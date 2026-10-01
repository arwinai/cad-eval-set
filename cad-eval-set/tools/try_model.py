"""Run one model against one task and keep what it produced for a person to judge.

    python3 tools/try_model.py claude tasks/sling_lift
    python3 tools/try_model.py gpt tasks/living_hinge
    python3 tools/try_model.py gpt:astra tasks/living_hinge
    python3 tools/try_model.py gemini:flash3.8 tasks/smartwatch
    python3 tools/try_model.py claude:fable5.1 helical_gear
    python3 tools/try_model.py claude:opus5.5 tasks/living_hinge
    python3 tools/try_model.py grok tasks/living_hinge
    python tools/try_model.py kimi:k27code tasks/helical_gear
    python3 tools/try_model.py deepseek tasks/living_hinge
    python3 tools/try_model.py glm tasks/sling_lift


The model is `route[:variant]` -- route is claude / gpt / gemini / grok /
kimi / deepseek / glm, variant is that route's own model key (claude:
sonnet5|opus5.5|fable5.1, gpt: gpt56|astra (GPT-6), gemini: pro3.1|flash3.8,
kimi: k2.7-code; grok, deepseek and glm each have a single deployment and
take no variant). The task
is just its folder name under `tasks/`, matched case-insensitively, so
`Sling_Lift` or `tasks/sling_lift` both find `tasks/sling_lift`. Which
CAD program the task uses is read off the before model's extension in
`environment/` (`input.SLDPRT` is SolidWorks, `input.FCStd` FreeCAD, ...).

Everything lands in `<task>/_runs/<model>_<timestamp>/`, which is gitignored:
the prompt actually sent, the model's raw answer, the "after" it built under
`after/`, its scratch files, and a short report. There is no grader: open
`after/` and decide whether it is good enough. Nothing outside that folder
is written, so a run can never disturb the task.

THE AGENT WORKS IN A REAL WORKSPACE, not a chat window. Every file from
`environment/` is copied into a scratch directory, the agent is given that
directory as its cwd, and its `bash` tool reaches the machine's actual CAD
toolchain -- `freecadcmd` for FreeCAD documents, the SolidWorks COM session
for SolidWorks ones, plain `python` with cadquery/trimesh/numpy for the
rest. So the deliverable is whatever the agent BUILT on disk, exactly as in
the container the real eval uses, and a `.FCStd` or `.SLDASM` is now a
normal outcome rather than something a text model cannot produce.

THAT WORKSPACE LIVES OUTSIDE THE REPOSITORY, under the system temp
directory, so a run can never write into the task folder, and the agent
cannot wander up into the checkout. Once it has finished, its scratch
files are copied back into the run folder for inspection.

Whatever it writes at the task's deliverable name is copied to `after/`.
If it never writes one, the run is reported as failed -- with the turn log
-- rather than left with an empty file. The one fallback
is a text deliverable (`solution.py`, a `.md` report) the model answered
with but never saved: that is written out from the answer, since the answer
IS the artefact there.

"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The toolchain paths (FREECAD_CMD) live in .env
# beside the checkout, and discover_toolchain() reads them from the
# environment -- so load it here rather than relying on whichever
# call_* module happens to be imported first.
import dotenv  # noqa: E402
dotenv.load_dotenv(ROOT.parent / ".env")

RUNS_DIRNAME = "_runs"

#: Generous but finite. A CAD build is not a chat turn -- opening a
#: document, measuring it, editing and re-checking costs turns -- but an
#: uncapped run can go for hours without saving anything. A run that hits
#: the ceiling is recorded as budget_exhausted in meta, so 'ran out of
#: turns' stays distinguishable from a crash.
AGENT_MAX_TURNS = 1000


#: The one folder under ROOT that holds the tasks. Which CAD
#: program a task uses is inferred from its input file's extension, not
#: from a parent folder name.
TASKS_DIR = "tasks"

#: What the model's answer can be written out as directly.
TEXT_CANDIDATE_EXTS = {".py", ".txt", ".md", ".json", ".scad", ".csv"}
#: A .docx deliverable is kept as Markdown when that is what the model wrote.
DOCX_FALLBACK_EXT = ".md"

#: How environment inputs are handed to `solve()`.
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PDF_EXTS = {".pdf"}
CLOUD_EXTS = {".stl", ".ply"}
TEXT_EXTS = {".py", ".txt", ".json", ".step", ".stp", ".scad", ".md", ".csv"}
#: Real inputs a text model cannot read; named in the prompt instead of
#: being silently dropped.
OPAQUE_EXTS = {".sldprt", ".sldasm", ".slddrw", ".fcstd", ".docx", ".xlsx",
               ".blend"}
SKIP_NAMES = {"Dockerfile", ".gitkeep"}


def is_task_file(p: Path) -> bool:
    """A real task input, as opposed to housekeeping. SolidWorks drops a
    `~$name.SLDPRT` lock file beside any document it has open; it vanishes
    when the document closes, which seal_solidworks() does before staging,
    so a run that counted it died copying a file that no longer existed."""
    return (p.is_file() and p.name not in SKIP_NAMES
            and not p.name.startswith("~$") and not p.name.startswith("."))

#: Checked when FREECAD_CMD is unset -- the Windows installer puts it under
#: AppData\Local\Programs, not Program Files, which is a common wrong guess.
FREECAD_FALLBACKS = (
    r"C:/Users/{}/AppData/Local/Programs/FreeCAD 1.1/bin/freecadcmd.exe".format(
        os.environ.get("USERNAME", "")),
    r"C:/Program Files/FreeCAD 1.1/bin/freecadcmd.exe",
    "/usr/bin/freecadcmd",
    "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",
)


# ---------------------------------------------------------------------------
# Resolving the arguments
# ---------------------------------------------------------------------------

def resolve_task(name: str) -> Path:
    """`tasks/Sling_Lift` or `sling_lift` -> the real directory,
    case-insensitively."""
    wanted = Path(name.replace("\\", "/"))
    parts = [p for p in wanted.parts if p not in (".", "")]
    if len(parts) > 1 and parts[-2].lower() != TASKS_DIR:
        raise SystemExit(f"{name!r}: tasks live under {TASKS_DIR}/<slug>; "
                         "there are no per-program folders")
    task = parts[-1]
    cands = [d for d in (ROOT / TASKS_DIR).glob("*")
             if d.is_dir() and d.name.lower() == task.lower()]
    cands = [c for c in cands if (c / "instruction.md").is_file()]
    if not cands:
        raise SystemExit(f"no task matching {name!r} (looked in {TASKS_DIR}/)")
    if len(cands) > 1:
        raise SystemExit(f"{name!r} is ambiguous: "
                         + ", ".join(str(c.relative_to(ROOT)) for c in cands))
    return cands[0]


#: Input extension -> CAD program. The primary input is `input.<ext>` in
#: environment/; the first match in this order wins when several exist
#: (an assembly over its parts, a CAD document over a neutral export).
PROGRAM_BY_EXT = (
    (".sldasm", "SolidWorks"), (".sldprt", "SolidWorks"),
    (".slddrw", "SolidWorks"),
    (".fcstd", "FreeCAD"),
    (".blend", "Blender"),
    (".py", "CadQuery"),
    (".step", "STEP"), (".stp", "STEP"),
)


def task_program(task_dir: Path) -> str:
    """CadQuery, FreeCAD, SolidWorks, STEP or Blender, from what is in
    environment/. This picks the container image and the sandbox."""
    env = task_dir / "environment"
    files = [p for p in env.rglob("*") if is_task_file(p)] if env.is_dir() else []
    exts = {p.suffix.lower() for p in files}
    for ext, program in PROGRAM_BY_EXT:
        if ext in exts:
            return program
    raise SystemExit(
        f"{env}: cannot tell which CAD program this task uses -- put the "
        "before model there as input.<ext> (one of "
        + ", ".join(e for e, _ in PROGRAM_BY_EXT) + ")")


def load_route(spec: str):
    """`claude:fable5.1` -> (module, variant, pretty label)."""
    route, _, variant = spec.partition(":")
    route = route.lower()
    if route == "claude":
        from common import call_claude as mod
    elif route == "gpt":
        from common import call_gpt as mod
    elif route in ("gemini", "google"):
        from common import call_gemini as mod
    elif route == "grok":
        from common import call_grok as mod
    elif route == "kimi":
        from common import call_kimi as mod
    elif route == "deepseek":
        from common import call_deepseek as mod
    elif route == "glm":
        from common import call_glm as mod
    else:
        raise SystemExit(f"unknown route {route!r} "
                         "(claude, gpt, gemini, grok, kimi, deepseek or glm)")

    if route in ("grok", "deepseek", "glm"):
        if variant:
            env_name = {
                "grok": "AZURE_GROK_DEPLOYMENT",
                "deepseek": "AZURE_DEEPSEEK_DEPLOYMENT",
                "glm": "AZURE_GLM_DEPLOYMENT",
            }[route]
            raise SystemExit(
                f"the {route} route has a single deployment and takes "
                f"no variant; set {env_name} to change it"
            )
        return mod, None, mod.label()

    variant = variant or mod.DEFAULT_MODEL
    variant = getattr(mod, "MODEL_ALIASES", {}).get(variant, variant)
    if variant not in mod.MODELS:
        raise SystemExit(f"unknown {route} model {variant!r}; "
                         f"known: {', '.join(mod.MODELS)}")
    return mod, variant, mod.label(variant)


# ---------------------------------------------------------------------------
# What the task asks for, and what it gives you
# ---------------------------------------------------------------------------

_APP_RE = re.compile(r"/app/([A-Za-z0-9_.\-]+)")
#: A deliverable named in prose instead of as a path, e.g. the report tasks'
#: "Deliver a text report ... as `report.docx` or `report.md`".
_TICKED_RE = re.compile(r"`([A-Za-z0-9_\-]+\.[A-Za-z0-9]{1,6})`")

#: Extensions a task can plausibly ask a candidate to produce.
DELIVERABLE_EXTS = {".py", ".txt", ".md", ".docx", ".json", ".scad",
                    ".fcstd", ".sldasm", ".sldprt", ".step", ".stp", ".blend"}

PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".stl", ".ply"}


def _is_input_name(name: str) -> bool:
    return Path(name).stem.lower().startswith("input")


def expected_candidate_name(task_dir: Path) -> str:
    """The filename the task wants written: instruction.md is the contract,
    `/app/<name>` where the task states a path, a backticked filename where
    it only says "deliver ... as `report.md`"."""
    instr = task_dir / "instruction.md"
    if instr.is_file():
        text = instr.read_text(encoding="utf-8", errors="replace")
        hits = [m.group(1) for m in _APP_RE.finditer(text)
                if not _is_input_name(m.group(1))
                and Path(m.group(1)).suffix.lower() in DELIVERABLE_EXTS]
        if hits:
            return hits[-1]
        ticked = [m.group(1) for m in _TICKED_RE.finditer(text)
                  if not _is_input_name(m.group(1))
                  and Path(m.group(1)).suffix.lower() in DELIVERABLE_EXTS]
        if ticked:
            # "as `report.docx` or `report.md`" -- take the first named.
            return ticked[0]

    raise SystemExit(f"cannot tell what {task_dir.name} wants written: "
                     "instruction.md must name the deliverable as /app/<name>")


def collect_inputs(task_dir: Path):
    """Environment files sorted into `solve()`'s buckets, plus the opaque ones."""
    env = task_dir / "environment"
    buckets = {"image_files": [], "pdf_files": [], "code_files": [],
               "pointcloud_files": []}
    opaque = []
    if not env.is_dir():
        return buckets, opaque
    for p in sorted(env.rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            buckets["image_files"].append(p)
        elif ext in PDF_EXTS:
            buckets["pdf_files"].append(p)
        elif ext in CLOUD_EXTS:
            buckets["pointcloud_files"].append(p)
        elif ext in TEXT_EXTS:
            buckets["code_files"].append(p)
        elif ext in OPAQUE_EXTS:
            opaque.append(p)
    return buckets, opaque


def build_prompt(task_dir: Path, candidate_name: str, tools: dict) -> str:
    """The task as written, plus what the machine can actually do."""
    instr = task_dir / "instruction.md"
    text = (instr.read_text(encoding="utf-8", errors="replace")
            if instr.is_file() else "")
    return (text.strip() + chr(10) + chr(10)
            + toolchain_brief(tools, candidate_name, task_dir))


def run_agent_for(mod, route: str, variant, prompt: str, *, cwd,
                  files, inline, require_file, max_turns, progress=None):
    """One call shape over three route signatures.

    Route modules differ in how they receive vision inputs and model variants,
    but try_model should not need provider-specific logic for interruption and
    resume accounting.

    `progress` is a provider-neutral live metadata channel.  A harness updates
    it while it works (most importantly with its current tool-call count), so
    try_model can recover accurate partial accounting even when run_agent()
    never reaches its normal return because of Ctrl-C, a 429, network failure,
    CLI crash, or another abnormal exit.

    Resume itself does NOT live in the route modules.  They only publish their
    existing live counters through this object; checkpointing, parent/child
    relationships and cumulative accounting remain try_model responsibilities.

    The routes still differ in two ordinary invocation details: GPT/Grok/
    DeepSeek/GLM take vision inputs as `images=`, while Claude/Gemini use
    `inline_files=`; Kimi takes `images=` plus an explicit model variant.
    """
    kw = dict(cwd=cwd, files=files, require_file=require_file,
              max_turns=max_turns, announce_files=True,)
    # Live progress is optional for backward compatibility.  try_model supplies
    # it for eval runs, while older/direct callers can continue using the same
    # helper without knowing about interruption/resume accounting.
    if progress is not None:
        kw["progress"] = progress

    if route in ("grok", "deepseek", "glm"):
        kw["images"] = inline
    elif route in ("kimi", "gpt"):
        kw["images"] = inline
        kw["model"] = variant
    else:
        kw["inline_files"] = inline
        kw["model"] = variant
    return mod.run_agent(prompt, **kw)


# ---------------------------------------------------------------------------
# Execution backends -- where the agent's `bash` actually runs
#
# The agent has a real shell. Running it in the program's container gives
# every run the same toolchain the task expects, keeps the checkout out of
# its reach, and lets `--network none` keep it offline.
#
# Only `bash` is redirected. `read_file` / `write_file` / `list_dir` keep
# operating on the host path, because the workspace is bind-mounted into the
# container: both sides see the same bytes, and the file tools were never the
# escape route -- they are confined to cwd already.
# ---------------------------------------------------------------------------

#: CAD program (see task_program) -> the base image that already
#: carries that program's toolchain. Built from common/docker/*.Dockerfile;
#: a CAD program without an image runs on the host.
PROGRAM_IMAGE = {
    "CadQuery": "cadquery-base:latest",
    "FreeCAD": "freecad-base:latest",
    # STEP tasks read their input with CadQuery's STEP importer, so they run
    # in the CadQuery image.
    # `step-base` only pip-installs: it has no libGL, so `import cadquery`
    # dies there with "libGL.so.1: cannot open shared object file" and the
    # agent loses the one library it needs to measure the part.
    "STEP": "cadquery-base:latest",
    # Blender 5.2.2 headless, numpy and scipy; nothing from CadQuery.
    "Blender": "blender-base:latest",
}

#: SolidWorks cannot be containerised -- COM, licensing and a GUI session --
#: so it gets the other mechanism: run the shell as a Windows account that
#: has no read access to this checkout.
SANDBOX_USER_ENV = "TRY_MODEL_SANDBOX_USER"
SANDBOX_PASS_ENV = "TRY_MODEL_SANDBOX_PASSWORD"

#: Where a sandboxed run's workspace goes. NOT %TEMP%: that lives inside the
#: launching user's profile, which the sandbox account cannot read, so every
#: command would fail on its working directory before it failed on anything
#: else. This directory is granted to the sandbox account instead:
#:   icacls C:	ry_model /grant "<user>:(OI)(CI)(M)"
SANDBOX_SHARED_ROOT = Path(os.environ.get("TRY_MODEL_SHARED_ROOT",
                                          "C:/try_model"))


def sandbox_credentials():
    """(user, password) when both are configured, else (None, None)."""
    return (os.environ.get(SANDBOX_USER_ENV),
            os.environ.get(SANDBOX_PASS_ENV))


def _under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def seal_solidworks(tools: dict, verbose: bool = True) -> dict:
    """Empty the shared SolidWorks session before the agent attaches to it.

    The agent attaches to the running SolidWorks instance -- the prompt
    tells it to -- and from there `GetDocuments` hands back every open
    document with its path, components, geometry and feature tree. A
    document left open from earlier work would be visible to it and never
    mentioned in the report, so the session is emptied first.

    Returns what it did, and raises SystemExit if a document from this
    checkout survives the close: a run that cannot be sealed is not a run
    whose score means anything, and going ahead quietly is how the
    duplicate-eval-set leak lasted as long as it did.
    """
    out = {"attached": False, "closed": 0, "before": [], "left": []}
    if not tools.get("solidworks_com") or not tools.get("solidworks_running"):
        return out
    try:
        from common import solidworks_session as sws
        app = sws.attach()
    except Exception as exc:                                # noqa: BLE001
        if verbose:
            print(f"solidworks: NOT sealed ({type(exc).__name__}: {exc}); "
                  "anything open in it is readable by the agent")
        return out
    if app is None:
        if verbose:
            print("solidworks: running but not reachable by COM; anything "
                  "open in it is readable by the agent")
        return out
    out["attached"] = True
    before = sws.open_documents(app) or []
    out["before"] = [d.get("title") for d in before]
    sws.close_all_documents(app)
    after = sws.open_documents(app) or []
    out["closed"] = max(0, len(before) - len(after))
    #: Only documents from THIS checkout are fatal. Something of the
    #: user's own left open elsewhere on the disk is their business and
    #: gives the agent nothing about the answer.
    out["left"] = [d.get("title") for d in after
                   if d.get("path") and _under(d["path"], ROOT)]
    if out["left"]:
        raise SystemExit(
            "REFUSING TO RUN: these documents are open in the shared "
            "SolidWorks and would not close, and they belong to this "
            "checkout -- the agent attaches to that same session and would "
            "see them:\n  "
            + "\n  ".join(out["left"])
            + "\nClose them in SolidWorks (or restart it) and run again.")
    if verbose and (before or after):
        print(f"solidworks: {out['closed']} document(s) closed before "
              f"handing the session over"
              + (f"; {len(after)} left open from outside the checkout"
                 if after else ""))
    return out

def workspace_root(task_dir: Path) -> Path:
    """%TEMP% normally; the shared root when the shell runs as another user."""
    user, pw = sandbox_credentials()
    if task_program(task_dir) == "SolidWorks" and user and pw:
        return SANDBOX_SHARED_ROOT
    return Path(tempfile.gettempdir()) / "try_model"


def _capped(cmd, timeout, cwd=None):
    """Run a command list with a timeout that actually kills the tree."""
    import subprocess
    kw = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            errors="replace", **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", err or ""
    except subprocess.TimeoutExpired:
        from common.agent_workspace import _kill_tree
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return None, out or "", err or ""


def docker_available() -> bool:
    try:
        rc, _, _ = _capped(["docker", "info"], 30)
        return rc == 0
    except (OSError, FileNotFoundError):
        return False


def image_present(image: str) -> bool:
    rc, out, _ = _capped(["docker", "images", "-q", image], 30)
    return rc == 0 and bool(out.strip())


#: A base image build downloads a CAD toolchain (FreeCAD, Blender, the
#: CadQuery wheels): minutes on a good connection, longer on a bad one.
IMAGE_BUILD_TIMEOUT_S = 3600


def build_image(image: str, program: str) -> bool:
    """Build the base image for `program` from common/docker/, streaming
    docker's output so a long build does not look like a hang."""
    import subprocess
    dockerfile = ROOT / "common" / "docker" / f"{program.lower()}-base.Dockerfile"
    if not dockerfile.is_file():
        print(f"  no Dockerfile for {program}: expected {dockerfile}")
        return False
    print(f"image     : {image} is missing -- building it from "
          f"{dockerfile.relative_to(ROOT).as_posix()} (first run only; "
          "this takes a few minutes)")
    try:
        rc = subprocess.call(["docker", "build", "-f", str(dockerfile),
                              "-t", image, str(ROOT)],
                             cwd=str(ROOT), timeout=IMAGE_BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  build timed out after {IMAGE_BUILD_TIMEOUT_S}s")
        return False
    return rc == 0 and image_present(image)


class DockerBackend:
    """The agent's shell runs inside the image for the task's CAD program.

    The container sees exactly two things: the image, and the workspace
    bind-mounted at /app. The repository is not on any path it can reach.
    `--network none` closes the other direction.
    """

    def __init__(self, image, workspace, run_id):
        self.image, self.workspace = image, Path(workspace)
        self.name = f"trymodel_{re.sub(r'[^A-Za-z0-9_.-]', '_', run_id)}"[:60]
        self.started = False

    def start(self):
        _capped(["docker", "rm", "-f", self.name], 60)
        rc, out, err = _capped([
            "docker", "run", "-d", "--name", self.name,
            "--network", "none",
            "-v", f"{self.workspace}:/app",
            "-w", "/app", self.image, "sleep", "infinity"], 300)
        if rc != 0:
            raise SystemExit(f"could not start the task container: "
                             f"{(err or out).strip()[:400]}")
        self.started = True

    def bash(self, cwd, command, timeout_s=None):
        from common.agent_workspace import (BASH_TIMEOUT_DEFAULT_S,
                                            BASH_TIMEOUT_MAX_S)
        t = min(int(timeout_s or BASH_TIMEOUT_DEFAULT_S), BASH_TIMEOUT_MAX_S)
        rc, out, err = _capped(
            ["docker", "exec", "-w", "/app", self.name,
             "sh", "-lc", command], t)
        if rc is None:
            return (f"TIMEOUT: killed after {t}s. The container process may "
                    "still be running; it is destroyed when the run ends. Do "
                    "not wait on it again -- check for a result file instead.")
        body = out
        if err:
            body += ("\n--- stderr ---\n" if body else "--- stderr ---\n") + err
        return f"exit={rc}\n{body}".strip()

    def stop(self):
        if self.started:
            _capped(["docker", "rm", "-f", self.name], 120)
            self.started = False

    def describe(self, task_dir):
        blender = "blender" in self.image
        return {
            "backend": f"docker ({self.image}, --network none)",
            "python": "python3 (inside the container)",
            # The Blender image carries numpy and scipy only; the others
            # carry env_requirements.txt.
            "python_modules": ("numpy, scipy" if blender else
                               "cadquery, trimesh, numpy, shapely, manifold3d, "
                               "pymupdf (per env_requirements.txt)"),
            "freecadcmd": ("/usr/local/bin/freecadcmd"
                           if "freecad" in self.image else None),
            "blender": "/usr/local/bin/blender" if blender else None,
            "workdir": "/app",
        }


class SandboxUserBackend:
    """The agent's shell runs as a second Windows account.

    For SolidWorks, which has no container: deny that account read access to
    the checkout (`icacls`) and the same three answer sources become
    unreachable, while COM still works -- PROVIDED SolidWorks itself is
    running as that account. COM attaches within a session; a process in
    another user's session cannot reach the primary user's SolidWorks, so
    running the shell as a second user while SolidWorks runs as you does not
    give you a sandbox, it gives you a SolidWorks that cannot be found.
    """

    def __init__(self, user, password, workspace):
        self.user, self.password = user, password
        self.workspace = Path(workspace)

    def bash(self, cwd, command, timeout_s=None):
        from common.agent_workspace import (BASH_TIMEOUT_DEFAULT_S,
                                            BASH_TIMEOUT_MAX_S)
        t = min(int(timeout_s or BASH_TIMEOUT_DEFAULT_S), BASH_TIMEOUT_MAX_S)
        out_f = self.workspace / "._sbx_out.txt"
        err_f = self.workspace / "._sbx_err.txt"
        for f in (out_f, err_f):
            try:
                f.unlink()
            except OSError:
                pass
        # Start-Process is the only scriptable way to launch as another user
        # without an interactive `runas` prompt; it cannot capture stdout, so
        # the streams are redirected to files in the shared workspace.
        # PowerShell single-quoted literals: a quote inside the command is
        # escaped by doubling it.
        def q(v):
            return "'" + str(v).replace("'", "''") + "'"

        ps = (
            "$sec = ConvertTo-SecureString $env:TRY_MODEL_SANDBOX_PASSWORD "
            "-AsPlainText -Force;"
            "$cred = New-Object System.Management.Automation.PSCredential("
            "$env:TRY_MODEL_SANDBOX_USER, $sec);"
            f"$p = Start-Process -FilePath cmd.exe -ArgumentList '/c', {q(command)} "
            f"-Credential $cred -WorkingDirectory {q(self.workspace)} "
            f"-RedirectStandardOutput {q(out_f)} "
            f"-RedirectStandardError {q(err_f)} -Wait -PassThru;"
            # Without -PassThru and this explicit exit, PowerShell's own exit
            # code is reported, so a command that failed comes back exit=0
            # and the agent reads a broken build as a successful one.
            "exit $p.ExitCode"
        )
        env_backup = dict(os.environ)
        os.environ[SANDBOX_USER_ENV] = self.user
        os.environ[SANDBOX_PASS_ENV] = self.password
        try:
            rc, out, err = _capped(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                t)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

        def _read(f):
            try:
                return f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
        if rc is None:
            return f"TIMEOUT: killed after {t}s (ran as {self.user})"
        body = _read(out_f)
        e = _read(err_f) or err
        if e:
            body += ("\n--- stderr ---\n" if body else "--- stderr ---\n") + e
        return f"exit={rc}\n{body}".strip()

    def stop(self):
        pass

    def describe(self, task_dir):
        d = discover_toolchain()
        d["backend"] = f"windows user {self.user} (no read access to the repo)"
        return d


def install_bash_backend(backend):
    """Route the shared `bash` tool through this backend.

    Patched rather than parameterised: `agent_workspace` is imported by all
    three call_* modules and its TOOL_IMPL is what `dispatch` looks up, so
    replacing the entry here redirects every route at once without any of
    them knowing there is more than one place a command can run.
    """
    from common import agent_workspace as ws
    previous = ws.TOOL_IMPL["bash"]
    ws.TOOL_IMPL["bash"] = backend.bash
    return previous


def restore_bash_backend(previous):
    from common import agent_workspace as ws
    ws.TOOL_IMPL["bash"] = previous


def choose_backend(task_dir: Path, workspace: Path, run_id: str,
                   allow_host: bool = False):
    """(backend, note). None means 'run on the host, as before'.

    A program that HAS an image never falls back to the host on its own: with
    Docker down or the image missing, the run stops before the agent gets a
    shell, unless `allow_host` (--allow-host) says to go ahead anyway: a
    host run uses whatever toolchain this machine happens to have, and the
    result should say so rather than be mistaken for a container run.
    """
    family = task_program(task_dir)
    image = PROGRAM_IMAGE.get(family)

    if image:
        problem = None
        if not docker_available():
            problem = ("docker is not running -- the agent's shell would be "
                       "on the HOST, with this machine's toolchain and "
                       "this checkout in reach.")
        elif not image_present(image) and not build_image(image, family):
            problem = (f"image {image} is missing and could not be built "
                       f"(see docker's output above; the Dockerfile is "
                       f"common/docker/{family.lower()}-base.Dockerfile); "
                       "without it the agent's shell would be on the HOST.")
        if problem:
            if not allow_host:
                raise SystemExit(f"refusing to run: {problem} Start Docker "
                                 "(or build the image) and run again, or pass "
                                 "--allow-host to run on the host anyway.")
            return None, (problem + " Running on the HOST (--allow-host); "
                          "scores from this run are not trustworthy.")
        return DockerBackend(image, workspace, run_id), None

    if family == "SolidWorks":
        user, pw = sandbox_credentials()
        if user and pw:
            return SandboxUserBackend(user, pw, workspace), None
        return None, (f"SolidWorks cannot be containerised; set "
                      f"{SANDBOX_USER_ENV} and {SANDBOX_PASS_ENV} in .env to "
                      "run the shell as a restricted account. Running on the "
                      "HOST meanwhile; scores are not trustworthy.")

    return None, f"no container image for program {family!r}; running on the host"


# ---------------------------------------------------------------------------
# The toolchain the agent may actually use
# ---------------------------------------------------------------------------

def discover_toolchain() -> dict:
    """What this machine can really run, checked rather than assumed.

    Told to the agent verbatim. A prompt that advertises freecadcmd on a box
    without it buys a dozen wasted turns and a confident wrong answer, and
    the reverse -- a machine that has SolidWorks while the prompt stays
    silent -- is how a solvable task gets scored as impossible.
    """
    import shutil as _sh
    import subprocess

    found = {}

    configured = os.environ.get("FREECAD_CMD") or os.environ.get("FREECADCMD")
    fc = (configured or _sh.which("freecadcmd") or _sh.which("FreeCADCmd")
          or next((p for p in FREECAD_FALLBACKS if Path(p).is_file()), None))
    if fc and Path(fc).is_file():
        found["freecadcmd"] = fc
    elif configured:
        # Set but wrong is the dangerous case: without this the agent is
        # simply never told FreeCAD exists, writes a build script it cannot
        # run, and the run ends with no output and nothing pointing at the
        # cause. A mangled path is the usual reason -- dotenv expands
        # backslash escapes, so `\bin\` becomes control characters.
        found["freecadcmd_BROKEN"] = (
            f"{configured!r} is set but is not a file "
            f"(use forward slashes in .env)")

    try:
        from common import solidworks_session as _sws
        if getattr(_sws, "_IMPORT_ERROR", None) is None:
            found["solidworks_com"] = "common.solidworks_session (pywin32)"
    except Exception:                                       # noqa: BLE001
        pass

    # Whether SolidWorks is up decides whether "attach" is even possible,
    # and the agent should be told rather than left to find out by hanging.
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq SLDWORKS.exe"],
                                 capture_output=True, text=True, timeout=20).stdout
            found["solidworks_running"] = "SLDWORKS" in (out or "").upper()
        except Exception:                                   # noqa: BLE001
            pass

    mods = []
    for m in ("cadquery", "trimesh", "numpy", "shapely", "manifold3d",
              "OCP", "ezdxf", "pymupdf"):
        try:
            __import__(m)
            mods.append(m)
        except Exception:                                   # noqa: BLE001
            pass
    if mods:
        found["python_modules"] = ", ".join(mods)
    found["python"] = sys.executable
    return found


def toolchain_brief(tools: dict, deliverable: str, task_dir: Path) -> str:
    """The part of the prompt that turns a chat answer into a build job."""
    where = tools.get("backend")
    lines = [
        "## Your workspace",
        "",
        ("Your `bash` tool runs inside a container: " + where + ". The "
         "working directory is `/app` and the task's input files are already "
         "in it. Only /app is shared with the outside; nothing else on the "
         "host is visible or reachable."
         if where and where.startswith("docker") else
         "You are working in a real directory on a real machine, and your "
         "`bash` tool runs there. The task's input files are already in it."),
        "",
        f"**Write your deliverable to `{deliverable}` in this directory.** "
        "That exact file is what gets reviewed -- not your chat reply. Build "
        "it, then check it opens/parses before you finish.",
        "",
        "Available here:",
        f"- `python` -> `{tools['python']}`",
    ]
    if where and not where.startswith("docker"):
        lines.insert(1, "")
        lines.insert(2, f"(shell backend: {where})")
    if tools.get("python_modules"):
        lines.append(f"  (importable: {tools['python_modules']})")
    if tools.get("freecadcmd"):
        lines += [
            f"- `freecadcmd` -> `{tools['freecadcmd']}`",
            '  Run headless FreeCAD scripting with '
            '`"<freecadcmd>" script.py`; inside it `import FreeCAD` works. '
            'Use it to open, edit and save `.FCStd` documents.',
        ]
    if tools.get("blender"):
        lines += [
            f"- `blender` -> `{tools['blender']}` (Blender 5.2 LTS, headless; "
            "no GUI)",
            "  Run Blender Python with `blender -b <file>.blend --python "
            "script.py`; inside it `import bpy` works, and numpy is bundled. "
            "Use it to open, edit and save `.blend` files "
            "(`bpy.ops.wm.save_as_mainfile(filepath=...)`).",
        ]
    if tools.get("solidworks_com"):
        running = tools.get("solidworks_running")
        lines += [
            f"- SolidWorks via COM -> `{tools['solidworks_com']}`",
            f"  SolidWorks is {'ALREADY RUNNING' if running else 'NOT running'} "
            "on this machine.",
            f"  `sys.path.insert(0, r\"{task_dir.parent.parent}\")` then "
            "`from common import solidworks_session as sws` and `sws.attach()` "
            "to reach that running instance.",
            "  **Open your documents WRITABLE.** `sws.open_document()` "
            "defaults to `OPEN_SILENT | OPEN_READONLY` because the tooling "
            "uses it for inspection and must not modify files -- with that "
            "default your edits cannot be saved and the title bar reads "
            "`[Read-only]`. Pass `options=sws.OPEN_SILENT` instead. Save "
            "explicitly when you are done (the parts as well as the "
            "assembly), and save before closing: a dirty document raises a "
            "'save changes?' dialog that blocks the COM thread with nobody "
            "there to click it.",
            "  **Save only with `doc.Extension.SaveAs3(path, 0, 1, None, "
            "None, err, warn)`** (err/warn from `sws.byref_i4()`; C#: "
            "`swSaveAsCurrentVersion`, `swSaveAsOptions_Silent`, `ref err, "
            "ref warn`) and check that it returned True and the file exists. "
            "**Never call `Save()` / `Save3()` on a document that has never "
            "been saved** -- one made by `NewDocument` or imported by "
            "`LoadFile2` / opening a STEP has no filename, so `Save()` opens "
            "the interactive Save As window and every later COM call, from "
            "any script, hangs behind it. The one-argument `doc.SaveAs(path)` "
            "is obsolete and can fail without saying so, leaving the "
            "document untitled.",
            "  **Arm the dialog watchdog before you touch anything.** "
            "`sws.arm_unattended(app)` right after `sws.attach()` puts "
            "SolidWorks in quiet mode and starts a thread that dismisses "
            "the modal windows it still raises -- a CAM add-in error, a "
            "rebuild notice, a 'save changes?' box. Any one of them blocks "
            "the COM thread and every later call from every script hangs "
            "behind it, with no error and no output: an earlier run lost "
            "half an hour to a `Ошибка SOLIDWORKS CAM` box and then killed "
            "its own process. `sws.session_report()` prints what fired. "
            "Call it in EVERY script you run, not just the first -- each "
            "one is a new process.",
            "  **Do not start a second SolidWorks.** `CreateInstance` / "
            "`new SldWorks()` / `Activator.CreateInstance(ProgID)` launches a "
            "fresh instance instead of attaching; with one already running "
            "and a licence held, the call blocks and never returns -- an "
            "earlier run sat on exactly that for 286 minutes. Attach to the "
            "existing session (GetActiveObject / the helpers above).",
        ]
    env = task_dir / "environment"
    pdfs = sorted(p.name for p in env.glob("*.pdf")) if env.is_dir() else []
    if pdfs:
        lines += [
            "",
            "The drawing set is staged as a file rather than pasted into this "
            "conversation, because it is megabytes and would ride along in "
            "every turn: " + ", ".join(f"`{n}`" for n in pdfs) + ".",
            "  Read the pages you need with PyMuPDF, e.g. `python3 -c \"import "
            "pymupdf; d=pymupdf.open('" + pdfs[0] + "'); print(d.page_count); "
            "print(d[0].get_text())\"`, and render a page to PNG with "
            "`d[0].get_pixmap(dpi=200).save('page0.png')` if you need to look "
            "at the geometry.",
        ]
    lines += [
        "",
        "Nothing outside this directory is yours to change, and nothing "
        "outside it is readable -- the rest of the machine is not part of "
        "this task. There is no grading script, rubric or reference for you "
        "to consult; searching the filesystem for one finds nothing and "
        "costs you turns. Judge your own work against the requirements "
        "above, and by measuring the model you build.",
    ]
    return chr(10).join(lines)


# ---------------------------------------------------------------------------
# Turning the answer into a candidate file
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)


def unfence(text: str) -> str:
    """The solvers return a script wrapped in a fence; the file wants it bare."""
    blocks = _FENCE_RE.findall(text or "")
    if not blocks:
        return text or ""
    return max(blocks, key=len).strip() + "\n"


def write_candidate(run_dir: Path, candidate_name: str, answer: str):
    """(path, note) -- path is None when there is nothing worth keeping.

    An empty answer is one of those cases and is the one worth naming: an
    agent that burns its whole turn budget without writing the deliverable
    returns "", and writing that out as a 0-byte file would look like an
    output. The run failed, and the report should say so.
    """
    if not (answer or "").strip():
        return None, ("the model returned no answer -- nothing was written. "
                      "Usually the turn budget ran out before it saved the "
                      "deliverable; see raw_response.md and the turn log.")
    ext = Path(candidate_name).suffix.lower()

    if ext in TEXT_CANDIDATE_EXTS:
        dest = run_dir / candidate_name
        body = unfence(answer) if ext in {".py", ".scad", ".json"} else (answer or "")
        dest.write_text(body, encoding="utf-8")
        return dest, None

    if ext == ".docx":
        dest = run_dir / (Path(candidate_name).stem + DOCX_FALLBACK_EXT)
        dest.write_text(answer or "", encoding="utf-8")
        return dest, (f"{candidate_name} is a .docx; wrote {dest.name} instead, "
                      "since that is what the model wrote")

    # A CAD document the agent was supposed to BUILD in the workspace and
    # did not. The toolchain is available to it -- this is not a limit of
    # the tool -- so the useful thing is the script it wrote plus a pointer
    # at the likely causes.
    script = run_dir / "candidate_script.py"
    script.write_text(unfence(answer), encoding="utf-8")
    return None, (f"the agent never wrote {candidate_name} to its workspace, "
                  f"so there is no after/; its build script was saved "
                  f"as {script.name}. It had a shell and the task's toolchain "
                  "-- check the run header for a `WARNING : freecadcmd ...` "
                  "line (a mis-set path means the agent was never told the "
                  "toolchain existed), then the transcript for a build that "
                  "failed or was never attempted.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def keep_native_logs_in_run(solve_meta, run_dir: Path) -> list:
    """Move a harness's own session logs into `<run>/native/`.

    Routes that drive a harness (Claude Code, Codex, Gemini CLI) hand back
    `native_logs`: copies of the harness's session files, taken before its
    throwaway home was deleted. Those are the full-fidelity record -- every
    message, retry, timeout and compaction. Other routes return none.
    """
    import shutil
    kept = []
    srcs = [Path(p) for p in ((solve_meta or {}).get("native_logs") or [])]
    if not srcs:
        return kept
    # every file sits under one temp dir named native_<label>_*
    base = next((a for a in srcs[0].parents
                 if a.name.startswith("native_")), srcs[0].parent)
    dest_root = run_dir / "native"
    for src in srcs:
        try:
            rel = src.relative_to(base)
        except ValueError:
            rel = Path(src.name)
        # Claude Code names its project folder after the whole workspace
        # path (C--Users-...-try-model-<run id>): under a run folder that is
        # already deep, the result passes Windows' 260-character limit and
        # the move fails. Such a folder name adds nothing the run folder
        # does not already say, so it is shortened.
        rel = Path(*[("workspace" if len(part) > 48 and part != rel.name else part)
                     for part in rel.parts])
        dest = dest_root / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dest)
        except OSError as e:
            # A log that cannot be kept must not cost the run its output.
            print(f"  WARNING : could not keep native log {src.name}: {e}")
            continue
        kept.append(dest.relative_to(run_dir).as_posix())
    shutil.rmtree(base, ignore_errors=True)
    return kept


def render_report(meta: dict) -> str:
    """A short, human-readable summary of the run. There is no score: a
    person opens `after/` and decides whether the result is good enough."""
    L = [f"# {meta['task']} -- {meta['model_label']}", "",
         f"- run: `{meta['run_id']}`",
         f"- status: {meta['status']}",
         f"- solved in: {meta['solve_seconds']}s"
         + (f", {meta['turns']} turns" if meta.get("turns") else "")
         + (" -- TURN BUDGET EXHAUSTED" if meta.get("budget_exhausted") else ""),
         f"- wanted: `{meta['wants']}`",
         f"- produced: `{meta.get('candidate') or '(none)'}`"
         + (f" -- {meta['candidate_note']}" if meta.get("candidate_note") else ""),
         f"- sandbox: {meta['sandbox']}"]
    if meta.get("sandbox_warning"):
        L.append(f"- WARNING: {meta['sandbox_warning']}")
    if meta.get("resume_sessions", 1) > 1:
        L.append(f"- cumulative: {meta.get('cumulative_solve_seconds')}s, "
                 f"{meta.get('cumulative_turns')} turns across "
                 f"{meta.get('resume_sessions')} sessions")
    if meta.get("after_files"):
        L += ["", "## after/", ""] + [f"- `{f}`" for f in meta["after_files"]]
    if meta.get("workspace_artifacts"):
        L += ["", "## workspace/ (the agent's scratch)", ""] + \
             [f"- `{f}`" for f in meta["workspace_artifacts"]]
    L += ["", "Open `after/` and judge the result by eye; `raw_response.md` "
          "and `transcript.json` say how the model got there.", ""]
    return "\n".join(L)


def _exhausted(solve_meta, max_turns=None) -> bool:
    """True when the agent stopped because it ran out of turns.

    `run_agent` reports the turn it reached; hitting the ceiling exactly is
    what "ran out" looks like from the outside, and it changes how a bad
    score should be read -- out of time, not out of ideas.
    """
    turns = (solve_meta or {}).get("turns")
    ceiling = max_turns or AGENT_MAX_TURNS
    return bool(turns) and turns >= ceiling


#: Where a run's deliverable lands inside its `_runs/<id>/` folder: the
#: "after", for a person to open and judge.
SOLUTION_DIR = "after"

#: What travels with the deliverable. An assembly is not one file, and the
#: renders are what a person actually looks at.
#: Named apart from the DELIVERABLE_EXTS above on purpose: this one was
#: once called the same, and being defined later it silently replaced the
#: first, so expected_candidate_name() read deliverable names against this
#: list -- the yeti mic "wanted" `Pattern.png`, and tasks 6, 7 and 74
#: could not be resolved at all.
DELIVERABLE_SET_EXTS = {".sldprt", ".sldasm", ".png", ".stl",
                        ".mp4"}   # a turntable video (Blender 5_ohlins_suspension)


def keep_deliverable_set(workspace: Path, run_dir: Path,
                         candidate_name: str) -> list:
    """Copy the whole CAD deliverable beside the candidate in `after/`.

    NOT just the named file, and NOT only files the agent created.

    An assembly carries its geometry in its COMPONENTS: a `Shampoo.SLDASM`
    holds none of the bottle's shape, and a candidate that thins the wall
    does it in `Shampoo Body.SLDPRT`. Keeping the assembly alone would show
    a person the seed parts beside a changed assembly.

    `keep_workspace_artifacts` cannot stand in for this. It skips
    anything whose NAME matches a staged input, so a staged part the
    agent MODIFIED is indistinguishable from one it never touched, and
    is dropped for being an input.
    """
    dest = run_dir / SOLUTION_DIR
    kept = []
    for item in sorted(workspace.iterdir()):
        if not item.is_file() or item.name == candidate_name:
            continue
        # A `~$` lock file is SolidWorks session scratch, not geometry.
        if item.name.startswith("~$"):
            continue
        # `input.<ext>` is the SEED, never the answer: filetree.MD fixes
        # that name for the before-model, and the staged copy sits in the
        # workspace beside the deliverable. Copying it here put an
        # `input.SLDASM` in a folder called `after/`, which reads as if
        # the deliverable were named input. The task's own inputs are
        # already in `environment/`; a component the agent MODIFIED keeps
        # its own name and is still picked up below.
        if item.stem.lower() == "input":
            continue
        if item.suffix.lower() not in DELIVERABLE_SET_EXTS:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(item, dest / item.name)
            kept.append(item.name)
        except OSError:
            pass
    return kept


def harvest(workspace: Path, run_dir: Path, candidate_name: str, answer: str):
    """(built_path, record_path, note). What the agent BUILT beats what it said.

    The deliverable is kept where it was built until the end of the run and
    copied into `after/` as the record a person opens.

    The deliverable is looked for in the workspace first. Only when it is
    absent does the answer get a say, and only for a text deliverable --
    there the answer genuinely is the artefact, and a model that printed a
    correct `solution.py` without ever saving it should not be failed on
    filing. That fallback is written INTO the workspace too, for the same
    reason.
    """
    def record(built, note=None):
        dest = run_dir / SOLUTION_DIR / built.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, dest)
        return built, dest, note

    built = workspace / candidate_name
    if built.is_file() and built.stat().st_size > 0:
        #: A deliverable is not always one file: an assembly's parts sit
        #: BESIDE it, a part may come with a report and material sheet.
        #: keep_deliverable_set below carries the companions along.
        beside = [q for q in workspace.iterdir()
                  if q.is_file() and q.name != candidate_name]
        note = (f"built in the workspace beside {len(beside)} other file(s)"
                if beside else None)
        out = record(built, note)
        kept = keep_deliverable_set(workspace, run_dir, candidate_name)
        if kept:
            note = ((note + "; " if note else "")
                    + f"after/ holds {len(kept) + 1} file(s)")
            out = (out[0], out[1], note)
        return out

    if Path(candidate_name).suffix.lower() == ".docx":
        alt = workspace / (Path(candidate_name).stem + DOCX_FALLBACK_EXT)
        if alt.is_file() and alt.stat().st_size > 0:
            return record(alt, f"kept {alt.name}, standing in for "
                               f"{candidate_name}")

    written, note = write_candidate(workspace, candidate_name, answer)
    if written is None:
        return None, None, note
    return record(written, note)


def save_checkpoint(workspace: Path, run_dir: Path) -> Path:
    """Save the complete agent workspace so an interrupted run can be resumed.

    Unlike keep_workspace_artifacts(), this intentionally keeps staged inputs
    too. A resume must reconstruct the exact working directory the agent had,
    not just the files it created.
    """
    dest = run_dir / "checkpoint"

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    shutil.copytree(workspace, dest)

    return dest


def resolve_resume_checkpoint(
        task_dir: Path, run_id: str, model_spec: str) -> tuple[Path, dict]:
    """Find and validate a checkpoint belonging to a previous run.

    Resume never modifies the parent run.  The old run is historical evidence
    of what happened before the interruption; a continuation gets a new run
    id and records the old one as its parent.

    The checkpoint must belong to THIS task and must contain a report.json
    whose metadata marks it resumable.  This prevents accidentally resuming
    from another task's run or from an arbitrary directory that merely happens
    to have the requested name.
    """
    parent_dir = task_dir / RUNS_DIRNAME / run_id
    report_file = parent_dir / "report.json"
    checkpoint = parent_dir / "checkpoint"

    if not parent_dir.is_dir():
        raise SystemExit(f"cannot resume {run_id!r}: no such run under "
                         f"{task_dir.relative_to(ROOT).as_posix()}"
                         f"/{RUNS_DIRNAME}/")

    if not report_file.is_file():
        raise SystemExit(f"cannot resume {run_id!r}: "
                         f"the run has no report.json")

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot resume {run_id!r}: report.json "
                         f"cannot be read: {exc}") from None

    meta = report.get("meta") or {}

    if meta.get("task") != task_dir.relative_to(ROOT).as_posix():
        raise SystemExit(f"cannot resume {run_id!r}: it belongs to task "
                         f"{meta.get('task')!r}, not "
                         f"{task_dir.relative_to(ROOT).as_posix()!r}")
    
    # A resumed run is a continuation of the same benchmark attempt, so its
    # model must match the parent. Mixing models would also mix their turns
    # and solve time in cumulative accounting and make the result incomparable
    # to an ordinary single-model run.
    if meta.get("model") != model_spec:
        raise SystemExit(f"cannot resume {run_id!r}: it was run with "
                         f"{meta.get('model')!r}, not {model_spec!r}")

    if not meta.get("resumable"):
        raise SystemExit(f"cannot resume {run_id!r}: "
                         f"the run is not marked resumable")

    if not checkpoint.is_dir():
        raise SystemExit(f"cannot resume {run_id!r}: "
                         f"checkpoint directory is missing")

    return checkpoint, meta


def find_latest_resumable_run(task_dir: Path, model_spec: str) -> str:
    """Return the newest resumable run id for this exact model and task.

    "Latest" must not mean merely the newest directory in `_runs`.  A task may
    contain completed runs, failed experiments and runs from several models.
    Resume-latest should only select a run whose report says that it is
    resumable AND whose model spec matches the model requested now.

    The report metadata is authoritative rather than the run-id prefix.  Model
    labels and filename slugs can change over time, while `meta["model"]`
    records the actual route specification used for that run.

    Runs without report.json are ignored.  They may be incomplete historical
    folders, but there is no reliable metadata proving that they are safe to
    resume automatically.
    """
    runs_dir = task_dir / RUNS_DIRNAME

    if not runs_dir.is_dir():
        raise SystemExit(f"cannot --resume-latest: "
                         f"{task_dir.relative_to(ROOT).as_posix()} "
                         f"has no runs")

    candidates = []

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        report_file = run_dir / "report.json"
        checkpoint = run_dir / "checkpoint"

        if not report_file.is_file() or not checkpoint.is_dir():
            continue

        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        meta = report.get("meta") or {}

        # Resume-latest is intentionally model-specific.  Continuing a Kimi
        # workspace with DeepSeek may be useful as a separate experiment, but
        # it must be explicit rather than something --resume-latest does by
        # accident.
        if meta.get("model") != model_spec:
            continue

        if not meta.get("resumable"):
            continue

        # Use the report/run directory mtime rather than parsing timestamps
        # from run ids.  The id format is an implementation detail, whereas
        # modification time directly represents which resumable run was
        # completed most recently.
        try:
            modified = max(run_dir.stat().st_mtime,
                           report_file.stat().st_mtime,)
        except OSError:
            continue

        candidates.append((modified, run_dir.name))

    if not candidates:
        raise SystemExit(f"cannot --resume-latest: no resumable run for "
                         f"{model_spec!r} on "
                         f"{task_dir.relative_to(ROOT).as_posix()}")

    candidates.sort(reverse=True)
    return candidates[0][1]


def resume_totals(resume_meta: dict | None) -> tuple[int, float, int]:
    """Return cumulative work inherited from the parent resume chain.

    A resumed run is a new model session, but it is NOT a fresh benchmark
    attempt.  Reporting only the child session's turns and solve time would
    make an interrupted 200-turn run followed by an 80-turn continuation look
    like an 80-turn solution.

    Every child therefore inherits the parent's cumulative totals.  Older
    reports created before resume support have no cumulative fields, so their
    own `turns` and `solve_seconds` are used as the starting point.

    Returns:
        (previous_turns, previous_seconds, previous_sessions)
    """
    if not resume_meta:
        return 0, 0.0, 0

    turns = resume_meta.get("cumulative_turns",
                            resume_meta.get("turns") or 0,)
    
    seconds = resume_meta.get("cumulative_solve_seconds",
                              resume_meta.get("solve_seconds") or 0.0,)
    
    sessions = resume_meta.get("resume_sessions", 1)

    try:
        turns = int(turns or 0)
    except (TypeError, ValueError):
        turns = 0

    try:
        seconds = float(seconds or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0

    try:
        sessions = int(sessions or 1)
    except (TypeError, ValueError):
        sessions = 1

    return turns, seconds, sessions


def restore_checkpoint(checkpoint: Path, workspace: Path) -> list[str]:
    """Restore a previous run's complete workspace into a new scratch workspace.

    Resume always gets a NEW temporary workspace.  The parent checkpoint is
    historical state and must never be modified in place: if the resumed agent
    damages a file, crashes, or is interrupted again, the previous run must
    remain reproducible and resumable.

    The checkpoint is copied before the agent starts, preserving the exact
    files the previous session had at interruption -- including staged inputs,
    build scripts, measurements, renders and any partial deliverable.

    Returns the names restored at the workspace root for reporting.
    """
    if not checkpoint.is_dir():
        raise SystemExit(f"resume checkpoint is missing: {checkpoint}")

    workspace.mkdir(parents=True, exist_ok=True)

    restored = []
    for item in sorted(checkpoint.iterdir()):
        dest = workspace / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
        restored.append(item.name)

    return restored


def keep_workspace_artifacts(workspace: Path, run_dir: Path, staged) -> list:
    """Copy what the agent made into the run folder, and say what that was.

    Only its own files: the staged inputs are already in `environment/`, and
    one of them here is a 57 MB document that would be copied on every run
    for nothing. This happens after the agent has stopped, so putting the
    files back beside the task cannot help it.
    """
    staged_names = {f.name for f in staged}
    dest = run_dir / "workspace"
    made = []
    for item in sorted(workspace.iterdir()):
        if item.name in staged_names or item.name == "__pycache__":
            continue
        # The deliverable itself lives in `after/`; this folder is the
        # agent's scratch -- its probe scripts, logs and notes. Keeping the
        # CAD and the renders in both would store an 8 MB STL twice.
        if item.is_file() and item.suffix.lower() in DELIVERABLE_SET_EXTS:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        try:
            if item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest / item.name)
            made.append(item.name)
        except OSError:
            pass
    return made


def main(argv=None) -> int:
    # The routes echo every tool call, and an agent's arguments carry any
    # character it likes. On Windows, stdout redirected to a log defaults to
    # cp1252, so one Greek letter in a tool call raised UnicodeEncodeError and
    # ended a two-hour Grok run as "solve FAILED".
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                       # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "model",
        help=("claude[:sonnet5|opus5.5|fable5.1] | gpt[:gpt56|astra] | "
              "gemini[:pro3.1|flash3.8] | grok | kimi[:k27code|k3] | "
              "deepseek | glm")
    )
    ap.add_argument("task",
                    help="e.g. tasks/sling_lift (case-insensitive)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help=f"tool-call budget for the run (default {AGENT_MAX_TURNS})")
    ap.add_argument("--allow-host", action="store_true",
                    help=("run on the host when the program's container "
                          "cannot start (Docker down, image missing); "
                          "without it the run stops instead"))
    
    # A continuation can name an exact parent run or ask try_model to find the
    # newest resumable run for this model/task pair.  These modes are mutually
    # exclusive: accepting both would make the parent ambiguous and could resume
    # from a different workspace than the user intended.
    resume_group = ap.add_mutually_exclusive_group()

    resume_group.add_argument("--resume", metavar="RUN_ID",
                              help=("continue from a previous run's "
                                    "checkpoint; the previous run is left "
                                    "unchanged and a new child run is created"))

    resume_group.add_argument("--resume-latest", action="store_true",
                              help=("continue from the newest resumable run "
                                    "for this model and task; "
                                    "the parent run is left unchanged "
                                    "and a new child run is created"))

    args = ap.parse_args(argv)

    task_dir = resolve_task(args.task)
    mod, variant, model_label = load_route(args.model)
    route = args.model.split(":")[0].lower()
    candidate_name = expected_candidate_name(task_dir)
    buckets, _opaque = collect_inputs(task_dir)
    tools = discover_toolchain()
    resume_checkpoint = None
    resume_meta = None

    # Resolve --resume-latest to a concrete parent id immediately.  Everything
    # below then has only one concept of a resume parent (`args.resume`), which
    # keeps checkpoint restoration, prompt construction, metadata and reporting
    # identical whether the user supplied the id or asked us to find it.
    if args.resume_latest:
        args.resume = find_latest_resumable_run(task_dir, args.model)

    if args.resume:
        resume_checkpoint, resume_meta = resolve_resume_checkpoint(
            task_dir, args.resume, args.model)

    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    # The random tail is not decoration. The id names the run folder, the
    # workspace under %TEMP%	ry_model and the container, and three runs
    # started in the same second once shared ONE workspace -- each agent
    # saw the others' input files -- while the container names collided.
    import secrets
    run_id = (f"{slug}_"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
              f"{secrets.token_hex(3)}")
    run_dir = task_dir / RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately NOT under run_dir: see the module docstring.
    workspace = workspace_root(task_dir) / run_id
    workspace.mkdir(parents=True, exist_ok=True)

    restored = []
    if resume_checkpoint is not None:
        # Never work directly inside the parent's checkpoint.  A resume is a new
        # run with its own scratch workspace; the previous run must remain an
        # immutable historical record even if this continuation corrupts files,
        # crashes, or is interrupted again.
        restored = restore_checkpoint(resume_checkpoint, workspace)

    # Everything from environment/ is staged, binaries included: with
    # freecadcmd and the SolidWorks session available they are openable, and
    # that is the whole point of running in a workspace.
    env_dir = task_dir / "environment"
    staged = sorted(f for f in env_dir.rglob("*") if is_task_file(f)) \
        if env_dir.is_dir() else []

    # Keep two views of the task inputs from this point on.
    #
    # `all_staged` is the complete set of files that belong to the task itself.
    # It is used later when workspace artifacts are collected, so a restored
    # input is not mistaken for a file created by the resumed agent.
    #
    # `staged` is allowed to become smaller on resume: it contains only the
    # inputs that still need to be copied into the new workspace.  Files already
    # restored from the parent checkpoint must NOT be staged again, because doing
    # so would overwrite the exact input version the parent session worked on.
    all_staged = list(staged)
    if resume_checkpoint is not None:
        restored_names = set(restored)

        # The checkpoint is the authoritative state of the interrupted session.
        # Preserve every task input already present there.  Only stage files that
        # the checkpoint did not contain, e.g. an auxiliary prompt file added
        # after the parent run.
        staged = [f for f in staged if f.name not in restored_names]

    
    # Images go inline -- a render is only useful if the model can see it,
    # and they are small. PDFs do NOT: a drawing set is megabytes, it rides
    # in EVERY turn once it is in the history, and it is the single largest
    # driver of the input-token quota. One 1.25 MB PDF inline preceded a
    # request that never returned at all. It is staged in the workspace
    # instead, where the agent reads the pages it actually needs.
    inline = buckets["image_files"]

    backend, backend_note = choose_backend(task_dir, workspace, run_id,
                                           allow_host=args.allow_host)
    if backend is not None:
        if isinstance(backend, DockerBackend):
            backend.start()
        tools = backend.describe(task_dir)
        tools = {k: v for k, v in tools.items() if v}

    prompt = build_prompt(task_dir, candidate_name, tools)

    if resume_checkpoint is not None:
        # Restoring the files is only half of resume.  The model itself starts a
        # fresh provider session, so without an explicit continuation note it may
        # see the normal task prompt and rebuild everything from scratch.  Tell it
        # that the existing workspace is previous work to inspect and continue.
        parent_status = (resume_meta or {}).get("status", "unknown")
        parent_turns = (resume_meta or {}).get("turns")
        parent_seconds = (resume_meta or {}).get("solve_seconds")

        prompt += (
            "\n\n## Continuation of an interrupted run\n\n"
            "This is NOT a fresh attempt.  You are continuing work from a previous "
            "agent session.  Its complete saved workspace has been restored into "
            "your current working directory.\n\n"
            "Before doing new work, inspect what is already here.  In particular, "
            f"look for the current `{candidate_name}`, build scripts, measurement "
            "or verification scripts, renders, notes and other diagnostic files.  "
            "Reuse and improve the existing work instead of rebuilding the task "
            "from scratch unless inspection shows that the previous approach is "
            "unusable.\n\n"
            f"Parent run: `{args.resume}`\n"
            f"Parent status: `{parent_status}`\n"
            f"Parent turns: `{parent_turns}`\n"
            f"Parent solve time: `{parent_seconds}s`\n\n"
            f"The same requirement still applies: your final deliverable must be "
            f"saved as `{candidate_name}` in the current working directory."
        )

    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    rel = task_dir.relative_to(ROOT).as_posix()
    max_turns = args.max_turns or AGENT_MAX_TURNS
    print(f"task      : {rel}")
    print(f"model     : {model_label}")
    print(f"deliver   : {candidate_name}")
    if resume_checkpoint is not None:
        print(f"restored  : {len(restored)} item(s) from parent checkpoint")
        print(f"staged    : {len(staged)} additional file(s) into workspace/")
    else:
        print(f"staged    : {len(staged)} file(s) into workspace/")
    print(f"vision    : {len(inline)} inline")
    print(f"toolchain : {', '.join(k for k in tools if k != 'python')}")
    print(f"sandbox   : {tools.get('backend', 'host (NOT isolated)')}")
    if backend_note:
        print(f"  WARNING : {backend_note}")
    for k, v in tools.items():
        if k.endswith("_BROKEN"):
            print(f"  WARNING : {k[:-7]} {v}")
    print(f"turns     : {max_turns}")
    print(f"run dir   : {run_dir.relative_to(ROOT).as_posix()}")
    print(f"workspace : {workspace}")

    if args.resume:
        print(f"resume    : {args.resume}")
        print(f"checkpoint: {resume_checkpoint.relative_to(ROOT).as_posix()}")

    # Before the agent exists: it is told to attach to the shared SolidWorks
    # session, so nothing of ours should be open in it.
    seal_solidworks(tools)

    print("solving ...")

    previous_bash = install_bash_backend(backend) if backend else None

    # Metadata normally arrives only when run_agent() returns. Keep a separate
    # live channel as well so an interrupted or crashed harness can leave behind
    # the progress it had already measured.
    from common import agent_workspace as ws
    progress = ws.AgentProgress()

    t0 = time.time()
    interrupted = False
    solve_error = None
    try:
        answer, solve_meta = run_agent_for(
            mod, route, variant, prompt, cwd=workspace, files=staged,
            inline=inline, require_file=candidate_name, max_turns=max_turns,
            progress=progress,)

    except KeyboardInterrupt:
        # Ctrl-C is a recoverable interruption, not a failed CAD attempt.
        #
        # run_agent() did not return, so its normal metadata is unavailable.
        # Production harnesses publish their live state through AgentProgress;
        # recover that snapshot here before adding try_model's own interruption
        # fields.  A harness that has not adopted AgentProgress yet simply leaves
        # an empty snapshot, preserving the old behaviour during migration.
        interrupted = True
        solve_error = "KeyboardInterrupt: interrupted by user"

        solve_meta = progress.snapshot()
        solve_meta["error"] = solve_error
        solve_meta["interrupted"] = True

        (run_dir / "error.txt").write_text(solve_error, encoding="utf-8",)
        print("\nINTERRUPTED: saving the agent's current work before exit...")
        answer = ""

    
    except Exception as exc:                                # noqa: BLE001
        # A route that dies mid-session (a CLI's own turn limit, a dropped
        # connection) may already have saved the deliverable. Record the
        # error and carry on to the harvest: the agent's files are copied
        # back and whatever it left at the deliverable name is kept, so
        # an hour of work is not discarded for how the session ended.
        solve_error = f"{type(exc).__name__}: {exc}"

        solve_meta = progress.snapshot()
        solve_meta["error"] = solve_error

        (run_dir / "error.txt").write_text(solve_error, encoding="utf-8")
        print(f"\nsolve FAILED: {solve_error}")
        answer = ""
    finally:
        # Always, even on the failure path: a leaked container holds a bind
        # mount on the workspace and the next run cannot clean it up.
        if previous_bash is not None:
            restore_bash_backend(previous_bash)
        if backend is not None:
            backend.stop()
    elapsed = round(time.time() - t0, 1)
    # A resume is a continuation of the same benchmark attempt.  Keep the current
    # session's measurements separately, but also accumulate all work inherited
    # from the parent chain so the final score cannot appear artificially cheap.
    previous_turns, previous_seconds, previous_sessions = resume_totals(resume_meta)

    session_turns = (solve_meta or {}).get("turns") or 0
    cumulative_turns = previous_turns + session_turns
    cumulative_solve_seconds = round(previous_seconds + elapsed, 1)
    resume_sessions = previous_sessions + 1

    (run_dir / "raw_response.md").write_text(answer or "", encoding="utf-8")
    # Everything the route reported beyond the transcript -- cost, token
    # usage, model round trips, harness version, tool list, errors -- for
    # whichever route ran. Routes differ in what they report; all of it is
    # kept rather than a hand-picked subset.
    solve_extra = {k: v for k, v in (solve_meta or {}).items()
                   if k not in ("transcript", "native_logs")}
    native_kept = keep_native_logs_in_run(solve_meta, run_dir)
    (run_dir / "transcript.json").write_text(
        json.dumps((solve_meta or {}).get("transcript", []), indent=2,
                   default=str), encoding="utf-8")
    graded, record, note = harvest(workspace, run_dir, candidate_name, answer)

    # Use `all_staged`, not the filtered `staged` list here.
    #
    # On a resumed run an input such as input.py may have come from the parent
    # checkpoint and therefore no longer appears in `staged`.  It is still a task
    # input, however.  Passing only `staged` would make
    # keep_workspace_artifacts() misclassify that restored input as something the
    # agent created and copy it into workspace/ as an agent artifact.
    produced = keep_workspace_artifacts(workspace, run_dir, all_staged)

    checkpoint = None
    if interrupted or solve_error or _exhausted(solve_meta, max_turns):
        checkpoint = save_checkpoint(workspace, run_dir)
        print(f"checkpoint: saved to {checkpoint.relative_to(ROOT).as_posix()}")

    meta = {"task": rel, "model": args.model, "model_label": model_label,
            "run_id": run_id, "solve_seconds": elapsed,
            "turns": (solve_meta or {}).get("turns"),
            # `turns` and `solve_seconds` above describe this model session only.
            # These cumulative fields describe the whole attempt across interruptions and
            # resumes and are the numbers that should be used for benchmark comparisons.
            "cumulative_turns": cumulative_turns,
            "cumulative_solve_seconds": cumulative_solve_seconds,
            "resume_sessions": resume_sessions,
            "nudged": (solve_meta or {}).get("nudged"),
            "budget_exhausted": _exhausted(solve_meta, max_turns),
            "interrupted": interrupted,
            # Reaching the tool-call ceiling is neither a successful completion nor a
            # provider failure.  Keep it as its own status so reports and resume tooling
            # can distinguish "the agent chose to finish" from "the harness stopped it".
            "status": (
                "interrupted" if interrupted
                else "failed" if solve_error
                else "budget_exhausted" if _exhausted(solve_meta, max_turns)
                else "completed"
            ),
            "resume": {
                "parent_run": args.resume,
                "restored_from": (
                    resume_checkpoint.relative_to(ROOT).as_posix()
                    if resume_checkpoint is not None else None
                ),
            } if args.resume else None,
            "resumable": checkpoint is not None,
            "checkpoint": (
                checkpoint.relative_to(ROOT).as_posix()
                if checkpoint is not None else None
            ),
            "wants": candidate_name,
            "candidate": record.name if record else None,
            "built_at": str(graded) if graded else None,
            "candidate_note": note,
            "staged": [f.name for f in staged],
            "toolchain": tools,
            "sandbox": tools.get("backend", "host (NOT isolated)"),
            "sandbox_warning": backend_note,
            "after_files": sorted(q.name for q in (run_dir / SOLUTION_DIR).iterdir())
            if (run_dir / SOLUTION_DIR).is_dir() else [],
            "workspace": str(workspace),
            "workspace_artifacts": produced,
            "native_logs": native_kept,
            "solve": solve_extra}

    if note:
        print(f"note      : {note}")
    if solve_error and graded is not None:
        print("note      : the solve failed, but the agent had saved "
              f"{candidate_name}; keeping what it left")

    (run_dir / "report.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(meta), encoding="utf-8")

    print()
    if graded is not None:
        print(f"DONE      : {(run_dir / SOLUTION_DIR).relative_to(ROOT).as_posix()}/"
              f"{record.name} -- open it and judge it")
    else:
        print(f"NO OUTPUT -- {note}")
    print(f"report    : {(run_dir / 'report.md').relative_to(ROOT).as_posix()}")

    shutil.rmtree(workspace, ignore_errors=True)
    if solve_error and graded is None:
        return 2
    return 0 if graded is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
