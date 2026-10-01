# cad-eval-set

A place to try out candidate CAD tasks. A task is a "before" (the files
an engineer starts from) and a prompt. You run a frontier model against
it, open the "after" it produced, and decide by eye whether it is good
enough or a failure. There are no reference solutions, adversarial
examples or automatic graders here.

New to the terminal, or coming from SolidWorks rather than code? Read
[GETTING_STARTED.md](cad-eval-set/GETTING_STARTED.md) first; it walks through every
step on Windows.

## Setup

```bash
# put the .env you were given at the git root, one level above this folder
pip install -r env_requirements.txt   # the model SDKs and agent tooling
```

`.env` lives at the git root, one level above this folder. FreeCAD tasks
also need `FREECAD_CMD` pointing at a `freecadcmd` binary; SolidWorks
tasks need Windows with SolidWorks running.

Runs default to the Docker container for the task's CAD program so every
run has the same toolchain. Build the base image once from this folder:

```bash
docker build -f common/docker/step-base.Dockerfile     -t step-base:latest .
docker build -f common/docker/cadquery-base.Dockerfile -t cadquery-base:latest .
docker build -f common/docker/freecad-base.Dockerfile  -t freecad-base:latest .
```

Without Docker, `--allow-host` runs the model on this machine instead.

## Adding a task

```bash
cp -r templates/task_template tasks/12_widget_bracket
```

Task folders are `tasks/<n>_<slug>/` where `n` is the task's number on the
tracking sheet. The CAD program (`CadQuery`, `FreeCAD`, `SolidWorks`,
`STEP`, `Blender`) is `program` in the task's `task.toml`; it picks the
container the model runs in. Inside, fill in:

| Path | What goes there |
|---|---|
| `environment/` | The "before" files the solver receives, main model renamed `input.<ext>`. Nothing else. |
| `instruction.md` | The prompt, verbatim. It must name the deliverable as `/app/<name>`. |
| `task.toml` | `program`, a name and a one-line description. |

That is the whole task. [filetree.MD](cad-eval-set/filetree.MD) is the full layout
spec, including how to bring a task in from a Drive folder. The "before"
CAD files are committed with the task; nothing else holds them.

## Trying a model on it

```bash
python3 tools/try_model.py claude tasks/12_widget_bracket
python3 tools/try_model.py gpt 12_widget_bracket
python3 tools/try_model.py claude:fable51 tasks/12_widget_bracket --max-turns 200
```

The model is `route[:variant]`: `claude[:sonnet5|opus55|fable51]`, `gpt`,
`gemini[:pro|flash]`, `grok`, `kimi`, `deepseek`, `glm`. The model gets a
real workspace with the task's inputs, a shell and the machine's CAD
toolchain, and works until it writes the deliverable or runs out of turns.

Everything lands in `<task>/_runs/<model>_<timestamp>/`, gitignored:

- `after/`: the file the prompt asked for, plus any parts that go with it
- `workspace/`: the model's scratch files
- `prompt.txt`, `raw_response.md`, `transcript.json`: what was sent, what
  the model said, and every command it ran
- `report.md`: turns, time, and whether it produced the file at all

Open `after/` and judge it as you would a colleague's work. Nothing is
scored.

## Deciding

A task is interesting when a strong model fails it on the engineering,
not on the tooling. Rough guide:

- **Solved cleanly on the first try**: too easy as written. Tighten the
  prompt or the constraints before spending time on an "after".
- **Failed because it could not open the file, find the tool, or ran out
  of turns**: an environment problem. Fix `environment/` or the prompt and
  run again; this says nothing about difficulty yet.
- **Produced something plausible that an engineer would reject**: the good
  case. Note what it got wrong.
- **Different models fail in different ways**: also good. Run two or three
  before deciding.

Record the verdict for each run, good enough or failure and why, on the
tracking sheet.

## Layout

```
cad-eval-set/
├── filetree.MD, GETTING_STARTED.md, CLAUDE.md
├── env_requirements.txt
├── common/                  # model routes and the agent's sandbox
│   ├── call_claude.py, call_gpt.py, call_gemini.py, ...
│   ├── agent_workspace.py, agent_cli.py, workspace_mcp.py
│   ├── solidworks_session.py
│   └── docker/              #   base images per CAD program
├── tools/
│   └── try_model.py         #   run one model on one task, keep its after/
├── templates/task_template/ # copy this to start a task
└── tasks/                   # tasks go here, one folder each
```

## Relationship to openai-eval-set

`common/` and `tools/` are copies of the same files in openai-eval-set as
of 30 September 2026. Fixes made there should be copied here and vice
versa. This repo differs on purpose: tasks live in one `tasks/` folder with
the CAD program in `task.toml` rather than in per-program folders, there
are no solutions, examples or graders, and only what `tools/try_model.py`
needs to run a model is kept.
