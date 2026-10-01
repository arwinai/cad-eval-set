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

CadQuery, FreeCAD, STEP and Blender tasks run the model inside a Docker
container so every run has the same toolchain. Install Docker Desktop and
have it running; the first run for each program builds its image from
`common/docker/` (a few minutes). SolidWorks tasks run on the host, since
SolidWorks cannot be containerised. Without Docker, `--allow-host` runs
any task on this machine instead.

## Adding a task

`tasks/playstation_controller/` is a finished example: a SolidWorks
controller body as the before, and a prompt asking for it to be widened
and converted to a left-handed layout. `tasks/template/` is the same
shape with the contents blanked out. Copy the template to
`tasks/<slug>/` and fill it in. Other task folders are not committed;
they live on your machine. The CAD program (`CadQuery`, `FreeCAD`,
`SolidWorks`, `STEP`, `Blender`) is read off the before model's
extension and picks the container the model runs in. Inside, fill in:

| Path | What goes there |
|---|---|
| `environment/` | The "before" files the solver receives, main model renamed `input.<ext>`. Nothing else. |
| `instruction.md` | The prompt, verbatim. It must name the deliverable as `/app/<name>`. |

That is the whole task. [filetree.MD](cad-eval-set/filetree.MD) is the full layout
spec, including how to bring a task in from a Drive folder.

## Trying a model on it

```bash
python3 tools/try_model.py claude tasks/playstation_controller
python3 tools/try_model.py gpt playstation_controller
python3 tools/try_model.py gpt:astra playstation_controller
python3 tools/try_model.py claude:fable51 tasks/playstation_controller --max-turns 200
```

The model is `route[:variant]`: `claude[:sonnet5|opus55|fable51]`,
`gpt[:gpt56|astra]` (astra is GPT-6),
`gemini[:pro|flash]`, `grok`, `kimi`, `deepseek`, `glm`. The model gets a
real workspace with the task's inputs, a shell and the machine's CAD
toolchain, and works until it writes the deliverable or runs out of turns.

The example is a SolidWorks task, so it needs Windows with SolidWorks
running. Everything lands in `<task>/_runs/<model>_<timestamp>/`, gitignored:

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
└── tasks/                   # tasks go here, one folder each (not committed)
    ├── playstation_controller/   # a finished example
    └── template/                 # copy this to start a task
```

## Relationship to openai-eval-set

`common/` and `tools/` are copies of the same files in openai-eval-set as
of 30 September 2026. Fixes made there should be copied here and vice
versa. This repo differs on purpose: tasks live in one `tasks/` folder with
the CAD program inferred from the input file rather than from
per-program folders, there
are no solutions, examples or graders, and only what `tools/try_model.py`
needs to run a model is kept.
