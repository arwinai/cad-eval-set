# Working in this repo

This is the staging ground for candidate CAD tasks. Read README.md first.

- A task is a before (`environment/`) and a prompt (`instruction.md`).
  There are no solutions, examples, or graders: a person opens the model's
  `after/` and judges it. Do not add grading code.
- A task under `tasks/<n>_<slug>/` follows `filetree.MD`; its CAD program
  is `program` in task.toml, not a parent folder. New tasks start as a
  copy of `tasks/1_playstation_controller/`.
- Task folders are not committed; `.gitignore` ignores everything under
  `tasks/` except `1_playstation_controller/`. Same for `.env` and `_runs/`.
- `tools/try_model.py` is the only thing that should run a model on a task.
  It works out of tree in the program's container.
- SolidWorks: one job per machine, with SolidWorks running.
- `common/` mirrors openai-eval-set's agent routes; port fixes both ways.
