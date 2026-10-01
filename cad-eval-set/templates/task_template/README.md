# Task template

Copy this folder into `tasks/` and rename it `<n>_<slug>`:

    cp -r templates/task_template tasks/12_widget_bracket

Then fill in:

1. `environment/` -- the "before": the files the solver receives, with the
   main model renamed `input.<ext>`. Nothing else goes here.
2. `instruction.md` -- the prompt, exactly as an engineer would receive it.
   It must name the deliverable as `/app/<name>`; that line is how
   `tools/try_model.py` knows what file to keep as the "after".
3. `task.toml` -- set `program`; it picks the container the model runs in.

Then run `python3 tools/try_model.py claude tasks/<n>_<slug>` and open
`_runs/<run>/after/` to judge the result.

Delete this README once the task is real.
