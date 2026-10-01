# Task directory layout

Every task lives at `tasks/<num>_<task_name>/`, where `<num>` is the task's
number on the tracking sheet. Which CAD program the task uses (`CadQuery`,
`FreeCAD`, `SolidWorks`, `STEP`, `Blender`) is `program` in its task.toml;
there are no per-program folders. `tasks/playstation_controller/` is a
finished example and `tasks/template/` a blank one to copy; other task
folders are not committed.

```
tasks/<num>_<task_name>/
├── task.toml            # program, name, one-line description
├── instruction.md       # the prompt (or a pointer to a prompt file in environment/)
├── environment/         # the "before": everything the solver receives
└── _runs/               # written by tools/try_model.py, gitignored
    └── <model>_<timestamp>/
        ├── after/       # what the model built -- open this and judge it
        ├── workspace/   # its scratch files
        ├── prompt.txt, raw_response.md, transcript.json
        └── report.md    # how the run went: turns, time, what it produced
```

## environment/  — what the solver gets

- The **before model**, renamed to `input.<ext>` (`input.sldprt`, `input.FCStd`,
  `input.py`, `input.STEP`). Multi-file assemblies go directly in
  `environment/` with the top-level assembly renamed `input.sldasm` and the
  component files keeping their real names.
- Any file the prompt references: drawings, a prompt deck renamed
  `input.ppt`, a brief renamed `input.docx`, PDFs.
- **Nothing else.** No author notes, no screenshots that aren't referenced by
  the prompt, no telemetry.

## instruction.md  — the prompt

The author's prompt text, verbatim. It must name the deliverable as
`/app/<name>`; `tools/try_model.py` reads that line to know which file to
keep as the "after". If the real prompt is a binary file (ppt/docx/pdf),
import it into `environment/` and have instruction.md point at it.

## There is no solution/, examples/ or tests/

A task here is a before and a prompt. The "after" is whatever the model
produces, and a person looks at it and decides whether it is good enough
or a failure. Nothing is graded automatically.

## Copying a task in from Google Drive — checklist

0. **Inventory first.** List EVERY file in the Drive folder — top level and all
   subfolders — and write down a disposition for each one (imported into
   environment/ / pasted into instruction.md / skipped and why) BEFORE
   moving anything. Prompt files are small and easy to lose next to
   hundreds of model files.
1. **instruction.md**: look for `task.txt`, `tasks.txt`, `notes.txt`,
   `prompt.txt`, or a Google Doc, at ANY level of the folder. Paste it
   verbatim.
2. **environment/**: the "Before" model (possibly named "Input", "Initial",
   "Original", "Unedited" etc.) → `input.<ext>`. Prompt deck/doc →
   `environment/input.ppt` / `input.docx`. Drawings the prompt cites come
   too.
3. **Skip**: "After" / "Solution" models, "Defective" / "Adversarial"
   variants, checklists and rubrics, `.swp`, `.mp4`, `.jsonl`, `~$*` lock
   files, screen recordings, click telemetry, and `.FCMacro` build steps.
4. **task.toml**: set `program`, `name`, `description`.
5. **Cross-check**: re-read the inventory and confirm every file landed
   where its disposition said.
