# Getting started (for SolidWorks engineers)

This guide takes you from nothing to running an AI model on a task you
made, on a Windows PC with SolidWorks. You do not need to know how to
program. You will type a handful of commands into a terminal; each one is
written out in full, and you can copy and paste them.

If anything here does not match what you see on screen, stop and ask.
That is a problem with the guide, not with you.

## 1. What you are setting up

- **The repo.** A folder of files, shared through GitHub, that holds the
  tools and the tasks. "Cloning" it means downloading a copy.
- **Python.** The language the tools are written in. You install it once.
- **A terminal.** A window where you type commands. On Windows, use
  **PowerShell** (press the Windows key, type `powershell`, press Enter).
- **A `.env` file.** A small text file with the keys the AI models need.
  You will be given the values; never share them or put them in a task.

## 2. Install the three programs (once)

Open PowerShell and paste these one at a time. Each one downloads and
installs a program; say yes to any prompt.

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id GitHub.cli -e
```

Close PowerShell and open it again so it picks up the new programs. Check
they worked:

```powershell
git --version
python --version
```

Each should print a version number. If `python --version` opens the
Microsoft Store instead, run this and try again:

```powershell
winget install --id Python.Python.3.11 -e --override "/passive PrependPath=1"
```

## 3. Download the repo

Pick where you want it. `C:\Dev` is a good choice. Then:

```powershell
mkdir C:\Dev
cd C:\Dev
gh auth login
git clone <REPO URL> cad-eval-set
cd cad-eval-set
```

`gh auth login` asks you to sign in to GitHub in your browser the first
time. Replace `<REPO URL>` with the address from the green **Code** button
on the repo's GitHub page.

You now have `C:\Dev\cad-eval-set`. Inside it is another folder also
called `cad-eval-set`; that inner one is where the tools live. The outer
one is where your keys go.

## 4. Add your keys

You were given a `.env` file with the keys in it. Put it in
`C:\Dev\cad-eval-set` (the outer folder), named exactly `.env`. To check
or edit it:

```powershell
notepad .env
```

Each key is a line of the form `NAME=value`. The two that matter most
for trying models are `AZURE_API_KEY` and `AZURE_API_KEY_GTM_RESEARCH`.
Leave `FREECAD_CMD` empty unless you have FreeCAD. Save and close Notepad.

If Windows Explorer hides file extensions, the copy may be called
`.env.txt` without you seeing it. Check the name in PowerShell with `dir`.

## 5. Install the Python packages (once)

```powershell
cd C:\Dev\cad-eval-set\cad-eval-set
pip install -r env_requirements.txt
```

This takes a few minutes and prints a lot. It is fine as long as it ends
without the word `ERROR` in red. The first run in step 7 is also the
check that the keys in `.env` work; a missing key is reported there.

## 6. Make a task

A task is a folder with two things: the "before" (the SolidWorks files
the engineer would start from) and the prompt they would be given.
There is a finished example at `tasks\playstation_controller`: a
controller body and a prompt asking for it to be widened and made
left-handed. You can run a model on it as-is in step 7 to see the whole
thing work. To make your own, copy `tasks\template` to a short name
for your task, for example `tasks\widget_bracket`, then, in Windows
Explorer, open the copy and:

1. **`environment\`**: put the "before" model here. Rename the main part
   or assembly to `input.sldprt` or `input.sldasm`. For an assembly, put
   its parts here too, with their real names. Nothing else goes in this
   folder.
2. **`instruction.md`**: open it in Notepad and replace the text with your
   prompt, worded exactly as you would give it to another engineer. Keep
   a last line that names the result file as `/app/<name>`, for example:
   `Save the finished assembly as /app/solution.sldasm.`

That is the whole task: a before and a prompt. There is no answer key.

## 7. Run a model on it

Start SolidWorks and leave it open. The model drives your SolidWorks
through its API, so it has to be running. Then:

```powershell
cd C:\Dev\cad-eval-set\cad-eval-set
python tools\try_model.py claude tasks\playstation_controller
```

The model reads your prompt, gets a working folder with your input files,
and starts working. You will see each step it takes scroll past. This can
take anywhere from a few minutes to over an hour. Do not use SolidWorks
yourself while it runs.

To try a different model, change the first word:

```powershell
python tools\try_model.py gpt tasks\playstation_controller
python tools\try_model.py gpt:astra tasks\playstation_controller
python tools\try_model.py claude:fable51 tasks\playstation_controller
```

You may see a warning that the run is "on the HOST". For a SolidWorks
task that is expected: SolidWorks cannot run in a container.

## 8. Look at what it did

Inside your task folder a new folder `_runs` appears, with one subfolder
per run named after the model and the time. Open the newest one. In it:

- **`after\`**: what it built, the file your prompt asked for plus any
  parts that go with it. Open it in SolidWorks and judge it as you would
  a colleague's work: good enough, or a failure.
- **`raw_response.md`**: what the model said it did, in its own words.
- **`transcript.json`**: every command it ran, if you want to see how it
  got there.
- **`report.md`**: how long it took, how many steps, and whether it
  produced the file at all.

## 9. Decide

The question is: **did it fail on the engineering, or on the tooling?**

- **It built the right thing on the first try.** The task is too easy as
  written. Make the prompt or the constraints harder and run again.
- **It could not open the file, got stuck, or ran out of turns without
  saving anything.** That is a setup problem, not a hard task. Check the
  input files and the prompt and run again.
- **It built something plausible that you would send back.** This is the
  good outcome. Write down what is wrong with it.
- **Two models fail in different ways.** Also good. Run two or three
  before deciding.

Record the verdict for each run on the tracking sheet: good enough, or a
failure, and why.

## Things that go wrong

**`python` is not recognised.** Python did not get added to your PATH.
Re-run the install line at the end of step 2, then close and reopen
PowerShell.

**`no Azure credential`** or **`key is missing`.** The `.env` file is in
the wrong place or has the wrong name. It must be
`C:\Dev\cad-eval-set\.env`, in the outer folder, with no `.txt` ending.

**`no task matching`.** The task folder is not under `tasks\`, or the
name is misspelled. The name is matched case-insensitively.

**`cannot tell what ... wants written`.** The last line of
`instruction.md` does not name the result as `/app/<something>`.

**The model says SolidWorks is not running.** Start SolidWorks, then run
again. If SolidWorks is open and it still says so, close SolidWorks
fully (check Task Manager for `SLDWORKS.exe`) and start it again.

**Something else.** Copy the last twenty lines from the terminal and send
them along with the task name.
