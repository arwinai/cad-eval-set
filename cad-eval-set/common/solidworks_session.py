"""Shared talking-to-SolidWorks plumbing (Windows + pywin32 only).

The transport layer every SolidWorks harness shares: attaching to the
running application, finding/opening/activating documents, byref VARIANT
helpers for the API's out-parameters, and the `z()` null-safe call idiom.
Measurement logic stays with each harness (or solidworks_capture/measure);
this module only owns the session.

UNVALIDATED ON WINDOWS: written on a non-Windows machine; run
SolidWorks/VALIDATION.md before trusting grades that flow through it.
"""
from __future__ import annotations

import os
import sys

# Why the reason is kept: on Linux this import fails because pywin32 ships no
# wheels there, and "runs only on Windows" is the whole story. On Windows the
# SAME ImportError means something else entirely -- pywin32 installed into a
# different interpreter than the one on PATH, or installed but unable to load
# pythoncomXX.dll because its post-install step never ran. Reporting the
# platform verdict for both is the failure mode this repo keeps paying for:
# one message reachable from two unrelated causes. Keep the exception.
_IMPORT_ERROR = None

try:
    import pythoncom
    import pywintypes
    import win32com.client
    import win32com.client.dynamic
    from win32com.client import VARIANT
except ImportError as exc:   # non-Windows: importable, unusable
    _IMPORT_ERROR = exc
    pythoncom = None
    pywintypes = None
    win32com = None
    VARIANT = None

def why_no_win32():
    """Why pywin32 is unusable in THIS process, in one line.

    "Not installed" and "installed and its DLL will not load" are
    different faults with different fixes, and a caller that printed only
    "measuring needs pywin32" made them look identical. A run that stops
    for the second reason sent whoever read it to `pip install pywin32`,
    which changes nothing.
    """
    if win32com is not None:
        return None
    import sys
    return (f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"
            if _IMPORT_ERROR is not None else
            f"pywin32 did not import and said nothing; python is "
            f"{sys.executable}")


# swDocumentTypes_e
DOC_PART = 1
DOC_ASSEMBLY = 2
# swOpenDocOptions_e
OPEN_SILENT = 1
OPEN_READONLY = 2
# swRebuildOnActivation_e
DONT_REBUILD_ON_ACTIVATE = 1
# swSaveAsVersion_e / swSaveAsOptions_e
SAVE_AS_CURRENT = 0
SAVE_AS_SILENT = 1


def z(member):
    """Call `member` if callable, tolerating COM members that surface as
    either properties or methods depending on the typelib state."""
    if not callable(member):
        return member
    try:
        return member()
    except Exception:
        return member


def byref_i4(value=0):
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, value)


def byref_bool(value=False):
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BOOL, value)


def byref_bstr(value=""):
    """A VT_BYREF|VT_BSTR out-parameter.

    IPartDoc::GetMaterialPropertyName2 hands the material database back
    through one. Handed a plain Python string instead it raises "type
    mismatch" -- which reads, from the outside, exactly like a build whose
    API lacks the call, so the material silently comes from whatever
    fallback answers next. That fallback (MaterialIdName) reports the
    default appearance rather than an assigned material, so the mistake
    ends in a graded number, not an error."""
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_BSTR, value)


def redispatch(obj):
    """Re-wrap a COM object so late-bound attribute access works on
    interfaces win32com typed too narrowly (PartDoc methods on a
    ModelDoc2, specific features, ...)."""
    if obj is None:
        return None
    if isinstance(obj, win32com.client.CDispatch):
        return win32com.client.Dispatch(obj._oleobj_)
    return win32com.client.Dispatch(obj)


def _no_pywin32_message():
    """Say which interpreter could not import pywin32, and why.

    On Windows this is nearly always the interpreter, not the platform:
    `pip install pywin32` went to one Python and the harness is running on
    another. The executable path is what settles that argument, so it is in
    the message rather than in a follow-up round of questions.
    """
    if sys.platform != "win32":
        return ("pywin32 not available - SolidWorks harnesses run only on "
                f"Windows (this is {sys.platform})")
    return (
        "pywin32 did not import, but this IS Windows - so this is about the "
        "interpreter, not the platform.\n"
        f"  interpreter: {sys.executable}\n"
        f"  python:      {sys.version.split()[0]}\n"
        f"  import error: {_IMPORT_ERROR}\n"
        "  fix: install it into THIS interpreter --\n"
        f'      "{sys.executable}" -m pip install pywin32\n'
        "      then, if the error was a DLL load failure, run\n"
        f'      "{sys.executable}" -m pywin32_postinstall -install\n'
        "  note: env_requirements.txt pins pywin32==306, which has no wheel "
        "for Python 3.13+; on a newer interpreter install a newer pywin32 "
        "or run the harness on the Python that already has it.")


def attach(unattended=True):
    """The running SolidWorks application, or raise.

    Arms unattended operation by default (see arm_unattended). Every attach
    in this repository is a grading or reconnaissance attach, and every one
    of them rebuilds documents -- so the alternative is remembering to wrap
    each of the six call sites, and the sixth is the one that hangs a
    nightly run. Pass unattended=False for interactive debugging. Never launches one:
    grading against a live session is the contract for these tasks.

    Deliberately forces DYNAMIC (late-bound) dispatch via
    win32com.client.dynamic.Dispatch on the raw IDispatch pointer, instead
    of win32com.client.GetActiveObject. GetActiveObject auto-upgrades to
    EARLY-bound dispatch whenever a makepy'd typelib module is sitting in
    the gen_py cache (which sw_constant()'s named-constant lookup requires
    and creates) -- and early-bound [in,out] byref params use a different
    calling convention (extra values in the return tuple) than the
    VARIANT(VT_BYREF, ...) idiom byref_i4/byref_bool implement below.
    Confirmed on the grading box 2026-08-15: once gen_py had the SldWorks
    typelib cached, plain GetActiveObject silently early-bound and
    OpenDoc6/GetErrorCode2/SaveAs2 broke with
    "TypeError: int() argument must be ... not 'VARIANT'". Forcing dynamic
    dispatch here keeps the byref helpers correct regardless of whether
    gen_py has been populated (by us or anything else on the box)."""
    if win32com is None:
        raise RuntimeError(_no_pywin32_message())
    try:
        clsid = pywintypes.IID("SldWorks.Application")
        raw = pythoncom.GetActiveObject(clsid).QueryInterface(
            pythoncom.IID_IDispatch)
        app = win32com.client.dynamic.Dispatch(raw)
    except Exception as exc:
        raise RuntimeError(
            "could not attach to a running SolidWorks (is it open?): "
            f"{exc}") from exc
    if unattended:
        arm_unattended(app)
    return app


def active_doc(app):
    return z(app.ActiveDoc)


def find_document(app, path):
    """An already-open document with the given path, or None."""
    target = str(path).lower()
    for doc in _open_docs(app):
        if _key(doc) == target:
            return doc
    return None


def dyn(obj):
    """Force dynamic (late-bound) dispatch on a COM object -- every
    harness was independently reimplementing this (as a local `_dyn`)
    because plain win32com.client.Dispatch() (which redispatch() above
    uses) silently upgrades to early-bound whenever a makepy'd typelib
    is sitting in the gen_py cache, exposing a different interface
    surface on some objects (e.g. IBody2 without GetFaces) and breaking
    the VARIANT(VT_BYREF, ...) byref idiom -- same landmine attach()'s
    docstring documents for the top-level Application object."""
    if obj is None:
        return None
    raw = obj._oleobj_ if hasattr(obj, "_oleobj_") else obj
    return win32com.client.dynamic.Dispatch(raw)


def close_all_documents(app):
    """Close every document currently open in this SolidWorks session.

    SolidWorks resolves referenced components by filename, not full
    path. Tasks whose environment/solution/examples folders each carry
    their own same-named copies of shared component files (a common
    pattern in this repo) hit a real bug because of this: if a document
    stays open from grading a previous candidate, the next assembly
    that references a component with that same filename silently
    reuses the stale already-open document instead of loading its own
    folder's copy, even though the assembly itself opens from the
    correct path. Confirmed live on 30_shampoo_bottle: this produced
    byte-identical geometry measurements across every adversarial
    example. Sweeping the whole session closed before every fresh open
    (see open_document) removes the possibility entirely -- a blunt
    instrument, but fine for harnesses that only ever need one
    assembly open at a time.

    A grading run must never stop on a modal dialog. CloseDoc alone can
    raise the "save changes?" prompt on a document the session dirtied --
    and every run dirties one, because ForceRebuild3 is how the rebuild
    census is taken. The prompt then blocks the COM thread until a human
    clicks it, which turns an unattended batch into a hung one.

    CloseAllDocuments(True) is the documented "close everything, discard
    unsaved" and does not prompt. Per-document CloseDoc stays as the
    fallback for builds where the bulk call refuses.

    Returns a diagnostic dict: which route ran, what it raised, what was
    still open afterwards. Returned rather than swallowed because the first
    attempt at this fix silently fell back to CloseDoc when the bulk call
    raised, and a swallowed exception is indistinguishable from a working
    fallback -- the same failure this repo has now paid for four times.
    """
    def _try(label, fn):
        key = label.split("(")[0]
        if key in _DEAD_ROUTES:
            return False
        try:
            r = fn()
            diag["routes"].append({"route": label, "returned": bool(r),
                                   "error": None})
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            # E_NOTIMPL / "not supported" is a property of the build, not
            # of this document. Learn it once.
            if _NOT_IMPLEMENTED in str(exc) or "not support" in str(exc).lower() \
                    or "не поддерживается" in str(exc).lower():
                _DEAD_ROUTES.add(key)
                text += "  [route disabled for this process]"
            diag["routes"].append({"route": label, "returned": None,
                                   "error": text})
        return not _open_titles(app)

    diag = {"before": _open_titles(app), "routes": [], "after": None,
            "count_before": document_count(app)}
    #: AN EMPTY ENUMERATION IS NOT AN EMPTY SESSION. The early return
    #: here used to fire on `before == []` alone, and on task 20 that is
    #: exactly what happened: nothing enumerated, twelve component parts
    #: still resident, every later assembly refused with
    #: swFileWithSameTitleAlreadyOpen. The bulk close is one COM call on
    #: an already-empty session, so the cost of running it anyway is
    #: nothing next to the cost of skipping it when it was needed.
    if not diag["before"] and not diag["count_before"]:
        _try("CloseAllDocuments(True)",
             lambda: z(app.CloseAllDocuments(True)))
        diag["after"] = _open_titles(app)
        return diag

    # CloseAllDocuments(True) first: one call for the whole session, and
    # measured on the grading box it is the one that actually works. The
    # per-document routes are the fallback.
    #
    # IModelDoc2::Quit led this list for one run, on the reasoning that it
    # promises "close without saving" most explicitly. It answers
    # E_NOTIMPL on this build -- for every open document, so a nine-model
    # batch spent ~300 failing COM calls and printed 300 identical errors
    # before the bulk call closed everything anyway. A route that is not
    # implemented is not a safer route; it is noise. Hence _DEAD_ROUTES:
    # an unsupported route is tried once per process, not once per
    # document forever.
    if _try("CloseAllDocuments(True)",
            lambda: z(app.CloseAllDocuments(True))):
        diag["after"] = []
        return diag
    for d in _open_docs(app):
        if _try("ModelDoc2.Quit()", lambda x=d: x.Quit()):
            break
    for title in _open_titles(app):
        if _try(f"QuitDoc({title!r})", lambda t=title: app.QuitDoc(t)):
            break
    for title in _open_titles(app):
        _try(f"CloseDoc({title!r})", lambda t=title: app.CloseDoc(t))
    diag["after"] = _open_titles(app)
    return diag


def quiet_mode(app, on=True):
    """Tell SolidWorks an API command is running, so it suppresses the
    dialogs it would otherwise raise at a user.

    Best-effort: the property is not present on every build, and its
    absence is not an error worth stopping a run for. Reported, not raised.
    """
    try:
        app.CommandInProgress = bool(on)
        return True
    except Exception:
        return False


_NOT_IMPLEMENTED = "-2147467263"      # E_NOTIMPL, as pywin32 renders it
_DEAD_ROUTES = set()                  # close routes this build does not have


def _docs_via_getdocuments(app):
    """Open documents through `GetDocuments`, HIDDEN COMPONENT DOCUMENTS
    INCLUDED. This is the route that tells the truth.

    MEASURED on task 20, and it cost a day. The walk below
    (GetFirstDocument/GetNext) reported an EMPTY session while twelve
    component parts from the previously graded folder were still
    resident: `close_all_documents` therefore decided there was nothing
    to close and returned without closing it, and every assembly opened
    after the first was refused with error 65536,
    swFileWithSameTitleAlreadyOpen -- because every candidate folder in
    that task ships its own copy of the same two dozen filenames, and
    SolidWorks resolves by filename. One model succeeded, thirteen
    failed, and the session insisted nothing was open.

    So both routes are read and unioned. A sweep that trusts a single
    enumeration is a sweep that can be told the room is empty.
    """
    raw = safe_call(lambda: z(app.GetDocuments))
    if raw is None:
        return []
    try:
        items = list(raw)
    except TypeError:
        items = [raw]
    return [dyn(x) for x in items if x is not None]


def _docs_via_walk(app):
    """Open documents through GetFirstDocument/GetNext. Kept because it
    is the only route on builds without GetDocuments -- not trusted on
    its own, see above."""
    out = []
    try:
        doc = z(app.GetFirstDocument)
    except Exception:
        return out
    while doc is not None:
        d = dyn(doc)
        out.append(d)
        try:
            doc = z(d.GetNext)
        except Exception:
            break
    return out


def document_count(app):
    """How many documents SolidWorks itself says are open, or None when
    the build will not say. A third opinion, cheaper than either walk and
    independent of both."""
    n = safe_call(lambda: z(app.GetDocumentCount))
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _key(d):
    """Identity for de-duplicating the two routes: the path, falling back
    to the title for a document that has never been saved."""
    k = safe_call(lambda: str(z(d.GetPathName) or ""), "") or ""
    if not k:
        k = safe_call(lambda: str(z(d.GetTitle) or ""), "") or ""
    return k.lower()


def _open_docs(app):
    """Every open document as a dispatchable object, by both routes."""
    out, seen = [], set()
    for d in _docs_via_walk(app) + _docs_via_getdocuments(app):
        if d is None:
            continue
        k = _key(d)
        if k and k in seen:
            continue
        if k:
            seen.add(k)
        out.append(d)
    return out


def open_documents(app):
    """(title, path) for every open document, BOTH enumeration routes.

    The public form of what the sweep reads, so a diagnostic that says
    "documents still open: []" is saying it on the same evidence the
    sweep acted on. A probe that asks a different, thinner question than
    the code it is diagnosing will confirm whatever that code believes.
    """
    out = []
    for d in _open_docs(app):
        out.append({"title": safe_call(lambda x=d: str(z(x.GetTitle)), "?"),
                    "path": safe_call(lambda x=d: str(z(x.GetPathName)), "")})
    return out


def _open_titles(app):
    """Titles of every open document, as a plain list.

    Taken before closing rather than holding the documents themselves:
    a CDispatch for a document that CloseAllDocuments already closed is a
    dangling pointer, and asking it for its own title is how a sweep ends
    in a COM error instead of a closed session."""
    out = []
    for d in _open_docs(app):
        t = safe_call(lambda x=d: str(z(x.GetTitle)))
        if t is not None:
            out.append(t)
    return out


def open_document(app, path, doc_type=None,
                  options=OPEN_SILENT | OPEN_READONLY):
    """(doc, opened_here). Reuses an already-open document with this
    exact path if found (e.g. the live document being graded);
    otherwise sweeps the session closed first (see close_all_documents)
    so no stale same-named component from a previous candidate/baseline
    can leak into this one, then opens fresh. doc_type defaults to
    inferring swDocASSEMBLY/swDocPART from the file extension.

    READ-ONLY by default, for two reasons that are really one.

    Grading must not change what it grades. SolidWorks rewrites parts it
    migrates on open, and that is not hypothetical here: it is how the
    `environment` folder of task 15 stopped matching its own manifest in
    19 files of 24, while `solution`, which nothing opened writable,
    still matches all 27. A grader that edits its inputs makes its own
    second run unrepeatable and the checksum audit meaningless.

    And a writable document that the session dirtied -- ForceRebuild3
    dirties every one -- raises the "save changes?" dialog on close, which
    blocks the COM thread until a human clicks it. Read-only documents
    cannot be saved, so the dialog has nothing to ask about.

    Callers that genuinely need to write pass options explicitly."""
    doc = find_document(app, path)
    if doc is not None:
        return dyn(doc), False
    close_all_documents(app)
    if doc_type is None:
        doc_type = (DOC_ASSEMBLY if str(path).lower().endswith(".sldasm")
                    else DOC_PART)
    errs, warns = byref_i4(), byref_i4()
    doc = app.OpenDoc6(os.path.abspath(str(path)), doc_type, options, "",
                       errs, warns)
    if doc is None:
        raise RuntimeError(f"could not open {path}")
    return dyn(doc), True


def activate(app, doc):
    """Make `doc` the active document without rebuilding it."""
    try:
        err = byref_i4()
        app.ActivateDoc3(z(doc.GetTitle), False,
                         DONT_REBUILD_ON_ACTIVATE, err)
    except Exception:
        pass


def sw_constant(name, fallback):
    """A named constant from the generated typelib module when available
    (win32com.client.constants requires a makepy'd SolidWorks typelib),
    else the documented numeric fallback.

    NOTE: attach() uses GetActiveObject, which returns a late-bound
    dispatch that does NOT populate win32com.client.constants on its own
    (unlike EnsureDispatch/makepy-run processes) -- so in practice this
    almost always falls through to `fallback`. Keep fallbacks verified
    against the real typelib, not guessed."""
    try:
        return getattr(win32com.client.constants, name)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# unattended operation: nothing may wait for a human
# ---------------------------------------------------------------------------

# Substrings that identify the "save your changes?" dialog, and the buttons
# that answer it with a no. Several languages because the grading box is not
# guaranteed to be an English install -- this repo has already been bitten
# once by assuming the console codepage.
_SAVE_DIALOG_HINTS = (
    "save", "сохран", "guardar", "enregistrer", "speichern", "salvar",
)
_DISMISS_CAPTIONS = (
    "don't save", "dont save", "do not save", "discard", "no",
    "не сохранять", "нет", "no guardar", "ne pas enregistrer",
    "nicht speichern", "não salvar", "nao salvar",
)
_DIALOG_CLASS = "#32770"          # the standard Win32 dialog class

# The OTHER modal that stops a grading run, and the reason it needs its own
# entry rather than a wider net on the one above.
#
# A part whose features do not all rebuild raises a warning when the
# document is opened or regenerated -- for task 8's reference, "unable to
# maintain existing wall faces, additional faces without draft were added
# to the model". It is a single-button notice: nothing is decided by it,
# the geometry is already what it is, and the only thing the button does
# is let the API carry on. Left unanswered it blocks the COM thread
# indefinitely, which turns a nine-model batch into a run that stops at
# the first one and waits for a person who is not watching.
#
# `CommandInProgress` suppresses many dialogs and does not suppress this
# one, which is why a watchdog is needed at all.
#
# Matched NARROWLY, and the narrowness is the safety: a dialog is only
# answered when its own text names a rebuild problem. "OK" on its own is
# far too common a caption to click on text this code has not read.
_REBUILD_DIALOG_HINTS = (
    "rebuild", "regenerat", "draft", "wall faces", "could not be",
    "unable to maintain", "перестро", "уклон", "грани стенок",
    "не удалось", "no se pudo", "impossible de", "konnte nicht",
)
#: A NOTE ON THE ONE DIALOG THAT LOOKED LIKE A MISCLASSIFICATION, kept
#: because the wrong reading of it cost a day.
#:
#: On task 20 a modal titled "Preduprezhdenie SOLIDWORKS CAM" stood open
#: while thirteen models failed to open, and the first reading was that
#: the net above had caught an add-in warning and clicked a button that
#: cancelled the open. It had not. The dialog's BODY reads "sokhranit'
#: izmeneniya v input.sldasm ?" -- an ordinary save prompt wearing the
#: add-in's title -- and the refusal is the right answer to it. A second
#: run that left the dialog untouched failed the same thirteen models,
#: which settles it: the dialog was never the cause.
#:
#: The rule that survives: classify on the body, where the question is,
#: never on the title, which only says who is asking.
_ACKNOWLEDGE_CAPTIONS = (
    "ok", "continue", "close", "ок", "продолжить", "закрыть",
    "aceptar", "continuar", "cerrar", "fermer", "schliessen", "weiter",
)


def _clean(text):
    return (text or "").replace("&", "").strip().lower()


def classify_dialog(blob):
    """(kind, acceptable button captions) for a dialog's whole text, or
    (None, ()) for one this code will not touch.

    A free function because it is the part worth testing: the rest of the
    watchdog is win32 calls that no test can make, and a rule that decides
    which button gets clicked in someone's live SolidWorks should not be
    reachable only through them.

    SAVE IS CHECKED FIRST and the order is load-bearing. A prompt that
    mentions both saving and rebuilding is a save prompt, and answering
    that one with "OK" writes the rebuilt document over the candidate's
    file -- the grader would destroy what it was asked to grade.
    """
    blob = _clean(blob)
    if any(h in blob for h in _SAVE_DIALOG_HINTS):
        return "save", _DISMISS_CAPTIONS
    if any(h in blob for h in _REBUILD_DIALOG_HINTS):
        return "rebuild-warning", _ACKNOWLEDGE_CAPTIONS
    return None, ()


class SaveDialogWatchdog:
    """Answers SolidWorks' save-on-close dialog with "Don't Save".

    Why this exists rather than "just do not dirty the document": the
    rebuild census IS the point of several criteria, ForceRebuild3 is how
    it is taken, and a rebuilt document is dirty by definition. Telling a
    harness not to rebuild to avoid a dialog would be letting the UI
    dictate the rubric.

    The documented no-save close routes (IModelDoc2::Quit, QuitDoc,
    CloseAllDocuments(True)) are tried first and usually suffice; this is
    the backstop for when they do not. It runs on its own thread because
    a modal dialog blocks the COM thread -- the code that would notice the
    problem is precisely the code that cannot run.

    Deliberately narrow, because it clicks buttons in the user's live
    SolidWorks:

      * only windows of the standard dialog class,
      * only in the SolidWorks process, when that process can be
        identified,
      * only when the dialog's own text names saving,
      * only buttons whose caption is a refusal to save.

    Every firing is recorded and meant to be printed. A watchdog that
    fires silently would hide exactly the condition it was built to
    reveal.
    """

    def __init__(self, app=None, interval=0.35):
        self.interval = interval
        self.fired = []
        self.available = win32com is not None
        self.reason = None if self.available else "pywin32 not importable"
        self._pid = None
        self._stop = None
        self._thread = None
        if self.available:
            try:
                import win32gui  # noqa: F401
                import win32con  # noqa: F401
                import win32process  # noqa: F401
            except ImportError as exc:
                self.available = False
                self.reason = f"win32gui/win32con/win32process: {exc}"
        if self.available and app is not None:
            self._pid = _solidworks_pid(app)
            if self._pid is None:
                # Not fatal: the text filters still apply. But say so --
                # a wider net is a fact the operator should know about.
                self.reason = ("SolidWorks pid not identified; matching on "
                               "dialog text alone")

    def _dismiss_once(self):
        import win32con
        import win32gui
        import win32process
        hits = []

        def visit(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            try:
                if win32gui.GetClassName(hwnd) != _DIALOG_CLASS:
                    return
                if self._pid is not None:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid != self._pid:
                        return
            except Exception:
                return
            hits.append(hwnd)

        try:
            win32gui.EnumWindows(visit, None)
        except Exception:
            return

        for hwnd in hits:
            texts, buttons = [], []

            def child(ch, _):
                try:
                    cls = win32gui.GetClassName(ch)
                    cap = win32gui.GetWindowText(ch)
                except Exception:
                    return
                texts.append(cap)
                if cls == "Button":
                    buttons.append((ch, cap))

            try:
                title = win32gui.GetWindowText(hwnd)
                texts.append(title)
                win32gui.EnumChildWindows(hwnd, child, None)
            except Exception:
                continue
            blob = " ".join(_clean(t) for t in texts)
            kind, wanted = classify_dialog(blob)
            if kind is None:
                continue
            if not wanted:
                #: SEEN AND NOT TOUCHED. Recorded once per dialog so the
                #: report can say what is standing in the way.
                if not any(f.get("dialog") == title and
                           f.get("kind") == kind for f in self.fired):
                    self.fired.append({"dialog": title, "clicked": None,
                                       "kind": kind, "text": blob[:300]})
                continue
            for ch, cap in buttons:
                if _clean(cap) in wanted:
                    try:
                        win32gui.SendMessage(ch, win32con.BM_CLICK, 0, 0)
                        self.fired.append({"dialog": title, "clicked": cap,
                                           "kind": kind,
                                           "text": blob[:300]})
                    except Exception:
                        pass
                    break

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._dismiss_once()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if not self.available or self._thread is not None:
            return self
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sw-save-dialog-watchdog")
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def _solidworks_pid(app):
    """PID behind the SolidWorks main frame, so the watchdog can stay
    inside that process. None when the frame cannot be reached."""
    try:
        import win32process
    except Exception:
        return None
    # Frame and GetHWnd each surface as either a property or a method
    # depending on typelib state -- that is what z() is for, and the first
    # version of this called GetHWnd() directly and silently produced
    # "pid not identified" on the grading box.
    for get in (lambda: z(z(app.Frame).GetHWnd),
                lambda: z(z(app.IFrameObject()).GetHWnd),
                lambda: z(app.IActiveDoc2.GetHWnd)):
        try:
            hwnd = int(get())
            if hwnd:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                return pid
        except Exception:
            continue
    return None


class unattended:
    """Everything needed for a run that no one is watching.

    Sets CommandInProgress so SolidWorks suppresses what it can, and runs
    SaveDialogWatchdog for what it cannot. Use it around any batch that
    rebuilds documents:

        with sws.unattended(app) as un:
            ...
        for f in un.watchdog.fired:
            print("dismissed:", f)

    `report()` returns the lines worth printing; empty when nothing fired
    and everything was available, so a clean run stays quiet.
    """

    def __init__(self, app, enabled=True):
        self.app = app
        self.enabled = enabled
        self.quiet = None
        self.watchdog = SaveDialogWatchdog(app)

    def __enter__(self):
        if self.enabled:
            self.quiet = quiet_mode(self.app, True)
            self.watchdog.start()
        return self

    def __exit__(self, *exc):
        self.watchdog.stop()
        if self.enabled:
            quiet_mode(self.app, False)
        return False

    def report(self):
        out = []
        if not self.enabled:
            return out
        if not self.watchdog.available:
            out.append("modal-dialog watchdog UNAVAILABLE "
                       f"({self.watchdog.reason}) -- a rebuilt document may "
                       "stop this run on a modal dialog")
        elif self.watchdog.reason:
            out.append(f"modal-dialog watchdog: {self.watchdog.reason}")
        for f in self.watchdog.fired:
            # The KIND matters to whoever reads this. A save prompt is
            # routine housekeeping; a rebuild warning is the model telling
            # you its features did not all regenerate, and that is a fact
            # about the file worth chasing rather than a click to forget.
            kind = f.get("kind", "save")
            if f.get("clicked") is None:
                out.append(f"{kind} dialog LEFT ALONE: {f['dialog']!r} -- "
                           f"this code does not know its buttons, so it "
                           f"answered nothing. If it blocked the run, "
                           f"switch the add-in off rather than guessing."
                           + (f"  [{f['text'][:140]}]"
                              if f.get("text") else ""))
                continue
            #: THE TEXT IS PRINTED FOR EVERY DISMISSAL, not just the
            #: rebuild ones. It used to be printed only for those, and
            #: that is why a CAM warning answered with the wrong button
            #: went by as a one-line note about a save prompt: the run
            #: said which dialog it clicked and never what the dialog
            #: said.
            out.append(f"{kind} dialog dismissed: {f['dialog']!r} -> "
                       f"{f['clicked']!r}"
                       + (f"  [{f['text'][:140]}]"
                          if f.get("text") else ""))
        return out


_SESSION = None


def arm_unattended(app, enabled=True):
    """Arm quiet mode and the save-dialog watchdog once per process.

    Idempotent on purpose: two watchdogs racing for the same dialog would
    each report a firing the other caused. The thread is a daemon, so it
    ends with the process -- which is the right lifetime, because the
    process exists only for the run.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = unattended(app, enabled=enabled)
        _SESSION.__enter__()
    return _SESSION


def session_report():
    """Lines worth printing about unattended operation; empty when the run
    was clean. Print these -- a watchdog nobody hears about is a silent
    workaround for a problem someone should know exists."""
    return _SESSION.report() if _SESSION is not None else []
