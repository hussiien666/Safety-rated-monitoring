"""
PHASE 8 - Voice Safety Alert & Greeting System (standalone)
==========================================================

The spoken-voice layer of the Cobot Vision Safety System, built and tested on
its own here (exactly like PHASE1-PHASE7) before it is wired into
`COBOT_GUI.py`. Nothing in this file imports the GUI, the camera, or YOLO - it
purely turns safety events ("hand entered the red zone", "this known person
appeared", "emergency stop") into spoken audio, and shows the resulting state.

What it speaks
--------------
    * Red-zone alert  - says "Red zone" once, then a repeating buzzer for a
      fixed number of seconds, then stops on its own.
    * Greeting        - says "Hello <name>" ONCE per known person when they
      appear; the same person is not greeted again until they have been out of
      frame for GREET_REENTRY_SECS (so brief recognition drop-outs don't repeat
      the greeting).
    * Emergency stop  - says "Emergency stop activated / released".

Priority (the clash rule)
-------------------------
Safety alerts always win over greetings. Emergency-stop phrases are spoken at
HIGH priority: they replace anything queued and block greetings until they
finish. A greeting is dropped if a high-priority phrase is in flight or the
red-zone buzzer is sounding, and greetings are suppressed entirely while the
emergency stop is engaged. This is the exact behaviour integrated into
`COBOT_GUI.py`.

Speech backend (same fallback chain as COBOT_GUI's AudioAlerter)
----------------------------------------------------------------
    1. pyttsx3       - in-process offline TTS (SAPI5 voice on Windows)
    2. Windows SAPI  - via PowerShell System.Speech, needs NO pip package
    3. winsound beep - last-resort tone if no TTS backend is available
So it talks on Windows out of the box with nothing to install.

Run modes
---------
    py PHASE8.py                 # simulation REPL: type events, hear the audio
    py PHASE8.py --demo          # scripted demo: greet -> red zone -> e-stop
    py PHASE8.py --selftest      # logic self-test, silent, no audio

Optional (nicer in-process TTS):
    py -m pip install pyttsx3
"""

from __future__ import annotations               # Lets type hints be lazy strings (e.g. "queue.Queue").

import argparse                                   # Parse the --demo / --selftest command-line flags.
import importlib.util                             # Detect whether the optional 'pyttsx3' package is installed.
import os                                         # Read environment variables (headless detection).
import queue                                      # Thread-safe FIFO that carries phrases to the worker thread.
import subprocess                                 # Run PowerShell for the Windows SAPI speech fallback.
import sys                                        # Platform check + process exit codes.
import threading                                  # Background worker + buzzer threads (non-blocking audio).
import time                                       # Monotonic clocks for cooldowns and alert deadlines.
from typing import Optional                       # Type hint for "X or None".


# ----------------------------------------------------------------------------
# Configuration  (mirrors the matching knobs in COBOT_GUI.py)
# ----------------------------------------------------------------------------
VOICE_ALERTS_DEFAULT = True                       # Whether voice is on by default.
RED_ZONE_PHRASE      = "Red zone"          # spoken once on entering the red zone
BUZZER_FREQ          = 600                  # Hz  - tone of the repeating buzzer
BUZZER_DUR_MS        = 300                  # ms  - length of each buzz
BUZZER_GAP_S         = 0.15                 # s   - silence between buzzes
ESTOP_ON_PHRASE      = "Emergency stop activated."   # Spoken when the e-stop engages.
ESTOP_OFF_PHRASE     = "Emergency stop released."    # Spoken when the e-stop releases.

# Red-zone alert episode length: the voice + buzzer run for this long, then stop.
RED_ALERT_SECS       = 5.0                        # Total duration of one red-zone alert episode.
# Don't greet the same known person again until they've been out of frame for
# this long (s) - stops brief recognition drop-outs re-triggering the greeting.
GREET_REENTRY_SECS   = 8.0                        # Per-person greeting cooldown.


# ----------------------------------------------------------------------------
# Core 1: the audio engine  (verbatim mirror of COBOT_GUI.AudioAlerter)
# ----------------------------------------------------------------------------
class AudioAlerter:
    """Speaks short safety phrases through the laptop speaker.

    Playback runs on a daemon worker thread so the caller never blocks. The
    backend is picked at runtime - the first that works wins (pyttsx3 -> SAPI ->
    beep). Calls collapse: a newer phrase replaces any still-queued one.

    Priority: a high-priority phrase (emergency stop / red zone) replaces what is
    queued and blocks low-priority phrases (greetings) until it finishes, so a
    safety alert always wins if the two clash. The red-zone alert says a short
    phrase once, then sounds a repeating buzzer (on its own thread) until the
    deadline or `stop_buzzer()`.
    """

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()    # Pending phrases (worker pulls from here).
        self._thread: Optional[threading.Thread] = None  # The speech worker thread (lazily created).
        self._buzzer_stop = threading.Event()     # Set this flag to stop the buzzer loop early.
        self._buzzer_thread: Optional[threading.Thread] = None  # The red-zone buzzer thread.
        # When a high-priority phrase (emergency stop / red zone) is queued or
        # speaking, low-priority phrases (greetings) are dropped so the safety
        # alert always wins if the two clash.
        self._lock = threading.Lock()             # Guards _priority_active + queue draining.
        self._priority_active = False             # True while a high-priority phrase is in flight.
        # Stay silent in headless / self-test runs (no speakers, would block).
        self._enabled = (VOICE_ALERTS_DEFAULT and  # Voice on by default...
                         os.environ.get("QT_QPA_PLATFORM") != "offscreen")  # ...but muted when headless.
        if importlib.util.find_spec("pyttsx3") is not None:  # If pyttsx3 is installed...
            self._backend = "pyttsx3"             # ...use the nicest in-process TTS.
        elif sys.platform.startswith("win"):      # Else on Windows...
            self._backend = "sapi"                # ...use built-in SAPI via PowerShell.
        else:                                     # Otherwise...
            self._backend = "beep"                # ...fall back to a plain beep.

    def backend_name(self) -> str:
        return self._backend                      # Expose which backend was selected.

    def set_enabled(self, on: bool):
        self._enabled = bool(on)                  # Turn voice on/off at runtime.
        if not self._enabled:                     # If turning off...
            self.stop_buzzer()                    # ...also silence any running buzzer.

    def alert_red_zone(self, duration_s=RED_ALERT_SECS):
        """Say the red-zone phrase once, then buzz for `duration_s` seconds and
        stop on its own - a fixed-length alert, not an open-ended one."""
        if not self._enabled:                     # Do nothing if voice is disabled.
            return
        self.stop_buzzer()                        # Cancel any previous buzzer first.
        self._buzzer_stop.clear()                 # Reset the stop flag for the new episode.
        self._buzzer_thread = threading.Thread(   # Run the alert on its own daemon thread...
            target=self._red_zone_loop, args=(duration_s,),
            name="RedZoneBuzzer", daemon=True)
        self._buzzer_thread.start()               # ...so the caller never blocks.

    def is_alerting(self) -> bool:
        """True while the red-zone alert episode (phrase + buzzer) is running."""
        t = self._buzzer_thread                   # Snapshot the handle (thread-safety).
        return t is not None and t.is_alive()     # Alive thread → still alerting.

    def stop_buzzer(self):
        """Stop the repeating red-zone buzzer (no-op if it is not running)."""
        self._buzzer_stop.set()                   # Signal the loop to exit.
        t = self._buzzer_thread                   # Snapshot the handle.
        if t is not None and t.is_alive() and t is not threading.current_thread():  # If running & not us...
            t.join(timeout=2.0)                   # ...wait briefly for it to finish.
        self._buzzer_thread = None                # Clear the handle.

    def say(self, text: str, priority: bool = False):
        """Queue a phrase to speak, dropping any older pending phrase.

        A high-priority phrase (emergency stop / red zone) replaces whatever is
        queued and blocks low-priority phrases until it finishes. A low-priority
        phrase (greeting) is dropped if a high-priority phrase is in flight or
        the red-zone alert is sounding - so safety alerts always win a clash.
        """
        if not self._enabled:                     # Silent when disabled.
            return
        self._ensure_thread()                     # Make sure the worker thread exists.
        with self._lock:                          # Atomically check priority + enqueue.
            if not priority and (self._priority_active or self.is_alerting()):  # Low-pri loses a clash...
                return                            # ...so drop the greeting.
            self._drain_queue()                   # Collapse: discard any older pending phrase.
            if priority:                          # If this is high priority...
                self._priority_active = True      # ...block greetings until it finishes.
            self._q.put((priority, text))         # Enqueue (priority flag, text) for the worker.

    def _drain_queue(self):
        try:
            while True:                           # Pull items until...
                self._q.get_nowait()              # ...the queue raises Empty.
        except queue.Empty:
            pass                                  # Queue is now empty.

    def shutdown(self):
        self._enabled = False                     # Stop accepting new phrases.
        self.stop_buzzer()                        # Silence the buzzer.
        if self._thread is not None:              # If the worker exists...
            self._q.put(None)                     # ...send the sentinel that tells it to exit.

    # -- internals ----------------------------------------------------------
    def _red_zone_loop(self, duration_s=RED_ALERT_SECS):
        # Speak the phrase once, then buzz until the deadline (or an early stop).
        deadline = time.monotonic() + max(0.5, float(duration_s))  # Absolute time the alert ends.
        try:
            self._speak(RED_ZONE_PHRASE)          # Say "Red zone" once.
        except Exception:
            pass                                  # Never let a speech error kill the thread.
        while not self._buzzer_stop.is_set() and time.monotonic() < deadline:  # Until stopped or expired...
            self._buzz_once()                     # ...emit one buzz...
            self._buzzer_stop.wait(BUZZER_GAP_S)  # ...then wait the gap (interruptible).
        # The episode finished on its own - drop our handle so state stays clean.
        if self._buzzer_thread is threading.current_thread():  # If we're still the active buzzer...
            self._buzzer_thread = None            # ...clear the handle.

    def _buzz_once(self):
        try:
            import winsound                       # Windows-only tone API.
            winsound.Beep(BUZZER_FREQ, BUZZER_DUR_MS)  # Emit one beep.
        except Exception:
            try:
                print("\a", end="", flush=True)   # Fallback: terminal bell character.
            except Exception:
                pass                              # Give up silently if even that fails.

    def _ensure_thread(self):
        if self._thread is None:                  # Lazily create the worker on first use...
            self._thread = threading.Thread(
                target=self._run, name="AudioAlerter", daemon=True)
            self._thread.start()                  # ...and start it.

    def _run(self):
        while True:                               # Worker loop.
            item = self._q.get()                  # Block until a phrase (or sentinel) arrives.
            if item is None:                      # None = shutdown sentinel...
                break                             # ...so exit the worker.
            priority, text = item                 # Unpack the queued tuple.
            try:
                self._speak(text)                 # Speak it (blocking on THIS thread only).
            except Exception:
                pass                              # Swallow speech errors.
            if priority:                          # After a high-priority phrase...
                with self._lock:
                    # Release the lock only once nothing else is waiting, so a
                    # burst of safety phrases keeps greetings out until it ends.
                    if self._q.empty():           # ...only clear the gate when nothing follows.
                        self._priority_active = False

    def _speak(self, text: str):
        if self._backend == "pyttsx3":            # --- Preferred backend ---
            try:
                import pyttsx3
                eng = pyttsx3.init()              # Create a TTS engine.
                try:
                    eng.setProperty("rate", 178)  # Speaking speed.
                    eng.setProperty("volume", 1.0)  # Max volume.
                except Exception:
                    pass                          # Property failures are non-fatal.
                eng.say(text)                     # Queue the text in the engine.
                eng.runAndWait()                  # Speak synchronously (on this worker thread).
                try:
                    eng.stop()                    # Release the engine.
                except Exception:
                    pass
                return                            # Done.
            except Exception:
                # Demote to the platform backend and fall through.
                self._backend = "sapi" if sys.platform.startswith("win") else "beep"

        if self._backend == "sapi":               # --- Windows SAPI fallback (no pip package) ---
            try:
                safe = text.replace("'", " ").replace('"', " ")  # Strip quotes to keep the PS string valid.
                ps = (                            # Build a PowerShell System.Speech one-liner...
                    "Add-Type -AssemblyName System.Speech; "
                    "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$v.Rate = 1; $v.Volume = 100; "
                    f"$v.Speak('{safe}')"
                )
                subprocess.run(                   # ...and run it.
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # Hide the console window.
                    timeout=12,                   # Don't hang forever.
                )
                return                            # Done.
            except Exception:
                self._backend = "beep"            # Demote to the last-resort beep.

        # Last resort: an attention-grabbing two-tone alarm.
        try:
            import winsound
            winsound.Beep(920, 220)               # High tone...
            winsound.Beep(680, 320)               # ...then a lower tone.
        except Exception:
            try:
                print("\a", end="", flush=True)   # Terminal bell as the absolute fallback.
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Core 2: the event -> voice policy  (the part the GUI reuses)
# ----------------------------------------------------------------------------
class VoiceAlertSystem:
    """Turns high-level safety events into spoken audio, applying the same
    greeting and priority rules as COBOT_GUI.

    Events (call these as they happen in the vision pipeline):
        on_faces(names)   - the set of KNOWN identities currently in frame
        on_red_zone(red)  - True while a hand is inside the red zone
        on_estop(on)      - emergency-stop latch engaged / released

    The policy:
        * Greet each newly-seen known person once ("Hello <name>"), re-greeting
          only after they've been gone for GREET_REENTRY_SECS.
        * Fire the red-zone alert once on ENTERING the red zone.
        * Emergency stop is high priority and suppresses greetings while engaged.
    """

    def __init__(self, audio: Optional[AudioAlerter] = None,
                 log=None):
        self.audio = audio if audio is not None else AudioAlerter()  # Reuse or create an audio engine.
        self._log = log or (lambda m: None)       # Optional logger; default is a no-op.
        self._was_red = False                     # Previous red-zone state (for edge detection).
        self._estop = False                       # Whether the emergency stop is engaged.
        self._last_seen: dict = {}     # known-name -> last-seen monotonic time   # Greeting cooldown table.

    def reset(self):
        """Forget transient state (call on START, like the GUI does)."""
        self._was_red = False                     # Reset red-zone edge.
        self._estop = False                       # Reset e-stop.
        self._last_seen.clear()                   # Forget who we've greeted.
        self.audio.stop_buzzer()                  # Silence any buzzer.

    # ---- events -----------------------------------------------------------
    def on_faces(self, names):
        """Greet newly-appeared known people. Suppressed while the emergency
        stop or the red-zone alert is active so the safety alert is never talked
        over."""
        names = [n for n in (names or ()) if n and n != "Unknown"]  # Keep only real, known names.
        if not names or self._estop or self.audio.is_alerting():    # Nothing to greet / safety active...
            return []                             # ...so greet no one.
        now = time.monotonic()                    # Current time.
        entered = []                              # People who newly qualify for a greeting.
        for name in names:                        # For each known face in frame...
            last = self._last_seen.get(name)      # When did we last see them?
            if last is None or (now - last) > GREET_REENTRY_SECS:  # Never, or back after the cooldown...
                entered.append(name)              # ...→ greet them.
            self._last_seen[name] = now           # Update their last-seen time either way.
        if entered:
            # One "Hello <name>" per newly-seen person, spoken as a single phrase
            # so none get dropped from the (collapsing) audio queue.
            self.audio.say(". ".join(f"Hello {name}" for name in entered))  # Speak all greetings together.
            self._log("Greeted: " + ", ".join(entered))  # Log who was greeted.
        return entered                            # Return the greeted names (used by tests/REPL).

    def on_red_zone(self, red: bool):
        """On *entering* the red zone, fire the fixed-length voice + buzzer
        alert. It stops on its own after RED_ALERT_SECS."""
        red = bool(red)                           # Normalise to a clean bool.
        if red and not self._was_red:             # Rising edge: just entered the red zone...
            self.audio.alert_red_zone(RED_ALERT_SECS)  # ...fire the alert (once).
            self._log("RED ZONE - alert fired")
        self._was_red = red                       # Remember state for next call.

    def on_estop(self, on: bool):
        """Engage / release the emergency stop. Spoken at HIGH priority so it
        wins over any greeting, and greetings stay suppressed while engaged."""
        on = bool(on)                             # Normalise.
        if on == self._estop:                     # No change → nothing to announce.
            return
        self._estop = on                          # Record the new state.
        if on:                                    # Engaging...
            self.audio.say(ESTOP_ON_PHRASE, priority=True)  # ...announce at high priority.
            self._log("EMERGENCY STOP engaged")
        else:                                     # Releasing...
            self.audio.say(ESTOP_OFF_PHRASE, priority=True)  # ...announce release.
            self._log("Emergency stop released")

    def close(self):
        self.audio.shutdown()                     # Tear down the audio engine.


# ----------------------------------------------------------------------------
# Run mode 1: interactive simulation REPL  (type events, hear the audio)
# ----------------------------------------------------------------------------
HELP = """
Type an event:
    face <name> [<name> ...]   a known person (or several) appears in frame
    clearfaces                 nobody known is in frame
    red                        a hand enters the red zone  -> alert + buzzer
    safe                       hands leave the red zone
    estop                      engage the emergency stop
    release                    release the emergency stop
    backend                    show the active TTS backend
    help                       show this help
    quit                       exit
"""


def run_sim():
    """Type safety events and hear exactly what the cobot would say."""
    vas = VoiceAlertSystem(log=lambda m: print(f"   [log] {m}"))  # System that prints its log lines.
    print("Phase 8 - Voice Safety Alert & Greeting SIMULATION")
    print(f"TTS backend: {vas.audio.backend_name()}")  # Show which speech backend is active.
    print(HELP)
    try:
        while True:                               # REPL loop.
            try:
                line = input("event> ").strip()   # Read a command line.
            except EOFError:                      # Ctrl-Z/Ctrl-D ends input...
                break
            if not line:                          # Ignore blank lines.
                continue
            parts = line.split()                  # Tokenise.
            cmd, rest = parts[0].lower(), parts[1:]  # First token = command, rest = args.

            if cmd in ("quit", "exit"):           # Exit commands.
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "backend":
                print(f"   TTS backend: {vas.audio.backend_name()}")
            elif cmd == "face":                   # Simulate known faces appearing.
                if not rest:
                    print("   usage: face <name> [<name> ...]")
                    continue
                greeted = vas.on_faces(rest)      # Fire the greeting policy.
                if not greeted:
                    print("   (no new greeting - already greeted, or suppressed)")
            elif cmd == "clearfaces":             # Simulate nobody in frame.
                vas.on_faces([])
            elif cmd == "red":                    # Hand enters red zone.
                vas.on_red_zone(True)
            elif cmd == "safe":                   # Hand leaves red zone.
                vas.on_red_zone(False)
            elif cmd == "estop":                  # Engage emergency stop.
                vas.on_estop(True)
            elif cmd == "release":                # Release emergency stop.
                vas.on_estop(False)
            else:
                print(f"   unknown event: {cmd!r}  (type 'help')")
    finally:
        vas.close()                               # Always tear down audio on exit.
        print("Voice alert system stopped.")


# ----------------------------------------------------------------------------
# Run mode 2: scripted demo  (greet -> red zone -> emergency-stop clash)
# ----------------------------------------------------------------------------
def run_demo():
    """A short scripted timeline that exercises greeting, the red-zone alert,
    and the emergency-stop-over-greeting priority - audibly."""
    vas = VoiceAlertSystem(log=lambda m: print(f"   [log] {m}"))
    print("Phase 8 - scripted voice demo")
    print(f"TTS backend: {vas.audio.backend_name()}\n")

    def step(desc, fn, pause=2.5):                # Helper: print a description, run it, then wait.
        print(f">> {desc}")
        fn()
        time.sleep(pause)

    step("Hussien appears  -> 'Hello Hussien'",   # Greeting on first sighting.
         lambda: vas.on_faces(["Hussien"]))
    step("Mazen also appears -> 'Hello Mazen'  (Hussien NOT greeted again)",  # Only the new person greeted.
         lambda: vas.on_faces(["Hussien", "Mazen"]))
    step("Hand enters the RED ZONE -> 'Red zone' + buzzer",  # Red-zone alert episode.
         lambda: vas.on_red_zone(True), pause=RED_ALERT_SECS + 0.5)
    step("A new face appears DURING the alert -> greeting is dropped",  # Safety beats greeting.
         lambda: vas.on_faces(["Sara"]), pause=1.0)
    step("EMERGENCY STOP engaged -> high-priority phrase, greetings suppressed",  # E-stop priority.
         lambda: vas.on_estop(True))
    step("Another face appears while e-stopped -> NO greeting",  # Suppressed while engaged.
         lambda: vas.on_faces(["Omar"]), pause=1.5)
    step("Emergency stop released",              # Release announcement.
         lambda: vas.on_estop(False))

    time.sleep(1.0)                               # Let the last phrase finish.
    vas.close()                                   # Tear down.
    print("\nDemo complete.")


# ----------------------------------------------------------------------------
# Run mode 3: self-test  (logic only - silent, no audio, no blocking)
# ----------------------------------------------------------------------------
def run_selftest() -> int:
    failures = 0                                  # Count of failed checks.

    def check(cond, msg):                         # Tiny assertion helper that records pass/fail.
        nonlocal failures
        ok = bool(cond)
        failures += not ok                        # +1 when the check fails (False→1).
        print(f"  [{'ok' if ok else 'XX'}] {msg}")

    # Run with audio disabled so nothing speaks or blocks.
    audio = AudioAlerter()
    audio.set_enabled(False)                      # Mute → pure logic test.
    vas = VoiceAlertSystem(audio=audio)

    # 1) a known person is greeted once, not twice in quick succession
    check(vas.on_faces(["Hussien"]) == ["Hussien"], "first sighting greets Hussien")
    check(vas.on_faces(["Hussien"]) == [], "immediate re-sighting does NOT re-greet")

    # 2) a second new person is greeted, the first is not greeted again
    check(vas.on_faces(["Hussien", "Mazen"]) == ["Mazen"],
          "new person Mazen greeted, Hussien not re-greeted")

    # 3) re-entry after the cooldown greets again
    vas._last_seen["Hussien"] = time.monotonic() - (GREET_REENTRY_SECS + 1)  # Pretend Hussien left long ago.
    check(vas.on_faces(["Hussien"]) == ["Hussien"],
          "re-greets after GREET_REENTRY_SECS away")

    # 4) emergency stop suppresses greetings (priority over greeting)
    vas.on_estop(True)
    check(vas.on_faces(["Sara"]) == [], "no greeting while emergency stop engaged")
    vas.on_estop(False)
    check(vas.on_faces(["Sara"]) == ["Sara"], "greeting resumes after release")

    # 5) red-zone edge: alert fires only on entering, not while held
    fired = {"n": 0}                              # Mutable counter captured by the stub below.
    audio.alert_red_zone = lambda *_a, **_k: fired.__setitem__("n", fired["n"] + 1)  # Stub counts calls.
    vas.on_red_zone(True)
    vas.on_red_zone(True)   # still red - must NOT fire again
    vas.on_red_zone(False)
    vas.on_red_zone(True)   # re-entered - fires again
    check(fired["n"] == 2, "red-zone alert fires once per entry (got %d)" % fired["n"])

    # 6) the priority gate: a queued high-priority phrase blocks a greeting
    a2 = AudioAlerter()
    a2.set_enabled(True)            # enabled, but we won't let the worker run
    a2._priority_active = True      # simulate an e-stop phrase in flight
    before = a2._q.qsize()                        # Queue size before the greeting attempt.
    a2.say("Hello Test", priority=False)          # Try to queue a greeting...
    check(a2._q.qsize() == before, "greeting dropped while a priority phrase is active")  # ...should be dropped.
    a2.set_enabled(False)

    if failures == 0:                             # Summarise the run.
        print("\n[selftest] PASS")
        return 0                                  # Exit code 0 = success.
    print(f"\n[selftest] FAIL ({failures} check(s))")
    return 1                                       # Non-zero = failure (for CI).


def main():
    parser = argparse.ArgumentParser(             # Set up CLI argument parsing.
        description="Phase 8 - standalone voice safety alert & greeting system")
    parser.add_argument("--demo", action="store_true",       # --demo flag.
                        help="run the scripted audible demo and exit")
    parser.add_argument("--selftest", action="store_true",   # --selftest flag.
                        help="run the silent logic self-test and exit")
    args = parser.parse_args()                    # Parse argv.

    if args.selftest:                             # Dispatch to the chosen mode.
        sys.exit(run_selftest())                  # Self-test → exit with its code.
    elif args.demo:
        run_demo()                                # Scripted audible demo.
    else:
        run_sim()                                 # Default: interactive REPL.


if __name__ == "__main__":                         # Only run when executed directly.
    main()
