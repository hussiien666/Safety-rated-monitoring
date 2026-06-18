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

from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional


# ----------------------------------------------------------------------------
# Configuration  (mirrors the matching knobs in COBOT_GUI.py)
# ----------------------------------------------------------------------------
VOICE_ALERTS_DEFAULT = True
RED_ZONE_PHRASE      = "Red zone"          # spoken once on entering the red zone
BUZZER_FREQ          = 600                  # Hz  - tone of the repeating buzzer
BUZZER_DUR_MS        = 300                  # ms  - length of each buzz
BUZZER_GAP_S         = 0.15                 # s   - silence between buzzes
ESTOP_ON_PHRASE      = "Emergency stop activated."
ESTOP_OFF_PHRASE     = "Emergency stop released."

# Red-zone alert episode length: the voice + buzzer run for this long, then stop.
RED_ALERT_SECS       = 5.0
# Don't greet the same known person again until they've been out of frame for
# this long (s) - stops brief recognition drop-outs re-triggering the greeting.
GREET_REENTRY_SECS   = 8.0


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
        self._q: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._buzzer_stop = threading.Event()
        self._buzzer_thread: Optional[threading.Thread] = None
        # When a high-priority phrase (emergency stop / red zone) is queued or
        # speaking, low-priority phrases (greetings) are dropped so the safety
        # alert always wins if the two clash.
        self._lock = threading.Lock()
        self._priority_active = False
        # Stay silent in headless / self-test runs (no speakers, would block).
        self._enabled = (VOICE_ALERTS_DEFAULT and
                         os.environ.get("QT_QPA_PLATFORM") != "offscreen")
        if importlib.util.find_spec("pyttsx3") is not None:
            self._backend = "pyttsx3"
        elif sys.platform.startswith("win"):
            self._backend = "sapi"
        else:
            self._backend = "beep"

    def backend_name(self) -> str:
        return self._backend

    def set_enabled(self, on: bool):
        self._enabled = bool(on)
        if not self._enabled:
            self.stop_buzzer()

    def alert_red_zone(self, duration_s=RED_ALERT_SECS):
        """Say the red-zone phrase once, then buzz for `duration_s` seconds and
        stop on its own - a fixed-length alert, not an open-ended one."""
        if not self._enabled:
            return
        self.stop_buzzer()
        self._buzzer_stop.clear()
        self._buzzer_thread = threading.Thread(
            target=self._red_zone_loop, args=(duration_s,),
            name="RedZoneBuzzer", daemon=True)
        self._buzzer_thread.start()

    def is_alerting(self) -> bool:
        """True while the red-zone alert episode (phrase + buzzer) is running."""
        t = self._buzzer_thread
        return t is not None and t.is_alive()

    def stop_buzzer(self):
        """Stop the repeating red-zone buzzer (no-op if it is not running)."""
        self._buzzer_stop.set()
        t = self._buzzer_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._buzzer_thread = None

    def say(self, text: str, priority: bool = False):
        """Queue a phrase to speak, dropping any older pending phrase.

        A high-priority phrase (emergency stop / red zone) replaces whatever is
        queued and blocks low-priority phrases until it finishes. A low-priority
        phrase (greeting) is dropped if a high-priority phrase is in flight or
        the red-zone alert is sounding - so safety alerts always win a clash.
        """
        if not self._enabled:
            return
        self._ensure_thread()
        with self._lock:
            if not priority and (self._priority_active or self.is_alerting()):
                return
            self._drain_queue()
            if priority:
                self._priority_active = True
            self._q.put((priority, text))

    def _drain_queue(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def shutdown(self):
        self._enabled = False
        self.stop_buzzer()
        if self._thread is not None:
            self._q.put(None)

    # -- internals ----------------------------------------------------------
    def _red_zone_loop(self, duration_s=RED_ALERT_SECS):
        # Speak the phrase once, then buzz until the deadline (or an early stop).
        deadline = time.monotonic() + max(0.5, float(duration_s))
        try:
            self._speak(RED_ZONE_PHRASE)
        except Exception:
            pass
        while not self._buzzer_stop.is_set() and time.monotonic() < deadline:
            self._buzz_once()
            self._buzzer_stop.wait(BUZZER_GAP_S)
        # The episode finished on its own - drop our handle so state stays clean.
        if self._buzzer_thread is threading.current_thread():
            self._buzzer_thread = None

    def _buzz_once(self):
        try:
            import winsound
            winsound.Beep(BUZZER_FREQ, BUZZER_DUR_MS)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass

    def _ensure_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="AudioAlerter", daemon=True)
            self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            priority, text = item
            try:
                self._speak(text)
            except Exception:
                pass
            if priority:
                with self._lock:
                    # Release the lock only once nothing else is waiting, so a
                    # burst of safety phrases keeps greetings out until it ends.
                    if self._q.empty():
                        self._priority_active = False

    def _speak(self, text: str):
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                eng = pyttsx3.init()
                try:
                    eng.setProperty("rate", 178)
                    eng.setProperty("volume", 1.0)
                except Exception:
                    pass
                eng.say(text)
                eng.runAndWait()
                try:
                    eng.stop()
                except Exception:
                    pass
                return
            except Exception:
                # Demote to the platform backend and fall through.
                self._backend = "sapi" if sys.platform.startswith("win") else "beep"

        if self._backend == "sapi":
            try:
                safe = text.replace("'", " ").replace('"', " ")
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$v.Rate = 1; $v.Volume = 100; "
                    f"$v.Speak('{safe}')"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=12,
                )
                return
            except Exception:
                self._backend = "beep"

        # Last resort: an attention-grabbing two-tone alarm.
        try:
            import winsound
            winsound.Beep(920, 220)
            winsound.Beep(680, 320)
        except Exception:
            try:
                print("\a", end="", flush=True)
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
        self.audio = audio if audio is not None else AudioAlerter()
        self._log = log or (lambda m: None)
        self._was_red = False
        self._estop = False
        self._last_seen: dict = {}     # known-name -> last-seen monotonic time

    def reset(self):
        """Forget transient state (call on START, like the GUI does)."""
        self._was_red = False
        self._estop = False
        self._last_seen.clear()
        self.audio.stop_buzzer()

    # ---- events -----------------------------------------------------------
    def on_faces(self, names):
        """Greet newly-appeared known people. Suppressed while the emergency
        stop or the red-zone alert is active so the safety alert is never talked
        over."""
        names = [n for n in (names or ()) if n and n != "Unknown"]
        if not names or self._estop or self.audio.is_alerting():
            return []
        now = time.monotonic()
        entered = []
        for name in names:
            last = self._last_seen.get(name)
            if last is None or (now - last) > GREET_REENTRY_SECS:
                entered.append(name)
            self._last_seen[name] = now
        if entered:
            # One "Hello <name>" per newly-seen person, spoken as a single phrase
            # so none get dropped from the (collapsing) audio queue.
            self.audio.say(". ".join(f"Hello {name}" for name in entered))
            self._log("Greeted: " + ", ".join(entered))
        return entered

    def on_red_zone(self, red: bool):
        """On *entering* the red zone, fire the fixed-length voice + buzzer
        alert. It stops on its own after RED_ALERT_SECS."""
        red = bool(red)
        if red and not self._was_red:
            self.audio.alert_red_zone(RED_ALERT_SECS)
            self._log("RED ZONE - alert fired")
        self._was_red = red

    def on_estop(self, on: bool):
        """Engage / release the emergency stop. Spoken at HIGH priority so it
        wins over any greeting, and greetings stay suppressed while engaged."""
        on = bool(on)
        if on == self._estop:
            return
        self._estop = on
        if on:
            self.audio.say(ESTOP_ON_PHRASE, priority=True)
            self._log("EMERGENCY STOP engaged")
        else:
            self.audio.say(ESTOP_OFF_PHRASE, priority=True)
            self._log("Emergency stop released")

    def close(self):
        self.audio.shutdown()


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
    vas = VoiceAlertSystem(log=lambda m: print(f"   [log] {m}"))
    print("Phase 8 - Voice Safety Alert & Greeting SIMULATION")
    print(f"TTS backend: {vas.audio.backend_name()}")
    print(HELP)
    try:
        while True:
            try:
                line = input("event> ").strip()
            except EOFError:
                break
            if not line:
                continue
            parts = line.split()
            cmd, rest = parts[0].lower(), parts[1:]

            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "backend":
                print(f"   TTS backend: {vas.audio.backend_name()}")
            elif cmd == "face":
                if not rest:
                    print("   usage: face <name> [<name> ...]")
                    continue
                greeted = vas.on_faces(rest)
                if not greeted:
                    print("   (no new greeting - already greeted, or suppressed)")
            elif cmd == "clearfaces":
                vas.on_faces([])
            elif cmd == "red":
                vas.on_red_zone(True)
            elif cmd == "safe":
                vas.on_red_zone(False)
            elif cmd == "estop":
                vas.on_estop(True)
            elif cmd == "release":
                vas.on_estop(False)
            else:
                print(f"   unknown event: {cmd!r}  (type 'help')")
    finally:
        vas.close()
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

    def step(desc, fn, pause=2.5):
        print(f">> {desc}")
        fn()
        time.sleep(pause)

    step("Hussien appears  -> 'Hello Hussien'",
         lambda: vas.on_faces(["Hussien"]))
    step("Mazen also appears -> 'Hello Mazen'  (Hussien NOT greeted again)",
         lambda: vas.on_faces(["Hussien", "Mazen"]))
    step("Hand enters the RED ZONE -> 'Red zone' + buzzer",
         lambda: vas.on_red_zone(True), pause=RED_ALERT_SECS + 0.5)
    step("A new face appears DURING the alert -> greeting is dropped",
         lambda: vas.on_faces(["Sara"]), pause=1.0)
    step("EMERGENCY STOP engaged -> high-priority phrase, greetings suppressed",
         lambda: vas.on_estop(True))
    step("Another face appears while e-stopped -> NO greeting",
         lambda: vas.on_faces(["Omar"]), pause=1.5)
    step("Emergency stop released",
         lambda: vas.on_estop(False))

    time.sleep(1.0)
    vas.close()
    print("\nDemo complete.")


# ----------------------------------------------------------------------------
# Run mode 3: self-test  (logic only - silent, no audio, no blocking)
# ----------------------------------------------------------------------------
def run_selftest() -> int:
    failures = 0

    def check(cond, msg):
        nonlocal failures
        ok = bool(cond)
        failures += not ok
        print(f"  [{'ok' if ok else 'XX'}] {msg}")

    # Run with audio disabled so nothing speaks or blocks.
    audio = AudioAlerter()
    audio.set_enabled(False)
    vas = VoiceAlertSystem(audio=audio)

    # 1) a known person is greeted once, not twice in quick succession
    check(vas.on_faces(["Hussien"]) == ["Hussien"], "first sighting greets Hussien")
    check(vas.on_faces(["Hussien"]) == [], "immediate re-sighting does NOT re-greet")

    # 2) a second new person is greeted, the first is not greeted again
    check(vas.on_faces(["Hussien", "Mazen"]) == ["Mazen"],
          "new person Mazen greeted, Hussien not re-greeted")

    # 3) re-entry after the cooldown greets again
    vas._last_seen["Hussien"] = time.monotonic() - (GREET_REENTRY_SECS + 1)
    check(vas.on_faces(["Hussien"]) == ["Hussien"],
          "re-greets after GREET_REENTRY_SECS away")

    # 4) emergency stop suppresses greetings (priority over greeting)
    vas.on_estop(True)
    check(vas.on_faces(["Sara"]) == [], "no greeting while emergency stop engaged")
    vas.on_estop(False)
    check(vas.on_faces(["Sara"]) == ["Sara"], "greeting resumes after release")

    # 5) red-zone edge: alert fires only on entering, not while held
    fired = {"n": 0}
    audio.alert_red_zone = lambda *_a, **_k: fired.__setitem__("n", fired["n"] + 1)
    vas.on_red_zone(True)
    vas.on_red_zone(True)   # still red - must NOT fire again
    vas.on_red_zone(False)
    vas.on_red_zone(True)   # re-entered - fires again
    check(fired["n"] == 2, "red-zone alert fires once per entry (got %d)" % fired["n"])

    # 6) the priority gate: a queued high-priority phrase blocks a greeting
    a2 = AudioAlerter()
    a2.set_enabled(True)            # enabled, but we won't let the worker run
    a2._priority_active = True      # simulate an e-stop phrase in flight
    before = a2._q.qsize()
    a2.say("Hello Test", priority=False)
    check(a2._q.qsize() == before, "greeting dropped while a priority phrase is active")
    a2.set_enabled(False)

    if failures == 0:
        print("\n[selftest] PASS")
        return 0
    print(f"\n[selftest] FAIL ({failures} check(s))")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 - standalone voice safety alert & greeting system")
    parser.add_argument("--demo", action="store_true",
                        help="run the scripted audible demo and exit")
    parser.add_argument("--selftest", action="store_true",
                        help="run the silent logic self-test and exit")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())
    elif args.demo:
        run_demo()
    else:
        run_sim()


if __name__ == "__main__":
    main()
