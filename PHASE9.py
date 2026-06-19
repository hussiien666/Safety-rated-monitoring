"""
PHASE 9 - Session Logging System (standalone)
=============================================

The per-session logging layer of the Cobot Vision Safety System, built and
tested on its own here (exactly like PHASE1-PHASE8) before it is wired into
`COBOT_GUI.py`. Nothing in this file imports the GUI, the camera, YOLO or audio
- it purely turns safety-monitor events ("hand entered the red zone", "emergency
stop engaged", "collision risk") into a timestamped log file, one file per
monitoring session.

What it writes
--------------
One file per session, created when the monitor STARTS and finalised when it
STOPS, named:

    logs/session_YYYYMMDD_HHMMSS.log

Each file looks like:

    ===== SESSION START  2026-06-19 14:56:01 =====
    [14:56:01] Starting subsystems...
    [14:56:02] SAFETY  ->  STOP (red zone - hand too close, arm frozen)
    [14:56:02] SAFETY  ->  COLLISION RISK (2 object(s) near a stopped arm)
    [14:56:05] SAFETY  ->  SAFE (green zone, arm resumed)
    ===== SESSION END    2026-06-19 14:56:30  (duration 00:29) =====

The design rule (same as COBOT_GUI)
-----------------------------------
Safety transitions are logged ONLY when the state actually changes, never every
frame, so the file reads as a clean timeline of the safety monitor rather than
thousands of identical lines. The four states mirror the GUI exactly:
    * SAFE            - green zone, the arm is allowed to move
    * STOP            - red zone, a hand is too close, the arm is frozen
    * EMERGENCY STOP  - the manual right-click latch is engaged
    * COLLISION RISK  - objects detected while the arm is stopped (onset only)

Run modes
---------
    py PHASE9.py                 # interactive REPL: type events, watch the log grow
    py PHASE9.py --demo          # scripted demo: a full session, then print the file
    py PHASE9.py --selftest      # logic self-test (uses a temp dir, no real logs)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tempfile
import shutil
from typing import Optional, Callable


LOG_DIR = "logs"


class SessionLogger:
    """Writes one timestamped log file per monitoring session.

    Lifecycle (matches the GUI's START -> run -> STOP):
        open_session()      - create logs/session_<stamp>.log + write the header
        log(msg)            - append one timestamped line
        update_safety(...)  - append a line ONLY when the safety state changes
        close_session()     - write the end footer (with duration) and close

    `echo` mirrors the GUI showing each line in its on-screen panel: when True
    every logged line is also printed to the console. `sink` is an optional
    callback (line: str) -> None for any other destination (e.g. a GUI widget),
    exactly how COBOT_GUI feeds the same text to its QPlainTextEdit.
    """

    def __init__(self, log_dir: str = LOG_DIR, echo: bool = True,
                 sink: Optional[Callable[[str], None]] = None):
        self._log_dir = log_dir
        self._echo = echo
        self._sink = sink
        self._file = None
        self._path = None
        self._t_start = None
        self._last_safe = None
        self._last_collision = False

    @property
    def path(self):
        return self._path

    def is_open(self) -> bool:
        return self._file is not None

    def open_session(self) -> Optional[str]:
        """Start a new session: create the timestamped file and write the header.
        Returns the file path, or None if the file could not be opened."""
        self.close_session()
        self._t_start = time.time()
        self._last_safe = None
        self._last_collision = False
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._path = os.path.join(self._log_dir, f"session_{stamp}.log")
            self._file = open(self._path, "a", encoding="utf-8")
            self._file.write(
                f"===== SESSION START  {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self._file.flush()
        except Exception:
            self._file = None
        if self._file is not None:
            self.log(f"Session log -> {self._path}")
        return self._path

    def log(self, msg: str):
        """Append one timestamped line to the log (and echo / sink it)."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self._echo:
            print(line)
        if self._sink is not None:
            try:
                self._sink(line)
            except Exception:
                pass
        if self._file is not None:
            try:
                self._file.write(line + "\n"); self._file.flush()
            except Exception:
                pass

    def update_safety(self, safe: bool, estop: bool = False,
                      collision: bool = False, objects: int = 0):
        """Feed the current safety-monitor state. Logs a SAFETY line ONLY on a
        change, using the same wording as COBOT_GUI:

            safe=True                 -> "SAFE (green zone, arm resumed)"
            safe=False, estop=True    -> "EMERGENCY STOP (manual latch)"
            safe=False, estop=False   -> "STOP (red zone - hand too close...)"
            collision rising edge     -> "COLLISION RISK (N object(s)...)"
        """
        safe = bool(safe)
        estop = bool(estop)
        collision = bool(collision)
        if safe != self._last_safe:
            if self._last_safe is not None:
                if safe:
                    self.log("SAFETY  ->  SAFE (green zone, arm resumed)")
                elif estop:
                    self.log("SAFETY  ->  EMERGENCY STOP (manual latch)")
                else:
                    self.log("SAFETY  ->  STOP (red zone - hand too close, arm frozen)")
            self._last_safe = safe
        if collision and not self._last_collision:
            self.log(f"SAFETY  ->  COLLISION RISK ({objects} object(s) near a stopped arm)")
        self._last_collision = collision

    def close_session(self):
        """Write the end footer (with the session duration) and close the file."""
        if self._file is not None:
            try:
                el = int(time.time() - self._t_start) if self._t_start else 0
                self._file.write(
                    f"===== SESSION END    {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    f"  (duration {el//60:02d}:{el%60:02d}) =====\n\n")
                self._file.flush(); self._file.close()
            except Exception:
                pass
            self._file = None


HELP = """
Type an event (the matching log line is written + printed):
    start                  begin a new session log file
    safe                   safety monitor reports GREEN zone (arm may move)
    stop                   hand enters the RED zone (arm frozen)
    estop                  engage the manual emergency-stop latch
    collide [n]            collision risk with n objects (default 1)
    msg <text>             write a free-text line (like "Starting subsystems...")
    stop_session           finalise + close the current session log
    where                  print the current log file path
    help                   show this help
    quit                   close the session (if any) and exit
"""


def run_sim():
    """Type safety events and watch exactly what the cobot would log."""
    logger = SessionLogger(echo=True)
    estop = False
    print("Phase 9 - Session Logging SIMULATION")
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
            elif cmd == "where":
                print(f"   log file: {logger.path or '(no session open)'}")
            elif cmd == "start":
                logger.open_session()
                logger.log("Starting subsystems...")
            elif cmd == "stop_session":
                logger.close_session()
                print("   session closed")
            elif cmd == "safe":
                estop = False
                logger.update_safety(True, estop=False)
            elif cmd == "stop":
                logger.update_safety(False, estop=False)
            elif cmd == "estop":
                estop = True
                logger.update_safety(False, estop=True)
            elif cmd == "collide":
                n = int(rest[0]) if rest and rest[0].isdigit() else 1
                logger.update_safety(False, estop=estop, collision=True, objects=n)
            elif cmd == "msg":
                logger.log(" ".join(rest) if rest else "(empty)")
            else:
                print(f"   unknown event: {cmd!r}  (type 'help')")
    finally:
        logger.close_session()
        print("Session logger stopped.")


def run_demo():
    """A short scripted session that exercises every kind of log line, then
    prints the resulting file so you can see exactly what a session log holds."""
    logger = SessionLogger(echo=True)
    print("Phase 9 - scripted session-log demo\n")

    logger.open_session()
    logger.log("Starting subsystems...")
    logger.log("Voice alerts: pyttsx3 (ON)")
    logger.update_safety(True)
    time.sleep(0.4)
    logger.update_safety(False, collision=True, objects=2)
    time.sleep(0.4)
    logger.update_safety(False)
    time.sleep(0.4)
    logger.update_safety(True)
    time.sleep(0.4)
    logger.update_safety(False, estop=True)
    time.sleep(0.4)
    path = logger.path
    logger.close_session()

    print("\n------ resulting log file ------")
    print(open(path, encoding="utf-8").read())
    print("Demo complete.  File:", path)


def run_selftest() -> int:
    failures = 0

    def check(cond, msg):
        nonlocal failures
        ok = bool(cond)
        failures += not ok
        print(f"  [{'ok' if ok else 'XX'}] {msg}")

    tmp = tempfile.mkdtemp(prefix="phase9_")
    try:
        logger = SessionLogger(log_dir=tmp, echo=False)

        path = logger.open_session()
        check(path and os.path.exists(path), "open_session creates a log file")
        check(logger.is_open(), "logger reports the session as open")

        logger.update_safety(True)
        body = open(path, encoding="utf-8").read()
        check("SAFETY" not in body, "first safety state is a silent baseline")

        logger.update_safety(False, collision=True, objects=3)
        body = open(path, encoding="utf-8").read()
        check("STOP (red zone" in body, "entering the red zone logs a STOP line")
        check("COLLISION RISK (3 object" in body, "collision risk logs with the object count")

        before = open(path, encoding="utf-8").read().count("SAFETY")
        logger.update_safety(False)
        after = open(path, encoding="utf-8").read().count("SAFETY")
        check(before == after, "no duplicate line while the state is held")

        logger.update_safety(True)
        logger.update_safety(False, estop=True)
        body = open(path, encoding="utf-8").read()
        check("EMERGENCY STOP (manual latch)" in body, "manual latch logs EMERGENCY STOP")

        logger.close_session()
        body = open(path, encoding="utf-8").read()
        check("SESSION END" in body and "duration" in body,
              "close_session writes the END footer with a duration")
        check(not logger.is_open(), "logger reports the session as closed")

        p2 = logger.open_session()
        check(logger.is_open() and os.path.exists(p2), "a new session opens a fresh file")
        logger.close_session()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures == 0:
        print("\n[selftest] PASS")
        return 0
    print(f"\n[selftest] FAIL ({failures} check(s))")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 - standalone session logging system")
    parser.add_argument("--demo", action="store_true",
                        help="run the scripted demo session and print the file")
    parser.add_argument("--selftest", action="store_true",
                        help="run the silent logic self-test (uses a temp dir)")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())
    elif args.demo:
        run_demo()
    else:
        run_sim()


if __name__ == "__main__":
    main()
