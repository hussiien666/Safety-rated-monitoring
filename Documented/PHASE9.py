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

from __future__ import annotations               # Allow lazy string type hints.

import argparse                                   # Parse --demo / --selftest flags.
import os                                         # Make the logs/ folder, join paths.
import sys                                        # argv + process exit codes.
import time                                       # Wall-clock timestamps + session duration.
import tempfile                                   # Isolated temp dir for the self-test.
import shutil                                     # Remove the temp dir after the self-test.
from typing import Optional, Callable             # Type hints ("X or None", callback).


# ----------------------------------------------------------------------------
# Configuration  (mirrors the matching knob in COBOT_GUI.py)
# ----------------------------------------------------------------------------
LOG_DIR = "logs"                                  # Folder where per-session log files are written.


# ----------------------------------------------------------------------------
# Core: the session logger  (verbatim mirror of COBOT_GUI's logging behaviour)
# ----------------------------------------------------------------------------
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
        self._log_dir = log_dir                   # Where session files live.
        self._echo = echo                         # Print each line to the console?
        self._sink = sink                         # Optional extra destination for each line.
        self._file = None                         # Open file handle (or None when no session).
        self._path = None                         # Path of the current session file.
        self._t_start = None                      # Session start time (for the duration footer).
        # Previous states - so we log only TRANSITIONS, not every update.
        self._last_safe = None                    # Last overall safe state (None = no prior frame).
        self._last_collision = False              # Last collision-risk state.

    # ---- properties -------------------------------------------------------
    @property
    def path(self):
        return self._path                         # Expose the current log path (read-only).

    def is_open(self) -> bool:
        return self._file is not None             # True while a session log is open.

    # ---- lifecycle --------------------------------------------------------
    def open_session(self) -> Optional[str]:
        """Start a new session: create the timestamped file and write the header.
        Returns the file path, or None if the file could not be opened."""
        self.close_session()                      # Finalise any previous session first.
        self._t_start = time.time()               # Record the session start time.
        self._last_safe = None                    # Reset the transition trackers...
        self._last_collision = False              # ...so the first state is a fresh baseline.
        try:
            os.makedirs(self._log_dir, exist_ok=True)  # Ensure logs/ exists.
            stamp = time.strftime("%Y%m%d_%H%M%S")     # Filename timestamp.
            self._path = os.path.join(self._log_dir, f"session_{stamp}.log")  # Full path.
            self._file = open(self._path, "a", encoding="utf-8")  # Open for appending (UTF-8).
            self._file.write(                     # Write the session-start banner.
                f"===== SESSION START  {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            self._file.flush()                    # Flush so the header hits disk now.
        except Exception:
            self._file = None                     # On any I/O error, disable file logging.
        if self._file is not None:                # If it opened...
            self.log(f"Session log -> {self._path}")  # ...note the path (also into the file).
        return self._path

    def log(self, msg: str):
        """Append one timestamped line to the log (and echo / sink it)."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"  # Prefix with HH:MM:SS.
        if self._echo:                            # Mirror the GUI's on-screen panel...
            print(line)                           # ...by printing to the console.
        if self._sink is not None:                # Optional extra destination (e.g. a widget)...
            try:
                self._sink(line)
            except Exception:
                pass                              # A bad sink must never break logging.
        if self._file is not None:                # If a session is open...
            try:
                self._file.write(line + "\n"); self._file.flush()  # ...persist to disk immediately.
            except Exception:
                pass                              # Never let logging crash the caller.

    def update_safety(self, safe: bool, estop: bool = False,
                      collision: bool = False, objects: int = 0):
        """Feed the current safety-monitor state. Logs a SAFETY line ONLY on a
        change, using the same wording as COBOT_GUI:

            safe=True                 -> "SAFE (green zone, arm resumed)"
            safe=False, estop=True    -> "EMERGENCY STOP (manual latch)"
            safe=False, estop=False   -> "STOP (red zone - hand too close...)"
            collision rising edge     -> "COLLISION RISK (N object(s)...)"
        """
        safe = bool(safe)                         # Normalise to clean bools.
        estop = bool(estop)
        collision = bool(collision)
        if safe != self._last_safe:               # Overall safe state changed since last call...
            if self._last_safe is not None:       # ...(skip the very first call, which has no prior).
                if safe:                          # Returned to safe.
                    self.log("SAFETY  ->  SAFE (green zone, arm resumed)")
                elif estop:                       # Unsafe due to the manual latch.
                    self.log("SAFETY  ->  EMERGENCY STOP (manual latch)")
                else:                             # Unsafe because a hand is too close.
                    self.log("SAFETY  ->  STOP (red zone - hand too close, arm frozen)")
            self._last_safe = safe                # Remember for next time.
        if collision and not self._last_collision:  # Collision risk just began (rising edge)...
            self.log(f"SAFETY  ->  COLLISION RISK ({objects} object(s) near a stopped arm)")
        self._last_collision = collision          # Remember collision state.

    def close_session(self):
        """Write the end footer (with the session duration) and close the file."""
        if self._file is not None:                # Only if a session is open...
            try:
                el = int(time.time() - self._t_start) if self._t_start else 0  # Duration in seconds.
                self._file.write(                 # Footer: end time + MM:SS duration.
                    f"===== SESSION END    {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    f"  (duration {el//60:02d}:{el%60:02d}) =====\n\n")
                self._file.flush(); self._file.close()  # Flush + close.
            except Exception:
                pass                              # Ignore errors during close.
            self._file = None                     # Drop the handle (session is over).


# ----------------------------------------------------------------------------
# Run mode 1: interactive REPL  (type events, watch the session log grow)
# ----------------------------------------------------------------------------
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
    logger = SessionLogger(echo=True)             # Print every line as it is written.
    estop = False                                 # Track the manual e-stop latch locally.
    print("Phase 9 - Session Logging SIMULATION")
    print(HELP)
    try:
        while True:                               # REPL loop.
            try:
                line = input("event> ").strip()   # Read a command.
            except EOFError:                      # Ctrl-Z / Ctrl-D ends input.
                break
            if not line:                          # Ignore blank lines.
                continue
            parts = line.split()                  # Tokenise.
            cmd, rest = parts[0].lower(), parts[1:]  # Command + args.

            if cmd in ("quit", "exit"):           # Exit.
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "where":                  # Show the current log path.
                print(f"   log file: {logger.path or '(no session open)'}")
            elif cmd == "start":                  # Open a new session log.
                logger.open_session()
                logger.log("Starting subsystems...")
            elif cmd == "stop_session":           # Close the current session log.
                logger.close_session()
                print("   session closed")
            elif cmd == "safe":                   # Safety monitor → GREEN.
                estop = False
                logger.update_safety(True, estop=False)
            elif cmd == "stop":                   # Safety monitor → RED (hand too close).
                logger.update_safety(False, estop=False)
            elif cmd == "estop":                  # Engage the manual e-stop latch.
                estop = True
                logger.update_safety(False, estop=True)
            elif cmd == "collide":                # Collision risk (optional object count).
                n = int(rest[0]) if rest and rest[0].isdigit() else 1
                logger.update_safety(False, estop=estop, collision=True, objects=n)
            elif cmd == "msg":                    # Free-text log line.
                logger.log(" ".join(rest) if rest else "(empty)")
            else:
                print(f"   unknown event: {cmd!r}  (type 'help')")
    finally:
        logger.close_session()                    # Always finalise the log on exit.
        print("Session logger stopped.")


# ----------------------------------------------------------------------------
# Run mode 2: scripted demo  (run a full session, then print the file)
# ----------------------------------------------------------------------------
def run_demo():
    """A short scripted session that exercises every kind of log line, then
    prints the resulting file so you can see exactly what a session log holds."""
    logger = SessionLogger(echo=True)
    print("Phase 9 - scripted session-log demo\n")

    logger.open_session()                         # START.
    logger.log("Starting subsystems...")          # A normal status line.
    logger.log("Voice alerts: pyttsx3 (ON)")      # Another status line.
    logger.update_safety(True)                    # First state = SAFE (baseline, not logged).
    time.sleep(0.4)
    logger.update_safety(False, collision=True, objects=2)  # -> STOP + COLLISION RISK.
    time.sleep(0.4)
    logger.update_safety(False)                   # Still stopped (no new line - no change).
    time.sleep(0.4)
    logger.update_safety(True)                    # -> SAFE again.
    time.sleep(0.4)
    logger.update_safety(False, estop=True)       # -> EMERGENCY STOP (manual latch).
    time.sleep(0.4)
    path = logger.path                            # Remember the file path...
    logger.close_session()                        # END (writes the footer).

    print("\n------ resulting log file ------")
    print(open(path, encoding="utf-8").read())    # Show the whole file.
    print("Demo complete.  File:", path)


# ----------------------------------------------------------------------------
# Run mode 3: self-test  (logic only - writes to a TEMP dir, then cleans up)
# ----------------------------------------------------------------------------
def run_selftest() -> int:
    failures = 0                                  # Count of failed checks.

    def check(cond, msg):                         # Tiny assertion helper.
        nonlocal failures
        ok = bool(cond)
        failures += not ok                        # +1 on failure (False -> 1).
        print(f"  [{'ok' if ok else 'XX'}] {msg}")

    tmp = tempfile.mkdtemp(prefix="phase9_")      # Isolated temp dir (no real logs touched).
    try:
        logger = SessionLogger(log_dir=tmp, echo=False)  # Silent logger into the temp dir.

        # 1) opening a session creates a file with the START header
        path = logger.open_session()
        check(path and os.path.exists(path), "open_session creates a log file")
        check(logger.is_open(), "logger reports the session as open")

        # 2) the first safety state is a baseline and is NOT logged
        logger.update_safety(True)               # baseline SAFE - should write nothing
        body = open(path, encoding="utf-8").read()
        check("SAFETY" not in body, "first safety state is a silent baseline")

        # 3) a real change IS logged, with the GUI's wording
        logger.update_safety(False, collision=True, objects=3)
        body = open(path, encoding="utf-8").read()
        check("STOP (red zone" in body, "entering the red zone logs a STOP line")
        check("COLLISION RISK (3 object" in body, "collision risk logs with the object count")

        # 4) holding the same state does NOT add a duplicate line
        before = open(path, encoding="utf-8").read().count("SAFETY")
        logger.update_safety(False)              # still stopped - no change
        after = open(path, encoding="utf-8").read().count("SAFETY")
        check(before == after, "no duplicate line while the state is held")

        # 5) emergency-stop wording when unsafe due to the manual latch
        logger.update_safety(True)              # back to safe first...
        logger.update_safety(False, estop=True)
        body = open(path, encoding="utf-8").read()
        check("EMERGENCY STOP (manual latch)" in body, "manual latch logs EMERGENCY STOP")

        # 6) closing writes the END footer and releases the file
        logger.close_session()
        body = open(path, encoding="utf-8").read()
        check("SESSION END" in body and "duration" in body,
              "close_session writes the END footer with a duration")
        check(not logger.is_open(), "logger reports the session as closed")

        # 7) a brand-new session goes to a different file (kept distinct)
        #    (we can't rely on the second clock; just confirm it opens cleanly)
        p2 = logger.open_session()
        check(logger.is_open() and os.path.exists(p2), "a new session opens a fresh file")
        logger.close_session()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)    # Always remove the temp dir.

    if failures == 0:                             # Summarise.
        print("\n[selftest] PASS")
        return 0                                   # Success exit code.
    print(f"\n[selftest] FAIL ({failures} check(s))")
    return 1                                        # Failure exit code.


def main():
    parser = argparse.ArgumentParser(             # CLI argument parsing.
        description="Phase 9 - standalone session logging system")
    parser.add_argument("--demo", action="store_true",       # --demo flag.
                        help="run the scripted demo session and print the file")
    parser.add_argument("--selftest", action="store_true",   # --selftest flag.
                        help="run the silent logic self-test (uses a temp dir)")
    args = parser.parse_args()                    # Parse argv.

    if args.selftest:                             # Dispatch to the chosen mode.
        sys.exit(run_selftest())
    elif args.demo:
        run_demo()
    else:
        run_sim()                                 # Default: interactive REPL.


if __name__ == "__main__":                         # Only run when executed directly.
    main()
