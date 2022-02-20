import logging
import subprocess
import threading
import time
from subprocess import Popen
from typing import Union


class RetroArchAPI:

    def __init__(self, retroarch: str, core: str, rom: str):
        """Run a RetroArch process and prepare it for stdin communication.

        :param retroarch: The path to a retroarch executable.
        :param core: The path to the libretro core used for running the game.
        :param rom: The path to the rom of the game to be run.
        """
        # Init the RetroArch process
        self._process = Popen([retroarch, "-L", core, rom, "--appendconfig", "config.cfg", "--verbose"], bufsize=1,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Init the stdout monitor thread
        stdout_monitor = threading.Thread(target=self._monitor_stdout, daemon=True)
        stdout_monitor.start()
        self._stdout_queue = []
        self._stdout_event = threading.Event()
        self._stdout_lock = threading.Lock()

        logging.info("Finished initialisation.")

    def _monitor_stdout(self):
        """Monitor RetroArch's stdout and stderr pipelines. Intended to be run as a separate thread.

        For some reason, RetroArch seems to log everything to stderr. This is redirected to stdout during
        process initialisation and also captured here.
        """
        logging.debug("Starting threaded stdout monitor.")

        while self._process.poll() is None:
            self._stdout_event.clear()
            # If stdout is currently empty, this will hang until there is something to be read.
            line = self._process.stdout.readline().rstrip()
            # Ensure the queue object is consistent across threads.
            with self._stdout_lock:
                # Notify the main thread that a change has occurred.
                self._stdout_event.set()
                self._stdout_queue.append(line)
            logging.debug("STDOUT: " + line)

        logging.debug("Process has ended, stopping stdout monitor thread.")

    def _flush_stdout(self):
        """Shorten the stdout queue by discarding old entries."""
        if len(self._stdout_queue) > 10:
            with self._stdout_lock:
                self._stdout_queue = self._stdout_queue[-10:]

    def _read_stdout(self) -> str:
        """Read the most recent addition to the stdout queue."""
        with self._stdout_lock:
            last_line = self._stdout_queue.pop()
        # Calling this here will probably keep the stdout queue reasonably short.
        self._flush_stdout()

        return last_line

    def _write_stdin(self, command: str):
        """Send a command to RetroArch via stdin."""
        # Ensure the command is properly formatted
        if not command.endswith("\n"):
            command += "\n"

        logging.info("Writing command to stdin: "+command.rstrip())
        self._process.stdin.write(command)

    def cmd_frame_advance(self):
        # TODO Seems to be kinda wacky. Needs more research.
        self._write_stdin("FRAMEADVANCE")

    def cmd_get_status(self) -> Union[str, None]:
        """Get information about the currently running content.

        :return: A string containing the content's CRC32 checksum, or None if no content is currently running.
        """
        self._write_stdin("GET_STATUS")
        # Wait for the monitoring thread to update the stdout queue.
        self._stdout_event.wait(3)
        status = self._read_stdout()

        if len(status) < 20:
            # Something went wrong along the way and this is likely not the actual status line.
            logging.warning("Failed to get status: response is too short!")
            return None

        # The first 18 characters of the line are just logging prefixes, splice those.
        return status[18:]

    def cmd_pause_toggle(self):
        self._write_stdin("PAUSE_TOGGLE")

    def cmd_quit(self, confirm: bool = False):
        """Exit RetroArch.

        Because RetroArch always asks for confirmation before quitting, it needs to receive two QUIT calls to actually
        do it; this is handled by the optional confirm parameter.

        :param confirm: Skip RetroArch asking for confirmation and quit immediately.
        """
        self._write_stdin("QUIT")
        if confirm:
            time.sleep(0.1)
            self._write_stdin("QUIT")

    def cmd_save_state(self):
        self._write_stdin("SAVE_STATE")

    def cmd_load_state(self):
        self._write_stdin("LOAD_STATE")
