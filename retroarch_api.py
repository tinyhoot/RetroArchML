import logging
import socket
import subprocess
import threading
import time
from subprocess import Popen
from typing import Union, Tuple


class RetroArchAPI:

    _socket = None
    _ip = "localhost"
    _port = 55355

    def __init__(self, retroarch: str, core: str, rom: str, enable_stdin: bool = False):
        """Run a RetroArch process and prepare it for stdin communication.

        :param retroarch: The path to a retroarch executable.
        :param core: The path to the libretro core used for running the game.
        :param rom: The path to the rom of the game to be run.
        :param enable_stdin: If enabled, will try to use stdin rather than network commands to control RetroArch.
        """
        # Init the RetroArch process
        self._process = Popen([retroarch, "-L", core, rom, "--appendconfig", "config.cfg", "--verbose"], bufsize=1,
                              stdin=subprocess.PIPE, stdout=None, stderr=None, text=True)

        # Init the stdout monitor thread
        if enable_stdin:
            self._stdout_queue = []
            self._stdout_event = threading.Event()
            self._stdout_lock = threading.Lock()
            stdout_monitor = threading.Thread(target=self._monitor_stdout, daemon=True)
            stdout_monitor.start()

        # Init the network socket
        if not enable_stdin:
            self._socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
            self._socket.settimeout(3)

        logging.info("Finished initialisation.")

    @staticmethod
    def _prepare_command(command: str) -> bytes:
        """Strip any special characters from the command and encode it into bytes."""
        command = command.rstrip().strip(r" \\?!-_./:")
        command += "\n"
        return command.encode()

    @staticmethod
    def _process_response(response: bytes) -> Tuple[str, str]:
        """Process the received byte response into something usable.

        :return: A tuple containing the issuing command and the actual content of the response.
        """
        response = response.decode().rstrip()

        # Filter out the input command
        index = response.find(" ")
        return response[:index], response[index+1:]

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
                self._stdout_queue.append(line)
                # Notify the main thread that a change has occurred.
                self._stdout_event.set()
            logging.debug("STDOUT: " + line)

        logging.debug("Process has ended, stopping stdout monitor thread.")

    def _flush_stdout(self):
        """Shorten the stdout queue by discarding old entries."""
        if len(self._stdout_queue) > 10:
            with self._stdout_lock:
                self._stdout_queue = self._stdout_queue[-10:]

    def _read_stdout(self) -> Union[str, None]:
        """Read the most recent addition to the stdout queue."""
        with self._stdout_lock:
            # If the queue is empty, just return None instead.
            if len(self._stdout_queue) == 0:
                return None
            last_line = self._stdout_queue.pop()
        # Calling this here will probably keep the stdout queue reasonably short.
        self._flush_stdout()

        return last_line

    def _write_stdin(self, command: str):
        """Send a command to RetroArch via stdin."""
        # Ensure the command is properly formatted
        command = command.rstrip() + "\n"

        logging.info("Writing command to stdin: "+command.rstrip())
        self._process.stdin.write(command)

    def _get_network_response(self, bufsize: int) -> bytes:
        """Receive a network response from RetroArch.

        This will time out and throw an exception after no response is captured for 3 seconds (default).

        :param bufsize: The maximum number of bytes to read.
        :return: Bytes received from RetroArch."""
        response, address = self._socket.recvfrom(bufsize)
        logging.debug(f"Received network response: {response}")
        return response

    def _send_network_cmd(self, command: str):
        """Send a command to RetroArch via the network command interface."""
        logging.debug("Send network cmd: " + command)
        command = self._prepare_command(command)
        self._socket.sendto(command, (self._ip, self._port))

    def frame_advance(self):
        # TODO Seems to be kinda wacky. Needs more research.
        self._send_network_cmd("FRAMEADVANCE")

    def get_status(self) -> str:
        """Get information about the currently running content.

        :return: The status string returned by RetroArch.
        """
        self._send_network_cmd("GET_STATUS")
        status = self._get_network_response(64)
        cmd, status_str = self._process_response(status)
        return status_str

    def get_version(self) -> str:
        """Get RetroArch's running version."""
        self._send_network_cmd("VERSION")
        response = self._get_network_response(16)
        return response.decode().rstrip()

    def pause_toggle(self):
        """Toggle pausing the currently running content."""
        self._send_network_cmd("PAUSE_TOGGLE")

    def quit(self, confirm: bool = False):
        """Exit RetroArch.

        Because RetroArch by default always asks for confirmation before quitting, it needs to receive two QUIT calls to
        actually do it; this is handled by the optional confirm parameter.

        :param confirm: Skip RetroArch asking for confirmation and quit immediately.
        """
        self._send_network_cmd("QUIT")
        if confirm:
            time.sleep(0.1)
            self._send_network_cmd("QUIT")

    def save_state(self):
        """Save the game state to the currently selected slot."""
        self._send_network_cmd("SAVE_STATE")

    def load_state(self):
        """Load a game state from the currently selected slot."""
        self._send_network_cmd("LOAD_STATE")

    def read_memory(self, address: str, byte_count: int) -> bytearray:
        """Read memory from the currently running content.

        Requires a core with memory mapping capabilities, otherwise RetroArch cannot read/write anything and this
        function will have no effect. Does not work via stdin.

        :param address: The address to read from, formatted in hex (e.g. 00ff)
        :param byte_count: The number of bytes to read.
        :return: A bytearray containing the bytes read at the specified address.
        """
        # Ensure any 0x prefixes are stripped and not sent along with the command.
        if address.startswith("0x"):
            address = address[2:]
        self._send_network_cmd(f"READ_CORE_MEMORY {address} {byte_count}")

        response = self._get_network_response(4096)

        if b"no memory map" in response:
            logging.error("Running core does not provide a memory map!")
            raise RuntimeError("The current running core does not provide a memory map. Cannot read from memory.")

        # This decodes the received bytes into a string, and then re-encodes it into a byte array.
        # Ultimately the simplest way to remove whitespace bytes in between the actual data.
        response = response.removeprefix(b"READ_CORE_MEMORY ").rstrip().decode()
        # The first "byte" in the remaining string is actually the address, filter it here.
        index = response.find(" ")
        b_arr = bytearray.fromhex(response[index+1:])

        return b_arr


