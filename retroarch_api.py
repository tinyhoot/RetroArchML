import logging
import socket
import subprocess
import time
from subprocess import Popen
from typing import Union, Tuple


class RetroArchAPI:

    _socket = None
    _ip = "localhost"
    _port = 55355

    def __init__(self, retroarch: str, core: str, rom: str):
        """Run a RetroArch process and prepare it for network communication.

        :param retroarch: The path to a retroarch executable.
        :param core: The path to the libretro core used for running the game.
        :param rom: The path to the rom of the game to be run.
        """
        # Init the RetroArch process
        self._process = Popen([retroarch, "-L", core, rom, "--appendconfig", "config.cfg", "--verbose"], bufsize=1,
                              stdin=subprocess.PIPE, stdout=None, stderr=None, text=True)

        # Init the network socket
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

    def _get_network_response(self, bufsize: int) -> bytes:
        """Receive a network response from RetroArch.

        This will time out and throw an exception after no response is captured for 3 seconds (default).

        :param bufsize: The maximum number of bytes to read.
        :return: Bytes received from RetroArch."""
        response, address = self._socket.recvfrom(bufsize)
        logging.debug(f"Received network response: {response}")
        return response

    def _send_network_cmd(self, command: str) -> bool:
        """Send a command to RetroArch via the network command interface."""
        logging.debug("Send network cmd: " + command)
        command = self._prepare_command(command)
        try:
            self._socket.sendto(command, (self._ip, self._port))
        except InterruptedError:
            logging.error(f"Failed to send command to socket! Command was: {command}")
            return False

        return True

    def frame_advance(self):
        # TODO Seems to be kinda wacky. Needs more research.
        self._send_network_cmd("FRAMEADVANCE")

    def get_status(self) -> Tuple[str]:
        """Get information about the currently running content.

        :return: A tuple containing pause status, platform name, content name, and crc32 checksum.
        """
        self._send_network_cmd("GET_STATUS")
        status = self._get_network_response(64)

        # Discard fluff and unhelpful parts.
        status_str = self._process_response(status)[1]
        status_str = status_str.replace("crc32=", "")

        # Handle pause status separately.
        stat_list = [status_str.split(" ")[0]]
        # Add comma-separated platform,content,crc32checksum
        stat_list += [x for x in status_str.split(" ")[1].split(",")]

        return tuple(stat_list)

    def get_version(self) -> str:
        """Get RetroArch's running version."""
        self._send_network_cmd("VERSION")
        response = self._get_network_response(16)
        return self._process_response(response)[1]

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

        response_str = self._process_response(response)[1]
        # The first "byte" in the remaining string is actually the address, filter it here.
        index = response_str.find(" ")
        b_arr = bytearray.fromhex(response_str[index+1:])

        return b_arr
