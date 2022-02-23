import logging
import os
import pytest
import subprocess
import time
from typing import AnyStr, IO, Tuple

from retroarch_api import RetroArchAPI


class TestRetroArchAPI:

    @pytest.fixture
    @pytest.mark.real_process
    def retroarch(self):
        retroarch = "retroarch"
        core = "../rom/snes9x_libretro.so"
        rom = "../rom/smw.sfc"

        process = subprocess.Popen([retroarch, "-L", core, rom, "--appendconfig", "config.cfg", "--verbose"], bufsize=1,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        time.sleep(5)
        yield process
        # After all the tests are done, shut down retroarch.
        process.terminate()

    @pytest.fixture
    def mock_retroarch(self, monkeypatch):
        # Replace the RetroArchAPI constructor with a hollow test case.
        monkeypatch.setattr(RetroArchAPI, "__init__", mock_api_init)
        retro = RetroArchAPI("", "", "")
        return retro

    @pytest.mark.real_process
    def test__init__is_alive(self, retroarch):
        """Check if the retroarch process started up correctly."""
        assert retroarch.poll() is None

    @pytest.mark.parametrize("response, expected",
                             [(b"GET_STATUS PLAYING super_snes,Super Mario World,crc32=08fdb21e", ("GET_STATUS", "PLAYING super_snes,Super Mario World,crc32=08fdb21e")),
                              (b"READ_CORE_MEMORY 10a 18 ac\n", ("READ_CORE_MEMORY", "10a 18 ac"))])
    def test_process_response(self, mock_retroarch, response: bytes, expected):
        assert isinstance(mock_retroarch._process_response(response), Tuple)
        assert mock_retroarch._process_response(response) == expected

    def test_get_network_response(self, mock_retroarch):
        assert mock_retroarch._get_network_response(16) == b"big response"

    def test_send_network_cmd(self, mock_retroarch):
        assert mock_retroarch._send_network_cmd("PAUSE_TOGGLE")

    def test_send_network_cmd_exception(self, mock_retroarch):
        assert not mock_retroarch._send_network_cmd("Exception")

    def test_get_config_param(self, mock_retroarch):
        assert mock_retroarch.get_config_param("GET_CONFIG_PARAM savestate_dir") == "/test/bin/retroarch/savestates"

    def test_get_config_param_unsupported(self, mock_retroarch):
        with pytest.raises(ValueError):
            mock_retroarch.get_config_param("GET_CONFIG_PARAM bad_parameter")

    def test_read_memory(self, mock_retroarch):
        assert isinstance(mock_retroarch.read_memory("1234", 128), bytearray)
        assert mock_retroarch.read_memory("1234", 128) == bytearray(b'Xi\x0fH\xca;\xfc\r')

    def test_read_memory_no_map(self, mock_retroarch):
        with pytest.raises(RuntimeError):
            mock_retroarch.read_memory("00ff", 4096)

    def test_write_memory(self, mock_retroarch):
        assert mock_retroarch.write_memory("7E0019", "03") == "7E0019 1"

    def test_write_memory_no_map(self, mock_retroarch):
        with pytest.raises(RuntimeError):
            mock_retroarch.write_memory("facf", "ff 26")


# Mock replacement for the RetroArch process
class MockProcess:

    stdin: IO[AnyStr] = None


# Mock replacement for the network socket
class MockSocket:

    response = b""

    def recvfrom(self, bufsize):
        if self.response == b"":
            return b"big response", b"bad address"
        return self.response, ""

    def sendto(self, command, ip_port_tuple):
        # test_send_network_cmd
        if b"Exception" in command:
            raise InterruptedError
        # test_get_config_param
        if b"PARAM savestate_dir" in command:
            self.response = b"GET_CONFIG_PARAM savestate_dir /test/bin/retroarch/savestates\n"
            return
        if b"PARAM bad" in command:
            self.response = b"GET_CONFIG_PARAM some_param unsupported\n"
            return
        # test_read_memory
        if b"MEMORY 00ff" in command:
            self.response = b"READ_CORE_MEMORY 96 -1 no memory map available\n"
            return
        if b"MEMORY 1234" in command:
            self.response = b"READ_CORE_MEMORY 1234 58 69 0f 48 ca 3b fc 0d\n"
            return
        # test_write_memory
        if b"MEMORY 7E" in command:
            self.response = b"WRITE_CORE_MEMORY 7E0019 1\n"
            return
        if b"MEMORY facf" in command:
            self.response = b"WRITE_CORE_MEMORY facf -1 no memory map defined\n"
            return


def mock_api_init(self, *args, **kwargs):
    self._log = logging.getLogger()
    self._process = MockProcess()
    self._socket = MockSocket()
