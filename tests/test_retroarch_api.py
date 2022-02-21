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
        monkeypatch.setattr(RetroArchAPI, "__init__", mock_process_get)
        retro = RetroArchAPI("", "", "")
        # Keep up appearances by pretending stdin exists.
        retro._process.stdin = open("tmp.txt", "w", encoding="utf-8")
        yield retro
        # Remove fake stdin after the tests are done.
        os.remove("tmp.txt")

    @pytest.mark.real_process
    def test__init__is_alive(self, retroarch):
        """Check if the retroarch process started up correctly."""
        assert retroarch.poll() is None

    @pytest.mark.parametrize("response", ["GET_STATUS PLAYING super_snes,bsnes,08fdb21e",
                                          "READ_CORE_MEMORY 10a 18 ac"])
    def test_process_response(self, mock_retroarch, response: bytes):
        assert isinstance(mock_retroarch._process_response(response), Tuple)

    @pytest.mark.parametrize("command", ["PAUSE_TOGGLE", "SAVE_STATE\n", "LOAD_STATE\\n"])
    def test_write_stdin(self, mock_retroarch, command: str):
        """Check if commands are being written to stdin properly."""
        mock_retroarch._write_stdin(command)
        mock_retroarch._process.stdin.flush()
        with open("tmp.txt", "r", encoding="utf-8") as mock_stdin:
            line = mock_stdin.readline()
            assert line.rstrip() == command.rstrip()
            assert line == command.rstrip() + "\n"

    def test_cmd_quit(self, mock_retroarch):
        mock_retroarch.cmd_quit()
        mock_retroarch._process.stdin.flush()
        with open("tmp.txt", "r", encoding="utf-8") as mock_stdin:
            assert mock_stdin.readlines() == ["QUIT\n"]

    def test_cmd_quit_confirm(self, mock_retroarch):
        mock_retroarch.cmd_quit(True)
        mock_retroarch._process.stdin.flush()
        with open("tmp.txt", "r", encoding="utf-8") as mock_stdin:
            assert mock_stdin.readlines() == ["QUIT\n", "QUIT\n"]


# Mock replacement for the RetroArch process
class MockProcess:

    stdin: IO[AnyStr] = None


def mock_process_get(self, *args, **kwargs):
    self._process = MockProcess()
