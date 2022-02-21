#!/usr/bin/env python3

import logging
import retroarch_api
import time
from argparse import ArgumentParser


def get_cli() -> ArgumentParser:
    """Get the command line interface for this program."""
    parser = ArgumentParser(description="")
    # TODO

    return parser


def main():
    retroarch_path = "retroarch"
    core_path = "rom/bsnes_mercury_balanced_libretro.so"
    rom_path = "rom/smwUSA.sfc"

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)

    retro = retroarch_api.RetroArchAPI(retroarch_path, core_path, rom_path)
    time.sleep(10)
    print("Testing commands")
    retro.cmd_pause_toggle()
    time.sleep(1)
    retro.cmd_pause_toggle()
    time.sleep(1)
    print(retro.cmd_get_status())
    time.sleep(1)
    retro.cmd_read_memory("96", 1)

    time.sleep(3)
    retro.cmd_quit(True)


if __name__ == "__main__":
    main()
