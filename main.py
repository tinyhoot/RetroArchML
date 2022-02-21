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
    retro.pause_toggle()
    time.sleep(1)
    print(retro.get_version())
    retro.get_status()
    retro.pause_toggle()
    time.sleep(1)
    print(retro.get_status())
    time.sleep(1)
    print(retro.read_memory("004e", 1))
    print(retro.read_memory("010A", 2))
    print(retro.read_memory("0068", 8))

    time.sleep(3)
    retro.quit(True)


if __name__ == "__main__":
    main()
