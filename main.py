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


def test(retro):
    #time.sleep(10)
    print("Testing commands")
    #retro.load_state()


    time.sleep(3)
    retro.quit(True)


def main():
    retroarch_path = "retroarch"
    core_path = "rom/bsnes_mercury_balanced_libretro.so"
    #core_path = "rom/snes9x_libretro.so"
    rom_path = "rom/smwUSA.sfc"

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)

    retro = retroarch_api.RetroArchAPI(retroarch_path, core_path, rom_path)
    time.sleep(10)
    test(retro)


if __name__ == "__main__":
    main()
