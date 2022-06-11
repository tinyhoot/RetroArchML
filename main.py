#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import retroarch_api
import time
from argparse import ArgumentParser
from games.smw import SuperMarioWorldWrapper


def get_cli() -> ArgumentParser:
    """Get the command line interface for this program."""
    parser = ArgumentParser(description="")
    # TODO

    return parser


def test(retro):
    time.sleep(3)
    print("Testing commands")
    retro.load_state()

    wrapper = SuperMarioWorldWrapper(retro)
    for i in range(5):
        time.sleep(5)
        print(wrapper.get_player_pos())
        wrapper.get_tiles()

    time.sleep(3)
    retro.quit(True)


def main():
    retroarch_path = "retroarch"
    core_path = "rom/bsnes_mercury_balanced_libretro.so"
    rom_path = "rom/smwUSA.sfc"

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)

    retro = retroarch_api.RetroArchAPI(retroarch_path, core_path, rom_path)
    time.sleep(10)
    test(retro)


if __name__ == "__main__":
    main()
