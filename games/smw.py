#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterator

from retroarch_api import RetroArchAPI


class SuperMarioWorldWrapper:
    """A wrapper for interfacing with Super Mario World (USA) through RetroArch."""

    def __init__(self, retroarch: RetroArchAPI):
        self.retroarch = retroarch

    def get_player_pos(self) -> tuple[int, int]:
        """Get Mario's current position within the level (not the screen!)."""
        x_bytes = self.retroarch.read_memory("7E00D1", 2)
        y_bytes = self.retroarch.read_memory("7E00D3", 2)
        x = int.from_bytes(x_bytes, "little")
        y = int.from_bytes(y_bytes, "little")

        return x, y

    def get_tiles(self) -> list[bytearray]:
        """Get the 16x16 tiles currently on screen."""
        # Here's how it works: The game loads the entire level into memory at two addresses, 0x7EC800 and 0x7FC800. The
        # space allocated to them is massive, clocking in at 14336 bytes each.
        # Each block in the level corresponds to a single byte in this address space, and every block has its data
        # split up across the two addresses; the high byte is stored in 0x7FC800, while the low byte is stored in the
        # other one. It just so happens that for the purposes of terrain detection, it is completely sufficient to read
        # the high byte, as that will generally be set to 1 if Mario can stand on the block.
        # For horizontal levels, the data is organised in sections, or screens, of 0x1B0 bytes. Each of these screens
        # holds its data left to right, top to bottom in a 24x16 grid (rows x columns). The uppermost 8 rows are not
        # rendered on screen unless Mario somehow jumps up there, leaving the camera focused on the lower 16x16 range.
        offset = int("7FC800", base=16)
        # Grab the position of the left screen edge by grabbing the current x position of layer 1, which are usually
        # equivalent except for special levels that do more elaborate visual things/effects.
        camera_b = self.retroarch.read_memory("7E001A", 4)
        cam_x = int.from_bytes(camera_b[:2], "little")
        cam_y = int.from_bytes(camera_b[2:], "little")
        # 16 pixels per block, 16 horizontal blocks per screen.
        block_x = cam_x // 16
        block_y = cam_y // 16
        screen_idx = block_x // 16
        # No matter what exact pixel the camera is currently on, grabbing two screens from the current index should
        # always yield the entirety of what is visible.
        screen_addr = str(hex(offset + (screen_idx * 432)))
        screens = self.retroarch.read_memory(screen_addr, 432 * 2)
        screen_l, screen_r = screens[:432], screens[432:]
        visible_screen = [row for row in self._get_visible_screen(screen_l, screen_r, block_x % 16, block_y)]

        return visible_screen

    def _get_visible_screen(self, screen_left: bytearray, screen_right: bytearray, offset_x: int = 0,
                            offset_y: int = 0) -> Iterator[bytearray]:
        """Get a generator representing the currently visible screen space.

        Each call to the generator yields one row of 16 bytes representing a horizontal line of blocks on screen.

        :param screen_left: The Map16 screen that is being left behind.
        :param screen_right: The Map16 screen that is being entered.
        :param offset_x: By how many blocks the left screen boundary (camera) is offset from the last Map16 screen index.
        :param offset_y: By how many blocks the top screen boundary (camera) is offset from the top of the map.
        """
        # Ensure no invalid offsets are given.
        offset_x = 15 if offset_x > 15 else 0 if offset_x < 0 else offset_x
        offset_y = 8 if offset_y > 8 else 0 if offset_y < 0 else offset_y
        # Apply the vertical offset from the top of the screen.
        screen_left = screen_left[offset_y*16:(offset_y+16)*16]
        screen_right = screen_right[offset_y*16:(offset_y+16)*16]
        # Yield a 16 byte long combined subset of the two screens.
        for i in range(0, len(screen_left), 16):
            if offset_x <= 0:
                yield screen_left[i:i+16]
            else:
                yield screen_left[i+offset_x:i+16] + screen_right[i:i+offset_x]
