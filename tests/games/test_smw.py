#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest

from games.smw import SuperMarioWorldWrapper


class TestSuperMarioWorldWrapper:

    @pytest.fixture
    def two_screens(self):
        # Normal screen with nothing but ground.
        screen = bytearray([0]*16*22 + [1]*16 + [0]*16)
        # Screen with staggered terrain similar to Donut Plains 1.
        screen2 = bytearray([0]*16*18 + [0]*8 + [1]*8 + [0]*16 + [1]*16 + [0]*16 + [1]*16 + [0]*16)
        yield screen, screen2

    def test_get_visible_screen_length(self, two_screens):
        wrap = SuperMarioWorldWrapper(None)
        result = [row for row in wrap._get_visible_screen(two_screens[0], two_screens[1], 5, 2)]
        assert len(result) == 16
        assert len(result[0]) == 16
        assert len(result[-1]) == 16

    @pytest.mark.parametrize("off_x", [-5, 16, 28])
    @pytest.mark.parametrize("off_y", [-3, 9, 24])
    def test_get_visible_screen_invalid_offsets(self, two_screens, off_x, off_y):
        wrap = SuperMarioWorldWrapper(None)
        result = [row for row in wrap._get_visible_screen(two_screens[0], two_screens[1], off_x, off_y)]
        assert len(result) == 16
        assert len(result[0]) == 16
        assert len(result[-1]) == 16

    def test_get_visible_screen_combining(self, two_screens):
        wrap = SuperMarioWorldWrapper(None)
        result = [row for row in wrap._get_visible_screen(two_screens[0], two_screens[1], 10, 12)]
        assert result[-6] == bytearray([0]*14 + [1]*2)
        assert result[-4] == bytearray([0]*6 + [1]*10)
        assert result[-2] == bytearray([1]*16)
