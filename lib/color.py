# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

__all__ = [
	'color_rgba_to_bgra',
	'color_alpha_percent',
]

def color_rgba_to_bgra(rgba):
	return (rgba[2], rgba[1], rgba[0], rgba[3])

def color_alpha_percent(xxxa):
	return xxxa[3] / 255
