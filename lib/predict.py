#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from types import SimpleNamespace as simple_namespace

from lib.kconfig import *

__all__ = [
	'pred_format_result',
]

def pred_format_result(res):
	pred = simple_namespace()

	pred.bboxes = res.boxes.xyxy
	pred.confs = res.boxes.conf
	pred.masks = res.masks.data
	pred.n = len(res)

	return pred
