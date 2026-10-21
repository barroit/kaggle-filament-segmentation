#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from shutil import copytree
from subprocess import run
from sys import executable

from lib.kconfig import *
from lib.path import *

def apply(name):
	run([executable, f"preprocess/{name}.py"], check=True)

copytree(MAGFILO_IMAGE_DIR, DATASET_SHARED_IMAGE_DIR)

apply('to_png')
apply('format_metadata')

if CONFIG_FILTER_ANNO:
	apply('filter_anno')
