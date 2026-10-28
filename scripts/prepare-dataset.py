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
apply('format_anno')

apply('partition_period')

apply('assign_dataset')
print('Annotation Distribution (assign_dataset)')
apply('show_set_anno_count')
print()

apply('enforce_set_gap')
print('Annotation Distribution (enforce_set_gap)')
apply('show_set_anno_count')
print()

if CONFIG_DROP_SUN_LIMB:
	apply('drop_sun_limb')

if CONFIG_DROP_CORRUPTED_ANNO:
	apply('drop_corrupted_anno')
	print('Annotation Distribution (drop_corrupted_anno)')
	apply('show_set_anno_count')
	print()

if CONFIG_DROP_CONFLICTED_ANNO:
	apply('drop_conflicted_anno')
	print('Annotation Distribution (drop_conflicted_anno)')
	apply('show_set_anno_count')
	print()

if CONFIG_DROP_OVERLAPPED_ANNO:
	apply('drop_overlapped_anno')
	print('Annotation Distribution (drop_overlapped_anno)')
	apply('show_set_anno_count')
