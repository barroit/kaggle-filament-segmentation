# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from multiprocessing import Pool as alloc_pool, cpu_count
from os import unlink

from cv2 import imread, imwrite

from lib.path import *

def to_png(src):
	dst = path_replace_suffix(src, 'png')
	image = imread(src)

	imwrite(dst, image)
	unlink(src)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)

	srcs = path_glob_sorted(f"{DATASET_SHARED_IMAGE_DIR}/*.jpeg")

	pool.map(to_png, srcs)
