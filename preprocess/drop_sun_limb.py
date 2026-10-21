# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from multiprocessing import Pool as alloc_pool, cpu_count

from cv2 import circle, imread, imwrite
from numpy import zeros

from lib.kconfig import *
from lib.path import *

def drop_sun_limb(src):
	image = imread(src)
	center = tuple(axis // 2 for axis in CONFIG_IMAGE_DIMENSION)
	mask = zeros(CONFIG_IMAGE_DIMENSION[::-1], dtype='uint8')

	circle(mask, center, CONFIG_SUN_DISK_RADIUS, 1, -1)
	image[mask == 0] = 0

	imwrite(src, image)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)

	srcs = path_glob_sorted(DATASET_SHARED_IMAGE_GLOB)

	pool.map(drop_sun_limb, srcs)
