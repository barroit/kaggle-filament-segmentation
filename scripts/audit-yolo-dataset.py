#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from glob import glob
from os.path import exists, samefile

from cv2 import imread
from numpy import array, float64
from pycocotools.coco import COCO

from lib.coord import *
from lib.path import *

coco = COCO(MAGFILO_METADATA_FILE)
images = {}

for image_id, info in coco.imgs.items():
	name = image_id.split('-')[1]
	if name not in images:
		images[name] = info

instance_count = 0
max_error = 0.0

for name, info in images.items():
	train_label = f"{DATASET_YOLO_TRAIN_LABEL_DIR}/{name}.txt"
	verify_label = f"{DATASET_YOLO_VERIFY_LABEL_DIR}/{name}.txt"
	is_train = exists(train_label)
	is_verify = exists(verify_label)
	assert is_train != is_verify, f"{name}: missing or duplicate label"

	label_path = train_label if is_train else verify_label
	image_dir = DATASET_YOLO_TRAIN_IMAGE_DIR if is_train \
		    else DATASET_YOLO_VERIFY_IMAGE_DIR
	image_path = f"{image_dir}/{info['file_name']}"

	assert samefile(image_path,
			f"{MAGFILO_IMAGE_DIR}/{info['file_name']}"),  name

	image = imread(image_path)
	height, width = image.shape[:2]
	assert (width, height) == (info['width'], info['height']), name

	annos = coco.imgToAnns[info['id']]
	lines = open(label_path).read().splitlines()
	assert len(lines) == len(annos), f"{name}: instance count mismatch"

	for index, (anno, line) in enumerate(zip(annos, lines)):
		assert len(anno['segmentation']) == 1, \
		       f"{name}: multipart polygon"

		cols = line.split()
		assert cols[0] == '0', f"{name}: wrong class"

		original = coord_pack_float64(anno['segmentation'][0])
		normalized = array(cols[1:], dtype=float64).reshape(-1, 2)

		assert len(original) >= 3, f"{name}: invalid polygon"
		assert original.shape == normalized.shape, \
		       f"{name}: vertex count mismatch"
		assert ((normalized >= 0) & (normalized <= 1)).all(), \
		       f"{name}: out-of-range coordinates"

		restored = normalized * [info['width'], info['height']]
		error = abs(restored - original).max()

		assert error <= 1e-6, \
		       f"{name}: polygon {index}: error {error} pixels"

		max_error = max(max_error, error)
		instance_count += 1

label_count = len(glob(f"{DATASET_YOLO_TRAIN_LABEL_DIR}/*.txt"))
label_count += len(glob(f"{DATASET_YOLO_VERIFY_LABEL_DIR}/*.txt"))

assert label_count == len(images), 'unexpected extra label files'

print(f"Validated {len(images)} images and {instance_count} instances")
print(f"Maximum coordinate error: {max_error:.12f} pixels")
