# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from os.path import basename, splitext

from lib.kconfig import *

__all__ = [
	'PREVIEW_PREFIX',
	'PREVIEW_TRAIN_DIR',
	'PREVIEW_YOLO_DIR',

	'TRAIN_PREFIX',
	'TRAIN_METADATA',
	'TRAIN_IMAGE_DIR',

	'YOLO_PREFIX',
	'YOLO_METADATA',
	'YOLO_TRAIN_IMAGE_DIR',
	'YOLO_TRAIN_LABEL_DIR',
	'YOLO_VERIFY_IMAGE_DIR',
	'YOLO_VERIFY_LABEL_DIR',

	'GLOB_TRAIN_IMAGES',
	'GLOB_YOLO_IMAGES',
	'GLOB_YOLO_LABELS',

	'path_resolve_stem',
]

PREVIEW_PREFIX = f"{CONFIG_DATASET_PREFIX}/preview"
PREVIEW_TRAIN_DIR = f"{PREVIEW_PREFIX}/train"
PREVIEW_YOLO_DIR = f"{PREVIEW_PREFIX}/yolo"

TRAIN_PREFIX = f"{CONFIG_DATASET_PREFIX}/train"
TRAIN_METADATA = f"{TRAIN_PREFIX}/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGE_DIR = f"{TRAIN_PREFIX}/train_images"

YOLO_PREFIX = f"{CONFIG_DATASET_PREFIX}/yolo"
YOLO_METADATA = f"{YOLO_PREFIX}/dataset.yaml"
YOLO_TRAIN_IMAGE_DIR = f"{YOLO_PREFIX}/train/images"
YOLO_TRAIN_LABEL_DIR = f"{YOLO_PREFIX}/train/labels"
YOLO_VERIFY_IMAGE_DIR = f"{YOLO_PREFIX}/verify/images"
YOLO_VERIFY_LABEL_DIR = f"{YOLO_PREFIX}/verify/labels"

GLOB_TRAIN_IMAGES = f"{TRAIN_IMAGE_DIR}/*.jpeg"
GLOB_YOLO_IMAGES = f"{YOLO_PREFIX}/*/images/*.jpeg"
GLOB_YOLO_LABELS = f"{YOLO_PREFIX}/*/labels/*.txt"

def path_resolve_stem(path):
	name = basename(path)
	parts = splitext(name)

	return parts[0]
