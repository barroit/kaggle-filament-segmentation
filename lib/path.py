# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from glob import glob
from os.path import basename, splitext

from lib.kconfig import *

__all__ = [
	'MAGFILO_METADATA_FILE',
	'MAGFILO_IMAGE_DIR',

	'DATASET_SHARED_DIR',
	'DATASET_SHARED_IMAGE_DIR',
	'DATASET_SHARED_ANNOTATIONS',

	'DATASET_YOLO_DIR',
	'DATASET_YOLO_METADATA',
	'DATASET_YOLO_TRAIN_IMAGE_DIR',
	'DATASET_YOLO_TRAIN_LABEL_DIR',
	'DATASET_YOLO_VERIFY_IMAGE_DIR',
	'DATASET_YOLO_VERIFY_LABEL_DIR',

	'DATASET_SHARED_IMAGE_GLOB',
	'DATASET_YOLO_IMAGE_GLOB',
	'DATASET_YOLO_LABEL_GLOB',

	'path_resolve_stem',
	'path_glob_sorted',
	'path_replace_suffix',
]

MAGFILO_METADATA_FILE = f"{CONFIG_MAGFILO_DIR}/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
MAGFILO_IMAGE_DIR = f"{CONFIG_MAGFILO_DIR}/train_images"

DATASET_SHARED_DIR = f"{CONFIG_DATASET_DIR}/shared"
DATASET_SHARED_IMAGE_DIR = f"{DATASET_SHARED_DIR}/images"
DATASET_SHARED_ANNOTATIONS = f"{DATASET_SHARED_DIR}/annotations.json"

DATASET_YOLO_DIR = f"{CONFIG_DATASET_DIR}/yolo"
DATASET_YOLO_METADATA = f"{DATASET_YOLO_DIR}/dataset.yaml"
DATASET_YOLO_TRAIN_IMAGE_DIR = f"{DATASET_YOLO_DIR}/train/images"
DATASET_YOLO_TRAIN_LABEL_DIR = f"{DATASET_YOLO_DIR}/train/labels"
DATASET_YOLO_VERIFY_IMAGE_DIR = f"{DATASET_YOLO_DIR}/verify/images"
DATASET_YOLO_VERIFY_LABEL_DIR = f"{DATASET_YOLO_DIR}/verify/labels"

DATASET_SHARED_IMAGE_GLOB = f"{DATASET_SHARED_IMAGE_DIR}/*.png"
DATASET_YOLO_IMAGE_GLOB = f"{DATASET_YOLO_DIR}/*/images/*.jpeg"
DATASET_YOLO_LABEL_GLOB = f"{DATASET_YOLO_DIR}/*/labels/*.txt"

def path_resolve_stem(path):
	filename = basename(path)
	prefix, _ = splitext(name)

	return prefix

def path_glob_sorted(pattern):
	files = glob(pattern)

	files.sort()
	return files

def path_replace_suffix(path, suffix):
	prefix, _ = splitext(path)

	return f"{prefix}.{suffix}"
