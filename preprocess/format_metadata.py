# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from collections import defaultdict
from itertools import repeat
from json import dump as json_dump

from pycocotools.coco import COCO as alloc_coco

from lib.path import *

coco = alloc_coco(MAGFILO_METADATA_FILE)
dst = open(DATASET_SHARED_METADATA, 'w')

metadata = defaultdict(list)

def format_anno(__anno, image, author):
	anno = dict()

	anno['id'] = __anno['id']
	anno['author'] = author
	anno['cat'] = __anno['category_id']
	anno['image'] = image
	anno['spine'] = __anno['spine']
	anno['polygon'] =  __anno['segmentation'][0]
	anno['bbox'] = __anno['bbox']
	anno['area'] = __anno['area']

	return anno

for image in coco.imgs.values():
	id = image['id']
	# The image ID prefix identifies the annotator's id in the annotation
	# batch, not necessarily a unique annotator across different batches.
	author, name = id.split('-')
	annos = map(format_anno, coco.imgToAnns[id], repeat(name),
		    repeat(author))

	metadata[name].extend(annos)

json_dump(metadata, dst, separators=(',', ':'))
