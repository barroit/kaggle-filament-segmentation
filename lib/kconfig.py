# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from glob import glob
from kconfiglib.kconfiglib import Kconfig as alloc_kconfig, BOOL, INT, HEX, \
				  STRING

__all__ = []
__globals__ = globals()

bool = {
	'y': True,
	'n': False,
}
kconfig = alloc_kconfig('Kconfig')

for name in glob('kconfig/gen/*'):
	if not name.endswith('.old'):
		kconfig.load_config(name, replace=False)

for symbol in kconfig.unique_defined_syms:
	name = f"CONFIG_{symbol.name}"
	val = symbol.str_value

	if not val and symbol.type in (BOOL, INT, HEX):
		continue

	if symbol.type == BOOL:
		val = bool[val]

	elif symbol.type == INT:
		val = int(val)

	elif symbol.type == HEX:
		val = int(val, 16)

	elif symbol.type == STRING:
		if symbol.name.endswith(('_RGBA', '_DIMENSION')):
			val = tuple(map(int, val.split()))

	__globals__[name] = val
	__all__.append(name)
