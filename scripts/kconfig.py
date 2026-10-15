# SPDX-License-Identifier: GPL-3.0-or-later

def __first_line(filename):
	file = open(filename, 'r')
	line = file.readline().rstrip()

	file.close()
	return line

def first_line(kconf, name, file):
	return __first_line(file)

def cut(kconf, name, line, field):
	cols = line.split('\t')

	return cols[int(field) - 1]

def warn_off(kconf, name):
	kconf.warn = False
	return ''

functions = {
	'first-line': (first_line, 1, 1),
	'cut': (cut, 2, 2),

	'warn-off': (warn_off, 0, 0),
}
