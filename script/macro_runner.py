#!D:\Python34\python.exe
# -*- coding: utf-8 -*-
import sys, importlib, os
from docopt import docopt
from python_macro import install_hook
importer = install_hook()

def execute(*args):
    basepath = os.path.abspath(os.path.curdir)
    for arg in args:
        sys.path.append(os.path.abspath(os.path.curdir))
        repath = os.path.relpath( arg, os.path.abspath(os.path.curdir))
        repath = repath.replace( '\\','.')[:-3]
        importlib.import_module(repath)

__doc__ = """python_macro runner

Usage:
  macro_runner <filename> | --file=<filename>

"""

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    filename = arguments['--file'] or arguments['<filename>']
    execute(filename)
