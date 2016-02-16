
# -*- coding: utf-8 -*-
from distutils.core import setup
import os
def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
        )

def find_packages(path, base="" ):
    """ Find all packages in path """
    packages = {}
    for item in os.listdir(path):
        dir = os.path.join(path, item)
        if is_package( dir ):
            if base:
                module_name = "%(base)s.%(item)s" % vars()
            else:
                module_name = item
            packages[module_name] = dir
            packages.update(find_packages(dir, module_name))
    return packages


setup(
    name='python_macro',
    version=str(__import__("python_macro").__version__),
    packages=['astutil'],
    scripts=['script/macro_runner.py'],
    include_package_data=False,
    install_requires=["nose","cython","typing","utils","pygments"],
)
