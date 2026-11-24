"""Run this fxbX installation with a user-selected SDK interpreter."""
import importlib.util
from pathlib import Path
import runpy
import sys


def worker_command(interpreter, module, *arguments):
    if module not in ('fxbx', 'fxbex.worker'):
        raise ValueError('Unsupported fxbX worker module')
    return [interpreter, '-u', str(Path(__file__).resolve()), module, *map(str, arguments)]


def main():
    module, *arguments = sys.argv[1:]
    if module not in ('fxbx', 'fxbex.worker'):
        raise ValueError('Unsupported fxbX worker module')
    root = Path(__file__).resolve().parent.parent
    # Load only our two namespaces; never expose another Python's site-packages.
    for name in ('fxbex', 'fxbx'):
        location = root / name
        spec = importlib.util.spec_from_file_location(
            name, location / '__init__.py', submodule_search_locations=[str(location)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
