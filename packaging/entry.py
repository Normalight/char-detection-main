"""Frozen entry: dispatch multiprocessing before importing Qt or application code."""
import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if '--smoke-test' in sys.argv:
        from smoke_test import smoke_test
        smoke_test()
    else:
        from app import main
        raise SystemExit(main())
