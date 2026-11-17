"""Friendly message when a student has not completed a TODO yet (instead of a bare traceback)."""
import os
import sys
import traceback


def install():
    def hook(t, v, tb):
        if t is NotImplementedError:
            fr = traceback.extract_tb(tb)[-1]
            print(f"\nNOT IMPLEMENTED YET: complete the TODO in  {os.path.basename(fr.filename)}  (function {fr.name}(), line {fr.lineno}),\n"
                  f"save the file and run this command again.", file=sys.stderr)
            sys.exit(1)
        sys.__excepthook__(t, v, tb)
    sys.excepthook = hook
