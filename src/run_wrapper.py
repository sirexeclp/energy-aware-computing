import os
import runpy
import argparse
import sys
parser = argparse.ArgumentParser(description='Monkey-Patch Keras to record energy measurements via nvidia-smi.')

# Optional argument
parser.add_argument("-d","--directory",type=str, default=None,
                    help='Change to this directory before running the module')

parser.add_argument("module_name", type=str, help="Module to execute.")

parser.add_argument('other', nargs='*', help="Additional args to pass on to the executed module.")


args = parser.parse_args()

# set correct args for wrapped module
new_args = [sys.argv[0]]
new_args += args.other
sys.argv=new_args

# chdir if requested
if args.directory is not None:
    print(f"chdir to {args.directory}")
    os.chdir(args.directory)

# patch keras and run module 
import monkey_patch
runpy.run_module(args.module_name)