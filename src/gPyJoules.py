import os
import runpy
import argparse
import sys
parser = argparse.ArgumentParser(description='Monkey-Patch Keras to record energy measurements via nvidia-smi.')

# Optional argument
parser.add_argument("-w","--working-directory",type=str, default=None,
                    help='Change to this directory before running the module')

parser.add_argument("-d", "--data-directory", type=str, required=True
                ,help="Data directory.")    

parser.add_argument("-e", "--predict-energy", action='store_true'
                    ,help='Enable live energy prediction')       

parser.add_argument("module_name", type=str, help="Module to execute.")

parser.add_argument('other', nargs='*', help="Additional args to pass on to the executed module.")


args = parser.parse_args()

# set correct args for wrapped module
new_args = [sys.argv[0]]
new_args += args.other
sys.argv=new_args

# chdir if requested
if args.working_directory is not None:
    print(f"chdir to {args.working_directory}")
    os.chdir(args.working_directory)

os.environ["CUDA_VISIBLE_DEVICES"]="1" 
# patch keras and run module 
import patch_keras
patch_keras.patch(args.data_directory, args.predict_energy)
runpy.run_module(args.module_name)