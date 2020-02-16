import os
import runpy
import argparse
import sys
from smi_wrapper import SMIWrapper



parser = argparse.ArgumentParser(description='Monkey-Patch Keras to record energy measurements via nvidia-smi.')

# Optional argument
parser.add_argument("-w","--working-directory",type=str, default=None,
                    help='Change to this directory before running the module')

parser.add_argument("-d", "--data-directory", type=str, required=True
                ,help="Data directory.")    

parser.add_argument("-e", "--predict-energy", action='store_true'
                    ,help='Enable live energy prediction')     

parser.add_argument("-pl", "--power-limit", type=int, default=None
                    ,help='Set power-limit before starting the training.')  

parser.add_argument("-v", "--visible-devices", type=str, default=None
                    ,help='Set CUDA_VISIBLE_DEVICES environment variable.')       

parser.add_argument("module_name", type=str, help="Module to execute.")

parser.add_argument('other', nargs='*', help="Additional args to pass on to the executed module.")


args = parser.parse_args()

# set correct args for wrapped module
new_args = [sys.argv[0]]
new_args += args.other
sys.argv=new_args

if args.visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices 

if args.power_limit is not None:
    SMIWrapper.set_power_limit(args.power_limit)


# patch keras
import patch_keras
patch_keras.patch(args.data_directory, args.predict_energy)

# chdir if requested
if args.working_directory is not None:
    print(f"chdir to {args.working_directory}")
    os.chdir(args.working_directory)

#run module 
runpy.run_module(args.module_name, run_name="__main__")