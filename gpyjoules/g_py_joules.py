import argparse
import os
import runpy
import sys
from enum import Enum
from multiprocessing import Event, Queue

from gpyjoules import data_collection
from gpyjoules import patch_keras


def parse_args() -> "Namespace":
    """Parse command line arguments using argparse.

    Returns:
        the parsed arguments

    """
    parser = argparse.ArgumentParser(description='Monkey-Patch Keras to record energy measurements via nvidia-smi.')

    # Optional argument
    parser.add_argument("-w", "--working-directory", type=str, default=None,
                        help='Change to this directory before running the module')

    parser.add_argument("-d", "--data-directory", type=str, required=True
                        , help="Data directory.")

    parser.add_argument("-e", "--predict-energy", action='store_true'
                        , help='Enable live energy prediction')

    parser.add_argument("-pl", "--power-limit", type=int, default=None
                        , help='Set power-limit before starting the training.')

    parser.add_argument("-v", "--visible-devices", type=int, default=None
                        , help='Set CUDA_VISIBLE_DEVICES environment variable.')

    parser.add_argument("module_name", type=str, help="Module to execute.")

    parser.add_argument('other', nargs='*', help="Additional args to pass on to the executed module.")
    return parser.parse_args()


class TfLogLevel(Enum):
    """This enum represents the different logging
    levels (verbosity) of tensorflow.
    """

    all = "0"
    """all messages are logged (default behavior)"""
    no_info = "1"
    """INFO messages are not printed"""
    no_warning = "2"
    """INFO and WARNING messages are not printed"""
    no_error = "3"
    """INFO, WARNING, and ERROR messages are not printed"""


def main() -> None:
    """Main function of this module.

    1. Set correct arguments for wrapped module.
    2. Set ``CUDA_VISIBLE_DEVICES`` environment variable
    3. Set tensorflow log level
    4. Monkey patch keras
    5. Chdir if needed
    6. Load and run the benchmark module
    """
    args = parse_args()

    # set correct args for wrapped module
    new_args = [sys.argv[0]]
    new_args += args.other
    sys.argv = new_args

    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)

    tf_log_level = TfLogLevel.no_warning.value
    if tf_log_level:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level

    # if args.power_limit is not None:
    #     SMIWrapper.set_power_limit(args.power_limit)

    data_event = Event()
    data_queue = Queue()

    data_collection.start_collecting(data_root=args.data_directory,
                                     visible_devices=args.visible_devices,
                                     data_event=data_event, data_queue=data_queue)

    # patch keras
    patch_keras.patch(args.data_directory, args.predict_energy,
                      args.visible_devices, data_event, data_queue)

    # chdir if requested
    if args.working_directory is not None:
        print(f"chdir to {args.working_directory}")
        os.chdir(args.working_directory)

    # run module
    runpy.run_module(args.module_name, run_name="__main__")


if __name__ == '__main__':
    main()
