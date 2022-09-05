"""This module prepares a single benchmark run.

It monkeypatches keras and starts data collection processes.
"""
import argparse
import os
import runpy
import sys
from enum import Enum
from multiprocessing import Event, Queue
from pathlib import Path

from gpyjoules import data_collection
from gpyjoules import patch_keras


def parse_args() -> "Namespace":
    """Parse command line arguments using argparse.

    Returns:
        the parsed arguments

    """
    parser = argparse.ArgumentParser(
        description="Monkey-Patch Keras to record energy measurements via nvidia-smi."
    )

    # Optional argument
    parser.add_argument(
        "-w",
        "--working-directory",
        type=str,
        default=None,
        help="Change to this directory before running the module",
    )

    parser.add_argument(
        "-d", "--data-directory", type=str, required=True, help="Data directory."
    )

    parser.add_argument(
        "-e",
        "--predict-energy",
        action="store_true",
        help="Enable live energy prediction",
    )

    parser.add_argument(
        "-pl",
        "--power-limit",
        type=int,
        default=None,
        help="Set power-limit before starting the training.",
    )

    parser.add_argument(
        "-v",
        "--visible-devices",
        type=int,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES environment variable.",
    )

    parser.add_argument(
        "-b",
        "--baseline",
        action="store_true",
        help="Record an idle, baseline measurement.",
    )

    parser.add_argument(
        "-bl",
        "--baseline-length",
        type=int,
        default=None,
        help="The length of the baseline measurement in seconds.",
    )

    parser.add_argument("module_name", type=str, help="Module to execute.")

    parser.add_argument(
        "other", nargs="*", help="Additional args to pass on to the executed module."
    )
    return parser.parse_args()


class TfLogLevel(Enum):
    """This enum represents the different logging
    levels (verbosity) of tensorflow.
    """

    ALL = "0"
    """all messages are logged (default behavior)"""
    NO_INFO = "1"
    """INFO messages are not printed"""
    NO_WARNING = "2"
    """INFO and WARNING messages are not printed"""
    NO_ERROR = "3"
    """INFO, WARNING, and ERROR messages are not printed"""


def get_baseline(args) -> None:
    """Collect baseline measurements."""
    data_collection.measure_baseline(
        data_root=args.data_directory,
        visible_devices=args.visible_devices,
        length=args.baseline_length,
    )


def main() -> None:
    """Main function of this module.

    1. Set correct arguments for wrapped module.
    2. Set ``CUDA_VISIBLE_DEVICES`` environment variable
    3. Set tensorflow log level
    4. Create event and cue for inter process communication
    5. Start data collection (pass event & queue)
    6. Monkey patch keras (pass event & queue)
    7. Chdir if needed
    8. Load and run the benchmark module
    """
    args = parse_args()

    if args.baseline:
        if args.baseline_length is None:
            print("Missing argument baseline-length!")
            return -1
        get_baseline(args)
        return 0

    # set correct args for wrapped module
    new_args = [sys.argv[0]]
    new_args += args.other
    sys.argv = new_args

    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_devices)

    tf_log_level = TfLogLevel.NO_WARNING.value
    if tf_log_level:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_log_level

    # if args.power_limit is not None:
    #     SMIWrapper.set_power_limit(args.power_limit)

    data_event = Event()
    data_queue = Queue()

    data_collection.start_collecting(
        data_root=args.data_directory,
        visible_devices=args.visible_devices,
        data_event=data_event,
        data_queue=data_queue,
    )

    # patch keras
    patch_keras.patch(
        args.data_directory,
        args.predict_energy,
        args.visible_devices,
        data_event,
        data_queue,
    )

    # chdir if requested
    if args.working_directory is not None:
        print(f"chdir to {Path(args.working_directory).absolute()}")
        os.chdir(args.working_directory)

    # run module
    runpy.run_module(args.module_name, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
