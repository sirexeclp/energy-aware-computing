"""
This module can be used to monkey patch the fit methods in keras,
to log timestamps of epochs, batches, etc., without the need to
change the sourcecode of training script.
"""
import atexit
import time
import os
import warnings

# with warnings.catch_warnings():
from typing import Union, Callable
from multiprocessing import Event, Queue

from gpyjoules.util import *

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow
import tensorflow.keras as keras


def get_log_path(data_root: Union[str, Path]) -> Path:
    """Get the filename for the timestamp log.
    Creates the `data_root` if it does not yet exist.

    Args:
        data_root: the data root or parent directory, in which to store the log

    Returns:
        the path object for the timestamp log

    """
    data_root = Path(data_root)
    timestamp_log_path = data_root / "timestamps.csv"
    data_root.mkdir(parents=True, exist_ok=True)
    return timestamp_log_path


def get_tensorboard_path(data_root: Union[str, Path]) -> Path:
    """Get the path for tensorboard.

    Args:
        data_root: the data root or parent directory

    Returns:
        the path object for the tensorboard

    """
    data_root = Path(data_root)
    tensorboard_path = data_root / "tensorboard"
    return tensorboard_path


class TimestampLogger:
    """A simple in-memory/csv based logger for timestamp events.
    The resulting csv file will have the fields:

    - timestamp: the timestamp in iso format
    - event: the event name as string
    - data: the integer index of this event

    """

    def __init__(self, log_path: Union[str, Path]):
        """Creates a new TimestampLogger instance.

        Args:
            log_path: the filename in which to store the log
        """
        self.timestamp_log = []
        self.log_path = log_path
        self.columns = ["timestamp", "event", "data"]

    def log_event(self, name: str, index: int = 0) -> None:
        """Log the given event and its index.
        The timestamp will be generated automatically.

        Args:
            name: the name of the event to log
            index: optional index of the event

        """
        tmp = {"timestamp": str(datetime.now()), "event": name, "data": index}
        self.timestamp_log.append(tmp)

    def save(self):
        """Save the log as a csv file."""
        df = self.get_df()
        df.to_csv(self.log_path, index=False)

    def get_df(self) -> pd.DataFrame:
        """Create a pandas dataframe from the internal log data.

        Returns:
            the log as a dataframe

        """
        return pd.DataFrame(self.timestamp_log, columns=self.columns)


class EnergyCallback(keras.callbacks.Callback):
    """Custom Keras Callback to log events, for energy measurements."""

    def __init__(
        self,
        enable_energy_prediction: bool,
        num_epochs: int,
        timestamp_logger: TimestampLogger,
        visible_devices: int,
        data_event: Event,
        data_queue: Queue,
    ):
        """Creates a new EnergyCallback to be used with keras.

        Args:
            enable_energy_prediction: if set to True, the total energy
                needed for training will be predicted after each epoch
                starting after the second epoch
            num_epochs: the number of epochs to train for
            timestamp_logger: an instance of a ``TimestampLogger``
                to use for logging
            visible_devices: index of the GPU that will be used for training
            data_event: event to request data from collection subprocess
            data_queue: queue to receive data from collection subprocess
        """
        super(EnergyCallback, self).__init__()
        self.timestamp_log = timestamp_logger
        self.enable_energy_prediction = enable_energy_prediction
        self.num_epochs = num_epochs

        self.total_batch = 0
        self.visible_devices = visible_devices
        self.last_time = time.time()
        self.summary_writer = tensorflow.summary.create_file_writer(
            "/tmp/energy-board/energy"
        )
        self.data_event = data_event
        self.data_queue = data_queue

    # def on_train_batch_end(self, batch: int, logs=None) -> None:
    #     """This callback gets called at the end of each training batch and
    #     logs the end of a training batch.
    #
    #     Args:
    #         batch: index of this batch
    #         logs: no idea, what this does
    #     """
    #     pass
    #     ## self.log_event("batch_end", batch)
    #     # self.total_batch += 1
    #     # if (self.total_batch % 100) == 0:
    #     #     with NVMLLib() as lib:
    #     #         device = Device.from_index(self.visible_devices)
    #     #         device.set_power_management_limit(device.get_power_management_limit()-50_000)
    #     #         t = time() - self.last_time
    #     #         print(f"Testing PL: {device.get_power_management_limit()}W, Time: {t}")
    #     #         self.last_time = time()
    #
    # def on_train_batch_begin(self, batch: int, logs=None) -> None:
    #     """Callback: Log the begin of a training batch.
    #
    #     Args:
    #         batch: index of this batch
    #         logs: no idea, what this does
    #     """
    #     # self.log_event("batch_begin", batch)
    #     pass

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        """Callback: Log the begin of an epoch.

        Args:
            epoch: index of this epoch
            logs: no idea, what this does
        """
        self.log_event("epoch_begin", epoch)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """Callback: Log the end of an epoch.
        If energy prediction is enabled, and at least
        two epochs have passed, print and predict the energy now.

        Args:
            epoch: index of this batch
            logs: no idea, what this does
        """
        self.log_event("epoch_end", epoch)

        if epoch > 0 and self.enable_energy_prediction:
            self.predict_energy(epoch)

    def on_train_begin(self, logs=None) -> None:
        """Callback: Log the beginning of training.

        Args:
            logs: no idea, what this does
        """
        self.log_event("train_begin")
        # use this to see if monkeypatching was successfull.
        print("Timestamp Logger: Train Begin")

    def on_train_end(self, logs=None) -> None:
        """Callback: Log the end of training.

        Args:
            logs: no idea, what this does
        """
        self.log_event("train_end")

    def log_event(self, name: str, index: int = 0) -> None:
        """Log the given event and its index using the timestamp_log.
        This is basically just a delegation to ``self.timestamp_log.log_event``.
        Args:
            name: the name of the event
            index: the index of the event
        """
        self.timestamp_log.log_event(name=name, index=index)

    def get_gpu_data(self) -> pd.DataFrame:
        """Request and get power data from the collection subprocess.

        Returns:
            the power data

        """
        self.data_event.set()
        while self.data_queue.empty():
            time.sleep(0.1)
        gpu_data = self.data_queue.get()
        gpu_data = pd.DataFrame(gpu_data)
        return gpu_data

    def predict_energy(self, epoch: int) -> None:
        """Predict the energy consumption for the rest of training, based on
        the consumption up to this point.

        Args:
            epoch: the current epoch
        """
        gpu_data = self.get_gpu_data()
        timestamps = self.timestamp_log.get_df()
        power_data = PowerData(gpu_data, timestamps)
        pred = predict_energy_live(power_data, [0], self.num_epochs, epoch)
        actual = calculate_total_energy(power_data, [0])
        with self.summary_writer.as_default():
            tensorflow.summary.scalar("energy", data=actual, step=epoch)
            tensorflow.summary.scalar("energy-predicted", data=pred, step=epoch)
        self.summary_writer.flush()
        print(f"\nConsumed Energy: {actual / 1_000:.3f}/{pred / 1_000:.3f}kJ")


def patch(
    data_root: Union[str, Path],
    enable_energy: bool,
    visible_devices: int,
    data_event: Event,
    data_queue: Queue,
) -> None:
    """This is the entrypoint for the `patch_keras` module.
    This function instantiates timestamp logger and callback and
    monkey patches the functions:

    - tensorflow.keras.models.Model.fit
    - tensorflow.keras.models.Model.fit_generator

    Monkey patching this functions allows to run any tensorflow.keras based
    training script to be run and monitored (timestamps logged= without
    the need to modify the original script source code.

    Args:
        data_root: directory to store timestamp-log in
        enable_energy: if set to True, realtime energy predictions will be made
        visible_devices: index of the GPU that will be used for training
        data_event: event to request data from collection subprocess
        data_queue: queue to receive data from collection subprocess

    """
    timestamp_log_path = get_log_path(data_root)
    logger = TimestampLogger(timestamp_log_path)

    @atexit.register
    def on_exit():
        logger.log_event("experiment_end")
        logger.save()

    def get_patched_fit(original_function: Callable) -> Callable:
        """Create a patched version of the given function.

            1. create energy callback, with the number of epochs (only known when the patched function is called)
            2. add energy callback to the list of callbacks
            3. call end return the result of the original function

        Args:
            original_function: a fit function to patch

        Returns:
            the patched fit funtion

        """

        def patched_fit(*args, **kwargs):
            num_epochs = kwargs.get("epochs")
            energy_callback = EnergyCallback(
                enable_energy,
                num_epochs,
                logger,
                visible_devices,
                data_event,
                data_queue,
            )

            # profile the first 10 batches with tensorboard
            # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=get_tensorboard_path(data_root),
            #                                                    profile_batch=(1, 10))

            callbacks = [
                energy_callback,
                # tensorboard_callback
            ]

            kwargs.setdefault("callbacks", list()).extend(callbacks)
            if not enable_energy:
                kwargs["verbose"] = 2
            return original_function(*args, **kwargs)

        return patched_fit

    model = tensorflow.keras.models.Model
    model.fit = get_patched_fit(model.fit)
    model.fit_generator = get_patched_fit(model.fit_generator)

    logger.log_event("experiment_begin")
