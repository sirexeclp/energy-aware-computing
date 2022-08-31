"""
This module provides classes to collect power and other data from gpus in
several ways.
"""
import abc
import atexit
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import pandas as pd

import pandas
from pynvml3 import Device, NVMLLib, SamplingType, NVMLErrorNotFound, ClockType, ClockId, TemperatureSensors, \
    PcieUtilCounter, NVMLErrorNotSupported
import pynpoint

SAMPLES_INTERVAL = 1.5

EXTERNAL_INTERVAL = 0.02
SLOW_INTERVAL = 0.15


class ProcessTimer(abc.ABC):
    """Abstract base class for a process based timer.
    It can be used to run a function in its own subprocess
    at a given interval.

    Note:
        This is a very naive implementation.
        Beware of race conditions.
    """

    def __init__(self, interval: float, args: Tuple = None, kwargs: Dict = None):
        """Create a new ProcessTimer object.
        Except it does not, because this is an abstract class, muhahahaaha.

        Args:
            interval: the interval (time) in seconds between ticks
            args: a collection of positional arguments,
                which will be passed to the ``_on_tick`` method for each tick
            kwargs: a dictionary of keyword arguments, which will be passed
                to the ``_on_tick`` method on each tick
        """
        self.interval = interval
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self._stop = multiprocessing.Event()
        self.last_time = None
        self.process = multiprocessing.Process(target=self._run)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> None:
        """This method is called from outside the process to start the timer.
        """

        if not self.process.is_alive():
            self._stop.clear()
            self.process.start()

    def stop(self, join: bool = False) -> None:
        """This method is called from outside the process to stop the timer
        before the next tick.
        The _on_stop will be called in the subprocess to allow for cleanup,
         before exiting the process.

        Args:
             join: if set to True, wait for the subprocess to join

        """
        self._stop.set()
        if join:
            self.process.join()

    @abc.abstractmethod
    def _on_start(self) -> None:
        """This function is executed in the context of the subprocess,
        when the timer is started.
        """
        pass

    @abc.abstractmethod
    def _on_stop(self) -> None:
        """This function is executed in the context of the subprocess,
        when the timer is stopped.
        """
        pass

    @abc.abstractmethod
    def _on_tick(self, args: Tuple, kwargs: Dict) -> None:
        """This function is executed inside the context of the subprocess,
        when the timer tics, ie. it is called repeatedly
        after the specified interval until the timer is stopped.

        Args:
            args: positional arguments
            kwargs: keyword arguments
        """
        pass

    def _run(self) -> None:
        """This is the main loop run inside the subprocess.
        This method should not be overridden by subclasses.
        Instead implement ``_on_start``, ``_on_stop`` and ``_on_tick``.

        Note:
            We use open loop linear control (without any filtering what so ever)
            which means we assumed that all operations take zero time except for
            _on_tick and that time.sleep delays the execution by exactly
            sleep_time seconds.

        """
        self._on_start()
        while not self._stop.is_set():
            pre_tick = time.time()
            self._on_tick(self.args, self.kwargs)
            post_tick = time.time()
            sleep_time = self.interval - (post_tick - pre_tick)
            if sleep_time < 0:
                # print(f"WARNING: can't keep up {(post_tick - pre_tick)}s")
                pass
            else:
                time.sleep(sleep_time)
        self._on_stop()


class Collector(ProcessTimer):
    """This is the base class for a ``ProcessTimer`` based data collector.
    """

    def __init__(self, device_id: Optional[int], interval: float, path: Union[str, Path], args=None, kwargs=None):
        """Constructor for the abstract Collector class.
        Args:
            device_id: the index of a GPU device
            interval: the interval (time) in seconds between collecting data
            path: the parent directory to store data in
            args: a collection of positional arguments, which will be passed
                to the ``_on_tick`` method of the timer for each tick
            kwargs: a dictionary of keyword arguments, which will be passed
                to the ``_on_tick`` method of the timer on each tick
        """
        super(Collector, self).__init__(interval, args, kwargs)
        self.data = []
        self.device_id = device_id
        self.lib = None
        self.device: Optional[Device] = None
        self.path = Path(path)

    def _on_start(self) -> None:
        """Instantiate NVMLLib and device on startup.

        Note:
            This will be executed inside the subprocess, when the timer starts.
        """
        self.lib = NVMLLib()
        self.lib.open()
        self.device = self.lib.device.from_index(self.device_id)

    def _on_stop(self) -> None:
        """Teardown NVMLLib and device on timer end and call ``_save`` to save
        the collected data to disk.

        Note:
            This will be executed inside the subprocess, when the timer stops.
        """
        self.device = None
        self.lib.close()
        self._save()

    @abc.abstractmethod
    def _get_save_path(self) -> Path:
        """Implement this to return a filename to which the data will be written.
        """
        pass

    @abc.abstractmethod
    def test(self, device: Device) -> bool:
        """Implement this method to test weather the collection is supported
        on this device or not.
        This method should return True, if the test was successfully, and
        the collector can be safely used or False, if the test did
        not succeed and the collector would not work on the given platform.

        Args:
            device: a pynvml3 GPU-Device object to use for testing

        Returns:
            True if the desired metric is supported, False otherwise

        """
        pass

    def _save(self) -> None:
        """Converts the collected data to a pandas data-frame
        and saves it as a csv file at the path returned from
        ``_get_save_path``.

        Note:
            This method is called in _on_stop and will be executed in
            the timer subprocess.

        """
        path = self._get_save_path()
        df = pd.DataFrame(self.data)
        df.to_csv(path)

    def get_len(self) -> int:
        """Returns the length of the internal data array.

        Returns:
            the number of datapoints

        """
        return len(self.data)


class SampleCollector(Collector):
    """This implements a collector using the ``get_sample`` (``nvmlDeviceGetSamples``) method of the
    gpu Device object.
    """

    def __init__(self, sample_type: SamplingType, device_id: int, interval: float, path: Union[str, Path]):
        """Creates a new sample collector for the given sample type.

        Args:
            sample_type: a nvml SamplingType to collect
            device_id: the index of the gpu device to collect from
            interval: the interval at which to query the samples
                Note: This is not the interval of the individual samples,
                    just the interval at which to get them from the buffer.
            path: the parent directory in which the data will be saved
        """
        super(SampleCollector, self).__init__(device_id, interval, path)
        self.sample_type = sample_type
        self.last_sample_time = 0

    def _on_tick(self, args, kwargs) -> None:
        """On each timer tick: try to get samples from the gpu.
        If there are any new samples, they are saved in memory.
        The timestamp of the last sample we got is saved and used
        for the next request.

        Args:
            args: no positional arguments are used
            kwargs: no keyword arguments are used

        """
        samples = self.device.try_get_samples(self.sample_type, self.last_sample_time)
        if samples:
            self.data.extend(samples)
            self.last_sample_time = samples[-1].timestamp

    def _get_save_path(self) -> Path:
        """Return the filename based on the sample type.

        Returns:
            the filename, to save data

        """
        return self.path / (self.sample_type.get_filename() + ".csv")

    def test(self, device: Device) -> bool:
        """Test if the ``get_samples`` is supported and does not throw.

        ``get_samples`` is called twice, because the second call will fail
        on some type of samples (which seem to be event based), if no new sample
        was produced since the last call.

        Args:
            device: the GPU device to test

        Returns:
            True if get_samples works for this sample_type, False if not

        """

        try:
            # first call sometimes works but then the second will fail; idk why.
            self.device = device
            self.device.get_samples(self.sample_type, self.last_sample_time)
            time.sleep(self.interval)
            self.device.get_samples(self.sample_type, self.last_sample_time)
            print(f"{self.sample_type}: OK")
            self.device = None
            return True
        except NVMLErrorNotFound:
            print(f"{self.sample_type} not supported on this device!")
            self.device = None
            return False


class SlowCollector(Collector):
    """
    This class implements a collector using the slower
    query methods, which return only
    one value at a time (e.g. ``device.get_power_usage()``).
    """

    def __init__(self, device_id: int, interval: float, path: str, args: Tuple[Event, Queue] = None, kwargs=None):
        """Create a new SlowCollector for the given gpu device.

        Args:
            device_id: the index of the gpu device to collect from
            interval: the interval at which to query the samples
            path: the parent directory in which the data will be saved
            args: a tuple of data_event, data_queue
            kwargs: no keyword arguments are used
        """
        super(SlowCollector, self).__init__(device_id, interval, path, args, kwargs)
        self.data_functions = {
            "timestamp": lambda: str(datetime.now())
            , "util": lambda: self.device.get_utilization_rates()
            , "clock-mem": lambda: self.device.get_clock(ClockType.MEM, ClockId.CURRENT)
            , "clock-gpu": lambda: self.device.get_clock(ClockType.SM, ClockId.CURRENT)
            , "app-clock-mem": lambda: self.device.get_applications_clock(ClockType.MEM)
            , "app-clock-gpu": lambda: self.device.get_applications_clock(ClockType.SM)
            , "enforced-power-limit": lambda: self.device.get_enforced_power_limit()
            , "total-energy": lambda: self.device.get_total_energy_consumption()
            # int representation to save on storage size
            , "power-state": lambda: self.device.get_power_state().value
            , "power": lambda: self.device.get_power_usage()
            , "tmp": lambda: self.device.get_temperature(TemperatureSensors.TEMPERATURE_GPU)
            , "pci-tx": lambda: self.device.get_pcie_throughput(PcieUtilCounter.TX_BYTES)
            , "pci-rx": lambda: self.device.get_pcie_throughput(PcieUtilCounter.RX_BYTES)
        }

    def _get_save_path(self) -> Path:
        """Get the filename to save the
        data collected with this slow collector.

        Returns:
            a filename

        """
        return self.path / "gpu-power.csv"

    def _on_tick(self, args: Tuple[Event, Queue], kwargs) -> None:
        """On each timer tick: Call all methods defined in data_functions,
        to get a set of measurements and a timestamp.
        Split utilization into gpu and memory.
        Send the data over a queue to an other process (KerasCallback),
        if it is requested via the event.

        Args:
            args: a tuple of data_event, data_queue
                the event should be used to request data,
                which will then be sent via the queue
            kwargs: not used
        """
        data_event, data_queue = args
        data = {key: function() for key, function in self.data_functions.items()}
        if "util" in data:
            data["util-gpu"] = data["util"].gpu
            data["util-mem"] = data["util"].memory
            del data["util"]

        self.data.append(data)

        if data_event.is_set():
            data_queue.put(self.data)
            data_event.clear()

    @staticmethod
    def _is_supported(key: str, function: callable) -> bool:
        """Test if the given function raises an NVMLErrorNotSupported exception.
        Return True if no exception was raised, print a message
        with the provided key and return False otherwise.
        Args:
            key: a key or name to identify the function
            function: a function to check

        Returns:

        """
        try:
            function()
            return True
        except NVMLErrorNotSupported:
            print(f"W: {key} is not supported on this system!")
            return False

    def test(self, device: Device) -> bool:
        """Filter all functions in self.data_functions with the ``_is_supported``
        function.
        After this self.data_functions should only contain functions,
        which are supported by the hardware.

        Args:
            device: a pynvml3 Device object

        Returns:
            True if any function is supported by the hardware

        """
        self.device = device
        self.data_functions = {key: value for key, value in
                               self.data_functions.items() if self._is_supported(key, value)}
        self.device = None
        return bool(self.data_functions)


class ExternalCollector(Collector):
    """
    This implements an external power measurement collector using the mcp device.
    """

    def __init__(self, interval, path):
        """Create a new ExternalCollector object.

        Args:
            interval: the interval at which to query the samples
            path: the parent directory in which the data will be saved
        """
        super().__init__(None, interval=interval, path=path)
        self.serial_devices = [
            "/dev/ttyACM0",
            "/dev/ttyACM1"
        ]
        self.handles = []

    def test(self, device: Optional[Device] = None) -> bool:
        """Test if the MCP devices can be opened.

        Args:
            device: not used

        Returns:
            True if all MCP devices could be initialized, False otherwise

        """
        try:
            for dev in self.serial_devices:
                self.handles.append(pynpoint.MCP(dev))
            return True
        except Exception as e:
            print("Initializing external measurements failed!")
            # print(e.message, type(e))
            return False

    def _on_start(self):
        """No setup needed."""
        pass

    def _on_stop(self):
        """Just call ``_save`` to save the collected data to disk.

        Note:
            This will be executed inside the subprocess, when the timer stops.
        """
        self._save()

    def _on_tick(self, args=None, kwargs=None) -> None:
        """On each timer tick: Get power measurements (from both channels) of
        all registered MCP handles and add a timestamp.

        Args:
            args: not used
            kwargs: not used
        """
        tmp = {"timestamp": str(datetime.now())}
        for index, h in enumerate(self.handles):
            c0, c1 = h.get_power()
            tmp[f"d{index}c0"] = c0
            tmp[f"d{index}c1"] = c1
        self.data.append(tmp)

    def _get_save_path(self) -> Path:
        """Get the filename to save the
        data collected with this external collector.

        Returns:
            a filename

        """
        return self.path / "power-external.csv"


class CollectorManager:
    """
    A class to manage multiple collector implementations.
    Collectors can be registered using the add method or the
    add_by_sampling_type method.
    All registered collectors can be started and stopped
    with the start and stop method.
    """

    def __init__(self, data_path: Union[str, Path], device_id: int):
        """Creates a new CollectorManager instance.
        Args:
            data_path: the parent path were all collectors will save data to
            device_id: the index of the gpu device to collect from
        """
        self.collectors: List[Collector] = []
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id

    def add(self, collector: Union[Collector, List[Collector]]) -> None:
        """Register a new collector or list of collectors.
        Each collector is tested using the Collector.test method and will
        only be registered if it passed the test.

        Args:
            collector: a single collector or list of collectors to register
        """
        if not isinstance(collector, list):
            collector = [collector]

        with NVMLLib() as lib:
            device = lib.device.from_index(self.device_id)
            for col in collector:
                if col.test(device):
                    self.collectors.append(col)

    def add_by_sampling_type(self,
                             sampling_type: Union[SamplingType,
                                                  List[SamplingType]]
                             , interval):
        """Create and register a new SampleCollector for each provided sampling
        type.
        Args:
            sampling_type: a sampling type, for which to register a collector
            interval: the interval at which the created collectors should be queried
        """
        if isinstance(sampling_type, SamplingType):
            sampling_type = [sampling_type]

        collectors = [SampleCollector(st, self.device_id, interval, self.data_path) for st in sampling_type]
        self.add(collectors)

    def start(self) -> None:
        """Start all collectors in the order they were registered."""
        for col in self.collectors:
            col.start()

    def stop(self):
        """Stop all collectors in the order they were registered."""
        for col in self.collectors:
            col.stop()


def start_collecting(data_root: str, visible_devices: int, data_event: Event, data_queue: Queue) -> None:
    """This is the entrypoint of this module.
    This function sets up a few collectors and a collector manager,
    then starts collecting.
    An @atexit.register callback is registered to stop the collector manager,
    when the python interpreter exits.

    Args:
        data_root: the parent directory in which to store all collected data
        visible_devices: the index of the gpu device to collect from
        data_event: event which will be used
            by the keras callback to request data
        data_queue: queue which will be used to send data to the keras callback
            upon request
    """
    sampling_types = [
        SamplingType.GPU_UTILIZATION_SAMPLES,
        SamplingType.MEMORY_UTILIZATION_SAMPLES,
        # SamplingType.MEMORY_CLK_SAMPLES,
        # SamplingType.PROCESSOR_CLK_SAMPLES,
        SamplingType.TOTAL_POWER_SAMPLES
    ]

    sampling_manager = CollectorManager(data_root, visible_devices)
    sampling_manager.add(SlowCollector(visible_devices,
                                       interval=SLOW_INTERVAL,
                                       path=data_root,
                                       args=(data_event, data_queue)))

    external_collector = ExternalCollector(interval=EXTERNAL_INTERVAL, path=data_root)
    sampling_manager.add(external_collector)

    sampling_manager.add_by_sampling_type(sampling_types, interval=SAMPLES_INTERVAL)

    @atexit.register
    def on_exit():
        sampling_manager.stop()

    sampling_manager.start()


def measure_baseline(data_root: str, visible_devices: int, length: int) -> None:
    """Measure baseline metrics on an idle system.
    Start measurements as usual then sleep for ``length`` seconds
    to get a baseline.

    Args:
        data_root: the parent directory in which to store all collected data
        visible_devices: the index of the gpu device to collect from
        length: the duration of the baseline measurement in seconds
    """
    start_collecting(data_root, visible_devices, Event(), Queue())
    time.sleep(length)
