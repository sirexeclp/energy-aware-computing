import warnings
#with warnings.catch_warnings():
warnings.filterwarnings('ignore')
import tensorflow
import tensorflow.keras as keras
from datetime import datetime
from pathlib import Path
from smi_wrapper import SMIWrapper
import atexit
import pandas as pd
from multiprocessing import  Process, Event, Queue
from time import sleep
from util import *


timestamp_log = []
stop_event = Event()
data_event = Event()
data_queue = Queue()


def log_event(name, index=""):
    #print(f"[Info] {name} {index}")
    tmp = {
        "timestamp": str(datetime.now())
        ,"event": name
        ,"data": index
    }
    timestamp_log.append(tmp)

    if name == "epoch_end" and index > 0 and enable_energy_prediction:
        data_event.set()
        while data_queue.empty():
            sleep(0.1)
        gpu_data = data_queue.get()
        gpu_data = pd.DataFrame(gpu_data)
        timestamps = pd.DataFrame(timestamp_log, columns=timestamp_log[0].keys())
        power_data = PowerData(gpu_data, timestamps)
        pred = predict_energy_live(power_data,[0],num_epochs, index)
        actual = calculate_total_energy(power_data, [0])
        print(f"\nConsumed Energy: {actual/1_000:.3f}/{pred/1_000:.3f}kJ")

            
    #timestamp_log.write(f"{str(datetime.now())},{name},{index}\n")

def set_log_path(new_path):
    global timestamp_log_path
    data_root = Path(new_path)
    timestamp_log_path = data_root / "timestamps.csv"
    data_root.mkdir(parents=True,exist_ok=True)


def collect_power_data(data_path, done, get_data, data):
    gpu_data = []
    
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    gpu_data_path = data_path / "gpu-power.csv"

    with SMIWrapper() as sw:
        print("start recording")
        while not done.is_set():
            sw.get_all_stats(gpu_data)
            if get_data.is_set():
                get_data.clear()
                data.put(gpu_data)
            #time.sleep(0.5-0.0752)
        
        df = pd.DataFrame(gpu_data)
        df.to_csv(gpu_data_path)



def patch(data_root, enable_energy):
    global timestamp_log_path, power_process, enable_energy_prediction
    enable_energy_prediction = enable_energy
    set_log_path(data_root)
    #timestamp_log = open(timestamp_log_path , "w", buffering=1)
    #timestamp_log.write(f"timestamp,event,data\n")
    
    power_process = Process(target=collect_power_data,args=[data_root, stop_event, data_event, data_queue])
    power_process.start()

    @atexit.register
    def goodbye():
        #print("You are now leaving the Python sector.")
        log_event("experiment_end")
        stop_event.set()
        df = pd.DataFrame(timestamp_log, columns=timestamp_log[0].keys())
        df.to_csv(timestamp_log_path, index=False)
        power_process.join()

    logger = keras.callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, logs: log_event("epoch_begin", epoch)
                ,on_epoch_end=lambda epoch, logs: log_event("epoch_end", epoch)
                ,on_train_begin=lambda logs: log_event("train_begin")
                ,on_train_end=lambda logs: log_event("train_end")
                #,on_batch_begin=lambda batch, logs: log_event("batch_begin", batch)
                )

    def get_patched_fit(original_function):
        def patched_fit(*args, **kwargs):
            global num_epochs
            num_epochs = kwargs.get("epochs")
            kwargs.setdefault("callbacks", list()).append(logger)
            if not enable_energy:
                kwargs["verbose"] = 2
            return original_function(*args, **kwargs)
        return patched_fit

    model = tensorflow.keras.models.Model
    model.fit = get_patched_fit(model.fit)
    model.fit_generator = get_patched_fit(model.fit_generator)

    log_event("experiment_begin")