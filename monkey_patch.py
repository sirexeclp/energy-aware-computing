import tensorflow
import tensorflow.keras as keras
from datetime import datetime
from pathlib import Path
from run_benchmark import GPU, SMIWrapper
import atexit


def log_event(name, index=""):
    #print(f"[Info] {name} {index}")
    timestamp_log.write(f"{str(datetime.now())},{name},{index}\n")


@atexit.register
def goodbye():
    #print("You are now leaving the Python sector.")
    log_event("experiment_end")


def my_fit(*args, **kwargs):
    callbacks = kwargs.get("callbacks",list())
    callbacks.append(logger)
    kwargs["callbacks"] = callbacks
    return tensorflow.keras.models.Sequential._fit(*args, **kwargs)


data_root = Path("/tmp")
timestamp_log_path = data_root / "timestamps.csv"
timestamp_log = open(timestamp_log_path , "w", buffering=1)
timestamp_log.write(f"timestamp,event,data\n")

logger = keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: log_event("epoch_begin", epoch)
            ,on_epoch_end=lambda epoch, logs: log_event("epoch_end", epoch)
            ,on_train_begin=lambda logs: log_event("train_begin")
            ,on_train_end=lambda logs: log_event("train_end")
            #,on_batch_begin=lambda batch, logs: log_event("batch_begin", batch)
            )

tensorflow.keras.models.Sequential._fit = tensorflow.keras.models.Sequential.fit
tensorflow.keras.models.Sequential.fit = my_fit

log_event("experiment_begin")