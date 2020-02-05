import warnings
#with warnings.catch_warnings():
warnings.filterwarnings('ignore')
import tensorflow
import tensorflow.keras as keras
from datetime import datetime
from pathlib import Path
#from run_benchmark import GPU, SMIWrapper
import atexit
import pandas as pd

timestamp_log = []

def log_event(name, index=""):
    print(f"[Info] {name} {index}")
    tmp = {
        "timestamp": str(datetime.now())
        ,"event": name
        ,"data": index
    }
    timestamp_log.append(tmp)
    #timestamp_log.write(f"{str(datetime.now())},{name},{index}\n")

def set_log_path(new_path):
    global timestamp_log_path
    data_root = Path(new_path)
    timestamp_log_path = data_root / "timestamps.csv"
    data_root.mkdir(parents=True,exist_ok=True)

def patch(data_root):
    global timestamp_log_path
    set_log_path(data_root)
    #timestamp_log = open(timestamp_log_path , "w", buffering=1)
    #timestamp_log.write(f"timestamp,event,data\n")
    
    @atexit.register
    def goodbye():
        #print("You are now leaving the Python sector.")
        log_event("experiment_end")
        df = pd.DataFrame(timestamp_log, columns=timestamp_log[0].keys())
        df.to_csv(timestamp_log_path, index=False)

    logger = keras.callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, logs: log_event("epoch_begin", epoch)
                ,on_epoch_end=lambda epoch, logs: log_event("epoch_end", epoch)
                ,on_train_begin=lambda logs: log_event("train_begin")
                ,on_train_end=lambda logs: log_event("train_end")
                #,on_batch_begin=lambda batch, logs: log_event("batch_begin", batch)
                )
    if not hasattr(tensorflow.keras.models.Sequential, "_fit"):
        def my_fit(*args, **kwargs):
            callbacks = kwargs.get("callbacks",list())
            callbacks.append(logger)
            kwargs["callbacks"] = callbacks
            return tensorflow.keras.models.Sequential._fit(*args, **kwargs)

        tensorflow.keras.models.Sequential._fit = tensorflow.keras.models.Sequential.fit
        tensorflow.keras.models.Sequential.fit = my_fit
    else:
        print("Keras was already patched!")

    log_event("experiment_begin")