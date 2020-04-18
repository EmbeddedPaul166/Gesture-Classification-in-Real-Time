from tensorflow.keras.models import load_model
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import numpy as np

output_saved_model_dir = "trt_cnn_model.pb"
input_saved_model_dir = "tf_model.pb"

num_runs = 50

def my_input_fn():
    for i in range(num_runs):
        inp = np.random.normal(size=(150, 150, 1)).astype(np.float32)
        yield inp

model = load_model("cnn_model.h5")
model.save(input_saved_model_dir, save_format="tf")

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(precision_mode=trt.TrtPrecisionMode.FP16)
conversion_params = conversion_params._replace(max_workspace_size_bytes=(6000000000))
conversion_params = conversion_params._replace(maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(input_saved_model_dir = input_saved_model_dir, conversion_params = conversion_params)
converter.convert()
#converter.save(output_saved_model_dir)
converter.build(input_fn=my_input_fn)
converter.save(output_saved_model_dir)
