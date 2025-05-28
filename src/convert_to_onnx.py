import tf2onnx
import tensorflow as tf
from keras.models import load_model
import numpy as np
from models.PINN import PinnModel

#!ONNX conversion -> required numpy downgrade to <2

custom_objects = {"PinnModel": PinnModel}
model = load_model("models/pinn_model_c0_c1.keras", custom_objects=custom_objects, compile=False)
input_shape = (1, 4)
print("***Loading model...", f"Input shape = {input_shape}")

model_input_signature = [
    tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input'),
]
model.output_names = ['output'] #had to add this to prevent error. I think its because model hasnt been called yet
onnx_model, _ = tf2onnx.convert.from_keras(model,
    output_path='models/onnx/pinn_c0_c1.onnx',
    input_signature=model_input_signature,
)