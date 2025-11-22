import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

# model path
model_path = "data/models/yolov8n_640.onnx"

# model weights
model_weights_path = "data/models/weights/yolov8n.pt"
model_weights = YOLO(model_weights_path)

# inference session creation
provider = ["CPUExecutionProvider"]   # specify CPU as provider
session = ort.InferenceSession(model_path, providers=provider)

print(f"============ ONNX Model Info ============")

# attain input node info of model(name, shape, type)
input_info = session.get_inputs()[0]
print(f"input name: {input_info.name}")      # default: images 
print(f"input shape: {input_info.shape}")    # default: [1, 3, 640, 640]
print(f"input type: {input_info.type}")      # default: tensor(float)

# attain output node info of model(name, shape)
output_info = session.get_outputs()[0]
print(f"output name: {output_info.name}")    # default: output
print(f"output shape: {output_info.shape}")  # default: [1, 25200, 85]


# randomly generate an input tensor to simulate a 640*640 img
# shape: [1, 3, 640, 640], dtype: float32
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# inference execution
outputs = session.run(None, {input_info.name: dummy_input})

# print output tensor shape
output_tensor = outputs[0]
#print(f"output tensor shape: {output_tensor.shape}")
print(f"Output tensor shape: {output_tensor.shape}")    # why shape cannot be used?
print(f"Inference successful! Model is usable.")

print(f"============= Pytorch Model Info (Model Weights) =============")
print(f"model weights path: {model_weights_path}")
print(f"model weights: {model_weights}")