
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")



import torch
print(torch.cuda.is_available())


from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to TensorRT format
model.export(format="engine", dynamic=False, imgsz=640, batch=3)
"""

import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())

import tensorrt_lean as trt
print(trt.__version__)
assert trt.Runtime(trt.Logger())


import tensorrt_dispatch as trt
print(trt.__version__)
assert trt.Runtime(trt.Logger())
"""