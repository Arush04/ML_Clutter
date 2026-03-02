import torch
import torch.nn as nn
from torchvision import models
from utils import FashionMNISTDataLoader 
import torch.profiler as profiler
from torch.ao.quantization import fuse_modules
torch.backends.cudnn.benchmark = True

# def fuse_densenet(model):
    # for module_name, module in model.named_children():
        # if hasattr(module, "features"):
            # for name, child in module.features.named_children():
                # if isinstance(child, torch.nn.Sequential):
                    # for block_name, block in child.named_children():
                        # try:
                            # fuse_modules(block, ["conv1", "norm1"], inplace=True)
                            # fuse_modules(block, ["conv2", "norm2"], inplace=True)
                        # except:
                            # pass
    # return model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader
data_path = "/home/arush/Arush/ML_Clutter/optimzations/data"
data = FashionMNISTDataLoader(data_path, batch_size=32, num_workers=2)
data_loader = data.get_loader()

# Model
model = models.densenet121(pretrained=False)

# Modify classifier for 10 classes
model.classifier = nn.Linear(model.classifier.in_features, 10)

# move to device
# model = model.to(device)
model.eval()
# model = fuse_densenet(model)

model.to(device)
model = model.to(memory_format=torch.channels_last)

# Warmup
print("Running warmup...")
with torch.no_grad():
    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)
        images = images.to(memory_format=torch.channels_last)
        _ = model(images)
        if i == 5:  # 5 warmup iterations
            break

print("Warmup done.")

# Profiling
print("Starting profiler...")

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            images = images.to(memory_format=torch.channels_last)
            outputs = model(images)

            if i == 10:   # profile 10 steps
                break
print("Profiling Finished...")

#export trace for Chrome tracing
tracer_folder = "/home/arush/Arush/ML_Clutter/optimzations/tracer_results"
prof.export_chrome_trace(f"{tracer_folder}/densenet_fashionmnist_trace4.1.json")
