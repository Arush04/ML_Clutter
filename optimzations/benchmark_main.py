import torch
import torch.nn as nn
import time
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from utils import FashionMNISTDataLoader 

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device >>>> {device}")

# Dataset
data_path = "/home/arush/Arush/ML_Clutter/optimzations/data"
# data = FashionMNISTDataLoader(data_path, batch_size=32, num_workers=2)
# data_loader = data.get_loader()

# Model
def create_model():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model = model.to(device)
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    return model


# Benchmark Function
def run_inference_benchmark(model, dataloader):
    total_images = 0
    total_batches = 0

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            images = images.to(memory_format=torch.channels_last)
            outputs = model(images)

            total_images += images.size(0)
            total_batches += 1

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    throughput = total_images / total_time
    latency_per_batch = total_time / total_batches

    return total_time, throughput, latency_per_batch


# Main Driver 
if __name__ == "__main__":

    # batch_sizes = [1, 8, 16, 32]
    batch_sizes = [32]
    data = FashionMNISTDataLoader(data_path, batch_size=32, num_workers=2)
    data_loader = data.get_loader()

    model = create_model()

    for bs in batch_sizes:
        # Warmup
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                images = images.to(device)
                images = images.to(memory_format=torch.channels_last)
                _ = model(images)
                if i == 5:
                    break

        total_time, throughput, latency = run_inference_benchmark(model, data_loader)

        print(f"\nBatch Size: {bs}")
        print(f"Total Time: {total_time:.3f} sec")
        print(f"Throughput: {throughput:.2f} images/sec")
        print(f"Latency per batch: {latency:.4f} sec")
