# Profiling DenseNet121

In this post, I document a series of inference performance experiments I ran on DenseNet-121 using my very powerful SOTA NVIDIA GeForce MX130 (sick!).
The goal was simple:  

Understand what actually improves inference performance.
No torch.compile (my GPU does not support CUDA version). No TensorRT. Just raw PyTorch eager inference and profiling.  

## 1. Baseline: Plain Model + Warmup + Profiling

```
model = models.densenet121(pretrained=False)

# warmup
for steps in range(5):
    output = model(input)

# profiler start
for steps in range(10):
    output = model(input)
# profiler stop
```
From the above pseudocode, we get the following trace:
![Run 1](/images/run_1.png)  

From this image we get two observations, one is the total time and the second is that we see a lot of space between. This is because GPU utilization not continuous which means CPU waits for GPU to finish.

## 2. Enabling Non-Blocking Transfers

When loading images into the model for inference just make this one line change, instead of doing:  
`images = images.to(device)`  
do this  
`images = images.to(device, non_blocking=True)`

With this GPU activity becomes more continuous. CPU schedules copy --> GPU performs async copy --> CPU prepares next batch.  
This enabled CPU–GPU overlap, improving pipeline efficiency.
![Run 2](/images/run_2.png) 

## 3. Trying Smaller Data Types (BF16 / INT8)

Generally transforming into smaller dtypes help in freeing up memory and also in reduced latency (with some effect on performance), but in my case, this did not yield any positive effects and instead backfired on me. This is because as earlier mentioned I am using MX130 for my experiments and it does not have  
- Native BF16 compute
- Tensor Core acceleration
- INT8 acceleration

So what happens under the hood is:  
`input -> bf16 -> gpu does not support hence upcast to fp32 -> compute -> downcasts back to bf16`  

This is introducing overhead due to the conversions and because of this my execution time increases to 24s.

## 4. Switching to Channels-Last Memory Format

The general rule for PyTorch memory format propagation is to preserve the input tensor’s memory format. Which means a Channels First input will generate a Channels First output and a Channels Last input will generate a Channels Last output.  
In PyTorch, a tensor is described in the NCHW(N: batch size, C: Channel, H&W: Image dimensions) memory format. No matter what the physical order is, tensor shape and stride will always be depicted in the order of NCHW.  
![channels flow](https://pytorch.org/wp-content/uploads/2024/11/accelerating-pytorch-vision-models-with-channels-last-on-cpu-2.png)

PyTorch uses oneDNN for optimized convolution and oneDNN does NOT compute efficiently in plain NCHW. This is because as seen from above diagram, there is an extra overhead due to conversions.

The change to make is quite simple: 

```
model = model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)
```

Result:
Execution time is halved to 8s.

![Run 3](/images/run_3.png)


More to read on Channels last here: https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
