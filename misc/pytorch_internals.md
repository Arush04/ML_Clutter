# Optimizing DL models using First Principles: 
referenced from: https://horace.io/brrr_intro.html

We can approach the problem by dividing it in the following regimes:

1. Compute: Time spent on your GPU computing actual floating point operations (FLOPS)
2. Memory: Time spent transferring tensors within a GPU
3. Overhead: Everything else

Just like with training ML models, knowing what regime you're in allows you to narrow in on optimizations that matters. For example, if you're spending all of your time doing memory transfers (i.e. you are in an memory-bandwidth bound regime), then increasing the FLOPS of your GPU won't help. On the other hand, if you're spending all of your time performing big chonky matmuls (i.e. a compute-bound regime), then rewriting your model logic into C++ to reduce overhead won't help.

## Compute:

memory (storage) --> send data to GPU (memory bandwidth) --> compute

if memory-bandwidth is small then the device cant perform to its max limit since not enough data is reaching it for operations(FLOPS)

## Bandwidth:

Bandwidth costs are essentially the cost paid to move data from one place to another. This might be moving the data from CPU to GPU, from one node to another, or even from CUDA global memory to CUDA shared memory. This last one, in particular, is what we'll be focusing on here, and is typically referred to as "bandwidth cost" or "memory bandwidth cost".

This cost of moving stuff to and from our compute units is what's called the "memory bandwidth" cost. As an aside, your GPU's DRAM is what shows up in nvidia-smi, and is the primary quantity responsible for your lovely "CUDA Out of Memory' errors.

One thing to note is that every single time we perform a GPU kernel, we need to move our data from and back to our GPU's DRAM (i.e. our warehouse).

Now, imagine what happens when we perform an unary operation like torch.cos. We need to ship our data from our storage to the warehouse, then perform a tiny bit of computation for each piece of data, and then ship that storage back. Shipping things around is quite expensive. 
### As a result, nearly all of our time here is spent shipping data around, and not on the actual computation itself.

Since we're spending all of our time on memory-bandwidth, such an operation is called a memory-bound operation, and it means that we're not spending a lot of time on compute. To resolve this as dicussed above we can use operator fusion.

Solution to this regime can in using torch.compile, writing custom kernels (Triton)

Finally, operator fusion leads to some surprising consequences. For one, a fused x.cos().cos() will take nearly the exact same time as calling x.cos() by itself.
This fact leads to some interesting consequences for rematerialization/activation checkpointing. Essentially, doing extra recomputation might lead to less memory-bandwidth, and thus less runtime. Thus, we can lower both memory and runtime through rematerialization, which we leveraged to build a neat min-cut optimization pass in AOTAutograd.

Memory bound = bandwidth is utilised, compute is underutilized.
Compute bound = compute utilized (achieving peak FLOPS), bandwith is underutilized.


## Overhead:

Overhead is when your code is spending time doing anything that's not transferring tensors or computing things. For example, time spent in the Python interpreter? Overhead. Time spent in the PyTorch framework? Overhead. Time spent launching CUDA kernels (but not executing them)? Also... overhead.

Primary reason is that python is slow and gpus are fast, and on top of using python we make use of pytorch, which goes though a bunch dispatches(as seen above in torch.compile) before firing up the kernel, hence all work done till now was overhead.

So, how do you tell if you're in this regime? Well, since overhead generally doesn't scale with problem size (while compute and memory do), the easiest way to tell is to simply increase the size of your data. If that doesn't increase the runtime proportionally, you're overhead bound. For example, if you double your batch size but your runtime only increases by 10%, you're likely overhead bound.

The primary reason this overhead exists is due to all of the flexibility frameworks like PyTorch have. Essentially, a lot of time needs to be spent on "figuring out what to do".


## Summary:

| Performance Regime  |	Plausible Solutions
-----------------------------------------------------------------------------------
| Overhead-Bound	  | Tracing, Operator Fusion, don't use Python, a real JIT :^)
| Bandwidth-Bound	  | Operator Fusion
| Compute-Bound	Use   | Tensor Cores, give Nvidia more money

# TORCH COMPILE:
reference from: https://medium.com/@jiminlee-ai/how-torch-compile-actually-works-from-python-bytecode-to-fused-triton-kernels-1c78721c3331

This is what happens during a standard CPU GPU conversation

f(x) = return x*2 + 1

The CPU issues the following instructions to the GPU:
1. Fetch: Go to the warehouse (HBM/VRAM), get x, bring it to the workbench (Registers).
2. Compute: Multiply by 2.
3. Store: Put the result back in the warehouse. (Round trip #1).
4. Fetch: Go back to the warehouse, get that result.
5. Compute: Add 1.
6. Store: Put the final result back in the warehouse. (Round trip #2).

We are wasting massive amounts of time just moving data back and forth.
“Kernel Launch” is the technical term for the CPU telling the GPU to do something.

For very small operations, the time it takes the CPU to write the “work order” and send it to the GPU takes longer than the actual calculation itself.
The goal of torch.compile() is Operator Fusion. (Launching as minimum kernels as possible and doing as much as possible in minimum kernels)

### Python Code (Pytorch Frontend) --> TorchDynamo (FX Graph | Forward pass) --> ATen IR --> TorchInductor

## TorchDynamo:
Analyzes python code and generates FX Graph -- a representation of your code consisting purely of PyTorch operations.

for f(x) = return x*2 + 1
It looks something like this (conceptually):
x -> multiply -> add -> output

Python code is compiled into bytecode before it runs. TorchDynamo hooks into the Python frame evaluation API (PEP 523) to "spy" on this bytecode right before it executes.
It scans the bytecode and identifies PyTorch operations. If it sees something it understands (like torch.add), it adds it to the FX Graph.
For things it does not understand it pauses the graph capture and produces a Graph Break. It splits the graph, lets Python handle the unsupported part, and then tries to resume capturing afterwards.

### Guards:
`torch.compile` makes some assumptions about runtime values as we trace through code. During tracing, we generate “guards”, which are runtime checks for these assumptions. Guards are run in future calls to the compiled function to determine if we can reuse previously compiled code. 
Ex: `if x.sum() > 0:` the execution path changes based on the data.
if the second time the function is called there is change in the condition then torch.compile needs to recompile the function.

Basically when you run your compiled model a second time, the Guard checks: “Does the input tensor have the exact same Shape and Dtype as last time?”

If the answer is yes, we use the cached, optimized kernel. If the answer is no trigger a recompilation.

### Note:
torch.compile initially assumes tensor shapes are static/constant and guards based on these assumptions. By using “dynamic shapes,” we can get torch.compile to produce compiled code that can accept tensor inputs with different shapes - we avoid recompiling every time shapes differ. By default, automatic dynamic shapes are enabled in torch.compile(dynamic=None) - if compilation fails due to shape mismatch, recompilation is attempted with dynamic shapes. Dynamic shapes can also be fully enabled (dynamic=True) or disabled (dynamic=False).

## AOTAutograd and ATen IR:

Dynamo only captures the Forward pass. But for deep learning training, we need backpropagation.
AOTAutograd (Ahead-Of-Time Autograd) takes the forward graph and simulates the PyTorch Autograd engine. It predicts, “To backpropagate through this, I will need these specific formulas later.” It then generates the Backward graph and stitches them together into a Joint Graph.
However the most important feature of AOTAutograd is called Decomposition(or Lowering)
The graph coming out of Dynamo is high-level. It contains complex instructions like torch.nn.CrossEntropyLoss.

AOTAutograd breaks these down into ATen IR (Intermediate Representation) — the lowest-level mathematical primitives.

CrossEntropy becomes a combination of aten.log_softmax, aten.neg, and aten.nll_loss.
Eventually, everything becomes simple adds, multiplies, and matrix matmuls.
So again we have a similifed grapgh with numbers wedges and simple ops like add, mult, mat muls as nodes.

## TorchInductor:

Now we have a graph of primitive math operations. We need to turn this into fast GPU code. This is the job of TorchInductor.

Loop Fusion & Scheduling
Inductor analyzes the graph and looks for Memory Bound operations to fuse.

Before: Read A -> Multiply -> Save to Temp -> Read Temp -> Add -> Save to B.
After (Fused): Read A -> Multiply & Add immediately (in registers) -> Save to B.
It eliminates the “Save to Temp” and “Read Temp” steps entirely. 

### Code Generation: Triton

Once the schedule is optimized, Inductor generates Triton code.

Triton operates on Blocks. You tell it, “Take this block of data and multiply it there,” and it handles the complex memory management and thread synchronization for you.

TorchInductor effectively acts as a Triton code generator. It writes the Triton kernel that maximizes cache (SRAM) usage on the GPU, so you don’t have to.

## Dispatcher:
referenced from: https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/

Dispatcher is a traffic controller that decides which implementation of an operator should run.
When we write `torch.add(x, y)`, there isn't just one add, there are actually the following:
CPU version, CUDA version, Autograd-tracked version, Tracing version, XLA version

The dispatcher decides: Which one should I actually run right now?
The dispatcher’s job is to compute a dispatch key, based on the input tensors and some other stuff (more on this shortly), and then do an indirect jump to the function pointed to by the table.

Why this complexity?:

Because PyTorch supports:

Multiple devices (CPU, CUDA, XLA)
Autograd
Tracing
Functionalization
Meta tensors
Custom backends

Each of these is a cross-cutting concern.

So how exactly do we compute the dispatch key which we use to index into the dispatch table?


The general concept is that we union together dispatch key sets from various sources (and in some case mask out some dispatch keys), giving us a final dispatch key set. Then, we pick the first dispatch key in the set (dispatch keys are implicitly ordered by some priority) and that is where we should dispatch to. What are these sources?

Ex: 
x = torch.tensor(..., device="cuda", requires_grad=True)

with this, internally it will have key set as:
{CUDA, Autograd}

Some modes are not properties of tensors like
torch.fx tracing mode

these are temporarily via thread-local state(TLS), and these contribute as extra keys
so we get
{CUDA, Autograd, Tracing}

Finally, we have a global set, which are dispatch keys that are always considered. There is also a local exclude set, which is used to exclude dispatch keys from dispatch. A common pattern is for some handler to handle a dispatch key, and then mask itself off via the local exclude set, so we don’t try reprocessing this dispatch key later.

Dispatch works like layers:
Example:

User calls torch.add

- Tracing wrapper runs
- Inside it, it redispatches
- Autograd wrapper runs
- Inside it, redispatch
- CUDA kernel runs

Each layer:

- Handles its concern
- Masks itself
- Redispatches

