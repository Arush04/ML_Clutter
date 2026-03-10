## Mixture of Experts (MoE)

### Very high level overview  
MoE enables models to be trained faster and also have less inference time.

MoE explained in terms of a regular transformer model:  
It consists of two important elements  
1. Sparse MoE layers: these are used in place of dense feed-forward network (FFN) layers. MoE layers have a certain number of “experts” (e.g. 8), where each expert is a neural network.  

2. A Gate network or router: it determines which tokens are sent to which expert.

![sparse switch ffn layer](images/moe_layer.png)

## Disadvatages

1. Training: MoE models have many more parameters than dense transformers, but only a subset of experts is active per token during training. The main challenges during fine-tuning are expert imbalance, router instability, and expert specialization rather than simple overfitting.
2. MoE might have a lot of parameters but only some of them are used during inference (only certain **expert(s)** needs to work at any given time), this leads to much faster inference but we still need to load up all parameters in RAM, so memmory requirements are high.

## Sparsity

Runing some parts of the whole system. The idea of conditional computation (parts of the network are active on a per-example basis) allows one to scale the size of the model without increasing the computation, and hence, this led to thousands of experts being used in each MoE layer.

This setup introduces some challenges. For example, although large batch sizes are usually better for performance, batch sizes in MOEs are effectively reduced as data flows through the active experts. For example, if our batched input consists of 10 tokens, five tokens might end in one expert, and the other five tokens might end in five different experts, leading to uneven batch sizes and underutilization.

How can we solve this? A learned gating network (G) decides which experts (E) to send a part of the input:  

In the most traditional setup, we just use a simple network with a softmax function.
$$G_{\sigma}(x) = Softmax(x.W_{g})$$

## Load balancing MoE:

As discussed before, if all our tokens are sent to just a few popular experts, that will make training inefficient. In a normal MoE training, the gating network converges to mostly activate the same few experts. This self-reinforces as favored experts are trained quicker and hence selected more. To mitigate this, an auxiliary loss is added to encourage giving all experts equal importance. This loss ensures that all experts receive a roughly equal number of training examples.


## Expert Parallelism with Load Balancing (EPLB)

In modern deployments, experts are distributed across multiple GPUs and nodes. Even if the gating network balances tokens across experts, some GPUs may still become overloaded if too many active experts or tokens are mapped to the same device. This leads to:

- GPU underutilization on some devices
- Increased communication overhead between nodes
- Slower overall throughput

To address this, systems implement Expert Parallel Load Balancing (EPLB), which balances experts and tokens across hardware resources.

There are different types of policies depeinding on the cluster configuration

### Heirarchical Load Balancing

When the number of server nodes divides evenly into the number of expert groups, EPLB applies a hierarchical load balancing policy.  

- Distribute expert groups accross nodes
- Replicate experts within each node
- Distribute replicated expertse across GPUs

### Global Load Balancing

When the cluster configuration does not align cleanly with expert groups.

- Experts are replicated globally across nodes, rather than being grouped locally.
- The replicated experts are then distributed across GPUs regardless of expert grouping.
- Tokens can be routed to any available replica of an expert.

Sources:
[1] Mixture of Experts Explained (https://huggingface.co/blog/moe)
[2] EPLB (https://github.com/deepseek-ai/EPLB)
