import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy, StateDictType
from torch.distributed.device_mesh import init_device_mesh
from ddmp_basic import ddpm_simple
from model import UNET
from utils import DDPM_Scheduler, set_seed, ImageOnlyDataset
from torch.utils.checkpoint import checkpoint

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key in ema_state.keys():
            if key in model_state:
                ema_param = ema_state[key]
                model_param = model_state[key]
                if hasattr(ema_param, "to_local") and hasattr(model_param, "to_local"):
                    ema_param_local = ema_param.to_local()
                    model_param_local = model_param.to_local()
                    if ema_param_local.shape == model_param_local.shape:
                        ema_param_local.mul_(decay).add_(model_param_local, alpha=1 - decay)
                    else:
                        raise ValueError(
                            f"Shape mismatch in EMA update for {key}: "
                            f"ema_param_local.shape={ema_param_local.shape}, "
                            f"model_param_local.shape={model_param_local.shape}"
                        )
                else:
                    if ema_param.shape == model_param.shape:
                        ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
                    else:
                        raise ValueError(
                            f"Shape mismatch in EMA update for {key}: "
                            f"ema_param.shape={ema_param.shape}, "
                            f"model_param.shape={model_param.shape}"
                        )
        ema_model.load_state_dict(ema_state)

def main(args):
    # Distributed setup
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=rank)
    set_seed(0)

    # Data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = ImageOnlyDataset(image_dir="DIV2K_train_HR", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    # Model: Initialize on meta device for FSDP2
    with torch.device("meta"):
        model = UNET(input_channels=3, output_channels=3, device=device)
    model.to_empty(device=device)
    with torch.device("meta"):
        ema_model = UNET(input_channels=3, output_channels=3, device=device)
    ema_model.to_empty(device=device)

    # FSDP2 sharding
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs["offload_policy"] = OffloadPolicy()
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp2",))
    fully_shard(model, mesh=device_mesh, **fsdp_kwargs)
    fully_shard(ema_model, mesh=device_mesh, **fsdp_kwargs)

    ema_model.load_state_dict(model.state_dict())

    # Optimizer, criterion, scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')
    scheduler = DDPM_Scheduler(num_time_steps=args.num_time_steps, device=device)
    scheduler = scheduler.to(device)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        ema_model.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        for bidx, x in enumerate(train_loader):
            x = x.to(device)
            t = torch.randint(0, args.num_time_steps, (args.batch_size,), device=device)
            e = torch.randn_like(x)
            _, a = scheduler(t)  # Use alpha_cumprod from forward
            a = a.view(args.batch_size, 1, 1, 1)
            x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)
            
            optimizer.zero_grad()
            output = model(x_noisy, t)
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)
            if bidx % 10 == 0 and rank == 0:
                print(f"Rank {rank} | Epoch {epoch+1} | Batch {bidx} | Loss {loss.item():.5f}")
        if rank == 0:
            print(f"Rank {rank} | Epoch {epoch+1} | Avg Loss {total_loss / len(train_loader):.5f}")

    # Save checkpoint
    if rank == 0:
        model_state = model.state_dict()
        ema_state = ema_model.state_dict()
        
        checkpoint = {
            'weights': model_state,
            'optimizer': optimizer.state_dict(),
            'ema': ema_state
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/ddpm_checkpoint_rank{rank}.pt')
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP2 Vision Model Training")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-time-steps", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    args = parser.parse_args()
    main(args)
