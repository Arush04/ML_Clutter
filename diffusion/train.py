import argparse
import os
import wandb
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.device_mesh import init_device_mesh
from timm.utils import ModelEmaV3
from model import UNET
from utils import DDPM_Scheduler, set_seed, ImageOnlyDataset

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
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=rank)
    set_seed(0)

    # Initialize wandb on rank 0 only
    if rank == 0:
        wandb.init(
            project="FSDP2_DDPM",
            name=f"fsdp2-run-{wandb.util.generate_id()}",
            config=vars(args)
        )

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = ImageOnlyDataset(image_dir="DIV2K_train_HR", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    with torch.device("meta"):
        model = UNET()
    with torch.device("meta"):
        ema = ModelEmaV3(model, decay=args.ema_decay)

    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs["offload_policy"] = OffloadPolicy()
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data_parallel",))
    print(f"Device Mesh: {device_mesh}")

    fully_shard(model, mesh=device_mesh, **fsdp_kwargs)
    model.to_empty(device="cuda")
    ema = ema.to_empty(device="cuda")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')
    scheduler = DDPM_Scheduler(num_time_steps=args.num_time_steps, device=device).to(device)

    if args.checkpoint_path is not None and rank == 0:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['weights'], strict=False)
        ema.load_state_dict(checkpoint['ema'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])

    dist.barrier()
    
    ctr = 0
    # Training loop with wandb logging
    for epoch in range(args.num_epochs):
        ctr += 1
        total_loss = 0
        for bidx, x in enumerate(train_loader):
            x = x.to(device)
            t = torch.randint(0, args.num_time_steps, (args.batch_size,), device=device)
            e = torch.randn_like(x)
            _, a = scheduler(t)
            a = a.view(args.batch_size, 1, 1, 1)
            x_noisy = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

            optimizer.zero_grad()
            output = model(x_noisy, t)
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)

            # Log loss per batch
            if bidx % 10 == 0 and rank == 0:
                print(f"Rank {rank} | Epoch {epoch+1} | Batch {bidx} | Loss {loss.item():.5f}")
                wandb.log({"batch_loss": loss.item(), "epoch": epoch+1, "batch": bidx})

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"Rank {rank} | Epoch {epoch+1} | Avg Loss {avg_loss:.5f}")
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})
    # Save model checkpoint
    if rank == 0:
        model_state = model.state_dict()
        ema_state = ema.state_dict()
        ema_state = {k.replace("module.", ""): v for k, v in ema_state.items()}
        checkpoint = {
            'weights': model_state,
            'optimizer': optimizer.state_dict(),
            'ema': ema_state
        }
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/ddpm_checkpoint_{ctr}.pt'
        torch.save(checkpoint, ckpt_path)

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
