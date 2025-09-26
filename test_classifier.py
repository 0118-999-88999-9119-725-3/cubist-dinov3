import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import time
import argparse

IMAGENET_VAL_DIR = "/datashare/imagenet/ILSVRC2012/val"

# ========================
# 1. Model setup
# ========================
REPO_DIR = ""
WEIGHTS = "~/scratch/dinov3_checkpoints/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth"
BACKBONE_WEIGHTS = "~/scratch/dinov3_checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

def main(args):
    # Load the model
    print("loading model")

    dinov3_vit7b16_lc = torch.hub.load(
        REPO_DIR, 
        'dinov3_vit7b16_lc', 
        source="local", 
        weights=WEIGHTS, 
        backbone_weights=BACKBONE_WEIGHTS,
    )
    dinov3_vit7b16_lc.eval()  # set to eval mode
    device = torch.device("cuda")
    dinov3_vit7b16_lc.to(device)
    print("loaded model")

    if args.m:
        dinov3_vit7b16_lc.backbone.config_token_merging(args.m, args.r, args.l, need_recover=False)
        print(f"Running with token merging method={args.m}, l={dinov3_vit7b16_lc.backbone.loc}, r={dinov3_vit7b16_lc.backbone.r}")
    else:
        print("Running baseline model")


    # ========================
    # 2. Data preparation
    # ========================
    from torchvision import transforms

    def make_transform(resize_size: int = 224):
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([resize, to_tensor, normalize])

    transform = make_transform(224)
    val_dataset = datasets.ImageFolder(IMAGENET_VAL_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    print("loaded data")

    # ========================
    # 3. Evaluation
    # ========================
    top1_correct = 0
    top5_correct = 0
    total = 0
    total_time = 0

    print("Total batches:", len(val_loader))
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            torch.cuda.synchronize()
            t = time.time()
            outputs = dinov3_vit7b16_lc(images)  # logits or features
            torch.cuda.synchronize()
            t = time.time() - t

            probs = F.softmax(outputs, dim=1)
            _, top5 = probs.topk(5, dim=1)

            total += targets.size(0)
            top1_correct += (top5[:, 0] == targets).sum().item()
            top5_correct += sum([t in top for t, top in zip(targets, top5)])
            total_time += t

            if i == args.s:
                print(f"Early stop at iter:{i}")
                break

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total

    print(f"ImageNet Validation Results:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Avg runtime: {total_time/total}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Script with conditional argument requirements')
    
    parser.add_argument('-l', type=int, help='Integer value for layer to apply token merging')
    parser.add_argument('-r', type=int, help='Int value for r parameter')
    parser.add_argument('-m', choices=['cume', 'tome', 'expedite'], 
                       help='Method choice: cume, tome, or expedite')
    parser.add_argument('-s', type=int, help='Integer value for iter to stop eval')
    
    args = parser.parse_args()
    
    # Check that either all three are specified or none are specified
    specified = [args.l is not None, args.r is not None, args.m is not None]
    
    if not (all(specified) or not any(specified)):
        parser.error("Either all three arguments (-l, -r, -m) must be specified, or none of them (run baseline)")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)