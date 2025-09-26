import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import os
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import time
import argparse

# ========================
# 1. Model setup
# ========================
REPO_DIR = ""
WEIGHTS = os.path.expanduser("~/scratch/dinov3_checkpoints/dinov3_vit7b16_coco_detr_head-b0235ff7.pth")
BACKBONE_WEIGHTS = os.path.expanduser("~/scratch/dinov3_checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
RESULT_PATH = os.path.expanduser("~/scratch/detections_dinov3_results.json")

slurm_tmp = os.environ["SLURM_TMPDIR"]
COCO_ANN_FILE = os.path.join(slurm_tmp, "coco/annotations/instances_val2017.json")
COCO_IMG_DIR = os.path.join(slurm_tmp, "coco/val2017")

def main(args):
    print("Loading model...")
    model = torch.hub.load(
        REPO_DIR,
        'dinov3_vit7b16_de',
        source="local",
        weights=WEIGHTS,
        backbone_weights=BACKBONE_WEIGHTS,
    )

    model.eval()
    device = torch.device("cuda")
    model.to(device)
    print("Loaded")

    if args.m:
        model.detector.backbone[0]._backbone.backbone.config_token_merging(args.m, args.r, args.l, need_recover=True)
        print(f"Running with token merging method={args.m}, l={model.detector.backbone[0]._backbone.backbone.loc}, r={model.detector.backbone[0]._backbone.backbone.r}")
    else:
        print("Running baseline model")

    # ========================
    # 2. Provided image transform
    # ========================
    def make_coco_eval_transform():
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    transform = make_coco_eval_transform()

    # ========================
    # 3. Dataset setup (COCO val2017)
    # ========================
    coco_dataset = torchvision.datasets.CocoDetection(
        COCO_IMG_DIR,
        COCO_ANN_FILE,
        transform=transform,
    )

    data_loader = DataLoader(
        coco_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: tuple(zip(*batch)),  # detection models expect lists
    )

    # ========================
    # 4. Evaluation loop
    # ========================
    coco_gt = COCO(COCO_ANN_FILE)
    results = []

    total = 0
    total_time = 0
    iter = 0
    total_flops = 0
    print("Running inference...")
    print("Total batches:", len(data_loader))

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            total += len(imgs)

            torch.cuda.synchronize()
            t = time.time()
            outputs = model(imgs)
            torch.cuda.synchronize()
            t = time.time() - t
            total_time += t

            for tgt, output in zip(targets, outputs):
                if len(tgt) == 0:   # image has no annotations
                        continue
                img_id = tgt[0]["image_id"]

                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": img_id,
                        "category_id": label.item(),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score.item(),
                    })
            iter += 1
            if iter == args.s:
                print(f"Early stop at iter:{iter}")
                break

    print(f"Avg runtime:{total_time/total}")
    # ========================
    # 5. Save & COCO eval
    # ========================
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f)

    img_ids = [r["image_id"] for r in results]
    coco_dt = coco_gt.loadRes(RESULT_PATH)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def parse_args():
    parser = argparse.ArgumentParser(description='Script with conditional argument requirements')
    
    parser.add_argument('-l', type=int, help='Integer value for layer to apply token merging')
    parser.add_argument('-r', type=float, help='Float value for r parameter')
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