from pathlib import Path

import torch
from tqdm import tqdm
from yolort.data import COCOEvaluator
from yolort.data.coco import COCODetection
from yolort.data.transforms import default_val_transforms, collate_fn
from yolort.data._helper import get_coco_api_from_dataset
from yolort.models import yolov5s

device = torch.device('cuda')


# Setup the coco dataset and dataloader for validation
# Acquire the images and labels from the coco128 dataset
data_path = Path('/home/pyc/data_pub/coco')
coco_path = data_path
image_root = coco_path / 'val'
annotation_file = coco_path / 'annotations' / 'instances_val2017.json'

# Define the dataloader
batch_size = 6
val_dataset = COCODetection(image_root, annotation_file, default_val_transforms())
# We adopt the sequential sampler in order to repeat the experiment
sampler = torch.utils.data.SequentialSampler(val_dataset)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size,
    sampler=sampler,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=8,
)
print("calc the coco map")
coco_gt = get_coco_api_from_dataset(val_dataset)
coco_evaluator = COCOEvaluator(coco_gt)

# Model Definition and Initialization
model = yolov5s(
    pretrained=True,
    size = (640,640),
    score_thresh=0.001,
    use_square_box=True
)
model = model.eval()
model = model.to(device)
print("calc the coco map")
# COCO evaluation
for images, targets in tqdm(val_dataloader):
    images = [image.to(device) for image in images]
    preds = model(images)
    coco_evaluator.update(preds, targets)

results = coco_evaluator.compute()

# Format the results
coco_evaluator.derive_coco_results()
print(results)
