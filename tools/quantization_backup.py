import time
import datetime
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.autograd import Function, Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
import onnx
import torch.optim as optim
import argparse
import os
import tqdm
from PIL import Image

from yolort.models._checkpoint import load_from_ultralytics
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.box_head import YOLOHead

class UniformDataset(Dataset):
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform 
        self.size = size
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        sample = (torch.randint(high=255, size=self.size).float() - 127.5) / 5418.75
        return sample

def getRandomData(size_ = (3, 416, 416), num_data_ = 100, batch_size_ = 4):
    dataset = UniformDataset(num_data_, size = size_, transform=None)
    dataloader_ = DataLoader(dataset, batch_size=batch_size_, shuffle=True, num_workers=8)
    return dataloader_

def own_loss(A, B):
    return (A - B).norm()**2 / B.size(0)

class output_hook(object):
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None
    
    def hook(self, module, input, output):
        self.outputs = output
    
    def clear(self):
        self.outputs = None

def getDistillData(
        path,
        teacher_model,
        size,
        batch_size,
        start_idx,
        iterations=500,
        num_batch=1):
    # init
    dataloader = getRandomData(size, num_batch, batch_size)

    eps = 1e-6
    # init hooks and single precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()])
    
    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the conv layers to get the intermediate output after conv and before bn:
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in bn layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var + eps).detach().clone().flatten().cuda())
            )

    assert len(hooks) == len(bn_stats)

    for i, gaussian_data in enumerate(tqdm.tqdm(dataloader)):
        if i == num_batch:
            break
        # init criterion, optimizer, scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1e-4,
                                                        verbose=False,
                                                        patience=100)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(iterations):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            # print(f"gaussian_data.shape = {gaussian_data.shape}")
            output = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0

            # compute the loss according to bn stats and intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                    tmp_output.size(1), -1), dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0), tmp_output.size(1), -1), dim=2) + eps)
                
                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3, -1), dim=2)
            tmp_std = torch.sqrt(torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1), dim=2) + eps)
            mean_loss += own_loss(input_mean, tmp_mean)
            std_loss += own_loss(input_std, tmp_std)
            total_loss = mean_loss + std_loss

            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
        # refined_gaussian.append(gaussian_data.detach().clone())
        gaussian_data = gaussian_data.clone().detach().to(torch.device("cpu"))
        # print(gaussian_data)
        # print(i)
        torchvision.utils.save_image(gaussian_data, os.path.join(path, f"{start_idx + i}.jpg"))


        # print(f"gaussian_data.shape = {gaussian_data.shape}")
        # print(f"len(refined_gaussian) = {len(refined_gaussian)}")
    
    for handle in hook_handles:
        handle.remove()
    
    # return refined_gaussian

class quantDatasets(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.images = [os.path.join(self.root, path) for path in os.listdir(self.root)]
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        # _, image_name = os.path.split(image_path)
        if self.transform:
            image = self.transform(image)
        return image

def prepare_data_loaders(data_path, shape):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # dataset = torchvision.datasets.ImageNet(
    #     data_path, split="train", transform=transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    dataset_test = quantDatasets(
                        data_path, 
                        transform=transforms.Compose([
                        transforms.Resize(shape[1]),
                        # transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        ]))
                        # normalize,]))

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler)

    return data_loader_test

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        # for image, _ in data_loader:
        for image in data_loader:
            model(image)

def get_parser():
    parser = argparse.ArgumentParser("ptq tool.", add_help=True)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default = "./model/yolov5s.pt",
        help="The path of checkpoint weights"
    )
    parser.add_argument(
        "--version",
        type=str,
        default = "r6.0",
        help="opset version"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default = 0.25,
        help="threshold"
    )
    parser.add_argument(
        "--distilled_data_path",
        type=str,
        default = "./distilled_data/",
        help="The path of distilled data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default = 1,
        help="batch size"
    )
    parser.add_argument(
        "--num_of_batches",
        type=int,
        default = 100,
        help="num of batches"
    )
    parser.add_argument(
        "--distill_iterations",
        type=int,
        default = 500,
        help="distill iterations"
    )
    parser.add_argument(
        "--input_size",
        default = [3, 640, 640],
        type = int,
        help="input size"
    )
    parser.add_argument(
        "--onnx_input_name",
        type=str,
        default = "dummy_input",
        help="onnx input name"
    )
    parser.add_argument(
        "--onnx_output_name",
        type=str,
        default = "dummy_output",
        help="onnx output name"
    )
    parser.add_argument(
        "--onnx_output_path",
        type=str,
        default = "./float_yolov5.onnx",
        help="onnx output name"
    )
    parser.add_argument(
        "--sim_onnx_output_path",
        type=str,
        default = "./sim_float_yolov5.onnx",
        help="simed onnx output name"
    )
    parser.add_argument(
        "--quantized_onnx_output_path",
        type=str,
        default = "./model/quantized_yolov5.onnx",
        help="simed onnx output name"
    )
    parser.add_argument(
        "--quantized_onnx_json_path",
        type=str,
        default = "./model/quantized_yolov5.json",
        help="simed onnx output name"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default = 11,
        help="opset version"
    )
    parser.add_argument(
        "--device",
        type=str,
        default = "cuda",
        help="opset version"
    )
    parser.add_argument(
        "--calib_steps",
        type=int,
        default = 64,
        help="opset version"
    )

    return parser

class YOLO(nn.Module):
    def __init__(self, backbone: nn.Module, strides, num_anchors, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = YOLOHead(backbone.out_channels, num_anchors, strides, num_classes)

    def forward(self, samples):

        # get the features from the backbone
        features = self.backbone(samples)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)
        return head_outputs

class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x):
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data

def make_model(checkpoint_path, version):

    model_info = load_from_ultralytics(checkpoint_path, version=version)

    backbone_name = f"darknet_{model_info['size']}_{version.replace('.', '_')}"
    depth_multiple = model_info["depth_multiple"]
    width_multiple = model_info["width_multiple"]
    use_p6 = model_info["use_p6"]
    backbone = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6)
    strides = model_info["strides"]
    num_anchors = len(model_info["anchor_grids"][0]) // 2 
    num_classes = model_info["num_classes"]
    model = YOLO(backbone, strides, num_anchors, num_classes)

    model.load_state_dict(model_info["state_dict"])
    model = ModelWrapper(model)

    model = model.eval()

    return model

def collate_fn(batch):
    return batch.to("cuda")