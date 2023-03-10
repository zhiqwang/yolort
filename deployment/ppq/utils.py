import os

import torch
import torchvision
import tqdm
from PIL import Image

from torch import nn, optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


def get_random_data(size_=(3, 416, 416), num_data_=100, batch_size_=4):
    dataset = UniformDataset(num_data_, size=size_, transform=None)
    dataloader_ = DataLoader(dataset, batch_size=batch_size_, shuffle=True, num_workers=8)
    return dataloader_


def own_loss(A, B):
    return (A - B).norm() ** 2 / B.size(0)


class OutputHook(object):
    def __init__(self):
        super(OutputHook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


def get_distill_data(path, teacher_model, size, batch_size, start_idx, iterations=500, num_batch=1):
    # init
    dataloader = get_random_data(size, num_batch, batch_size)

    eps = 1e-6
    # init hooks and single precision model
    hooks, hook_handles, bn_stats = [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([1 if isinstance(layer, nn.BatchNorm2d) else 0 for layer in teacher_model.modules()])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the conv layers to get the intermediate output after conv and before bn:
            hook = OutputHook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in bn layers
            bn_stats.append(
                (
                    m.running_mean.detach().clone().flatten().cuda(),
                    torch.sqrt(m.running_var + eps).detach().clone().flatten().cuda(),
                )
            )

    assert len(hooks) == len(bn_stats)

    for i, gaussian_data in enumerate(tqdm.tqdm(dataloader)):
        if i == num_batch:
            break
        # init criterion, optimizer, scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True

        optimizer = optim.Adam([gaussian_data], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-4, verbose=False, patience=100)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(iterations):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()

            _ = teacher_model(gaussian_data)
            mean_loss = 0
            std_loss = 0

            # compute the loss according to bn stats and intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0), tmp_output.size(1), -1), dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0), tmp_output.size(1), -1), dim=2) + eps
                )

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
        gaussian_data = gaussian_data.clone().detach().to(torch.device("cpu"))
        torchvision.utils.save_image(gaussian_data, os.path.join(path, f"{start_idx + i}.jpg"))

    for handle in hook_handles:
        handle.remove()


class QuantDatasets(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.images = [os.path.join(self.root, path) for path in os.listdir(self.root)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image


def prepare_data_loaders(data_path, shape):

    dataset_test = QuantDatasets(
        data_path,
        transform=transforms.Compose(
            [
                transforms.Resize(shape[1]),
                transforms.ToTensor(),
            ]
        ),
    )

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler)

    return data_loader_test


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image in data_loader:
            model(image)


def collate_fn(batch):
    return batch.to("cuda")
