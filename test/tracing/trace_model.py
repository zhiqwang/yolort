import torch

from hubconf import yolov5s
from models.transform import WrappedNestedTensor


if __name__ == "__main__":

    model = yolov5s(pretrained=True)
    model = WrappedNestedTensor(model)
    model.eval()

    traced_model = torch.jit.script(model)
    traced_model.save("./test/tracing/yolov5s.torchscript.pt")
