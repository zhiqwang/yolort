import torch
from hubconf import yolov5s


if __name__ == "__main__":

    model = yolov5s(pretrained=True)
    model.eval()

    traced_model = torch.jit.trace(model)
    traced_model.save("./test/tracing/yolov5s.torchscript.pt")
