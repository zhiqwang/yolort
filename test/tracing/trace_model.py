import torch

from hubconf import yolov5


if __name__ == "__main__":

    model = yolov5(pretrained=True)
    model.eval()

    traced_model = torch.jit.script(model)
    traced_model.save("./test/tracing/yolov5s.torchscript.pt")
