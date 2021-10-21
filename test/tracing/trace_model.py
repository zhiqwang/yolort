import torch
from hubconf import yolov5s


if __name__ == "__main__":

    model = yolov5s(pretrained=True)
    model.eval()

    dummy_inputs = [torch.rand(3, 416, 352), torch.rand(3, 480, 384)]
    traced_model = torch.jit.trace(model, dummy_inputs)
    traced_model.save("./test/tracing/yolov5s.torchscript.pt")
