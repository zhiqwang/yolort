import torch

from hubconf import yolov5


if __name__ == "__main__":

    model = yolov5(
        cfg_path='./models/yolov5s.yaml',
    )
    model.eval()

    traced_model = torch.jit.script(model)
    traced_model.save("./test/tracing/yolov5s.torchscript.pt")
