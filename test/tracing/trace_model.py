from yolort.models import yolov5s
from yolort.relay import get_trace_module


if __name__ == "__main__":

    model = yolov5s(pretrained=True)
    traced_model = get_trace_module(model, input_shape=(416, 352))

    traced_model.save("./test/tracing/yolov5s.torchscript.pt")
