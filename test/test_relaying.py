from torch.jit._trace import TopLevelTracedModule
from yolort.models import yolov5s
from yolort.relaying import get_trace_module


def test_get_trace_module():
    model_func = yolov5s(pretrained=True)
    script_module = get_trace_module(model_func, input_shape=(416, 320))
    assert isinstance(script_module, TopLevelTracedModule)
    assert script_module.code is not None
