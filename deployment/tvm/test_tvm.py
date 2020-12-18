import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import numpy as np
import cv2

import torch
from hubconf import yolov5s


in_size = 300

def get_test_img():
    img_path = './street_small.jpg'
    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return img


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def get_script_module():
    inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))
    model_func = yolov5s
    model = TraceWrapper(model_func(pretrained=True))
    model.eval()

    with torch.no_grad():
        out = model(inp)
        script_module = do_trace(model, inp)

    return script_module


def retina_net_lab():

    img = get_test_img()

    input_name = "input0"
    input_shape = (1, 3, in_size, in_size)
    shape_list = [(input_name, input_shape)]
    # script_module = get_script_module()
    # mod, params = relay.frontend.from_pytorch(script_module, shape_list)
    # print(mod["main"])

    # with open("yolov5s_mod.json", "w") as fo:
    #     fo.write(tvm.ir.save_json(mod))
    # with open("yolov5s.params", "wb") as fo:
    #     fo.write(relay.save_param_dict(params))

    with open("yolov5s_mod.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open("yolov5s.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        vm_exec = relay.vm.compile(mod, target=target, params=params)

    print("compile finished")
    return

    ctx = tvm.cpu()
    vm = VirtualMachine(vm_exec, ctx)
    vm.set_input("main", **{input_name: img})
    tvm_res = vm.run()

    score_threshold = 0.9
    boxes = tvm_res[0].asnumpy().tolist()
    valid_boxes = []
    for i, score in enumerate(tvm_res[1].asnumpy().tolist()):
        if score > score_threshold:
            valid_boxes.append(boxes[i])
        else:
            break

    print("Get {} valid boxes".format(len(valid_boxes)))


if __name__ == '__main__':
    retina_net_lab()
