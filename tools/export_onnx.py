#coding:utf-8

import argparse
import os
import sys
sys.path.append("../")

import onnx
import torch
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from yolort.models import yolov5s

def get_parser():
    parser = argparse.ArgumentParser('export onnx CLI for yolort', add_help=True)

    parser.add_argument('--image_size', default=640, type=int,
                        help='Image size for evaluation (default: 640)')
    
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--output_dir', default='.',
                        help='Path where to save')

    parser.add_argument('--input_names', default='input', type=str,
                        help='input name of model')
    
    parser.add_argument('--output_names', nargs='+', default=['scores', 'label', 'boxes'],
                        help='output name of model')

    parser.add_argument('--opset_version', default=11, type=int,
                        help='opset_version')

    parser.add_argument('--do_constant', default=True, type=bool,
                        help='do_constant')

    parser.add_argument('--dynamic', default=True, type=bool,
                        help='dynamic_axes')

    parser.add_argument('--simplify', default=True, type=bool,
                        help='simplified ONNX model')
    return parser


def export_onnx(model, input, save_name, input_names='input',
               output_names=['scores', 'label', 'boxes'],
               opset_version=11, do_constant=True,
               dynamic=True, simplify=False,
               batch_size=1, image_size=640) :

    print(input_names)
    print(type(output_names[0]))
    print('opset version: {}'.format(_onnx_opset_version))
    torch.onnx.export(
        model,
        input,
        save_name,
        do_constant_folding = do_constant,
        opset_version=opset_version,
        input_names = [input_names],
        output_names = output_names,
        dynamic_axes = {
            output_names[0]: [0, 1],
            output_names[1]: [0],
            output_names[2]: [0],
        } if dynamic else None
    )

    if (simplify):
        import onnxsim

        print("Starting simplifing with onnxsim {}".format(onnxsim.__version__))
        base_dir, base_name = os.path.split(save_name)
        file_name = base_name.split('.')[0]
        onnxsim_path = os.path.join(base_dir, "{}_sim.onnx".format(file_name))

        #load onnx mode
        onnx_model = onnx.load(save_name)

        #conver mode
        model_sim, check = onnxsim.simplify(
            onnx_model,
            input_shapes={input_names: [batch_size, 3, image_size, image_size]},
            dynamic_input_shape=dynamic,
        )

        assert check, "Simplified ONNX model could not be validated"

        onnx.save(model_sim, onnxsim_path)
        print("End of Simplified!")

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print('Command Line Args: {}'.format(args))

    image_size    = args.image_size
    batch_size    = args.batch_size

    output_dir    = args.output_dir
    input_names   = args.input_names
    output_names  = args.output_names
    opset_version = args.opset_version
    do_constant   = args.do_constant
    dynamic       = args.dynamic
    simplify      = args.simplify

    if (not os.path.isdir(output_dir)) :
        output_dir = "."

    output_path = os.path.join(output_dir, "yolov5s.onnx")
    
    # input data
    input = torch.randn(batch_size, 3, image_size, image_size)
    model = yolov5s(pretrained = True, export_friendly = True)
    model.eval()

    # export onnx
    export_onnx(
        model,
        input,
        output_path,
        input_names,
        output_names,
        opset_version,
        do_constant,
        dynamic,
        simplify,
    )

if __name__ == "__main__":
    cli_main()