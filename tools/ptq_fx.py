import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import ao
import torchvision
import onnx
from onnxsim import simplify

import os
from PIL import Image
import argparse
from pathlib import Path

from quantization_backup import getDistillData, prepare_data_loaders, calibrate, get_parser, make_model
from yolort.models import YOLOv5


def main():

    print("In main 1")
    # parser
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    print("In main 2")
    # model initilize
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"
    model = make_model(checkpoint_path, args.version)
    model.eval()

    print("In main 3")
    # distill data
    distilled_data_path = Path(args.distilled_data_path)
    if not os.path.exists(distilled_data_path):
        os.makedirs(distilled_data_path)
    imgs_lists = os.listdir(distilled_data_path)
    sorted(imgs_lists)
    if len(imgs_lists) < args.num_of_batches:
        args.num_of_batches = args.num_of_batches - len(imgs_lists)
        getDistillData(
            args.distilled_data_path,
            model,
            args.input_size,
            args.batch_size,
            len(imgs_lists) + 1,
            args.distill_iterations,
            args.num_of_batches
        )
    
    print("In main 4")
    # dataloader
    dataloader = prepare_data_loaders(distilled_data_path, args.input_size)

    print("In main 5")
    # prepare model
    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    dummy_inputs = torch.randn(args.input_size)
    print(f"1 dummy_inputs.shape = {dummy_inputs.shape}")
    dummy_inputs = dummy_inputs.unsqueeze(0)
    print(f"2 dummy_inputs.shape = {dummy_inputs.shape}")
    prepared_model = prepare_fx(model, qconfig_mapping, dummy_inputs)

    print("In main 6")
    # calibrate
    calibrate(prepared_model, dataloader)

    print("In main 7")
    # convert
    quantized_model = convert_fx(prepared_model)

    print("In main 8")
    # export quantized model
    torch.onnx.export(
        quantized_model,
        dummy_inputs,
        args.onnx_output_path,
        args.opset_version,
        do_constant_folding=False,
        input_names=[args.onnx_input_name],
        output_names=[args.onnx_output_name]
    )
    print("In main 9")

    # sim and output
    quantized_onnx = onnx.load(args.onnx_output_path)
    sim_quantized_onnx, _ = simplify(quantized_onnx)
    onnx.save(sim_quantized_onnx, args.sim_onnx_output_path)

if __name__ == "__main__":
    main()