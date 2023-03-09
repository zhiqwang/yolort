import onnx
import ppq.lib as PFL
import torch
import torchvision
from onnx import helper, version_converter
from onnxsim import simplify

from ppq import BaseGraph, graphwise_error_analyse, QuantizationSettingFactory, TargetPlatform, TorchExecutor
from ppq.api import (
    ENABLE_CUDA_KERNEL,
    export_ppq_graph,
    load_onnx_graph,
    load_torch_model,
    quantize_onnx_model,
)
from torch.utils.data import DataLoader
from ppq.quantization.optim import *


import argparse
import os
from pathlib import Path

from PIL import Image

from utils import calibrate, collate_fn, get_distill_data, get_parser, make_model, prepare_data_loaders
from yolort.models import YOLOv5

PLATFORM = TargetPlatform.TRT_INT8


def main():

    # parser
    parser = get_parser()
    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    # model initilize
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f"Not found checkpoint file at '{checkpoint_path}'"
    model = make_model(checkpoint_path, args.version)
    model.eval()

    # distill data
    if args.distill_data:
        distilled_data_path = Path(args.distilled_data_path)
        args.calibration_data_path = distilled_data_path
        if args.regenerate_data:
            os.rmdir(distilled_data_path)
        if not os.path.exists(distilled_data_path):
            os.makedirs(distilled_data_path)
        imgs_lists = os.listdir(distilled_data_path)
        sorted(imgs_lists)
        if len(imgs_lists) < args.num_of_batches:
            args.num_of_batches = args.num_of_batches - len(imgs_lists)
            get_distill_data(
                args.distilled_data_path,
                model,
                args.input_size,
                args.batch_size,
                len(imgs_lists) + 1,
                args.distill_iterations,
                args.num_of_batches,
            )

    if args.export2onnx:
        # torch to onnx
        dummy_inputs = torch.randn([1] + args.input_size, device="cpu")
        torch.onnx.export(
            model,
            dummy_inputs,
            args.onnx_output_path,
            args.opset_version,
            do_constant_folding=False,
            input_names=[args.onnx_input_name],
            output_names=[args.onnx_output_name],
        )

        # quantization
        onnx_model = onnx.load(args.onnx_output_path)
        simplified, _ = simplify(onnx_model)
        onnx.save(simplified, args.sim_onnx_output_path)
        graph = load_onnx_graph(args.sim_onnx_output_path)

        quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_INT8, graph=graph)
        dispatching = {op.name: TargetPlatform.FP32 for op in graph.operations.values()}

    if args.ptq:
        # dataloader
        dataloader = prepare_data_loaders(args.calibration_data_path, args.input_size)
        # 从第一个卷积到最后的卷积中间的所有算子量化，其他算子不量化
        from ppq.IR import SearchableGraph

        search_engine = SearchableGraph(graph)
        for op in search_engine.opset_matching(
            sp_expr=lambda x: x.type == "Conv",
            rp_expr=lambda x, y: True,
            ep_expr=lambda x: x.type == "Conv",
            direction="down",
        ):
            dispatching[op.name] = TargetPlatform.TRT_INT8

        # 为算子初始化量化信息
        for op in graph.operations.values():
            if dispatching[op.name] == TargetPlatform.TRT_INT8:
                quantizer.quantize_operation(op_name=op.name, platform=dispatching[op.name])

        # 初始化执行器
        collate_fn = lambda x: x.to("cuda")
        executor = TorchExecutor(graph=graph, device="cuda")
        executor.tracing_operation_meta(inputs=torch.zeros(size=[1] + args.input_size).cuda())
        executor.load_graph(graph=graph)

        # 创建优化管线
        pipeline = PFL.Pipeline(
            [
                QuantizeSimplifyPass(),
                QuantizeFusionPass(activation_type=quantizer.activation_fusion_types),
                ParameterQuantizePass(),
                RuntimeCalibrationPass(),
                PassiveParameterQuantizePass(),
                QuantAlignmentPass(force_overlap=True),
                # 微调你的网络
                # LearnedStepSizePass(steps=1500)
                # 如果需要训练网络，训练过程必须发生在ParameterBakingPass之前
                # ParameterBakingPass()
            ]
        )

        # 调用管线完成量化
        pipeline.optimize(
            graph=graph,
            dataloader=dataloader,
            verbose=True,
            calib_steps=args.calib_steps,
            collate_fn=collate_fn,
            executor=executor,
        )

        if args.show_error_cal:
            graphwise_error_analyse(
                graph=graph, running_device="cuda", dataloader=dataloader, collate_fn=lambda x: x.cuda()
            )

        if not os.path.exists(args.quantized_onnx_output_path):
            os.makedirs(args.quantized_onnx_output_path)
        if not os.path.exists(args.quantized_onnx_json_path):
            os.makedirs(args.quantized_onnx_json_path)

        export_ppq_graph(
            graph=graph,
            platform=TargetPlatform.TRT_INT8,
            graph_save_to=args.quantized_onnx_output_path,
            config_save_to=args.quantized_onnx_json_path,
        )


if __name__ == "__main__":
    main()