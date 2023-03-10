import argparse
import os

import onnx
import ppq.lib as PFL
import torch

from onnxsim import simplify

from ppq import graphwise_error_analyse, TargetPlatform, TorchExecutor
from ppq.api import export_ppq_graph, load_onnx_graph
from ppq.quantization.optim import (
    ParameterQuantizePass,
    PassiveParameterQuantizePass,
    QuantAlignmentPass,
    QuantizeFusionPass,
    QuantizeSimplifyPass,
    RuntimeCalibrationPass,
)

from utils import collate_fn, prepare_data_loaders


PLATFORM = TargetPlatform.TRT_INT8


def main():
    parser = argparse.ArgumentParser("ptq tool.", add_help=True)

    parser.add_argument(
        "--calibration_data_path",
        type=str,
        default="./distilled_data/",
        help="The path of calibration data, if zeroq is not used, you should set it",
    )
    parser.add_argument("--show_error_cal", type=int, default=1, help="flag to show error cal")

    parser.add_argument("--input_size", default=[3, 640, 640], type=int, help="input size")

    parser.add_argument("--calib_steps", type=int, default=64, help="opset version")

    parser.add_argument("--onnx_input_path", type=str, default="./model/yolov5.onnx", help="onnx input path")

    parser.add_argument(
        "--quantized_res_path",
        type=str,
        default="./model/",
        help="quantized outputs",
    )

    parser.add_argument("--device", type=str, default="cuda", help="opset version")

    args = parser.parse_args()

    sim_onnx_output_name = "sim_float_yolov5.onnx"
    quantized_onnx_output_name = "quantized_float_yolov5.onnx"
    quantized_json_output_name = "quantized_json_yolov5.onnx"

    print(f"Command Line Args: {args}")

    # quantization
    onnx_model = onnx.load(args.onnx_input_path)
    simplified, _ = simplify(onnx_model)
    onnx.save(simplified, os.path.join(args.quantized_res_path, sim_onnx_output_name))
    graph = load_onnx_graph(os.path.join(args.quantized_res_path, sim_onnx_output_name))

    quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_INT8, graph=graph)
    dispatching = {op.name: TargetPlatform.FP32 for op in graph.operations.values()}
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
    executor = TorchExecutor(graph=graph, device=args.device)
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
            graph=graph,
            running_device=args.device,
            dataloader=dataloader,
            collate_fn=collate_fn,
        )

    export_ppq_graph(
        graph=graph,
        platform=TargetPlatform.TRT_INT8,
        graph_save_to=os.path.join(args.quantized_res_path, quantized_onnx_output_name),
        config_save_to=os.path.join(args.quantized_res_path, quantized_json_output_name),
    )


if __name__ == "__main__":
    main()
