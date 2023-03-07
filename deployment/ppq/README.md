# ppq int8 ptq Example

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

The ppq int8 ptq example of `yolort`.

## Dependencies

- ppq
- torch
- OpenCV
- onnx

## Usage

Here we will mainly discuss how to use the ppq interface, we recommend that you check out  [tutorial](https://github.com/openppl-public/ppq/tree/master/ppq/samples) first.

1. Distill your calibration data (Optional: If you don't have images for calibration and bn is in your model, you can use this)
1. Export your custom model to onnx format
1. Quantization onnx-format float model to a json file and a float model
