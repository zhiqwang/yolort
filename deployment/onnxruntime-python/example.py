import argparse
from typing import Tuple, List

import numpy as np
import cv2
import onnxruntime as ort


class YOLOv5Detector:
    def __init__(self, model_path: str, gpu: bool = True) -> None:
        ort_device = ort.get_device()
        self._providers = None

        if ort_device == 'GPU' and gpu:
            self._providers = ['CUDAExecutionProvider']
            print('Inference device: GPU')
        elif gpu:
            print('GPU is not supported by your ONNXRuntime build. Fallback to CPU.')
        else:
            self._providers = ['CPUExecutionProvider']
            print('Inference device: CPU')

        self._model = ort.InferenceSession(model_path, providers=self._providers)
        self._input_names = self._model.get_inputs()[0].name
        print('Model was initialized.')

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = rgb_image.astype(np.float32) / 255.0
        blob = np.transpose(blob, axes=(2, 0, 1))

        return blob

    def __call__(self, image: np.ndarray) -> Tuple[List[float], List[int], List[List[int]]]:
        blob = self._preprocessing(image)
        scores, class_ids, boxes = self._model.run(output_names=None,
                                                   input_feed={self._input_names: blob})
        boxes = boxes.astype(np.int32)
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]  # from xyxy to xywh format

        return scores.tolist(), class_ids.tolist(), boxes.tolist()


def visualize_detection(image: np.ndarray,
                        class_names: List[str],
                        scores: List[float],
                        class_ids: List[int],
                        boxes: List[List[int]]) -> None:

    for i, class_id in enumerate(class_ids):
        x, y, w, h = boxes[i]
        conf = round(scores[i], 2)

        label = class_names[class_id] + " " + str(conf)
        text_size = cv2.getTextSize(label, cv2.FONT_ITALIC, 0.8, 2)[0]

        cv2.rectangle(image, (x, y), (x + w, y + h), (229, 160, 21), 2)
        cv2.rectangle(image, (x, y - 25), (x + text_size[0], y), (229, 160, 21), -1)
        cv2.putText(image, label, (x, y - 3), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('ONNXRuntime inference script.', add_help=True)
    parser.add_argument('--model_path', default='yolov5.onnx', type=str, required=True,
                        help='Path to onnx model.')
    parser.add_argument('--image', default='bus.jpg', type=str, required=True,
                        help='Image source to be detected.')
    parser.add_argument('--class_names', default='coco.names', type=str, required=True,
                        help='Path of dataset labels.')
    parser.add_argument('--gpu', action='store_true',
                        help='To enable inference on GPU.')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('Command Line Args: ', args)

    with open(args.class_names, 'r', encoding='utf-8') as f:
        class_names = f.readlines()
        class_names = [class_name.strip() for class_name in class_names]

    detector = YOLOv5Detector(model_path=args.model_path, gpu=args.gpu)

    image = cv2.imread(args.image)
    scores, class_ids, boxes = detector(image)

    visualize_detection(image, class_names, scores, class_ids, boxes)

    cv2.imshow('result', image)
    cv2.waitKey(0)
