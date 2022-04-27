.. _models:

yolort.models
#############

Models structure
================

The models expect a list of ``Tensor[C, H, W]``, in the range ``0-1``.
The models internally resize the images but the behaviour varies depending
on the model. Check the constructor of the models for more information.

.. autofunction:: yolort.models.YOLOv5

Pre-trained weights
===================

The pre-trained models return the predictions of the following classes:

  .. code-block:: python

      COCO_INSTANCE_CATEGORY_NAMES = [
         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
         'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
         'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
         'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'
      ]

.. autofunction:: yolort.models.yolov5n
.. autofunction:: yolort.models.yolov5n6
.. autofunction:: yolort.models.yolov5s
.. autofunction:: yolort.models.yolov5s6
.. autofunction:: yolort.models.yolov5m
.. autofunction:: yolort.models.yolov5m6
.. autofunction:: yolort.models.yolov5l
.. autofunction:: yolort.models.yolov5ts
