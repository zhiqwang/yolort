##################################
Welcome to yolort's documentation!
##################################

**What is yolort?**

``yolort`` focus on making the training and inference of the object detection task
integrate more seamlessly together. ``yolort`` now adopts the same model
structure as the official ``YOLOv5``. The significant difference is that we adopt
the dynamic shape mechanism, and within this, we can embed both pre-processing
(``letterbox``) and post-processing (``nms``) into the model graph, which
simplifies the deployment strategy. In this sense, ``yolort`` makes it possible
to be deployed more friendly on ``LibTorch``, ``ONNX Runtime``, ``TensorRT``, ``TVM``
and so on.

.. _about-the-code:

**About the code**

Follow the design principle of `detr <https://github.com/facebookresearch/detr>`_:

..

   object detection should not be more difficult than classification, and should
   not require complex libraries for training and inference.

``yolort`` is very simple to implement and experiment with. You like the implementation
of torchvision's faster-rcnn, retinanet or detr? You like yolov5? You love yolort!

Quick get stated
================

Read a source of image(s) and detect its objects:

.. code:: python

   from yolort.models import yolov5s

   # Load model
   model = yolov5s(pretrained=True, score_thresh=0.45)
   model.eval()

   # Perform inference on an image file
   predictions = model.predict("bus.jpg")
   # Perform inference on a list of image files
   predictions = model.predict(["bus.jpg", "zidane.jpg"])

**Loading checkpoint from official yolov5**

And we support loading the trained weights from YOLOv5. Please see our documents on what
we `share`_ and how we `differ`_ from yolov5 for more details.

.. _share: https://zhiqwang.com/yolov5-rt-stack/notebooks/how-to-align-with-ultralytics-yolov5.html
.. _differ: https://zhiqwang.com/yolov5-rt-stack/notebooks/comparison-between-yolort-vs-yolov5.html

.. code:: python

   from yolort.models import YOLOv5
   from yolort.v5 import attempt_download

   # will downloaded from 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt'
   model_path = "yolov5n6.pt"
   checkpoint_path = attempt_download(model_path)

   model = YOLOv5.load_from_yolov5(checkpoint_path, score_thresh=0.25)

   model.eval()
   img_path = "bus.jpg"
   predictions = model.predict(img_path)

Use Cases and Solutions
=======================

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Getting Started

   installation
   notebooks/why-yolort

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Tutorials

   notebooks/inference-pytorch-export-libtorch
   notebooks/comparison-between-yolort-vs-yolov5
   notebooks/how-to-align-with-ultralytics-yolov5
   notebooks/anchor-label-assignment-visualization
   notebooks/model-graph-visualization

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Deployment

   notebooks/export-onnx-inference-onnxruntime
   notebooks/onnx-graphsurgeon-inference-tensorrt
   notebooks/export-relay-inference-tvm

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   models
   yolov5
