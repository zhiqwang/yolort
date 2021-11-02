What is yolort?
====================

Runtime stack for YOLOv5 on specialized accelerators such as libtorch, onnxruntime, tvm and ncnn.

.. image:: _static/yolort_logo.png
    :width: 500px
    :align: center


`yolort` focus on making the training and inference of the object detection integrate more seamlessly
together. `yolort` now adopts the same model structure as the official YOLOv5. The significant
difference is that we adopt the dynamic shape mechanism, and within this, we can embed both
pre-processing (`letterbox`) and post-processing (`nms`) into the model graph, which simplifies
the deployment strategy. In this sense, `yolort` makes it possible to be deployed more friendly
on `LibTorch`, `ONNXRuntime`, `TVM` and so on.
