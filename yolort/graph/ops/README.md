# XGraph operators

Inspired by and based on TVM Relay operators: https://docs.tvm.ai/langref/relay_op.html

## L0: Input and Other (Utility) Operators

| Operator      | Description   |
| ------------- | ------------- |
| Constant | Operator representing a constant tensor |
| Input | Operator representing a variable input tensor |
| StrInput | Operator representing a variable string input |
| Tuple | Operator representing a tuple of tensors |
| RelayOp | Generic Relay operator (can be used for importing unimplemented operators) |

## L1: Basic MLP (multi-layer perceptron) Operators

| Operator      | Description        |
| ------------- | ------------- |
| Add | Addition with numpy-style broadcasting |
| BiasAdd | Specific operator for addition of bias |
| Concat | Concatenation of input tensor along provided axes |
| Eltwise | Eltwise addition of input tensors |
| Dense | Dense/fully connected layer  |
| Divide | Division with numpy-style broadcasting |
| Dropout | Dropout layer |
| Exp | Elementwise exponential layer (Y = e^X) |
| ExpandDims | Expand dimensions at the given axis |
| Log | Elementwise natural logarithm |
| Multiply| Multiplication with numpy-style broadcasting |
| ReLU | Rectified Linear Unit, Y = max(X, 0) |
| rSqrt | Elementwise inverse of the square root, Y = 1 / sqrt(X) |
| Scale | Scale input tensor along axis and add bias |
| Sigmoid | Sigmoid nonlinearity,  Y = 1 / (1 + e^(-X)) |
| Sqrt | Elementwise square root |
| Sub | Subtraction with numpy-style broadcasting |
| Tanh | Hyperbolic tangent non-linearity, Y = (e^X - e^-X) / (e^X + e^-X) |

## L2: Convolution Related Operators

| Operator      | Description   |
| ------------- | ------------- |
| BatchNorm | Batch normalization operation |
| Convolution | General 2D convolution operation (also depthwise) |
| Conv2DTranspose | Transpose 2D convolution operation |
| Flatten | Flatten all input dimensions into one dimension except for batch dimension |
| Pad | Padding operation |
| Pooling | General pooling operation (both max and average) |

## L3: Other Math and Transformation Operators

| Operator      | Description   |
| ------------- | ------------- |
| Cast | Cast input tensor to the provided data type |
| Clip | Clip the input tensor between a specified minimum and maximum |
| pReLU | Parameterized ReLU operation, y = x > 0 ? x : alpha * x |
| Reshape | Reshape input tensor to specified shape |
| Squeeze | Squeeze input tensor along specified axes |
| Take | Slice input tensor according to specified axis and indices |
| Transpose | Transpose input tensor according to specified axes |

## L4: Broadcast and Reduction Operators

| Operator      | Description   |
| ------------- | ------------- |
| Mean | Take mean of input tensor along specified axes |


## L5: Vision Operators

| Operator      | Description   |
| ------------- | ------------- |
| Cvx | Operation for Cvx preprocessing (https://github.com/jtuyls/cvx) |
| YoloReorg | Shuffle and shape transform based on specified stride |

## L11: Quantization Operators

| Operator      | Description   |
| ------------- | ------------- |
| Quantize | Quantization of an input tensor based on threshold |
| QuantizeBias | Quantization of an input bias tensor |
| QuantizeInter | Requantize input tensor to new quantization threshold |
| QuantizeScaleBias | Quantization of an input bias tensor to a scaling layer |
| UnQuantize | Undo quantization an input tensor |
