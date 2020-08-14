#include <iostream>

#include <torchvision/models/mobilenet.h>

int main() {
  auto model = vision::models::MobileNet();
  model->eval();

  // Create a random input tensor and run it through the model.
  auto in = torch::rand({1, 3, 224, 224});
  auto out = model->forward(in);

  std::cout << out;
}
