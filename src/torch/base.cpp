#include "base.h"

void createTensor() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Tensor: " << tensor << std::endl;
}