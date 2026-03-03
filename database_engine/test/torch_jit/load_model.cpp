#include "c10/core/Device.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

void load_model(std::string model_path) {
    try {
        torch::jit::script::Module module = torch::jit::load(model_path);
        torch::Tensor input = torch::rand({1, 128});

        auto output = module.forward({input}).toTensor();
        std::cout << output << std::endl;
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
    }
}
