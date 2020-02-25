#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

int main() {
    std::cout << "Hello, World!" << std::endl;
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("my_net.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    auto image = torch::randn({2, 784});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);

    auto rst = module.forward(inputs).toTensor();

    std::cout<<rst.sizes()<<std::endl;

    return 0;
}
