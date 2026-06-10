#include <iostream>
#include <dlfcn.h>
#include <torch/torch.h>
#include <torch/script.h> 



torch::Device device(torch::kPrivateUse1, 0);    /* backend : torch::kPrivateUse1, torch::kCPU  */


void load_device() {
    const char* lib_path = "<path-to-pt_vk.so>";
    /* load dynamic library */
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Failed to load " << lib_path << ": " << dlerror() << std::endl;
        exit(1);
    }
    std::cout << "Dynamic library loaded successfully: " << lib_path << std::endl;
}


void infer_net() {
    /* create dummy input */
    torch::Tensor input = torch::randn({1, 1, 28, 28}).to(device);
    std::cout << "Input : " << input << std::endl;

    /* load scripted module */
    std::string scripted_model = "<path-to-mnist_cnn-scripted.pt>";
    std::cout << "Loading scripted module: " << scripted_model << std::endl;
    auto module = torch::jit::load(scripted_model);
    module.to(device);
    module.eval();

    /* inference */
    auto out = module.forward({input}).toTensor();
    auto probs = torch::exp(out);
    auto top = std::get<1>(probs.max(1));
    std::cout << "Predicted class (scripted): " << top.item<int>() << std::endl;
}


int main() {
    load_device();
    infer_net();
    return 0;
} 
