#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    cv::VideoCapture cap(0);

    while(1)
    {
        cv::Mat frame;
        torch::Tensor tensor_image = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kByte);
        std::cout<<tensor_image<<std::endl;
        cap>>frame;
        //cap.read(frame);
        imshow("opencv", frame);
        waitKey(30);
    }

    return 0;

//    std::cout << "Hello, World!" << std::endl;
//    torch::jit::script::Module module;
//    try {
//        // Deserialize the ScriptModule from a file using torch::jit::load().
//        module = torch::jit::load("my_net.pt");
//    }
//    catch (const c10::Error& e) {
//        std::cerr << "error loading the model\n";
//        return -1;
//    }
//
//    auto image = torch::randn({2, 784});
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(image);
//
//    auto rst = module.forward(inputs).toTensor();
//
//    std::cout<<rst.sizes()<<std::endl;


}
