#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <torch/script.h> // 一站式头文件
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

// 函数：填充图像，使其尺寸可被patch_size整除
void pad_image(std::vector<uint8_t>& image, int& width, int& height, int channels, int patch_size) {
    int padded_width = std::ceil(static_cast<float>(width) / patch_size) * patch_size;
    int padded_height = std::ceil(static_cast<float>(height) / patch_size) * patch_size;

    if (padded_width == width && padded_height == height) {
        return; // 无需填充
    }

    std::vector<uint8_t> padded_image(padded_width * padded_height * channels, 0); // 初始化为0（黑色）

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                padded_image[(y * padded_width + x) * channels + c] = image[(y * width + x) * channels + c];
            }
        }
    }

    image = std::move(padded_image);
    width = padded_width;
    height = padded_height;
}

int main(int argc, char* argv[]) {

    auto start = std::chrono::high_resolution_clock::now();

    const char* input_image_path = "../small-input.jpg";
    const char* output_image_path = "../whole-output.jpg";

    // 使用stb_image加载图像
    int width, height, channels;
    // 强制加载为RGB
    uint8_t* img = stbi_load(input_image_path, &width, &height, &channels, 3);
    if (!img) {
        std::cerr << "加载图像失败: " << input_image_path << "\n";
        return -1;
    }
    channels = 3; // 因为强制为3通道

    std::cout << "加载图像，宽度: " << width << ", 高度: " << height << ", 通道数: " << channels << "\n";

    // 将图像数据转换为vector，便于操作
    std::vector<uint8_t> image_data(img, img + (width * height * channels));
    stbi_image_free(img);

    // 填充图像，使尺寸可被patch_size整除（如果需要）
    const int patch_size = 128; // 可以保留，尽管现在不切分
    pad_image(image_data, width, height, channels, patch_size);
    std::cout << "填充后的图像宽度: " << width << ", 高度: " << height << "\n";

    // 分离各个通道的数据
    std::vector<std::vector<uint8_t>> channels_data(3, std::vector<uint8_t>(width * height, 0));
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < width * height; ++i) {
            channels_data[c][i] = image_data[i * channels + c];
        }
    }

    // 加载PyTorch模型
    torch::jit::Module module;
    try {
        module = torch::jit::load("../weights.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "加载模型时出错\n";
        return -1;
    }

    // 设置设备为MPS（如果可用），否则使用CPU
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "使用CUDA设备。\n";
    }
    device = torch::Device(torch::kMPS);
    std::cout << "使用MPS设备。\n";
    module.to(device);
    module.eval(); // 设置模型为评估模式

    // 处理每个通道
    std::vector<std::vector<uint8_t>> reconstructed_channels(3);
    const int upscale_factor = 2; // 上采样因子为2
    for (int c = 0; c < 3; ++c) {
        const auto& channel = channels_data[c];
        int upscaled_width = width * upscale_factor;
        int upscaled_height = height * upscale_factor;

        std::cout << "开始处理通道: " << c << "\n";

        // 创建一个输入张量，形状为 (1, 1, height, width)
        std::vector<float> channel_float(channel.begin(), channel.end());
        std::vector<float> normalized_channel;
        normalized_channel.reserve(channel_float.size());
        for (auto val : channel_float) {
            normalized_channel.push_back(static_cast<float>(val) / 255.0f); // 归一化到[0,1]
        }

        at::Tensor input_tensor = torch::from_blob(normalized_channel.data(), {1, 1, height, width}, torch::kFloat32).clone();
        input_tensor = input_tensor.to(device);

        // 准备输入
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // 前向传播
        at::Tensor output;
        try {
            torch::NoGradGuard no_grad; // 禁用梯度计算
            output = module.forward(inputs).toTensor();
        }
        catch (const c10::Error& e) {
            std::cerr << "通道 " << c << " 的模型推理时出错\n";
            return -1;
        }

        // 打印输出张量形状
        std::cout << "通道 " << c << " 输出张量形状: " << output.sizes() << "\n";

        // 检查输出形状是否符合预期
        if (output.dim() != 4 || output.size(0) != 1 || output.size(1) != 1 ||
            output.size(2) != height * upscale_factor || output.size(3) != width * upscale_factor) {
            std::cerr << "输出张量形状与预期不符: " << output.sizes() << "\n";
            return -1;
        }

        // 将输出移动到CPU并转换为uint8
        output = output.to(torch::kCPU);
        output = output.squeeze(); // 移除批次和通道维度，变为 (upscaled_height, upscaled_width)
        output = output.clamp(0, 1).mul(255).to(torch::kU8);
        auto output_accessor = output.accessor<uint8_t, 2>();

        // 将张量数据复制到重建的通道中
        reconstructed_channels[c].resize(upscaled_width * upscaled_height);
        for (int y = 0; y < upscaled_height; ++y) {
            for (int x = 0; x < upscaled_width; ++x) {
                reconstructed_channels[c][y * upscaled_width + x] = output_accessor[y][x];
            }
        }

        std::cout << "处理完通道 " << c << "\n";
    }

    // 将各个通道合并为单一图像（上采样后的图像）
    int upscaled_width = width * upscale_factor;
    int upscaled_height = height * upscale_factor;
    std::vector<uint8_t> output_image_data(upscaled_width * upscaled_height * 3, 0);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < upscaled_width * upscaled_height; ++i) {
            output_image_data[i * 3 + c] = reconstructed_channels[c][i];
        }
    }

    // 使用stb_image_write保存还原后的图像
    if (!stbi_write_jpg(output_image_path, upscaled_width, upscaled_height, 3, output_image_data.data(), 100)) {
        std::cerr << "保存图像失败: " << output_image_path << "\n";
        return -1;
    }

    std::cout << "已保存处理后的图像到 " << output_image_path << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "代码块运行时间: " << duration.count() << " 毫秒" << std::endl;

    return 0;
}
