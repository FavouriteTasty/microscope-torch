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

// 函数：将单个通道分割为128x128的块
std::vector<at::Tensor> split_channel_into_patches(const std::vector<uint8_t>& channel_data, int width, int height, int patch_size) {
    std::vector<at::Tensor> patches;
    int num_patches_x = width / patch_size;
    int num_patches_y = height / patch_size;

    for (int y = 0; y < num_patches_y; ++y) {
        for (int x = 0; x < num_patches_x; ++x) {
            // 提取块
            std::vector<float> patch;
            patch.reserve(patch_size * patch_size);
            for (int dy = 0; dy < patch_size; ++dy) {
                for (int dx = 0; dx < patch_size; ++dx) {
                    int idx = ((y * patch_size + dy) * width) + (x * patch_size + dx);
                    patch.push_back(static_cast<float>(channel_data[idx]) / 255.0f); // 归一化到[0,1]
                }
            }
            // 创建张量，形状为 [1, 128, 128]
            at::Tensor patch_tensor = torch::from_blob(patch.data(), {1, patch_size, patch_size}, torch::kFloat32).clone();
            patches.push_back(patch_tensor);
        }
    }

    return patches;
}

// 函数：从块还原单个通道（上采样为256x256）
std::vector<uint8_t> reconstruct_channel_from_patches(const std::vector<at::Tensor>& patches, int width, int height, int patch_size, int upscale_factor) {
    int upscaled_width = width * upscale_factor;
    int upscaled_height = height * upscale_factor;
    std::vector<uint8_t> channel_data(upscaled_width * upscaled_height, 0);

    int num_patches_x = width / patch_size;
    int num_patches_y = height / patch_size;

    int patch_idx = 0;
    for (int y = 0; y < num_patches_y; ++y) {
        for (int x = 0; x < num_patches_x; ++x) {
            if (patch_idx >= patches.size()) {
                std::cerr << "块索引超出范围！\n";
                break;
            }
            const at::Tensor& patch = patches[patch_idx];
            patch_idx++;

            // 假设patch形状为 (256, 256)
            auto patch_accessor = patch.accessor<float, 2>();
            for (int dy = 0; dy < patch.size(0); ++dy) {
                for (int dx = 0; dx < patch.size(1); ++dx) {
                    int img_x = x * patch.size(1) + dx;
                    int img_y = y * patch.size(0) + dy;
                    if (img_x >= upscaled_width || img_y >= upscaled_height) {
                        continue; // 防止越界
                    }
                    float val = patch_accessor[dy][dx] * 255.0f;
                    val = std::clamp(val, 0.0f, 255.0f);
                    channel_data[img_y * upscaled_width + img_x] = static_cast<uint8_t>(val);
                }
            }
        }
    }

    return channel_data;
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    const char* input_image_path = "../very-small-input.jpg";
    const char* output_image_path = "../split-output.jpg";

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

    // 填充图像，使尺寸可被128整除
    const int patch_size = 128;
    pad_image(image_data, width, height, channels, patch_size);
    std::cout << "填充后的图像宽度: " << width << ", 高度: " << height << "\n";

    // 分离各个通道的数据
    std::vector<std::vector<uint8_t>> channels_data(3, std::vector<uint8_t>(width * height, 0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                channels_data[c][y * width + x] = image_data[(y * width + x) * 3 + c];
            }
        }
    }

    // 将每个通道分割为块
    std::vector<std::vector<at::Tensor>> all_patches(3);
    for (int c = 0; c < 3; ++c) {
        all_patches[c] = split_channel_into_patches(channels_data[c], width, height, patch_size);
        std::cout << "通道 " << c << " 拥有 " << all_patches[c].size() << " 个块。\n";
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
    device = torch::Device(torch::kMPS);
    std::cout << "使用MPS设备。\n";
    module.to(device);
    module.eval(); // 设置模型为评估模式

    // 分别处理每个通道
    std::vector<std::vector<at::Tensor>> processed_patches(3);
    const int upscale_factor = 2; // 上采样因子为2
    for (int c = 0; c < 3; ++c) {
        const auto& patches = all_patches[c];
        int num_patches = patches.size();

        std::cout << "开始处理通道: " << c << " 输入 batch形状: " << num_patches << "\n";

        // 创建一个批次张量
        // 形状: (批次大小, 1, 128, 128)
        at::Tensor input_batch = torch::zeros({num_patches, 1, patch_size, patch_size}, torch::kFloat32);
        for (int i = 0; i < num_patches; ++i) {
            input_batch[i][0] = patches[i][0];
        }

        // 打印输入张量形状
        std::cout << "通道 " << c << " 输入张量形状: " << input_batch.sizes() << "\n";

        input_batch = input_batch.to(device);

        // 准备输入
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_batch);

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

        // 假设输出形状为 (num_patches, 1, 256, 256)
        // 如果不同，请相应调整
        if (output.dim() != 4 || output.size(0) != num_patches || output.size(1) != 1 ||
            output.size(2) != patch_size * upscale_factor || output.size(3) != patch_size * upscale_factor) {
            std::cerr << "输出张量形状与预期不符: " << output.sizes() << "\n";
            return -1;
        }

        // 分离并移动到CPU
        output = output.to(torch::kCPU);

        // 将输出拆分为单独的块
        for (int i = 0; i < num_patches; ++i) {
            // 假设输出形状为 (num_patches, 1, 256, 256)
            at::Tensor patch = output[i][0].clone(); // 克隆以拥有数据，形状: (256, 256)
            processed_patches[c].push_back(patch);
        }

        std::cout << "处理完通道 " << c << "\n";
    }

    // 从处理后的块还原各个通道（上采样）
    std::vector<std::vector<uint8_t>> reconstructed_channels(3);
    for (int c = 0; c < 3; ++c) {
        reconstructed_channels[c] = reconstruct_channel_from_patches(processed_patches[c], width, height, patch_size, upscale_factor);
        std::cout << "还原完通道 " << c << "\n";
    }

    // 将各个通道合并为单一图像（上采样后的图像）
    int upscaled_width = width * upscale_factor;
    int upscaled_height = height * upscale_factor;
    std::vector<uint8_t> output_image_data(upscaled_width * upscaled_height * 3, 0);
    for (int y = 0; y < upscaled_height; ++y) {
        for (int x = 0; x < upscaled_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                output_image_data[(y * upscaled_width + x) * 3 + c] = reconstructed_channels[c][y * upscaled_width + x];
            }
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
