#ifndef TFLITE_POLICY_HPP_
#define TFLITE_POLICY_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <string>
#include <iostream>

// TensorFlow Lite headers (REQUIRED)
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Constants from model_info.txt (50Hz version for track_test)
constexpr int BUFFER_SIZE = 50;        // 动作-状态缓冲区大小（50Hz版本）
constexpr int ACTION_DIM = 4;          // 动作维度（四旋翼控制指令）
constexpr int AUX_OUTPUT_DIM = 3;      // 辅助输出维度（目标速度预测，机体系）
constexpr int TOTAL_OUTPUT_DIM = ACTION_DIM + AUX_OUTPUT_DIM;  // 总输出维度 = 7
constexpr int OBS_DIM = 9;             // 观测维度（机体系速度3 + 机体系重力3 + 机体系目标位置3）
constexpr int INPUT_DIM = BUFFER_SIZE * (ACTION_DIM + OBS_DIM);  // 总输入维度 = 50*(4+9) = 650

/**
 * 动作-状态缓冲区类
 * 维护最近BUFFER_SIZE个时间步的观测和动作
 */
class ActionObsBuffer {
public:
    ActionObsBuffer(int buffer_size, int obs_dim, int action_dim)
        : buffer_size_(buffer_size), obs_dim_(obs_dim), action_dim_(action_dim) {
        // 初始化缓冲区为全零
        for (int i = 0; i < buffer_size_; ++i) {
            obs_buffer_.push_back(std::vector<float>(obs_dim_, 0.0f));
            action_buffer_.push_back(std::vector<float>(action_dim_, 0.0f));
        }
    }
    
    // 更新缓冲区（添加新的观测和动作）
    void update(const std::vector<float>& obs, const std::vector<float>& action) {
        // 移除最旧的数据
        obs_buffer_.pop_front();
        action_buffer_.pop_front();
        
        // 添加最新的数据
        obs_buffer_.push_back(obs);
        action_buffer_.push_back(action);
    }
    
    // 只更新最后一个entry的action部分（obs不变）
    // 用于：先用empty_action推理后，用真实action替换empty_action
    void update_last(const std::vector<float>& action) {
        if (!action_buffer_.empty()) {
            action_buffer_.back() = action;
        }
    }
    
    // 获取展平的缓冲区数据作为网络输入
    // 注意：顺序必须与训练时一致！训练代码中是 [action, obs]
    std::vector<float> get_flattened_buffer() const {
        std::vector<float> flattened;
        flattened.reserve(buffer_size_ * (action_dim_ + obs_dim_));
        
        for (int i = 0; i < buffer_size_; ++i) {
            // 先添加动作（与训练代码一致）
            flattened.insert(flattened.end(), action_buffer_[i].begin(), action_buffer_[i].end());
            // 后添加观测
            flattened.insert(flattened.end(), obs_buffer_[i].begin(), obs_buffer_[i].end());
        }
        
        return flattened;
    }
    
    // 重置缓冲区
    void reset() {
        for (auto& obs : obs_buffer_) {
            std::fill(obs.begin(), obs.end(), 0.0f);
        }
        for (auto& action : action_buffer_) {
            std::fill(action.begin(), action.end(), 0.0f);
        }
    }

private:
    int buffer_size_;
    int obs_dim_;
    int action_dim_;
    std::deque<std::vector<float>> obs_buffer_;
    std::deque<std::vector<float>> action_buffer_;
};

/**
 * TFLite策略推理器类
 */
class TFLitePolicyInference {
public:
    TFLitePolicyInference(const std::string& model_path)
        : buffer_(BUFFER_SIZE, OBS_DIM, ACTION_DIM), 
          last_action_(ACTION_DIM, 0.0f),
          initialized_(false),
          is_new_inference_(false) {
        
        // 加载模型
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model_) {
            std::cerr << "[TFLite] Failed to load model from: " << model_path << std::endl;
            return;
        }
        
        // 创建解释器
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model_, resolver);
        builder(&interpreter_);
        
        if (!interpreter_) {
            std::cerr << "[TFLite] Failed to create interpreter" << std::endl;
            return;
        }
        
        // 分配张量
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::cerr << "[TFLite] Failed to allocate tensors" << std::endl;
            return;
        }
        
        initialized_ = true;
        std::cout << "[TFLite] Model loaded successfully from: " << model_path << std::endl;
        print_model_info();
    }
    
    /**
     * 获取动作和辅助输出
     * @param obs 当前观测
     * @return 完整输出 [action(4), aux_output(3)] = 7维
     * 
     * 注意：逻辑必须与训练代码 bptt.py 一致！
     * 训练时的步骤：
     * 先用【空动作+新观测】更新buffer → 推理 → 用【新动作+新观测】更新buffer
     */
    std::vector<float> get_action_and_aux(const std::vector<float>& obs) {
        if (!initialized_) {
            return std::vector<float>(TOTAL_OUTPUT_DIM, 0.0f);
        }
        
        // 步骤1：先用【空动作 + 新观测】临时更新buffer（用于推理）
        // 这与训练代码一致：先用empty_action + obs更新buffer再推理
        std::vector<float> empty_action(ACTION_DIM, 0.0f);
        buffer_.update(obs, empty_action);
        
        // 步骤2：用临时buffer获取新输出（推理），返回完整7维输出
        std::vector<float> full_output = compute_action_and_aux();  // 返回7维
        
        // 提取动作部分（前4维）
        last_action_.assign(full_output.begin(), full_output.begin() + ACTION_DIM);
        
        // 步骤3：用【新动作 + 新观测】更新buffer（真正保存）
        // 替换刚才的empty_action为真实action
        buffer_.update_last(last_action_);
        
        is_new_inference_ = true;  // 标记为新推理
        
        return full_output;  // 返回完整7维输出
    }
    
    /**
     * 获取动作（仅返回前4维，保持向后兼容）
     * @param obs 当前观测
     * @return 控制动作（4维）
     */
    std::vector<float> get_action(const std::vector<float>& obs) {
        std::vector<float> full_output = get_action_and_aux(obs);
        // 只返回前4维动作
        return std::vector<float>(full_output.begin(), full_output.begin() + ACTION_DIM);
    }
    
    /**
     * 检查上次get_action是否进行了新的推理
     */
    bool is_new_inference() const {
        return is_new_inference_;
    }
    
    /**
     * 重置推理器状态
     * @param initial_obs 初始观测（用于填充buffer）
     * @param hovering_action 悬停动作（归一化后的，默认为[0,0,0,0]）
     * 
     * 注意：必须与训练代码一致！
     * 训练时用归一化的悬停动作和当前观测填充buffer
     */
    void reset(const std::vector<float>& initial_obs, 
               const std::vector<float>& hovering_action = std::vector<float>(ACTION_DIM, 0.0f)) {
        // 清空buffer并用悬停动作+初始观测填充（与训练代码一致）
        buffer_.reset();  // 先清空
        
        // 用悬停动作和初始观测填充整个buffer（50个历史步）
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            buffer_.update(initial_obs, hovering_action);
        }
        
        last_action_ = hovering_action;  // 初始动作设为悬停动作
        is_new_inference_ = false;
    }
    
    /**
     * 检查是否已成功初始化
     */
    bool is_initialized() const {
        return initialized_;
    }
    
    /**
     * 获取当前展平的缓冲区数据（用于调试）
     * @return 展平的缓冲区数据 [action(4), obs(9)] * BUFFER_SIZE (650 dims)
     */
    std::vector<float> get_flattened_buffer() const {
        return buffer_.get_flattened_buffer();
    }

private:
    /**
     * 计算动作和辅助输出（实际推理）
     * 使用当前buffer进行推理（buffer已经包含了obs）
     * @return 完整输出 [action(4), aux_output(3)] = 7维
     */
    std::vector<float> compute_action_and_aux() {
        // 获取展平的缓冲区数据
        std::vector<float> input_data = buffer_.get_flattened_buffer();
        
        // 获取输入张量
        float* input = interpreter_->typed_input_tensor<float>(0);
        
        // 复制输入数据
        std::copy(input_data.begin(), input_data.end(), input);
        
        // 推理
        if (interpreter_->Invoke() != kTfLiteOk) {
            std::cerr << "[TFLite] Failed to invoke interpreter" << std::endl;
            return std::vector<float>(TOTAL_OUTPUT_DIM, 0.0f);
        }
        
        // 获取输出（7维）
        float* output = interpreter_->typed_output_tensor<float>(0);
        
        // 复制输出数据（完整7维）
        std::vector<float> full_output(output, output + TOTAL_OUTPUT_DIM);
        
        return full_output;
    }
    
    /**
     * 计算动作（实际推理，仅返回前4维，保持向后兼容）
     * 使用当前buffer进行推理（buffer已经包含了obs）
     * @return 控制动作（4维）
     */
    std::vector<float> compute_action() {
        std::vector<float> full_output = compute_action_and_aux();
        // 只返回前4维动作
        return std::vector<float>(full_output.begin(), full_output.begin() + ACTION_DIM);
    }
    
    /**
     * 打印模型信息
     */
    void print_model_info() {
        std::cout << "[TFLite] Model Information:" << std::endl;
        
        // 输入信息
        const auto* input_tensor = interpreter_->input_tensor(0);
        std::cout << "  Input Tensor: ";
        for (int i = 0; i < input_tensor->dims->size; ++i) {
            std::cout << input_tensor->dims->data[i];
            if (i < input_tensor->dims->size - 1) std::cout << " x ";
        }
        std::cout << std::endl;
        
        // 输出信息
        const auto* output_tensor = interpreter_->output_tensor(0);
        std::cout << "  Output Tensor: ";
        for (int i = 0; i < output_tensor->dims->size; ++i) {
            std::cout << output_tensor->dims->data[i];
            if (i < output_tensor->dims->size - 1) std::cout << " x ";
        }
        std::cout << std::endl;
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    ActionObsBuffer buffer_;
    std::vector<float> last_action_;
    bool initialized_;
    bool is_new_inference_;  // 标记上次get_action是否进行了新推理
};

#endif  // TFLITE_POLICY_HPP_



