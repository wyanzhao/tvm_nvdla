
#include <NvDlaLib.h>

#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/Reshape.h>
#include <onnc/IR/Compute/MaxPool.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>

using namespace onnc;

int main(void)
{
    uint32_t ret = 0;

    auto p_module = new(onnc::Module);
    onnc::IRBuilder builder(*p_module);
    auto cg = builder.CreateComputeGraph("lenet_test");
    NvDlaLib* nvdla_lib = new NvDlaLib(p_module, cg);

    // Create Input.
    {
    cg->addOperator<InputOperator>()->setTensor(
    * (nvdla_lib->create_float_compute_tensor("data_0", {1, 1, 28, 28})));
    }
    
    // Conv1
    {
    // Create weights.
    nvdla_lib->create_float_weight_tensor_from_file("conv1_w_0", {32, 1, 5, 5}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/weight1.pb");
      // create nodes (layers)
    onnc::Conv* conv_op1 = nvdla_lib->create_compute_operator<Conv>({"data_0", "conv1_w_0"});
    // set strides
    auto strides = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op1->setStrides(std::move(strides));
    auto dilations = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op1->setDilations(std::move(dilations));
    auto kernel_shape = nvdla_lib->get_values<int64_t>({5, 5});
    conv_op1->setKernelShape(std::move(kernel_shape));
    auto pads = nvdla_lib->get_values<int64_t>({2, 2, 2, 2});
    conv_op1->setPads(std::move(pads));
    conv_op1->addOutput(*nvdla_lib->create_float_compute_tensor("output1", {1, 32, 28, 28}));
    }

    // Reshape1
    {
    nvdla_lib->create_float_weight_tensor_from_file("reshape_data1", {32}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/reshape1.pb");
    nvdla_lib->create_weight_operator<Int64Tensor>("reshape_dummy_1", {32, 1, 1});
    onnc::Reshape* reshape_op1 = nvdla_lib->create_compute_operator<Reshape>({"reshape_data1", "reshape_dummy_1"});
    reshape_op1->addOutput(*nvdla_lib->create_float_compute_tensor("reshaped1", {32, 1, 1}));
    }

    // Add1
    {
    onnc::Add* add_op1 = nvdla_lib->create_compute_operator<Add>({"output1", "reshaped1"});
    add_op1->addOutput(*nvdla_lib->create_float_compute_tensor("BiasAdd_1", {1, 32, 28, 28}));
    }

    // Relu1
    {
    onnc::Relu* relu_op1 = nvdla_lib->create_compute_operator<Relu>({"BiasAdd_1"});
    relu_op1->addOutput(*nvdla_lib->create_float_compute_tensor("Relu_1", {1, 32, 28, 28}));
    }

    // MaxPool1
    {
    onnc::MaxPool* maxpool_op1 = nvdla_lib->create_compute_operator<MaxPool>({"Relu_1"}, nvdla_lib->get_values<int64_t>({2, 2}));
    auto pads = nvdla_lib->get_values<int64_t>({0, 0, 0, 0});
    maxpool_op1->setPads(pads);
    auto strides = nvdla_lib->get_values<int64_t>({2, 2});
    maxpool_op1->setStrides(std::move(strides));
    auto kernel_shape = nvdla_lib->get_values<int64_t>({2, 2});
    maxpool_op1->setKernelShape(std::move(kernel_shape));

    maxpool_op1->addOutput(*nvdla_lib->create_float_compute_tensor("MaxPool_0", {1, 32, 14, 14}));
    }

    // Conv2
    {
    // Create weights.
    nvdla_lib->create_float_weight_tensor_from_file("conv2_w_0", {64, 32, 5, 5}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/weight2.pb");
      // create nodes (layers)
    onnc::Conv* conv_op2 = nvdla_lib->create_compute_operator<Conv>({"MaxPool_0", "conv2_w_0"});
    // set strides
    auto strides = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op2->setStrides(std::move(strides));
    auto dilations = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op2->setDilations(std::move(dilations));
    auto kernel_shape = nvdla_lib->get_values<int64_t>({5, 5});
    conv_op2->setKernelShape(std::move(kernel_shape));
    auto pads = nvdla_lib->get_values<int64_t>({2, 2, 2, 2});
    conv_op2->setPads(std::move(pads));
    conv_op2->addOutput(*nvdla_lib->create_float_compute_tensor("output2", {1, 64, 14, 14}));
    }

    // Reshape2
    {
    nvdla_lib->create_float_weight_tensor_from_file("reshape_data2", {64}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/reshape2.pb");
    nvdla_lib->create_weight_operator<Int64Tensor>("reshape_dummy_2", {64, 1, 1});
    onnc::Reshape* reshape_op2 = nvdla_lib->create_compute_operator<Reshape>({"reshape_data2", "reshape_dummy_2"});
    reshape_op2->addOutput(*nvdla_lib->create_float_compute_tensor("reshaped2", {64, 1, 1}));
    }

    // Add2
    {
    onnc::Add* add_op2 = nvdla_lib->create_compute_operator<Add>({"output2", "reshaped2"});
    add_op2->addOutput(*nvdla_lib->create_float_compute_tensor("BiasAdd_2", {1, 64, 14, 14}));
    }

    // Relu2
    {
    onnc::Relu* relu_op2 = nvdla_lib->create_compute_operator<Relu>({"BiasAdd_2"});
    relu_op2->addOutput(*nvdla_lib->create_float_compute_tensor("Relu_2", {1, 64, 14, 14}));
    }

    // MaxPool2
    {
    onnc::MaxPool* maxpool_op2 = nvdla_lib->create_compute_operator<MaxPool>({"Relu_2"}, nvdla_lib->get_values<int64_t>({2, 2}));
    auto pads = nvdla_lib->get_values<int64_t>({0, 0, 0, 0});
    maxpool_op2->setPads(pads);
    auto strides = nvdla_lib->get_values<int64_t>({2, 2});
    maxpool_op2->setStrides(std::move(strides));
    auto kernel_shape = nvdla_lib->get_values<int64_t>({2, 2});
    maxpool_op2->setKernelShape(std::move(kernel_shape));
    maxpool_op2->addOutput(*nvdla_lib->create_float_compute_tensor("MaxPool_1", {1, 64, 7, 7}));
    }

    // Conv3
    {
    // Create weights.
    nvdla_lib->create_float_weight_tensor_from_file("conv3_w_0", {1024, 64, 7, 7}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/weight3.pb");
      // create nodes (layers)
    onnc::Conv* conv_op3 = nvdla_lib->create_compute_operator<Conv>({"MaxPool_1", "conv3_w_0"});
    // set strides
    auto strides = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op3->setStrides(std::move(strides));
    auto dilations = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op3->setDilations(std::move(dilations));
    auto kernel_shape = nvdla_lib->get_values<int64_t>({7, 7});
    conv_op3->setKernelShape(std::move(kernel_shape));
    conv_op3->addOutput(*nvdla_lib->create_float_compute_tensor("output3", {1, 1024, 1, 1}));
    }

    // Reshape3
    {
    nvdla_lib->create_float_weight_tensor_from_file("reshape_data3", {1024}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/reshape3.pb");
    nvdla_lib->create_weight_operator<Int64Tensor>("reshape_dummy_3", {1024, 1, 1});
    onnc::Reshape* reshape_op3 = nvdla_lib->create_compute_operator<Reshape>({"reshape_data3", "reshape_dummy_3"});
    reshape_op3->addOutput(*nvdla_lib->create_float_compute_tensor("reshaped3", {1024, 1, 1}));
    }

    // Add3
    {
    onnc::Add* add_op3 = nvdla_lib->create_compute_operator<Add>({"output3", "reshaped3"});
    add_op3->addOutput(*nvdla_lib->create_float_compute_tensor("BiasAdd_3", {1, 1024, 1, 1}));
    }

    // Relu3
    {
    onnc::Relu* relu_op3 = nvdla_lib->create_compute_operator<Relu>({"BiasAdd_3"});
    relu_op3->addOutput(*nvdla_lib->create_float_compute_tensor("Relu_3", {1, 1024, 1, 1}));
    }

    // Conv4
    {
    // Create weights.
    nvdla_lib->create_float_weight_tensor_from_file("conv4_w_0", {10, 1024, 1, 1}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/weight4.pb");
      // create nodes (layers)
    onnc::Conv* conv_op4 = nvdla_lib->create_compute_operator<Conv>({"Relu_3", "conv4_w_0"});
    // set strides
    auto strides = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op4->setStrides(std::move(strides));
    auto dilations = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op4->setDilations(std::move(dilations));
    auto kernel_shape = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op4->setKernelShape(std::move(kernel_shape));
    conv_op4->addOutput(*nvdla_lib->create_float_compute_tensor("output4", {1, 10, 1, 1}));
    }

    // Reshape4
    {
    nvdla_lib->create_float_weight_tensor_from_file("reshape_data4", {10}, "/home/dev/Workspace/tvm/nvdla/python/lenet_data/reshape4.pb");
    nvdla_lib->create_weight_operator<Int64Tensor>("reshape_dummy_4", {10, 1, 1});
    onnc::Reshape* reshape_op4 = nvdla_lib->create_compute_operator<Reshape>({"reshape_data4", "reshape_dummy_4"});
    reshape_op4->addOutput(*nvdla_lib->create_float_compute_tensor("reshaped4", {10, 1, 1}));
    }

    // Add4
    {
    onnc::Add* add_op4 = nvdla_lib->create_compute_operator<Add>({"output4", "reshaped4"});
    add_op4->addOutput(*nvdla_lib->create_float_compute_tensor("Y", {1, 10, 1, 1}));
    }

    // Output of the model
    nvdla_lib->create_compute_operator<OutputOperator>({"Y"});

    nvdla_lib->optimize();
    nvdla_lib->compile();

    return ret;
}