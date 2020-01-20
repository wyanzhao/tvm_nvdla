
#include <NvDlaLib.h>

#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/Add.h>
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
    auto cg = builder.CreateComputeGraph("add_test");
    NvDlaLib* nvdla_lib = new NvDlaLib(p_module, cg);

    
  // Create Input.
    cg->addOperator<InputOperator>()->setTensor(
    * (nvdla_lib->create_float_compute_tensor("data_0", {1, 1, 3, 3})));

    // Create weights.
    nvdla_lib->create_float_weight_tensor_from_file("conv1_w_0", {2, 1, 1, 1}, "/home/dev/Workspace/tvm/nvdla/tensor1.pb");
    nvdla_lib->create_float_weight_tensor_from_file("conv2_w_0", {2, 1, 1, 1}, "/home/dev/Workspace/tvm/nvdla/tensor2.pb");

    {
    // create nodes (layers)
    onnc::Conv* conv_op1 = nvdla_lib->create_compute_operator<Conv>({"data_0", "conv1_w_0"});
    // set strides
    auto v = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op1->setStrides(std::move(v));
    conv_op1->addOutput(*nvdla_lib->create_float_compute_tensor("output1", {1, 2, 3, 3}));
    } 

    {
    onnc::Conv* conv_op2 = nvdla_lib->create_compute_operator<Conv>({"data_0", "conv2_w_0"});
    auto v = nvdla_lib->get_values<int64_t>({1, 1});
    conv_op2->setStrides(std::move(v));
    conv_op2->addOutput(*nvdla_lib->create_float_compute_tensor("output2", {1, 2, 3, 3}));
    }
    
    {
    onnc::Add* add_op = nvdla_lib->create_compute_operator<Add>({"output1", "output2"});
    add_op->addOutput(*nvdla_lib->create_float_compute_tensor("Y", {1, 2, 3, 3}));
    }

    nvdla_lib->create_compute_operator<OutputOperator>({"Y"});

    nvdla_lib->optimize();
    nvdla_lib->compile();

    return ret;
}