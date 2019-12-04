
#include <NvDlaLib.h>

#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/Conv.h>
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

    onnc::Module module;
    onnc::IRBuilder builder(module);
    NvDlaLib* nvdla_lib = new NvDlaLib();

    auto& cg = *builder.CreateComputeGraph("conv_test");

  // Create Input.
    cg.addOperator<InputOperator>()->setTensor(
    * (nvdla_lib->create_float_compute_tensor(cg, "data_0", {1, 1, 12, 8})));

    // Create weights.
    nvdla_lib->create_float_weight_operator(cg, "conv1_w_0", {13, 1, 1, 1}, "/home/dev/Workspace/tvm/nvdla/tensor1.pb");

    // create nodes (layers)
    onnc::Conv* conv_op1 = nvdla_lib->create_compute_operator<Conv>(cg,    {"data_0", "conv1_w_0"});
    // set strides
    {
      auto v = nvdla_lib->get_values<int64_t>({1, 1});
      conv_op1->setStrides(std::move(v));
    }

    conv_op1->addOutput(*nvdla_lib->create_float_compute_tensor(cg, "output1", {1, 13, 12, 8}));

    nvdla_lib->create_float_weight_operator(cg, "conv2_w_0", {8, 13, 2, 8}, "/home/dev/Workspace/tvm/nvdla/tensor2.pb");
    onnc::Conv* conv_op2 = nvdla_lib->create_compute_operator<Conv>(cg,    {"output1", "conv2_w_0"});
        //set conv2 attributes
    {
      auto dialations = nvdla_lib->get_values<int64_t>({1, 1});
      conv_op2->setDilations(dialations);

      auto groups = IntAttr(1);
      conv_op2->setGroup(groups);

      auto kernel_shape = nvdla_lib->get_values<int64_t>({2, 8});
      conv_op2->setKernelShape(kernel_shape);

      auto pads = nvdla_lib->get_values<int64_t>({0, 0, 0, 0});
      conv_op2->setPads(pads);

      auto strides = nvdla_lib->get_values<int64_t>({1, 1});
      conv_op1->setStrides(std::move(strides));
    }

    conv_op2->addOutput(*nvdla_lib->create_float_compute_tensor(cg, "output2", {1, 8, 11, 1}));
    nvdla_lib->create_compute_operator<OutputOperator>(cg, {"output2"});


    nvdla_lib->compile(module, cg);

    return ret;
}