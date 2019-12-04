#include <NvDlaLib.h>

#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>


int main(void)
{
    uint32_t ret = 0;

    onnc::Module module;
    onnc::IRBuilder builder(module);
    NvDlaLib* nvdla_lib = new NvDlaLib();

    auto& cg = *builder.CreateComputeGraph("relu_test");
  // Create Input.
    cg.addOperator<InputOperator>()->setTensor(
    * (nvdla_lib->create_float_compute_tensor(cg, "data0", {1, 1, 3, 3})));
  
    auto op2 = nvdla_lib->create_compute_operator<Relu>(cg,    {"data0"});

    op2->addOutput(*nvdla_lib->create_float_compute_tensor(cg, "activation0", {1, 1, 3, 3}));


    nvdla_lib->create_compute_operator<OutputOperator>(cg, {"activation0"});

    nvdla_lib->compile(module, cg);
    return ret;
}