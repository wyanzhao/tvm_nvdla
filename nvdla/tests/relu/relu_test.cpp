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

    auto p_module = new(onnc::Module);
    onnc::IRBuilder builder(*p_module);
    auto cg = builder.CreateComputeGraph("relu_test");
    NvDlaLib* nvdla_lib = new NvDlaLib(p_module, cg);
    
  // Create Input.
    cg->addOperator<InputOperator>()->setTensor(
    * (nvdla_lib->create_float_compute_tensor("data0", {1, 1, 3, 3})));
  
    auto op2 = nvdla_lib->create_compute_operator<Relu>({"data0"});

    op2->addOutput(*nvdla_lib->create_float_compute_tensor("activation0", {1, 1, 3, 3}));


    nvdla_lib->create_compute_operator<OutputOperator>({"activation0"});

    nvdla_lib->optimize();
    nvdla_lib->compile();
    return ret;
}