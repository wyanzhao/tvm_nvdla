#include <nvdla_meta.h>
#include <nvdla_lib.h>

#include <onnc/ADT/StringList.h>
#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/CodeGen/BuildMemOperand.h>
#include <onnc/CodeGen/SetMemOperand.h>


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>

using namespace onnc;




int main(void)
{
uint32_t ret = 0;

onnc::Module module;
IRBuilder builder(module);

xTensorProto tensor;

std::ifstream input_fin("/home/dev/Workspace/tvm/tensor.pb");
tensor.ParseFromIstream(&input_fin);
const std::string &raw_data_str = tensor.raw_data();

ComputeGraph& cg = *builder.CreateComputeGraph("mxnet");

FloatTensor* t = cg.addValue<FloatTensor>("A");

Tensor* result = nullptr;

const size_t numElems = raw_data_str.size() / (sizeof(float)); 
float* d = (float*)raw_data_str.c_str(); 
t->getValues().resize(numElems); 
for (size_t i = 0; i < numElems; ++i) 
  t->getValues()[i] = d[i]; 

result = t;


return ret;
}