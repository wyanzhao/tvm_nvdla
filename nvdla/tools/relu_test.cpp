#include <nvdla_meta.h>
#include <nvdla_lib.h>
#include <onnc.h>
#include <nvdla_op.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>

//===----------------------------------------------------------------------===//
// Create Compute Graph Helper
//===----------------------------------------------------------------------===//
template<typename TensorTy>
static Tensor* CreateComputeTensor(ComputeGraph& pCG, const StringRef& pName,
                                   const Tensor::Dimensions& pDims)
{
  Tensor* t = pCG.addValue<TensorTy>(pName);
  t->setDimensions(pDims);
  return t;
}

template<typename OpTy, typename ... NodeCtorParams>
OpTy* CreateComputeOperator(onnc::ComputeGraph& pCG,
                                   const onnc::StringList& pInputNames,
                                   NodeCtorParams&& ... pParams)
{
  OpTy* op = pCG.addOperator<OpTy>(pParams...);
  for (auto& iname : pInputNames)
    op->addInput(*pCG.getValue<Tensor>(iname));
  return op;
}

// Create Compute Graph Helper
//===----------------------------------------------------------------------===//
onnc::Tensor*
CreateFloatComputeTensor(onnc::ComputeGraph& pCG, const onnc::StringRef& pName,
                         const onnc::Tensor::Dimensions& pDims)
{
  return CreateComputeTensor<FloatTensor>(pCG, pName, pDims);
}

onnc::Initializer *
CreateFloatWeightOperator(onnc::ComputeGraph& pCG, const std::string& pName,
                          const onnc::Tensor::Dimensions& pDims)
{
  //CreateWeightOperator<FloatTensor>(pCG, pName, pDims);
  Initializer* init = pCG.addOperator<Initializer>(pName);
  
  //Tensor* value = CreateComputeTensor<TensorTy>(pCG, pName, pDims);
  FloatTensor* t = pCG.addValue<FloatTensor>(pName);
  
  xTensorProto tensor;

  std::ifstream input_fin("/home/dev/Workspace/tvm/tensor.pb");
  tensor.ParseFromIstream(&input_fin);
  const std::string &raw_data_str = tensor.raw_data();

  const size_t numElems = raw_data_str.size() / (sizeof(float)); 
  float* d = (float*)raw_data_str.c_str(); 
  t->getValues().resize(numElems); 
  for (size_t i = 0; i < numElems; ++i) 
    t->getValues()[i] = d[i]; 
  
  t->setDimensions(pDims);
  init->setTensor(*t);

  return init;
}


int main(void)
{
    uint32_t ret = 0;

    onnc::Module module;
    onnc::IRBuilder builder(module);

    onnc::ComputeGraph& cg = *builder.CreateComputeGraph("mxnet");

  // Create Input.
    cg.addOperator<InputOperator>()->setTensor(
    *CreateFloatComputeTensor(cg, "data0", {1, 1, 3, 3}));
  
    auto op2 = CreateComputeOperator<Relu>(cg,    {"data0"});

    op2->addOutput(*CreateFloatComputeTensor(cg, "activation0", {1, 1, 3, 3}));

    CreateComputeOperator<OutputOperator>(cg, {"activation0"});

    createMemOperandsOfGraph(cg);
    setMemOperand(module);

    NvDlaBackendMeta* pMeta = new (NvDlaBackendMeta);
    init_nvdla_memory(module, pMeta);

    ComputeGraph::iterator nodeIt, nEnd = cg.end();
    for (nodeIt = cg.begin(); nodeIt != nEnd; ++nodeIt) {
        const onnc::ComputeOperator *node = nodeIt;
        std::cout<< node->name() << std::endl;
        
        if(node->name() == "Relu"){
            relu(* (Relu *) node, pMeta);
        }
  }
    task_submit(pMeta);
    nvdla_filegen(pMeta);
    
    return ret;
}