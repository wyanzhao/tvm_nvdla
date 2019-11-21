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
#include <memory>
#include <fstream>
#include <string>

using namespace onnc;



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

static Tensor*
CreateFloatComputeTensor(ComputeGraph& pCG, const StringRef& pName,
                         const Tensor::Dimensions& pDims)
{
  return CreateComputeTensor<FloatTensor>(pCG, pName, pDims);
}

static onnc::Initializer *
CreateFloatWeightOperator(ComputeGraph& pCG, const std::string& pName,
                          const Tensor::Dimensions& pDims)
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

template<typename OpTy, typename ... NodeCtorParams>
static OpTy* CreateComputeOperator(ComputeGraph& pCG,
                                   const StringList& pInputNames,
                                   NodeCtorParams&& ... pParams)
{
  OpTy* op = pCG.addOperator<OpTy>(pParams...);
  for (auto& iname : pInputNames)
    op->addInput(*pCG.getValue<Tensor>(iname));
  return op;
}

void createMemOperandsOfNode(ComputeGraph& pCG,
                                              ComputeOperator& pNode,
                                              ComputeOperand::Residence pResd)
{
  // Create memory operand for each output value
  for (unsigned i = 0; i < pNode.getNumOfOutputs(); ++i) {
    onnc::Value* value = pNode.getOutput(i);
    for (auto& use : value->getUses()) {
      pCG.addOperand<ComputeMemOperand>(pNode, *use.getUser(),
                                        *value, pResd);
    }
  }
}


void createMemOperandsOfGraph(ComputeGraph& pCG)
{
  ComputeGraph::iterator nodeIt, nEnd = pCG.end();
  for (nodeIt = pCG.begin(); nodeIt != nEnd; ++nodeIt) {
    ComputeOperator *node = nodeIt;
    ComputeOperand::Residence resd;
    if (isa<Initializer>(node))
      resd = ComputeOperand::kWeightResidence;
    else if (isa<InputOperator>(node))
      resd = ComputeOperand::kInputResidence;
    else if (isa<OutputOperator>(node))
      resd = ComputeOperand::kOutputResidence;
    else
      resd = ComputeOperand::kInternalResidence;

    createMemOperandsOfNode(pCG, *node, resd);
  }
}

void setMemOperand(Module& pModule)
{
  //MemAllocData* memAllocData = getAnalysis<MemAllocData>();

  for (ComputeOperand* opnd : pModule.getComputeOperands()) {
    Value* v = opnd->getValue();
    //if (!v || !memAllocData->hasAlloc(v))
    //  continue;

   // MemAllocData::AllocEntry alloc = memAllocData->getAlloc(v);
    // FIXME: need some check, e.g isa<>
    ComputeMemOperand* memOpnd = (ComputeMemOperand*)opnd;
    memOpnd->setStart(0);
    memOpnd->setLength(12);
  }
}

int main(void)
{
    uint32_t ret = 0;

    onnc::Module module;
    IRBuilder builder(module);

  ComputeGraph& cg = *builder.CreateComputeGraph("mxnet");

  // Create Input.
  cg.addOperator<InputOperator>()->setTensor(
    *CreateFloatComputeTensor(cg, "data0", {1, 1, 3, 3}));
  
  //CreateFloatWeightOperator(cg, "A", {3});
  
    // create nodes (layers)

  ComputeOperator* op2 = CreateComputeOperator<Relu>(cg,    {"data0"});

  op2->addOutput(*CreateFloatComputeTensor(cg, "activation0", {1, 1, 3, 3}));

  ComputeOperator* op3 = CreateComputeOperator<OutputOperator>(cg, {"activation0"});

  createMemOperandsOfGraph(cg);
  setMemOperand(module);

  NvDlaBackendMeta* pMeta = new (NvDlaBackendMeta);
  init_nvdla_memory(module, pMeta);

  ComputeGraph::iterator nodeIt, nEnd = cg.end();
  for (nodeIt = cg.begin(); nodeIt != nEnd; ++nodeIt) {
    const onnc::ComputeOperator *node = nodeIt;
    ///beforeEmit(node);
    std::cout<< node->name() << std::endl;
    if(node->name() == "Relu"){

      relu(* (Relu *) node, pMeta);
    }
  }
    task_submit(pMeta);
    nvdla_filegen(pMeta);
    
    return ret;
}