#include <onnc.h>

using namespace onnc;

void setMemOperand(onnc::Module& pModule)
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
    memOpnd->setLength(48);
  }
}

static void createMemOperandsOfNode(onnc::ComputeGraph& pCG,
                                              onnc::ComputeOperator& pNode,
                                              onnc::ComputeOperand::Residence pResd)
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

void createMemOperandsOfGraph(onnc::ComputeGraph& pCG)
{
  ComputeGraph::iterator nodeIt, nEnd = pCG.end();
  for (nodeIt = pCG.begin(); nodeIt != nEnd; ++nodeIt) {
    ComputeOperator *node = nodeIt;
    ComputeOperand::Residence resd;
    if (isa<onnc::Initializer>(node))
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
