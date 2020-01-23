#include <opt_pass/nvdla_collect_reshape_info_pass.h>


bool NvDlaCollectReshapeInfoPass::run(Module& pModule, NvDlaBackendMeta& m_pMeta)
{
  for (ComputeOperator& cm : *pModule.getRootComputeGraph()) {
    if (const Reshape* const reshape = dyn_cast<Reshape>(&cm)) {
      const Tensor* const          shape         = reshape->getShape();
      const ComputeOperator* const shapeProducer = static_cast<const ComputeOperator*>(shape->getDefine());
      assert(isa<Initializer>(shapeProducer));

      const Tensor* const input  = reshape->getInput(0);
      const Tensor* const output = reshape->getOutput(0);
      m_pMeta.markAsReshaped(*input, *output);
    }
  }

  return false;
}