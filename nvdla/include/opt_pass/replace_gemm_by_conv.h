#ifndef REPLACE_GEMM_BY_CONV_H_
#define REPLACE_GEMM_BY_CONV_H_

#include <NvDlaLib.h>
#include <NvDlaOp.h>

#include <string>

using namespace onnc;

class ReplaceGemmByConv {
    public:
    static bool run(ComputeGraph& pCG);

    private:

  static Tensor* addReshapeBefore(ComputeGraph& pCG, Tensor* inputTensor);
  static ComputeOperator* constructConvAndReshape(ComputeGraph& pCG,
                                           Tensor* tA, Tensor* tB, Tensor* tC);

    private:
  static const std::string convPrefixName;
  static const std::string reshapePrefixName;
  static const std::string shapePrefixName;

  static unsigned convIdx;
  static unsigned shapeIdx;
  static unsigned reshapeIdx;
};

#endif