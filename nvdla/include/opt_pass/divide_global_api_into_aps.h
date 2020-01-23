#ifndef DEVIDE_GLOBAL_API_INTO_APS_H_
#define DEVIDE_GLOBAL_API_INTO_APS_H_

#include <NvDlaLib.h>
#include <NvDlaOp.h>

using namespace onnc;

typedef int64_t ValueType;
static const ValueType m_MaxKernelSize = 8;

class DivideGlobalAPIntoAPs {
public:
    typedef int64_t ValueType;
    typedef std::vector<ValueType> VectorType;

public:
    static bool run(ComputeGraph& pCG);
private:
    
     // Given a kernel size, return a list of kernel sizes of AveragePool
    static VectorType divideKernelSizeOf(const ValueType& kernelSize, const ValueType& maxKernelSize);

  // Given a kernel size, return a list of kernel sizes of AveragePool
  // These number can compose the kernel size without pads
  // If there is no solution, the size of return IntsAttr will be zero
    static VectorType canBeComposedOf(const ValueType& kernelSize, const ValueType& maxKernelSize);

  // Get k, where maxKernelSize ^ (k-1) < kernelSize <= maxKernelSize ^ k
    static ValueType getBestSize(const ValueType& kernelSize, const ValueType& maxKernelSize);

  template <int T = 0>
  static std::pair<ComputeOperator*, ComputeOperator*>
    genListOfAPsMul(ComputeGraph& pCG,
                    const Tensor* inputTensor,
                    const VectorType& lstOfKernels);

  template <typename FirstTensorType, typename... RestTensorTypes, int T = 0>
  static std::pair<ComputeOperator*, ComputeOperator*>
    genListOfAPsMul(ComputeGraph& pCG,
                    const Tensor* inputTensor,
                    const VectorType& lstOfKernels);

private:
    static unsigned tensorIdx;
};

#endif