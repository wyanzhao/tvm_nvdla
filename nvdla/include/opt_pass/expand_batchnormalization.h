#ifndef EXPAND_BATCHNORMALIZATION_H_
#define EXPAND_BATCHNORMALIZATION_H_

#include <NvDlaLib.h>
#include <NvDlaOp.h>

#include <string>

using namespace onnc;

class ExpandBatchNormalization {
public:
    static bool run(ComputeGraph& pCG);
private:
    static void expandBNToAddAndMul(ComputeGraph& pCG, BatchNormalization& batchNormalization);
private:
    static unsigned tensorIdx;
};

#endif