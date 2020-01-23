#ifndef PROPAGATE_CONST_WITH_DIFF_SHAPE_H_
#define PROPAGATE_CONST_WITH_DIFF_SHAPE_H_

#include <NvDlaLib.h>
#include <NvDlaOp.h>

#include <string>

using namespace onnc;

class PropagateConstWithDiffShape {
    public:
    static bool run(ComputeGraph& pCG);
};

#endif