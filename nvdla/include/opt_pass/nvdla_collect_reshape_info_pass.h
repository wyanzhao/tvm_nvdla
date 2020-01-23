#ifndef NVDLA_COLLECT_RESHAPE_INFO_PASS_H_
#define NVDLA_COLLECT_RESHAPE_INFO_PASS_H_

#include <NvDlaLib.h>
#include <NvDlaOp.h>

#include <string>

using namespace onnc;

class NvDlaCollectReshapeInfoPass {
public:
    static bool run(Module& pModule, NvDlaBackendMeta& m_pMeta);
};

#endif