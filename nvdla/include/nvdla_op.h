#ifndef NVDLA_OP_H
#define NVDLA_OP_H

#include <onnc/Core/CustomPass.h>
#include <onnc/IR/Compute/Relu.h>
#include <nvdla_lib.h>

void relu(const onnc::Relu& pOp, NvDlaBackendMeta* m_pMeta);

#endif