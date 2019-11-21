#ifndef NVDLA_LIB_H_
#define NVDLA_LIB_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <onnc/Core/CustomPass.h>
#include <onnc/IR/Compute/Relu.h>

#include "nvdla_meta.h"

using namespace onnc;

//void init_nvdla_memory(NvDlaBackendMeta* pMeta);
void init_nvdla_memory(Module& pModule, NvDlaBackendMeta* pMeta);
void task_submit(NvDlaBackendMeta* pMeta);
void nvdla_filegen(NvDlaBackendMeta* pMeta);
void relu(const Relu& pOp, NvDlaBackendMeta* pMeta);

int issueDlaAddr(int mid, NvDlaCubeInfo cube, int groups, int gidx, int ofs, NvDlaBackendMeta* m_pMeta);
void issueDlaOp(NvDlaDlaOperation* op, NvDlaDlaOperation* op_fuse, NvDlaDlaOperation* op_prev, NvDlaBackendMeta* m_pMeta);
int submitEvent(int task_id, int event_type, NvDlaBackendMeta* pMeta);
int submitMemAllocAddress(int size, std::string blob_name, NvDlaBackendMeta* pMeta);

#endif