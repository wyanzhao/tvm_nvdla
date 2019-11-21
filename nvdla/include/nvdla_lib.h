#ifndef NVDLA_LIB_H_
#define NVDLA_LIB_H_

#include <nvdla_meta.h>
#include <onnc/IR/Module.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


void init_nvdla_memory(onnc::Module& pModule, NvDlaBackendMeta* pMeta);
void task_submit(NvDlaBackendMeta* pMeta);
void nvdla_filegen(NvDlaBackendMeta* pMeta);

int issueDlaAddr(int mid, NvDlaCubeInfo cube, int groups, int gidx, int ofs, NvDlaBackendMeta* m_pMeta);
void issueDlaOp(NvDlaDlaOperation* op, NvDlaDlaOperation* op_fuse, NvDlaDlaOperation* op_prev, NvDlaBackendMeta* m_pMeta);

#endif