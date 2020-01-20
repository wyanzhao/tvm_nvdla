#ifndef NVDLA_RUNTIME_H_
#define NVDLA_RUNTIME_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

TVM_DLL void NvDlaInit(const char* cg_name);
TVM_DLL void AddInputOp(void* tensor);
TVM_DLL void* AddFloatTensor(const char* p_name, uint64_t ndim, ...);
TVM_DLL void AddFloatWeightTensorFromNumpy(const char* p_name, uint64_t ndim, void* data, ...);
TVM_DLL void AddFloatWeightTensor(const char* p_name, uint64_t ndim, ...);
TVM_DLL void AddOutput(void *op, void* tensor);
TVM_DLL void* AddConvOp(const char* input_name, const char* weight_name);
TVM_DLL void* AddReluOp(const char* input_name);
TVM_DLL void* AddAddOp(const char* input_name1, const char* input_name2);
TVM_DLL void* AddReshapeOp(const char* input_name1, const char* input_name2);
TVM_DLL void* AddMaxPoolOp(const char* input_name, uint64_t ndim, ...);
TVM_DLL void AddOutputOp(const char* input_name);
TVM_DLL void Compile();
TVM_DLL void* GetOpPointer(const char *input_name);

// Conv Related
TVM_DLL void SetConvDilations(void *conv_op, uint64_t ndim, ...);
TVM_DLL void SetConvGroup(void *conv_op, uint64_t ngroups);
TVM_DLL void SetConvKernelShape(void *conv_op, uint64_t ndim, ...);
TVM_DLL void SetConvPads(void *conv_op, uint64_t ndim, ...);
TVM_DLL void SetConvStrides(void *conv_op, uint64_t ndim, ...);

// MaxPool Related
TVM_DLL void SetMaxPoolKernelShape(void *maxpool_op, uint64_t ndim, ...);
TVM_DLL void SetMaxPoolPads(void *maxpool_op, uint64_t ndim, ...);
TVM_DLL void SetMaxPoolStrides(void *maxpool_op, uint64_t ndim, ...);

#ifdef __cplusplus
}
#endif

#endif