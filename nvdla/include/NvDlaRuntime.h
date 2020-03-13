#ifndef NVDLA_RUNTIME_H_
#define NVDLA_RUNTIME_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>
#include <stdio.h>

TVM_DLL void NvDlaInit(const char* cg_name);
TVM_DLL void AddInputOp(void* tensor);
TVM_DLL void AddInputOpByName(const char* p_name);
TVM_DLL void* AddFloatTensor(const void* p_name, uint64_t ndim, ...);
TVM_DLL void AddFloatWeightTensorFromNumpy(const void* p_name, const char* comment,uint64_t ndim, void* data, ...);
TVM_DLL void AddOutput(void *op, void* tensor);
TVM_DLL void* AddConvOp(const void* input_name, const void* weight_name);
TVM_DLL void* AddReluOp(const void* input_name);
TVM_DLL void* AddAddOp(const void* input_name1, const void* input_name2);
TVM_DLL void* AddMulOp(const char* input_name1, const char* input_name2);
TVM_DLL void* AddReshapeOp(const void* input_name1, uint64_t ndim, ...);
TVM_DLL void* AddMaxPoolOp(const void* input_name, uint64_t ndim, ...);
TVM_DLL void* AddGlobalAveragePoolOp(const void* input_name);
TVM_DLL void* AddAveragePoolOp(const char* input_name, uint64_t ndim, ...);
TVM_DLL void* AddGemmOp(const void* input_name, const void* weight, const void* bias);
TVM_DLL void* AddBatchNormOp(const void* input_name, const void*scale, const void* B, const void* mean, const void* var);

TVM_DLL void GetOpPointerAddress(const void* input_ptr, const char* word)
{   
    size_t addr = (size_t) input_ptr;
    printf("%s Ptr Addr: %p\n", word, addr);
}

TVM_DLL void AddOutputOp(const void* input_name);
TVM_DLL void NvDlaCompile();
TVM_DLL void* GetOpPointer(const void *input_name);

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
/*
// AveragePool Related
TVM_DLL void SetAveragePoolKernelShape(void *averagepool_op, uint64_t ndim, ...);
TVM_DLL void SetAveragePoolAutoPad(void *averagepool_op, const char* auto_pads);
TVM_DLL void SetAveragePoolCountIncludePad(void *averagepool_op, uint64_t ncount);
TVM_DLL void SetAveragePoolPads(void *averagepool_op, uint64_t ndim, ...);
TVM_DLL void SetAveragePoolStrides(void *averagepool_op, uint64_t ndim, ...);
*/
// Gemm Related
TVM_DLL void SetGemmAlpha(void *gemm_op, float alpha);
TVM_DLL void SetGemmBeta(void *gemm_op, float beta);
TVM_DLL void SetGemmTransA(void *gemm_op, int32_t trans_a);
TVM_DLL void SetGemmTransB(void *gemm_op, int32_t trans_b);

// BatchNorm Related
TVM_DLL void SetBatchNormEpsilon(void *batch_norm_op, float epsilon);
TVM_DLL void SetBatchNormMomentum(void *batch_norm_op, float momentum);
TVM_DLL void SetBatchNormSpatial(void *batch_norm_op, int32_t spatial);

#ifdef __cplusplus
}
#endif

#endif