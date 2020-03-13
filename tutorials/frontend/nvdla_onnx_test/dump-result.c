#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
extern void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  float* compute = (float*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  (void)NvDlaInit("Init");
  (void)AddInputOp(AddFloatTensor(56243696, 4, 1, 1, 6, 3));
  (void)AddFloatWeightTensorFromNumpy(56243456, 4, placeholder1, 1, 1, 1, 1);
  (void)AddOutput(AddConvOp(56243696, 56243456), AddFloatTensor(56244576, 4, 1, 1, 6, 3));
  (void)SetConvDilations(GetOpPointer(56243456), 2, 1, 1);
  (void)SetConvGroup(GetOpPointer(56243456), 1);
  (void)SetConvKernelShape(GetOpPointer(56243456), 2, 1, 1);
  (void)SetConvPads(GetOpPointer(56243456), 4, 0, 0, 0, 0);
  (void)SetConvStrides(GetOpPointer(56243456), 2, 1, 1);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_reshape( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  float* T_reshape = (float*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  (void)AddOutput(AddReshapeOp(56244576, 2, 1, 18), AddFloatTensor(56244224, 2, 1, 18));
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_gemm_1( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  float* placeholder2 = (float*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  float* T_gemm = (float*)(((DLTensor*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((DLTensor*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  (void)AddFloatWeightTensorFromNumpy(56243392, 2, placeholder1, 18, 7);
  (void)AddFloatWeightTensorFromNumpy(56221600, 2, placeholder2, 1, 7);
  (void)AddOutput(AddGemmOp(56244224, 56243392, 56221600), AddFloatTensor(56243920, 2, 1, 7));
  (void)SetGemmAlpha(GetOpPointer(56244224), 1.000000e+00f);
  (void)SetGemmBeta(GetOpPointer(56244224), 1.000000e+00f);
  (void)SetGemmTransA(GetOpPointer(56244224), 0);
  (void)SetGemmTransB(GetOpPointer(56244224), 0);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_gemm( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  float* placeholder2 = (float*)(((DLTensor*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((DLTensor*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((DLTensor*)arg2)[0].strides);
  float* T_gemm = (float*)(((DLTensor*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((DLTensor*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((DLTensor*)arg3)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  if (!(arg3_strides == NULL)) {
  }
  (void)AddFloatWeightTensorFromNumpy(56240192, 2, placeholder1, 7, 1);
  (void)AddFloatWeightTensorFromNumpy(56240432, 2, placeholder2, 1, 1);
  (void)AddOutput(AddGemmOp(56243616, 56240192, 56240432), AddFloatTensor(56245088, 2, 1, 1));
  (void)SetGemmAlpha(GetOpPointer(56243616), 1.000000e+00f);
  (void)SetGemmBeta(GetOpPointer(56243616), 1.000000e+00f);
  (void)SetGemmTransA(GetOpPointer(56243616), 0);
  (void)SetGemmTransB(GetOpPointer(56243616), 0);
  (void)AddOutputOp(56245088);
  (void)NvDlaCompile();
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_relu( void* args,  void* arg_type_ids, int32_t num_args,  void* out_ret_value,  void* out_ret_tcode) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((DLTensor*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((DLTensor*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((DLTensor*)arg0)[0].strides);
  int32_t dev_type = (((DLTensor*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((DLTensor*)arg0)[0].ctx.device_id);
  float* T_relu = (float*)(((DLTensor*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((DLTensor*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  (void)AddOutput(AddReluOp(56243920), AddFloatTensor(56243616, 2, 1, 7));
  return 0;
}
