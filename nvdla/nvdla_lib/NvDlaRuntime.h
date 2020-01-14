#ifndef NVDLA_RUNTIME_H_
#define NVDLA_RUNTIME_H_


#ifdef __cplusplus
extern "C" {
#endif


void NvDlaInit(const char* cg_name);
void AddInputOp(void* tensor);
void* AddFloatTensor(const char* p_name, int *dims, int ndim);
void AddOutput(void *op, void* tensor);
void* AddReluOp(const char* input_name);
void AddOutputOp(const char* input_name);
void Compile();

#ifdef __cplusplus
}
#endif

#endif