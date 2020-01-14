


#include "NvDlaRuntime.h"

#include <NvDlaLib.h>
#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/Relu.h>

#ifdef __cplusplus
extern "C" {
#endif

//#include <tvm/runtime/c_runtime_api.h>

static NvDlaLib* p_nvdla_lib = nullptr;
static onnc::ComputeGraph* p_cg = nullptr;
static onnc::Module* p_module = nullptr;

void NvDlaInit(const char* cg_name)
{
    std::string tmp_cg_name(cg_name);

    p_module = new(onnc::Module);
    onnc::IRBuilder builder(*p_module);
    p_cg = builder.CreateComputeGraph(tmp_cg_name);
    p_nvdla_lib = new NvDlaLib(p_module, p_cg);
}

void AddInputOp(void* tensor)
{   
    p_cg->addOperator<InputOperator>()->setTensor(*(onnc::Tensor*)tensor);
}

void* AddFloatTensor(const char* p_name, int *dims, int ndim)
{
    std::string name(p_name);
    Tensor::Dimensions pDims(dims, dims + ndim);

    return (void *) p_nvdla_lib->create_float_compute_tensor(p_name, pDims);
}

void* AddReluOp(const char* input_name)
{
    std::string name(input_name);
    return (void *) p_nvdla_lib->create_compute_operator<Relu>({name});
}


void AddOutput(void *op, void* tensor) {
    
    ((ComputeOperator *)op)->addOutput(*(onnc::Tensor *) tensor);
}

void AddOutputOp(const char* input_name)
{
    std::string name(input_name);
    p_nvdla_lib->create_compute_operator<OutputOperator>({name});
}

void Compile()
{
    p_nvdla_lib->optimize();
    p_nvdla_lib->compile();
}

#ifdef __cplusplus
}
#endif
