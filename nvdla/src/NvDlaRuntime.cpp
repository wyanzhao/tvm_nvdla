


#include <NvDlaRuntime.h>

#include <NvDlaLib.h>
#include <cstdarg>
#include <utility>
#include <unordered_map>
#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>

#ifdef __cplusplus
extern "C" {
#endif

//#include <tvm/runtime/c_runtime_api.h>

static NvDlaLib* p_nvdla_lib = nullptr;
static onnc::ComputeGraph* p_cg = nullptr;
static onnc::Module* p_module = nullptr;
static std::unordered_map<std::string, void*> op_symbol_table;

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

void* AddFloatTensor(const char* p_name, uint64_t ndim, ...)
{
    std::vector<uint64_t> dim_vec(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, uint64_t);
    }
    va_end(valist);

    std::string name(p_name);
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());

    #ifdef NVDLA_DEBUG
    printf("Tensor Dim Begin\n");
    for(uint64_t i : pDims)
    {
        printf("%lu ", i);
    }
    printf("\nTensor Dim End\n");
    #endif

    return (void *) p_nvdla_lib->create_float_compute_tensor(p_name, pDims);
}

void AddReshapeTensor(const char* p_name, uint64_t ndim, ...)
{
    std::vector<uint64_t> dim_vec(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, uint64_t);
    }
    va_end(valist);

    std::string name(p_name);
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());

    #ifdef NVDLA_DEBUG
    printf("Reshape Dim Begin\n");
    for(uint64_t i : pDims)
    {
        printf("%lu ", i);
    }
    printf("\nReshape Dim End\n");
    #endif

    p_nvdla_lib->create_weight_operator<Int64Tensor>(std::move(p_name), std::move(pDims));
}

void AddFloatWeightTensor(const char* p_name, uint64_t ndim, ...)
{
    std::vector<uint64_t> dim_vec(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, uint64_t);
    }
    va_end(valist);

    std::string name(p_name);
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());

    #ifdef NVDLA_DEBUG
    printf("Weight Dim Begin\n");
    for(uint64_t i : pDims)
    {
        printf("%lu ", i);
    }
    printf("\nWeight Dim End\n");
    #endif

    p_nvdla_lib->create_weight_operator<FloatTensor>(std::move(p_name), std::move(pDims));
}

void AddFloatWeightTensorFromNumpy(const char* p_name, uint64_t ndim, void* data, ...)
{
    std::vector<uint64_t> dim_vec(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, uint64_t);
    }
    va_end(valist);

    std::string name(p_name);
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());

    #ifdef NVDLA_DEBUG
    printf("Weight From Numpy Dim Begin\n");
    for(uint64_t i : pDims)
    {
        printf("%lu ", i);
    }
    printf("\nWeight From Numpy Dim End\n");
    #endif

    p_nvdla_lib->create_float_weight_tensor_from_numpy(std::move(p_name), std::move(pDims), data);
}

void* AddConvOp(const char* input_name, const char* weight_name)
{
    std::string input_name_(input_name);
    std::string weight_name_(weight_name);

    auto p = (void *) p_nvdla_lib->create_compute_operator<Conv>({std::move(input_name_), std::move(weight_name_)});
    auto op_in_symbol_table = op_symbol_table.find(input_name);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name_] = p;
    }
    return p;
}

TVM_DLL void* AddAddOp(const char* input_name1, const char* input_name2)
{
    std::string input_name1_(input_name1);
    std::string input_name2_(input_name2);

    auto p =  (void *) p_nvdla_lib->create_compute_operator<Add>({std::move(input_name1_), std::move(input_name2_)});

    auto op_in_symbol_table = op_symbol_table.find(input_name1_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name1_] = p;
    }
    return p;
}

void* AddReluOp(const char* input_name)
{
    std::string name(input_name);

    auto p =  (void *) p_nvdla_lib->create_compute_operator<Relu>({std::move(name)});

    auto op_in_symbol_table = op_symbol_table.find(name);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[name] = p;
    }
    return p;
}

TVM_DLL void* AddReshapeOp(const char* input_name1, const char* input_name2)
{
    std::string input_name1_(input_name1);
    std::string input_name2_(input_name2);

    auto p = (void *) p_nvdla_lib->create_compute_operator<Reshape>({std::move(input_name1_), std::move(input_name2_)});
    auto op_in_symbol_table = op_symbol_table.find(input_name1_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name1_] = p;
    }
    return p;
}

TVM_DLL void* AddGemmOp(const char* input_name, const char* weight, const char* bias)
{
    std::string input_name_(input_name);
    std::string weight_(weight);
    std::string bias_(bias);

    auto p = (void *) p_nvdla_lib->create_compute_operator<Gemm>({std::move(input_name_), std::move(weight_),
     std::move(bias_)});

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name_] = p;
    }
    return p;
}

TVM_DLL void* AddAveragePoolOp(const char* input_name, uint64_t ndim, ...)
{
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    std::string input_name_(input_name);

    #ifdef NVDLA_DEBUG
    printf("kernel_shape Begin\n");
    for(uint64_t i : kernel_shape)
    {
        printf("%lu ", i);
    }
    printf("kernel_shape End\n");
    #endif

    auto p = (void *) p_nvdla_lib->create_compute_operator<AveragePool>({input_name_}, \
    p_nvdla_lib->get_values<int64_t>(kernel_shape));

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name_] = p;
    }
    return p;
}

TVM_DLL void* AddGlobalAveragePoolOp(const char* input_name)
{
    std::string input_name_(input_name);

    auto p = (void *) p_nvdla_lib->create_compute_operator<GlobalAveragePool>({input_name_});

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name_] = p;
    }
    return p;
}


TVM_DLL void* AddMaxPoolOp(const char* input_name, uint64_t ndim, ...)
{
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    std::string input_name_(input_name);

    #ifdef NVDLA_DEBUG
    printf("kernel_shape Begin\n");
    for(uint64_t i : kernel_shape)
    {
        printf("%lu ", i);
    }
    printf("kernel_shape End\n");
    #endif

    auto p = (void *) p_nvdla_lib->create_compute_operator<MaxPool>({input_name_}, \
    p_nvdla_lib->get_values<int64_t>(kernel_shape));

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name_] = p;
    }
    return p;
}

void* GetOpPointer(const char *input_name)
{
    std::string input_name_(input_name);

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        return op_symbol_table[input_name_];
    } else {
        printf("Can't find Op Pointer by input_name\n");
        return nullptr;
    }
}


// Conv Related
TVM_DLL void SetConvDilations(void *conv_op, uint64_t ndim, ...)
{
    std::vector<int64_t> dilations(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dilations[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set Conv Dilations Begin\n");
    for(uint64_t i : dilations)
    {
        printf("%lu ", i);
    }
    printf("Set Conv Dilations End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(dilations);
    ((onnc::Conv *) conv_op)->setDilations(std::move(v));
}

TVM_DLL void SetConvGroup(void *conv_op, uint64_t ngroups)
{
    auto groups = IntAttr(ngroups);
    ((onnc::Conv *) conv_op)->setGroup(std::move(groups));
}

TVM_DLL void SetConvKernelShape(void *conv_op, uint64_t ndim, ...)
{
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set Conv KernelShape Begin\n");
    for(uint64_t i : kernel_shape)
    {
        printf("%lu ", i);
    }
    printf("Set Conv KernelShape End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(kernel_shape);
    ((onnc::Conv *) conv_op)->setKernelShape(std::move(v));
}

TVM_DLL void SetConvPads(void *conv_op, uint64_t ndim, ...)
{
    std::vector<int64_t> pads(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        pads[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set Conv Pads Begin\n");
    for(uint64_t i : pads)
    {
        printf("%lu ", i);
    }
    printf("Set Conv Pads End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(pads);
    ((onnc::Conv *) conv_op)->setPads(std::move(v));
}

TVM_DLL void SetConvStrides(void *conv_op, uint64_t ndim, ...)
{
    std::vector<int64_t> strides(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        strides[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set Conv Strides Begin\n");
    for(uint64_t i : strides)
    {
        printf("%lu ", i);
    }
    printf("Set Conv Strides End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(strides);
    ((onnc::Conv *) conv_op)->setStrides(std::move(v));
}

// MaxPool Related
TVM_DLL void SetMaxPoolKernelShape(void *maxpool_op, uint64_t ndim, ...)
{
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set MaxPool KernelShape Begin\n");
    for(uint64_t i : kernel_shape)
    {
        printf("%lu ", i);
    }
    printf("Set MaxPool KernelShape End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(kernel_shape);
    ((onnc::MaxPool *) maxpool_op)->setKernelShape(std::move(v));
}

TVM_DLL void SetMaxPoolPads(void *maxpool_op, uint64_t ndim, ...)
{
    std::vector<int64_t> pads(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        pads[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set MaxPool Pads Begin\n");
    for(uint64_t i : pads)
    {
        printf("%lu ", i);
    }
    printf("Set MaxPool Pads End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(pads);
    ((onnc::MaxPool *) maxpool_op)->setPads(std::move(v));
}

TVM_DLL void SetMaxPoolStrides(void *maxpool_op, uint64_t ndim, ...)
{
    std::vector<int64_t> strides(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        strides[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set MaxPool Strides Begin\n");
    for(uint64_t i : strides)
    {
        printf("%lu ", i);
    }
    printf("Set MaxPool Strides End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(strides);
    ((onnc::MaxPool *) maxpool_op)->setStrides(std::move(v));
}

TVM_DLL void SetAveragePoolKernelShape(void *averagepool_op, uint64_t ndim, ...)
{
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set AveragePool KernelShape Begin\n");
    for(uint64_t i : kernel_shape)
    {
        printf("%lu ", i);
    }
    printf("Set AveragePool KernelShape End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(kernel_shape);
    ((onnc::AveragePool *) averagepool_op)->setKernelShape(std::move(v));
}

TVM_DLL void SetAveragePoolAutoPad(void *averagepool_op, const char* auto_pads)
{
    std::string autopads(auto_pads);
    ((onnc::AveragePool *) averagepool_op)->setAutoPad(std::move(autopads));
}

TVM_DLL void SetAveragePoolCountIncludePad(void *averagepool_op, uint64_t ncount)
{
    ((onnc::AveragePool *) averagepool_op)->setCountIncludePad(std::move(ncount));
}

TVM_DLL void SetAveragePoolPads(void *averagepool_op, uint64_t ndim, ...)
{
    std::vector<int64_t> pads(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        pads[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set AveragePool Pads Begin\n");
    for(uint64_t i : pads)
    {
        printf("%lu ", i);
    }
    printf("Set AveragePool Pads End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(pads);
    ((onnc::AveragePool *) averagepool_op)->setPads(std::move(v));
}

TVM_DLL void SetAveragePoolStrides(void *averagepool_op, uint64_t ndim, ...)
{
    std::vector<int64_t> strides(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        strides[i] = va_arg(valist, int64_t);
    }
    va_end(valist);

    #ifdef NVDLA_DEBUG
    printf("Set AveragePool Strides Begin\n");
    for(uint64_t i : strides)
    {
        printf("%lu ", i);
    }
    printf("Set AveragePool Strides End\n");
    #endif

    auto v = p_nvdla_lib->get_values<int64_t>(strides);
    ((onnc::AveragePool *) averagepool_op)->setStrides(std::move(v));
}

TVM_DLL void* AddBatchNormOp(const char* input_name, const char*scale, const char* B, const char* mean, const char* var)
{
    std::string input_name_(input_name);
    std::string scale_(scale);
    std::string B_(B);
    std::string mean_(mean);
    std::string var_(var);

    auto p = (void *) p_nvdla_lib->create_compute_operator<BatchNormalization>({
    std::move(input_name_), 
    std::move(scale_), 
    std::move(B_),
    std::move(mean_), 
    std::move(var_)});

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name_] = p;
    }
    return p;
}

// Gemm Releated
TVM_DLL void SetGemmAlpha(void *gemm_op, float alpha)
{
    ((onnc::Gemm *) gemm_op)->setAlpha(std::move(alpha));
}

TVM_DLL void SetGemmBeta(void *gemm_op, float beta)
{
    ((onnc::Gemm *) gemm_op)->setBeta(std::move(beta));
}

TVM_DLL void SetGemmTransA(void *gemm_op, int64_t trans_a)
{
    ((onnc::Gemm *) gemm_op)->setTransA(std::move(trans_a));
}

TVM_DLL void SetGemmTransB(void *gemm_op, int64_t trans_b)
{
    ((onnc::Gemm *) gemm_op)->setTransB(std::move(trans_b));
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
