#include <NvDlaRuntime.h>

#include <NvDlaLib.h>
#include <cstdarg>
#include <utility>
#include <unordered_map>
#include <memory>
#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>


#ifdef __cplusplus
extern "C" {
#endif

//#include <tvm/runtime/c_runtime_api.h>

static std::unique_ptr<NvDlaLib> p_nvdla_lib;
static std::unique_ptr<onnc::ComputeGraph> p_cg;
static std::unique_ptr<onnc::Module> p_module;
static std::unordered_map<std::string, void*> op_symbol_table;
static std::string input_name = "";
static std::unordered_map<std::string, void*> tensor_sym_table;


void NvDlaInit(const char* cg_name)
{
    std::string tmp_cg_name(cg_name);

    p_module = std::make_unique<onnc::Module>();
    onnc::IRBuilder builder(*p_module);
    p_cg.reset(builder.CreateComputeGraph(std::move(tmp_cg_name)));
    p_nvdla_lib = std::make_unique<NvDlaLib>(p_module.get(), p_cg.get());
}

void AddInputOp(void* tensor)
{   
    p_cg->addOperator<InputOperator>()->setTensor(*(onnc::Tensor*)tensor);
}

TVM_DLL void AddInputOpByName(const char* p_name)
{
    if (input_name == "")
    {
        input_name = std::move(std::string(p_name));
    } else
    {
        printf("Error Graph alread had a input\n");
    }
}

void* AddFloatTensor(const void* p_name, uint64_t ndim, ...)
{
    size_t input_ = (size_t) p_name;
    std::string tensor_name(std::to_string(input_));
    #ifdef NVDLA_DEBUG
    printf("Enter AddFloatTensor, Input Name:%x\n", input_);
    #endif

    auto tensor = tensor_sym_table.find(tensor_name);
    if (tensor != tensor_sym_table.end())
    {
        return tensor->second;
    }
    
    std::vector<uint64_t> dim_vec(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, int32_t);
    }
    va_end(valist);

    std::string name(std::to_string(input_));
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());

    #ifdef NVDLA_DEBUG
    printf("Tensor Dim Begin\n");
    for(uint64_t i : pDims)
    {
        printf("%lu ", i);
    }
    printf("\nTensor Dim End\n");
    #endif

    auto input_tensor = p_nvdla_lib->create_float_compute_tensor(name, pDims);
    tensor_sym_table[tensor_name] = input_tensor;
    if(std::string(name) == input_name)
    {
        printf("Input name:%s\n", name);
        p_cg->addOperator<InputOperator>()->setTensor(*(onnc::Tensor*)input_tensor);
    }
    #ifdef NVDLA_DEBUG
    printf("Exit AddFloatTensor\n");
    #endif
    return input_tensor;
}


void AddFloatWeightTensorFromNumpy(const void *p_name, const char* comment_name, uint64_t ndim, void* data, ...)
{
    std::vector<uint64_t> dim_vec(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, int32_t);
    }
    va_end(valist);

    //char *p = (char *)p_name;
    std::size_t p = (size_t) p_name;

    #ifdef NVDLA_DEBUG
    printf("Enter AddFloatWeightTensorFromNumpy Input Name:%x %s\n", p, comment_name);
    #endif

    std::string name(std::to_string(p));
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());

    #ifdef NVDLA_DEBUG
    printf("Weight From Numpy Dim Begin\n");
    for(uint64_t i : pDims)
    {
        printf("%d ", i);
    }
    printf("\nWeight From Numpy Dim End\n");
    #endif

    p_nvdla_lib->create_float_weight_tensor_from_numpy(std::move(name), std::move(pDims), data);
    #ifdef NVDLA_DEBUG
    printf("Exit AddFloatWeightTensorFromNumpy\n");
    #endif
}


TVM_DLL void* AddConvOp(const void* input_name, const void* weight_name)
{
    size_t input_ = (size_t)input_name;
    size_t weight_ = (size_t)weight_name;
    #ifdef NVDLA_DEBUG
    printf("Exter AddConvOp, Input Name:%x weight_:%x\n", input_, weight_);
    #endif

    std::string input_name_(std::to_string(input_));
    std::string weight_name_(std::to_string(weight_));

    auto p = (void *) p_nvdla_lib->create_compute_operator<Conv>({std::move(input_name_), std::move(weight_name_)});
    auto op_in_symbol_table = op_symbol_table.find(weight_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[weight_name_] = p;
    }

    #ifdef NVDLA_DEBUG
    printf("Exit AddConvOp\n");
    #endif
    return p;
}

TVM_DLL void* AddAddOp(const void* input_name1, const void* input_name2)
{
    size_t input1_ = (size_t) input_name1;
    size_t input2_ = (size_t) input_name2;
    #ifdef NVDLA_DEBUG
    printf("Exter AddAddOp, Input Name1:%x Input Name2:%x\n", input1_, input2_);
    #endif

    std::string input_name1_(std::to_string(input1_));
    std::string input_name2_(std::to_string(input2_));

    auto p =  (void *) p_nvdla_lib->create_compute_operator<Add>({std::move(input_name1_), std::move(input_name2_)});

    auto op_in_symbol_table = op_symbol_table.find(input_name2_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name2_] = p;
    }
    #ifdef NVDLA_DEBUG
    printf("Exit AddAddOp\n");
    #endif
    return p;
}


TVM_DLL void* AddMulOp(const char* input_name1, const char* input_name2)
{   
    std::size_t input1_ = (std::size_t) input_name1;
    std::size_t input2_ = (std::size_t) input_name2;

    std::string input_name1_(std::to_string(input1_));
    std::string input_name2_(std::to_string(input2_));

    auto p =  (void *) p_nvdla_lib->create_compute_operator<Mul>({std::move(input_name1_), std::move(input_name2_)});

    auto op_in_symbol_table = op_symbol_table.find(input_name2);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name2] = p;
    }
    return p;
}


void* AddReluOp(const void* input_name)
{
    size_t input_ = (size_t)input_name;
    #ifdef NVDLA_DEBUG
    printf("Exter AddReluOp, Input Name:%x\n", input_);
    #endif
    
    std::string name(std::to_string(input_));

    auto p =  (void *) p_nvdla_lib->create_compute_operator<Relu>({std::move(name)});

    auto op_in_symbol_table = op_symbol_table.find(name);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[name] = p;
    }

    #ifdef NVDLA_DEBUG
    printf("Exit AddReluOp, Input Name:%x\n", input_name);
    #endif
    return p;
}

TVM_DLL void* AddReshapeOp(const void* input_name1, uint64_t ndim, ...)
{
    size_t input_ = (size_t) input_name1;
    #ifdef NVDLA_DEBUG
    printf("Exter AddReshapeOp, Input Name:%x\n", input_);
    #endif
    std::string input_name1_(std::to_string(input_));

    std::vector<uint64_t> dim_vec(ndim);
    va_list valist;
    va_start(valist, ndim);
    for(uint64_t i = 0; i < ndim; ++i)
    {
        dim_vec[i] = va_arg(valist, int32_t);
    }
    va_end(valist);

    std::string reshape_tensor_name(input_name1_ + "_reshape_tensor");
    Tensor::Dimensions pDims(dim_vec.begin(), dim_vec.end());
    #ifdef NVDLA_DEBUG
    printf("Reshape Dim Begin\n");
    for(auto i : pDims)
    {
        printf("%lu ", i);
    }
    printf("\nReshape Dim End\n");
    #endif
    p_nvdla_lib->create_weight_operator<Int64Tensor>(std::move(reshape_tensor_name), std::move(pDims));

    auto p = (void *) p_nvdla_lib->create_compute_operator<Reshape>({std::move(input_name1_), std::move(reshape_tensor_name)});
    auto op_in_symbol_table = op_symbol_table.find(input_name1_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
    } else {
        op_symbol_table[input_name1_] = p;
    }

    #ifdef NVDLA_DEBUG
    printf("Exit AddReshapeOp, Input Name:%x\n", input_name1);
    #endif
    return p;
}

TVM_DLL void* AddGemmOp(const void* input_name, const void* weight, const void* bias)
{
    size_t input_ = (size_t)input_name;
    size_t weight_ = (size_t) weight;
    size_t bias_ = (size_t) bias;
    #ifdef NVDLA_DEBUG
    printf("Exter AddGemmOp, Input Name:%x Weight Name:%x Bias Name:%x\n", input_, weight_, bias_);
    #endif

    std::string input_name_(std::to_string(input_));
    std::string weight_name_(std::to_string(weight_));
    std::string bias_name_(std::to_string(bias_));

    auto p = (void *) p_nvdla_lib->create_compute_operator<Gemm>({std::move(input_name_), std::move(weight_name_),
     std::move(bias_name_)});

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
        exit(-1);
    } else {
        op_symbol_table[input_name_] = p;
    }

     #ifdef NVDLA_DEBUG
    printf("Exit AddGemmOp, Input Name:%x\n", input_name);
    #endif
    return p;
}

TVM_DLL void* AddAveragePoolOp(const char* input_name, uint64_t ndim, ...)
{
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(auto i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int32_t);
    }
    va_end(valist);
    std::size_t input_ = (std::size_t) input_name;

    #ifdef NVDLA_DEBUG
    printf("Exter AddAveragePoolOp, Input Name:%x\n", input_);
    #endif

    std::string input_name_(std::to_string(input_));

    #ifdef NVDLA_DEBUG
    printf("kernel_shape Begin\n");
    for(auto i : kernel_shape)
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

    #ifdef NVDLA_DEBUG
    printf("Exter AddAveragePoolOp, Input Name:%x\n", input_name);
    #endif
    return p;
}

TVM_DLL void* AddGlobalAveragePoolOp(const void* input_name)
{
    std::size_t input_ = (std::size_t) input_name;
    #ifdef NVDLA_DEBUG
    printf("Exter AddGlobalAveragePoolOp, Input Name:%x\n", input_);
    #endif
    std::string input_name_(std::to_string(input_));

    auto p = (void *) p_nvdla_lib->create_compute_operator<GlobalAveragePool>({input_name_});

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        printf("Input name existed in symbol table");
        exit(-1);
    } else {
        op_symbol_table[input_name_] = p;
    }

    #ifdef NVDLA_DEBUG
    printf("Exter AddGlobalAveragePoolOp, Input Name:%x\n", input_name);
    #endif
    return p;
}


TVM_DLL void* AddMaxPoolOp(const void* input_name, uint64_t ndim, ...)
{
    std::size_t input_ = (std::size_t) input_name;
    #ifdef NVDLA_DEBUG
    printf("Exter AddMaxPoolOp, Input Name:%x\n", input_);
    #endif
    std::vector<int64_t> kernel_shape(ndim);

    va_list valist;
    va_start(valist, ndim);
    for(auto i = 0; i < ndim; ++i)
    {
        kernel_shape[i] = va_arg(valist, int32_t);
    }
    va_end(valist);

    std::string input_name_(std::to_string(input_));

    #ifdef NVDLA_DEBUG
    printf("kernel_shape Begin\n");
    for(auto i : kernel_shape)
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
        exit(-1);
    } else {
        op_symbol_table[input_name_] = p;
    }

    #ifdef NVDLA_DEBUG
    printf("Exit AddMaxPoolOp, Input Name:%x\n", input_name);
    #endif
    return p;
}

void* GetOpPointer(const void *input_name)
{
    std::size_t input_ = (size_t) input_name;
    std::string input_name_(std::to_string(input_));

    auto op_in_symbol_table = op_symbol_table.find(input_name_);
    if (op_in_symbol_table != op_symbol_table.end())
    {
        return op_symbol_table[input_name_];
    } else {
        printf("Can't find Op Pointer by input_name\n");
        exit(-1);
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
        dilations[i] = va_arg(valist, int32_t);
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
    #ifdef NVDLA_DEBUG
    printf("Called SetConvGroup with value:%d\n", ngroups);
    #endif
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
        kernel_shape[i] = va_arg(valist, int32_t);
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
        pads[i] = va_arg(valist, int32_t);
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
        strides[i] = va_arg(valist, int32_t);
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
        kernel_shape[i] = va_arg(valist, int32_t);
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
        pads[i] = va_arg(valist, int32_t);
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
        strides[i] = va_arg(valist, int32_t);
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
        kernel_shape[i] = va_arg(valist, int32_t);
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
        pads[i] = va_arg(valist, int32_t);
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
        strides[i] = va_arg(valist, int32_t);
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

TVM_DLL void* AddBatchNormOp(const void* input_name, const void*scale, const void* B, const void* mean, const void* var)
{
    std::size_t input_ = (size_t) input_name;
    std::size_t s = (size_t) scale;
    std::size_t b = (size_t) B;
    std::size_t m = (size_t) mean;
    std::size_t v = (size_t) var;

    #ifdef NVDLA_DEBUG
    printf("Exter AddBatchNormOp, Input Name:%x s:%x b:%x m:%x v:%x\n", input_, s, b, m, v);
    #endif

    std::string input_name_(std::to_string(input_));
    std::string scale_(std::to_string(s));
    std::string B_(std::to_string(b));
    std::string mean_(std::to_string(m));
    std::string var_(std::to_string(v));

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

    #ifdef NVDLA_DEBUG
    printf("EXit AddBatchNormOp, Input Name:%x s:%x b:%x m:%x v:%x\n", input_, s, b, m, v);
    #endif
    return p;
}

// Gemm Releated
TVM_DLL void SetGemmAlpha(void *gemm_op, float alpha)
{
    #ifdef NVDLA_DEBUG
    printf("Called SetGemmAlpha with value:%f\n", alpha);
    #endif
    ((onnc::Gemm *) gemm_op)->setAlpha(std::move(alpha));
}

TVM_DLL void SetGemmBeta(void *gemm_op, float beta)
{
    #ifdef NVDLA_DEBUG
    printf("Called SetGemmBeta with value:%f\n", beta);
    #endif
    ((onnc::Gemm *) gemm_op)->setBeta(std::move(beta));
}

TVM_DLL void SetGemmTransA(void *gemm_op, int32_t trans_a)
{
        #ifdef NVDLA_DEBUG
    printf("Called SetGemmTransA with value:%d\n", trans_a);
    #endif
    ((onnc::Gemm *) gemm_op)->setTransA(std::move(trans_a));
}

TVM_DLL void SetGemmTransB(void *gemm_op, int32_t trans_b)
{
        #ifdef NVDLA_DEBUG
    printf("Called SetGemmTransB with value:%d\n", trans_b);
    #endif
    ((onnc::Gemm *) gemm_op)->setTransB(std::move(trans_b));
}

// BatchNorm Releated
TVM_DLL void SetBatchNormEpsilon(void *batch_norm_op, float epsilon)
{
    #ifdef NVDLA_DEBUG
    printf("Called SetBatchNormEpsilon with value:%f\n", epsilon);
    #endif
    ((onnc::BatchNormalization *) batch_norm_op)->setEpsilon(std::move(epsilon));
}

// BatchNorm Releated
TVM_DLL void SetBatchNormMomentum(void *batch_norm_op, float momentum)
{
    #ifdef NVDLA_DEBUG
    printf("Called SetBatchNormMomentum with value:%f\n", momentum);
    #endif
    ((onnc::BatchNormalization *) batch_norm_op)->setMomentum(std::move(momentum));
}

// BatchNorm Releated
TVM_DLL void SetBatchNormSpatial(void *batch_norm_op, int32_t spatial)
{
    #ifdef NVDLA_DEBUG
    printf("Called SetBatchNormSpatial with value:%d\n", spatial);
    #endif
    ((onnc::BatchNormalization *) batch_norm_op)->setSpatial(std::move(spatial));
}



void AddOutput(void *op, void* tensor) {
    #ifdef NVDLA_DEBUG
    printf("Exter AddOutput\n");
    #endif
    ((ComputeOperator *)op)->addOutput(*(onnc::Tensor *) tensor);

    #ifdef NVDLA_DEBUG
    printf("Exit AddOutput\n");
    #endif
}

void AddOutputOp(const void* input_name)
{   
    size_t input_ = (size_t)input_name;

    #ifdef NVDLA_DEBUG
    printf("Exter AddOutputOp, Input Name:%x\n", input_);
    #endif

    std::string name(std::to_string(input_));
    p_nvdla_lib->create_compute_operator<OutputOperator>({name});

    #ifdef NVDLA_DEBUG
    printf("Exit AddOutputOp, Input Name:%x\n", input_);
    #endif
}

void NvDlaCompile()
{
    p_nvdla_lib->optimize();
    p_nvdla_lib->compile();
    
    // p_cg->clear();
    // p_cg.release();
    // p_module.release();
    // p_nvdla_lib.release();
}

#ifdef __cplusplus
}
#endif
