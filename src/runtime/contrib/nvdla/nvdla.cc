#include <NvDlaRuntime.h>

#include <tvm/runtime/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>


namespace tvm {
namespace contrib {

using namespace runtime;


extern "C" void NvDlaAdd(const std::string& data, const std::string& weight, std::string& out, int p_N_,
                         int p_C_, int p_H_, int p_W_) 
{
    int64_t input_shape[4] = {p_N_, p_C_, p_H_, p_W_};
    auto input_tensor = AddFloatTensor(data.c_str(), 4, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    auto weight_tensor = AddFloatTensor(weight.c_str(), 4, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    auto op = AddAddOp(data.c_str(), weight.c_str());

    int64_t output_shape[4] = {p_N_, p_C_, p_H_, p_W_};
    auto output_tensor = AddFloatTensor(out.c_str(), 4, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    AddOutput(op, output_tensor);
};

extern "C" void NvDlaRelu(const std::string& data, const std::string& out, int p_N_,
                         int p_C_, int p_H_, int p_W_) 
{
    int64_t input_shape[4] = {p_N_, p_C_, p_H_, p_W_};
    auto input_tensor = AddFloatTensor(data.c_str(), 4, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

    auto op = AddReluOp(data.c_str());

    int64_t output_shape[4] = {p_N_, p_C_, p_H_, p_W_};
    auto output_tensor = AddFloatTensor(out.c_str(), 4, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    AddOutput(op, output_tensor);
};

extern "C" void NvDlaConv2D(const std::string& data, float* weights, const std::string& out, int p_N_,
                            int p_C_, int p_H_, int p_W_, int p_O_, int p_G_,
                            int p_Ph_, int p_Pw_, int p_Kh_, int p_Kw_,
                            int p_Sh_, int p_Sw_, int p_Dh_, int p_Dw_) 
{
    uint64_t input_shape[4] = {p_N_, p_C_, p_H_, p_W_};
    auto input_tensor = AddFloatTensor(data.c_str(), 4, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);


    uint64_t weight_shape[4] = {p_O_, p_C_, p_Kh_, p_Kw_};
    if(p_G_ != 1)
    {  
        printf("Unsupport Group Num:%d\n", p_G_);
        return;
    }

    uint64_t output_shape[4] = {p_N_, p_O_,
                                (p_H_ - p_Kh_ + 2 * p_Ph_ + p_Sh_) / p_Sh_,
                                (p_W_ - p_Kw_ + 2 * p_Pw_ + p_Sw_) / p_Sw_};
    
    auto weight_name = data + "weight" + out;
    AddFloatWeightTensorFromNumpy(weight_name.c_str(), "Add Conv2d Weight",4, weights, weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);

    auto op = AddConvOp(data.c_str(), weight_name.c_str());

    auto output_tensor = AddFloatTensor(out.c_str(), 4, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    AddOutput(op, output_tensor);

    int64_t dilations[2] = {p_Dh_, p_Dw_};
    int64_t pads[4] = {p_Ph_, p_Pw_, p_Ph_, p_Pw_};
    int64_t strides[2] = {p_Sh_, p_Sw_};

    SetConvDilations(op, 2, dilations[0], dilations[1]);
    SetConvGroup(op, 1);
    SetConvPads(op, 4, pads[0], pads[1], pads[2], pads[3]);
    SetConvStrides(op, 2, strides[0], strides[1]);
};

}};
