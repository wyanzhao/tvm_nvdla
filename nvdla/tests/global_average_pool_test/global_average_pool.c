#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("GlobalAveragePoolTest");
    
    void* input_tensor = AddFloatTensor("data0", 4, 1, 1, 1, 1);

    AddInputOp(input_tensor);

    float weight[] ={1, 2, 1, 2};

    AddFloatWeightTensorFromNumpy("weight0", 4, weight, 4, 1, 1, 1);

    void* conv_op = AddConvOp("data0", "weight0");

    void* output_tensor = AddFloatTensor("activation0", 4, 1, 4, 1, 1);
    AddOutput(conv_op, output_tensor);

    void* global_averagepool = AddGlobalAveragePoolOp("activation0");
    
    AddOutput(global_averagepool, AddFloatTensor("output0", 4, 1, 4, 1, 1));

    AddOutputOp("output0");
    Compile();

    return 0;
}