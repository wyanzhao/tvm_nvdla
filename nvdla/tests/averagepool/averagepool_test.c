#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("ConvTest");
    
    void* input_tensor = AddFloatTensor("data0", 4, 1, 1, 4, 2);

    AddInputOp(input_tensor);

    float weight[] ={3};

    AddFloatWeightTensorFromNumpy("weight0", 4, weight, 1, 1, 1, 1);

    void* conv_op = AddConvOp("data0", "weight0");

    void* output_tensor = AddFloatTensor("activation0", 4, 1, 1, 4, 2);
    AddOutput(conv_op, output_tensor);

    void* averagepool_op = AddAveragePoolOp("activation0", 2, 3, 1);
    SetAveragePoolAutoPad(averagepool_op, "NOTSET");
    SetAveragePoolCountIncludePad(averagepool_op, 1);
    SetAveragePoolKernelShape(averagepool_op, 2, 3, 1);
    SetAveragePoolPads(averagepool_op, 4, 0, 0, 0, 0);
    SetAveragePoolStrides(averagepool_op, 2, 1, 1);
    
    void* output_tensor2 = AddFloatTensor("output0", 4, 1, 1, 2, 2);
    AddOutput(averagepool_op, output_tensor2);

    AddOutputOp("output0");
    Compile();

    return 0;
}