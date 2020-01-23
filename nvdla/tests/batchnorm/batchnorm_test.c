#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("BatchNormTest");
    
    void* input_tensor = AddFloatTensor("data0", 4, 1, 1, 13, 16);

    AddInputOp(input_tensor);

    float weight[] ={3, 2, 3, 1, 2, 4, 2, 3, 1, 2, 4, 1, 3, 1, 3, 4, 1, 2, 2, 3, 3};

    AddFloatWeightTensorFromNumpy("weight0", 4, weight, 21, 1, 1, 1);

    void* conv_op = AddConvOp("data0", "weight0");

    void* output_tensor = AddFloatTensor("activation0", 4, 1, 21, 13, 16);
    AddOutput(conv_op, output_tensor);



    
    float scale[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    float b2[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    float mean3[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    float var4[] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    AddFloatWeightTensorFromNumpy("scale", 1, scale, 21);
    AddFloatWeightTensorFromNumpy("b2", 1, b2, 21);
    AddFloatWeightTensorFromNumpy("mean3", 1, mean3, 21);
    AddFloatWeightTensorFromNumpy("var4", 1, var4, 21);
    void* batchnorm_op = AddBatchNormOp("activation0", "scale", "b2", "mean3", "var4");
    void* output_tensor2 = AddFloatTensor("output0", 4, 1, 21, 13, 16);
    AddOutput(batchnorm_op, output_tensor2);

    AddOutputOp("output0");
    Compile();

    return 0;
}