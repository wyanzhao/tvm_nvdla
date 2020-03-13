#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("ReluTest");

    AddInputOp(AddFloatTensor("data0", 4, 1, 1, 3, 3));

    float weight[] = {3};
    AddFloatWeightTensorFromNumpy("weight0", 4, weight, 1, 1, 1, 1);

    void* conv_op = AddConvOp("data0", "weight0");
    
    AddOutput(conv_op, AddFloatTensor("buf0", 4, 1, 1, 3, 3));

    float weight2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    AddFloatWeightTensorFromNumpy("weight1", 4, weight2, 1, 1, 3, 3);
    void* add_op = AddAddOp("buf0", "weight1");  
        
    AddOutput(add_op, AddFloatTensor("output", 4, 1, 1, 3, 3));
    AddOutputOp("output");
    NvDlaCompile();

    return 0;
}