#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("GemmTest");
    
    void* input_tensor = AddFloatTensor("data0", 4, 1, 1, 2, 1);
    AddInputOp(input_tensor);

    float weight[] ={3};
    AddFloatWeightTensorFromNumpy("weight0", 4, weight, 1, 1, 1, 1);

    void* conv_op = AddConvOp("data0", "weight0");
    SetConvStrides(conv_op, 2, 1, 1);
    AddOutput(conv_op, AddFloatTensor("activation0", 4, 1, 1, 2, 1));

    AddReshapeTensor("reshape_dummy_1", 2, 1, 2);    
    void* reshape_op = AddReshapeOp("activation0", "reshape_dummy_1");
    AddOutput(reshape_op, AddFloatTensor("reshape_output", 2, 1, 2));

    float b12[] = {2, 4, 1, 2, 4, 4, 4, 4, 3, 1, 1, 3, 3, 1};
    AddFloatWeightTensorFromNumpy("B12", 2, b12, 2, 7);
    float c13[] = {4, 1, 3, 4, 1, 2, 4};
    AddFloatWeightTensorFromNumpy("C13", 2, c13, 1, 7);
    void* gemm_op = AddGemmOp("reshape_output", "B12", "C13");
    AddOutput(gemm_op, AddFloatTensor("gemm_output", 2, 1, 7));

    float b24[] = { 4,
        1,
        2,
        4,
        2,
        4,
        3,
        2,
        3,
        4,
        3,
        3,
        4,
        1,
        3,
        3,
        4,
        1,
        2,
        1,
        1,
        1,
        3,
        4,
        2,
        2,
        4,
        1,
        2,
        4,
        1,
        1,
        3,
        2,
        4,
        3,
        1,
        4,
        1,
        2,
        1,
        2,
        2,
        3,
        4,
        1,
        2,
        2,
        2,
        1,
        3,
        1,
        2,
        4,
        3,
        3};
    AddFloatWeightTensorFromNumpy("B24", 2, b24, 7, 8);
    float c25[] = {3, 3, 3, 4, 1, 3, 1, 4};
    AddFloatWeightTensorFromNumpy("C25", 2, c25, 1, 8);
    void* gemm_op2 = AddGemmOp("gemm_output", "B24", "C25");
    AddOutput(gemm_op2, AddFloatTensor("gemm_output2", 2, 1, 8));

    AddOutputOp("gemm_output2");
    Compile();

    return 0;
}