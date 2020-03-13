#include <NvDlaRuntime.h>

int main()
{
    float weight[] ={3};
    float B_12[] = {4,4,3,1,4,1,2,2,1,3,3,3,1,1,3,3,1,4,2,2,2,1,4,1,4,4,1,4,2,2,1,4,1,2,4,3
        ,1,4,4,3,3,2,3,1,4,3,3,2,3,2,4,4,4,1,1,4,4,3,4,3,3,1,4,4,4,1,4,4,4,2,1,3,2,1,4,2,2,1,2,
        4,1,2,1,3,3,4,2,1,2,3,4,1,4,1,3,1,2,4,3,4,1,2,1,4,1,1,1,3,3,1,1,2,4,2,3,2,4,1,2,3,2,3,2,1,1,3};
    float c13[] = {3,3,1,2,3,3,4};
    float B_24[] =  {4, 1,1,1,2, 1, 2};
    float C_25[] = {2};
    (void)NvDlaInit("Init");
    (void)AddInputOp(AddFloatTensor(40811168, 4, 1, 1, 6, 3));
    (void)AddFloatWeightTensorFromNumpy(40810928, 4, weight, 1, 1, 1, 1);
    (void)AddOutput(AddConvOp(40811168, 40810928), AddFloatTensor(40812048, 4, 1, 1, 6, 3));
    (void)AddOutput(AddReshapeOp(40812048, 2, 1, 18), AddFloatTensor(40811696, 2, 1, 18));

    (void)AddFloatWeightTensorFromNumpy(40810864, 2, B_12, 18, 7);
    (void)AddFloatWeightTensorFromNumpy(40807488, 2, c13, 1, 7);
    (void)AddOutput(AddGemmOp(40811696, 40810864, 40807488), AddFloatTensor(40811392, 2, 1, 7));
    (void)SetGemmAlpha(GetOpPointer(40811696), 1.0);
    (void)SetGemmBeta(GetOpPointer(40811696), 1.0);
    (void)SetGemmTransA(GetOpPointer(40811696), 0);
    (void)SetGemmTransB(GetOpPointer(40811696), 0);
    (void)AddOutput(AddReluOp(40811392), AddFloatTensor(40811088, 2, 1, 7));

    (void)AddFloatWeightTensorFromNumpy(40807664, 2, B_24, 7, 1);
    (void)AddFloatWeightTensorFromNumpy(40807904, 2, C_25, 1, 1);
    (void)AddOutput(AddGemmOp(40811088, 40807664, 40807904), AddFloatTensor(40812560, 2, 1, 1));
    (void)SetGemmAlpha(GetOpPointer(40811088), 1.0);
    (void)SetGemmBeta(GetOpPointer(40811088), 1.0);
     (void)SetGemmTransA(GetOpPointer(40811088), 0);
    (void)SetGemmTransB(GetOpPointer(40811088), 0);
    ( void)AddOutputOp(40812560);
    ( void)NvDlaCompile();
    return 0;
}