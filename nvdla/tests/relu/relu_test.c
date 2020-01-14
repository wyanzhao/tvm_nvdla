#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("ReluTest");
    
    int dims1[4] = {1, 1, 3, 3};    
    void* input_tensor = AddFloatTensor("data0", dims1, 4);

    AddInputOp(input_tensor);

    void* op = AddReluOp("data0");
    int dims2[4] = {1, 1, 3, 3}; 
    void* output_tensor = AddFloatTensor("activation0", dims2, 4);
    AddOutput(op, output_tensor);

    AddOutputOp("activation0");
    Compile();

    return 0;
}