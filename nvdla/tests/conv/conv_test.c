#include <NvDlaRuntime.h>

int main()
{
    NvDlaInit("ConvTest");
    
    void* input_tensor = AddFloatTensor("data0", 4, 1, 1, 3, 3);

    AddInputOp(input_tensor);

    void* op = AddReluOp("data0");
    void* output_tensor = AddFloatTensor("activation0", 4, 1, 1, 3, 3);
    AddOutput(op, output_tensor);

    AddOutputOp("activation0");
    Compile();

    return 0;
}