#include <NvDlaRuntime.h>

int main()
{
    float placeholder1[] = {3};
  (void)NvDlaInit("Init");
  (void)AddInputOp(AddFloatTensor(58048944, 4, 1, 1, 2, 3));
  (void)AddFloatWeightTensorFromNumpy(57957712, 4, placeholder1, 1, 1, 1, 1);
  (void)AddOutput(AddConvOp(58048944, 57957712), AddFloatTensor(57957312, 4, 1, 1, 2, 3));

  (void)AddOutput(AddMaxPoolOp(57957312, 2, 1, 1), AddFloatTensor(57938752, 4, 1, 1, 2, 3));
  (void)SetMaxPoolPads(GetOpPointer(57957312), 4, 0, 0, 0, 0);
  (void)SetMaxPoolStrides(GetOpPointer(57957312), 2, 1, 1);
  (void)AddOutputOp(57938752);
  (void)NvDlaCompile();
  
  return 0;
}