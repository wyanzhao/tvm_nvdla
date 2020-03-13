#include <NvDlaRuntime.h>

int main()
{
    float w0[] = {3,
                2
                ,3
                ,1
                ,2
                ,4
                ,2
                ,3
                ,1
                ,2
                ,4
                ,1
                ,3
                ,1
                ,3
                ,4
                ,1
                ,2
                ,2
                ,3
                ,3};
  (void)NvDlaInit("Init");
  (void)AddInputOp(AddFloatTensor(54671808, 4, 1, 1, 13, 16));
  (void)AddFloatWeightTensorFromNumpy(54671744, 4, w0, 21, 1, 1, 1);
  (void)AddOutput(AddConvOp(54671808, 54671744), AddFloatTensor(54672272, 4, 1, 21, 13, 16));
  float scala1[] = {1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
};
  (void)AddFloatWeightTensorFromNumpy(54671504, 1, scala1, 21);
  (void)AddFloatWeightTensorFromNumpy(54673040, 1, scala1, 21);
  (void)AddFloatWeightTensorFromNumpy(54669664, 1, scala1, 21);
  (void)AddFloatWeightTensorFromNumpy(54669824, 1, scala1, 21);
  (void)AddOutput(AddBatchNormOp(54672272, 54671504, 54673040, 54669664, 54669824), AddFloatTensor(54671664, 4, 1, 21, 13, 16));
  (void)AddOutputOp(54671664);
  (void)NvDlaCompile();
    return 0;
}