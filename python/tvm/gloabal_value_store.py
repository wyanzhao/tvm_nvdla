def gemm_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 3
    i1, i2, i3 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    i3 = [int(x) for x in i3.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1] and i3 == op_input_shapes[2]:
        return True
    else:
        return False

def add_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 2
    i1, i2 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1]:
        return True
    else:
        return False

def reshape_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def relu_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def conv_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 2
    i1, i2 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1]:
        return True
    else:
        return False

def maxpool_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def global_avg_pool2d_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def batch_norm_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 5
    i1, i2, i3, i4, i5 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    i3 = [int(x) for x in i3.shape]
    i4 = [int(x) for x in i4.shape]
    i5 = [int(x) for x in i5.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1] and i3 == op_input_shapes[2] and \
    i4 == op_input_shapes[3] and i5 == op_input_shapes[4]:
        return True
    else:
        return False


global_stores = {
    'op_infos': [],
    "input_op": [],
    'op_maps': {
        "add":[],
        'reshape':[],
        'batch_norm':[],
        'relu':[],
        "copy":[],
        'conv2d':[],
        'max_pool2d':[],
        'global_avg_pool2d':[],
        'gemm':[]
    },
    "shape_functions": {
        "gemm": gemm_input_shapes,
        "add": add_input_shapes,
        "reshape": reshape_input_shapes,
        "relu": relu_input_shapes,
        "conv2d": conv_input_shapes,
        "max_pool2d": maxpool_input_shapes,
        "global_avg_pool2d": global_avg_pool2d_input_shapes,
        "batch_norm": batch_norm_input_shapes
    }
    ,
    "output_op": None
}


