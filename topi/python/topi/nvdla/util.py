
from tvm.gloabal_value_store import global_stores


def find_op_info(op_name, op_shape, op_inputs):
    global global_stores
    op_list = global_stores['op_maps'].get(op_name)
    if op_list == None:
        raise ValueError("Can't find op_list in op_maps:{}".format(op_name))
    op_info = None

    if len(global_stores['op_maps'].get(op_name)) <= 0:
        raise ValueError("Empty Op_Map List:{}".format(op_name))

    for x in range(len(global_stores['op_maps'].get(op_name)) - 1, -1, -1):
        node = global_stores['op_maps'][op_name][x]
        func = global_stores['shape_functions'][op_name]
        if node['op_shape'] == op_shape and func(node['input_shapes'], op_inputs):
            op_info = node
            del global_stores['op_maps'][op_name][x]
            break
        
    if op_info == None:
        raise ValueError("Can't find Op in Relay Graph")
    return op_info 