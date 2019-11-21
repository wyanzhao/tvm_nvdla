#include <nvdla_lib.h>

#include <onnc/IR/Compute/Concat.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Reshape.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/IR/Compute/Tensor.h>
#include <onnc/Support/Casting.h>
#include <onnc/Support/IOStream.h>
#include <onnc/Support/Timer.h>

using namespace onnc;

using namespace nvdla;

//void init_nvdla_memory(NvDlaBackendMeta* pMeta)
void init_nvdla_memory(Module& pModule, NvDlaBackendMeta* pMeta)
{
    // [0] entry of memory & address list
  {
    ILoadable::MemoryListEntry mle;
    mle.id             = pMeta->m_MemoryListEntries.size();
    mle.alignment      = 4096;
    mle.bind_id        = 0;
    mle.domain         = nvdla::ILoadable::MemoryDomain_SYSMEM;
    mle.flags          = nvdla::ILoadable::MemoryFlags_ALLOC;
    mle.size           = 4096;
    mle.tensor_desc_id = 0;
    pMeta->m_MemoryListEntries.push_back(mle);

    ILoadable::AddressListEntry ale;
    ale.size   = 4096;
    ale.offset = 0;
    ale.mem_id = mle.id;
    ale.id     = pMeta->m_AddressListEntries.size();
    pMeta->m_AddressListEntries.push_back(ale);
  }

  
  std::unordered_map<Value*, bool> isOutputMap;
  for (ComputeOperator& cm : *pModule.getRootComputeGraph()) {
    if (OutputOperator* out = dyn_cast<OutputOperator>(&cm)) {
      for (int i = 0; i < out->getNumOfInputs(); ++i) {
        Value* v       = out->getInput(i);
        isOutputMap[v] = true;
        Tensor* t      = static_cast<Tensor*>(v);
        for (auto j : t->getDimensions()) {
          NVDLA_DBG("output dim[%ld]\n", j);
        }
      }
    } 
  }
  printf("operands size:%d\n", pModule.getComputeOperands().size());

  for (ComputeOperand* co : pModule.getComputeOperands()) {
    NVDLA_DBG("ComputeOperand: ");
    if (ComputeMemOperand* mem = dyn_cast<ComputeMemOperand>(co)) {
      Value* v = co->getValue();
      if (mem->isWeight()) {
        // for weight, memory buffers are allocated & blob files are also generated in ComputeOperator.

        FloatTensor* t = static_cast<FloatTensor*>(v);
        NVDLA_DBG("weight size:%d %d\n", mem->length(), t->getValues().size());

      } else {
        NVDLA_DBG("operand size:%d\n", mem->length());
        // alocation only, no blobs
        Tensor* t = static_cast<Tensor*>(v);

        ILoadable::MemoryListEntry     mle;
        ILoadable::TensorDescListEntry tle;
        mle.id = pMeta->m_MemoryListEntries.size();

        int dims[4] = {1, 1, 1, 1};
        int idx     = 0;
        for (auto i : t->getDimensions())
          dims[idx++] = i;
        NvDlaCubeInfo cubeinfo(NVDLA_CUBE_FEATURE, dims[0], dims[1], dims[2], dims[3], sizeof(unsigned short));
        mle.size = cubeinfo.size;

        mle.alignment      = 4096;
        mle.flags          = nvdla::ILoadable::MemoryFlags_ALLOC;
        mle.domain         = nvdla::ILoadable::MemoryDomain_SYSMEM;
        mle.bind_id        = 0;
        mle.tensor_desc_id = 0;

        if (mem->isInput()) {
          mle.flags |= nvdla::ILoadable::MemoryFlags_INPUT;
          mle.tensor_desc_id = 0;

          tle.name   = "data";
          tle.id     = 0;
          tle.memId  = mle.id;
          tle.size   = mle.size;
          tle.offset = 0;

          tle.dims.n       = cubeinfo.dim_n;
          tle.dims.c       = cubeinfo.dim_c;
          tle.dims.h       = cubeinfo.dim_h;
          tle.dims.w       = cubeinfo.dim_w;
          tle.dataFormat   = 3;
          tle.dataType     = nvdla::loadable::DataType_HALF;
          tle.dataCategory = nvdla::loadable::DataCategory_FEATURE;
          tle.pixelFormat  = TENSOR_PIXEL_FORMAT_FEATURE;
          tle.pixelMapping = 0;

          tle.stride[0] = cubeinfo.stride_channel;
          tle.stride[1] = cubeinfo.stride_line;
          tle.stride[2] = cubeinfo.stride_surface;
          tle.stride[3] = 0;
          tle.stride[4] = 0;
          tle.stride[5] = 0;
          tle.stride[6] = 0;
          tle.stride[7] = 0;

          pMeta->m_TensorDescListEntries.emplace(pMeta->m_TensorDescListEntries.begin(), tle);
        } else if (isOutputMap.find(v) != isOutputMap.end()) {
          mle.flags |= nvdla::ILoadable::MemoryFlags_OUTPUT;
          mle.tensor_desc_id = 1;

          tle.name   = "probe";
          tle.id     = 1;
          tle.memId  = mle.id;
          tle.size   = mle.size;
          tle.offset = 0;

          tle.dims.n       = cubeinfo.dim_n;
          tle.dims.c       = cubeinfo.dim_c;
          tle.dims.h       = cubeinfo.dim_h;
          tle.dims.w       = cubeinfo.dim_w;
          tle.dataFormat   = 3;
          tle.dataType     = nvdla::loadable::DataType_HALF;
          tle.dataCategory = nvdla::loadable::DataCategory_FEATURE;
          tle.pixelFormat  = TENSOR_PIXEL_FORMAT_FEATURE;
          tle.pixelMapping = 0;

          tle.stride[0] = cubeinfo.stride_channel;
          tle.stride[1] = cubeinfo.stride_line;
          tle.stride[2] = cubeinfo.stride_surface;
          tle.stride[3] = 0;
          tle.stride[4] = 0;
          tle.stride[5] = 0;
          tle.stride[6] = 0;
          tle.stride[7] = 0;

          pMeta->m_TensorDescListEntries.push_back(tle);
        }
        pMeta->m_MemIdxTable[v] = pMeta->m_MemoryListEntries.size();
        pMeta->m_MemoryListEntries.push_back(mle);
      }
    }
  }
  
}
using namespace onnc;

void relu(const Relu& pOp, NvDlaBackendMeta* m_pMeta)
{
  pOp.print(errs());
  errs() << "\n";

  const Tensor* input_X_t       = pOp.getInput(0);
  int32_t       input_X_ndim    = input_X_t->getNumOfDimensions();
  int32_t       input_X_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_X_ndim; ++i)
    input_X_dims[i] = input_X_t->dimension(i);
  int                        X_mid = m_pMeta->m_MemIdxTable[(Tensor*)input_X_t];
  ILoadable::MemoryListEntry X_mle = m_pMeta->m_MemoryListEntries[X_mid];
  NvDlaCubeInfo X_cube(NVDLA_CUBE_FEATURE, input_X_dims[0], input_X_dims[1], input_X_dims[2], input_X_dims[3],
                       sizeof(short));

  const Tensor* output_Y_t       = pOp.getOutput(0);
  int32_t       output_Y_ndim    = output_Y_t->getNumOfDimensions();
  int32_t       output_Y_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_Y_ndim; ++i)
    output_Y_dims[i] = output_Y_t->dimension(i);

  int                        Y_mid;
  ILoadable::MemoryListEntry Y_mle;
  //concat_meta                meta;
//  if (m_pMeta->m_ConcatTable.find(output_Y_t) == m_pMeta->m_ConcatTable.end()) {
  //{
    Y_mid = m_pMeta->m_MemIdxTable[(Tensor*)output_Y_t];
    Y_mle = m_pMeta->m_MemoryListEntries[Y_mid];
  //}
  /* 
  else {
    printf("Concat Relu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    meta  = m_pMeta->m_ConcatTable[output_Y_t];
    Y_mid = m_pMeta->m_MemIdxTable[(Tensor*)meta.t];
    Y_mle = m_pMeta->m_MemoryListEntries[Y_mid];
  }*/
  NvDlaCubeInfo Y_cube(NVDLA_CUBE_FEATURE, output_Y_dims[0], output_Y_dims[1], output_Y_dims[2], output_Y_dims[3],
                       sizeof(short));

  NvDlaDlaOperation* relu_op = new NvDlaDlaOperation();
  relu_op->op_dep.op_type    = DLA_OP_SDP;

  struct dla_sdp_op_desc* relu_desc     = (struct dla_sdp_op_desc*)(&(relu_op->op_desc));
  relu_desc->src_precision              = PRECISION_FP16;
  relu_desc->dst_precision              = PRECISION_FP16;
  relu_desc->lut_index                  = -1;
  relu_desc->conv_mode                  = 0;
  relu_desc->out_cvt.scale              = 1;
  relu_desc->out_cvt.truncate           = 0;
  relu_desc->out_cvt.enable             = 1;
  relu_desc->out_cvt.offset             = 0;
  relu_desc->conv_mode                  = CONV_MODE_DIRECT;
  relu_desc->batch_num                  = 1;
  relu_desc->batch_stride               = 0;
  relu_desc->x1_op.enable               = 1;
  relu_desc->x1_op.alu_type             = SDP_ALU_OP_SUM;
  relu_desc->x1_op.type                 = SDP_OP_NONE;
  relu_desc->x1_op.mode                 = SDP_OP_PER_LAYER;
  relu_desc->x1_op.act                  = ACTIVATION_RELU;
  relu_desc->x1_op.shift_value          = 0;
  relu_desc->x1_op.truncate             = 0;
  relu_desc->x1_op.precision            = PRECISION_FP16;
  relu_desc->x1_op.alu_operand          = 0;
  relu_desc->x1_op.mul_operand          = 1;
  relu_desc->x1_op.cvt.alu_cvt.scale    = 0;
  relu_desc->x1_op.cvt.alu_cvt.truncate = 0;
  relu_desc->x1_op.cvt.alu_cvt.enable   = 0;
  relu_desc->x1_op.cvt.alu_cvt.offset   = 0;
  relu_desc->x1_op.cvt.mul_cvt.scale    = 0;
  relu_desc->x1_op.cvt.mul_cvt.truncate = 0;
  relu_desc->x1_op.cvt.mul_cvt.enable   = 0;
  relu_desc->x1_op.cvt.mul_cvt.offset   = 0;

  struct dla_sdp_surface_desc* relu_surf = (struct dla_sdp_surface_desc*)(&(relu_op->op_surf));
  relu_surf->src_data.type               = DLA_MEM_MC;
  relu_surf->src_data.address            = issueDlaAddr(X_mid, X_cube, 1, 0, 0, m_pMeta);
  relu_surf->src_data.size               = X_mle.size;
  relu_surf->src_data.width              = X_cube.dim_w;
  relu_surf->src_data.height             = X_cube.dim_h;
  relu_surf->src_data.channel            = X_cube.dim_c;
  relu_surf->src_data.line_stride        = X_cube.stride_line;
  relu_surf->src_data.surf_stride        = X_cube.stride_surface;
  relu_surf->src_data.plane_stride       = X_cube.stride_plane;

  relu_surf->dst_data.type = DLA_MEM_MC;
  //if (m_pMeta->m_ConcatTable.find(output_Y_t) == m_pMeta->m_ConcatTable.end())
  relu_surf->dst_data.address = issueDlaAddr(Y_mid, Y_cube, 1, 0, 0, m_pMeta);
  //else
      //relu_surf->dst_data.address = issueDlaAddr(Y_mid, Y_cube, -1, 0, meta.ofs);
  relu_surf->dst_data.size         = Y_mle.size;
  relu_surf->dst_data.width        = Y_cube.dim_w;
  relu_surf->dst_data.height       = Y_cube.dim_h;
  relu_surf->dst_data.channel      = Y_cube.dim_c;
  relu_surf->dst_data.line_stride  = Y_cube.stride_line;
  relu_surf->dst_data.surf_stride  = Y_cube.stride_surface;
  relu_surf->dst_data.plane_stride = Y_cube.stride_plane;

  issueDlaOp(relu_op, NULL, m_pMeta->m_pPrevOp, m_pMeta);
}

int issueDlaAddr(int mid, NvDlaCubeInfo cube, int groups, int gidx, int ofs, NvDlaBackendMeta* m_pMeta)
{
  int aid = m_pMeta->m_AddressListEntries.size();

  ILoadable::AddressListEntry ale;
  ILoadable::MemoryListEntry  mle = m_pMeta->m_MemoryListEntries[mid];

  ale.size = mle.size;
  if (groups >= 0) {
    int h_offset = ofs * cube.stride_line;
    ale.offset = ((gidx * (cube.dim_n * cube.dim_c * cube.dim_h * cube.dim_w * cube.element_size)) / groups) + h_offset;
  } else {
    int surf_offset = ofs / (32 / sizeof(short));
    ale.offset      = surf_offset * cube.stride_surface;
  }
  ale.mem_id = mid;
  ale.id     = aid;
  NVDLA_DBG("cube(%d %d %d %d %d), group(%d/%d) ofs %d\n", cube.dim_n, cube.dim_c, cube.dim_h, cube.dim_w,
            cube.element_size, groups, gidx, ofs);
  NVDLA_DBG("AddressEntry s:%9d o:%9d mid:%3d id:%3d\n", ale.size, ale.offset, ale.mem_id, ale.id);

  m_pMeta->m_AddressListEntries.push_back(ale);
  return aid;
}

void issueDlaOp(NvDlaDlaOperation* op, NvDlaDlaOperation* op_fuse, NvDlaDlaOperation* op_prev, NvDlaBackendMeta* m_pMeta)
{
  struct dla_common_op_desc* op_desc = &(op->op_dep);
  int                        op_type = op_desc->op_type;
  NVDLA_DBG("issueDlaOp: %d\n", op_type);
  op_desc->index            = m_pMeta->m_DLAOperationList.size();
  op_desc->roi_index        = 0;
  op_desc->dependency_count = 0;

  if (op_prev != NULL) {
    struct dla_common_op_desc* prev_op_desc = &(op_prev->op_dep);
    prev_op_desc->consumers[op_type].index  = op_desc->index;
    prev_op_desc->consumers[op_type].event  = 1;
    op_desc->dependency_count++;
  }

  if (m_pMeta->m_pDepOp[op_type] != NULL) {
    struct dla_common_op_desc* dep_op_desc = &(m_pMeta->m_pDepOp[op_type]->op_dep);
    if (m_pMeta->m_pDepOp[op_type] != op_prev) {
      dep_op_desc->consumers[op_type].index = op_desc->index;
      dep_op_desc->consumers[op_type].event = 2;
      op_desc->dependency_count++;
    }
  }

  m_pMeta->m_DlaNetworkDesc.op_head[op_type] = (m_pMeta->m_DlaNetworkDesc.op_head[op_type] < 0)
                                                ? m_pMeta->m_DLAOperationList.size()
                                                : m_pMeta->m_DlaNetworkDesc.op_head[op_type];
  m_pMeta->m_DLAOperationList.push_back(op);
  m_pMeta->m_pDepOp[op_type] = op;

  if (op_fuse != NULL) {
    struct dla_common_op_desc* fuse_op_desc = &(op_fuse->op_dep);
    int                        op_fuse_type = fuse_op_desc->op_type;
    fuse_op_desc->index                     = m_pMeta->m_DLAOperationList.size();
    fuse_op_desc->roi_index                 = 0;
    fuse_op_desc->dependency_count          = 1;

    fuse_op_desc->fused_parent.index = op_desc->index;
    fuse_op_desc->fused_parent.event = 3;

    op_desc->consumers[op_fuse_type].index = fuse_op_desc->index;
    op_desc->consumers[op_fuse_type].event = 2;
    if (op_prev != NULL) {
      struct dla_common_op_desc* prev_op_desc = &(op_prev->op_dep);
      prev_op_desc->consumers[op_type].event  = 1;
    }
    op_desc->dependency_count++;

    if (m_pMeta->m_pDepOp[op_fuse_type] != NULL) {
      struct dla_common_op_desc* dep_op_desc     = &(m_pMeta->m_pDepOp[op_fuse_type]->op_dep);
      dep_op_desc->consumers[op_fuse_type].index = fuse_op_desc->index;
      dep_op_desc->consumers[op_fuse_type].event = 2;

      fuse_op_desc->dependency_count++;
    }
    m_pMeta->m_pDepOp[op_fuse_type] = op_fuse;

    m_pMeta->m_DlaNetworkDesc.op_head[op_fuse_type] = (m_pMeta->m_DlaNetworkDesc.op_head[op_fuse_type] < 0)
                                                       ? m_pMeta->m_DLAOperationList.size()
                                                       : m_pMeta->m_DlaNetworkDesc.op_head[op_fuse_type];
    m_pMeta->m_DLAOperationList.push_back(op_fuse);
    m_pMeta->m_pPrevOp = op_fuse;
  } else {
    m_pMeta->m_pPrevOp = op;
  }
}

void task_submit(NvDlaBackendMeta* pMeta)
{
  int dla_start;
  {
    std::string blob_name = "task-0-addr0";

    ILoadable::Blob b;
    b.name              = blob_name;
    b.size              = sizeof(struct dla_network_desc);
    b.version.major     = 0;
    b.version.minor     = 11;
    b.version.sub_minor = 0;
    b.interface         = ILoadable::Interface_DLA1;
    b.subInterface      = 0;
    NvU8* blob_data     = new NvU8[b.size];

    pMeta->m_DlaNetworkDesc.operation_desc_index   = pMeta->m_AddressListEntries.size() + 2;
    pMeta->m_DlaNetworkDesc.surface_desc_index     = pMeta->m_AddressListEntries.size() + 3;
    pMeta->m_DlaNetworkDesc.dependency_graph_index = pMeta->m_AddressListEntries.size() + 1;
    pMeta->m_DlaNetworkDesc.lut_data_index         = pMeta->m_AddressListEntries.size() + 4;
    pMeta->m_DlaNetworkDesc.roi_array_index        = -1;
    pMeta->m_DlaNetworkDesc.surface_index          = -1;
    pMeta->m_DlaNetworkDesc.stat_list_index        = -1;
    pMeta->m_DlaNetworkDesc.stat_list_index        = -1;

    pMeta->m_DlaNetworkDesc.num_rois       = 1;
    pMeta->m_DlaNetworkDesc.num_operations = pMeta->m_DLAOperationList.size();
    pMeta->m_DlaNetworkDesc.num_luts       = pMeta->m_NumLUTs;
    pMeta->m_DlaNetworkDesc.num_addresses  = pMeta->m_AddressListEntries.size() + 5;

    pMeta->m_DlaNetworkDesc.input_layer = 0;
    pMeta->m_DlaNetworkDesc.dynamic_roi = 0;

    memcpy(blob_data, &(pMeta->m_DlaNetworkDesc), sizeof(struct dla_network_desc));

    pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
    dla_start = submitMemAllocAddress(b.size, blob_name, pMeta);
  }

  {
    std::string     blob_name = "task-0-dep_graph";
    ILoadable::Blob b;
    b.name              = blob_name;
    b.size              = pMeta->m_DLAOperationList.size() * sizeof(struct dla_common_op_desc);
    b.version.major     = 0;
    b.version.minor     = 11;
    b.version.sub_minor = 0;
    b.interface         = ILoadable::Interface_DLA1;
    b.subInterface      = 0;

    NvU8*                      blob_data = new NvU8[b.size];
    struct dla_common_op_desc* op_blob   = (struct dla_common_op_desc*)blob_data;
    for (int i = 0; i < pMeta->m_DLAOperationList.size(); i++) {
      NvDlaDlaOperation* op = pMeta->m_DLAOperationList[i];
      memcpy(op_blob + i, &(op->op_dep), sizeof(struct dla_common_op_desc));
    }

    pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
    submitMemAllocAddress(b.size, blob_name, pMeta);
  }

  {
    std::string     blob_name = "task-0-op_list";
    ILoadable::Blob b;
    b.name              = blob_name;
    b.size              = pMeta->m_DLAOperationList.size() * sizeof(union dla_operation_container);
    b.version.major     = 0;
    b.version.minor     = 11;
    b.version.sub_minor = 0;
    b.interface         = ILoadable::Interface_DLA1;
    b.subInterface      = 0;

    NvU8*                          blob_data = new NvU8[b.size];
    union dla_operation_container* op_blob   = (union dla_operation_container*)blob_data;
    for (int i = 0; i < pMeta->m_DLAOperationList.size(); i++) {
      NvDlaDlaOperation* op = pMeta->m_DLAOperationList[i];
      memcpy(op_blob + i, &(op->op_desc), sizeof(union dla_operation_container));
    }

    pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
    submitMemAllocAddress(b.size, blob_name, pMeta);
  }

  {
    std::string     blob_name = "task-0-surf_list";
    ILoadable::Blob b;
    b.name              = blob_name;
    b.size              = pMeta->m_DLAOperationList.size() * sizeof(union dla_surface_container);
    b.version.major     = 0;
    b.version.minor     = 11;
    b.version.sub_minor = 0;
    b.interface         = ILoadable::Interface_DLA1;
    b.subInterface      = 0;

    NvU8*                        blob_data = new NvU8[b.size];
    union dla_surface_container* op_blob   = (union dla_surface_container*)blob_data;
    for (int i = 0; i < pMeta->m_DLAOperationList.size(); i++) {
      NvDlaDlaOperation* op = pMeta->m_DLAOperationList[i];
      memcpy(op_blob + i, &(op->op_surf), sizeof(union dla_surface_container));
    }

    pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
    submitMemAllocAddress(b.size, blob_name, pMeta);
  }

  {
    std::string     blob_name = "task-0-lut_list";
    ILoadable::Blob b;
    b.name              = blob_name;
    b.size              = pMeta->m_LUTList.size() * sizeof(struct dla_lut_param);
    b.version.major     = 0;
    b.version.minor     = 11;
    b.version.sub_minor = 0;
    b.interface         = ILoadable::Interface_DLA1;
    b.subInterface      = 0;

    NvU8*                 blob_data = new NvU8[b.size];
    struct dla_lut_param* op_blob   = (struct dla_lut_param*)blob_data;
    for (int i = 0; i < pMeta->m_LUTList.size(); i++) {
      struct dla_lut_param* lut = pMeta->m_LUTList[i];
      memcpy(op_blob + i, lut, sizeof(struct dla_lut_param));
    }

    pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
    submitMemAllocAddress(b.size, blob_name, pMeta);
  }

  {
    ILoadable::MemoryListEntry mle;
    mle.id             = pMeta->m_MemoryListEntries.size();
    mle.alignment      = 4096;
    mle.bind_id        = 0;
    mle.domain         = nvdla::ILoadable::MemoryDomain_SYSMEM;
    mle.flags          = nvdla::ILoadable::MemoryFlags_ALLOC;
    mle.size           = 4096;
    mle.tensor_desc_id = 0;
    pMeta->m_MemoryListEntries.push_back(mle);

    ILoadable::AddressListEntry ale;
    ale.size   = 0;
    ale.offset = 0;
    ale.mem_id = mle.id;
    ale.id     = pMeta->m_AddressListEntries.size();
    pMeta->m_AddressListEntries.push_back(ale);
  }

  {
    ILoadable::SubmitListEntry sle;
    ILoadable::TaskListEntry   tle;

    tle.id        = pMeta->m_TaskListEntries.size();
    tle.interface = ILoadable::Interface_DLA1;
    tle.instance  = -1;

    tle.preactions.push_back(submitEvent(tle.id, NVDLA_LOADABLE_EVENT_OP_SIGNAL, pMeta));

    tle.address_list.push_back(dla_start);
    for (int i = 1; i < pMeta->m_AddressListEntries.size(); i++)
      tle.address_list.push_back(i);
    pMeta->m_TaskListEntries.push_back(tle);

    sle.id = pMeta->m_SubmitListEntries.size();
    sle.tasks.push_back(tle.id);
    pMeta->m_SubmitListEntries.push_back(sle);
  }

  if (pMeta->m_EMUOperationList.size()) {
    int emu_start;
    {
      std::string blob_name = "task-1-addr0";

      ILoadable::Blob b;
      b.name              = blob_name;
      b.size              = sizeof(struct emu_network_desc);
      b.version.major     = 0;
      b.version.minor     = 0;
      b.version.sub_minor = 1;
      b.interface         = ILoadable::Interface_EMU1;
      b.subInterface      = 0;
      NvU8* blob_data     = new NvU8[b.size];

      pMeta->m_EmuNetworkDesc.operation_desc_index        = pMeta->m_AddressListEntries.size() + 1;
      pMeta->m_EmuNetworkDesc.operation_buffer_desc_index = pMeta->m_AddressListEntries.size() + 2;
      pMeta->m_EmuNetworkDesc.num_operations              = pMeta->m_EMUOperationList.size();
      memcpy(blob_data, &(pMeta->m_EmuNetworkDesc), sizeof(struct emu_network_desc));

      pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
      emu_start = submitMemAllocAddress(b.size, blob_name, pMeta);
    }

    {
      std::string blob_name = "task-1-op_list";

      ILoadable::Blob b;
      b.name              = blob_name;
      b.size              = pMeta->m_EMUOperationList.size() * sizeof(union emu_operation_container);
      b.version.major     = 0;
      b.version.minor     = 0;
      b.version.sub_minor = 1;
      b.interface         = ILoadable::Interface_EMU1;
      b.subInterface      = 0;
      NvU8*                          blob_data = new NvU8[b.size];
      union emu_operation_container* op_blob   = (union emu_operation_container*)blob_data;
      for (int i = 0; i < pMeta->m_EMUOperationList.size(); i++) {
        NvDlaEmuOperation* op = pMeta->m_EMUOperationList[i];
        memcpy(op_blob + i, &(op->op_desc), sizeof(union emu_operation_container));
      }

      pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
      submitMemAllocAddress(b.size, blob_name, pMeta);
    }

    {
      std::string blob_name = "task-1-op_buf_list";

      ILoadable::Blob b;
      b.name              = blob_name;
      b.size              = pMeta->m_EMUOperationList.size() * sizeof(union emu_operation_buffer_container);
      b.version.major     = 0;
      b.version.minor     = 0;
      b.version.sub_minor = 1;
      b.interface         = ILoadable::Interface_EMU1;
      b.subInterface      = 0;
      NvU8*                                 blob_data = new NvU8[b.size];
      union emu_operation_buffer_container* op_blob   = (union emu_operation_buffer_container*)blob_data;
      for (int i = 0; i < pMeta->m_EMUOperationList.size(); i++) {
        NvDlaEmuOperation* op = pMeta->m_EMUOperationList[i];
        memcpy(op_blob + i, &(op->op_buf), sizeof(union emu_operation_buffer_container));
      }

      pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
      submitMemAllocAddress(b.size, blob_name, pMeta);
    }

    {
      for (int i = 0; i < 3; i++) {
        ILoadable::MemoryListEntry mle;
        mle.id             = pMeta->m_MemoryListEntries.size();
        mle.alignment      = 4096;
        mle.bind_id        = 0;
        mle.domain         = nvdla::ILoadable::MemoryDomain_SYSMEM;
        mle.flags          = nvdla::ILoadable::MemoryFlags_ALLOC;
        mle.size           = 4096;
        mle.tensor_desc_id = 0;
        pMeta->m_MemoryListEntries.push_back(mle);

        ILoadable::AddressListEntry ale;
        ale.size   = 0;
        ale.offset = 0;
        ale.mem_id = mle.id;
        ale.id     = pMeta->m_AddressListEntries.size();
        pMeta->m_AddressListEntries.push_back(ale);
      }

      int                        prepost_cnt = 0;
      ILoadable::SubmitListEntry sle;
      ILoadable::TaskListEntry   tle;

      tle.id        = pMeta->m_TaskListEntries.size();
      tle.interface = ILoadable::Interface_EMU1;
      tle.instance  = -1;

      tle.preactions.push_back(submitEvent(tle.id, NVDLA_LOADABLE_EVENT_OP_WAIT, pMeta));
      tle.postactions.push_back(submitEvent(tle.id, NVDLA_LOADABLE_EVENT_OP_SIGNAL, pMeta));
      tle.address_list.push_back(emu_start);
      for (int i = 1; i < pMeta->m_AddressListEntries.size(); i++)
        tle.address_list.push_back(i);
      pMeta->m_TaskListEntries.push_back(tle);

      sle.id = pMeta->m_SubmitListEntries.size();
      sle.tasks.push_back(tle.id);
      pMeta->m_SubmitListEntries.push_back(sle);
    }
  }
}

int submitEvent(int task_id, int event_type, NvDlaBackendMeta* pMeta)
{
  ILoadable::EventListEntry ele;
  ele.id     = pMeta->m_EventListEntries.size();
  ele.op     = event_type;
  ele.target = 0;
  ele.val    = task_id + ele.op;

  pMeta->m_EventListEntries.push_back(ele);
  return ele.id;
}

int submitMemAllocAddress(int size, std::string blob_name, NvDlaBackendMeta* pMeta)
{
  int aid = pMeta->m_AddressListEntries.size();

  ILoadable::AddressListEntry ale;

  ILoadable::MemoryListEntry mle;
  mle.size           = size;
  mle.id             = pMeta->m_MemoryListEntries.size();
  mle.alignment      = 4096;
  mle.flags          = nvdla::ILoadable::MemoryFlags_ALLOC | nvdla::ILoadable::MemoryFlags_SET;
  mle.domain         = nvdla::ILoadable::MemoryDomain_SYSMEM;
  mle.bind_id        = 0;
  mle.tensor_desc_id = 0;
  mle.contents.push_back(blob_name);
  mle.offsets.push_back(0);
  pMeta->m_MemoryListEntries.push_back(mle);

  ale.size   = 0;
  ale.offset = 0;
  ale.mem_id = mle.id;
  ale.id     = aid;

  NVDLA_DBG("AddressEntry s:%9d o:%9d mid:%3d id:%3d\n", ale.size, ale.offset, ale.mem_id, ale.id);
  pMeta->m_AddressListEntries.push_back(ale);
  return aid;
}

void nvdla_filegen(NvDlaBackendMeta* pMeta)
{
  pMeta->m_Loadable.priv()->setMemoryListEntries(pMeta->m_MemoryListEntries);
  pMeta->m_Loadable.priv()->setTensorDescListEntries(pMeta->m_TensorDescListEntries);
  pMeta->m_Loadable.priv()->setAddressListEntries(pMeta->m_AddressListEntries);
  pMeta->m_Loadable.priv()->setEventListEntries(pMeta->m_EventListEntries);
  pMeta->m_Loadable.priv()->setTaskListEntries(pMeta->m_TaskListEntries);
  pMeta->m_Loadable.priv()->setSubmitListEntries(pMeta->m_SubmitListEntries);
  pMeta->m_Loadable.priv()->serialize();
}