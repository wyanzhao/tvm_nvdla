#include <NvDlaLib.h>
#include <NvDlaUtil.h>
#include <NvDlaMeta.h>

#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Tensor.h>
#include <onnc/Support/Casting.h>
#include <onnc/Support/IOStream.h>
#include <onnc/Support/String.h>
#include <onnc/Support/Timer.h>
#include <onnc/Support/Algorithm.h>
#include <onnc/Core/PassSupport.h>
#include <onnc/IR/Compute/Reshape.h>

#include <cassert>
#include <unordered_set>
#include <fstream>

NvDlaLib::~NvDlaLib()
{
  delete m_pModule;
  delete m_pCG;
}


Tensor* NvDlaLib::create_float_compute_tensor(const StringRef& pName,
                         const Tensor::Dimensions& pDims)
{
  return this->create_compute_tensor<FloatTensor>(pName, pDims);
}


int NvDlaLib::submit_event(int task_id, int event_type)
{
  ILoadable::EventListEntry ele;
  ele.id     = m_pMeta->m_EventListEntries.size();
  ele.op     = event_type;
  ele.target = 0;
  ele.val    = task_id + ele.op;

  m_pMeta->m_EventListEntries.push_back(ele);
  return ele.id;
}

int NvDlaLib::submit_mem_alloc_address(int size, std::string& blob_name)
{
  int aid = m_pMeta->m_AddressListEntries.size();

  ILoadable::AddressListEntry ale;

  ILoadable::MemoryListEntry mle;
  mle.size           = size;
  mle.id             = m_pMeta->m_MemoryListEntries.size();
  mle.alignment      = 4096;
  mle.flags          = ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_SET;
  mle.domain         = ILoadable::MemoryDomain_SYSMEM;
  mle.bind_id        = 0;
  mle.tensor_desc_id = 0;
  mle.contents.push_back(blob_name);
  mle.offsets.push_back(0);
  m_pMeta->m_MemoryListEntries.push_back(mle);

  ale.size   = 0;
  ale.offset = 0;
  ale.mem_id = mle.id;
  ale.id     = aid;

  m_pMeta->m_AddressListEntries.push_back(ale);
  return aid;
}

void NvDlaLib::task_submit()
{
  using OperationCategory = NvDlaBackendMeta::OperationMeta::Category;

  unsigned taskIndex = 0;
  for (std::size_t iTaskStart = 0; iTaskStart < m_pMeta->m_OperationMetas.size(); ++taskIndex) {
    const OperationCategory category = m_pMeta->m_OperationMetas[iTaskStart].category;

    // find last operation (in task) which has same category
    std::size_t iTaskEnd = iTaskStart + 1;
    for (; iTaskEnd < m_pMeta->m_OperationMetas.size(); ++iTaskEnd) {
      if (m_pMeta->m_OperationMetas[iTaskEnd].category != category) {
        break;
      }
    }
    assert(iTaskEnd <= m_pMeta->m_OperationMetas.size());

    const std::size_t numTasks = (iTaskEnd - iTaskStart);
    // submit for different type tasks
    if (category == OperationCategory::dla) {
      int dla_start;
      {
        std::string blob_name = to_string("task-", taskIndex, "-addr0");

        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = sizeof(struct dla_network_desc);
        b.interface    = ILoadable::Interface_DLA1;
        b.subInterface = 0;
        //assign(b.version, m_DlaVersion);

        NvU8* blob_data = new NvU8[b.size];

        m_pMeta->m_DlaNetworkDesc.operation_desc_index   = m_pMeta->m_AddressListEntries.size() + 2;
        m_pMeta->m_DlaNetworkDesc.surface_desc_index     = m_pMeta->m_AddressListEntries.size() + 3;
        m_pMeta->m_DlaNetworkDesc.dependency_graph_index = m_pMeta->m_AddressListEntries.size() + 1;
        m_pMeta->m_DlaNetworkDesc.lut_data_index =
          (m_pMeta->m_LUTList.empty() ? -1 : m_pMeta->m_AddressListEntries.size() + 4);
        m_pMeta->m_DlaNetworkDesc.roi_array_index = -1;
        m_pMeta->m_DlaNetworkDesc.surface_index   = -1;
        m_pMeta->m_DlaNetworkDesc.stat_list_index = -1;
        m_pMeta->m_DlaNetworkDesc.stat_list_index = -1;

        m_pMeta->m_DlaNetworkDesc.num_rois       = 1;
        m_pMeta->m_DlaNetworkDesc.num_operations = numTasks;
        m_pMeta->m_DlaNetworkDesc.num_luts       = m_pMeta->m_NumLUTs;
        m_pMeta->m_DlaNetworkDesc.num_addresses  = m_pMeta->m_AddressListEntries.size() + 5;

        m_pMeta->m_DlaNetworkDesc.input_layer = 0;
        m_pMeta->m_DlaNetworkDesc.dynamic_roi = 0;

        memcpy(blob_data, &(m_pMeta->m_DlaNetworkDesc), sizeof(struct dla_network_desc));

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        dla_start = submit_mem_alloc_address(b.size, blob_name);
      }

      {
        std::string     blob_name = to_string("task-", taskIndex, "-dep_graph");
        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = numTasks * sizeof(struct dla_common_op_desc);
        b.interface    = ILoadable::Interface_DLA1;
        b.subInterface = 0;
        //assign(b.version, m_DlaVersion);

        NvU8*                      blob_data = new NvU8[b.size];
        struct dla_common_op_desc* op_blob   = (struct dla_common_op_desc*)blob_data;
        for (std::size_t i = iTaskStart; i < iTaskEnd; i++) {
          const auto& opMeta = m_pMeta->m_OperationMetas[i];

          NvDlaDlaOperation* op = m_pMeta->m_DLAOperationList[opMeta.index];
          memcpy(op_blob + (i - iTaskStart), &(op->op_dep), sizeof(struct dla_common_op_desc));
        }

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        submit_mem_alloc_address(b.size, blob_name);
      }

      {
        std::string     blob_name = to_string("task-", taskIndex, "-op_list");
        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = numTasks * sizeof(union dla_operation_container);
        b.interface    = ILoadable::Interface_DLA1;
        b.subInterface = 0;
        //assign(b.version, m_DlaVersion);

        NvU8*                          blob_data = new NvU8[b.size];
        union dla_operation_container* op_blob   = (union dla_operation_container*)blob_data;
        for (std::size_t i = iTaskStart; i < iTaskEnd; i++) {
          const auto& opMeta = m_pMeta->m_OperationMetas[i];

          NvDlaDlaOperation* op = m_pMeta->m_DLAOperationList[opMeta.index];
          memcpy(op_blob + (i - iTaskStart), &(op->op_desc), sizeof(union dla_operation_container));
        }

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        submit_mem_alloc_address(b.size, blob_name);
      }

      {
        std::string     blob_name = to_string("task-", taskIndex, "-surf_list");
        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = numTasks * sizeof(union dla_surface_container);
        b.interface    = ILoadable::Interface_DLA1;
        b.subInterface = 0;
        //assign(b.version, m_DlaVersion);

        NvU8*                        blob_data = new NvU8[b.size];
        union dla_surface_container* op_blob   = (union dla_surface_container*)blob_data;
        for (std::size_t i = iTaskStart; i < iTaskEnd; i++) {
          const auto& opMeta = m_pMeta->m_OperationMetas[i];

          NvDlaDlaOperation* op = m_pMeta->m_DLAOperationList[opMeta.index];
          memcpy(op_blob + (i - iTaskStart), &(op->op_surf), sizeof(union dla_surface_container));
        }

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        submit_mem_alloc_address(b.size, blob_name);
      }

      if (!m_pMeta->m_LUTList.empty()) {
        std::string     blob_name = to_string("task-", taskIndex, "-lut_list");
        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = m_pMeta->m_LUTList.size() * sizeof(struct dla_lut_param);
        b.interface    = ILoadable::Interface_DLA1;
        b.subInterface = 0;
        //assign(b.version, m_DlaVersion);

        NvU8*                 blob_data = new NvU8[b.size];
        struct dla_lut_param* op_blob   = (struct dla_lut_param*)blob_data;
        for (int i = 0; i < m_pMeta->m_LUTList.size(); i++) {
          struct dla_lut_param* lut = m_pMeta->m_LUTList[i];
          memcpy(op_blob + i, lut, sizeof(struct dla_lut_param));
        }

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        submit_mem_alloc_address(b.size, blob_name);
      }

      {
        ILoadable::MemoryListEntry mle;
        mle.id             = m_pMeta->m_MemoryListEntries.size();
        mle.alignment      = 4096;
        mle.bind_id        = 0;
        mle.domain         = ILoadable::MemoryDomain_SYSMEM;
        mle.flags          = ILoadable::MemoryFlags_ALLOC;
        mle.size           = 4096;
        mle.tensor_desc_id = 0;
        m_pMeta->m_MemoryListEntries.push_back(mle);

        ILoadable::AddressListEntry ale;
        ale.size   = 0;
        ale.offset = 0;
        ale.mem_id = mle.id;
        ale.id     = m_pMeta->m_AddressListEntries.size();
        m_pMeta->m_AddressListEntries.push_back(ale);
      }

      {
        ILoadable::SubmitListEntry sle;
        ILoadable::TaskListEntry   tle;

        tle.id        = m_pMeta->m_TaskListEntries.size();
        tle.interface = ILoadable::Interface_DLA1;
        tle.instance  = -1;

        if (0 < taskIndex) {
          tle.preactions.push_back(submit_event(tle.id, NVDLA_LOADABLE_EVENT_OP_WAIT));
          tle.postactions.push_back(submit_event(tle.id, NVDLA_LOADABLE_EVENT_OP_SIGNAL));
        } else {
          tle.preactions.push_back(submit_event(tle.id, NVDLA_LOADABLE_EVENT_OP_SIGNAL));
        }

        tle.address_list.push_back(dla_start);
        for (int i = 1; i < m_pMeta->m_AddressListEntries.size(); i++)
          tle.address_list.push_back(i);
        m_pMeta->m_TaskListEntries.push_back(tle);

        sle.id = m_pMeta->m_SubmitListEntries.size();
        sle.tasks.push_back(tle.id);
        m_pMeta->m_SubmitListEntries.push_back(sle);
      }
    } else if (category == OperationCategory::emu) {
      int emu_start;
      {
        std::string blob_name = to_string("task-", taskIndex, "-addr0");

        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = sizeof(struct emu_network_desc);
        b.interface    = ILoadable::Interface_EMU1;
        b.subInterface = 0;
        //assign(b.version, m_EmuVersion);

        NvU8* blob_data = new NvU8[b.size];

        m_pMeta->m_EmuNetworkDesc.operation_desc_index        = m_pMeta->m_AddressListEntries.size() + 1;
        m_pMeta->m_EmuNetworkDesc.operation_buffer_desc_index = m_pMeta->m_AddressListEntries.size() + 2;
        m_pMeta->m_EmuNetworkDesc.num_operations              = numTasks;
        memcpy(blob_data, &(m_pMeta->m_EmuNetworkDesc), sizeof(struct emu_network_desc));

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        emu_start = submit_mem_alloc_address(b.size, blob_name);
      }

      {
        std::string blob_name = to_string("task-", taskIndex, "-op_list");

        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = numTasks * sizeof(union emu_operation_container);
        b.interface    = ILoadable::Interface_EMU1;
        b.subInterface = 0;
        //assign(b.version, m_EmuVersion);

        NvU8*                          blob_data = new NvU8[b.size];
        union emu_operation_container* op_blob   = (union emu_operation_container*)blob_data;
        for (std::size_t i = iTaskStart; i < iTaskEnd; i++) {
          const auto& opMeta = m_pMeta->m_OperationMetas[i];

          NvDlaEmuOperation* op = m_pMeta->m_EMUOperationList[opMeta.index];
          memcpy(op_blob + (i - iTaskStart), &(op->op_desc), sizeof(union emu_operation_container));
        }

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        submit_mem_alloc_address(b.size, blob_name);
      }

      {
        std::string blob_name = to_string("task-", taskIndex, "-op_buf_list");

        ILoadable::Blob b;
        b.name         = blob_name;
        b.size         = numTasks * sizeof(union emu_operation_buffer_container);
        b.interface    = ILoadable::Interface_EMU1;
        b.subInterface = 0;
        //assign(b.version, m_EmuVersion);

        NvU8*                                 blob_data = new NvU8[b.size];
        union emu_operation_buffer_container* op_blob   = (union emu_operation_buffer_container*)blob_data;
        for (std::size_t i = iTaskStart; i < iTaskEnd; i++) {
          const auto& opMeta = m_pMeta->m_OperationMetas[i];

          NvDlaEmuOperation* op = m_pMeta->m_EMUOperationList[opMeta.index];
          memcpy(op_blob + (i - iTaskStart), &(op->op_buf), sizeof(union emu_operation_buffer_container));
        }

        m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);
        submit_mem_alloc_address(b.size, blob_name);
      }

      {
        for (int i = 0; i < 3; i++) {
          ILoadable::MemoryListEntry mle;
          mle.id             = m_pMeta->m_MemoryListEntries.size();
          mle.alignment      = 4096;
          mle.bind_id        = 0;
          mle.domain         = ILoadable::MemoryDomain_SYSMEM;
          mle.flags          = ILoadable::MemoryFlags_ALLOC;
          mle.size           = 4096;
          mle.tensor_desc_id = 0;
          m_pMeta->m_MemoryListEntries.push_back(mle);

          ILoadable::AddressListEntry ale;
          ale.size   = 0;
          ale.offset = 0;
          ale.mem_id = mle.id;
          ale.id     = m_pMeta->m_AddressListEntries.size();
          m_pMeta->m_AddressListEntries.push_back(ale);
        }

        int                        prepost_cnt = 0;
        ILoadable::SubmitListEntry sle;
        ILoadable::TaskListEntry   tle;

        tle.id        = m_pMeta->m_TaskListEntries.size();
        tle.interface = ILoadable::Interface_EMU1;
        tle.instance  = -1;

        tle.preactions.push_back(submit_event(tle.id, NVDLA_LOADABLE_EVENT_OP_WAIT));
        tle.postactions.push_back(submit_event(tle.id, NVDLA_LOADABLE_EVENT_OP_SIGNAL));
        tle.address_list.push_back(emu_start);
        for (int i = 1; i < m_pMeta->m_AddressListEntries.size(); i++)
          tle.address_list.push_back(i);
        m_pMeta->m_TaskListEntries.push_back(tle);

        sle.id = m_pMeta->m_SubmitListEntries.size();
        sle.tasks.push_back(tle.id);
        m_pMeta->m_SubmitListEntries.push_back(sle);
      }
    }

    iTaskStart = iTaskEnd;
  }

}


void NvDlaLib::init_nvdla_memory()
{
  //using namespace nvdla;
  // [0] entry of memory & address list
  {
    const MemoryListEntryId memoryId =
      m_pMeta->allocateMemory(ILoadable::MemoryDomain_SYSMEM, ILoadable::MemoryFlags_ALLOC, 4096);
    m_pMeta->acquireMemory(memoryId, 0, 4096);
  }

  using std::begin;
  using std::end;

  std::unordered_set<const Tensor*> outputTensors;
  std::vector<const Tensor*>        tensors;
  for (ComputeOperator& cm : *(m_pModule->getRootComputeGraph())) {
    if (OutputOperator* outputOperator = dyn_cast<OutputOperator>(&cm)) {
      for (unsigned idx = 0; idx < outputOperator->getNumOfInputs(); ++idx) {
        outputTensors.insert(static_cast<const Tensor*>(outputOperator->getInput(idx)));
      }
    }

    for (unsigned idx = 0; idx < cm.getNumOfOutputs(); ++idx) {
      const Tensor* output = static_cast<const Tensor*>(cm.getOutput(idx));
      if (std::find(begin(tensors), end(tensors), output) == end(tensors)) {
        tensors.emplace_back(output);
      }
    }
  }
  assert(!outputTensors.empty());

  using std::end;
  const auto isOutput = [&outputTensors](const Tensor* tensor) {
    return outputTensors.find(tensor) != end(outputTensors);
  };

  for (const Tensor* tensor : tensors) {
    if (isConstant(*tensor)) {
      continue;
    }

    // skip allocating memory for Reshape input tensors
    if (!(m_pMeta->shouldOwnMemory(*tensor) || isOutput(tensor))) {
      continue;
    }

    // skip for already-allocated-memory tensors
    if (m_pMeta->hasMemoryListEntry(*tensor)) {
      continue;
    }

    int dims[4] = {1, 1, 1, 1};
    int idx     = 0;
    for (auto i : tensor->getDimensions())
      dims[idx++] = i;

    const NvDlaCubeInfo cubeinfo(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, dims[0], dims[1], dims[2], dims[3], 0, 0);
   
    const bool isInput = isa<InputOperator>(getProducer(*tensor));
    if (isInput && !m_pMeta->hasMemoryListEntry(*tensor)) {
      const MemoryListEntryId memoryId =
        m_pMeta->allocateMemoryFor(*tensor, ILoadable::MemoryDomain_SYSMEM,
                                   ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_INPUT, cubeinfo.size);

      ILoadable::TensorDescListEntry tle;
      tle.name   = "data";
      tle.id     = 0;
      tle.memId  = memoryId;
      tle.size   = m_pMeta->getMemoryListEntrySize(memoryId);
      tle.offset = 0;

      tle.dims.n       = cubeinfo.dim_n;
      tle.dims.c       = cubeinfo.dim_c;
      tle.dims.h       = cubeinfo.dim_h;
      tle.dims.w       = cubeinfo.dim_w;
      tle.dataFormat   = 3;
      tle.dataType     = m_nvdla_constants.DATA_TYPE;
      tle.dataCategory = nvdla::loadable::DataCategory_FEATURE;
      tle.pixelFormat  = m_nvdla_constants.INPUT_PIXEL_FORMAT;
      tle.pixelMapping = 0;

      tle.stride[0] = cubeinfo.stride_channel;
      tle.stride[1] = cubeinfo.stride_line;
      tle.stride[2] = cubeinfo.stride_surface;
      tle.stride[3] = 0;
      tle.stride[4] = 0;
      tle.stride[5] = 0;
      tle.stride[6] = 0;
      tle.stride[7] = 0;

      m_pMeta->m_TensorDescListEntries.emplace(m_pMeta->m_TensorDescListEntries.begin(), tle);
    } else if (isOutput(tensor)) {
      const NvDlaBackendMeta::MemoryFlags flags    = ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_OUTPUT;
      MemoryListEntryId                   memoryId = NvDlaBackendMeta::getInvalidMemoryListEntryId();
      if (m_pMeta->shouldOwnMemory(*tensor)) {
        memoryId = m_pMeta->allocateMemoryFor(*tensor, ILoadable::MemoryDomain_SYSMEM, flags, cubeinfo.size,
                                              true /* is output */);
      } else {
        NvDlaBackendMeta::MemoryListEntry& memory = m_pMeta->getMemoryListEntry(*tensor);

        memory.tensor_desc_id = 1; // mark this MemoryListEntry is for output
        memory.flags          = flags;
        memoryId              = memory.id;
      }
      assert(memoryId != NvDlaBackendMeta::getInvalidMemoryListEntryId());

      ILoadable::TensorDescListEntry tle;
      tle.name   = "probe";
      tle.id     = 1;
      tle.memId  = memoryId;
      tle.size   = m_pMeta->getMemoryListEntrySize(memoryId);
      tle.offset = 0;

      tle.dims.n       = cubeinfo.dim_n;
      tle.dims.c       = cubeinfo.dim_c;
      tle.dims.h       = cubeinfo.dim_h;
      tle.dims.w       = cubeinfo.dim_w;
      tle.dataFormat   = 3;
      tle.dataType     = m_nvdla_constants.DATA_TYPE;
      tle.dataCategory = nvdla::loadable::DataCategory_FEATURE;
      tle.pixelFormat  = m_nvdla_constants.OUTPUT_PIXEL_FORMAT;
      tle.pixelMapping = 0;

      tle.stride[0] = cubeinfo.stride_channel;
      tle.stride[1] = cubeinfo.stride_line;
      tle.stride[2] = cubeinfo.stride_surface;
      tle.stride[3] = 0;
      tle.stride[4] = 0;
      tle.stride[5] = 0;
      tle.stride[6] = 0;
      tle.stride[7] = 0;

      m_pMeta->m_TensorDescListEntries.push_back(tle);
    } else {
      m_pMeta->tryAllocateMemoryFor(*tensor, ILoadable::MemoryDomain_SYSMEM, ILoadable::MemoryFlags_ALLOC,
                                    cubeinfo.size);
    }
  }
}

void NvDlaLib::nvdla_filegen()
{
  m_pMeta->m_Loadable.priv()->setMemoryListEntries(m_pMeta->m_MemoryListEntries);
  m_pMeta->m_Loadable.priv()->setTensorDescListEntries(m_pMeta->m_TensorDescListEntries);
  m_pMeta->m_Loadable.priv()->setAddressListEntries(m_pMeta->m_AddressListEntries);
  m_pMeta->m_Loadable.priv()->setEventListEntries(m_pMeta->m_EventListEntries);
  m_pMeta->m_Loadable.priv()->setTaskListEntries(m_pMeta->m_TaskListEntries);
  m_pMeta->m_Loadable.priv()->setSubmitListEntries(m_pMeta->m_SubmitListEntries);
  m_pMeta->m_Loadable.priv()->serialize();
}

CbufAllocType NvDlaLib::getCbufAllocType(const NvDlaCubeInfo& xinfo, const NvDlaCubeInfo& winfo, Tensor::Dimension yDilation,
                                 unsigned& minNumNeededDataBanks)
{
    const unsigned input_height  = xinfo.dim_h;
  const unsigned kernel_height = winfo.dim_h;
  const unsigned kernel_width  = winfo.dim_w;
  minNumNeededDataBanks        = DIV_ROUNDUP(
    (xinfo.eps * min(kernel_height + (kernel_height - 1) * (yDilation - 1), input_height)), m_nvdla_constants.CBUF_BANK_DEPTH);

  if ((xinfo.banks + winfo.getBanksForFullWeights()) <= m_nvdla_constants.CBUF_BANK_NUM) {
    // Full data and full weights
    return CbufAllocType::kFullDataFullWeight;
  } else if ((xinfo.banks + winfo.getBanksForPartialWeights()) <= m_nvdla_constants.CBUF_BANK_NUM) {
    // Full data and partial weights
    return CbufAllocType::kFullDataPartialWeight;
  } else if ((minNumNeededDataBanks + winfo.getBanksForFullWeights()) <= m_nvdla_constants.CBUF_BANK_NUM) {
    // Split data and full weights
    return CbufAllocType::kSplitDataFullWeight;
  } else if ((minNumNeededDataBanks + winfo.getBanksForPartialWeights()) <= m_nvdla_constants.CBUF_BANK_NUM) {
    // Split data and partial weights
    return CbufAllocType::kSplitDataPartialWeight;
  } else if ((xinfo.banks + winfo.getBanksForMinimumWeights()) <= m_nvdla_constants.CBUF_BANK_NUM) {
    // Full data and minimum weights
    return CbufAllocType::kFullDataMinimumWeight;
  } else if ((minNumNeededDataBanks + winfo.getBanksForMinimumWeights()) <= m_nvdla_constants.CBUF_BANK_NUM) {
    // Split data and minimum weights
    return CbufAllocType::kSplitDataMinimumWeight;
  }

  return CbufAllocType::kUnfeasible;
}                                

void
NvDlaLib::create_float_weight_tensor_from_numpy(const std::string& pName,
                          const Tensor::Dimensions& pDims, void* data)
{
  assert(data != nullptr);

  Initializer* init = m_pCG->addOperator<Initializer>(pName);
  Tensor* value = this->create_compute_tensor<FloatTensor>(pName, pDims);

  float * weight = (float *)data;

  size_t numElems = 1;
  for(auto i : pDims)
  {
    numElems *= i;
  }

  ((FloatTensor *)value)->getValues().resize(numElems); 
  for (size_t i = 0; i < numElems; ++i) {
    ((FloatTensor *)value)->getValues()[i] = weight[i];
    // printf("%f\n", ((FloatTensor *)value)->getValues()[i]);
  }
  /*
  #ifdef NVDLA_DEBUG
  for(auto i = 0; i < numElems; ++i)
  {
    std::cout<< weight[i]<< std::endl;
  }
  #endif
  */
  init->setTensor(*value);
}

void
NvDlaLib::create_float_weight_tensor_from_file(const std::string& pName,
                          const Tensor::Dimensions& pDims, const std::string& weight_path)
{
  assert(!weight_path.empty());

  Initializer* init = m_pCG->addOperator<Initializer>(pName);
  Tensor* value = this->create_compute_tensor<FloatTensor>(pName, pDims);

  xTensorProto tensor;

  std::ifstream input_fin(weight_path);

  tensor.ParseFromIstream(&input_fin);
  const std::string &raw_data_str = tensor.raw_data();

  const size_t numElems = raw_data_str.size() / (sizeof(float)); 
  float* d = (float*)raw_data_str.c_str(); 
  ((FloatTensor *)value)->getValues().resize(numElems); 
  printf("INPUT PATH:%s\n", weight_path.c_str());
  for (size_t i = 0; i < numElems; ++i) {

    ((FloatTensor *)value)->getValues()[i] = d[i];
    }

  
  init->setTensor(*value);
}

void NvDlaLib::optimize()
{
  // Should follow optimize order
  // 1.divide_globalap_into_aps
  // 2.propagate_const_with_diff_shape
  // 3.expand_batch_normalization
  // 4.replace_gemm_by_conv
  // 5.eliminate_identity

  // optinal
  // 6.SplitConvPass

  // NvdDla specific
  // 7.EliminateIdentity
  // 8.NvDlaCalibrateAveragePoolResultPass
  // 9.NvDlaIdentifyShufflePass
  // 10.SplitGroupConvPass
  // 11.NvDlaCollectReshapeInfoPass
  DivideGlobalAPIntoAPs::run(*m_pCG);
  PropagateConstWithDiffShape::run(*m_pCG);
  ExpandBatchNormalization::run(*m_pCG);
  ReplaceGemmByConv::run(*m_pCG);

  NvDlaCollectReshapeInfoPass::run(*m_pModule, *m_pMeta);
}

void NvDlaLib::compile()
{
  this->init_nvdla_memory();

  ComputeGraph::iterator nodeIt, nEnd = m_pCG->end();
    for (nodeIt = m_pCG->begin(); nodeIt != nEnd; ++nodeIt) {
        const onnc::ComputeOperator *node = nodeIt;
        //std::cout<< node->name()<< std::endl;
        if(node->name() == "Relu"){
            this->relu(* (Relu *) node);
        } else if(node->name() == "Conv") {
            this->conv(* (Conv *) node);
        } else if(node->name() == "Add") {
            this->add(* (Add *) node);
        } else if(node->name() == "Mul") {
            this->mul(* (Mul *) node);
        } else if(node->name() == "Reshape") {
            this->reshape(* (Reshape *) node);
        } else if(node->name() == "MaxPool") {
            this->max_pool(* (MaxPool *) node);
        } else if(node->name() == "AveragePool") {
            this->average_pool(* (AveragePool *) node);
        } else if(node->name() == "Gemm") {
            this->gemm(* (Gemm *) node);
        } else if(node->name() == "BatchNormalization") {
            this->batchnorm(* (BatchNormalization *) node);
        } else if(node->name() == "GlobalAveragePool") {
            this->global_average_pool(* (GlobalAveragePool *) node);
        } 
        else {
          //assert((node->name() == "InputOperator" || node->name() == "Initializer" || node->name() == "OutputOperator"));
          if (!(node->name() == "InputOperator" || node->name() == "Initializer" || node->name() == "OutputOperator")) {
            std::cout<< "Unsupport Op:"<< node->name()<< std::endl;
            //std::abort();
            //exit(0)
          }
        }
        //node->dump();
  }
  this->task_submit();
  this->nvdla_filegen();


}