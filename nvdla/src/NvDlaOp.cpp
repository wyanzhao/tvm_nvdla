#include <NvDlaOp.h>
#include <NvDlaUtil.h>

#include <onnc/Support/Algorithm.h>
#include <onnc/Support/IOStream.h>
#include <onnc/Support/Match.h>
#include <onnc/Support/Preprocessor.h>
#include <onnc/Support/Range.h>
#include <onnc/Support/String.h>
#include <onnc/Support/View.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace onnc;

enum class VerboseLevels : unsigned
{
  TRACE = 1,
  DEBUG = 2,
  INFO  = 3,
  WARN  = 4,
};

inline bool operator<=(VerboseLevels lhs, unsigned rhs) noexcept
{
  using underlying_type = std::underlying_type<VerboseLevels>::type;
  static_assert(std::is_unsigned<underlying_type>::value, "underlying type of verbose should be unsigned");

  return static_cast<underlying_type>(lhs) <= rhs;
}

std::ostream& operator<<(std::ostream& stream, const ComputeOperator& op)
{
  op.print(stream);
  return stream;
}

//namespace internal {

enum class BroadcastCategory : std::int8_t
{
  UNSUPPORT = -1,
  LAYER     = 0,
  CHANNEL   = 1,
  ELEMENT   = 2,
};

BroadcastCategory getBroadcastCategory(const Tensor& fromTensor, const Tensor& toTensor)
{
  const Tensor::Dimensions& fromDims = fromTensor.getDimensions();
  const Tensor::Dimensions& toDims   = toTensor.getDimensions();

  if (!isConstant(fromTensor) && !isConstant(toTensor)) {
    if (size(fromDims) == 4 && size(toDims) == 4) {
      return (equal(fromDims, begin(toDims)) ? BroadcastCategory::ELEMENT : BroadcastCategory::UNSUPPORT);
    } else {
      return BroadcastCategory::UNSUPPORT;
    }
    // only constant tensor can broadcast to non-constant tensor
  } else if (!isConstant(fromTensor) || isConstant(toTensor)) {
    return BroadcastCategory::UNSUPPORT;
  }

  assert(size(toDims) == 4);

  using Any     = MatchAny<Tensor::Dimension>;
  using AnyList = std::vector<Any>;

  using std::begin;
  switch (size(fromDims)) {
  case 4: {
    assert(toDims[0] == 1);

    const Tensor::Dimension c = toDims[1];
    const Tensor::Dimension h = toDims[2];
    const Tensor::Dimension w = toDims[3];

    if (equal({1, 1, 1, 1}, begin(fromDims))) {
      return BroadcastCategory::LAYER;
    } else if (equal(asRange<AnyList>(1, c, 1, 1), begin(fromDims))) {
      return BroadcastCategory::CHANNEL;
    } else if (equal(asRange<AnyList>(1, c, h, w), begin(fromDims))) {
      return BroadcastCategory::ELEMENT;
    } else {
      return BroadcastCategory::UNSUPPORT;
    }
  } break;
  case 3: {
    const Tensor::Dimension c = toDims[1];

    if (equal({1, 1, 1}, begin(fromDims))) {
      return BroadcastCategory::LAYER;
    } else if (equal(asRange<AnyList>(c, 1, 1), begin(fromDims))) {
      return BroadcastCategory::CHANNEL;
    } else {
      return BroadcastCategory::UNSUPPORT;
    }
  } break;
  case 2: {
    if (equal({1, 1}, begin(fromDims))) {
      return BroadcastCategory::LAYER;
    } else {
      return BroadcastCategory::UNSUPPORT;
    }
  } break;
  case 1:
    return (fromDims[0] == 1 ? BroadcastCategory::LAYER : BroadcastCategory::UNSUPPORT);
  default:
    return BroadcastCategory::UNSUPPORT;
  }
}

std::uint8_t getSdpOpMode(BroadcastCategory category)
{
  assert(category != BroadcastCategory::UNSUPPORT);

  return static_cast<std::uint8_t>(category);
}

nvdla_cube_type getSdpXSingleCubeType(const Tensor& tensor, std::uint8_t precision)
{
  if (!isConstant(tensor)) {
    return NVDLA_CUBE_FEATURE;
  }

  switch (precision) {
  case PRECISION_INT8:
    return NVDLA_CUBE_SDP_X_ALU_OR_MUL_ONE_BYTE;
  case PRECISION_INT16:
    [[fallthrough]];
  case PRECISION_FP16:
    return NVDLA_CUBE_SDP_X_ALU_OR_MUL_TWO_BYTE;
  default:
    break;
  }

  assert(false && "meet unsupported tensor");
  return NVLDA_CUBE_UNKNOWN;
}

enum class NvDlaOpType : std::uint8_t
{
  bdma  = DLA_OP_BDMA,
  conv  = DLA_OP_CONV,
  sdp   = DLA_OP_SDP,
  pdp   = DLA_OP_PDP,
  cdp   = DLA_OP_CDP,
  rubik = DLA_OP_RUBIK,
};

bool operator==(std::underlying_type<NvDlaOpType>::type lhs, NvDlaOpType rhs)
{
  return lhs == static_cast<std::underlying_type<NvDlaOpType>::type>(rhs);
}

std::unique_ptr<NvDlaDlaOperation> makeNvDlaOp(NvDlaOpType type)
{
  auto operation            = std::make_unique<NvDlaDlaOperation>();
  operation->op_dep.op_type = static_cast<std::underlying_type<NvDlaOpType>::type>(type);
  return operation;
}

template <NvDlaOpType type>
struct nvdla_op_desc;

template <>
struct nvdla_op_desc<NvDlaOpType::sdp>
{
  using type = dla_sdp_op_desc;
};

template <NvDlaOpType type>
typename nvdla_op_desc<type>::type& getDesc(NvDlaDlaOperation& operation);

template <>
nvdla_op_desc<NvDlaOpType::sdp>::type& getDesc<NvDlaOpType::sdp>(NvDlaDlaOperation& operation)
{
  assert(operation.op_dep.op_type == NvDlaOpType::sdp);
  return operation.op_desc.sdp_op;
}

template <NvDlaOpType type>
struct nvdla_op_surface;

template <>
struct nvdla_op_surface<NvDlaOpType::sdp>
{
  using type = dla_sdp_surface_desc;
};

template <NvDlaOpType type>
typename nvdla_op_surface<type>::type& getSurface(NvDlaDlaOperation& operation);

template <>
nvdla_op_surface<NvDlaOpType::sdp>::type& getSurface<NvDlaOpType::sdp>(NvDlaDlaOperation& operation)
{
  assert(operation.op_dep.op_type == NvDlaOpType::sdp);
  return operation.op_surf.sdp_surface;
}

NvDlaCubeInfo makeCubeInfo(const NvDlaConstants& constants, nvdla_cube_type type, Tensor::Dimension n,
                           Tensor::Dimension c, Tensor::Dimension h, Tensor::Dimension w)
{
  return NvDlaCubeInfo(constants, type, n, c, h, w);
}

NvDlaCubeInfo makeCubeInfo(const NvDlaConstants& constants, nvdla_cube_type type, const Tensor& tensor)
{
  const NvDlaDims dimensions(tensor);
  return makeCubeInfo(constants, type, dimensions.n, dimensions.c, dimensions.h, dimensions.w);
}

enum class NvDlaMemType : std::uint16_t
{
  mc = DLA_MEM_MC,
  cv = DLA_MEM_CV,
  hw = DLA_MEM_HW,
};

class NvDlaDataCubeModifier
{
public:
  NvDlaDataCubeModifier(dla_data_cube& cube, NvDlaMemType type)
    : cube_{cube}
  {
    cube_.type = static_cast<std::underlying_type<NvDlaMemType>::type>(type);
  }

  NvDlaDataCubeModifier& setAddress(std::int16_t address)
  {
    cube_.address = address;
    return *this;
  }

  NvDlaDataCubeModifier& setSize(std::uint32_t size)
  {
    cube_.size = size;
    return *this;
  }

  NvDlaDataCubeModifier& setInfo(const NvDlaCubeInfo& info)
  {
    cube_.width        = info.dim_w;
    cube_.height       = info.dim_h;
    cube_.channel      = info.dim_c;
    cube_.line_stride  = info.stride_line;
    cube_.surf_stride  = info.stride_surface;
    cube_.plane_stride = info.stride_plane;
    return *this;
  }

private:
  dla_data_cube& cube_;
};

//} // namespace internal


using namespace nvdla;


MemoryListEntryId NvDlaOp::packWeight(const Tensor& weight, NvDlaDims destDims,
                                              Tensor::Dimension numFrontPaddingChannels,
                                              Tensor::Dimension outputChannelOffset)
{
  return packWeight(weight, NvDlaDims(weight), destDims, numFrontPaddingChannels, outputChannelOffset);
}

MemoryListEntryId NvDlaOp::packWeight(const Tensor& weight, NvDlaDims srcDims, NvDlaDims destDims,
                                              Tensor::Dimension numFrontPaddingChannels,
                                              Tensor::Dimension outputChannelOffset)
{
  if (const auto* const floatTensor = dynamic_cast<const FloatTensor*>(&weight)) {
    return packWeight(floatTensor->getValues(), floatTensor, srcDims, destDims, numFrontPaddingChannels,
                      outputChannelOffset);
  }

  return MemoryListEntryId(-1);
}

MemoryListEntryId NvDlaOp::packWeight(span<const float> weight, const Tensor* weightTensor, NvDlaDims srcDims,
                                              NvDlaDims destDims, Tensor::Dimension numFrontPaddingChannels,
                                              Tensor::Dimension outputChannelOffset)
{
  assert(size(weight) == srcDims.size());

  std::string blob_name = "tb-" + std::to_string(m_pMeta->m_NumBlobs++);

  const Tensor::Dimension numDestChannels = numFrontPaddingChannels + destDims.c;
  const NvDlaDims         destDimsWithFrontPadding(destDims.n, numDestChannels, destDims.h, destDims.w);

  ILoadable::Blob b;
  b.name              = blob_name;
  b.size              = UNIT_ALIGNMENT(destDimsWithFrontPadding.size() * m_nvdla_constants.ELEMENT_SIZE, m_nvdla_constants.WEIGHT_ATOM_CUBE_SIZE);
  b.version.major     = 0;
  b.version.minor     = 0;
  b.version.sub_minor = 0;
  b.interface         = ILoadable::Interface_NONE;
  b.subInterface      = 0;

  NvU8* blob_data = new NvU8[b.size];
  memset(blob_data, 0, b.size);

  const auto* srcData = weight.data();

  using weight_type           = nv_weight_t<nvdla::ConfigSet::nv_full>;
  weight_type* const destData = reinterpret_cast<weight_type*>(blob_data);

  packWeightImpl(destData, destDimsWithFrontPadding, weightTensor, srcData, srcDims, numFrontPaddingChannels,
                 outputChannelOffset);

  m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);

  const MemoryListEntryId memoryId = m_pMeta->allocateMemory(
    ILoadable::MemoryDomain_SYSMEM, ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_SET, b.size);

  ILoadable::MemoryListEntry& memory = m_pMeta->getMemoryListEntry(memoryId);
  memory.contents.push_back(blob_name);
  memory.offsets.push_back(0);

  return memoryId;
}

MemoryListEntryId NvDlaOp::packBias(const Tensor& bias, Tensor::Dimension numDestChannels,
                                            Tensor::Dimension srcChannelOffset)
{
   std::string   blob_name = "tb-" + std::to_string(m_pMeta->m_NumBlobs++);
  NvDlaCubeInfo finfo(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, 1, numDestChannels, 1, 1);

  ILoadable::Blob b;
  b.name              = blob_name;
  b.size              = m_nvdla_constants.ELEMENT_SIZE * UNIT_ALIGNMENT(numDestChannels, m_nvdla_constants.MAC_ATOMIC_K);
  b.version.major     = 0;
  b.version.minor     = 0;
  b.version.sub_minor = 0;
  b.interface         = ILoadable::Interface_NONE;
  b.subInterface      = 0;

  NvU8* blob_data = new NvU8[b.size];
  std::memset(blob_data, 0, b.size);

  if (const FloatTensor* floatTensor = dynamic_cast<const FloatTensor*>(&bias)) {
    const auto* srcData = reinterpret_cast<const float*>(floatTensor->getValues().data());

    using weight_type           = nv_weight_t<nvdla::ConfigSet::nv_full>;
    weight_type* const destData = reinterpret_cast<weight_type*>(blob_data);

    packBiasImpl(destData, numDestChannels, &bias, srcData, srcChannelOffset);
  }

  m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);

  const MemoryListEntryId memoryId = m_pMeta->allocateMemory(
    ILoadable::MemoryDomain_SYSMEM, ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_SET, b.size);

  ILoadable::MemoryListEntry& memory = m_pMeta->getMemoryListEntry(memoryId);
  memory.contents.push_back(blob_name);
  memory.offsets.push_back(0);

  return memoryId;
}

template <typename Type>
void NvDlaOp::packWeightImpl(Type* destData, NvDlaDims destDimsWithFrontPadding, const Tensor* tensor,
                                     const float* srcData, NvDlaDims srcDims, Tensor::Dimension numFrontPaddingChannels,
                                     Tensor::Dimension outputChannelOffset)
{
  const NvDlaDims::value_type N = destDimsWithFrontPadding.n;
  const NvDlaDims::value_type C = destDimsWithFrontPadding.c;
  const NvDlaDims::value_type H = destDimsWithFrontPadding.h;
  const NvDlaDims::value_type W = destDimsWithFrontPadding.w;

  const int channel_per_cube = m_nvdla_constants.WEIGHT_ATOM_CUBE_SIZE / m_nvdla_constants.ELEMENT_SIZE;
  const int w_stride_kgrp    = m_nvdla_constants.MAC_ATOMIC_K * C * H * W;

  using weight_t = typename std::decay<decltype(*destData)>::type;

  for (Tensor::Dimension n = 0; n < (N / m_nvdla_constants.MAC_ATOMIC_K + 1); n++) {
    int n_size        = (N - n * m_nvdla_constants.MAC_ATOMIC_K >= m_nvdla_constants.MAC_ATOMIC_K) ? m_nvdla_constants.MAC_ATOMIC_K : N - n * m_nvdla_constants.MAC_ATOMIC_K;
    int w_stride_surf = W * H * n_size * channel_per_cube;
    for (Tensor::Dimension h = 0; h < H; h++) {
      for (Tensor::Dimension w = 0; w < W; w++) {
        for (int n_ofs = 0; n_ofs < n_size; n_ofs++) {
          for (Tensor::Dimension c = 0; c < C; c++) {
            int surf_ofs  = c / channel_per_cube;
            int ch_ofs    = c % channel_per_cube;
            int cube_size = (C - surf_ofs * channel_per_cube) >= channel_per_cube ? channel_per_cube
                                                                                  : (C - surf_ofs * channel_per_cube);
            int w_stride_line = W * n_size * cube_size;

            int dest_ofs = (n * w_stride_kgrp) + (surf_ofs * w_stride_surf) + (h * w_stride_line) +
                           w * n_size * cube_size + (n_ofs * cube_size) + ch_ofs;

            const Tensor::Dimension srcChannel = c - numFrontPaddingChannels;
            int src_ofs = ((n * m_nvdla_constants.MAC_ATOMIC_K + n_ofs + outputChannelOffset) * srcDims.c * srcDims.h * srcDims.w) +
                          (srcChannel * (srcDims.h * srcDims.w)) + (h * srcDims.w) + w;

            // fill zero at front if necessary
            if (c < numFrontPaddingChannels) {
              *(destData + dest_ofs) = 0;
              continue;
            }

            assert(srcChannel < srcDims.c);
            assert(src_ofs < srcDims.size());

            *(destData + dest_ofs) = f2float16_ieee(*(srcData + src_ofs));
          }
        }
      }
    }
  }
}

template <typename Type>
void NvDlaOp::packBiasImpl(Type* destData, Tensor::Dimension numDestChannels, const Tensor* tensor,
                                   const float* srcData, Tensor::Dimension srcChannelOffset)
{
  assert(srcData != nullptr);
  assert(destData != nullptr);

  using weight_t = typename std::decay<decltype(*destData)>::type;

  for (Tensor::Dimension destChannel = 0; destChannel < numDestChannels; ++destChannel) {
    *(destData + destChannel) =
      f2float16_ieee(*(srcData + srcChannelOffset + destChannel)); // FIXME: bad hack method.
  }
}

MemoryListEntryId NvDlaOp::packFeature(const Tensor& tensor, const NvDlaCubeInfo& cube)
{
  assert(false && "not implemented");
  return 0;
}

AddressListEntryId NvDlaOp::issueEmuAddr(MemoryListEntryId mid)
{
  AddressListEntryId aid = m_pMeta->m_AddressListEntries.size();

  ILoadable::AddressListEntry ale;
  ILoadable::MemoryListEntry  mle = m_pMeta->getMemoryListEntry(mid);

  ale.size   = 0;
  ale.offset = 0;
  ale.mem_id = mid;
  ale.id     = aid;

  m_pMeta->m_AddressListEntries.push_back(ale);
  return aid;
}

void NvDlaOp::issueEmuOp(NvDlaEmuOperation* op)
{
  m_pMeta->m_EMUOperationList.push_back(op);

  m_pMeta->appendOperationMeta(m_pMeta->m_EMUOperationList.size() - 1,
                              NvDlaBackendMeta::OperationMeta::Category::emu);
}

AddressListEntryId NvDlaOp::issueDlaAddr(const Tensor& tensor, const NvDlaCubeInfo& cube,
                                                 Tensor::Dimension channelOffset, NvDlaBackendMeta::Offset hOffset)
{
  using offset_type = NvDlaBackendMeta::Offset;

  const offset_type h_offset     = hOffset * cube.stride_line;
  const offset_type memoryOffset = (channelOffset * (cube.dim_h * cube.dim_w * m_nvdla_constants.ELEMENT_SIZE)) + h_offset;

  return m_pMeta->acquireMemory(m_pMeta->getMemoryListEntryId(tensor), memoryOffset);
}

AddressListEntryId NvDlaOp::issueDlaAddr(const Tensor& tensor, const NvDlaCubeInfo& cube)
{
  const MemoryListEntryId memoryId = m_pMeta->getMemoryListEntryId(tensor);

  return m_pMeta->acquireMemory(memoryId, 0);
}

AddressListEntryId NvDlaOp::issueSDPOperand(const Tensor& tensor, const NvDlaCubeInfo& cube,
                                                    MemoryListEntryId& memoryId)
{
  if (isConstant(tensor)) {
    memoryId = packSDPOperand(&tensor, nullptr, cube);
    return issueDlaAddr(memoryId, cube);
  }

  memoryId = m_pMeta->getMemoryListEntryId(tensor);
  return issueDlaAddr(tensor, cube);
}

AddressListEntryId NvDlaOp::issueDlaAddr(MemoryListEntryId memoryId, const NvDlaCubeInfo& cube)
{
  assert(m_pMeta->hasMemoryListEntry(memoryId));

  return m_pMeta->acquireMemory(memoryId, 0);
}

void NvDlaOp::issueDlaOp(NvDlaDlaOperation* op, NvDlaDlaOperation* op_fuse, NvDlaDlaOperation* op_prev)
{
  struct dla_common_op_desc* op_desc = &(op->op_dep);
  int                        op_type = op_desc->op_type;
  op_desc->index            = m_pMeta->m_DLAOperationList.size();
  op_desc->roi_index        = 0;
  op_desc->dependency_count = 0;

  if (op_prev != NULL && !m_pMeta->isLastDlaOperationInTaskEntry(*op_prev)) {
    struct dla_common_op_desc* prev_op_desc = &(op_prev->op_dep);
    prev_op_desc->consumers[op_type].index  = op_desc->index;
    prev_op_desc->consumers[op_type].event  = 1;
    op_desc->dependency_count++;
  }

  if (m_pMeta->m_pDepOp[op_type] != NULL && !m_pMeta->isLastDlaOperationInTaskEntry(*m_pMeta->m_pDepOp[op_type])) {
    struct dla_common_op_desc* dep_op_desc = &(m_pMeta->m_pDepOp[op_type]->op_dep);
    if (m_pMeta->m_pDepOp[op_type] != op_prev) {
      dep_op_desc->consumers[op_type].index = op_desc->index;
      if ((op_type == DLA_OP_CONV) && (op_prev == NULL)) { // splitted convolution layers other than the first one
        dep_op_desc->consumers[op_type].event = 3;
      } else { // first splitted layer or other normal layers
        dep_op_desc->consumers[op_type].event = 2;
      }
      op_desc->dependency_count++;
    }
  }

  m_pMeta->m_DlaNetworkDesc.op_head[op_type] = (m_pMeta->m_DlaNetworkDesc.op_head[op_type] < 0)
                                                ? m_pMeta->m_DLAOperationList.size()
                                                : m_pMeta->m_DlaNetworkDesc.op_head[op_type];

  m_pMeta->m_DLAOperationList.push_back(op);
  m_pMeta->m_pDepOp[op_type] = op;

  m_pMeta->appendOperationMeta(m_pMeta->m_DLAOperationList.size() - 1,
                              NvDlaBackendMeta::OperationMeta::Category::dla);

  if (op_fuse != NULL) {
    struct dla_common_op_desc* fuse_op_desc = &(op_fuse->op_dep);
    int                        op_fuse_type = fuse_op_desc->op_type;
    fuse_op_desc->index                     = m_pMeta->m_DLAOperationList.size();
    fuse_op_desc->roi_index                 = 0;
    fuse_op_desc->dependency_count          = 1;
    fuse_op_desc->fused_parent.index        = op_desc->index;
    fuse_op_desc->fused_parent.event        = 3;

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

    m_pMeta->appendOperationMeta(m_pMeta->m_DLAOperationList.size() - 1,
                                NvDlaBackendMeta::OperationMeta::Category::dla);
  } else {
    m_pMeta->m_pPrevOp = op;
  }
}

void NvDlaOp::issueDlaOp(std::unique_ptr<NvDlaDlaOperation> operation)
{
  issueDlaOp(operation.release(), nullptr, m_pMeta->m_pPrevOp);
}

void NvDlaOp::emitSdp(std::uint8_t opType, const Tensor& first, const Tensor& second, const Tensor& output)
{
  assert(!(isConstant(first) && isConstant(second)) && "cannot support 2 constant tensors");
  assert(opType == SDP_OP_ADD || opType == SDP_OP_MUL);

  // make sure the 'first' tensor is always non-constant
  if (isConstant(first)) {
    emitSdp(opType, second, first, output);
    return;
  }

  assert(!isConstant(first));

  const BroadcastCategory category =
    (isConstant(second) ? getBroadcastCategory(second, first) : getBroadcastCategory(first, second));
  auto operation = makeNvDlaOp(NvDlaOpType::sdp);

  auto& desc             = getDesc<NvDlaOpType::sdp>(*operation);
  desc.src_precision     = m_nvdla_constants.DLA_PRECISION;
  desc.dst_precision     = m_nvdla_constants.DLA_PRECISION;
  desc.lut_index         = -1;
  desc.out_cvt.scale     = 1;
  desc.out_cvt.truncate  = 0;
  desc.out_cvt.enable    = 1;
  desc.out_cvt.offset    = 0;
  desc.conv_mode         = CONV_MODE_DIRECT;
  desc.batch_num         = 1;
  desc.batch_stride      = 0;
  desc.x1_op.enable      = 1;
  desc.x1_op.alu_type    = SDP_ALU_OP_SUM;
  desc.x1_op.type        = opType;
  desc.x1_op.mode        = getSdpOpMode(category);
  desc.x1_op.act         = ACTIVATION_NONE;
  desc.x1_op.shift_value = 0;
  desc.x1_op.truncate    = 0;
  desc.x1_op.precision   = m_nvdla_constants.DLA_PRECISION;
  if (category == BroadcastCategory::LAYER) {
    const auto operand = to_<std::vector<float>>(second);
    assert(size(operand) == 1);

    switch (opType) {
    case SDP_OP_ADD:
      desc.x1_op.mul_operand = 0;
      desc.x1_op.alu_operand = f2float16_ieee(*begin(operand));
      break;
    case SDP_OP_MUL:
      desc.x1_op.alu_operand = 0;
      desc.x1_op.mul_operand = f2float16_ieee(*begin(operand));
      break;
    default:
      assert(false && "should not reach here");
    }
  }

  auto& surface = getSurface<NvDlaOpType::sdp>(*operation);

  const NvDlaCubeInfo firstCubeInfo = makeCubeInfo(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, first);
  NvDlaDataCubeModifier(surface.src_data, NvDlaMemType::mc)
    .setSize(m_pMeta->getMemoryListEntrySize(first))
    .setAddress(issueDlaAddr(first, firstCubeInfo))
    .setInfo(firstCubeInfo);

  if (category == BroadcastCategory::LAYER) {
    NvDlaDataCubeModifier(surface.x1_data, NvDlaMemType::hw).setAddress(-1);
  } else {
    MemoryListEntryId   memoryId;
    const NvDlaCubeInfo secondCubeInfo = makeCubeInfo(this->m_nvdla_constants, getSdpXSingleCubeType(second, m_nvdla_constants.DLA_PRECISION), second);
    NvDlaDataCubeModifier(surface.x1_data, NvDlaMemType::mc)
      .setAddress(issueSDPOperand(second, secondCubeInfo, memoryId))
      .setSize(m_pMeta->getMemoryListEntrySize(memoryId))
      .setInfo(secondCubeInfo);
  }

  const NvDlaCubeInfo outputCubeInfo = makeCubeInfo(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, output);
  NvDlaDataCubeModifier(surface.dst_data, NvDlaMemType::mc)
    .setSize(m_pMeta->getMemoryListEntrySize(output))
    .setAddress(issueDlaAddr(output, outputCubeInfo))
    .setInfo(outputCubeInfo);

  issueDlaOp(std::move(operation));
}

void NvDlaOp::packSDPOperandImpl(NvU8* blob, const Tensor* aluTensor, const float* aluData,
                                         const Tensor* mulTensor, const float* mulData, const NvDlaCubeInfo& cubeInfo)
{
  int64_t tmpdims[4];
  tmpdims[0] = cubeInfo.dim_n;
  tmpdims[1] = cubeInfo.dim_c;
  tmpdims[2] = cubeInfo.dim_h;
  tmpdims[3] = cubeInfo.dim_w;
  NvDlaDims srcDims(tmpdims);

  const float*  srcData = nullptr;
  const Tensor* srcTensor;
  if (cubeInfo.mode == NVDLA_CUBE_SDP_X_ALU_OR_MUL_ONE_BYTE || cubeInfo.mode == NVDLA_CUBE_SDP_Y_ALU_OR_MUL_ONE_BYTE ||
      cubeInfo.mode == NVDLA_CUBE_SDP_X_ALU_OR_MUL_TWO_BYTE || cubeInfo.mode == NVDLA_CUBE_SDP_Y_ALU_OR_MUL_TWO_BYTE) {
    if (aluData != nullptr) {
      srcData   = aluData;
      srcTensor = aluTensor;
    } else if (mulData != nullptr) {
      srcData   = mulData;
      srcTensor = mulTensor;
    }

    assert(srcData != nullptr);
  }

  assert(cubeInfo.dim_n == 1);
  for (int c = 0; c < cubeInfo.dim_c; c++) {     // kernel channel
    for (int h = 0; h < cubeInfo.dim_h; h++) {   // kernel height
      for (int w = 0; w < cubeInfo.dim_w; w++) { // kernel width

        int blob_ofs = m_nvdla_constants.getBlobOffsetForSDPOperand(c, h, w, cubeInfo);
        // printf("(c,h,w) = (%.2d,%.2d,%.2d)  blob_ofs=%.5d   (KC,KH,KW)=(%d,%d,%d)\n", c, h, w, blob_ofs,
        // cubeInfo.dim_n,
        //        cubeInfo.dim_c, cubeInfo.dim_h, cubeInfo.dim_w);

        int src_ofs = m_nvdla_constants.getONNXInitializerOffset(0, c, h, w, srcDims);

        switch (cubeInfo.mode) {
        case NVDLA_CUBE_SDP_X_ALU_OR_MUL_ONE_BYTE:
        case NVDLA_CUBE_SDP_Y_ALU_OR_MUL_ONE_BYTE:
          break;

        case NVDLA_CUBE_SDP_X_BOTH_ONE_BYTE:
          break;

        case NVDLA_CUBE_SDP_Y_BOTH_ONE_BYTE:
          break;

        case NVDLA_CUBE_SDP_X_ALU_OR_MUL_TWO_BYTE:
        case NVDLA_CUBE_SDP_Y_ALU_OR_MUL_TWO_BYTE:
          uint16_t data;
          data = (uint16_t)f2float16_ieee(*(srcData + src_ofs));
          // NVDLA uses little endian to interpret data in cube.
          *(blob + 2 * blob_ofs)     = (NvU8)(data & 0xFF);
          *(blob + 2 * blob_ofs + 1) = (NvU8)((data >> 8) & 0xFF);
          break;

        case NVDLA_CUBE_SDP_X_BOTH_TWO_BYTE:
        case NVDLA_CUBE_SDP_Y_BOTH_TWO_BYTE:
          uint16_t alu;
          uint16_t mul;
          alu = (uint16_t)f2float16_ieee(*(aluData + src_ofs));
          mul = (uint16_t)f2float16_ieee(*(mulData + src_ofs));

          if (cubeInfo.mode == NVDLA_CUBE_SDP_X_BOTH_TWO_BYTE) {
            *(blob + 4 * blob_ofs)     = (NvU8)(alu & 0xFF);
            *(blob + 4 * blob_ofs + 1) = (NvU8)((alu >> 8) & 0xFF);
            *(blob + 4 * blob_ofs + 2) = (NvU8)(mul & 0xFF);
            *(blob + 4 * blob_ofs + 3) = (NvU8)((mul >> 8) & 0xFF);
          } else {
            assert(cubeInfo.mode == NVDLA_CUBE_SDP_Y_BOTH_TWO_BYTE);

            *(blob + 4 * blob_ofs)     = (NvU8)(mul & 0xFF);
            *(blob + 4 * blob_ofs + 1) = (NvU8)((mul >> 8) & 0xFF);
            *(blob + 4 * blob_ofs + 2) = (NvU8)(alu & 0xFF);
            *(blob + 4 * blob_ofs + 3) = (NvU8)((alu >> 8) & 0xFF);
          }
          break;

        default:
          assert(0 && "Unsupported SDP cube mode.");
          break;
        } // switch (cubeInfo.mode)

      } // for (int w
    }   // for (int h
  }     // for (int c
}

// NvDlaCubeInfo BN_OPERAND
MemoryListEntryId NvDlaOp::packSDPOperand(const Tensor* aluTensor, const Tensor* mulTensor,
                                                  const NvDlaCubeInfo& cubeInfo)
{
  if (cubeInfo.mode == NVDLA_CUBE_SDP_X_BOTH_ONE_BYTE || cubeInfo.mode == NVDLA_CUBE_SDP_X_BOTH_TWO_BYTE) {
    assert(aluTensor != nullptr && mulTensor != nullptr);
  }

  if (cubeInfo.mode == NVDLA_CUBE_SDP_X_ALU_OR_MUL_ONE_BYTE || cubeInfo.mode == NVDLA_CUBE_SDP_X_ALU_OR_MUL_TWO_BYTE) {
    assert((aluTensor != nullptr && mulTensor == nullptr) || (aluTensor == nullptr && mulTensor != nullptr));
  }

  assert((aluTensor == nullptr || mulTensor == nullptr) || (NvDlaDims(*aluTensor) == NvDlaDims(*mulTensor)));

  std::string blob_name = "tb-" + std::to_string(m_pMeta->m_NumBlobs++);

  ILoadable::Blob b;
  b.name              = blob_name;
  b.size              = cubeInfo.size;
  b.version.major     = 0;
  b.version.minor     = 0;
  b.version.sub_minor = 0;
  b.interface         = ILoadable::Interface_NONE;
  b.subInterface      = 0;

  NvU8* blob_data = new NvU8[b.size];
  memset(blob_data, 0, b.size);

  int64_t tmpdims[4];
  tmpdims[0] = cubeInfo.dim_n;
  tmpdims[1] = cubeInfo.dim_c;
  tmpdims[2] = cubeInfo.dim_h;
  tmpdims[3] = cubeInfo.dim_w;
  NvDlaDims srcDims(tmpdims);

  // Get data of ALU and/or MUL.
  const float* aluData;
  const float* mulData;
  if (aluTensor != nullptr) {
    if (const FloatTensor* floatTensor = dynamic_cast<const FloatTensor*>(aluTensor)) {
      aluData = reinterpret_cast<const float*>(floatTensor->getValues().data());
    }
  }
  if (mulTensor != nullptr) {
    if (const FloatTensor* floatTensor = dynamic_cast<const FloatTensor*>(mulTensor)) {
      mulData = reinterpret_cast<const float*>(floatTensor->getValues().data());
    }
  }

  packSDPOperandImpl(blob_data, aluTensor, aluData, mulTensor, mulData, cubeInfo);

  m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);

  const MemoryListEntryId memoryId = m_pMeta->allocateMemory(
    ILoadable::MemoryDomain_SYSMEM, ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_SET, b.size);

  ILoadable::MemoryListEntry& memory = m_pMeta->getMemoryListEntry(memoryId);
  memory.contents.push_back(blob_name);
  memory.offsets.push_back(0);

  return memoryId;
}

template <typename Type>
void NvDlaOp::packImageWeightImpl(Type* blob, NvDlaDims blobDims, const Tensor* tensor, const float* srcData,
                                          NvDlaDims srcDims, Tensor::Dimension outputChannelOffset)
{
  assert((blobDims.c == 4) && "Kernel channel must be 4 in image mode.");

  using weight_t = typename std::decay<decltype(*blob)>::type;

  for (int k = 0; k < blobDims.n; k++) {       // kernel number
    for (int c = 0; c < blobDims.c; c++) {     // kernel channel
      for (int h = 0; h < blobDims.h; h++) {   // kernel height
        for (int w = 0; w < blobDims.w; w++) { // kernel width

          if (c >= srcDims.c)
            continue;

          int src_ofs  = m_nvdla_constants.getONNXInitializerOffset(outputChannelOffset + k, c, h, w, srcDims);
          int blob_ofs = m_nvdla_constants.getBlobOffsetForImageWeight(k, c, h, w, blobDims);

          *(blob + blob_ofs) = (weight_t)f2float16_ieee(*(srcData + src_ofs));
        }
      }
    }
  }
}

MemoryListEntryId NvDlaOp::packImageWeight(const Tensor& weight, NvDlaDims destDims,
                                                   Tensor::Dimension outputChannelOffset)
{
  const std::string blob_name = "tb-" + std::to_string(m_pMeta->m_NumBlobs++);

  ILoadable::Blob b;
  b.name              = blob_name;
  b.size              = UNIT_ALIGNMENT(destDims.size() * m_nvdla_constants.ELEMENT_SIZE, m_nvdla_constants.WEIGHT_ATOM_CUBE_SIZE);
  b.version.major     = 0;
  b.version.minor     = 0;
  b.version.sub_minor = 0;
  b.interface         = ILoadable::Interface_NONE;
  b.subInterface      = 0;

  NvU8* blob_data = new NvU8[b.size];
  memset(blob_data, 0, b.size);

  // Pack weights here.
  if (const FloatTensor* floatTensor = dynamic_cast<const FloatTensor*>(&weight)) {
    const auto* srcData = reinterpret_cast<const float*>(floatTensor->getValues().data());

    packImageWeightImpl(reinterpret_cast<nv_weight_t<nvdla::ConfigSet::nv_full>*>(blob_data), destDims, &weight,
                        srcData, NvDlaDims(weight), outputChannelOffset);
  }

  m_pMeta->m_Loadable.priv()->setSymbolContent(blob_name, b, blob_data);

  const MemoryListEntryId memoryId = m_pMeta->allocateMemory(
    ILoadable::MemoryDomain_SYSMEM, ILoadable::MemoryFlags_ALLOC | ILoadable::MemoryFlags_SET, b.size);

  ILoadable::MemoryListEntry& memory = m_pMeta->getMemoryListEntry(memoryId);
  memory.contents.push_back(blob_name);
  memory.offsets.push_back(0);

  return memoryId;
}

// lut_param->linear_exp_offset.exp_offset = 0
// lut_param->linear_exp_offset.frac_bits = 0
// lut_param->linear_exp_start = 0
// lut_param->linear_exp_end = 0
// lut_param->linear_only_offset.frac_bits = 0
// lut_param->linear_only_start = 1
// lut_param->linear_only_end = 1
// lut_param->method=LUT_METHOD_EXPONENTIAL
// REVIEW:
//   1. fp32 to fp16 conversion
//   2. pack LRN into function, pass function pointer and parameters?
//   3. General function todos?
//   4. How to verify?
//   5. Documentation or Description on how to setup 8
void NvDlaOp::SetLUTParam(dla_lut_param* lut_param, float alpha, float beta, float bias, int fsize, float outdata_scale, float outdata_offset)
{
  // LUT X contents
  float indata;
  if (lut_param->linear_exp_offset.exp_offset > 0)
    indata = (float)(1 << lut_param->linear_exp_offset.exp_offset);
  else
    indata = 1.0 / (1 << (-lut_param->linear_exp_offset.exp_offset));

  for (int i = 0; i < 65; i++) {
    float outdata = 1.0f / std::pow((bias + (alpha * indata / fsize)), beta);

    if (lut_param->method == LUT_METHOD_EXPONENTIAL)
      indata *= 2.0f;
    else
      indata += (1 << lut_param->linear_exp_offset.frac_bits);

    lut_param->linear_exp_table[i] = f2float16_ieee(outdata);
  }

  lut_param->hybrid_priority    = 0;
  lut_param->underflow_priority = 0;
  lut_param->overflow_priority  = 0;

  // calculate slope
  // for LRN and sigmoid, overflow and under flow are all zero
  // two sets are union, only one can be set at a time
  lut_param->linear_exp_underflow_slope.data_f = 0.000000;
  lut_param->linear_exp_overflow_slope.data_f  = 0.000000;

  lut_param->linear_only_underflow_slope.data_f = 0.000000;
  lut_param->linear_only_overflow_slope.data_f  = 0.000000;
}


std::pair<unsigned, bool> NvDlaOp::tryAllocateDataAndWeightsIntoCBuf(const NvDlaCubeInfo& data,
                                                                             NvDlaCubeInfo&       weight,
                                                                             Tensor::Dimension    yDilation) const
{
  unsigned minNumNeededDataBanks = 0;
  bool     shouldReuseWeight     = false;

  const CbufAllocType cbufAllocType = m_CbufAllocTypeGetter(data, weight, yDilation, minNumNeededDataBanks);
  switch (cbufAllocType) {
  case CbufAllocType::kFullDataFullWeight:
    // Full data and full weights
    weight.setBanksForFullWeights();
    shouldReuseWeight = true;
    break;
  case CbufAllocType::kFullDataPartialWeight:
    // Full data and partial weights
    weight.setBanksForPartialWeights();
    break;
  case CbufAllocType::kFullDataMinimumWeight:
    // Full data and minimum weights
    weight.setBanksForMinimumWeights();
    break;
  case CbufAllocType::kSplitDataFullWeight:
    // Split data and full weights
    weight.setBanksForFullWeights();
    shouldReuseWeight = true;
    break;
  case CbufAllocType::kSplitDataPartialWeight:
    // Split data and partial weights
    weight.setBanksForPartialWeights();
    break;
  case CbufAllocType::kSplitDataMinimumWeight:
    // Split data and minimum weights
    weight.setBanksForMinimumWeights();
    break;
  case CbufAllocType::kUnfeasible:
    weight.setBanksForMinimumWeights();
    std::ostringstream os;
    os << "data_banks(" << minNumNeededDataBanks << ") + weight_banks(" << weight.banks << ") > " << m_nvdla_constants.CBUF_BANK_NUM;
    //fatal(nvdla_exceed_hardware_limit) << os.str();
    std::cout<< os.str()<< std::endl;
    break;
  }

  return std::make_pair(minNumNeededDataBanks, shouldReuseWeight);
}


void NvDlaOp::relu(const onnc::Relu& pOp)
{
  const Tensor* input_X_t       = pOp.getInput(0);
  int32_t       input_X_ndim    = input_X_t->getNumOfDimensions();
  int32_t       input_X_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_X_ndim; ++i)
    input_X_dims[i] = input_X_t->dimension(i);
  NvDlaCubeInfo X_cube(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, input_X_dims[0], input_X_dims[1], input_X_dims[2], input_X_dims[3]);

  const Tensor* output_Y_t       = pOp.getOutput(0);
  int32_t       output_Y_ndim    = output_Y_t->getNumOfDimensions();
  int32_t       output_Y_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_Y_ndim; ++i)
    output_Y_dims[i] = output_Y_t->dimension(i);

  NvDlaCubeInfo Y_cube(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, output_Y_dims[0], output_Y_dims[1], output_Y_dims[2],
                       output_Y_dims[3]);

  NvDlaDlaOperation* relu_op = new NvDlaDlaOperation();
  relu_op->op_dep.op_type    = DLA_OP_SDP;

  struct dla_sdp_op_desc* relu_desc     = (struct dla_sdp_op_desc*)(&(relu_op->op_desc));
  relu_desc->src_precision              = m_nvdla_constants.DLA_PRECISION;
  relu_desc->dst_precision              = m_nvdla_constants.DLA_PRECISION;
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
  relu_desc->x1_op.precision            = m_nvdla_constants.DLA_PRECISION;
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
  relu_surf->src_data.address            = issueDlaAddr(*input_X_t, X_cube);
  relu_surf->src_data.size               = m_pMeta->getMemoryListEntrySize(*input_X_t);
  relu_surf->src_data.width              = X_cube.dim_w;
  relu_surf->src_data.height             = X_cube.dim_h;
  relu_surf->src_data.channel            = X_cube.dim_c;
  relu_surf->src_data.line_stride        = X_cube.stride_line;
  relu_surf->src_data.surf_stride        = X_cube.stride_surface;
  relu_surf->src_data.plane_stride       = X_cube.stride_plane;

  relu_surf->dst_data.type         = DLA_MEM_MC;
  relu_surf->dst_data.address      = issueDlaAddr(*output_Y_t, Y_cube);
  relu_surf->dst_data.size         = m_pMeta->getMemoryListEntrySize(*output_Y_t);
  relu_surf->dst_data.width        = Y_cube.dim_w;
  relu_surf->dst_data.height       = Y_cube.dim_h;
  relu_surf->dst_data.channel      = Y_cube.dim_c;
  relu_surf->dst_data.line_stride  = Y_cube.stride_line;
  relu_surf->dst_data.surf_stride  = Y_cube.stride_surface;
  relu_surf->dst_data.plane_stride = Y_cube.stride_plane;

  issueDlaOp(relu_op, NULL, m_pMeta->m_pPrevOp);
}

void NvDlaOp::conv(const onnc::Conv& pOp)
{
    const Tensor* input_X_t       = pOp.getInput(0);
  int32_t       input_X_ndim    = input_X_t->getNumOfDimensions();
  int32_t       input_X_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_X_ndim; ++i)
    input_X_dims[i] = input_X_t->dimension(i);

  const Tensor* input_W_t       = pOp.getInput(1);
  int32_t       input_W_ndim    = input_W_t->getNumOfDimensions();
  int32_t       input_W_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_W_ndim; ++i)
    input_W_dims[i] = input_W_t->dimension(i);

  const Tensor* const     input_B_t    = (pOp.hasBias() ? pOp.getB() : nullptr);
  const Tensor::Dimension input_B_size = (pOp.hasBias() ? input_B_t->dimension(0) : 1);

  const Tensor* output_Y_t       = pOp.getOutput(0);
  int32_t       output_Y_ndim    = output_Y_t->getNumOfDimensions();
  int32_t       output_Y_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_Y_ndim; ++i)
    output_Y_dims[i] = output_Y_t->dimension(i);
  NvDlaCubeInfo Y_cube(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, output_Y_dims[0], output_Y_dims[1], output_Y_dims[2],
                       output_Y_dims[3]);

  // Prepare attributes
  int32_t number_of_dilations = pOp.getDilations().vector().size();
  int32_t dilations[2]        = {1, 1};
  for (int i = 0; i < number_of_dilations; ++i)
    dilations[i] = pOp.getDilations().at(i);
  const int32_t numGroups              = pOp.getGroup().value();
  int32_t       number_of_kernel_shape = pOp.getKernelShape().vector().size();
  int32_t       kernel_shape[2]        = {1, 1};
  for (int i = 0; i < number_of_kernel_shape; ++i)
    kernel_shape[i] = pOp.getKernelShape().at(i);
  int32_t number_of_pads = pOp.getPads().vector().size();
  int32_t pads[4]        = {0, 0, 0, 0};
  for (int i = 0; i < number_of_pads; ++i)
    pads[i] = pOp.getPads().at(i);
  int32_t number_of_strides = pOp.getStrides().vector().size();
  int32_t strides[2]        = {1, 1};
  for (int i = 0; i < number_of_strides; ++i)
    strides[i] = pOp.getStrides().at(i);

  NvDlaCubeInfo X_cube(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, input_X_dims[0],
                       input_X_dims[1], input_X_dims[2], input_X_dims[3], pads[1], pads[3]);

  input_X_dims[1] /= numGroups;
  input_W_dims[0] /= numGroups;

  NvDlaCubeInfo fcube_group(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, input_X_dims[0],
                            input_X_dims[1], input_X_dims[2], input_X_dims[3], pads[1], pads[3]);
  const Tensor::Dimension numInputChannelsPerGroup  = input_X_dims[1];
  const Tensor::Dimension numOutputChannelsPerGroup = input_W_dims[0];
  for (std::int32_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
    // output/input channel offsets w/o alignment by atomic k
    const Tensor::Dimension inputChannelOffset        = groupIdx * numInputChannelsPerGroup;
    const Tensor::Dimension alignedInputChannelOffset = ((inputChannelOffset / m_nvdla_constants.MAC_ATOMIC_K) * m_nvdla_constants.MAC_ATOMIC_K);
    const Tensor::Dimension outputChannelOffset       = groupIdx * numOutputChannelsPerGroup;
    const Tensor::Dimension alignedOutputChannelOffset =
      groupIdx * UNIT_ALIGNMENT(numOutputChannelsPerGroup, m_nvdla_constants.MAC_ATOMIC_K);

    const Tensor::Dimension numWeightFrontPaddingChannels = inputChannelOffset - alignedInputChannelOffset;

    NvDlaCubeInfo winfo(this->m_nvdla_constants, NVDLA_CUBE_WEIGHT, input_W_dims[0],
                        numWeightFrontPaddingChannels + numInputChannelsPerGroup, input_W_dims[2], input_W_dims[3]);

    // Weight Memory allocation, repacking by groups
    int W_mid = -1;
    W_mid = packWeight(*input_W_t, NvDlaDims(input_W_dims), numWeightFrontPaddingChannels, outputChannelOffset);

    int W_addr = issueDlaAddr(W_mid, winfo);
    int B_mid  = -1;
    int B_addr = -1;

    NvDlaCubeInfo B_info(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, numOutputChannelsPerGroup, 1, 1, 1);

    if (pOp.hasBias()) {
      B_mid  = packBias(*input_B_t, numOutputChannelsPerGroup, outputChannelOffset);
      B_addr = issueDlaAddr(B_mid, B_info);
    }

    int pad_top       = pads[0];
    int pad_left      = pads[1];
    int pad_bottom    = pads[2];
    int pad_right     = pads[3];
    int kernel_height = input_W_dims[2] + (dilations[0] - 1) * (input_W_dims[2] - 1);
    int stride_y      = strides[0];
    int input_height  = input_X_dims[2];
    int output_height = output_Y_dims[2];
    assert(output_height == ((input_height + pad_top + pad_bottom - kernel_height) / stride_y + 1));

    unsigned min_data_banks_needed = 0;
    bool     is_weight_reuse       = false;
    std::tie(min_data_banks_needed, is_weight_reuse) =
      tryAllocateDataAndWeightsIntoCBuf(fcube_group, winfo, dilations[0]);

    int affordable_conv_height = (m_nvdla_constants.CBUF_BANK_DEPTH * (m_nvdla_constants.CBUF_BANK_NUM - winfo.banks)) / fcube_group.eps;

    int data_banks_of_first_split_layer = -1;
    int input_h_idx         = -pad_top; // the starting H where the input data of each split convolution comes from.
    int output_h_idx        = 0;        // the starting H where the output data of each split convolution goes to.
    int is_first_split      = true;
    int is_last_split       = false;
    int unused_input_height = (input_height + pad_top) - (kernel_height + stride_y * (output_height - 1));
    if (unused_input_height < 0)
      unused_input_height = 0;

    do { // for each split convolution
      is_last_split = (input_height - unused_input_height - std::max(input_h_idx, 0)) <= affordable_conv_height;

      int input_split_height;
      int output_split_height;
      if (is_first_split && is_last_split) { // No split
        output_split_height = output_height;
        input_split_height  = input_height;
      } else if (is_last_split) {
        input_split_height  = input_height - unused_input_height - std::max(input_h_idx, 0);
        output_split_height = output_height - output_h_idx;
        assert(output_split_height == ((input_split_height + pad_bottom - kernel_height) / stride_y + 1));
      } else if (is_first_split) {
        output_split_height = (affordable_conv_height + pad_top - kernel_height) / stride_y + 1;
        input_split_height  = kernel_height + stride_y * (output_split_height - 1) - pad_top;
      } else { // Split convolution in middle
        output_split_height = (affordable_conv_height - kernel_height) / stride_y + 1;
        input_split_height  = kernel_height + stride_y * (output_split_height - 1);
      }

      int split_pad_top    = (is_first_split) ? pad_top : 0;
      int split_pad_bottom = (is_last_split) ? pad_bottom : 0;

      NvDlaCubeInfo finfo(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, input_X_dims[0],
                          numWeightFrontPaddingChannels + numInputChannelsPerGroup, input_split_height,
                          input_X_dims[3], pads[1], pads[3]);

      if (is_first_split) {
        data_banks_of_first_split_layer = finfo.banks;
      } else {                 // The rest split layers
        if (is_weight_reuse) { // All weights are buffered in the CBUF, so weights are reused among split layers.
          if (!(finfo.banks <= data_banks_of_first_split_layer)) {
            //fatal(nvdla_unexpected_num_of_banks) << PP_STRINGIFY(finfo.banks) << finfo.banks;
            printf("nvdla_unexpected_num_of_banks:%d\n", finfo.banks);
          }
          // All split layers must have the same data_bank setting.
          finfo.banks = data_banks_of_first_split_layer;
        }
      }

      NvDlaCubeInfo oinfo(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, output_Y_dims[0], numOutputChannelsPerGroup, output_split_height,
                          output_Y_dims[3]);

      NvDlaDlaOperation* const conv_op = new NvDlaDlaOperation();
      conv_op->op_dep.op_type          = DLA_OP_CONV;

      struct dla_conv_op_desc* conv_desc = (struct dla_conv_op_desc*)(&(conv_op->op_desc));
      conv_desc->conv_mode               = CONV_MODE_DIRECT;
      conv_desc->data_reuse              = 0;
      conv_desc->weight_reuse            = (is_weight_reuse && !is_first_split) ? 1 : 0;
      conv_desc->skip_data_rls           = 0;
      conv_desc->skip_weight_rls         = (is_weight_reuse && !is_last_split) ? 1 : 0;
      conv_desc->entry_per_slice         = finfo.eps;
      conv_desc->data_format = FORMAT_FEATURE;
      conv_desc->pixel_mapping = 0;
      conv_desc->fetch_grain   = 1;
      conv_desc->batch         = 1;
      conv_desc->weight_format = WEIGHT_FORMAT_UNCOMPRESSED;
      conv_desc->data_bank     = finfo.banks;
      conv_desc->weight_bank   = winfo.banks;
      conv_desc->batch_stride   = 0;
      conv_desc->post_extension = 0;
      conv_desc->pixel_override = 0;
      conv_desc->release = input_split_height;
      conv_desc->input_width_csc    = finfo.dim_w;
      conv_desc->input_height_csc   = finfo.dim_h;
      conv_desc->input_channel_csc  = finfo.dim_c;
      conv_desc->kernel_channel_csc = winfo.dim_c;
      conv_desc->kernel_width_csc   = winfo.dim_w;
      conv_desc->kernel_height_csc  = winfo.dim_h;
      conv_desc->input_width_cmac   = output_Y_dims[3];
      conv_desc->input_height_cmac  = output_split_height;
      conv_desc->bytes_per_kernel   = winfo.dim_c * winfo.dim_h * winfo.dim_w * m_nvdla_constants.ELEMENT_SIZE;
      conv_desc->mean_ry            = 0;
      conv_desc->mean_gu            = 0;
      conv_desc->mean_bv            = 0;
      conv_desc->mean_ax            = 0;
      conv_desc->mean_format        = 0;
      conv_desc->conv_stride_x      = strides[1];
      conv_desc->conv_stride_y      = strides[0];
      conv_desc->pad_x_left         = pad_left;
      conv_desc->pad_x_right        = pad_right;
      conv_desc->pad_y_top          = split_pad_top;
      conv_desc->pad_y_bottom       = split_pad_bottom;
      conv_desc->dilation_x         = dilations[1];
      conv_desc->dilation_y         = dilations[0];
      conv_desc->pra_truncate       = 0;
      conv_desc->in_precision       = m_nvdla_constants.DLA_PRECISION;
      conv_desc->out_precision      = m_nvdla_constants.DLA_PRECISION;
      conv_desc->out_cvt.scale      = 1;
      conv_desc->out_cvt.enable     = 1;
      conv_desc->pad_val            = 0;

      struct dla_conv_surface_desc* conv_surf = (struct dla_conv_surface_desc*)(&(conv_op->op_surf));
      conv_surf->weight_data.type             = DLA_MEM_MC;
      conv_surf->weight_data.address          = W_addr;
      conv_surf->weight_data.size             = m_pMeta->getMemoryListEntrySize(W_mid);
      conv_surf->weight_data.width            = winfo.dim_w;
      conv_surf->weight_data.height           = winfo.dim_h;
      conv_surf->weight_data.channel          = winfo.dim_c;
      conv_surf->weight_data.line_stride      = 0;
      conv_surf->weight_data.surf_stride      = 0;
      conv_surf->weight_data.plane_stride     = 0;

      conv_surf->wmb_data.type    = DLA_MEM_HW;
      conv_surf->wmb_data.address = -1;

      conv_surf->wgs_data.type    = DLA_MEM_HW;
      conv_surf->wgs_data.address = -1;

      conv_surf->src_data.type = DLA_MEM_MC;
      conv_surf->src_data.address =
        issueDlaAddr(*input_X_t, X_cube, alignedInputChannelOffset, std::max(input_h_idx, 0));
      conv_surf->src_data.size         = finfo.size;
      conv_surf->src_data.width        = finfo.dim_w;
      conv_surf->src_data.height       = finfo.dim_h;
      conv_surf->src_data.channel      = finfo.dim_c;
      conv_surf->src_data.line_stride  = fcube_group.stride_line;
      conv_surf->src_data.surf_stride  = fcube_group.stride_surface;
      conv_surf->src_data.plane_stride = fcube_group.stride_plane;

      conv_surf->dst_data.type         = DLA_MEM_HW;
      conv_surf->dst_data.address      = -1;
      conv_surf->dst_data.size         = oinfo.size;
      conv_surf->dst_data.width        = oinfo.dim_w;
      conv_surf->dst_data.height       = oinfo.dim_h;
      conv_surf->dst_data.channel      = oinfo.dim_c;
      conv_surf->dst_data.line_stride  = Y_cube.stride_line;
      conv_surf->dst_data.surf_stride  = Y_cube.stride_surface;
      conv_surf->dst_data.plane_stride = Y_cube.stride_plane;

      // Bias Add
      NvDlaDlaOperation* const add_op = new NvDlaDlaOperation();
      add_op->op_dep.op_type          = DLA_OP_SDP;

      struct dla_sdp_op_desc* add_desc = (struct dla_sdp_op_desc*)(&(add_op->op_desc));
      add_desc->src_precision          = m_nvdla_constants.DLA_PRECISION;
      add_desc->dst_precision          = m_nvdla_constants.DLA_PRECISION;
      add_desc->lut_index              = -1;
      add_desc->conv_mode              = 0;
      add_desc->out_cvt.scale          = 1;
      add_desc->out_cvt.truncate       = 0;
      add_desc->out_cvt.enable         = 1;
      add_desc->out_cvt.offset         = 0;
      add_desc->conv_mode              = CONV_MODE_DIRECT;
      add_desc->batch_num              = 1;
      add_desc->batch_stride           = 0;
      if (pOp.hasBias()) {
        add_desc->x1_op.enable               = 1;
        add_desc->x1_op.alu_type             = SDP_ALU_OP_SUM;
        add_desc->x1_op.type                 = SDP_OP_ADD;
        add_desc->x1_op.mode                 = SDP_OP_PER_KERNEL;
        add_desc->x1_op.act                  = ACTIVATION_NONE;
        add_desc->x1_op.shift_value          = 0;
        add_desc->x1_op.truncate             = 0;
        add_desc->x1_op.precision            = m_nvdla_constants.DLA_PRECISION;
        add_desc->x1_op.alu_operand          = 0;
        add_desc->x1_op.mul_operand          = 0;
        add_desc->x1_op.cvt.alu_cvt.scale    = 0;
        add_desc->x1_op.cvt.alu_cvt.truncate = 0;
        add_desc->x1_op.cvt.alu_cvt.enable   = 0;
        add_desc->x1_op.cvt.alu_cvt.offset   = 0;
        add_desc->x1_op.cvt.mul_cvt.scale    = 0;
        add_desc->x1_op.cvt.mul_cvt.truncate = 0;
        add_desc->x1_op.cvt.mul_cvt.enable   = 0;
        add_desc->x1_op.cvt.mul_cvt.offset   = 0;
      } else {
        add_desc->x1_op.enable = 0;
      }

      add_desc->x2_op.enable = 0;
      add_desc->y_op.enable  = 0;

      struct dla_sdp_surface_desc* add_surf = (struct dla_sdp_surface_desc*)(&(add_op->op_surf));
      add_surf->src_data.type               = DLA_MEM_HW;
      add_surf->src_data.address            = -1;
      add_surf->src_data.size               = conv_surf->dst_data.size;
      add_surf->src_data.width              = conv_surf->dst_data.width;
      add_surf->src_data.height             = conv_surf->dst_data.height;
      add_surf->src_data.channel            = conv_surf->dst_data.channel;
      add_surf->src_data.line_stride        = 0;
      add_surf->src_data.surf_stride        = 0;
      add_surf->src_data.plane_stride       = 0;

      if (pOp.hasBias()) {
        add_surf->x1_data.type         = DLA_MEM_MC;
        add_surf->x1_data.address      = B_addr;
        add_surf->x1_data.size         = m_pMeta->getMemoryListEntrySize(B_mid);
        add_surf->x1_data.width        = 1;
        add_surf->x1_data.height       = 1;
        add_surf->x1_data.channel      = numOutputChannelsPerGroup;
        add_surf->x1_data.line_stride  = B_info.stride_line;
        add_surf->x1_data.surf_stride  = B_info.stride_surface;
        add_surf->x1_data.plane_stride = B_info.stride_plane;
      }

      add_surf->dst_data.type         = DLA_MEM_MC;
      add_surf->dst_data.address      = issueDlaAddr(*output_Y_t, Y_cube, alignedOutputChannelOffset, output_h_idx);
      add_surf->dst_data.size         = conv_surf->dst_data.size;
      add_surf->dst_data.width        = conv_surf->dst_data.width;
      add_surf->dst_data.height       = conv_surf->dst_data.height;
      add_surf->dst_data.channel      = conv_surf->dst_data.channel;
      add_surf->dst_data.line_stride  = conv_surf->dst_data.line_stride;
      add_surf->dst_data.surf_stride  = conv_surf->dst_data.surf_stride;
      add_surf->dst_data.plane_stride = conv_surf->dst_data.plane_stride;

      NvDlaDlaOperation* prev_op = (is_first_split) ? m_pMeta->m_pPrevOp : NULL;
      issueDlaOp(conv_op, add_op, prev_op);

      output_h_idx = output_h_idx + output_split_height;
      input_h_idx  = stride_y * output_h_idx - pad_top;

      is_first_split = false;
    } while (!is_last_split);
  }
}

void NvDlaOp::add(const onnc::Add& pOp)
{
  this->emitSdp(SDP_OP_ADD, *pOp.getA(), *pOp.getB(), *pOp.getOutput(0));
}


void NvDlaOp::max_pool(const onnc::MaxPool& pOp)
{
  const Tensor* input_X_t       = pOp.getInput(0);
  int32_t       input_X_ndim    = input_X_t->getNumOfDimensions();
  int32_t       input_X_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_X_ndim; ++i)
    input_X_dims[i] = input_X_t->dimension(i);
  MemoryListEntryId          X_mid = m_pMeta->getMemoryListEntryId(*input_X_t);
  ILoadable::MemoryListEntry X_mle = m_pMeta->getMemoryListEntry(X_mid);
  NvDlaCubeInfo X_cube(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, input_X_dims[0], input_X_dims[1], input_X_dims[2], input_X_dims[3]);

  // Prepare output
  const Tensor* output_Y_t       = pOp.getOutput(0);
  int32_t       output_Y_ndim    = output_Y_t->getNumOfDimensions();
  int32_t       output_Y_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_Y_ndim; ++i)
    output_Y_dims[i] = output_Y_t->dimension(i);
  MemoryListEntryId          Y_mid = m_pMeta->getMemoryListEntryId(*output_Y_t);
  ILoadable::MemoryListEntry Y_mle = m_pMeta->getMemoryListEntry(Y_mid);
  NvDlaCubeInfo Y_cube(this->m_nvdla_constants, NVDLA_CUBE_FEATURE, output_Y_dims[0], output_Y_dims[1], output_Y_dims[2], output_Y_dims[3]);

  const Tensor* output_Indices_t    = NULL;
  void*         output_Indices      = NULL;
  int32_t       output_Indices_ndim = 0;
  if (pOp.getNumOfOutputs() > 1) {
    output_Indices_t    = pOp.getOutput(1);
    output_Indices_ndim = output_Indices_t->getNumOfDimensions();
  }
  int32_t output_Indices_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_Indices_ndim; ++i)
    output_Indices_dims[i] = output_Indices_t->dimension(i);

  // Prepare attributes
  int32_t number_of_kernel_shape = pOp.getKernelShape().vector().size();
  int32_t kernel_shape[number_of_kernel_shape];
  for (int i = 0; i < number_of_kernel_shape; ++i)
    kernel_shape[i] = pOp.getKernelShape().at(i);
  int32_t number_of_pads = pOp.getPads().vector().size();
  int32_t pad_shapes[number_of_pads];
  for (int i = 0; i < number_of_pads; ++i)
    pad_shapes[i] = pOp.getPads().at(i);
  int32_t storage_order     = pOp.getStorageOrder().value();
  int32_t number_of_strides = pOp.getStrides().vector().size();
  int32_t strides[number_of_strides];
  for (int i = 0; i < number_of_strides; ++i)
    strides[i] = pOp.getStrides().at(i);

  NvDlaDlaOperation* maxpool_op = new NvDlaDlaOperation();
  maxpool_op->op_dep.op_type    = DLA_OP_PDP;

  struct dla_pdp_op_desc* maxpool_desc = (struct dla_pdp_op_desc*)(&(maxpool_op->op_desc));
  maxpool_desc->partial_in_width_first = 0;
  maxpool_desc->partial_in_width_mid   = 0;
  maxpool_desc->partial_in_width_last  = 0;
  maxpool_desc->partial_width_first    = 0;
  maxpool_desc->partial_width_mid      = 0;
  maxpool_desc->partial_width_last     = 0;
  maxpool_desc->split_num              = 1;
  maxpool_desc->pool_mode              = POOL_MODE_MAX;
  maxpool_desc->pool_width             = kernel_shape[1] - 1;
  maxpool_desc->pool_height            = kernel_shape[0] - 1;
  maxpool_desc->stride_x               = strides[1];
  maxpool_desc->stride_y               = strides[0];
  maxpool_desc->pad_top                = pad_shapes[0]; // pad_shape - H
  maxpool_desc->pad_left               = pad_shapes[1]; // pad_shape - W
  maxpool_desc->pad_bottom             = pad_shapes[2];
  maxpool_desc->pad_right              = pad_shapes[3];

  maxpool_desc->precision = m_nvdla_constants.DLA_PRECISION;

  struct dla_pdp_surface_desc* maxpool_surf = (struct dla_pdp_surface_desc*)(&(maxpool_op->op_surf));
  maxpool_surf->src_data.type               = DLA_MEM_MC;
  maxpool_surf->src_data.address            = issueDlaAddr(X_mid, X_cube);
  maxpool_surf->src_data.size               = X_mle.size;
  maxpool_surf->src_data.width              = X_cube.dim_w;
  maxpool_surf->src_data.height             = X_cube.dim_h;
  maxpool_surf->src_data.channel            = X_cube.dim_c;
  maxpool_surf->src_data.line_stride        = X_cube.stride_line;
  maxpool_surf->src_data.surf_stride        = X_cube.stride_surface;
  maxpool_surf->src_data.plane_stride       = X_cube.stride_plane;

  maxpool_surf->dst_data.type         = DLA_MEM_MC;
  maxpool_surf->dst_data.address      = issueDlaAddr(Y_mid, Y_cube);
  maxpool_surf->dst_data.size         = Y_mle.size;
  maxpool_surf->dst_data.width        = Y_cube.dim_w;
  maxpool_surf->dst_data.height       = Y_cube.dim_h;
  maxpool_surf->dst_data.channel      = Y_cube.dim_c;
  maxpool_surf->dst_data.line_stride  = Y_cube.stride_line;
  maxpool_surf->dst_data.surf_stride  = Y_cube.stride_surface;
  maxpool_surf->dst_data.plane_stride = Y_cube.stride_plane;

  issueDlaOp(maxpool_op, NULL, m_pMeta->m_pPrevOp);
}


template<typename T>
void NvDlaOp::set_default_strides(T &pOp)
{
  auto input = pOp.getInput(0);
  std::vector<int64_t> v(input->getNumOfDimensions() - 2, 1);

  #ifdef NVDLA_DEBUG
  std::cout<< "default strides"<< std::endl;
  for(auto i : v)
  {
    std::cout<< i;
    std::cout<< " ,";
  }
  std::cout<< std::endl;
  #endif
  
  pOp.setStrides(std::move(v));
}


template<typename T>
void NvDlaOp::set_default_dilations(T &pOp)
{
    auto input = pOp.getInput(0);
    std::vector<int64_t> v(input->getNumOfDimensions() - 2, 1);
    
    #ifdef NVDLA_DEBUG
    std::cout<< "default dilations"<< std::endl;
    for(auto i: v)
    {
      std::cout<< i;
      std::cout<< " ,";
    }
    std::cout<< std::endl;
    #endif

    pOp.setDilations(std::move(v));
}


template<typename T>
void NvDlaOp::set_default_kernel_shape(T &pOp) {
    auto weight = pOp.getInput(1);
    std::vector<int64_t> v(weight->getNumOfDimensions() - 2);
    auto& dimensions = weight->getDimensions();

    for (int i = 0; i < v.size(); ++i) {
        v[i] = dimensions[i + 2];
    }
    
    #ifdef NVDLA_DEBUG
    std::cout<< "default kernel shape:"<< std::endl;
    for(auto i : v)
    {
      std::cout<< i;
      std::cout<< " ,";
    }
    std::cout<< std::endl;
    #endif

    pOp.setKernelShape(v);
}
  
template<typename T>
void NvDlaOp::set_default_pads(T &pOp)
{
    auto& kernel_shape = pOp.getKernelShape();
    assert(2 <= kernel_shape.vector().size());

    std::vector<int64_t> v(kernel_shape.vector().size() * 2, 0);
    
    #ifdef NVDLA_DEBUG
    std::cout<< "default pads"<< std::endl;
    for(auto i: v)
    { 
      std::cout<< i;
      std::cout<< " ,";
    }
    std::cout<< std::endl;
    #endif

    pOp.setPads(std::move(v));
}

void NvDlaOp::set_default_attributs(Conv &pConv)
{
    this->set_default_dilations(pConv);
    this->set_default_kernel_shape(pConv);
    this->set_default_pads(pConv);
    this->set_default_strides(pConv);
}

void NvDlaOp::set_default_attributs(MaxPool &pMaxPool)
{
    this->set_default_pads(pMaxPool);
    this->set_default_strides(pMaxPool);
}
