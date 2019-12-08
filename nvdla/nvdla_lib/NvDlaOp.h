#ifndef NVDLA_NVDLA_LIB_NVDLA_OP_H
#define NVDLA_NVDLA_LIB_NVDLA_OP_H

#include "NvDlaMeta.h"

#include <onnc/IR/Compute/Relu.h>
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/Add.h>
#include <onnc/IR/Compute/MaxPool.h>

#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Tensor.h>
#include <onnc/IR/CustomVisitor.h>
#include <onnc/Core/CustomPass.h>
#include <onnc/Support/Preprocessor.h>
#include <onnc/Support/Span.h>

#include <functional>

using namespace onnc;

class NvDlaOp {
public:
  using CbufAllocTypeGetter =
    std::function<CbufAllocType(const NvDlaCubeInfo&, const NvDlaCubeInfo&, Tensor::Dimension, unsigned&)>;

public:
  void relu(const onnc::Relu& pOp);
  void conv(const onnc::Conv& pOp);
  void add (const onnc::Add & pOp);
  void max_pool(const onnc::MaxPool& pOp);
  void reshape(const onnc::Reshape& pOp) {};
  
  void set_default_attributs(Conv &pOp);
  void set_default_attributs(MaxPool &pOp);
  void set_default_attributs(Relu &pOp) {}
  void set_default_attributs(Reshape &pOp) {}
  void set_default_attributs(Add &pOp) {}
  void set_default_attributs(OutputOperator &pOp) {}
  void set_default_attributs(InputOperator &pOp) {}
  void set_default_attributs(Initializer &pOp) {}
protected:
  NvDlaOp(const NvDlaConstants& constants, CbufAllocTypeGetter cbufAllocTypeGetter) noexcept
  : m_nvdla_constants(constants)
  , m_CbufAllocTypeGetter{std::move(cbufAllocTypeGetter)}
  { 
    auto tmp = new NvDlaBackendMeta(m_nvdla_constants);
    m_pMeta = std::move(tmp);
  }


  MemoryListEntryId packWeight(const Tensor& weight, NvDlaDims destDims, Tensor::Dimension numFrontPaddingChannels,
                               Tensor::Dimension outputChannelOffset);
  MemoryListEntryId packImageWeight(const Tensor& weight, NvDlaDims destDims, Tensor::Dimension outputChannelOffset);
  MemoryListEntryId packBias(const Tensor& bias, Tensor::Dimension numDestChannels,
                             Tensor::Dimension srcChannelOffset = 0);
  MemoryListEntryId packSDPOperand(const Tensor* aluTensor, const Tensor* mulTensor, const NvDlaCubeInfo& cubeInfo);

  MemoryListEntryId  packFeature(const Tensor& tensor, const NvDlaCubeInfo& cube);
  void               issueEmuOp(NvDlaEmuOperation* op);
  AddressListEntryId issueEmuAddr(MemoryListEntryId mid);
  void               issueDlaOp(NvDlaDlaOperation* op, NvDlaDlaOperation* op_fuse, NvDlaDlaOperation* op_prev);
  void               issueDlaOp(std::unique_ptr<NvDlaDlaOperation> op);
  AddressListEntryId issueDlaAddr(const Tensor& tensor, const NvDlaCubeInfo& cube, Tensor::Dimension channelOffset,
                                  NvDlaBackendMeta::Offset hOffset);
  AddressListEntryId issueDlaAddr(const Tensor& tensor, const NvDlaCubeInfo& cube);
  AddressListEntryId issueDlaAddr(MemoryListEntryId memoryId, const NvDlaCubeInfo& cube);
  AddressListEntryId issueSDPOperand(const Tensor& tensor, const NvDlaCubeInfo& cube, MemoryListEntryId& memoryId);

  void SetLUTParam(dla_lut_param* lut_param, float alpha, float beta, float bias, int size, float outdata_scale, float outdata_offset);

  // Perform SDP for 2 input tensors and an output tensor,
  // the possible value for parameter 'opType' is:
  //
  //   1. SDP_OP_ADD
  //   2. SDP_OP_MUL
  //
  void emitSdp(std::uint8_t opType, const Tensor& firstInput, const Tensor& secondInput, const Tensor& output);
  std::pair<unsigned, bool> tryAllocateDataAndWeightsIntoCBuf(const NvDlaCubeInfo& data, NvDlaCubeInfo& weight,
                                                              Tensor::Dimension yDilation) const;
protected:
  template<typename T>
  void set_default_strides(T &pOp);

  template<typename T>
  void set_default_dilations(T &pOp); 

  template<typename T>
  void set_default_kernel_shape(T &pOp);
  
  template<typename T>
  void set_default_pads(T &pOp);

private:

  MemoryListEntryId packWeight(span<const float> weight, const Tensor* weightTensor, NvDlaDims srcDims,
                               NvDlaDims destDims, Tensor::Dimension numFrontPaddingChannels,
                               Tensor::Dimension outputChannelOffset);

  MemoryListEntryId packWeight(const Tensor& weight, NvDlaDims srcDims, NvDlaDims destDims,
                               Tensor::Dimension numFrontPaddingChannels, Tensor::Dimension outputChannelOffset);

  template <typename Type>
  void packWeightImpl(Type* destData, NvDlaDims destDimsWithFrontPadding, const Tensor* tensor, const float* srcData,
                      NvDlaDims srcDims, Tensor::Dimension numFrontPaddingChannels,
                      Tensor::Dimension outputChannelOffset);

  template <typename Type>
  void packImageWeightImpl(Type* blob, NvDlaDims blobDims, const Tensor* tensor, const float* srcData,
                           NvDlaDims srcDims, Tensor::Dimension outputChannelOffset);

  template <typename Type>
  void packBiasImpl(Type* destData, Tensor::Dimension numDestChannels, const Tensor* tensor, const float* srcData,
                    Tensor::Dimension srcChannelOffset);

  void packSDPOperandImpl(NvU8* blob, const Tensor* aluTensor, const float* aluData, const Tensor* mulTensor,
                          const float* mulData, const NvDlaCubeInfo& cubeInfo);
protected:
  NvDlaConstants            m_nvdla_constants;
  NvDlaBackendMeta*         m_pMeta;
  const CbufAllocTypeGetter m_CbufAllocTypeGetter;
};


#endif