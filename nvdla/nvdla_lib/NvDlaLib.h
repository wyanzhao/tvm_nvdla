#ifndef NVDLA_NVDLA_LIB_NVDLA_LIB_H
#define NVDLA_NVDLA_LIB_NVDLA_LIB_H

#include "NvDlaOp.h"
#include "NvDlaDefine.h"

#include <onnc/ADT/StringList.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Tensor.h>
#include <onnc/IR/CustomVisitor.h>
#include <onnc/Support/Preprocessor.h>
#include <onnc/Support/Span.h>

#include <functional>

using namespace onnc;

class NvDlaLib : public NvDlaOp
{
public:
  using CbufAllocTypeGetter =
    std::function<CbufAllocType(const NvDlaCubeInfo&, const NvDlaCubeInfo&, Tensor::Dimension, unsigned&)>;

public:
  NvDlaLib()
  :
    NvDlaOp(getConfig(nvdla::ConfigSet::nv_full, nvdla::ExecutionMode::direct, false), std::bind(&NvDlaLib::getCbufAllocType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4))
  {
    m_nvdla_constants = NvDlaConstants( getConfig(nvdla::ConfigSet::nv_full, nvdla::ExecutionMode::direct, false));
  }

  void compile(onnc::Module& pModule, onnc::ComputeGraph& pCG);

template<typename TensorTy>
Tensor* create_compute_tensor(ComputeGraph& pCG, const StringRef& pName,
                                   const Tensor::Dimensions& pDims);
                                   
Tensor* create_float_compute_tensor(ComputeGraph& pCG, const StringRef& pName,
                         const Tensor::Dimensions& pDims);

template<typename TensorTy>
void create_weight_operator(ComputeGraph& pCG, const std::string& pName,
                                 const Tensor::Dimensions& pDims);

void
create_float_weight_operator(ComputeGraph& pCG, const std::string& pName,
                          const Tensor::Dimensions& pDims, const std::string& weight_path);

template<typename OpTy, typename ... NodeCtorParams>
OpTy* create_compute_operator(ComputeGraph& pCG,
                                   const StringList& pInputNames,
                                   NodeCtorParams&& ... pParams)
{
  OpTy* op = pCG.addOperator<OpTy>(pParams...);
  for (auto& iname : pInputNames)
    op->addInput(*pCG.getValue<Tensor>(iname));
  return op;
}                                   

template<typename VTy>
std::vector<VTy> get_values(const std::vector<VTy>& pVec)
{
  return pVec;
}                                          

private:
  CbufAllocType getCbufAllocType(const NvDlaCubeInfo& xinfo, const NvDlaCubeInfo& winfo, Tensor::Dimension yDilation,
                                 unsigned& minNumNeededDataBanks);

  int submitEvent(int task_id, int event_type);
  int submitMemAllocAddress(int size, std::string& blob_name);
  void init_nvdla_memory(onnc::Module& pModule);
  void task_submit();
  void nvdla_filegen();

protected:
  NvDlaConstants m_nvdla_constants;
};

#endif