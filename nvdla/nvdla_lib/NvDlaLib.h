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
  NvDlaLib(onnc::Module* pModule, onnc::ComputeGraph* pCG)
  : NvDlaOp(getConfig(nvdla::ConfigSet::nv_full, nvdla::ExecutionMode::direct, false), std::bind(&NvDlaLib::getCbufAllocType, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4))
  {
    m_pModule = pModule;
    m_pCG = pCG;
    m_nvdla_constants = NvDlaConstants( getConfig(nvdla::ConfigSet::nv_full, nvdla::ExecutionMode::direct, false));
  }
  ~NvDlaLib();

  void optimize();

  void compile();

  template<typename TensorTy>
  Tensor* create_compute_tensor(const StringRef& pName,
                                   const Tensor::Dimensions& pDims)
  {
    Tensor* t = m_pCG->addValue<TensorTy>(pName);
    t->setDimensions(pDims);
    return t;
  }                                   
                                   
  Tensor* create_float_compute_tensor(const StringRef& pName,
                         const Tensor::Dimensions& pDims);
                         
  template<typename TensorTy>
  void create_weight_operator(const std::string& pName,
                                 const Tensor::Dimensions& pDims){
  Initializer* init = m_pCG->addOperator<Initializer>(pName);
  Tensor* value = this->create_compute_tensor<TensorTy>(pName, pDims);
  init->setTensor(*value);
  };

  void
  create_float_weight_operator(const std::string& pName,
                          const Tensor::Dimensions& pDims, const std::string& weight_path);

  template<typename OpTy, typename ... NodeCtorParams>
  OpTy* create_compute_operator(const StringList& pInputNames,
                                   NodeCtorParams&& ... pParams) {
  OpTy* op = m_pCG->addOperator<OpTy>(pParams...);
  for (auto& iname : pInputNames)
    op->addInput(*(m_pCG->getValue<Tensor>(iname)));
  this->set_default_attributs(*op);
  return op;
  }                                   

  template<typename VTy>
  std::vector<VTy> get_values(const std::vector<VTy>& pVec) {
  return pVec;
  }

template <typename TensorTy>
void create_weight_operator_with_values(const std::string &pName,
                                    const Tensor::Dimensions &pDims,
                                    const typename TensorTy::ValueList& values)
{
  Initializer *init = m_pCG->addOperator<Initializer>(pName);
  TensorTy *value = m_pCG->addValue<TensorTy>(pName);
  value->setDimensions(pDims);
  value->getValues() = values;
  init->setTensor(*value);
}

private:
  CbufAllocType getCbufAllocType(const NvDlaCubeInfo& xinfo, const NvDlaCubeInfo& winfo, Tensor::Dimension yDilation,
                                 unsigned& minNumNeededDataBanks);

  // code generate pass
  int submit_event(int task_id, int event_type);
  int submit_mem_alloc_address(int size, std::string& blob_name);
  void init_nvdla_memory();
  void task_submit();
  void nvdla_filegen();

  // optimize pass
  void propagate_const_with_diff_shape();

protected:
  NvDlaConstants m_nvdla_constants;
  onnc::Module* m_pModule;
  onnc::ComputeGraph* m_pCG;
};

#endif