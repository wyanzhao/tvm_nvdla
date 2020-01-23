#ifndef NVDLA_LIB_H
#define NVDLA_LIB_H

#include <NvDlaOp.h>
#include <NvDlaDefine.h>

// OPT PASS
#include <opt_pass/replace_gemm_by_conv.h>
#include <opt_pass/propagate_const_with_diff_shape.h>
#include <opt_pass/expand_batchnormalization.h>
#include <opt_pass/divide_global_api_into_aps.h>
#include <opt_pass/nvdla_collect_reshape_info_pass.h>

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

  template<typename VTy>
  std::vector<VTy> get_values(const std::vector<VTy>& pVec) {
    return pVec;
  }

  template<typename OpTy, typename ... NodeCtorParams>
  OpTy* create_compute_operator(const StringList& pInputNames,
                                   NodeCtorParams&& ... pParams) {
    OpTy* op = m_pCG->addOperator<OpTy>(pParams...);
    for (auto& iname : pInputNames)
      op->addInput(*(m_pCG->getValue<Tensor>(iname)));
    this->set_default_attributs(*op);
    return op;
  }                                   

  template<typename TensorTy>
  void create_weight_operator(const std::string& pName,
                                   const Tensor::Dimensions& pDims){
    Initializer* init = m_pCG->addOperator<Initializer>(pName);
    Tensor* value = this->create_compute_tensor<TensorTy>(pName, pDims);
    init->setTensor(*value);
  };
                                                         
  Tensor* create_float_compute_tensor(const StringRef& pName,
                         const Tensor::Dimensions& pDims);

  void
  create_float_weight_tensor_from_numpy(const std::string& pName,
                          const Tensor::Dimensions& pDims, void* data);

  void
  create_float_weight_tensor_from_file(const std::string& pName,
                          const Tensor::Dimensions& pDims, const std::string& weight_path);

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
protected:
  NvDlaConstants m_nvdla_constants;
  onnc::Module* m_pModule;
  onnc::ComputeGraph* m_pCG;
};

#endif