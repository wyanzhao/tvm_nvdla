/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <sstream>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief An example codegen that is only used for quick prototyping and testing
 * purpose. Only several binary options are covered. Users
 * may need to extend them to cover more operators.
 */
class CodegenNvDla : public ExprVisitor, public CodegenCBase {
 public:
  explicit CodegenNvDla(const std::string& id) { this->ext_func_id_ = id; }

  void VisitExpr_(const VarNode* node) {
    ext_func_args_.push_back(node->name_hint());
    out_.clear();
    out_.push_back({node->name_hint(), 0});
  }

  void VisitExpr_(const CallNode* call) final {
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;
    // Args: ID
    std::vector<std::string> args;

    // Get the arguments for various DNNL kernels.
    if (IsOp(call, "nn.conv2d")) {
      decl_stream << "NvDlaConv2D";
      args = Conv2d(call);
    } else if (IsOp(call, "add")) {
      decl_stream << "NvDlaAdd";
      args = Add(call);
    } else if (IsOp(call, "nn.relu")) {
      decl_stream << "NvDlaRelu";
      args = Relu(call);
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    }

    // Make function call with input buffers when visiting arguments
    bool first = true;
    decl_stream << "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]);
      for (auto out : out_) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        if(IsOp(call, "nn.conv2d") && i == 1)
        {
          decl_stream <<out.first;
        } else
        {
          decl_stream << "\"" <<out.first<< "\"";  
        }
      }
    }

    // Analyze the output buffer
    auto type_node = call->checked_type().as<TensorTypeNode>();
    CHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
        << "Only support single output tensor with float type";
    std::string out = "buf_" + std::to_string(buf_idx_++);
    //auto out_shape = GetShape(call->checked_type());
    auto out_size = 1;
    //for (size_t i = 0; i < out_shape.size(); ++i) {
    //  out_size *= out_shape[i];
    //}
    this->PrintIndents();
    //buf_stream << "float* " << out << " = (float*)std::malloc(4 * " << out_size << ");";
    //buf_decl_.push_back(buf_stream.str());
    decl_stream << ", " << "\""<< out <<"\"" ;

    // Attach attribute arguments
    for (size_t i = 0; i < args.size(); ++i) {
      decl_stream << ", " << args[i];
    }
    decl_stream << ");";
    ext_func_body.push_back(decl_stream.str());

    // Update output buffer
    out_.clear();
    out_.push_back({out, out_size});
  }

    std::string JitImpl(std::string ext_func_id, std::vector<std::string> args,
                      std::vector<std::string> buf_decl, std::vector<std::string> body,
                      std::vector<std::pair<std::string, int>> out) {
    // Create the signature. For example, it could be:
    // extern "C" void dnnl_0_(float* input0, float* input1, float* out, int M, int N) {}
    code_stream_ << "extern \"C\" void " << ext_func_id << "_(";

    for (const auto& arg : args) {
      code_stream_ << "float* " << arg << ", ";
    }
    code_stream_ << "float* out) {\n";
    this->EnterScope();
    
    // Init NvDla
    this->PrintIndents();
    code_stream_ << "NvDlaInit(\"NvDla\");" << "\n";

    this->PrintIndents();
    code_stream_ << "AddInputOpByName("<< "\""<< args.front() << "\"" << ");\n";

    // Function body
    for (auto decl : buf_decl) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : body) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }


    // Copy output
    CHECK_EQ(out.size(), 1U) << "Internal error: only single output is support.";
    this->PrintIndents();
    //code_stream_ << "std::memcpy(out, " << out[0].first << ", 4 * " << out[0].second << ");\n";
    code_stream_ << "AddOutputOp(" << "\"" << out[0].first << "\"" << ");\n";

    this->PrintIndents();
    code_stream_ << "NvDlaCompile();" << "\n";

    // Free buffers
    //for (size_t i = 0; i < buf_decl.size(); i++) {
    //  this->PrintIndents();
    //  code_stream_ << "std::free(buf_" << i << ");\n";
   // }

    this->ExitScope();
    code_stream_ << "}\n";

    // Create the wrapper to call the ext_func
    this->GenerateBackendCFunc(ext_func_id, args.size() + 1 /* output */);
    return code_stream_.str();
  }

  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  std::string JIT() {
    // Write function macros

    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return this->JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
  }

 private:
  /*! \brief The function id that represents a C source function. */
  std::string ext_func_id_ = "";
  /*! \brief The index of a wrapped C function. */
  int func_idx = 0;
  /*! \brief The index of allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  std::vector<std::string> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  std::vector<std::string> ext_func_body;
  /*! \brief The declaration statements of a C compiler compatible function. */
  std::vector<std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The name and index pairs for output. */
  std::vector<std::pair<std::string, int>> out_;
  private:

    std::vector<std::string> Add(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    return args;
  }

    std::vector<std::string> Relu(const CallNode* call) {
    std::vector<std::string> args;
    auto ishape = GetShape(call->args[0]->checked_type());

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    return args;
  }

    std::vector<std::string> Conv2d(const CallNode* call) {
    std::vector<std::string> args;
    const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
    CHECK(conv2d_attr);

    auto ishape = GetShape(call->args[0]->checked_type());
    CHECK(ishape.size() == 4);
    auto wshape = GetShape(call->args[1]->checked_type());
    CHECK(wshape.size() == 4);

    // Args: N, C, H, W
    for (auto s : ishape) {
      args.push_back(std::to_string(s));
    }

    // Args: O, G, Ph, Pw, Kh, Kw, Sh, Sw
    args.push_back(std::to_string(wshape[0]));
    args.push_back(std::to_string(conv2d_attr->groups));
    args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
    args.push_back(std::to_string(wshape[2]));
    args.push_back(std::to_string(wshape[3]));
    args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->dilation[0].as<IntImmNode>()->value));
    args.push_back(std::to_string(conv2d_attr->dilation[1].as<IntImmNode>()->value));

    return args;
  }
};

class NvDlaCodegen : public CSourceModuleCodegenBase {
 public:
  void GenCFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodegenNvDla builder(sid);
    builder.VisitExpr(func->body);
    code_stream_ << builder.JIT();
  }


  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    code_stream_ << "#include <nvdla/nvdla.h>\n";
    code_stream_ << "using namespace tvm::runtime::contrib;\n";
    code_stream_ << "\n";


    if (ref->IsInstance<FunctionNode>()) {
      GenCFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenCFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("module.csource_module_create");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc");
  }

 private:
  std::ostringstream code_stream_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 *
 * The external codegen tool should have been registered similiarly to LLVM,
 * CUDA, etc, under TVM, so the generated code could be packed in a runtime
 * module. This module simplifies code serialization and invocation.
 */
runtime::Module NvDlaCompiler(const ObjectRef& ref) {
  NvDlaCodegen csource;
  return csource.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.nvdla").set_body_typed(NvDlaCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
