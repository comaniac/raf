/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dist_params.cc
 * \brief Transform the IR for distributed input parameters (ZeRO-3).
 */
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "raf/value.h"
#include "./let_list.h"

namespace raf {
namespace pass {
namespace dist_params {

/*! \brief 1) Modify the main function arguments to match the type. 2) Insert AllGather. */
class DistParamsMutator : public ExprMutator {
 public:
  DistParamsMutator(const ir::Map<tvm::String, tvm::Type> partitioned_types)
      : partitioned_types_(partitioned_types) {
    scopes_.emplace_back(new LetList);
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      curr_let_ = node->var;
      auto new_value = VisitExpr(node->value);
      scope->Push(curr_let_, new_value);
      let_vars_.emplace(curr_let_, new_value);

      body = node->body;
      node = body.as<LetNode>();
    } while (node);

    auto new_body = VisitExpr(body);
    auto ret = scopes_.back()->Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) {
    Array<Expr> args;
    for (auto arg : node->args) {
      auto new_arg = arg;
      if (auto var_node = arg.as<VarNode>()) {
        auto var = GetRef<Var>(var_node);
        if (map_part_params_.count(var) > 0) {
          new_arg = GatherVar(map_part_params_[var].first, map_part_params_[var].second);
        }
      }
      args.push_back(new_arg);
    }
    return Call(node->op, args, node->attrs, node->type_args);
  }

  Expr VisitExpr_(const FunctionNode* node) {
    Array<Var> params;
    for (const auto& param : node->params) {
      auto name = param->name_hint();
      auto new_param = param;
      if (partitioned_types_.count(name) > 0) {
        new_param = Var(name, partitioned_types_[name], param->span);
        map_part_params_[param] = {new_param, param->type_annotation};
      }
      params.push_back(new_param);
    }
    Expr body = VisitExpr(node->body);
    Function func(params, body, node->ret_type, {}, node->attrs);
    return func;
  }

 private:
  Var GatherVar(const Var& var, const Type& orig_type) {
    auto part_ttype = var->type_annotation.as<TensorTypeNode>();
    CHECK(part_ttype != nullptr);
    auto orig_ttype = orig_type.as<TensorTypeNode>();
    CHECK(orig_ttype != nullptr);

    static const Op& allgather_op = Op::Get("raf.op._allgather");
    static const Op& strided_slice_op = Op::Get("raf.op.strided_slice");
    auto scope = scopes_.back().get();
    auto new_var =
        scope->Push(Call(allgather_op, {var, MakeConstant(value::ScalarValue::make(0))}, {}));
    auto part_dim0 = part_ttype->shape[0].as<IntImmNode>()->value;
    auto orig_dims = orig_ttype->shape[0].as<IntImmNode>()->value;

    // Slice to remove the zero-padding if needed.
    if (orig_dims % part_dim0 != 0) {
      new_var = scope->Push(Call(strided_slice_op,
                                 {new_var,
                                  /*begin=*/{MakeConstant(value::ScalarValue::make(0))},
                                  /*end=*/{MakeConstant(value::ScalarValue::make(orig_dims))},
                                  /*stide=*/{MakeConstant(value::ScalarValue::make(1))}},
                                 {}));
    }
    return new_var;
  }

  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The current processing let-binding var. */
  Var curr_let_;
  /*! \brief The mapping from let bound var to its expr. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> let_vars_;
  /*! \brief Mapping from parameter name to the partitioned type. */
  ir::Map<tvm::String, tvm::Type> partitioned_types_;
  /*! \brief Mapping from original parameter to the partitioned one with the original type. */
  std::unordered_map<Var, std::pair<Var, Type>, ObjectPtrHash, ObjectPtrEqual> map_part_params_;
};
}  // namespace dist_params

Pass DistParams(ir::Map<tvm::String, tvm::Type> partitioned_types) {
  TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m, PassContext pc) {
    // Copy the module to avoid any in-place module update
    ir::IRModule mod = ir::IRModule(m->functions);

    auto func = Downcast<Function>(mod->Lookup("main"));
    auto updated_func =
        Downcast<Function>(dist_params::DistParamsMutator(partitioned_types).Mutate(func));
    mod->Update(mod->GetGlobalVar("main"), updated_func);
    return mod;
  };
  return CreateModulePass(pass_func, 0, "DistParams", {"InferType"});
}

RAF_REGISTER_GLOBAL("raf.pass_.DistParams").set_body_typed(DistParams);

}  // namespace pass
}  // namespace raf
