/*!
 * Copyright (c) 2021 by Contributors
 * \file src/impl/base_ops.cc
 * \brief Implementation of some OpEnvs registered on base ops.
 */
#include "./schema/transform.h"
#include "./schema/ufunc.h"
#include "mnm/op.h"
#include "mnm/device_api.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;

void SizeImpl(const DLTensor* x, const Value& axis, Value& out) {
  if (axis.defined()) {
    const auto* v = axis.as<IntValueObj>();
    ICHECK(v != nullptr);
    DLTensor* out_tensor = out;
    ICHECK_EQ(out_tensor->device.device_type, kDLCPU);
    auto out_ptr = static_cast<int32_t*>(out_tensor->data);
    *out_ptr = int32_t(x->shape[v->value]);
  } else {
    Array<Value> outs = Downcast<TupleValue>(out)->fields;
    for (int i = 0; i < x->ndim; ++i) {
      DLTensor* out_tensor = outs[i];
      ICHECK_EQ(out_tensor->device.device_type, kDLCPU);
      auto out_ptr = static_cast<int32_t*>(out_tensor->data);
      *out_ptr = int32_t(x->shape[i]);
    }
  }
}

class SizeOpEnv : public mnm::op::OpEnv {
  std::string env_name_;

 public:
  explicit SizeOpEnv(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    static const std::string op_name = "mnm.op.size";
    static const auto op = ir::Op::Get(op_name);
    this->arg_indices = {
        fschema_index[op]("x"),
        fschema_index[op]("axis"),
    };
    env_name_ = TruncateName(GetUniqueName(op_name));
  }

  std::string name() const override {
    return env_name_;
  }

  void Execute(const CallValues& cv) override {
    const auto* args = cv->args.as<op::schema::SizeArgs>();
    ICHECK(args != nullptr);
    SizeImpl(args->x, args->axis, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    SizeImpl(inputs[0], inputs[1], output);
  }

  static OpEnv* make(const CallValues& cv) {
    return new SizeOpEnv(cv);
  }
};

MNM_OP_ENV_MAKER("mnm.op.size", SizeOpEnv::make);

void NumelImpl(const DLTensor* x, DLTensor* y) {
  ICHECK_EQ(y->device.device_type, kDLCPU);
  int32_t num_elems = 1;
  for (int i = 0; i < x->ndim; ++i) {
    num_elems *= x->shape[i];
  }
  auto y_ptr = static_cast<int32_t*>(y->data);
  *y_ptr = num_elems;
}

class NumelOpEnv : public mnm::op::OpEnv {
  std::string env_name_;

 public:
  explicit NumelOpEnv(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    static const std::string op_name = "mnm.op.numel";
    static const auto op = ir::Op::Get(op_name);
    this->arg_indices = {fschema_index[op]("x")};
    env_name_ = TruncateName(GetUniqueName(op_name));
  }

  std::string name() const override {
    return env_name_;
  }

  void Execute(const CallValues& cv) override {
    const auto* args = cv->args.as<op::schema::UnaryArgs>();
    ICHECK(args != nullptr);
    NumelImpl(args->x, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    NumelImpl(inputs[0], output);
  }

  static OpEnv* make(const CallValues& cv) {
    return new NumelOpEnv(cv);
  }
};

MNM_OP_ENV_MAKER("mnm.op.numel", NumelOpEnv::make);

void ShapeAsTensorImpl(const DLTensor* x, DLTensor* y) {
  ICHECK_EQ(y->device.device_type, kDLCPU);
  auto y_ptr = static_cast<int32_t*>(y->data);
  for (int i = 0; i < x->ndim; ++i) {
    *(y_ptr + i) = int32_t(x->shape[i]);
  }
}

class ShapeAsTensorOpEnv : public mnm::op::OpEnv {
  std::string env_name_;

 public:
  explicit ShapeAsTensorOpEnv(const CallValues& cv) {
    static auto fschema_index =
        ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    static const std::string op_name = "mnm.op.shape_as_tensor";
    static const auto op = ir::Op::Get(op_name);
    this->arg_indices = {fschema_index[op]("x")};
    env_name_ = TruncateName(GetUniqueName(op_name));
  }

  std::string name() const override {
    return env_name_;
  }

  void Execute(const CallValues& cv) override {
    const auto* args = cv->args.as<op::schema::UnaryArgs>();
    ICHECK(args != nullptr);
    ShapeAsTensorImpl(args->x, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, Value output) override {
    ShapeAsTensorImpl(inputs[0], output);
  }

  static OpEnv* make(const CallValues& cv) {
    return new ShapeAsTensorOpEnv(cv);
  }
};

MNM_OP_ENV_MAKER("mnm.op.shape_as_tensor", ShapeAsTensorOpEnv::make);

}  // namespace op
}  // namespace mnm