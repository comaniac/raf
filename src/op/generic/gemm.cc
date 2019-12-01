/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/generic/gemm.cc
 * \brief Declaration of genmm-related operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/gemm.h"

namespace mnm {
namespace op {
namespace generic {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.matmul", [](const CallValues& call) {
  /*
   * This is essentially transposed matrix multiplication.
   * [n, m] * [m, k] => [n, k]
   */
  // TODO(@junrushao1994): sanity check
  const auto* args = call->args.as<MatmulArgs>();
  CHECK(args != nullptr);
  const DLTensor* a = args->a;
  const DLTensor* b = args->b;
  // a is of shape [n1, m1]
  // b is of shape [n2, m2]
  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(b->ndim, 2);
  int64_t n1 = a->shape[0];
  int64_t m1 = a->shape[1];
  int64_t n2 = b->shape[0];
  int64_t m2 = b->shape[1];
  if (args->transpose_a) {
    std::swap(n1, m1);
  }
  if (args->transpose_b) {
    std::swap(n2, m2);
  }
  CHECK_EQ(m1, n2);
  call->out = TensorValue::Assemble(/*ctx=*/a->ctx, /*dtype=*/a->dtype, /*shape=*/{n1, m2});
  call->ctx = a->ctx;
});

}  // namespace generic
}  // namespace op
}  // namespace mnm
