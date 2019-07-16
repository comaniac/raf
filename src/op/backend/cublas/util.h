#pragma once

#include <cublas_v2.h>

#include <mnm/base.h>
#include <mnm/enum_base.h>
#include <mnm/rly.h>

#include "../../../common/cuda.h"

#define CUBLAS_CALL(func)                                                        \
  do {                                                                           \
    cublasStatus_t e = (func);                                                   \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS) << "cublas: " << cublasGetErrorString(e); \
  } while (false)

namespace mnm {
namespace op {
namespace backend {
namespace cublas {

inline const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      LOG(FATAL) << "ValueError: Unknown error!\n";
      throw;
  }
  LOG(FATAL) << "ValueError: Unknown error!\n";
  throw;
}

class CUBlasThreadEntry {
 public:
  CUBlasThreadEntry();
  static CUBlasThreadEntry* ThreadLocal();

 public:
  cublasHandle_t handle{nullptr};
};

}  // namespace cublas
}  // namespace backend
}  // namespace op
}  // namespace mnm