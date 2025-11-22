#ifndef RVV_RVVPASSES_H
#define RVV_RVVPASSES_H

#include "RVV/RVVDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rvv {

std::unique_ptr<Pass> createRVVToEmitCPass();

#define GEN_PASS_REGISTRATION
#include "RVV/RVVPasses.h.inc"

} // namespace rvv
} // namespace mlir

#endif // RVV_RVVPASSES_H
