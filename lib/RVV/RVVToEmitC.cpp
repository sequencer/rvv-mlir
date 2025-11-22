#include "RVV/RVVAttributes.h"
#include "RVV/RVVDialect.h"
#include "RVV/RVVOps.h"
#include "RVV/RVVPasses.h"
#include "RVV/RVVTypes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::rvv;

namespace mlir {
namespace rvv {

#define GEN_PASS_DEF_RVVTOEMITC
#include "RVV/RVVPasses.h.inc"

namespace {

void populateRVVToEmitCConversionPatterns(RewritePatternSet &patterns) {
  // TODO: Add patterns here when operations are defined
  (void)patterns; // Suppress unused parameter warning
}

struct RVVToEmitCPass : public impl::RVVToEmitCBase<RVVToEmitCPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<RVVDialect>();

    RewritePatternSet patterns(&getContext());
    populateRVVToEmitCConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRVVToEmitCPass() {
  return std::make_unique<RVVToEmitCPass>();
}

} // namespace rvv
} // namespace mlir