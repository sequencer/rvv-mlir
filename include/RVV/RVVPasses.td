#ifndef RVV_PASSES
#define RVV_PASSES

include "mlir/Pass/PassBase.td"

def RVVToEmitC : Pass<"convert-rvv-to-emitc"> {
  let summary = "Convert RVV dialect to EmitC dialect";
  let description = [{
    This pass converts RVV operations to EmitC operations.
  }];
  let constructor = "mlir::rvv::createRVVToEmitCPass()";
  let dependentDialects = ["emitc::EmitCDialect"];
}

#endif // RVV_PASSES
