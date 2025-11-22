#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "RVV/RVVDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // Register RVV dialect
  registry.insert<mlir::rvv::RVVDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "RVV optimizer driver\n", registry));
}
