#include "RVV/RVVDialect.h"
#include "RVV/RVVAttributes.h"
#include "RVV/RVVOps.h"
#include "RVV/RVVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rvv;

// Include the generated dialect definitions
#include "RVV/RVVDialect.cpp.inc"

void RVVDialect::initialize() {
  registerTypes();
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "RVV/RVVOps.cpp.inc"
      >();
}
