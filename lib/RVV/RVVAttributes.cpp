#include "RVV/RVVAttributes.h"
#include "RVV/RVVDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rvv;

// Include generated enum definitions
#include "RVV/RVVEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RVV/RVVAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Attribute Registration
//===----------------------------------------------------------------------===//

void RVVDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RVV/RVVAttributes.cpp.inc"
      >();
}
