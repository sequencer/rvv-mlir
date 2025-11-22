#include "RVV/RVVOps.h"
#include "RVV/RVVAttributes.h"
#include "RVV/RVVDialect.h"
#include "RVV/RVVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::rvv;

//===----------------------------------------------------------------------===//
// SetvlOp
//===----------------------------------------------------------------------===//

std::string SetvlOp::getIntrinsicName() {
  std::string name = "vsetvl_e";

  // Append SEW (Selected Element Width)
  name += std::to_string(getSew());

  // Append LMUL (use TableGen-generated stringifyLMUL)
  name += stringifyLMUL(getLmul()).str();

  return name;
}

//===----------------------------------------------------------------------===//
// SetvlMaxOp
//===----------------------------------------------------------------------===//

std::string SetvlMaxOp::getIntrinsicName() {
  std::string name = "vsetvlmax_e";

  // Append SEW (Selected Element Width)
  name += std::to_string(getSew());

  // Append LMUL (use TableGen-generated stringifyLMUL)
  name += stringifyLMUL(getLmul()).str();

  return name;
}

//===----------------------------------------------------------------------===//
// VSExtOp
//===----------------------------------------------------------------------===//

LogicalResult VSExtOp::verify() {
  int32_t factor = getFactor();
  if (factor != 2 && factor != 4 && factor != 8) {
    return emitOpError("factor attribute must be 2, 4, or 8, but got ")
           << factor;
  }

  // Check that the result element width matches the input element width times
  // the factor
  auto vs1Type = llvm::dyn_cast<VectorType>(getVs1().getType());
  auto vdType = llvm::dyn_cast<VectorType>(getVd().getType());

  if (vs1Type && vdType) {
    unsigned vs1Width = vs1Type.getElementWidth();
    unsigned vdWidth = vdType.getElementWidth();
    unsigned expectedWidth = vs1Width * factor;

    if (vdWidth != expectedWidth) {
      return emitOpError("result element width must be input element width "
                         "times factor")
             << ": input width=" << vs1Width << ", factor=" << factor
             << ", expected output width=" << expectedWidth
             << ", actual output width=" << vdWidth;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VZExtOp
//===----------------------------------------------------------------------===//

LogicalResult VZExtOp::verify() {
  int32_t factor = getFactor();
  if (factor != 2 && factor != 4 && factor != 8) {
    return emitOpError("factor attribute must be 2, 4, or 8, but got ")
           << factor;
  }

  // Check that the result element width matches the input element width times
  // the factor
  auto vs1Type = llvm::dyn_cast<VectorType>(getVs1().getType());
  auto vdType = llvm::dyn_cast<VectorType>(getVd().getType());

  if (vs1Type && vdType) {
    unsigned vs1Width = vs1Type.getElementWidth();
    unsigned vdWidth = vdType.getElementWidth();
    unsigned expectedWidth = vs1Width * factor;

    if (vdWidth != expectedWidth) {
      return emitOpError("result element width must be input element width "
                         "times factor")
             << ": input width=" << vs1Width << ", factor=" << factor
             << ", expected output width=" << expectedWidth
             << ", actual output width=" << vdWidth;
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "RVV/RVVOps.cpp.inc"
