#include "RVV/RVVTypes.h"
#include "RVV/RVVDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rvv;

#define GET_TYPEDEF_CLASSES
#include "RVV/RVVTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

unsigned mlir::rvv::VectorType::getElementWidth() const {
  Type elemType = getElementType();

  // Both IntegerType and FloatType have getWidth() method
  if (auto intType = llvm::dyn_cast<IntegerType>(elemType)) {
    return intType.getWidth();
  } else if (auto floatType = llvm::dyn_cast<FloatType>(elemType)) {
    return floatType.getWidth();
  }

  return 0;
}

std::string mlir::rvv::VectorType::getVectorTypeFullString() const {
  std::string result;
  Type elemType = getElementType();
  LMUL lmul = getLmul();

  // Determine signedness and base type
  if (auto intType = llvm::dyn_cast<IntegerType>(elemType)) {
    result = "vint"; // RVV intrinsics use signed by default
    result += std::to_string(intType.getWidth());
  } else if (auto floatType = llvm::dyn_cast<FloatType>(elemType)) {
    result = "vfloat";
    if (floatType.isF16()) {
      result += "16";
    } else if (floatType.isBF16()) {
      result = "vbfloat16";
    } else if (floatType.isF32()) {
      result += "32";
    } else if (floatType.isF64()) {
      result += "64";
    }
  }

  // Append LMUL suffix
  result += stringifyLMUL(lmul).str();

  result += "_t";
  return result;
}

std::string mlir::rvv::VectorType::getVectorTypeString() const {
  // Get the full string and transform it to the required format
  std::string fullString = getVectorTypeFullString();

  // Transform "vint32m1_t" -> "_i32m1", "vfloat64m4_t" -> "_f64m4", etc.
  std::string result = "_";

  // Remove 'v' prefix and '_t' suffix
  std::string core = fullString.substr(1, fullString.size() - 3);

  if (core.starts_with("int")) {
    result += "i" + core.substr(3); // "int32m1" -> "i32m1"
  } else if (core.starts_with("bfloat")) {
    result += "bf" + core.substr(6); // "bfloat16m1" -> "bf16m1"
  } else if (core.starts_with("float")) {
    result += "f" + core.substr(5); // "float64m4" -> "f64m4"
  }

  return result;
}

void mlir::rvv::VectorType::print(AsmPrinter &printer) const {
  printer << "<" << getVectorTypeFullString() << ">";
}

Type mlir::rvv::VectorType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  // Parse the type name as a bare identifier
  std::string typeName;
  if (parser.parseKeywordOrString(&typeName))
    return Type();

  if (parser.parseGreater())
    return Type();

  // Parse intrinsic type name (e.g., "vint32m1_t")
  if (!typeName.starts_with("v") || !typeName.ends_with("_t")) {
    parser.emitError(parser.getNameLoc(), "invalid RVV type name: ")
        << typeName;
    return Type();
  }

  // Remove 'v' prefix and '_t' suffix
  typeName = typeName.substr(1, typeName.size() - 3);

  // Determine if it's int or float
  bool isFloat = false;
  bool isBFloat = false;
  if (typeName.starts_with("int")) {
    typeName = typeName.substr(3);
  } else if (typeName.starts_with("bfloat")) {
    isFloat = true;
    isBFloat = true;
    typeName = typeName.substr(6);
  } else if (typeName.starts_with("float")) {
    isFloat = true;
    typeName = typeName.substr(5);
  } else {
    parser.emitError(parser.getNameLoc(), "invalid RVV type prefix");
    return Type();
  }

  // Parse bit width
  unsigned bitWidth = 0;
  size_t pos = 0;
  while (pos < typeName.size() && std::isdigit(typeName[pos])) {
    bitWidth = bitWidth * 10 + (typeName[pos] - '0');
    pos++;
  }

  if (bitWidth == 0) {
    parser.emitError(parser.getNameLoc(), "invalid bit width");
    return Type();
  }

  // Parse LMUL
  std::string lmulStr = typeName.substr(pos);
  auto lmulOpt = symbolizeLMUL(lmulStr);
  if (!lmulOpt.has_value()) {
    parser.emitError(parser.getNameLoc(), "invalid LMUL: ") << lmulStr;
    return Type();
  }
  LMUL lmul = lmulOpt.value();

  // Construct element type
  Type elemType;
  if (isFloat) {
    if (isBFloat && bitWidth == 16) {
      elemType = BFloat16Type::get(parser.getContext());
    } else if (bitWidth == 16) {
      elemType = Float16Type::get(parser.getContext());
    } else if (bitWidth == 32) {
      elemType = Float32Type::get(parser.getContext());
    } else if (bitWidth == 64) {
      elemType = Float64Type::get(parser.getContext());
    } else {
      parser.emitError(parser.getNameLoc(), "invalid float bit width: ")
          << bitWidth;
      return Type();
    }
  } else {
    elemType = IntegerType::get(parser.getContext(), bitWidth);
  }

  return mlir::rvv::VectorType::get(parser.getContext(), elemType, lmul);
}

//===----------------------------------------------------------------------===//
// MaskType
//===----------------------------------------------------------------------===//

LogicalResult
MaskType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                 MaskLayout layout) {
  // Mask layout is always valid since it's an enum
  return success();
}

std::string mlir::rvv::MaskType::getMaskTypeFullString() const {
  MaskLayout layout = getLayout();

  // Convert MaskLayout enum to string (e.g., N1 -> "vbool1_t", N8 ->
  // "vbool8_t") stringifyMaskLayout returns "N1", "N8", etc., so we need to
  // extract the number
  std::string layoutStr = stringifyMaskLayout(layout).str();

  // Remove 'N' prefix from layout string (e.g., "N1" -> "1", "N64" -> "64")
  std::string result = "vbool" + layoutStr.substr(1) + "_t";

  return result;
}

std::string mlir::rvv::MaskType::getMaskTypeString() const {
  // Get the full string and transform it to the required format
  std::string fullString = getMaskTypeFullString();

  // Transform "vbool1_t" -> "_b1", "vbool8_t" -> "_b8", etc.
  // Remove 'vbool' prefix and '_t' suffix
  std::string core = fullString.substr(5, fullString.size() - 7);

  return "_b" + core;
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

unsigned mlir::rvv::getScalarElementWidth(Type scalarType) {
  // Both IntegerType and FloatType have getWidth() method
  if (auto intType = llvm::dyn_cast<IntegerType>(scalarType)) {
    return intType.getWidth();
  } else if (auto floatType = llvm::dyn_cast<FloatType>(scalarType)) {
    return floatType.getWidth();
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Dialect Type Registration
//===----------------------------------------------------------------------===//

void RVVDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RVV/RVVTypes.cpp.inc"
      >();
}
