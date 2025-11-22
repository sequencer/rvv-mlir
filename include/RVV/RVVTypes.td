#ifndef RVV_TYPES
#define RVV_TYPES

include "RVV/RVVDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

class RVV_Type<string name, string typeMnemonic> : TypeDef<RVV_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Define valid RVV element types

// Define valid RVV element types
def RVV_ElementType : AnyTypeOf<[
  I8, I16, I32, I64,
  F16, BF16, F32, F64
], "RVV element type (i8, i16, i32, i64, u8, u16, u32, u64, f16, bf16, f32, f64)">;

// Define valid RVV scalar types
// Note: Use mlir::rvv::getScalarElementWidth(Type) to get element width in bits
def RVV_ScalarType : AnyTypeOf<[
  I32, I64,
  F32, F64
], "RVV scalar type (i32, i64, f32, f64)">;

// RVV Vector Type - Explicitly encodes element type and LMUL (register grouping)
// Maps to vint*m*_t / vfloat*m*_t / vuint*m*_t types in C intrinsics
def RVV_VectorType : RVV_Type<"Vector", "vector"> {
  let summary = "RISC-V Vector Type with explicit LMUL";
  let description = [{
    A custom parameterized type that explicitly encodes both the element type 
    and the register grouping (LMUL). This resolves the ambiguity of generic 
    scalable vectors by making LMUL an explicit compile-time parameter.
    
    Syntax: `!rvv.vector< element_type, lmul >`
    
    Examples:
    - `!rvv.vector<i32, m1>` maps to `vint32m1_t`
    - `!rvv.vector<f64, m4>` maps to `vfloat64m4_t`
    - `!rvv.vector<i8, mf8>` maps to `vint8mf8_t`
    
    The element type must be a standard MLIR integer or floating-point type
    (i8, i16, i32, i64, f16, bf16, f32, f64).
    
    The LMUL parameter controls register grouping and directly affects VLMAX.
  }];
  let parameters = (ins RVV_ElementType:$elementType, "::mlir::rvv::LMUL":$lmul);
  
  // Custom printer and parser to match intrinsic type names (e.g., vint32m1_t)
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    /// Get the element width in bits (e.g., 32 for i32, 16 for f16)
    unsigned getElementWidth() const;
    
    /// Get the full intrinsic type name (e.g., "vint32m1_t", "vfloat64m4_t")
    std::string getVectorTypeFullString() const;
    
    /// Get the vector type string suffix for intrinsic names (e.g., "_i32m1", "_f64m4")
    std::string getVectorTypeString() const;
  }];
}

// RVV Mask Type - Parameterized by layout ratio (n = SEW/LMUL)
// Maps to vbool*_t types in C intrinsics
def RVV_MaskType : RVV_Type<"Mask", "mask"> {
  let summary = "RISC-V Vector Mask Type";
  let description = [{
    A custom type for RVV masks, parameterized by the layout ratio 
    (n = SEW/LMUL). This ensures correct mask register usage for 
    different data vector configurations.
    
    Syntax: `!rvv.mask< layout >`
    
    Examples:
    - `!rvv.mask<n1>` maps to `vbool1_t` (one bit per element, LMUL=SEW)
    - `!rvv.mask<n8>` maps to `vbool8_t` (one bit per 8 elements)
    - `!rvv.mask<n64>` maps to `vbool64_t` (one bit per 64 elements)
    
    The layout parameter must be compatible with the data vector it masks.
    For a vector `!rvv.vector<T, M>`, the required mask layout is 
    n = sizeof(T) / value(M).
  }];
  let parameters = (ins "::mlir::rvv::MaskLayout":$layout);
  let assemblyFormat = "`<` $layout `>`";
  let genVerifyDecl = 1;
  
  let extraClassDeclaration = [{
    /// Get the full intrinsic type name (e.g., "vbool1_t", "vbool8_t", "vbool64_t")
    std::string getMaskTypeFullString() const;
    
    /// Get the mask type string suffix for intrinsic names (e.g., "_b1", "_b8", "_b64")
    std::string getMaskTypeString() const;
  }];
}

// Union type for operations that can work with both vectors and masks
def RVV_VectorOrMaskType : AnyTypeOf<[RVV_VectorType, RVV_MaskType], 
  "RVV vector or mask type">;

#endif // RVV_TYPES
