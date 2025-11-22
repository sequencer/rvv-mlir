#ifndef RVV_ATTRIBUTES
#define RVV_ATTRIBUTES

include "RVV/RVVDialect.td"
include "mlir/IR/EnumAttr.td"

class RVV_Attr<string name, string attrMnemonic> : AttrDef<RVV_Dialect, name> {
  let mnemonic = attrMnemonic;
}

// LMUL (Length Multiplier) - Register Grouping
def RVV_LMUL_MF8 : I32EnumAttrCase<"MF8", 0, "mf8">;
def RVV_LMUL_MF4 : I32EnumAttrCase<"MF4", 1, "mf4">;
def RVV_LMUL_MF2 : I32EnumAttrCase<"MF2", 2, "mf2">;
def RVV_LMUL_M1  : I32EnumAttrCase<"M1",  3, "m1">;
def RVV_LMUL_M2  : I32EnumAttrCase<"M2",  4, "m2">;
def RVV_LMUL_M4  : I32EnumAttrCase<"M4",  5, "m4">;
def RVV_LMUL_M8  : I32EnumAttrCase<"M8",  6, "m8">;

def RVV_LMULAttr : I32EnumAttr<"LMUL", "RISC-V Vector Length Multiplier",
  [RVV_LMUL_MF8, RVV_LMUL_MF4, RVV_LMUL_MF2, RVV_LMUL_M1, 
   RVV_LMUL_M2,  RVV_LMUL_M4,  RVV_LMUL_M8]> {
  let cppNamespace = "::mlir::rvv";
}

// Mask Layout (n = SEW/LMUL ratio)
def RVV_MaskLayout_N1  : I32EnumAttrCase<"N1",  0, "n1">;
def RVV_MaskLayout_N2  : I32EnumAttrCase<"N2",  1, "n2">;
def RVV_MaskLayout_N4  : I32EnumAttrCase<"N4",  2, "n4">;
def RVV_MaskLayout_N8  : I32EnumAttrCase<"N8",  3, "n8">;
def RVV_MaskLayout_N16 : I32EnumAttrCase<"N16", 4, "n16">;
def RVV_MaskLayout_N32 : I32EnumAttrCase<"N32", 5, "n32">;
def RVV_MaskLayout_N64 : I32EnumAttrCase<"N64", 6, "n64">;

def RVV_MaskLayoutAttr : I32EnumAttr<"MaskLayout", "RISC-V Vector Mask Layout",
  [RVV_MaskLayout_N1, RVV_MaskLayout_N2, RVV_MaskLayout_N4, RVV_MaskLayout_N8,
   RVV_MaskLayout_N16, RVV_MaskLayout_N32, RVV_MaskLayout_N64]> {
  let cppNamespace = "::mlir::rvv";
}

// Tail Policy (Agnostic/Undisturbed)
def RVV_TailPolicy_Agnostic    : I32EnumAttrCase<"Agnostic", 0, "agnostic">;
def RVV_TailPolicy_Undisturbed : I32EnumAttrCase<"Undisturbed", 1, "undisturbed">;

def RVV_TailPolicyAttr : I32EnumAttr<"TailPolicy", "RISC-V Vector Tail Policy",
  [RVV_TailPolicy_Agnostic, RVV_TailPolicy_Undisturbed]> {
  let cppNamespace = "::mlir::rvv";
}

// Mask Policy (Agnostic/Undisturbed)
def RVV_MaskPolicy_Agnostic    : I32EnumAttrCase<"Agnostic", 0, "agnostic">;
def RVV_MaskPolicy_Undisturbed : I32EnumAttrCase<"Undisturbed", 1, "undisturbed">;

def RVV_MaskPolicyAttr : I32EnumAttr<"MaskPolicy", "RISC-V Vector Mask Policy",
  [RVV_MaskPolicy_Agnostic, RVV_MaskPolicy_Undisturbed]> {
  let cppNamespace = "::mlir::rvv";
}

// Floating-Point Rounding Mode (FRM)
def RVV_FRM_RNE : I32EnumAttrCase<"RNE", 0, "rne">; // Round to Nearest, ties to Even
def RVV_FRM_RTZ : I32EnumAttrCase<"RTZ", 1, "rtz">; // Round towards Zero
def RVV_FRM_RDN : I32EnumAttrCase<"RDN", 2, "rdn">; // Round Down (towards -infinity)
def RVV_FRM_RUP : I32EnumAttrCase<"RUP", 3, "rup">; // Round Up (towards +infinity)
def RVV_FRM_RMM : I32EnumAttrCase<"RMM", 4, "rmm">; // Round to Nearest, ties to Max Magnitude
def RVV_FRM_DYN : I32EnumAttrCase<"DYN", 7, "dyn">; // Dynamic (use frm CSR)

def RVV_FRMAttr : I32EnumAttr<"FRM", "RISC-V Floating-Point Rounding Mode",
  [RVV_FRM_RNE, RVV_FRM_RTZ, RVV_FRM_RDN, RVV_FRM_RUP, RVV_FRM_RMM, RVV_FRM_DYN]> {
  let cppNamespace = "::mlir::rvv";
}

#endif // RVV_ATTRIBUTES
