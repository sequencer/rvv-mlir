#ifndef RVV_FLOAT_OPS
#define RVV_FLOAT_OPS

def VFAddVVOp : RVV_Op<"vfadd.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point addition";
  let description = [{
    Performs element-wise floating-point addition.
    Maps to vfadd.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfadd.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfadd_vv_<type><policy>
      // Example: __riscv_vfadd_vv_f32m1_tu, __riscv_vfadd_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFAddVFOp : RVV_Op<"vfadd.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point addition";
  let description = [{
    Performs element-wise floating-point addition with scalar.
    Maps to vfadd.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfadd.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfadd_vf_<type><policy>
      // Example: __riscv_vfadd_vf_f32m1_tu, __riscv_vfadd_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfadd_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSubVVOp : RVV_Op<"vfsub.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point subtraction";
  let description = [{
    Performs element-wise floating-point subtraction: vd[i] = vs2[i] - vs1[i].
    Maps to vfsub.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsub.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsub_vv_<type><policy>
      // Example: __riscv_vfsub_vv_f32m1_tu, __riscv_vfsub_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSubVFOp : RVV_Op<"vfsub.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point subtraction";
  let description = [{
    Performs element-wise floating-point subtraction with scalar: vd[i] = vs2[i] - rs1.
    Maps to vfsub.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsub.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsub_vf_<type><policy>
      // Example: __riscv_vfsub_vf_f32m1_tu, __riscv_vfsub_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsub_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRSubVFOp : RVV_Op<"vfrsub.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point reverse subtraction";
  let description = [{
    Performs element-wise floating-point reverse subtraction: vd[i] = rs1 - vs2[i].
    This is the reverse of vfsub.vf, where the scalar is the minuend and vector elements are subtrahends.
    Maps to vfrsub.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfrsub.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfrsub_vf_<type><policy>
      // Example: __riscv_vfrsub_vf_f32m1_tu, __riscv_vfrsub_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfrsub_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMulVVOp : RVV_Op<"vfmul.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point multiplication";
  let description = [{
    Performs element-wise floating-point multiplication: vd[i] = vs2[i] * vs1[i].
    Maps to vfmul.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfmul.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmul_vv_<type><policy>
      // Example: __riscv_vfmul_vv_f32m1_tu, __riscv_vfmul_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmul_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMulVFOp : RVV_Op<"vfmul.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point multiplication";
  let description = [{
    Performs element-wise floating-point multiplication with scalar: vd[i] = vs2[i] * rs1.
    Maps to vfmul.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfmul.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmul_vf_<type><policy>
      // Example: __riscv_vfmul_vf_f32m1_tu, __riscv_vfmul_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmul_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFDivVVOp : RVV_Op<"vfdiv.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point division";
  let description = [{
    Performs element-wise floating-point division: vd[i] = vs2[i] / vs1[i].
    Maps to vfdiv.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfdiv.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfdiv_vv_<type><policy>
      // Example: __riscv_vfdiv_vv_f32m1_tu, __riscv_vfdiv_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfdiv_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFDivVFOp : RVV_Op<"vfdiv.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point division";
  let description = [{
    Performs element-wise floating-point division with scalar: vd[i] = vs2[i] / rs1.
    Maps to vfdiv.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfdiv.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfdiv_vf_<type><policy>
      // Example: __riscv_vfdiv_vf_f32m1_tu, __riscv_vfdiv_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfdiv_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRDivVFOp : RVV_Op<"vfrdiv.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point reverse division";
  let description = [{
    Performs element-wise floating-point reverse division: vd[i] = rs1 / vs2[i].
    This is the reverse of vfdiv.vf, where the scalar is the dividend and vector elements are divisors.
    Maps to vfrdiv.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfrdiv.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfrdiv_vf_<type><policy>
      // Example: __riscv_vfrdiv_vf_f32m1_tu, __riscv_vfrdiv_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfrdiv_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMinVVOp : RVV_Op<"vfmin.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point minimum";
  let description = [{
    Performs element-wise floating-point minimum: vd[i] = min(vs2[i], vs1[i]).
    Maps to vfmin.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfmin.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmin_vv_<type><policy>
      // Example: __riscv_vfmin_vv_f32m1_tu, __riscv_vfmin_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmin_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMinVFOp : RVV_Op<"vfmin.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point minimum";
  let description = [{
    Performs element-wise floating-point minimum with scalar: vd[i] = min(vs2[i], rs1).
    Maps to vfmin.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfmin.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmin_vf_<type><policy>
      // Example: __riscv_vfmin_vf_f32m1_tu, __riscv_vfmin_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmin_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMaxVVOp : RVV_Op<"vfmax.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point maximum";
  let description = [{
    Performs element-wise floating-point maximum: vd[i] = max(vs2[i], vs1[i]).
    Maps to vfmax.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfmax.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmax_vv_<type><policy>
      // Example: __riscv_vfmax_vv_f32m1_tu, __riscv_vfmax_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmax_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMaxVFOp : RVV_Op<"vfmax.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point maximum";
  let description = [{
    Performs element-wise floating-point maximum with scalar: vd[i] = max(vs2[i], rs1).
    Maps to vfmax.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfmax.vf %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmax_vf_<type><policy>
      // Example: __riscv_vfmax_vf_f32m1_tu, __riscv_vfmax_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmax_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNegVOp : RVV_Op<"vfneg.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point negation";
  let description = [{
    Performs element-wise floating-point negation: vd[i] = -vs2[i].
    Maps to vfneg.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfneg.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfneg_v_<type><policy>
      // Example: __riscv_vfneg_v_f32m1_tu, __riscv_vfneg_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfneg_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFAbsVOp : RVV_Op<"vfabs.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point absolute value";
  let description = [{
    Performs element-wise floating-point absolute value: vd[i] = |vs1[i]|.
    Maps to vfabs.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfabs.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfabs_v_<type><policy>
      // Example: __riscv_vfabs_v_f32m1_tu, __riscv_vfabs_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfabs_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMFEqVVOp : RVV_Op<"vmfeq.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point equality comparison";
  let description = [{
    Performs element-wise floating-point equality comparison: vd[i] = (vs2[i] == vs1[i]).
    Returns a mask indicating which elements are equal.
    Maps to vmfeq.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfeq.vv %vs1, %vs2, %vl : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfeq_vv_<type>_b<lmul>
      // Example: __riscv_vmfeq_vv_f32m1_b4
      return "__riscv_vmfeq_vv" + 
             getVs2().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFEqVFOp : RVV_Op<"vmfeq.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point equality comparison";
  let description = [{
    Performs element-wise floating-point equality comparison with scalar: vd[i] = (vs2[i] == rs1).
    Returns a mask indicating which elements are equal to the scalar.
    Maps to vmfeq.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfeq.vf %vs2, %rs1, %vl : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfeq_vf_<type>_b<lmul>
      // Example: __riscv_vmfeq_vf_f32m1_b4
      return "__riscv_vmfeq_vf" + 
             getVs1().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFNeVVOp : RVV_Op<"vmfne.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point inequality comparison";
  let description = [{
    Performs element-wise floating-point inequality comparison: vd[i] = (vs2[i] != vs1[i]).
    Returns a mask indicating which elements are not equal.
    Maps to vmfne.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfne.vv %vs1, %vs2, %vl : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfne_vv_<type>_b<lmul>
      // Example: __riscv_vmfne_vv_f32m1_b4
      return "__riscv_vmfne_vv" + 
             getVs2().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFNeVFOp : RVV_Op<"vmfne.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point inequality comparison";
  let description = [{
    Performs element-wise floating-point inequality comparison with scalar: vd[i] = (vs2[i] != rs1).
    Returns a mask indicating which elements are not equal to the scalar.
    Maps to vmfne.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfne.vf %vs2, %rs1, %vl : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfne_vf_<type>_b<lmul>
      // Example: __riscv_vmfne_vf_f32m1_b4
      return "__riscv_vmfne_vf" + 
             getVs1().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFLtVVOp : RVV_Op<"vmflt.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point less-than comparison";
  let description = [{
    Performs element-wise floating-point less-than comparison: vd[i] = (vs2[i] < vs1[i]).
    Returns a mask indicating which elements are less than.
    Maps to vmflt.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmflt.vv %vs1, %vs2, %vl : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmflt_vv_<type>_b<lmul>
      // Example: __riscv_vmflt_vv_f32m1_b4
      return "__riscv_vmflt_vv" + 
             getVs2().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFLtVFOp : RVV_Op<"vmflt.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point less-than comparison";
  let description = [{
    Performs element-wise floating-point less-than comparison with scalar: vd[i] = (vs2[i] < rs1).
    Returns a mask indicating which elements are less than the scalar.
    Maps to vmflt.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmflt.vf %vs2, %rs1, %vl : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmflt_vf_<type>_b<lmul>
      // Example: __riscv_vmflt_vf_f32m1_b4
      return "__riscv_vmflt_vf" + 
             getVs1().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFLeVVOp : RVV_Op<"vmfle.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point less-than-or-equal comparison";
  let description = [{
    Performs element-wise floating-point less-than-or-equal comparison: vd[i] = (vs2[i] <= vs1[i]).
    Returns a mask indicating which elements are less than or equal.
    Maps to vmfle.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfle.vv %vs1, %vs2, %vl : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfle_vv_<type>_b<lmul>
      // Example: __riscv_vmfle_vv_f32m1_b4
      return "__riscv_vmfle_vv" + 
             getVs2().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFLeVFOp : RVV_Op<"vmfle.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point less-than-or-equal comparison";
  let description = [{
    Performs element-wise floating-point less-than-or-equal comparison with scalar: vd[i] = (vs2[i] <= rs1).
    Returns a mask indicating which elements are less than or equal to the scalar.
    Maps to vmfle.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfle.vf %vs2, %rs1, %vl : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfle_vf_<type>_b<lmul>
      // Example: __riscv_vmfle_vf_f32m1_b4
      return "__riscv_vmfle_vf" + 
             getVs1().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFGtVVOp : RVV_Op<"vmfgt.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point greater-than comparison";
  let description = [{
    Performs element-wise floating-point greater-than comparison: vd[i] = (vs2[i] > vs1[i]).
    Returns a mask indicating which elements are greater than.
    Maps to vmfgt.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfgt.vv %vs1, %vs2, %vl : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfgt_vv_<type>_b<lmul>
      // Example: __riscv_vmfgt_vv_f32m1_b4
      return "__riscv_vmfgt_vv" + 
             getVs2().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFGtVFOp : RVV_Op<"vmfgt.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point greater-than comparison";
  let description = [{
    Performs element-wise floating-point greater-than comparison with scalar: vd[i] = (vs2[i] > rs1).
    Returns a mask indicating which elements are greater than the scalar.
    Maps to vmfgt.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfgt.vf %vs2, %rs1, %vl : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfgt_vf_<type>_b<lmul>
      // Example: __riscv_vmfgt_vf_f32m1_b4
      return "__riscv_vmfgt_vf" + 
             getVs1().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFGeVVOp : RVV_Op<"vmfge.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point greater-than-or-equal comparison";
  let description = [{
    Performs element-wise floating-point greater-than-or-equal comparison: vd[i] = (vs2[i] >= vs1[i]).
    Returns a mask indicating which elements are greater than or equal.
    Maps to vmfge.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfge.vv %vs1, %vs2, %vl : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfge_vv_<type>_b<lmul>
      // Example: __riscv_vmfge_vv_f32m1_b4
      return "__riscv_vmfge_vv" + 
             getVs2().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VMFGeVFOp : RVV_Op<"vmfge.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point greater-than-or-equal comparison";
  let description = [{
    Performs element-wise floating-point greater-than-or-equal comparison with scalar: vd[i] = (vs2[i] >= rs1).
    Returns a mask indicating which elements are greater than or equal to the scalar.
    Maps to vmfge.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmfge.vf %vs2, %rs1, %vl : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmfge_vf_<type>_b<lmul>
      // Example: __riscv_vmfge_vf_f32m1_b4
      return "__riscv_vmfge_vf" + 
             getVs1().getType().getVectorTypeString() + 
             "_" + getVd().getType().getMaskTypeString();
    }
  }];
}

def VFClassVOp : RVV_Op<"vfclass.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point classify";
  let description = [{
    Classifies each floating-point element into one of ten categories and returns
    an integer bitmask indicating the classification. The result is an unsigned integer
    with the same SEW as the input floating-point vector.
    
    Classification categories (bit positions):
    - bit 0: negative infinity
    - bit 1: negative normal number
    - bit 2: negative subnormal number
    - bit 3: negative zero
    - bit 4: positive zero
    - bit 5: positive subnormal number
    - bit 6: positive normal number
    - bit 7: positive infinity
    - bit 8: signaling NaN
    - bit 9: quiet NaN
    
    Maps to vfclass.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfclass.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<u32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfclass_v_<type><policy>
      // Example: __riscv_vfclass_v_f32m1_tu, __riscv_vfclass_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfclass_v" + 
             getVs1().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRec7VOp : RVV_Op<"vfrec7.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point reciprocal estimate (7-bit accuracy)";
  let description = [{
    Computes an approximation of the reciprocal (1/x) for each floating-point element.
    The result has a relative error of at most 2^-7 (about 0.78%).
    Maps to vfrec7.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfrec7.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfrec7_v_<type><policy>
      // Example: __riscv_vfrec7_v_f32m1_tu, __riscv_vfrec7_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfrec7_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRSqrt7VOp : RVV_Op<"vfrsqrt7.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point reciprocal square root estimate (7-bit accuracy)";
  let description = [{
    Computes an approximation of the reciprocal square root (1/sqrt(x)) for each 
    floating-point element. The result has a relative error of at most 2^-7 (about 0.78%).
    Maps to vfrsqrt7.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfrsqrt7.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfrsqrt7_v_<type><policy>
      // Example: __riscv_vfrsqrt7_v_f32m1_tu, __riscv_vfrsqrt7_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfrsqrt7_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMAccVVOp : RVV_Op<"vfmacc.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point multiply-accumulate";
  let description = [{
    Performs element-wise floating-point multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfmacc.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfmacc.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmacc_vv_<type><policy>
      // Example: __riscv_vfmacc_vv_f32m1_tu, __riscv_vfmacc_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmacc_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMAccVFOp : RVV_Op<"vfmacc.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point multiply-accumulate";
  let description = [{
    Performs element-wise floating-point multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfmacc.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfmacc.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmacc_vf_<type><policy>
      // Example: __riscv_vfmacc_vf_f32m1_tu, __riscv_vfmacc_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmacc_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMAccVVOp : RVV_Op<"vfnmacc.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point negated multiply-accumulate";
  let description = [{
    Performs element-wise floating-point negated multiply-accumulate: vd[i] = -(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfnmacc.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfnmacc.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmacc_vv_<type><policy>
      // Example: __riscv_vfnmacc_vv_f32m1_tu, __riscv_vfnmacc_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmacc_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMAccVFOp : RVV_Op<"vfnmacc.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point negated multiply-accumulate";
  let description = [{
    Performs element-wise floating-point negated multiply-accumulate with scalar: vd[i] = -(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfnmacc.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfnmacc.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmacc_vf_<type><policy>
      // Example: __riscv_vfnmacc_vf_f32m1_tu, __riscv_vfnmacc_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmacc_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMSacVVOp : RVV_Op<"vfmsac.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point multiply-subtract-accumulate";
  let description = [{
    Performs element-wise floating-point multiply-subtract-accumulate: vd[i] = +(vs1[i] * vs2[i]) - vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfmsac.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfmsac.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmsac_vv_<type><policy>
      // Example: __riscv_vfmsac_vv_f32m1_tu, __riscv_vfmsac_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmsac_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMSacVFOp : RVV_Op<"vfmsac.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point multiply-subtract-accumulate";
  let description = [{
    Performs element-wise floating-point multiply-subtract-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) - vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfmsac.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfmsac.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmsac_vf_<type><policy>
      // Example: __riscv_vfmsac_vf_f32m1_tu, __riscv_vfmsac_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmsac_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMSacVVOp : RVV_Op<"vfnmsac.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point negated multiply-subtract-accumulate";
  let description = [{
    Performs element-wise floating-point negated multiply-subtract-accumulate: vd[i] = -(vs1[i] * vs2[i]) - vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfnmsac.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfnmsac.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmsac_vv_<type><policy>
      // Example: __riscv_vfnmsac_vv_f32m1_tu, __riscv_vfnmsac_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmsac_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMSacVFOp : RVV_Op<"vfnmsac.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point negated multiply-subtract-accumulate";
  let description = [{
    Performs element-wise floating-point negated multiply-subtract-accumulate with scalar: vd[i] = -(rs1 * vs1[i]) - vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vfnmsac.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfnmsac.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmsac_vf_<type><policy>
      // Example: __riscv_vfnmsac_vf_f32m1_tu, __riscv_vfnmsac_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmsac_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMAddVVOp : RVV_Op<"vfmadd.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point multiply-add";
  let description = [{
    Performs element-wise floating-point multiply-add: vd[i] = +(vs1[i] * vs3[i]) + vs2[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfmadd.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfmadd.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmadd_vv_<type><policy>
      // Example: __riscv_vfmadd_vv_f32m1_tu, __riscv_vfmadd_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMAddVFOp : RVV_Op<"vfmadd.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point multiply-add";
  let description = [{
    Performs element-wise floating-point multiply-add with scalar: vd[i] = +(vs1[i] * rs1) + vs3[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfmadd.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfmadd.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmadd_vf_<type><policy>
      // Example: __riscv_vfmadd_vf_f32m1_tu, __riscv_vfmadd_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmadd_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMAddVVOp : RVV_Op<"vfnmadd.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point negated multiply-add";
  let description = [{
    Performs element-wise floating-point negated multiply-add: vd[i] = -(vs1[i] * vs3[i]) + vs2[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfnmadd.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfnmadd.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmadd_vv_<type><policy>
      // Example: __riscv_vfnmadd_vv_f32m1_tu, __riscv_vfnmadd_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMAddVFOp : RVV_Op<"vfnmadd.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point negated multiply-add";
  let description = [{
    Performs element-wise floating-point negated multiply-add with scalar: vd[i] = -(vs1[i] * rs1) + vs3[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfnmadd.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfnmadd.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmadd_vf_<type><policy>
      // Example: __riscv_vfnmadd_vf_f32m1_tu, __riscv_vfnmadd_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmadd_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMSubVVOp : RVV_Op<"vfmsub.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point multiply-subtract";
  let description = [{
    Performs element-wise floating-point multiply-subtract: vd[i] = +(vs1[i] * vs3[i]) - vs2[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfmsub.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfmsub.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmsub_vv_<type><policy>
      // Example: __riscv_vfmsub_vv_f32m1_tu, __riscv_vfmsub_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMSubVFOp : RVV_Op<"vfmsub.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point multiply-subtract";
  let description = [{
    Performs element-wise floating-point multiply-subtract with scalar: vd[i] = +(vs1[i] * rs1) - vs3[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfmsub.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfmsub.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmsub_vf_<type><policy>
      // Example: __riscv_vfmsub_vf_f32m1_tu, __riscv_vfmsub_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfmsub_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMSubVVOp : RVV_Op<"vfnmsub.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector floating-point negated multiply-subtract";
  let description = [{
    Performs element-wise floating-point negated multiply-subtract: vd[i] = -(vs1[i] * vs3[i]) - vs2[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfnmsub.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfnmsub.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmsub_vv_<type><policy>
      // Example: __riscv_vfnmsub_vv_f32m1_tu, __riscv_vfnmsub_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNMSubVFOp : RVV_Op<"vfnmsub.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar floating-point negated multiply-subtract";
  let description = [{
    Performs element-wise floating-point negated multiply-subtract with scalar: vd[i] = -(vs1[i] * rs1) - vs3[i].
    This is a destructive operation where vs1 is the multiplicand input.
    Maps to vfnmsub.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs1 serves as the multiplicand input.
    
    Example:
    ```mlir
    %result = rvv.vfnmsub.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfnmsub_vf_<type><policy>
      // Example: __riscv_vfnmsub_vf_f32m1_tu, __riscv_vfnmsub_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfnmsub_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSgnjVVOp : RVV_Op<"vfsgnj.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point sign-injection";
  let description = [{
    Performs element-wise floating-point sign-injection: vd[i] = {vs1[i][31:0], vs2[i][31]}.
    Copies the magnitude from vs2 and the sign from vs1.
    Maps to vfsgnj.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsgnj.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsgnj_vv_<type><policy>
      // Example: __riscv_vfsgnj_vv_f32m1_tu, __riscv_vfsgnj_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsgnj_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSgnjVFOp : RVV_Op<"vfsgnj.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point sign-injection";
  let description = [{
    Performs element-wise floating-point sign-injection with scalar: vd[i] = {rs1[31:0], vs1[i][31]}.
    Copies the magnitude from vs1 and the sign from rs1.
    Maps to vfsgnj.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsgnj.vf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsgnj_vf_<type><policy>
      // Example: __riscv_vfsgnj_vf_f32m1_tu, __riscv_vfsgnj_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsgnj_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSgnjnVVOp : RVV_Op<"vfsgnjn.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point negated sign-injection";
  let description = [{
    Performs element-wise floating-point negated sign-injection: vd[i] = {~vs1[i][31], vs2[i][30:0]}.
    Copies the magnitude from vs2 and the negated sign from vs1.
    Maps to vfsgnjn.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsgnjn.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsgnjn_vv_<type><policy>
      // Example: __riscv_vfsgnjn_vv_f32m1_tu, __riscv_vfsgnjn_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsgnjn_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSgnjnVFOp : RVV_Op<"vfsgnjn.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point negated sign-injection";
  let description = [{
    Performs element-wise floating-point negated sign-injection with scalar: vd[i] = {~rs1[31], vs1[i][30:0]}.
    Copies the magnitude from vs1 and the negated sign from rs1.
    Maps to vfsgnjn.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsgnjn.vf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsgnjn_vf_<type><policy>
      // Example: __riscv_vfsgnjn_vf_f32m1_tu, __riscv_vfsgnjn_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsgnjn_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSgnjxVVOp : RVV_Op<"vfsgnjx.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector floating-point XOR sign-injection";
  let description = [{
    Performs element-wise floating-point XOR sign-injection: vd[i] = {vs1[i][31] ^ vs2[i][31], vs2[i][30:0]}.
    Copies the magnitude from vs2 and XORs the signs from vs1 and vs2.
    Maps to vfsgnjx.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsgnjx.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsgnjx_vv_<type><policy>
      // Example: __riscv_vfsgnjx_vv_f32m1_tu, __riscv_vfsgnjx_vv_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsgnjx_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSgnjxVFOp : RVV_Op<"vfsgnjx.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar floating-point XOR sign-injection";
  let description = [{
    Performs element-wise floating-point XOR sign-injection with scalar: vd[i] = {rs1[31] ^ vs1[i][31], vs1[i][30:0]}.
    Copies the magnitude from vs1 and XORs the signs from rs1 and vs1.
    Maps to vfsgnjx.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsgnjx.vf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsgnjx_vf_<type><policy>
      // Example: __riscv_vfsgnjx_vf_f32m1_tu, __riscv_vfsgnjx_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsgnjx_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSqrtVOp : RVV_Op<"vfsqrt.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point square root";
  let description = [{
    Performs element-wise floating-point square root: vd[i] = sqrt(vs1[i]).
    Maps to vfsqrt.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfsqrt.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfsqrt_v_<type><policy>
      // Example: __riscv_vfsqrt_v_f32m1_tu, __riscv_vfsqrt_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfsqrt_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWAddVVOp : RVV_Op<"vfwadd.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point addition";
  let description = [{
    Performs element-wise widening floating-point addition.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwadd.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwadd.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwadd_vv_<type><policy>
      // Example: __riscv_vfwadd_vv_f64m2_tu, __riscv_vfwadd_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWAddVFOp : RVV_Op<"vfwadd.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point addition";
  let description = [{
    Performs element-wise widening floating-point addition with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwadd.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwadd.vf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwadd_vf_<type><policy>
      // Example: __riscv_vfwadd_vf_f64m2_tu, __riscv_vfwadd_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwadd_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWAddWVOp : RVV_Op<"vfwadd.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point addition (wide-narrow)";
  let description = [{
    Performs element-wise widening floating-point addition with wide first operand.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwadd.wv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwadd.wv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwadd_wv_<type><policy>
      // Example: __riscv_vfwadd_wv_f64m2_tu, __riscv_vfwadd_wv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwadd_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWAddWFOp : RVV_Op<"vfwadd.wf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point addition (wide-narrow)";
  let description = [{
    Performs element-wise widening floating-point addition with wide vector and narrow scalar.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwadd.wf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwadd.wf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, f32, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwadd_wf_<type><policy>
      // Example: __riscv_vfwadd_wf_f64m2_tu, __riscv_vfwadd_wf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwadd_wf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWMulVVOp : RVV_Op<"vfwmul.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point multiplication";
  let description = [{
    Performs element-wise widening floating-point multiplication.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Returns the full 2*SEW-bit floating-point product.
    Maps to vfwmul.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwmul.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwmul_vv_<type><policy>
      // Example: __riscv_vfwmul_vv_f64m2_tu, __riscv_vfwmul_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwmul_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWMulVFOp : RVV_Op<"vfwmul.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point multiplication";
  let description = [{
    Performs element-wise widening floating-point multiplication with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: wide scalar (2*SEW bits) - same width as result
    - vd: wide vector (2*SEW bits)
    Returns the full 2*SEW-bit floating-point product.
    Maps to vfwmul.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwmul.vf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f64, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwmul_vf_<type><policy>
      // Example: __riscv_vfwmul_vf_f64m2_tu, __riscv_vfwmul_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwmul_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWMAccVVOp : RVV_Op<"vfwmacc.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point multiply-accumulate";
  let description = [{
    Performs element-wise widening floating-point multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwmacc.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwmacc.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwmacc_vv_<type><policy>
      // Example: __riscv_vfwmacc_vv_f64m2_tu, __riscv_vfwmacc_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwmacc_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWMAccVFOp : RVV_Op<"vfwmacc.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point multiply-accumulate";
  let description = [{
    Performs element-wise widening floating-point multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwmacc.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwmacc.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwmacc_vf_<type><policy>
      // Example: __riscv_vfwmacc_vf_f64m2_tu, __riscv_vfwmacc_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwmacc_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWNMAccVVOp : RVV_Op<"vfwnmacc.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point negated multiply-accumulate";
  let description = [{
    Performs element-wise widening floating-point negated multiply-accumulate: vd[i] = -(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwnmacc.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwnmacc.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwnmacc_vv_<type><policy>
      // Example: __riscv_vfwnmacc_vv_f64m2_tu, __riscv_vfwnmacc_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwnmacc_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWNMAccVFOp : RVV_Op<"vfwnmacc.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point negated multiply-accumulate";
  let description = [{
    Performs element-wise widening floating-point negated multiply-accumulate with scalar: vd[i] = -(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwnmacc.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwnmacc.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwnmacc_vf_<type><policy>
      // Example: __riscv_vfwnmacc_vf_f64m2_tu, __riscv_vfwnmacc_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwnmacc_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWMSacVVOp : RVV_Op<"vfwmsac.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point multiply-subtract-accumulate";
  let description = [{
    Performs element-wise widening floating-point multiply-subtract-accumulate: vd[i] = +(vs1[i] * vs2[i]) - vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwmsac.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwmsac.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwmsac_vv_<type><policy>
      // Example: __riscv_vfwmsac_vv_f64m2_tu, __riscv_vfwmsac_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwmsac_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWMSacVFOp : RVV_Op<"vfwmsac.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point multiply-subtract-accumulate";
  let description = [{
    Performs element-wise widening floating-point multiply-subtract-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) - vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwmsac.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwmsac.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwmsac_vf_<type><policy>
      // Example: __riscv_vfwmsac_vf_f64m2_tu, __riscv_vfwmsac_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwmsac_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWNMSacVVOp : RVV_Op<"vfwnmsac.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point negated multiply-subtract-accumulate";
  let description = [{
    Performs element-wise widening floating-point negated multiply-subtract-accumulate: vd[i] = -(vs1[i] * vs2[i]) - vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwnmsac.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwnmsac.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwnmsac_vv_<type><policy>
      // Example: __riscv_vfwnmsac_vv_f64m2_tu, __riscv_vfwnmsac_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwnmsac_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWNMSacVFOp : RVV_Op<"vfwnmsac.vf", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point negated multiply-subtract-accumulate";
  let description = [{
    Performs element-wise widening floating-point negated multiply-subtract-accumulate with scalar: vd[i] = -(rs1 * vs1[i]) - vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vs3: wide vector (2*SEW bits) - accumulator
    - vd: wide vector (2*SEW bits)
    Maps to vfwnmsac.vf intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vfwnmsac.vf %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f32, !rvv.vector<f64, m2>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs3,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwnmsac_vf_<type><policy>
      // Example: __riscv_vfwnmsac_vf_f64m2_tu, __riscv_vfwnmsac_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwnmsac_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWSubVVOp : RVV_Op<"vfwsub.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point subtraction";
  let description = [{
    Performs element-wise widening floating-point subtraction.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwsub.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwsub.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwsub_vv_<type><policy>
      // Example: __riscv_vfwsub_vv_f64m2_tu, __riscv_vfwsub_vv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWSubVFOp : RVV_Op<"vfwsub.vf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point subtraction";
  let description = [{
    Performs element-wise widening floating-point subtraction with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: wide scalar (2*SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwsub.vf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwsub.vf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, f64, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwsub_vf_<type><policy>
      // Example: __riscv_vfwsub_vf_f64m2_tu, __riscv_vfwsub_vf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwsub_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWSubWVOp : RVV_Op<"vfwsub.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening floating-point subtraction (wide-narrow)";
  let description = [{
    Performs element-wise widening floating-point subtraction with wide and narrow vectors.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwsub.wv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwsub.wv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, !rvv.vector<f32, m1>, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwsub_wv_<type><policy>
      // Example: __riscv_vfwsub_wv_f64m2_tu, __riscv_vfwsub_wv_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwsub_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWSubWFOp : RVV_Op<"vfwsub.wf", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-scalar widening floating-point subtraction (wide-narrow)";
  let description = [{
    Performs element-wise widening floating-point subtraction with wide vector and narrow scalar.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vfwsub.wf intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwsub.wf %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, f32, i64) 
      -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwsub_wf_<type><policy>
      // Example: __riscv_vfwsub_wf_f64m2_tu, __riscv_vfwsub_wf_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwsub_wf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRedOSumVSOp : RVV_Op<"vfredosum.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point ordered sum reduction";
  let description = [{
    Performs an ordered sum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    The reduction is performed in element order to maintain reproducibility.
    Maps to vfredosum.vs intrinsics.
  }];
  let arguments = (ins
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfredosum_vs_<type><policy>
      // Example: __riscv_vfredosum_vs_f32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfredosum_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRedUSumVSOp : RVV_Op<"vfredusum.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point unordered sum reduction";
  let description = [{
    Performs an unordered sum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    The reduction order is unspecified, which may allow for better performance.
    Maps to vfredusum.vs intrinsics.
  }];
  let arguments = (ins
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfredusum_vs_<type><policy>
      // Example: __riscv_vfredusum_vs_f32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfredusum_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRedMaxVSOp : RVV_Op<"vfredmax.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point maximum reduction";
  let description = [{
    Performs a maximum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vfredmax.vs intrinsics.
  }];
  let arguments = (ins
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfredmax_vs_<type><policy>
      // Example: __riscv_vfredmax_vs_f32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfredmax_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFRedMinVSOp : RVV_Op<"vfredmin.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector floating-point minimum reduction";
  let description = [{
    Performs a minimum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vfredmin.vs intrinsics.
  }];
  let arguments = (ins
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfredmin_vs_<type><policy>
      // Example: __riscv_vfredmin_vs_f32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfredmin_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWRedOSumVSOp : RVV_Op<"vfwredosum.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector widening floating-point ordered sum reduction";
  let description = [{
    Performs a widening ordered sum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value (2*SEW bits).
    The result is 2*SEW bits wide.
    The reduction is performed in element order to maintain reproducibility.
    Maps to vfwredosum.vs intrinsics.
  }];
  let arguments = (ins
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwredosum_vs_<type><policy>
      // Example: __riscv_vfwredosum_vs_f32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwredosum_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWRedUSumVSOp : RVV_Op<"vfwredusum.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector widening floating-point unordered sum reduction";
  let description = [{
    Performs a widening unordered sum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value (2*SEW bits).
    The result is 2*SEW bits wide.
    The reduction order is unspecified, which may allow for better performance.
    Maps to vfwredusum.vs intrinsics.
  }];
  let arguments = (ins
    RVV_VectorType:$vs2,
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwredusum_vs_<type><policy>
      // Example: __riscv_vfwredusum_vs_f32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwredusum_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSlideDownVFOp : RVV_Op<"vslidedown.vf", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector slide down by floating-point scalar offset";
  let description = [{
    Slides floating-point vector elements down by a scalar offset.
    Maps to vslidedown.vf intrinsics.
    
    For each element i: vd[i] = (i + rs1 < vl) ? vs1[i + rs1] : 0
    
    Arguments:
    - `vs1`: Source floating-point vector to slide
    - `rs1`: Scalar offset (number of positions to slide down)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result floating-point vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vslidedown.vf %src, %offset, %vl : !rvv.vector<f32, m1>, i64, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vslidedown_vf_<type><policy>
      // Example: __riscv_vslidedown_vf_f32m1_tu, __riscv_vslidedown_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vslidedown_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSlide1DownVFOp : RVV_Op<"vfslide1down.vf", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Floating-point vector slide1 down with scalar fill";
  let description = [{
    Slides floating-point vector elements down by 1 position, filling with scalar value.
    Maps to vfslide1down.vf intrinsics.
    
    For each element i: vd[i] = (i == vl-1) ? rs1 : vs1[i + 1]
    
    Arguments:
    - `vs1`: Source floating-point vector to slide
    - `rs1`: Floating-point scalar value to insert at position vl-1
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result floating-point vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vfslide1down.vf %src, %scalar, %vl : !rvv.vector<f32, m1>, f32, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfslide1down_vf_<type><policy>
      // Example: __riscv_vfslide1down_vf_f32m1_tu, __riscv_vfslide1down_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfslide1down_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSlideUpVFOp : RVV_Op<"vslideup.vf", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector slide up by floating-point scalar offset";
  let description = [{
    Slides floating-point vector elements up by a scalar offset.
    Maps to vslideup.vf intrinsics.
    
    For each element i: vd[i] = (i < rs1) ? vd[i] : vs1[i - rs1]
    
    Arguments:
    - `vs1`: Source floating-point vector to slide
    - `rs1`: Scalar offset (number of positions to slide up)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result floating-point vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vslideup.vf %src, %offset, %vl : !rvv.vector<f32, m1>, i64, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vslideup_vf_<type><policy>
      // Example: __riscv_vslideup_vf_f32m1_tu, __riscv_vslideup_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vslideup_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSlide1UpVFOp : RVV_Op<"vfslide1up.vf", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Floating-point vector slide1 up with scalar fill";
  let description = [{
    Slides floating-point vector elements up by 1 position, filling with scalar value.
    Maps to vfslide1up.vf intrinsics.
    
    For each element i: vd[i] = (i == 0) ? rs1 : vs1[i - 1]
    
    Arguments:
    - `vs1`: Source floating-point vector to slide
    - `rs1`: Floating-point scalar value to insert at position 0
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result floating-point vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vfslide1up.vf %src, %scalar, %vl : !rvv.vector<f32, m1>, f32, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfslide1up_vf_<type><policy>
      // Example: __riscv_vfslide1up_vf_f32m1_tu, __riscv_vfslide1up_vf_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfslide1up_vf" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFMvFSOp : RVV_Op<"vfmv.f.s", [Pure]> {
  let summary = "Move vector element to floating-point scalar register";
  let description = [{
    Moves the first element (element 0) from a floating-point vector register
    to a scalar floating-point register.
    
    fd = vs1[0]
    
    Arguments:
    - `vs1`: Source floating-point vector
    
    Results:
    - `fd`: Scalar floating-point result (type matching vector element type)
    
    Example:
    ```mlir
    %scalar = rvv.vfmv.f.s %vec : !rvv.vector<f32, m1> -> f32
    ```
  }];
  let arguments = (ins RVV_VectorType:$vs1);
  let results = (outs RVV_ScalarType:$fd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmv_f_s_<type>
      // Example: __riscv_vfmv_f_s_f32m1
      return "__riscv_vfmv_f_s" + 
             getVs1().getType().getVectorTypeString();
    }
  }];
}

def VFMergeVFMOp : RVV_Op<"vfmerge.vfm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Floating-point vector-scalar merge with mask";
  let description = [{
    Merges a floating-point scalar value and a floating-point vector based on a mask.
    For each element position, if the mask bit is 1, the element is set to the scalar
    value, otherwise taken from vs2.
    
    For each element i: vd[i] = mask[i] ? fs1 : vs2[i]
    
    Arguments:
    - `fs1`: Floating-point scalar value (selected when mask bit is 1)
    - `vs2`: Source floating-point vector (selected when mask bit is 0)
    - `mask`: Mask controlling the merge (must be provided)
    - `vl`: Vector length - number of elements to process
    
    Results:
    - `vd`: Result floating-point vector with merged elements
    
    Example:
    ```mlir
    %result = rvv.vfmerge.vfm %scalar, %vs2, %mask, %vl : f32, !rvv.vector<f32, m1>, !rvv.mask<m1>, i64
    ```
  }];
  let arguments = (ins
    RVV_ScalarType:$fs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$mask,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmerge_vfm_<type>
      // Example: __riscv_vfmerge_vfm_f32m1
      return "__riscv_vfmerge_vfm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VFMvVFOp : RVV_Op<"vfmv.v.f", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Move floating-point scalar to vector";
  let description = [{
    Moves (broadcasts) a scalar floating-point value to all elements of a 
    floating-point vector register. Each element in the destination vector
    is set to the scalar value.
    
    For each element i (where i < vl): vd[i] = fs1
    
    Arguments:
    - `fs1`: Scalar floating-point value to broadcast
    - `vl`: Vector length - number of elements to set
    
    Results:
    - `vd`: Destination floating-point vector with all elements set to the scalar value
    
    Example:
    ```mlir
    %result = rvv.vfmv.v.f %scalar, %vl : f32, i64 -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_ScalarType:$fs1,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmv_v_f_<type>
      // Example: __riscv_vfmv_v_f_f32m1
      return "__riscv_vfmv_v_f" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VFMvSFOp : RVV_Op<"vfmv.s.f", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Move floating-point scalar to vector element 0";
  let description = [{
    Moves a scalar floating-point value to element 0 of a floating-point vector register.
    Other elements in the vector are preserved from the source vector.
    
    vd[0] = fs1
    vd[i] = vs2[i] for i > 0
    
    Arguments:
    - `fs1`: Scalar floating-point value to move into element 0
    - `vs2`: Source floating-point vector (elements 1..N-1 are preserved)
    - `vl`: Vector length (must be at least 1)
    
    Results:
    - `vd`: Destination floating-point vector with element 0 set to scalar value
    
    Example:
    ```mlir
    %result = rvv.vfmv.s.f %scalar, %vec, %vl : f32, !rvv.vector<f32, m1>, i64 
      -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_ScalarType:$fs1,
    RVV_VectorType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfmv_s_f_<type>
      // Example: __riscv_vfmv_s_f_f32m1
      return "__riscv_vfmv_s_f" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VFWCvtFFVOp : RVV_Op<"vfwcvt.f.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening floating-point conversion";
  let description = [{
    Performs element-wise widening of floating-point values, doubling the element width.
    Each SEW-bit floating-point element is widened to 2*SEW bits with exact conversion.
    
    - vs1: narrow floating-point vector (SEW bits)
    - vd: wide floating-point vector (2*SEW bits)
    
    For example: f32 → f64, f16 → f32
    
    Maps to __riscv_vfwcvt_f_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.f.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_f_f_v_<type><policy>
      // Example: __riscv_vfwcvt_f_f_v_f64m2_tu, __riscv_vfwcvt_f_f_v_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_f_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtFFWOp : RVV_Op<"vfncvt.f.f.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing floating-point conversion";
  let description = [{
    Performs element-wise narrowing of floating-point values, halving the element width.
    Each 2*SEW-bit floating-point element is narrowed to SEW bits using the current
    rounding mode.
    
    - vs1: wide floating-point vector (2*SEW bits)
    - vd: narrow floating-point vector (SEW bits)
    
    For example: f64 → f32, f32 → f16
    
    The conversion follows IEEE 754 rounding rules based on the current frm (floating-point
    rounding mode) setting.
    
    Maps to __riscv_vfncvt_f_f_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.f.f.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, i64) -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_f_f_w_<type><policy>
      // Example: __riscv_vfncvt_f_f_w_f32m1_tu, __riscv_vfncvt_f_f_w_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_f_f_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtRodFFWOp : RVV_Op<"vfncvt.rod.f.f.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing floating-point conversion with round-to-odd";
  let description = [{
    Performs element-wise narrowing of floating-point values, halving the element width,
    using the round-to-odd (ROD) rounding mode.
    
    - vs1: wide floating-point vector (2*SEW bits)
    - vd: narrow floating-point vector (SEW bits)
    
    For example: f64 → f32, f32 → f16
    
    Round-to-odd is a special rounding mode where:
    - If the result is exact, return the exact value
    - If the result needs rounding, round to the nearest value with an odd significand
    
    This mode is useful for implementing correctly-rounded multi-step floating-point
    operations, as it preserves information about whether rounding occurred.
    
    Maps to __riscv_vfncvt_rod_f_f_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.rod.f.f.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, i64) -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_rod_f_f_w_<type><policy>
      // Example: __riscv_vfncvt_rod_f_f_w_f32m1_tu, __riscv_vfncvt_rod_f_f_w_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_rod_f_f_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFCvtFXVOp : RVV_Op<"vfcvt.f.x.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Convert signed integer to floating-point";
  let description = [{
    Performs element-wise conversion from signed integer to floating-point.
    The input and output have the same SEW (element width).
    
    - vs1: signed integer vector (SEW bits)
    - vd: floating-point vector (SEW bits)
    
    For example: i32 → f32, i64 → f64
    
    The conversion uses the current floating-point rounding mode (frm) when the
    integer value cannot be exactly represented in the floating-point format.
    
    Maps to __riscv_vfcvt_f_x_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfcvt.f.x.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfcvt_f_x_v_<type><policy>
      // Example: __riscv_vfcvt_f_x_v_f32m1_tu, __riscv_vfcvt_f_x_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfcvt_f_x_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFCvtFXUVOp : RVV_Op<"vfcvt.f.xu.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Convert unsigned integer to floating-point";
  let description = [{
    Performs element-wise conversion from unsigned integer to floating-point.
    The input and output have the same SEW (element width).
    
    - vs1: unsigned integer vector (SEW bits)
    - vd: floating-point vector (SEW bits)
    
    For example: u32 → f32, u64 → f64
    
    The conversion uses the current floating-point rounding mode (frm) when the
    integer value cannot be exactly represented in the floating-point format.
    
    Maps to __riscv_vfcvt_f_xu_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfcvt.f.xu.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfcvt_f_xu_v_<type><policy>
      // Example: __riscv_vfcvt_f_xu_v_f32m1_tu, __riscv_vfcvt_f_xu_v_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfcvt_f_xu_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtFXWOp : RVV_Op<"vfncvt.f.x.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing convert wide signed integer to narrow floating-point";
  let description = [{
    Performs element-wise conversion from wide signed integer to narrow floating-point,
    halving the element width.
    
    - vs1: wide signed integer vector (2*SEW bits)
    - vd: narrow floating-point vector (SEW bits)
    
    For example: i64 → f32, i32 → f16
    
    The conversion uses the current floating-point rounding mode (frm) when the
    integer value cannot be exactly represented in the narrow floating-point format.
    
    This is a narrowing operation where the input integer is twice as wide as the
    output floating-point value.
    
    Maps to __riscv_vfncvt_f_x_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.f.x.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i64) -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_f_x_w_<type><policy>
      // Example: __riscv_vfncvt_f_x_w_f32m1_tu, __riscv_vfncvt_f_x_w_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_f_x_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtFXUWOp : RVV_Op<"vfncvt.f.xu.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing convert wide unsigned integer to narrow floating-point";
  let description = [{
    Performs element-wise conversion from wide unsigned integer to narrow floating-point,
    halving the element width.
    
    - vs1: wide unsigned integer vector (2*SEW bits)
    - vd: narrow floating-point vector (SEW bits)
    
    For example: u64 → f32, u32 → f16
    
    The conversion uses the current floating-point rounding mode (frm) when the
    integer value cannot be exactly represented in the narrow floating-point format.
    
    This is a narrowing operation where the input integer is twice as wide as the
    output floating-point value.
    
    Maps to __riscv_vfncvt_f_xu_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.f.xu.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i64) -> !rvv.vector<f32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_f_xu_w_<type><policy>
      // Example: __riscv_vfncvt_f_xu_w_f32m1_tu, __riscv_vfncvt_f_xu_w_f32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_f_xu_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWCvtFXVOp : RVV_Op<"vfwcvt.f.x.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening convert narrow signed integer to wide floating-point";
  let description = [{
    Performs element-wise conversion from narrow signed integer to wide floating-point,
    doubling the element width.
    
    - vs1: narrow signed integer vector (SEW bits)
    - vd: wide floating-point vector (2*SEW bits)
    
    For example: i32 → f64, i16 → f32
    
    The conversion is typically exact as the wider floating-point format can usually
    represent all values from the narrower integer type. For very large integers,
    the current floating-point rounding mode (frm) is used if needed.
    
    This is a widening operation where the output floating-point is twice as wide
    as the input integer.
    
    Maps to __riscv_vfwcvt_f_x_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.f.x.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_f_x_v_<type><policy>
      // Example: __riscv_vfwcvt_f_x_v_f64m2_tu, __riscv_vfwcvt_f_x_v_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_f_x_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWCvtFXUVOp : RVV_Op<"vfwcvt.f.xu.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening convert narrow unsigned integer to wide floating-point";
  let description = [{
    Performs element-wise conversion from narrow unsigned integer to wide floating-point,
    doubling the element width.
    
    - vs1: narrow unsigned integer vector (SEW bits)
    - vd: wide floating-point vector (2*SEW bits)
    
    For example: u32 → f64, u16 → f32
    
    The conversion is typically exact as the wider floating-point format can usually
    represent all values from the narrower unsigned integer type. For very large
    unsigned integers, the current floating-point rounding mode (frm) is used if needed.
    
    This is a widening operation where the output floating-point is twice as wide
    as the input integer.
    
    Maps to __riscv_vfwcvt_f_xu_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.f.xu.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_f_xu_v_<type><policy>
      // Example: __riscv_vfwcvt_f_xu_v_f64m2_tu, __riscv_vfwcvt_f_xu_v_f64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_f_xu_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWCvtXFVOp : RVV_Op<"vfwcvt.x.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening convert narrow floating-point to wide signed integer";
  let description = [{
    Performs element-wise conversion from narrow floating-point to wide signed integer,
    doubling the element width.
    
    - vs1: narrow floating-point vector (SEW bits)
    - vd: wide signed integer vector (2*SEW bits)
    
    For example: f32 → i64, f16 → i32
    
    The conversion uses the current floating-point rounding mode (frm) from the fcsr.
    Values outside the representable range saturate to the maximum/minimum integer values.
    NaN converts to the maximum positive integer value.
    
    This is a widening operation where the output integer is twice as wide
    as the input floating-point value.
    
    Maps to __riscv_vfwcvt_x_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.x.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_x_f_v_<type><policy>
      // Example: __riscv_vfwcvt_x_f_v_i64m2_tu, __riscv_vfwcvt_x_f_v_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_x_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWCvtXUFVOp : RVV_Op<"vfwcvt.xu.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening convert narrow floating-point to wide unsigned integer";
  let description = [{
    Performs element-wise conversion from narrow floating-point to wide unsigned integer,
    doubling the element width.
    
    - vs1: narrow floating-point vector (SEW bits)
    - vd: wide unsigned integer vector (2*SEW bits)
    
    For example: f32 → u64, f16 → u32
    
    The conversion uses the current floating-point rounding mode (frm) from the fcsr.
    Values outside the representable range saturate to the maximum/minimum unsigned values.
    Negative values saturate to zero. NaN converts to the maximum unsigned integer value.
    
    This is a widening operation where the output integer is twice as wide
    as the input floating-point value.
    
    Maps to __riscv_vfwcvt_xu_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.xu.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_xu_f_v_<type><policy>
      // Example: __riscv_vfwcvt_xu_f_v_u64m2_tu, __riscv_vfwcvt_xu_f_v_u64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_xu_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWCvtRtzXFVOp : RVV_Op<"vfwcvt.rtz.x.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening convert narrow float to wide signed int with round-to-zero";
  let description = [{
    Performs element-wise conversion from narrow floating-point to wide signed integer
    using round-to-zero (truncate) rounding mode, doubling the element width.
    
    - vs1: narrow floating-point vector (SEW bits)
    - vd: wide signed integer vector (2*SEW bits)
    
    For example: f32 → i64, f16 → i32
    
    Round-to-zero (RTZ) always truncates toward zero, regardless of the current frm:
    - Positive values round down
    - Negative values round up
    
    This is equivalent to C-style type casting. Values outside the representable
    range saturate to the maximum/minimum integer values. NaN converts to the
    maximum positive integer value.
    
    This is a widening operation where the output integer is twice as wide
    as the input floating-point value.
    
    Maps to __riscv_vfwcvt_rtz_x_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.rtz.x.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_rtz_x_f_v_<type><policy>
      // Example: __riscv_vfwcvt_rtz_x_f_v_i64m2_tu, __riscv_vfwcvt_rtz_x_f_v_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_rtz_x_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFWCvtRtzXUFVOp : RVV_Op<"vfwcvt.rtz.xu.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening convert narrow float to wide unsigned int with round-to-zero";
  let description = [{
    Performs element-wise conversion from narrow floating-point to wide unsigned integer
    using round-to-zero (truncate) rounding mode, doubling the element width.
    
    - vs1: narrow floating-point vector (SEW bits)
    - vd: wide unsigned integer vector (2*SEW bits)
    
    For example: f32 → u64, f16 → u32
    
    Round-to-zero (RTZ) always truncates toward zero (rounds down for positive values),
    regardless of the current frm setting.
    
    This is equivalent to C-style type casting for unsigned types. Values outside the
    representable range saturate to the maximum/minimum unsigned values. Negative values
    saturate to zero. NaN converts to the maximum unsigned integer value.
    
    This is a widening operation where the output integer is twice as wide
    as the input floating-point value.
    
    Maps to __riscv_vfwcvt_rtz_xu_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfwcvt.rtz.xu.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfwcvt_rtz_xu_f_v_<type><policy>
      // Example: __riscv_vfwcvt_rtz_xu_f_v_u64m2_tu, __riscv_vfwcvt_rtz_xu_f_v_u64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfwcvt_rtz_xu_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFCvtXFVOp : RVV_Op<"vfcvt.x.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Convert floating-point to signed integer";
  let description = [{
    Performs element-wise conversion from floating-point to signed integer.
    The input and output have the same SEW (element width).
    
    - vs1: floating-point vector (SEW bits)
    - vd: signed integer vector (SEW bits)
    
    For example: f32 → i32, f64 → i64
    
    The conversion uses the current floating-point rounding mode (frm) from the fcsr.
    Values outside the representable range saturate to the maximum/minimum integer values.
    NaN converts to the maximum positive integer value.
    
    Maps to __riscv_vfcvt_x_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfcvt.x.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfcvt_x_f_v_<type><policy>
      // Example: __riscv_vfcvt_x_f_v_i32m1_tu, __riscv_vfcvt_x_f_v_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfcvt_x_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFCvtXUFVOp : RVV_Op<"vfcvt.xu.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Convert floating-point to unsigned integer";
  let description = [{
    Performs element-wise conversion from floating-point to unsigned integer.
    The input and output have the same SEW (element width).
    
    - vs1: floating-point vector (SEW bits)
    - vd: unsigned integer vector (SEW bits)
    
    For example: f32 → u32, f64 → u64
    
    The conversion uses the current floating-point rounding mode (frm) from the fcsr.
    Values outside the representable range saturate to the maximum/minimum unsigned values.
    Negative values saturate to zero. NaN converts to the maximum unsigned integer value.
    
    Maps to __riscv_vfcvt_xu_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfcvt.xu.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfcvt_xu_f_v_<type><policy>
      // Example: __riscv_vfcvt_xu_f_v_u32m1_tu, __riscv_vfcvt_xu_f_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfcvt_xu_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFCvtRtzXFVOp : RVV_Op<"vfcvt.rtz.x.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Convert floating-point to signed integer with round-to-zero";
  let description = [{
    Performs element-wise conversion from floating-point to signed integer using
    round-to-zero (truncate) rounding mode, regardless of the current frm setting.
    The input and output have the same SEW (element width).
    
    - vs1: floating-point vector (SEW bits)
    - vd: signed integer vector (SEW bits)
    
    For example: f32 → i32, f64 → i64
    
    Round-to-zero (RTZ) always truncates toward zero:
    - Positive values round down
    - Negative values round up
    
    This is equivalent to C-style type casting. Values outside the representable
    range saturate to the maximum/minimum integer values. NaN converts to the
    maximum positive integer value.
    
    Maps to __riscv_vfcvt_rtz_x_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfcvt.rtz.x.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfcvt_rtz_x_f_v_<type><policy>
      // Example: __riscv_vfcvt_rtz_x_f_v_i32m1_tu, __riscv_vfcvt_rtz_x_f_v_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfcvt_rtz_x_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFCvtRtzXUFVOp : RVV_Op<"vfcvt.rtz.xu.f.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Convert floating-point to unsigned integer with round-to-zero";
  let description = [{
    Performs element-wise conversion from floating-point to unsigned integer using
    round-to-zero (truncate) rounding mode, regardless of the current frm setting.
    The input and output have the same SEW (element width).
    
    - vs1: floating-point vector (SEW bits)
    - vd: unsigned integer vector (SEW bits)
    
    For example: f32 → u32, f64 → u64
    
    Round-to-zero (RTZ) always truncates toward zero (rounds down for positive values).
    
    This is equivalent to C-style type casting for unsigned types. Values outside the
    representable range saturate to the maximum/minimum unsigned values. Negative values
    saturate to zero. NaN converts to the maximum unsigned integer value.
    
    Maps to __riscv_vfcvt_rtz_xu_f_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfcvt.rtz.xu.f.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f32, m1>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfcvt_rtz_xu_f_v_<type><policy>
      // Example: __riscv_vfcvt_rtz_xu_f_v_u32m1_tu, __riscv_vfcvt_rtz_xu_f_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfcvt_rtz_xu_f_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtXFWOp : RVV_Op<"vfncvt.x.f.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing convert wide floating-point to narrow signed integer";
  let description = [{
    Performs element-wise conversion from wide floating-point to narrow signed integer,
    halving the element width.
    
    - vs1: wide floating-point vector (2*SEW bits)
    - vd: narrow signed integer vector (SEW bits)
    
    For example: f64 → i32, f32 → i16
    
    The conversion uses the current floating-point rounding mode (frm) from the fcsr.
    Values outside the representable range saturate to the maximum/minimum integer values.
    NaN converts to the maximum positive integer value.
    
    This is a narrowing operation where the input floating-point is twice as wide
    as the output integer.
    
    Maps to __riscv_vfncvt_x_f_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.x.f.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_x_f_w_<type><policy>
      // Example: __riscv_vfncvt_x_f_w_i32m1_tu, __riscv_vfncvt_x_f_w_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_x_f_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtXUFWOp : RVV_Op<"vfncvt.xu.f.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing convert wide floating-point to narrow unsigned integer";
  let description = [{
    Performs element-wise conversion from wide floating-point to narrow unsigned integer,
    halving the element width.
    
    - vs1: wide floating-point vector (2*SEW bits)
    - vd: narrow unsigned integer vector (SEW bits)
    
    For example: f64 → u32, f32 → u16
    
    The conversion uses the current floating-point rounding mode (frm) from the fcsr.
    Values outside the representable range saturate to the maximum/minimum unsigned values.
    Negative values saturate to zero. NaN converts to the maximum unsigned integer value.
    
    This is a narrowing operation where the input floating-point is twice as wide
    as the output integer.
    
    Maps to __riscv_vfncvt_xu_f_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.xu.f.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_xu_f_w_<type><policy>
      // Example: __riscv_vfncvt_xu_f_w_u32m1_tu, __riscv_vfncvt_xu_f_w_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_xu_f_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtRtzXFWOp : RVV_Op<"vfncvt.rtz.x.f.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing convert wide float to narrow signed int with round-to-zero";
  let description = [{
    Performs element-wise conversion from wide floating-point to narrow signed integer
    using round-to-zero (truncate) rounding mode, halving the element width.
    
    - vs1: wide floating-point vector (2*SEW bits)
    - vd: narrow signed integer vector (SEW bits)
    
    For example: f64 → i32, f32 → i16
    
    Round-to-zero (RTZ) always truncates toward zero, regardless of the current frm:
    - Positive values round down
    - Negative values round up
    
    This is equivalent to C-style type casting. Values outside the representable
    range saturate to the maximum/minimum integer values. NaN converts to the
    maximum positive integer value.
    
    This is a narrowing operation where the input floating-point is twice as wide
    as the output integer.
    
    Maps to __riscv_vfncvt_rtz_x_f_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.rtz.x.f.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_rtz_x_f_w_<type><policy>
      // Example: __riscv_vfncvt_rtz_x_f_w_i32m1_tu, __riscv_vfncvt_rtz_x_f_w_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_rtz_x_f_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFNCvtRtzXUFWOp : RVV_Op<"vfncvt.rtz.xu.f.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing convert wide float to narrow unsigned int with round-to-zero";
  let description = [{
    Performs element-wise conversion from wide floating-point to narrow unsigned integer
    using round-to-zero (truncate) rounding mode, halving the element width.
    
    - vs1: wide floating-point vector (2*SEW bits)
    - vd: narrow unsigned integer vector (SEW bits)
    
    For example: f64 → u32, f32 → u16
    
    Round-to-zero (RTZ) always truncates toward zero (rounds down for positive values),
    regardless of the current frm setting.
    
    This is equivalent to C-style type casting for unsigned types. Values outside the
    representable range saturate to the maximum/minimum unsigned values. Negative values
    saturate to zero. NaN converts to the maximum unsigned integer value.
    
    This is a narrowing operation where the input floating-point is twice as wide
    as the output integer.
    
    Maps to __riscv_vfncvt_rtz_xu_f_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vfncvt.rtz.xu.f.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<f64, m2>, i64) -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfncvt_rtz_xu_f_w_<type><policy>
      // Example: __riscv_vfncvt_rtz_xu_f_w_u32m1_tu, __riscv_vfncvt_rtz_xu_f_w_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfncvt_rtz_xu_f_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

#endif // RVV_FLOAT_OPS
