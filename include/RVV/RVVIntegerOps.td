#ifndef RVV_INTEGER_OPS
#define RVV_INTEGER_OPS

def VAddVVOp : RVV_Op<"vadd.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector integer addition";
  let description = [{
    Performs element-wise integer addition.
    Maps to vadd.vv intrinsics.
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
      // Format: __riscv_vadd_vv_<type><policy>
      // Example: __riscv_vadd_vv_i32m1_tu, __riscv_vadd_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAddVXOp : RVV_Op<"vadd.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar integer addition";
  let description = [{
    Performs element-wise integer addition with scalar.
    Maps to vadd.vx intrinsics.
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
      // Format: __riscv_vadd_vx_<type><policy>
      // Example: __riscv_vadd_vx_i32m1_tu, __riscv_vadd_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vadd_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSAddVVOp : RVV_Op<"vsadd.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector saturating integer addition";
  let description = [{
    Performs element-wise saturating integer addition.
    Maps to vsadd.vv intrinsics.
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
      // Format: __riscv_vsadd_vv_<type><policy>
      // Example: __riscv_vsadd_vv_i32m1_tu, __riscv_vsadd_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSAddVXOp : RVV_Op<"vsadd.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar saturating integer addition";
  let description = [{
    Performs element-wise saturating integer addition with scalar.
    Maps to vsadd.vx intrinsics.
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
      // Format: __riscv_vsadd_vx_<type><policy>
      // Example: __riscv_vsadd_vx_i32m1_tu, __riscv_vsadd_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsadd_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSAddUVVOp : RVV_Op<"vsaddu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned saturating integer addition";
  let description = [{
    Performs element-wise unsigned saturating integer addition.
    Maps to vsaddu.vv intrinsics.
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
      // Format: __riscv_vsaddu_vv_<type><policy>
      // Example: __riscv_vsaddu_vv_u32m1_tu, __riscv_vsaddu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsaddu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSAddUVXOp : RVV_Op<"vsaddu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned saturating integer addition";
  let description = [{
    Performs element-wise unsigned saturating integer addition with scalar.
    Maps to vsaddu.vx intrinsics.
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
      // Format: __riscv_vsaddu_vx_<type><policy>
      // Example: __riscv_vsaddu_vx_u32m1_tu, __riscv_vsaddu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsaddu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddVVOp : RVV_Op<"vwadd.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening integer addition";
  let description = [{
    Performs element-wise widening integer addition.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vwadd.vv intrinsics.
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
      // Format: __riscv_vwadd_vv_<type><policy>
      // Example: __riscv_vwadd_vv_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddVXOp : RVV_Op<"vwadd.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening integer addition";
  let description = [{
    Performs element-wise widening integer addition with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: wide scalar (2*SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vwadd.vx intrinsics.
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
      // Format: __riscv_vwadd_vx_<type><policy>
      // Example: __riscv_vwadd_vx_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwadd_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddWVOp : RVV_Op<"vwadd.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening integer addition (wide-narrow)";
  let description = [{
    Performs element-wise widening integer addition with wide first operand.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vwadd.wv intrinsics.
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
      // Format: __riscv_vwadd_wv_<type><policy>
      // Example: __riscv_vwadd_wv_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwadd_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddWXOp : RVV_Op<"vwadd.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening integer addition (wide-narrow)";
  let description = [{
    Performs element-wise widening integer addition with wide vector and narrow scalar.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vd: wide vector (2*SEW bits)
    Maps to vwadd.wx intrinsics.
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
      // Format: __riscv_vwadd_wx_<type><policy>
      // Example: __riscv_vwadd_wx_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwadd_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddUVVOp : RVV_Op<"vwaddu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening unsigned integer addition";
  let description = [{
    Performs element-wise widening unsigned integer addition.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Maps to vwaddu.vv intrinsics.
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
      // Format: __riscv_vwaddu_vv_<type><policy>
      // Example: __riscv_vwaddu_vv_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwaddu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddUVXOp : RVV_Op<"vwaddu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening unsigned integer addition";
  let description = [{
    Performs element-wise widening unsigned integer addition with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - rs1: wide unsigned scalar (2*SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Maps to vwaddu.vx intrinsics.
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
      // Format: __riscv_vwaddu_vx_<type><policy>
      // Example: __riscv_vwaddu_vx_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwaddu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddUWVOp : RVV_Op<"vwaddu.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening unsigned integer addition (wide-narrow)";
  let description = [{
    Performs element-wise widening unsigned integer addition with wide first operand.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide unsigned vector (2*SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Maps to vwaddu.wv intrinsics.
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
      // Format: __riscv_vwaddu_wv_<type><policy>
      // Example: __riscv_vwaddu_wv_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwaddu_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWAddUWXOp : RVV_Op<"vwaddu.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening unsigned integer addition (wide-narrow)";
  let description = [{
    Performs element-wise widening unsigned integer addition with wide vector and narrow scalar.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide unsigned vector (2*SEW bits)
    - rs1: narrow unsigned scalar (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Maps to vwaddu.wx intrinsics.
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
      // Format: __riscv_vwaddu_wx_<type><policy>
      // Example: __riscv_vwaddu_wx_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwaddu_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSubVVOp : RVV_Op<"vsub.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector integer subtraction";
  let description = [{
    Performs element-wise integer subtraction: vd[i] = vs2[i] - vs1[i].
    Maps to vsub.vv intrinsics.
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
      // Format: __riscv_vsub_vv_<type><policy>
      // Example: __riscv_vsub_vv_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSubVXOp : RVV_Op<"vsub.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar integer subtraction";
  let description = [{
    Performs element-wise integer subtraction with scalar: vd[i] = vs1[i] - rs1.
    Maps to vsub.vx intrinsics.
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
      // Format: __riscv_vsub_vx_<type><policy>
      // Example: __riscv_vsub_vx_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsub_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRSubVXOp : RVV_Op<"vrsub.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar integer reverse subtraction";
  let description = [{
    Performs element-wise integer reverse subtraction: vd[i] = rs1 - vs1[i].
    Maps to vrsub.vx intrinsics.
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
      // Format: __riscv_vrsub_vx_<type><policy>
      // Example: __riscv_vrsub_vx_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vrsub_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSubVVOp : RVV_Op<"vssub.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector saturating integer subtraction";
  let description = [{
    Performs element-wise saturating signed integer subtraction: vd[i] = saturate(vs2[i] - vs1[i]).
    Maps to vssub.vv intrinsics.
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
      // Format: __riscv_vssub_vv_<type><policy>
      // Example: __riscv_vssub_vv_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSubVXOp : RVV_Op<"vssub.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar saturating integer subtraction";
  let description = [{
    Performs element-wise saturating signed integer subtraction with scalar: vd[i] = saturate(vs1[i] - rs1).
    Maps to vssub.vx intrinsics.
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
      // Format: __riscv_vssub_vx_<type><policy>
      // Example: __riscv_vssub_vx_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssub_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSubUVVOp : RVV_Op<"vssubu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector saturating unsigned integer subtraction";
  let description = [{
    Performs element-wise saturating unsigned integer subtraction: vd[i] = saturate(vs2[i] - vs1[i]).
    Maps to vssubu.vv intrinsics.
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
      // Format: __riscv_vssubu_vv_<type><policy>
      // Example: __riscv_vssubu_vv_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssubu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSubUVXOp : RVV_Op<"vssubu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar saturating unsigned integer subtraction";
  let description = [{
    Performs element-wise saturating unsigned integer subtraction with scalar: vd[i] = saturate(vs1[i] - rs1).
    Maps to vssubu.vx intrinsics.
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
      // Format: __riscv_vssubu_vx_<type><policy>
      // Example: __riscv_vssubu_vx_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssubu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAAddVVOp : RVV_Op<"vaadd.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed averaging addition";
  let description = [{
    Performs element-wise signed averaging addition: vd[i] = (vs2[i] + vs1[i]) >> 1.
    The addition is performed with one extra bit of precision and then right-shifted.
    Maps to vaadd.vv intrinsics.
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
      // Format: __riscv_vaadd_vv_<type><policy>
      // Example: __riscv_vaadd_vv_i32m1_tu, __riscv_vaadd_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vaadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAAddVXOp : RVV_Op<"vaadd.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed averaging addition";
  let description = [{
    Performs element-wise signed averaging addition with scalar: vd[i] = (vs1[i] + rs1) >> 1.
    The addition is performed with one extra bit of precision and then right-shifted.
    Maps to vaadd.vx intrinsics.
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
      // Format: __riscv_vaadd_vx_<type><policy>
      // Example: __riscv_vaadd_vx_i32m1_tu, __riscv_vaadd_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vaadd_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAAddUVVOp : RVV_Op<"vaaddu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned averaging addition";
  let description = [{
    Performs element-wise unsigned averaging addition: vd[i] = (vs2[i] + vs1[i]) >> 1.
    The addition is performed with one extra bit of precision and then right-shifted.
    Maps to vaaddu.vv intrinsics.
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
      // Format: __riscv_vaaddu_vv_<type><policy>
      // Example: __riscv_vaaddu_vv_u32m1_tu, __riscv_vaaddu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vaaddu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAAddUVXOp : RVV_Op<"vaaddu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned averaging addition";
  let description = [{
    Performs element-wise unsigned averaging addition with scalar: vd[i] = (vs1[i] + rs1) >> 1.
    The addition is performed with one extra bit of precision and then right-shifted.
    Maps to vaaddu.vx intrinsics.
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
      // Format: __riscv_vaaddu_vx_<type><policy>
      // Example: __riscv_vaaddu_vx_u32m1_tu, __riscv_vaaddu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vaaddu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VASubVVOp : RVV_Op<"vasub.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed averaging subtraction";
  let description = [{
    Performs element-wise signed averaging subtraction: vd[i] = (vs2[i] - vs1[i]) >> 1.
    The subtraction is performed with one extra bit of precision and then right-shifted.
    Maps to vasub.vv intrinsics.
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
      // Format: __riscv_vasub_vv_<type><policy>
      // Example: __riscv_vasub_vv_i32m1_tu, __riscv_vasub_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vasub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VASubVXOp : RVV_Op<"vasub.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed averaging subtraction";
  let description = [{
    Performs element-wise signed averaging subtraction with scalar: vd[i] = (vs1[i] - rs1) >> 1.
    The subtraction is performed with one extra bit of precision and then right-shifted.
    Maps to vasub.vx intrinsics.
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
      // Format: __riscv_vasub_vx_<type><policy>
      // Example: __riscv_vasub_vx_i32m1_tu, __riscv_vasub_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vasub_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VASubUVVOp : RVV_Op<"vasubu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned averaging subtraction";
  let description = [{
    Performs element-wise unsigned averaging subtraction: vd[i] = (vs2[i] - vs1[i]) >> 1.
    The subtraction is performed with one extra bit of precision and then right-shifted.
    Maps to vasubu.vv intrinsics.
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
      // Format: __riscv_vasubu_vv_<type><policy>
      // Example: __riscv_vasubu_vv_u32m1_tu, __riscv_vasubu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vasubu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VASubUVXOp : RVV_Op<"vasubu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned averaging subtraction";
  let description = [{
    Performs element-wise unsigned averaging subtraction with scalar: vd[i] = (vs1[i] - rs1) >> 1.
    The subtraction is performed with one extra bit of precision and then right-shifted.
    Maps to vasubu.vx intrinsics.
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
      // Format: __riscv_vasubu_vx_<type><policy>
      // Example: __riscv_vasubu_vx_u32m1_tu, __riscv_vasubu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vasubu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubVVOp : RVV_Op<"vwsub.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening integer subtraction";
  let description = [{
    Performs element-wise widening integer subtraction.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow vector (SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Computes vd[i] = vs2[i] - vs1[i] with widening.
    Maps to vwsub.vv intrinsics.
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
      // Format: __riscv_vwsub_vv_<type><policy>
      // Example: __riscv_vwsub_vv_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubVXOp : RVV_Op<"vwsub.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening integer subtraction";
  let description = [{
    Performs element-wise widening integer subtraction with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow vector (SEW bits)
    - rs1: wide scalar (2*SEW bits)
    - vd: wide vector (2*SEW bits)
    Computes vd[i] = vs1[i] - rs1 with widening.
    Maps to vwsub.vx intrinsics.
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
      // Format: __riscv_vwsub_vx_<type><policy>
      // Example: __riscv_vwsub_vx_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsub_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubWVOp : RVV_Op<"vwsub.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening integer subtraction (wide-narrow)";
  let description = [{
    Performs element-wise widening integer subtraction with wide first operand.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - vs2: narrow vector (SEW bits)
    - vd: wide vector (2*SEW bits)
    Computes vd[i] = vs1[i] - vs2[i] with widening.
    Maps to vwsub.wv intrinsics.
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
      // Format: __riscv_vwsub_wv_<type><policy>
      // Example: __riscv_vwsub_wv_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsub_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubWXOp : RVV_Op<"vwsub.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening integer subtraction (wide-narrow)";
  let description = [{
    Performs element-wise widening integer subtraction with wide vector and narrow scalar.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide vector (2*SEW bits)
    - rs1: narrow scalar (SEW bits)
    - vd: wide vector (2*SEW bits)
    Computes vd[i] = vs1[i] - rs1 with widening.
    Maps to vwsub.wx intrinsics.
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
      // Format: __riscv_vwsub_wx_<type><policy>
      // Example: __riscv_vwsub_wx_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsub_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubUVVOp : RVV_Op<"vwsubu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening unsigned integer subtraction";
  let description = [{
    Performs element-wise widening unsigned integer subtraction.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Computes vd[i] = vs2[i] - vs1[i] with widening.
    Maps to vwsubu.vv intrinsics.
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
      // Format: __riscv_vwsubu_vv_<type><policy>
      // Example: __riscv_vwsubu_vv_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsubu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubUVXOp : RVV_Op<"vwsubu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening unsigned integer subtraction";
  let description = [{
    Performs element-wise widening unsigned integer subtraction with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - rs1: wide unsigned scalar (2*SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Computes vd[i] = vs1[i] - rs1 with widening.
    Maps to vwsubu.vx intrinsics.
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
      // Format: __riscv_vwsubu_vx_<type><policy>
      // Example: __riscv_vwsubu_vx_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsubu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubUWVOp : RVV_Op<"vwsubu.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening unsigned integer subtraction (wide-narrow)";
  let description = [{
    Performs element-wise widening unsigned integer subtraction with wide first operand.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide unsigned vector (2*SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Computes vd[i] = vs1[i] - vs2[i] with widening.
    Maps to vwsubu.wv intrinsics.
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
      // Format: __riscv_vwsubu_wv_<type><policy>
      // Example: __riscv_vwsubu_wv_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsubu_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWSubUWXOp : RVV_Op<"vwsubu.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening unsigned integer subtraction (wide-narrow)";
  let description = [{
    Performs element-wise widening unsigned integer subtraction with wide vector and narrow scalar.
    The result and operands follow wide-narrow-wide pattern:
    - vs1: wide unsigned vector (2*SEW bits)
    - rs1: narrow unsigned scalar (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Computes vd[i] = vs1[i] - rs1 with widening.
    Maps to vwsubu.wx intrinsics.
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
      // Format: __riscv_vwsubu_wx_<type><policy>
      // Example: __riscv_vwsubu_wx_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsubu_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulVVOp : RVV_Op<"vmul.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector integer multiplication";
  let description = [{
    Performs element-wise integer multiplication: vd[i] = vs2[i] * vs1[i].
    Returns the low SEW bits of the 2*SEW-bit product.
    Maps to vmul.vv intrinsics.
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
      // Format: __riscv_vmul_vv_<type><policy>
      // Example: __riscv_vmul_vv_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmul_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulVXOp : RVV_Op<"vmul.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar integer multiplication";
  let description = [{
    Performs element-wise integer multiplication with scalar: vd[i] = vs1[i] * rs1.
    Returns the low SEW bits of the 2*SEW-bit product.
    Maps to vmul.vx intrinsics.
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
      // Format: __riscv_vmul_vx_<type><policy>
      // Example: __riscv_vmul_vx_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmul_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSMulVVOp : RVV_Op<"vsmul.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed saturating multiply";
  let description = [{
    Performs element-wise signed saturating multiplication: vd[i] = saturate((vs2[i] * vs1[i]) >> SEW-1).
    Multiplies two signed integers, shifts right by SEW-1 bits, and saturates the result.
    This is useful for fixed-point multiplication where the result is in the same format as inputs.
    Maps to vsmul.vv intrinsics.
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
      // Format: __riscv_vsmul_vv_<type><policy>
      // Example: __riscv_vsmul_vv_i32m1_tu, __riscv_vsmul_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsmul_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSMulVXOp : RVV_Op<"vsmul.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed saturating multiply";
  let description = [{
    Performs element-wise signed saturating multiplication with scalar: vd[i] = saturate((vs1[i] * rs1) >> SEW-1).
    Multiplies a signed integer vector with a signed scalar, shifts right by SEW-1 bits, and saturates.
    This is useful for fixed-point multiplication where the result is in the same format as inputs.
    Maps to vsmul.vx intrinsics.
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
      // Format: __riscv_vsmul_vx_<type><policy>
      // Example: __riscv_vsmul_vx_i32m1_tu, __riscv_vsmul_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsmul_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulHVVOp : RVV_Op<"vmulh.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer multiply high";
  let description = [{
    Performs element-wise signed integer multiplication: vd[i] = (vs2[i] * vs1[i]) >> SEW.
    Returns the high SEW bits of the 2*SEW-bit signed product.
    Maps to vmulh.vv intrinsics.
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
      // Format: __riscv_vmulh_vv_<type><policy>
      // Example: __riscv_vmulh_vv_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmulh_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulHVXOp : RVV_Op<"vmulh.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer multiply high";
  let description = [{
    Performs element-wise signed integer multiplication with scalar: vd[i] = (vs1[i] * rs1) >> SEW.
    Returns the high SEW bits of the 2*SEW-bit signed product.
    Maps to vmulh.vx intrinsics.
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
      // Format: __riscv_vmulh_vx_<type><policy>
      // Example: __riscv_vmulh_vx_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmulh_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulHSUVVOp : RVV_Op<"vmulhsu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed-unsigned integer multiply high";
  let description = [{
    Performs element-wise signed-unsigned integer multiplication: vd[i] = (vs2[i] * vs1[i]) >> SEW.
    vs2 is treated as signed, vs1 is treated as unsigned.
    Returns the high SEW bits of the 2*SEW-bit product.
    Maps to vmulhsu.vv intrinsics.
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
      // Format: __riscv_vmulhsu_vv_<type><policy>
      // Example: __riscv_vmulhsu_vv_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmulhsu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulHSUVXOp : RVV_Op<"vmulhsu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed-unsigned integer multiply high";
  let description = [{
    Performs element-wise signed-unsigned integer multiplication with scalar: vd[i] = (vs1[i] * rs1) >> SEW.
    vs1 is treated as signed, rs1 is treated as unsigned.
    Returns the high SEW bits of the 2*SEW-bit product.
    Maps to vmulhsu.vx intrinsics.
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
      // Format: __riscv_vmulhsu_vx_<type><policy>
      // Example: __riscv_vmulhsu_vx_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmulhsu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulHUVVOp : RVV_Op<"vmulhu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer multiply high";
  let description = [{
    Performs element-wise unsigned integer multiplication: vd[i] = (vs2[i] * vs1[i]) >> SEW.
    Returns the high SEW bits of the 2*SEW-bit unsigned product.
    Maps to vmulhu.vv intrinsics.
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
      // Format: __riscv_vmulhu_vv_<type><policy>
      // Example: __riscv_vmulhu_vv_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmulhu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMulHUVXOp : RVV_Op<"vmulhu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer multiply high";
  let description = [{
    Performs element-wise unsigned integer multiplication with scalar: vd[i] = (vs1[i] * rs1) >> SEW.
    Returns the high SEW bits of the 2*SEW-bit unsigned product.
    Maps to vmulhu.vx intrinsics.
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
      // Format: __riscv_vmulhu_vx_<type><policy>
      // Example: __riscv_vmulhu_vx_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmulhu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMulVVOp : RVV_Op<"vwmul.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening signed integer multiplication";
  let description = [{
    Performs element-wise widening signed integer multiplication.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - vs2: narrow signed vector (SEW bits)
    - vd: wide signed vector (2*SEW bits)
    Returns the full 2*SEW-bit signed product.
    Maps to vwmul.vv intrinsics.
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
      // Format: __riscv_vwmul_vv_<type><policy>
      // Example: __riscv_vwmul_vv_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmul_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMulVXOp : RVV_Op<"vwmul.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening signed integer multiplication";
  let description = [{
    Performs element-wise widening signed integer multiplication with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - rs1: wide signed scalar (2*SEW bits)
    - vd: wide signed vector (2*SEW bits)
    Returns the full 2*SEW-bit signed product.
    Maps to vwmul.vx intrinsics.
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
      // Format: __riscv_vwmul_vx_<type><policy>
      // Example: __riscv_vwmul_vx_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmul_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMulSUVVOp : RVV_Op<"vwmulsu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening signed-unsigned integer multiplication";
  let description = [{
    Performs element-wise widening signed-unsigned integer multiplication.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - vs2: narrow signed vector (SEW bits)
    - vd: wide signed vector (2*SEW bits)
    vs2 is treated as signed, vs1 is treated as unsigned.
    Returns the full 2*SEW-bit product.
    Maps to vwmulsu.vv intrinsics.
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
      // Format: __riscv_vwmulsu_vv_<type><policy>
      // Example: __riscv_vwmulsu_vv_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmulsu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMulSUVXOp : RVV_Op<"vwmulsu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening signed-unsigned integer multiplication";
  let description = [{
    Performs element-wise widening signed-unsigned integer multiplication with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - rs1: wide unsigned scalar (2*SEW bits)
    - vd: wide signed vector (2*SEW bits)
    vs1 is treated as signed, rs1 is treated as unsigned.
    Returns the full 2*SEW-bit product.
    Maps to vwmulsu.vx intrinsics.
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
      // Format: __riscv_vwmulsu_vx_<type><policy>
      // Example: __riscv_vwmulsu_vx_i64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmulsu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMulUVVOp : RVV_Op<"vwmulu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening unsigned integer multiplication";
  let description = [{
    Performs element-wise widening unsigned integer multiplication.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Returns the full 2*SEW-bit unsigned product.
    Maps to vwmulu.vv intrinsics.
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
      // Format: __riscv_vwmulu_vv_<type><policy>
      // Example: __riscv_vwmulu_vv_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmulu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMulUVXOp : RVV_Op<"vwmulu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening unsigned integer multiplication";
  let description = [{
    Performs element-wise widening unsigned integer multiplication with scalar.
    The result and operands follow narrow-wide-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - rs1: wide unsigned scalar (2*SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    Returns the full 2*SEW-bit unsigned product.
    Maps to vwmulu.vx intrinsics.
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
      // Format: __riscv_vwmulu_vx_<type><policy>
      // Example: __riscv_vwmulu_vx_u64m2_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmulu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSeqVVOp : RVV_Op<"vmseq.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector integer equality comparison";
  let description = [{
    Performs element-wise integer equality comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] == vs2[i].
    Maps to vmseq.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmseq.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmseq_vv_<element_type><lmul><policy>
      // Example: __riscv_vmseq_vv_i32m1_b32_tu, __riscv_vmseq_vv_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmseq_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSeqVXOp : RVV_Op<"vmseq.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar integer equality comparison";
  let description = [{
    Performs element-wise integer equality comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] == rs1.
    Maps to vmseq.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmseq.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmseq_vx_<element_type><lmul><policy>
      // Example: __riscv_vmseq_vx_i32m1_b32_tu, __riscv_vmseq_vx_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmseq_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSneVVOp : RVV_Op<"vmsne.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector integer inequality comparison";
  let description = [{
    Performs element-wise integer inequality comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] != vs2[i].
    Maps to vmsne.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsne.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsne_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsne_vv_i32m1_b32_tu, __riscv_vmsne_vv_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsne_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSneVXOp : RVV_Op<"vmsne.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar integer inequality comparison";
  let description = [{
    Performs element-wise integer inequality comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] != rs1.
    Maps to vmsne.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsne.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsne_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsne_vx_i32m1_b32_tu, __riscv_vmsne_vx_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsne_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSltVVOp : RVV_Op<"vmslt.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer less than comparison";
  let description = [{
    Performs element-wise signed integer less than comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] < vs2[i] (signed).
    Maps to vmslt.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmslt.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmslt_vv_<element_type><lmul><policy>
      // Example: __riscv_vmslt_vv_i32m1_b32_tu, __riscv_vmslt_vv_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmslt_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSltVXOp : RVV_Op<"vmslt.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer less than comparison";
  let description = [{
    Performs element-wise signed integer less than comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] < rs1 (signed).
    Maps to vmslt.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmslt.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmslt_vx_<element_type><lmul><policy>
      // Example: __riscv_vmslt_vx_i32m1_b32_tu, __riscv_vmslt_vx_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmslt_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSleVVOp : RVV_Op<"vmsle.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer less than or equal comparison";
  let description = [{
    Performs element-wise signed integer less than or equal comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] <= vs2[i] (signed).
    Maps to vmsle.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsle.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsle_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsle_vv_i32m1_b32_tu, __riscv_vmsle_vv_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsle_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSleVXOp : RVV_Op<"vmsle.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer less than or equal comparison";
  let description = [{
    Performs element-wise signed integer less than or equal comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] <= rs1 (signed).
    Maps to vmsle.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsle.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsle_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsle_vx_i32m1_b32_tu, __riscv_vmsle_vx_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsle_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgtVVOp : RVV_Op<"vmsgt.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer greater than comparison";
  let description = [{
    Performs element-wise signed integer greater than comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] > vs2[i] (signed).
    Maps to vmsgt.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsgt.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsgt_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsgt_vv_i32m1_b32_tu, __riscv_vmsgt_vv_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsgt_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgtVXOp : RVV_Op<"vmsgt.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer greater than comparison";
  let description = [{
    Performs element-wise signed integer greater than comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] > rs1 (signed).
    Maps to vmsgt.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsgt.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsgt_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsgt_vx_i32m1_b32_tu, __riscv_vmsgt_vx_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsgt_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgeVVOp : RVV_Op<"vmsge.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer greater than or equal comparison";
  let description = [{
    Performs element-wise signed integer greater than or equal comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] >= vs2[i] (signed).
    Maps to vmsge.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsge.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsge_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsge_vv_i32m1_b32_tu, __riscv_vmsge_vv_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsge_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgeVXOp : RVV_Op<"vmsge.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer greater than or equal comparison";
  let description = [{
    Performs element-wise signed integer greater than or equal comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] >= rs1 (signed).
    Maps to vmsge.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsge.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsge_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsge_vx_i32m1_b32_tu, __riscv_vmsge_vx_i32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsge_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSltuVVOp : RVV_Op<"vmsltu.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer less than comparison";
  let description = [{
    Performs element-wise unsigned integer less than comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] < vs2[i] (unsigned).
    Maps to vmsltu.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsltu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsltu_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsltu_vv_u32m1_b32_tu, __riscv_vmsltu_vv_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsltu_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSltuVXOp : RVV_Op<"vmsltu.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer less than comparison";
  let description = [{
    Performs element-wise unsigned integer less than comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] < rs1 (unsigned).
    Maps to vmsltu.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsltu.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsltu_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsltu_vx_u32m1_b32_tu, __riscv_vmsltu_vx_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsltu_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSleuVVOp : RVV_Op<"vmsleu.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer less than or equal comparison";
  let description = [{
    Performs element-wise unsigned integer less than or equal comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] <= vs2[i] (unsigned).
    Maps to vmsleu.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsleu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsleu_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsleu_vv_u32m1_b32_tu, __riscv_vmsleu_vv_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsleu_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSleuVXOp : RVV_Op<"vmsleu.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer less than or equal comparison";
  let description = [{
    Performs element-wise unsigned integer less than or equal comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] <= rs1 (unsigned).
    Maps to vmsleu.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsleu.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsleu_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsleu_vx_u32m1_b32_tu, __riscv_vmsleu_vx_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsleu_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgtuVVOp : RVV_Op<"vmsgtu.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer greater than comparison";
  let description = [{
    Performs element-wise unsigned integer greater than comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] > vs2[i] (unsigned).
    Maps to vmsgtu.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsgtu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsgtu_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsgtu_vv_u32m1_b32_tu, __riscv_vmsgtu_vv_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsgtu_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgtuVXOp : RVV_Op<"vmsgtu.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer greater than comparison";
  let description = [{
    Performs element-wise unsigned integer greater than comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] > rs1 (unsigned).
    Maps to vmsgtu.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsgtu.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsgtu_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsgtu_vx_u32m1_b32_tu, __riscv_vmsgtu_vx_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsgtu_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgeuVVOp : RVV_Op<"vmsgeu.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer greater than or equal comparison";
  let description = [{
    Performs element-wise unsigned integer greater than or equal comparison between two vectors.
    Produces a mask result where each bit indicates if vs1[i] >= vs2[i] (unsigned).
    Maps to vmsgeu.vv intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsgeu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsgeu_vv_<element_type><lmul><policy>
      // Example: __riscv_vmsgeu_vv_u32m1_b32_tu, __riscv_vmsgeu_vv_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsgeu_vv_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMSgeuVXOp : RVV_Op<"vmsgeu.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer greater than or equal comparison";
  let description = [{
    Performs element-wise unsigned integer greater than or equal comparison between a vector and a scalar.
    Produces a mask result where each bit indicates if vs1[i] >= rs1 (unsigned).
    Maps to vmsgeu.vx intrinsics.
    
    Example:
    ```mlir
    %mask = rvv.vmsgeu.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.mask<n32>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl,
    Optional<RVV_MaskType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsgeu_vx_<element_type><lmul><policy>
      // Example: __riscv_vmsgeu_vx_u32m1_b32_tu, __riscv_vmsgeu_vx_u32m1_b32_tumu
      RVVPolicyInterface policyIf = *this;
      
      // For comparison ops, we need to determine the element type from the operands
      // since the result is a mask type
      auto vs1Type = ::llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      
      return "__riscv_vmsgeu_vx_" + 
             vs1Type.getVectorTypeString() + "_" +
             vdType.getMaskTypeString() +
             policyIf.getPolicySuffix();
    }
  }];
}

def VMinVVOp : RVV_Op<"vmin.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer minimum";
  let description = [{
    Performs element-wise signed integer minimum operation between two vectors.
    Returns the smaller of vs1[i] and vs2[i] using signed comparison.
    Maps to vmin.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmin.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmin_vv_<type><policy>
      // Example: __riscv_vmin_vv_i32m1_tu, __riscv_vmin_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmin_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMinVXOp : RVV_Op<"vmin.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer minimum";
  let description = [{
    Performs element-wise signed integer minimum operation between a vector and a scalar.
    Returns the smaller of vs1[i] and rs1 using signed comparison.
    Maps to vmin.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmin.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmin_vx_<type><policy>
      // Example: __riscv_vmin_vx_i32m1_tu, __riscv_vmin_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmin_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMinuVVOp : RVV_Op<"vminu.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer minimum";
  let description = [{
    Performs element-wise unsigned integer minimum operation between two vectors.
    Returns the smaller of vs1[i] and vs2[i] using unsigned comparison.
    Maps to vminu.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vminu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vminu_vv_<type><policy>
      // Example: __riscv_vminu_vv_u32m1_tu, __riscv_vminu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vminu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMinuVXOp : RVV_Op<"vminu.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer minimum";
  let description = [{
    Performs element-wise unsigned integer minimum operation between a vector and a scalar.
    Returns the smaller of vs1[i] and rs1 using unsigned comparison.
    Maps to vminu.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vminu.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vminu_vx_<type><policy>
      // Example: __riscv_vminu_vx_u32m1_tu, __riscv_vminu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vminu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMaxVVOp : RVV_Op<"vmax.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer maximum";
  let description = [{
    Performs element-wise signed integer maximum operation between two vectors.
    Returns the larger of vs1[i] and vs2[i] using signed comparison.
    Maps to vmax.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmax.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmax_vv_<type><policy>
      // Example: __riscv_vmax_vv_i32m1_tu, __riscv_vmax_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmax_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMaxVXOp : RVV_Op<"vmax.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer maximum";
  let description = [{
    Performs element-wise signed integer maximum operation between a vector and a scalar.
    Returns the larger of vs1[i] and rs1 using signed comparison.
    Maps to vmax.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmax.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmax_vx_<type><policy>
      // Example: __riscv_vmax_vx_i32m1_tu, __riscv_vmax_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmax_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMaxuVVOp : RVV_Op<"vmaxu.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer maximum";
  let description = [{
    Performs element-wise unsigned integer maximum operation between two vectors.
    Returns the larger of vs1[i] and vs2[i] using unsigned comparison.
    Maps to vmaxu.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmaxu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmaxu_vv_<type><policy>
      // Example: __riscv_vmaxu_vv_u32m1_tu, __riscv_vmaxu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmaxu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMaxuVXOp : RVV_Op<"vmaxu.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer maximum";
  let description = [{
    Performs element-wise unsigned integer maximum operation between a vector and a scalar.
    Returns the larger of vs1[i] and rs1 using unsigned comparison.
    Maps to vmaxu.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmaxu.vx %vs1, %rs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i32, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmaxu_vx_<type><policy>
      // Example: __riscv_vmaxu_vx_u32m1_tu, __riscv_vmaxu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmaxu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNegOp : RVV_Op<"vneg.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector integer negate";
  let description = [{
    Performs element-wise two's complement negation of the source vector.
    Computes vd[i] = -vs1[i] for each element.
    Maps to vneg.v intrinsics (pseudo-instruction for vrsub.vx with x0).
    
    Example:
    ```mlir
    %result = rvv.vneg.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<ta>,
      mask_policy = #rvv.mask_policy<ma>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vneg_v_<type><policy>
      // Example: __riscv_vneg_v_i32m1_tu, __riscv_vneg_v_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vneg_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAdcVVMOp : RVV_Op<"vadc.vvm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorVectorMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector integer add-with-carry";
  let description = [{
    Performs element-wise integer addition with carry-in from mask register.
    Computes vd[i] = vs2[i] + vs1[i] + vs3[i] where vs3 is the carry-in mask.
    Maps to vadc.vvm intrinsics.
    
    Note: vadc does not support tail/mask policies as it always operates on
    all elements and the mask operand (vs3) is used as carry-in, not for masking.
    
    Example:
    ```mlir
    %result = rvv.vadc.vvm %vs1, %vs2, %carry_in, %vl 
      : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.mask<m1>, i64) 
      -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vadc_vvm_<type>
      // Example: __riscv_vadc_vvm_i32m1
      return "__riscv_vadc_vvm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VAdcVXMOp : RVV_Op<"vadc.vxm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorScalarMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar integer add-with-carry";
  let description = [{
    Performs element-wise integer addition with scalar and carry-in from mask register.
    Computes vd[i] = vs2[i] + rs1 + vs3[i] where vs3 is the carry-in mask.
    Maps to vadc.vxm intrinsics.
    
    Note: vadc does not support tail/mask policies as it always operates on
    all elements and the mask operand (vs3) is used as carry-in, not for masking.
    
    Example:
    ```mlir
    %result = rvv.vadc.vxm %vs2, %rs1, %carry_in, %vl 
      : (!rvv.vector<i32, m1>, i32, !rvv.mask<m1>, i64) 
      -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vadc_vxm_<type>
      // Example: __riscv_vadc_vxm_i32m1
      return "__riscv_vadc_vxm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VMadcVVMOp : RVV_Op<"vmadc.vvm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorVectorMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-vector integer add-with-carry-in, produce carry-out";
  let description = [{
    Performs element-wise integer addition with carry-in and produces carry-out mask.
    Computes carry_out[i] = carry_out(vs2[i] + vs1[i] + vs3[i]) where vs3 is the carry-in mask.
    Maps to vmadc.vvm intrinsics.
    
    This operation produces a mask result indicating whether a carry occurred at each element position.
    
    Example:
    ```mlir
    %carry_out = rvv.vmadc.vvm %vs1, %vs2, %carry_in, %vl 
      : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.mask<m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmadc_vvm_<type>
      // Example: __riscv_vmadc_vvm_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmadc_vvm" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMadcVXMOp : RVV_Op<"vmadc.vxm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorScalarMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-scalar integer add-with-carry-in, produce carry-out";
  let description = [{
    Performs element-wise integer addition with scalar and carry-in, produces carry-out mask.
    Computes carry_out[i] = carry_out(vs2[i] + rs1 + vs3[i]) where vs3 is the carry-in mask.
    Maps to vmadc.vxm intrinsics.
    
    This operation produces a mask result indicating whether a carry occurred at each element position.
    
    Example:
    ```mlir
    %carry_out = rvv.vmadc.vxm %vs2, %rs1, %carry_in, %vl 
      : (!rvv.vector<i32, m1>, i32, !rvv.mask<m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmadc_vxm_<type>
      // Example: __riscv_vmadc_vxm_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmadc_vxm" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMadcVVOp : RVV_Op<"vmadc.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-vector integer add, produce carry-out";
  let description = [{
    Performs element-wise integer addition and produces carry-out mask.
    Computes carry_out[i] = carry_out(vs2[i] + vs1[i]).
    Maps to vmadc.vv intrinsics.
    
    This operation produces a mask result indicating whether a carry occurred at each element position.
    Unlike vmadc.vvm, this variant does not take a carry-in.
    
    Example:
    ```mlir
    %carry_out = rvv.vmadc.vv %vs1, %vs2, %vl 
      : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmadc_vv_<type>
      // Example: __riscv_vmadc_vv_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmadc_vv" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMadcVXOp : RVV_Op<"vmadc.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-scalar integer add, produce carry-out";
  let description = [{
    Performs element-wise integer addition with scalar and produces carry-out mask.
    Computes carry_out[i] = carry_out(vs2[i] + rs1).
    Maps to vmadc.vx intrinsics.
    
    This operation produces a mask result indicating whether a carry occurred at each element position.
    Unlike vmadc.vxm, this variant does not take a carry-in.
    
    Example:
    ```mlir
    %carry_out = rvv.vmadc.vx %vs2, %rs1, %vl 
      : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmadc_vx_<type>
      // Example: __riscv_vmadc_vx_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmadc_vx" + vs1Type.getVectorTypeString();
    }
  }];
}

def VSbcVVMOp : RVV_Op<"vsbc.vvm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorVectorMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector integer subtract-with-borrow";
  let description = [{
    Performs element-wise integer subtraction with borrow-in from mask register.
    Computes vd[i] = vs2[i] - vs1[i] - vs3[i] where vs3 is the borrow-in mask.
    Maps to vsbc.vvm intrinsics.
    
    Note: vsbc does not support tail/mask policies as it always operates on
    all elements and the mask operand (vs3) is used as borrow-in, not for masking.
    
    Example:
    ```mlir
    %result = rvv.vsbc.vvm %vs1, %vs2, %borrow_in, %vl 
      : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.mask<m1>, i64) 
      -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vsbc_vvm_<type>
      // Example: __riscv_vsbc_vvm_i32m1
      return "__riscv_vsbc_vvm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VSbcVXMOp : RVV_Op<"vsbc.vxm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorScalarMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar integer subtract-with-borrow";
  let description = [{
    Performs element-wise integer subtraction with scalar and borrow-in from mask register.
    Computes vd[i] = vs2[i] - rs1 - vs3[i] where vs3 is the borrow-in mask.
    Maps to vsbc.vxm intrinsics.
    
    Note: vsbc does not support tail/mask policies as it always operates on
    all elements and the mask operand (vs3) is used as borrow-in, not for masking.
    
    Example:
    ```mlir
    %result = rvv.vsbc.vxm %vs2, %rs1, %borrow_in, %vl 
      : (!rvv.vector<i32, m1>, i32, !rvv.mask<m1>, i64) 
      -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vsbc_vxm_<type>
      // Example: __riscv_vsbc_vxm_i32m1
      return "__riscv_vsbc_vxm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VMsbcVVMOp : RVV_Op<"vmsbc.vvm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorVectorMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-vector integer subtract-with-borrow-in, produce borrow-out";
  let description = [{
    Performs element-wise integer subtraction with borrow-in and produces borrow-out mask.
    Computes borrow_out[i] = borrow_out(vs2[i] - vs1[i] - vs3[i]) where vs3 is the borrow-in mask.
    Maps to vmsbc.vvm intrinsics.
    
    This operation produces a mask result indicating whether a borrow occurred at each element position.
    
    Example:
    ```mlir
    %borrow_out = rvv.vmsbc.vvm %vs1, %vs2, %borrow_in, %vl 
      : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.mask<m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsbc_vvm_<type>
      // Example: __riscv_vmsbc_vvm_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmsbc_vvm" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMsbcVXMOp : RVV_Op<"vmsbc.vxm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorScalarMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-scalar integer subtract-with-borrow-in, produce borrow-out";
  let description = [{
    Performs element-wise integer subtraction with scalar and borrow-in, produces borrow-out mask.
    Computes borrow_out[i] = borrow_out(vs2[i] - rs1 - vs3[i]) where vs3 is the borrow-in mask.
    Maps to vmsbc.vxm intrinsics.
    
    This operation produces a mask result indicating whether a borrow occurred at each element position.
    
    Example:
    ```mlir
    %borrow_out = rvv.vmsbc.vxm %vs2, %rs1, %borrow_in, %vl 
      : (!rvv.vector<i32, m1>, i32, !rvv.mask<m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    RVV_MaskType:$vs3,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsbc_vxm_<type>
      // Example: __riscv_vmsbc_vxm_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmsbc_vxm" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMsbcVVOp : RVV_Op<"vmsbc.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-vector integer subtract, produce borrow-out";
  let description = [{
    Performs element-wise integer subtraction and produces borrow-out mask.
    Computes borrow_out[i] = borrow_out(vs2[i] - vs1[i]).
    Maps to vmsbc.vv intrinsics.
    
    This operation produces a mask result indicating whether a borrow occurred at each element position.
    Unlike vmsbc.vvm, this variant does not take a borrow-in.
    
    Example:
    ```mlir
    %borrow_out = rvv.vmsbc.vv %vs1, %vs2, %vl 
      : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsbc_vv_<type>
      // Example: __riscv_vmsbc_vv_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmsbc_vv" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMsbcVXOp : RVV_Op<"vmsbc.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>
  ]> {
  let summary = "Vector-scalar integer subtract, produce borrow-out";
  let description = [{
    Performs element-wise integer subtraction with scalar and produces borrow-out mask.
    Computes borrow_out[i] = borrow_out(vs2[i] - rs1).
    Maps to vmsbc.vx intrinsics.
    
    This operation produces a mask result indicating whether a borrow occurred at each element position.
    Unlike vmsbc.vxm, this variant does not take a borrow-in.
    
    Example:
    ```mlir
    %borrow_out = rvv.vmsbc.vx %vs2, %rs1, %vl 
      : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_ScalarType:$rs1,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsbc_vx_<type>
      // Example: __riscv_vmsbc_vx_i32m1
      auto vs1Type = llvm::cast<::mlir::rvv::VectorType>(getVs1().getType());
      return "__riscv_vmsbc_vx" + vs1Type.getVectorTypeString();
    }
  }];
}

def VMAccVVOp : RVV_Op<"vmacc.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector integer multiply-accumulate";
  let description = [{
    Performs element-wise integer multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vmacc.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vmacc.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmacc_vv_<type><policy>
      // Example: __riscv_vmacc_vv_i32m1_tu, __riscv_vmacc_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmacc_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMAccVXOp : RVV_Op<"vmacc.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar integer multiply-accumulate";
  let description = [{
    Performs element-wise integer multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vmacc.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vmacc.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmacc_vx_<type><policy>
      // Example: __riscv_vmacc_vx_i32m1_tu, __riscv_vmacc_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmacc_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNMSacVVOp : RVV_Op<"vnmsac.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector integer negated multiply-subtract-accumulate";
  let description = [{
    Performs element-wise integer negated multiply-subtract-accumulate: vd[i] = -(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vnmsac.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vnmsac.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnmsac_vv_<type><policy>
      // Example: __riscv_vnmsac_vv_i32m1_tu, __riscv_vnmsac_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnmsac_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNMSacVXOp : RVV_Op<"vnmsac.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar integer negated multiply-subtract-accumulate";
  let description = [{
    Performs element-wise integer negated multiply-subtract-accumulate with scalar: vd[i] = -(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the accumulator input.
    Maps to vnmsac.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vnmsac.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnmsac_vx_<type><policy>
      // Example: __riscv_vnmsac_vx_i32m1_tu, __riscv_vnmsac_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnmsac_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMAddVVOp : RVV_Op<"vmadd.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector integer multiply-add";
  let description = [{
    Performs element-wise integer multiply-add: vd[i] = (vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is one of the multiplicands.
    Maps to vmadd.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as one of the multiplicands.
    
    Example:
    ```mlir
    %result = rvv.vmadd.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmadd_vv_<type><policy>
      // Example: __riscv_vmadd_vv_i32m1_tu, __riscv_vmadd_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmadd_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMAddVXOp : RVV_Op<"vmadd.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar integer multiply-add";
  let description = [{
    Performs element-wise integer multiply-add with scalar: vd[i] = (vs3[i] * rs1) + vs1[i].
    This is a destructive operation where vs3 is one of the multiplicands.
    Maps to vmadd.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as one of the multiplicands.
    
    Example:
    ```mlir
    %result = rvv.vmadd.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vmadd_vx_<type><policy>
      // Example: __riscv_vmadd_vx_i32m1_tu, __riscv_vmadd_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vmadd_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNMSubVVOp : RVV_Op<"vnmsub.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector integer negated multiply-subtract";
  let description = [{
    Performs element-wise integer negated multiply-subtract: vd[i] = -(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is one of the multiplicands.
    Maps to vnmsub.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as one of the multiplicands.
    
    Example:
    ```mlir
    %result = rvv.vnmsub.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnmsub_vv_<type><policy>
      // Example: __riscv_vnmsub_vv_i32m1_tu, __riscv_vnmsub_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnmsub_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNMSubVXOp : RVV_Op<"vnmsub.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar integer negated multiply-subtract";
  let description = [{
    Performs element-wise integer negated multiply-subtract with scalar: vd[i] = -(vs3[i] * rs1) + vs1[i].
    This is a destructive operation where vs3 is one of the multiplicands.
    Maps to vnmsub.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as one of the multiplicands.
    
    Example:
    ```mlir
    %result = rvv.vnmsub.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnmsub_vx_<type><policy>
      // Example: __riscv_vnmsub_vx_i32m1_tu, __riscv_vnmsub_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnmsub_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccVVOp : RVV_Op<"vwmacc.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening signed integer multiply-accumulate";
  let description = [{
    Performs element-wise widening signed integer multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - vs2: narrow signed vector (SEW bits)
    - vs3: wide signed vector (2*SEW bits) - accumulator
    - vd: wide signed vector (2*SEW bits)
    Maps to vwmacc.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmacc.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmacc_vv_<type><policy>
      // Example: __riscv_vwmacc_vv_i64m2_tu, __riscv_vwmacc_vv_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmacc_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccVXOp : RVV_Op<"vwmacc.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening signed integer multiply-accumulate";
  let description = [{
    Performs element-wise widening signed integer multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - rs1: narrow signed scalar (SEW bits)
    - vs3: wide signed vector (2*SEW bits) - accumulator
    - vd: wide signed vector (2*SEW bits)
    Maps to vwmacc.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmacc.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmacc_vx_<type><policy>
      // Example: __riscv_vwmacc_vx_i64m2_tu, __riscv_vwmacc_vx_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmacc_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccSUVVOp : RVV_Op<"vwmaccsu.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening signed-unsigned integer multiply-accumulate";
  let description = [{
    Performs element-wise widening signed-unsigned integer multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vs3: wide signed vector (2*SEW bits) - accumulator
    - vd: wide signed vector (2*SEW bits)
    vs1 is treated as signed, vs2 is treated as unsigned.
    Maps to vwmaccsu.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmaccsu.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmaccsu_vv_<type><policy>
      // Example: __riscv_vwmaccsu_vv_i64m2_tu, __riscv_vwmaccsu_vv_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmaccsu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccSUVXOp : RVV_Op<"vwmaccsu.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening signed-unsigned integer multiply-accumulate";
  let description = [{
    Performs element-wise widening signed-unsigned integer multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow signed vector (SEW bits)
    - rs1: narrow unsigned scalar (SEW bits)
    - vs3: wide signed vector (2*SEW bits) - accumulator
    - vd: wide signed vector (2*SEW bits)
    vs1 is treated as signed, rs1 is treated as unsigned.
    Maps to vwmaccsu.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmaccsu.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmaccsu_vx_<type><policy>
      // Example: __riscv_vwmaccsu_vx_i64m2_tu, __riscv_vwmaccsu_vx_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmaccsu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccUSVVOp : RVV_Op<"vwmaccus.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening unsigned-signed integer multiply-accumulate";
  let description = [{
    Performs element-wise widening unsigned-signed integer multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - vs2: narrow signed vector (SEW bits)
    - vs3: wide signed vector (2*SEW bits) - accumulator
    - vd: wide signed vector (2*SEW bits)
    vs1 is treated as unsigned, vs2 is treated as signed.
    Maps to vwmaccus.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmaccus.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmaccus_vv_<type><policy>
      // Example: __riscv_vwmaccus_vv_i64m2_tu, __riscv_vwmaccus_vv_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmaccus_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccUSVXOp : RVV_Op<"vwmaccus.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening unsigned-signed integer multiply-accumulate";
  let description = [{
    Performs element-wise widening unsigned-signed integer multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - rs1: narrow signed scalar (SEW bits)
    - vs3: wide signed vector (2*SEW bits) - accumulator
    - vd: wide signed vector (2*SEW bits)
    vs1 is treated as unsigned, rs1 is treated as signed.
    Maps to vwmaccus.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmaccus.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmaccus_vx_<type><policy>
      // Example: __riscv_vwmaccus_vx_i64m2_tu, __riscv_vwmaccus_vx_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmaccus_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccUVVOp : RVV_Op<"vwmaccu.vv", [
  DeclareOpInterfaceMethods<RVVVectorVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>
  ]> {
  let summary = "Vector-vector widening unsigned integer multiply-accumulate";
  let description = [{
    Performs element-wise widening unsigned integer multiply-accumulate: vd[i] = +(vs1[i] * vs2[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - vs2: narrow unsigned vector (SEW bits)
    - vs3: wide unsigned vector (2*SEW bits) - accumulator
    - vd: wide unsigned vector (2*SEW bits)
    Maps to vwmaccu.vv intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmaccu.vv %vs1, %vs2, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmaccu_vv_<type><policy>
      // Example: __riscv_vwmaccu_vv_i64m2_tu, __riscv_vwmaccu_vv_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmaccu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWMAccUVXOp : RVV_Op<"vwmaccu.vx", [
  DeclareOpInterfaceMethods<RVVVectorScalarVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVPolicyInterface, ["getMask", "getTailPolicy", "getMaskPolicy", "getPolicySuffix"]>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar widening unsigned integer multiply-accumulate";
  let description = [{
    Performs element-wise widening unsigned integer multiply-accumulate with scalar: vd[i] = +(rs1 * vs1[i]) + vs3[i].
    This is a destructive operation where vs3 is the wide accumulator input.
    The result and operands follow narrow-narrow-wide pattern:
    - vs1: narrow unsigned vector (SEW bits)
    - rs1: narrow unsigned scalar (SEW bits)
    - vs3: wide unsigned vector (2*SEW bits) - accumulator
    - vd: wide unsigned vector (2*SEW bits)
    Maps to vwmaccu.vx intrinsics.
    
    Note: Unlike most RVV operations, this does not support passthru semantics
    because vs3 serves as the accumulator input.
    
    Example:
    ```mlir
    %result = rvv.vwmaccu.vx %vs1, %rs1, %vs3, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwmaccu_vx_<type><policy>
      // Example: __riscv_vwmaccu_vx_i64m2_tu, __riscv_vwmaccu_vx_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwmaccu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VDivVVOp : RVV_Op<"vdiv.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer division";
  let description = [{
    Performs element-wise signed integer division: vd[i] = vs2[i] / vs1[i].
    Maps to vdiv.vv intrinsics.
    
    Note: Division by zero and signed overflow (INT_MIN / -1) behavior 
    follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vdiv.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vdiv_vv_<type><policy>
      // Example: __riscv_vdiv_vv_i32m1_tu, __riscv_vdiv_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vdiv_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VDivVXOp : RVV_Op<"vdiv.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer division";
  let description = [{
    Performs element-wise signed integer division with scalar: vd[i] = vs2[i] / rs1.
    Maps to vdiv.vx intrinsics.
    
    Note: Division by zero and signed overflow (INT_MIN / -1) behavior 
    follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vdiv.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vdiv_vx_<type><policy>
      // Example: __riscv_vdiv_vx_i32m1_tu, __riscv_vdiv_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vdiv_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VDivUVVOp : RVV_Op<"vdivu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer division";
  let description = [{
    Performs element-wise unsigned integer division: vd[i] = vs2[i] / vs1[i].
    Maps to vdivu.vv intrinsics.
    
    Note: Division by zero behavior follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vdivu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vdivu_vv_<type><policy>
      // Example: __riscv_vdivu_vv_u32m1_tu, __riscv_vdivu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vdivu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VDivUVXOp : RVV_Op<"vdivu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer division";
  let description = [{
    Performs element-wise unsigned integer division with scalar: vd[i] = vs2[i] / rs1.
    Maps to vdivu.vx intrinsics.
    
    Note: Division by zero behavior follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vdivu.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vdivu_vx_<type><policy>
      // Example: __riscv_vdivu_vx_u32m1_tu, __riscv_vdivu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vdivu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRemVVOp : RVV_Op<"vrem.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector signed integer remainder";
  let description = [{
    Performs element-wise signed integer remainder: vd[i] = vs2[i] % vs1[i].
    Maps to vrem.vv intrinsics.
    
    Note: Division by zero and signed overflow (INT_MIN % -1) behavior 
    follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vrem.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vrem_vv_<type><policy>
      // Example: __riscv_vrem_vv_i32m1_tu, __riscv_vrem_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vrem_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRemVXOp : RVV_Op<"vrem.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar signed integer remainder";
  let description = [{
    Performs element-wise signed integer remainder with scalar: vd[i] = vs2[i] % rs1.
    Maps to vrem.vx intrinsics.
    
    Note: Division by zero and signed overflow (INT_MIN % -1) behavior 
    follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vrem.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vrem_vx_<type><policy>
      // Example: __riscv_vrem_vx_i32m1_tu, __riscv_vrem_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vrem_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRemUVVOp : RVV_Op<"vremu.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector unsigned integer remainder";
  let description = [{
    Performs element-wise unsigned integer remainder: vd[i] = vs2[i] % vs1[i].
    Maps to vremu.vv intrinsics.
    
    Note: Division by zero behavior follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vremu.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vremu_vv_<type><policy>
      // Example: __riscv_vremu_vv_u32m1_tu, __riscv_vremu_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vremu_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRemUVXOp : RVV_Op<"vremu.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar unsigned integer remainder";
  let description = [{
    Performs element-wise unsigned integer remainder with scalar: vd[i] = vs2[i] % rs1.
    Maps to vremu.vx intrinsics.
    
    Note: Division by zero behavior follows RISC-V specification.
    
    Example:
    ```mlir
    %result = rvv.vremu.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vremu_vx_<type><policy>
      // Example: __riscv_vremu_vx_u32m1_tu, __riscv_vremu_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vremu_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSExtOp : RVV_Op<"vsext", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector sign-extend";
  let description = [{
    Performs element-wise sign-extension, widening the element width by a factor.
    Each SEW-bit signed element is sign-extended to (SEW * factor) bits.
    The factor must be 2, 4, or 8.
    Maps to vsext.vf2, vsext.vf4, or vsext.vf8 intrinsics depending on the factor.
    
    Example:
    ```mlir
    %result = rvv.vsext %vs1, %vl {
      factor = 2 : i32,
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<i64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    I32Attr:$factor,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vsext_vf<factor>_<type><policy>
      // Example: __riscv_vsext_vf2_i64m2_tu, __riscv_vsext_vf4_i64m4_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsext_vf" + std::to_string(getFactor()) + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VZExtOp : RVV_Op<"vzext", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector zero-extend";
  let description = [{
    Performs element-wise zero-extension, widening the element width by a factor.
    Each SEW-bit unsigned element is zero-extended to (SEW * factor) bits.
    The factor must be 2, 4, or 8.
    Maps to vzext.vf2, vzext.vf4, or vzext.vf8 intrinsics depending on the factor.
    
    Example:
    ```mlir
    %result = rvv.vzext %vs1, %vl {
      factor = 2 : i32,
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<i64, m2>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl,
    I32Attr:$factor,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vzext_vf<factor>_<type><policy>
      // Example: __riscv_vzext_vf2_u64m2_tu, __riscv_vzext_vf4_u64m4_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vzext_vf" + std::to_string(getFactor()) + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedSumVSOp : RVV_Op<"vredsum.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector integer sum reduction";
  let description = [{
    Performs a sum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredsum.vs intrinsics.
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
      // Format: __riscv_vredsum_vs_<type><policy>
      // Example: __riscv_vredsum_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredsum_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWRedSumVSOp : RVV_Op<"vwredsum.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector widening signed integer sum reduction";
  let description = [{
    Performs a widening sum reduction of the signed vector elements.
    The scalar input vs1 is the initial accumulator value (2*SEW bits).
    The result is 2*SEW bits wide.
    Maps to vwredsum.vs intrinsics.
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
      // Format: __riscv_vwredsum_vs_<type><policy>
      // Example: __riscv_vwredsum_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwredsum_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWRedSumUVSOp : RVV_Op<"vwredsumu.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector widening unsigned integer sum reduction";
  let description = [{
    Performs a widening sum reduction of the unsigned vector elements.
    The scalar input vs1 is the initial accumulator value (2*SEW bits).
    The result is 2*SEW bits wide.
    Maps to vwredsumu.vs intrinsics.
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
      // Format: __riscv_vwredsumu_vs_<type><policy>
      // Example: __riscv_vwredsumu_vs_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwredsumu_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedAndVSOp : RVV_Op<"vredand.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector bitwise AND reduction";
  let description = [{
    Performs a bitwise AND reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredand.vs intrinsics.
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
      // Format: __riscv_vredand_vs_<type><policy>
      // Example: __riscv_vredand_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredand_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedOrVSOp : RVV_Op<"vredor.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector bitwise OR reduction";
  let description = [{
    Performs a bitwise OR reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredor.vs intrinsics.
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
      // Format: __riscv_vredor_vs_<type><policy>
      // Example: __riscv_vredor_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredor_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedXorVSOp : RVV_Op<"vredxor.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector bitwise XOR reduction";
  let description = [{
    Performs a bitwise XOR reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredxor.vs intrinsics.
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
      // Format: __riscv_vredxor_vs_<type><policy>
      // Example: __riscv_vredxor_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredxor_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedMaxVSOp : RVV_Op<"vredmax.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector signed maximum reduction";
  let description = [{
    Performs a signed maximum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredmax.vs intrinsics.
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
      // Format: __riscv_vredmax_vs_<type><policy>
      // Example: __riscv_vredmax_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredmax_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedMaxUVSOp : RVV_Op<"vredmaxu.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector unsigned maximum reduction";
  let description = [{
    Performs an unsigned maximum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredmaxu.vs intrinsics.
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
      // Format: __riscv_vredmaxu_vs_<type><policy>
      // Example: __riscv_vredmaxu_vs_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredmaxu_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedMinVSOp : RVV_Op<"vredmin.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector signed minimum reduction";
  let description = [{
    Performs a signed minimum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredmin.vs intrinsics.
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
      // Format: __riscv_vredmin_vs_<type><policy>
      // Example: __riscv_vredmin_vs_i32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredmin_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRedMinUVSOp : RVV_Op<"vredminu.vs", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector unsigned minimum reduction";
  let description = [{
    Performs an unsigned minimum reduction of the vector elements.
    The scalar input vs1 is the initial accumulator value.
    Maps to vredminu.vs intrinsics.
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
      // Format: __riscv_vredminu_vs_<type><policy>
      // Example: __riscv_vredminu_vs_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vredminu_vs" + 
             getVs2().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VmAndMMOp : RVV_Op<"vmand.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask AND operation";
  let description = [{
    Performs bitwise AND operation on two mask registers.
    vd[i] = vs1[i] & vs2[i]
    Maps to vmand.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmand.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmand_mm_<mask_type>
      // Example: __riscv_vmand_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmand_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmAndNMMOp : RVV_Op<"vmandn.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask AND-NOT operation";
  let description = [{
    Performs bitwise AND-NOT operation on two mask registers.
    vd[i] = vs1[i] & ~vs2[i]
    Maps to vmandn.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmandn.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmandn_mm_<mask_type>
      // Example: __riscv_vmandn_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmandn_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmNandMMOp : RVV_Op<"vmnand.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask NAND operation";
  let description = [{
    Performs bitwise NAND operation on two mask registers.
    vd[i] = ~(vs1[i] & vs2[i])
    Maps to vmnand.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmnand.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmnand_mm_<mask_type>
      // Example: __riscv_vmnand_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmnand_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmNorMMOp : RVV_Op<"vmnor.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask NOR operation";
  let description = [{
    Performs bitwise NOR operation on two mask registers.
    vd[i] = ~(vs1[i] | vs2[i])
    Maps to vmnor.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmnor.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmnor_mm_<mask_type>
      // Example: __riscv_vmnor_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmnor_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmOrMMOp : RVV_Op<"vmor.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask OR operation";
  let description = [{
    Performs bitwise OR operation on two mask registers.
    vd[i] = vs1[i] | vs2[i]
    Maps to vmor.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmor.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmor_mm_<mask_type>
      // Example: __riscv_vmor_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmor_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmOrNMMOp : RVV_Op<"vmorn.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask OR-NOT operation";
  let description = [{
    Performs bitwise OR-NOT operation on two mask registers.
    vd[i] = vs1[i] | ~vs2[i]
    Maps to vmorn.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmorn.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmorn_mm_<mask_type>
      // Example: __riscv_vmorn_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmorn_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmXnorMMOp : RVV_Op<"vmxnor.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask XNOR operation";
  let description = [{
    Performs bitwise XNOR operation on two mask registers.
    vd[i] = ~(vs1[i] ^ vs2[i])
    Maps to vmxnor.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmxnor.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmxnor_mm_<mask_type>
      // Example: __riscv_vmxnor_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmxnor_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmXorMMOp : RVV_Op<"vmxor.mm", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask-mask XOR operation";
  let description = [{
    Performs bitwise XOR operation on two mask registers.
    vd[i] = vs1[i] ^ vs2[i]
    Maps to vmxor.mm intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmxor.mm %vs1, %vs2, %vl : (!rvv.mask<m1>, !rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    RVV_MaskType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmxor_mm_<mask_type>
      // Example: __riscv_vmxor_mm_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmxor_mm_" + vdType.getMaskTypeString();
    }
  }];
}

def VmNotMOp : RVV_Op<"vmnot.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask NOT operation";
  let description = [{
    Performs bitwise NOT operation on a mask register.
    vd[i] = ~vs1[i]
    This is a pseudo-instruction that is actually implemented as vmnand.mm vd, vs1, vs1.
    Maps to vmnot.m pseudo-intrinsic.
    
    Example:
    ```mlir
    %result = rvv.vmnot.m %vs1, %vl : (!rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmnot_m_<mask_type>
      // Example: __riscv_vmnot_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmnot_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VFirstMOp : RVV_Op<"vfirst.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Find-first-set mask bit";
  let description = [{
    Finds the index of the first set bit in a mask register.
    Returns -1 if no bits are set.
    Maps to vfirst.m intrinsics.
    
    Example:
    ```mlir
    %index = rvv.vfirst.m %vs1, %vl, %mask : (!rvv.mask<m1>, i64, !rvv.mask<m1>) -> i64
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs I64:$rd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vfirst_m_<mask_type>
      // Example: __riscv_vfirst_m_b32
      auto vs1Type = ::llvm::cast<::mlir::rvv::MaskType>(getVs1().getType());
      return "__riscv_vfirst_m_" + vs1Type.getMaskTypeString();
    }
  }];
}

def VCPopMOp : RVV_Op<"vcpop.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Count population of mask bits";
  let description = [{
    Counts the number of set bits in a mask register.
    Maps to vcpop.m intrinsics.
    
    Example:
    ```mlir
    %count = rvv.vcpop.m %vs1, %vl, %mask : (!rvv.mask<m1>, i64, !rvv.mask<m1>) -> i64
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs I64:$rd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vcpop_m_<mask_type>
      // Example: __riscv_vcpop_m_b32
      auto vs1Type = ::llvm::cast<::mlir::rvv::MaskType>(getVs1().getType());
      return "__riscv_vcpop_m_" + vs1Type.getMaskTypeString();
    }
  }];
}

def VMSbfMOp : RVV_Op<"vmsbf.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Set-before-first mask bit";
  let description = [{
    Sets all mask bits before the first set bit (found set bit becomes 0).
    Example: 00100 -> 00011
    Maps to vmsbf.m intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmsbf.m %vs1, %vl, %mask : (!rvv.mask<m1>, i64, !rvv.mask<m1>) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsbf_m_<mask_type>
      // Example: __riscv_vmsbf_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmsbf_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VMSifMOp : RVV_Op<"vmsif.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Set-including-first mask bit";
  let description = [{
    Sets all mask bits up to and including the first set bit.
    Example: 00100 -> 00111
    Maps to vmsif.m intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmsif.m %vs1, %vl, %mask : (!rvv.mask<m1>, i64, !rvv.mask<m1>) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsif_m_<mask_type>
      // Example: __riscv_vmsif_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmsif_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VMSofMOp : RVV_Op<"vmsof.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Set-only-first mask bit";
  let description = [{
    Sets only the first set bit, clearing all others.
    Example: 00100 -> 00100, 01100 -> 00100
    Maps to vmsof.m intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vmsof.m %vs1, %vl, %mask : (!rvv.mask<m1>, i64, !rvv.mask<m1>) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl,
    Optional<RVV_MaskType>:$mask
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmsof_m_<mask_type>
      // Example: __riscv_vmsof_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmsof_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VMMvMOp : RVV_Op<"vmmv.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskOpInterface>,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Mask move operation";
  let description = [{
    Copies a mask register to another mask register.
    This is a pseudo-instruction that is actually implemented as vmand.mm vd, vs, vs.
    Maps to vmmv.m pseudo-intrinsic.
    
    Example:
    ```mlir
    %result = rvv.vmmv.m %vs1, %vl : (!rvv.mask<m1>, i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$vs1,
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmmv_m_<mask_type>
      // Example: __riscv_vmmv_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmmv_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VMSetMOp : RVV_Op<"vmset.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Set mask register to all ones";
  let description = [{
    Sets all bits in a mask register to 1.
    This is a pseudo-instruction that is actually implemented as vmxnor.mm vd, vd, vd.
    Maps to vmset.m pseudo-intrinsic.
    
    Example:
    ```mlir
    %result = rvv.vmset.m %vl : (i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmset_m_<mask_type>
      // Example: __riscv_vmset_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmset_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VMClrMOp : RVV_Op<"vmclr.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVMaskResultInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Clear mask register to all zeros";
  let description = [{
    Clears all bits in a mask register to 0.
    This is a pseudo-instruction that is actually implemented as vmxor.mm vd, vd, vd.
    Maps to vmclr.m pseudo-intrinsic.
    
    Example:
    ```mlir
    %result = rvv.vmclr.m %vl : (i64) -> !rvv.mask<m1>
    ```
  }];
  let arguments = (ins
    I64:$vl
  );
  let results = (outs RVV_MaskType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmclr_m_<mask_type>
      // Example: __riscv_vmclr_m_b32
      auto vdType = ::llvm::cast<::mlir::rvv::MaskType>(getVd().getType());
      return "__riscv_vmclr_m_" + vdType.getMaskTypeString();
    }
  }];
}

def VSllVVOp : RVV_Op<"vsll.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector logical left shift";
  let description = [{
    Performs element-wise logical left shift: vd[i] = vs2[i] << (vs1[i] & (SEW-1)).
    The shift amount is taken from the lower log2(SEW) bits of vs1[i].
    Maps to vsll.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vsll.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vsll_vv_<type><policy>
      // Example: __riscv_vsll_vv_i32m1_tu, __riscv_vsll_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsll_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSllVXOp : RVV_Op<"vsll.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar logical left shift";
  let description = [{
    Performs element-wise logical left shift with scalar: vd[i] = vs2[i] << (rs1 & (SEW-1)).
    The shift amount is taken from the lower log2(SEW) bits of rs1.
    Maps to vsll.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vsll.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vsll_vx_<type><policy>
      // Example: __riscv_vsll_vx_i32m1_tu, __riscv_vsll_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsll_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSrlVVOp : RVV_Op<"vsrl.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector logical right shift";
  let description = [{
    Performs element-wise logical right shift: vd[i] = vs2[i] >> (vs1[i] & (SEW-1)).
    The shift amount is taken from the lower log2(SEW) bits of vs1[i].
    Zero-extends (unsigned shift).
    Maps to vsrl.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vsrl.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vsrl_vv_<type><policy>
      // Example: __riscv_vsrl_vv_u32m1_tu, __riscv_vsrl_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsrl_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSrlVXOp : RVV_Op<"vsrl.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar logical right shift";
  let description = [{
    Performs element-wise logical right shift with scalar: vd[i] = vs2[i] >> (rs1 & (SEW-1)).
    The shift amount is taken from the lower log2(SEW) bits of rs1.
    Zero-extends (unsigned shift).
    Maps to vsrl.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vsrl.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vsrl_vx_<type><policy>
      // Example: __riscv_vsrl_vx_u32m1_tu, __riscv_vsrl_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsrl_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSraVVOp : RVV_Op<"vsra.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector arithmetic right shift";
  let description = [{
    Performs element-wise arithmetic right shift: vd[i] = vs2[i] >> (vs1[i] & (SEW-1)).
    The shift amount is taken from the lower log2(SEW) bits of vs1[i].
    Sign-extends (signed shift).
    Maps to vsra.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vsra.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vsra_vv_<type><policy>
      // Example: __riscv_vsra_vv_i32m1_tu, __riscv_vsra_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsra_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSraVXOp : RVV_Op<"vsra.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar arithmetic right shift";
  let description = [{
    Performs element-wise arithmetic right shift with scalar: vd[i] = vs2[i] >> (rs1 & (SEW-1)).
    The shift amount is taken from the lower log2(SEW) bits of rs1.
    Sign-extends (signed shift).
    Maps to vsra.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vsra.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vsra_vx_<type><policy>
      // Example: __riscv_vsra_vx_i32m1_tu, __riscv_vsra_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vsra_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSrlVVOp : RVV_Op<"vssrl.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector saturating logical right shift";
  let description = [{
    Performs element-wise saturating logical right shift with rounding:
    vd[i] = round(vs2[i] >> (vs1[i] & (SEW-1))).
    
    The shift includes rounding based on the current vxrm rounding mode.
    This instruction is useful for fixed-point arithmetic where scaled values
    need to be shifted with proper rounding.
    
    Maps to vssrl.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vssrl.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vssrl_vv_<type><policy>
      // Example: __riscv_vssrl_vv_u32m1_tu, __riscv_vssrl_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssrl_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSrlVXOp : RVV_Op<"vssrl.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar saturating logical right shift";
  let description = [{
    Performs element-wise saturating logical right shift with rounding and scalar shift amount:
    vd[i] = round(vs2[i] >> (rs1 & (SEW-1))).
    
    The shift includes rounding based on the current vxrm rounding mode.
    This instruction is useful for fixed-point arithmetic where scaled values
    need to be shifted with proper rounding.
    
    Maps to vssrl.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vssrl.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vssrl_vx_<type><policy>
      // Example: __riscv_vssrl_vx_u32m1_tu, __riscv_vssrl_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssrl_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSraVVOp : RVV_Op<"vssra.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector saturating arithmetic right shift";
  let description = [{
    Performs element-wise saturating arithmetic right shift with rounding:
    vd[i] = round(vs2[i] >> (vs1[i] & (SEW-1))).
    
    The shift includes rounding based on the current vxrm rounding mode.
    Sign-extends (signed shift).
    This instruction is useful for fixed-point arithmetic where signed scaled values
    need to be shifted with proper rounding.
    
    Maps to vssra.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vssra.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vssra_vv_<type><policy>
      // Example: __riscv_vssra_vv_i32m1_tu, __riscv_vssra_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssra_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSSraVXOp : RVV_Op<"vssra.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar saturating arithmetic right shift";
  let description = [{
    Performs element-wise saturating arithmetic right shift with rounding and scalar shift amount:
    vd[i] = round(vs2[i] >> (rs1 & (SEW-1))).
    
    The shift includes rounding based on the current vxrm rounding mode.
    Sign-extends (signed shift).
    This instruction is useful for fixed-point arithmetic where signed scaled values
    need to be shifted with proper rounding.
    
    Maps to vssra.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vssra.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vssra_vx_<type><policy>
      // Example: __riscv_vssra_vx_i32m1_tu, __riscv_vssra_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vssra_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWsllVVOp : RVV_Op<"vwsll.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector widening logical left shift";
  let description = [{
    Performs element-wise widening logical left shift: vd[i] = (2*SEW)vs2[i] << (vs1[i] & (SEW-1)).
    The result is 2*SEW bits wide, the source operands are SEW bits wide.
    Maps to vwsll.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vwsll.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwsll_vv_<type><policy>
      // Example: __riscv_vwsll_vv_i64m2_tu, __riscv_vwsll_vv_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsll_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWsllVXOp : RVV_Op<"vwsll.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar widening logical left shift";
  let description = [{
    Performs element-wise widening logical left shift with scalar: vd[i] = (2*SEW)vs2[i] << (rs1 & (SEW-1)).
    The result is 2*SEW bits wide, the source vector is SEW bits wide.
    Maps to vwsll.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vwsll.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwsll_vx_<type><policy>
      // Example: __riscv_vwsll_vx_i64m2_tu, __riscv_vwsll_vx_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwsll_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNsrlWVOp : RVV_Op<"vnsrl.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector narrowing logical right shift";
  let description = [{
    Performs element-wise narrowing logical right shift: vd[i] = vs2[i] >> (vs1[i] & (2*SEW-1)).
    The source vs2 is 2*SEW bits wide, the result is SEW bits wide.
    Zero-extends (unsigned shift).
    Maps to vnsrl.wv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnsrl.wv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnsrl_wv_<type><policy>
      // Example: __riscv_vnsrl_wv_u32m1_tu, __riscv_vnsrl_wv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnsrl_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNsrlWXOp : RVV_Op<"vnsrl.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar narrowing logical right shift";
  let description = [{
    Performs element-wise narrowing logical right shift with scalar: vd[i] = vs2[i] >> (rs1 & (2*SEW-1)).
    The source vs2 is 2*SEW bits wide, the result is SEW bits wide.
    Zero-extends (unsigned shift).
    Maps to vnsrl.wx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnsrl.wx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnsrl_wx_<type><policy>
      // Example: __riscv_vnsrl_wx_u32m1_tu, __riscv_vnsrl_wx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnsrl_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNsraWVOp : RVV_Op<"vnsra.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector narrowing arithmetic right shift";
  let description = [{
    Performs element-wise narrowing arithmetic right shift: vd[i] = vs2[i] >> (vs1[i] & (2*SEW-1)).
    The source vs2 is 2*SEW bits wide, the result is SEW bits wide.
    Sign-extends (signed shift).
    Maps to vnsra.wv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnsra.wv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnsra_wv_<type><policy>
      // Example: __riscv_vnsra_wv_i32m1_tu, __riscv_vnsra_wv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnsra_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNsraWXOp : RVV_Op<"vnsra.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar narrowing arithmetic right shift";
  let description = [{
    Performs element-wise narrowing arithmetic right shift with scalar: vd[i] = vs2[i] >> (rs1 & (2*SEW-1)).
    The source vs2 is 2*SEW bits wide, the result is SEW bits wide.
    Sign-extends (signed shift).
    Maps to vnsra.wx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnsra.wx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnsra_wx_<type><policy>
      // Example: __riscv_vnsra_wx_i32m1_tu, __riscv_vnsra_wx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnsra_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNClipWVOp : RVV_Op<"vnclip.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector narrowing signed saturating clip";
  let description = [{
    Performs element-wise narrowing signed saturating right shift with rounding:
    vd[i] = saturate(round(vs2[i] >> (vs1[i] & (2*SEW-1)))).
    
    The source vs2 is 2*SEW bits wide (signed), the result is SEW bits wide (signed).
    The shift includes rounding based on the current vxrm rounding mode.
    Results are saturated to the destination's signed range.
    
    Maps to vnclip.wv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnclip.wv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnclip_wv_<type><policy>
      // Example: __riscv_vnclip_wv_i32m1_tu, __riscv_vnclip_wv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnclip_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNClipWXOp : RVV_Op<"vnclip.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar narrowing signed saturating clip";
  let description = [{
    Performs element-wise narrowing signed saturating right shift with rounding and scalar shift amount:
    vd[i] = saturate(round(vs2[i] >> (rs1 & (2*SEW-1)))).
    
    The source vs2 is 2*SEW bits wide (signed), the result is SEW bits wide (signed).
    The shift includes rounding based on the current vxrm rounding mode.
    Results are saturated to the destination's signed range.
    
    Maps to vnclip.wx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnclip.wx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnclip_wx_<type><policy>
      // Example: __riscv_vnclip_wx_i32m1_tu, __riscv_vnclip_wx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnclip_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNClipUWVOp : RVV_Op<"vnclipu.wv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector narrowing unsigned saturating clip";
  let description = [{
    Performs element-wise narrowing unsigned saturating right shift with rounding:
    vd[i] = saturate(round(vs2[i] >> (vs1[i] & (2*SEW-1)))).
    
    The source vs2 is 2*SEW bits wide (unsigned), the result is SEW bits wide (unsigned).
    The shift includes rounding based on the current vxrm rounding mode.
    Results are saturated to the destination's unsigned range [0, 2^SEW - 1].
    
    Maps to vnclipu.wv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnclipu.wv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i64, m2>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnclipu_wv_<type><policy>
      // Example: __riscv_vnclipu_wv_u32m1_tu, __riscv_vnclipu_wv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnclipu_wv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNClipUWXOp : RVV_Op<"vnclipu.wx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar narrowing unsigned saturating clip";
  let description = [{
    Performs element-wise narrowing unsigned saturating right shift with rounding and scalar shift amount:
    vd[i] = saturate(round(vs2[i] >> (rs1 & (2*SEW-1)))).
    
    The source vs2 is 2*SEW bits wide (unsigned), the result is SEW bits wide (unsigned).
    The shift includes rounding based on the current vxrm rounding mode.
    Results are saturated to the destination's unsigned range [0, 2^SEW - 1].
    
    Maps to vnclipu.wx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vnclipu.wx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnclipu_wx_<type><policy>
      // Example: __riscv_vnclipu_wx_u32m1_tu, __riscv_vnclipu_wx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnclipu_wx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAndVVOp : RVV_Op<"vand.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector bitwise AND";
  let description = [{
    Performs element-wise bitwise AND: vd[i] = vs2[i] & vs1[i].
    Maps to vand.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vand.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vand_vv_<type><policy>
      // Example: __riscv_vand_vv_i32m1_tu, __riscv_vand_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vand_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAndVXOp : RVV_Op<"vand.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar bitwise AND";
  let description = [{
    Performs element-wise bitwise AND with scalar: vd[i] = vs2[i] & rs1.
    Maps to vand.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vand.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vand_vx_<type><policy>
      // Example: __riscv_vand_vx_i32m1_tu, __riscv_vand_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vand_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VOrVVOp : RVV_Op<"vor.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector bitwise OR";
  let description = [{
    Performs element-wise bitwise OR: vd[i] = vs2[i] | vs1[i].
    Maps to vor.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vor.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vor_vv_<type><policy>
      // Example: __riscv_vor_vv_i32m1_tu, __riscv_vor_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vor_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VOrVXOp : RVV_Op<"vor.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar bitwise OR";
  let description = [{
    Performs element-wise bitwise OR with scalar: vd[i] = vs2[i] | rs1.
    Maps to vor.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vor.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vor_vx_<type><policy>
      // Example: __riscv_vor_vx_i32m1_tu, __riscv_vor_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vor_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VXorVVOp : RVV_Op<"vxor.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector bitwise XOR";
  let description = [{
    Performs element-wise bitwise XOR: vd[i] = vs2[i] ^ vs1[i].
    Maps to vxor.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vxor.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vxor_vv_<type><policy>
      // Example: __riscv_vxor_vv_i32m1_tu, __riscv_vxor_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vxor_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VXorVXOp : RVV_Op<"vxor.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar bitwise XOR";
  let description = [{
    Performs element-wise bitwise XOR with scalar: vd[i] = vs2[i] ^ rs1.
    Maps to vxor.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vxor.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vxor_vx_<type><policy>
      // Example: __riscv_vxor_vx_i32m1_tu, __riscv_vxor_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vxor_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAndNVVOp : RVV_Op<"vandn.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector bitwise AND-NOT";
  let description = [{
    Performs element-wise bitwise AND-NOT: vd[i] = vs2[i] & ~vs1[i].
    Maps to vandn.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vandn.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vandn_vv_<type><policy>
      // Example: __riscv_vandn_vv_i32m1_tu, __riscv_vandn_vv_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vandn_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VAndNVXOp : RVV_Op<"vandn.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar bitwise AND-NOT";
  let description = [{
    Performs element-wise bitwise AND-NOT with scalar: vd[i] = vs2[i] & ~rs1.
    Maps to vandn.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vandn.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vandn_vx_<type><policy>
      // Example: __riscv_vandn_vx_i32m1_tu, __riscv_vandn_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vandn_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNotVOp : RVV_Op<"vnot.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector bitwise NOT";
  let description = [{
    Performs element-wise bitwise NOT: vd[i] = ~vs1[i].
    This is a pseudo-instruction implemented as vxor.vi vd, vs1, -1.
    Maps to vnot.v pseudo-intrinsic.
    
    Example:
    ```mlir
    %result = rvv.vnot.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vnot_v_<type><policy>
      // Example: __riscv_vnot_v_i32m1_tu, __riscv_vnot_v_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vnot_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VClzVOp : RVV_Op<"vclz.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector count leading zeros";
  let description = [{
    Counts the number of leading zero bits in each element.
    Maps to vclz.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vclz.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vclz_v_<type><policy>
      // Example: __riscv_vclz_v_u32m1_tu, __riscv_vclz_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vclz_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VCtzVOp : RVV_Op<"vctz.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector count trailing zeros";
  let description = [{
    Counts the number of trailing zero bits in each element.
    Maps to vctz.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vctz.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vctz_v_<type><policy>
      // Example: __riscv_vctz_v_u32m1_tu, __riscv_vctz_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vctz_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VCpopVOp : RVV_Op<"vcpop.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector count population (popcount)";
  let description = [{
    Counts the number of set bits (1s) in each element.
    Maps to vcpop.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vcpop.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vcpop_v_<type><policy>
      // Example: __riscv_vcpop_v_u32m1_tu, __riscv_vcpop_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vcpop_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VBrev8VOp : RVV_Op<"vbrev8.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector reverse bits in bytes";
  let description = [{
    Reverses the bits in each byte of each element.
    For example, byte 0b10110010 becomes 0b01001101.
    Maps to vbrev8.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vbrev8.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vbrev8_v_<type><policy>
      // Example: __riscv_vbrev8_v_u32m1_tu, __riscv_vbrev8_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vbrev8_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VBrevVOp : RVV_Op<"vbrev.v", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector reverse bits in elements";
  let description = [{
    Reverses all bits in each element.
    For a 32-bit element, bit 0 and bit 31 are swapped, bit 1 and bit 30 are swapped, etc.
    Maps to vbrev.v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vbrev.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vbrev_v_<type><policy>
      // Example: __riscv_vbrev_v_u32m1_tu, __riscv_vbrev_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vbrev_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRorVVOp : RVV_Op<"vror.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector rotate right";
  let description = [{
    Performs element-wise rotate right: vd[i] = rotate_right(vs2[i], vs1[i] & (SEW-1)).
    The rotation amount is taken from the lower log2(SEW) bits of vs1[i].
    Maps to vror.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vror.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vror_vv_<type><policy>
      // Example: __riscv_vror_vv_u32m1_tu, __riscv_vror_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vror_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRorVXOp : RVV_Op<"vror.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar rotate right";
  let description = [{
    Performs element-wise rotate right with scalar: vd[i] = rotate_right(vs2[i], rs1 & (SEW-1)).
    The rotation amount is taken from the lower log2(SEW) bits of rs1.
    Maps to vror.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vror.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vror_vx_<type><policy>
      // Example: __riscv_vror_vx_u32m1_tu, __riscv_vror_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vror_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRolVVOp : RVV_Op<"vrol.vv", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-vector rotate left";
  let description = [{
    Performs element-wise rotate left: vd[i] = rotate_left(vs2[i], vs1[i] & (SEW-1)).
    The rotation amount is taken from the lower log2(SEW) bits of vs1[i].
    Maps to vrol.vv intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vrol.vv %vs1, %vs2, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vrol_vv_<type><policy>
      // Example: __riscv_vrol_vv_u32m1_tu, __riscv_vrol_vv_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vrol_vv" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRolVXOp : RVV_Op<"vrol.vx", [
  Pure, 
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>, 
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector-scalar rotate left";
  let description = [{
    Performs element-wise rotate left with scalar: vd[i] = rotate_left(vs2[i], rs1 & (SEW-1)).
    The rotation amount is taken from the lower log2(SEW) bits of rs1.
    Maps to vrol.vx intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vrol.vx %vs2, %rs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i32, i64) 
      -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vrol_vx_<type><policy>
      // Example: __riscv_vrol_vx_u32m1_tu, __riscv_vrol_vx_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vrol_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VRGatherVVOp : RVV_Op<"vrgather.vv", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVVectorVectorOpInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector register gather (vector-vector)";
  let description = [{
    Performs a vector register gather operation, reordering elements from a source
    vector according to indices specified in an index vector.
    
    For each element i: vd[i] = (vs2[i] >= VLMAX) ? 0 : vs1[vs2[i]]
    
    Arguments:
    - `vs1`: Source vector from which elements are gathered
    - `vs2`: Index vector specifying which elements to gather (must be unsigned integer type)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are gathered
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector with gathered elements
    
    Lowering Rules:
    - If index vector has uint16 element type: lowers to `vrgatherei16_vv_<type><policy>`
    - Otherwise: lowers to `vrgather_vv_<type><policy>`
    - Policy suffixes `_tu`, `_ta`, `_mu`, `_ma` are appended based on tail/mask policies
    - Masked variants append `_m` and include mask operand
    
    Example:
    ```mlir
    // Standard gather with same-width indices
    %result = rvv.vrgather.vv %src, %indices, %vl : !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, i64
    
    // Gather with 16-bit indices (maps to vrgatherei16)
    %result = rvv.vrgather.vv %src, %indices16, %vl : !rvv.vector<i32, m1>, !rvv.vector<i16, m1>, i64
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
      // Determine if we need vrgather or vrgatherei16 based on index element type
      auto indexVectorType = getVs2().getType();
      auto indexElementType = indexVectorType.getElementType();
      
      RVVPolicyInterface policyIf = *this;
      
      // Check if index is uint16 (i16 in MLIR)
      if (indexElementType.isUnsignedInteger(16) || 
          (indexElementType.isInteger(16) && indexVectorType.getElementWidth() == 16)) {
        // Format: __riscv_vrgatherei16_vv_<type><policy>
        return "__riscv_vrgatherei16_vv" + 
               getVd().getType().getVectorTypeString() + 
               policyIf.getPolicySuffix();
      } else {
        // Format: __riscv_vrgather_vv_<type><policy>
        return "__riscv_vrgather_vv" + 
               getVd().getType().getVectorTypeString() + 
               policyIf.getPolicySuffix();
      }
    }
  }];
}

def VRGatherVXOp : RVV_Op<"vrgather.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector register gather (vector-scalar)";
  let description = [{
    Performs a vector register gather operation, setting all elements from a source
    vector to the element at the scalar index position.
    
    For each element i: vd[i] = (rs1 >= VLMAX) ? 0 : vs1[rs1]
    
    Arguments:
    - `vs1`: Source vector from which element is gathered
    - `rs1`: Scalar index specifying which element to gather (must be unsigned integer)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are gathered
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector with gathered element replicated
    
    Lowering Rules:
    - Lowers to `vrgather_vx_<type><policy>`
    - Policy suffixes `_tu`, `_ta`, `_mu`, `_ma` are appended based on tail/mask policies
    - Masked variants append `_m` and include mask operand
    
    Example:
    ```mlir
    %result = rvv.vrgather.vx %src, %index, %vl : !rvv.vector<i32, m1>, i64, i64
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
      // Format: __riscv_vrgather_vx_<type><policy>
      // Example: __riscv_vrgather_vx_i32m1_tu, __riscv_vrgather_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vrgather_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSlideDownVXOp : RVV_Op<"vslidedown.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector slide down by scalar offset";
  let description = [{
    Slides vector elements down by a scalar offset.
    Maps to vslidedown.vx intrinsics.
    
    For each element i: vd[i] = (i + rs1 < vl) ? vs1[i + rs1] : 0
    
    Arguments:
    - `vs1`: Source vector to slide
    - `rs1`: Scalar offset (number of positions to slide down)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vslidedown.vx %src, %offset, %vl : !rvv.vector<i32, m1>, i64, i64
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
      // Format: __riscv_vslidedown_vx_<type><policy>
      // Example: __riscv_vslidedown_vx_i32m1_tu, __riscv_vslidedown_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vslidedown_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSlide1DownVXOp : RVV_Op<"vfslide1down.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector slide1 down with scalar fill";
  let description = [{
    Slides vector elements down by 1 position, filling with scalar value.
    Maps to vfslide1down.vx intrinsics.
    
    For each element i: vd[i] = (i == vl-1) ? rs1 : vs1[i + 1]
    
    Arguments:
    - `vs1`: Source vector to slide
    - `rs1`: Scalar value to insert at position vl-1
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vfslide1down.vx %src, %scalar, %vl : !rvv.vector<i32, m1>, i32, i64
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
      // Format: __riscv_vfslide1down_vx_<type><policy>
      // Example: __riscv_vfslide1down_vx_i32m1_tu, __riscv_vfslide1down_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfslide1down_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VSlideUpVXOp : RVV_Op<"vslideup.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector slide up by scalar offset";
  let description = [{
    Slides vector elements up by a scalar offset.
    Maps to vslideup.vx intrinsics.
    
    For each element i: vd[i] = (i < rs1) ? vd[i] : vs1[i - rs1]
    
    Arguments:
    - `vs1`: Source vector to slide
    - `rs1`: Scalar offset (number of positions to slide up)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vslideup.vx %src, %offset, %vl : !rvv.vector<i32, m1>, i64, i64
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
      // Format: __riscv_vslideup_vx_<type><policy>
      // Example: __riscv_vslideup_vx_i32m1_tu, __riscv_vslideup_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vslideup_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VFSlide1UpVXOp : RVV_Op<"vfslide1up.vx", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVectorScalarOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector slide1 up with scalar fill";
  let description = [{
    Slides vector elements up by 1 position, filling with scalar value.
    Maps to vfslide1up.vx intrinsics.
    
    For each element i: vd[i] = (i == 0) ? rs1 : vs1[i - 1]
    
    Arguments:
    - `vs1`: Source vector to slide
    - `rs1`: Scalar value to insert at position 0
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are processed
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector with slid elements
    
    Example:
    ```mlir
    %result = rvv.vfslide1up.vx %src, %scalar, %vl : !rvv.vector<i32, m1>, i32, i64
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
      // Format: __riscv_vfslide1up_vx_<type><policy>
      // Example: __riscv_vfslide1up_vx_i32m1_tu, __riscv_vfslide1up_vx_i32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vfslide1up_vx" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VCompressVMOp : RVV_Op<"vcompress.vm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector compress instruction";
  let description = [{
    Compresses active elements from a source vector based on a mask.
    Only elements whose corresponding mask bit is 1 are written to contiguous
    positions in the destination vector, starting from position 0.
    
    For each active element (mask[i] == 1): vd[offset++] = vs1[i]
    
    Arguments:
    - `vs1`: Source vector to compress
    - `mask`: Mask specifying which elements to compress (must be provided)
    - `vl`: Vector length - number of elements to process
    
    Results:
    - `vd`: Result vector with compressed elements
    
    Example:
    ```mlir
    %result = rvv.vcompress.vm %src, %mask, %vl : !rvv.vector<i32, m1>, !rvv.mask<m1>, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_MaskType:$mask,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vcompress_vm_<type>
      // Example: __riscv_vcompress_vm_i32m1
      return "__riscv_vcompress_vm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VMvXSOp : RVV_Op<"vmv.x.s", [Pure]> {
  let summary = "Move vector element to scalar register";
  let description = [{
    Moves the first element (element 0) from a vector register to a scalar
    integer register.
    
    rd = vs1[0]
    
    Arguments:
    - `vs1`: Source vector
    
    Results:
    - `rd`: Scalar result (integer type matching vector element type)
    
    Example:
    ```mlir
    %scalar = rvv.vmv.x.s %vec : !rvv.vector<i32, m1> -> i32
    ```
  }];
  let arguments = (ins RVV_VectorType:$vs1);
  let results = (outs RVV_ScalarType:$rd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmv_x_s_<type>
      // Example: __riscv_vmv_x_s_i32m1
      return "__riscv_vmv_x_s" + 
             getVs1().getType().getVectorTypeString();
    }
  }];
}

def VLMulExtVOp : RVV_Op<"vlmul_ext_v", [Pure]> {
  let summary = "Vector LMUL extension";
  let description = [{
    Extends a vector register group to a larger LMUL (Length Multiplier).
    This operation takes a vector with smaller LMUL and produces a vector
    with larger LMUL, effectively widening the register group.
    
    For example, extending from m1 to m2, m2 to m4, etc.
    The operation preserves the element values in the lower portion and
    the upper portion behavior depends on the implementation.
    
    Arguments:
    - `vs1`: Source vector with smaller LMUL
    
    Results:
    - `vd`: Result vector with larger LMUL
    
    Example:
    ```mlir
    %result = rvv.vlmul_ext_v %src : !rvv.vector<i32, m1> -> !rvv.vector<i32, m2>
    %result2 = rvv.vlmul_ext_v %src : !rvv.vector<i32, m2> -> !rvv.vector<i32, m4>
    ```
  }];
  let arguments = (ins RVV_VectorType:$vs1);
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vlmul_ext_v_<src_type>_<dst_type>
      // Example: __riscv_vlmul_ext_v_i32m1_i32m2
      return "__riscv_vlmul_ext_v" + 
             getVs1().getType().getVectorTypeString() +
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VLMulTruncOp : RVV_Op<"vlmul_trunc", [Pure]> {
  let summary = "Vector LMUL truncation";
  let description = [{
    Truncates a vector register group to a smaller LMUL (Length Multiplier).
    This operation takes a vector with larger LMUL and produces a vector
    with smaller LMUL, effectively narrowing the register group.
    
    For example, truncating from m4 to m2, m2 to m1, etc.
    The operation extracts the lower portion of the register group.
    
    Arguments:
    - `vs1`: Source vector with larger LMUL
    
    Results:
    - `vd`: Result vector with smaller LMUL
    
    Example:
    ```mlir
    %result = rvv.vlmul_trunc %src : !rvv.vector<i32, m4> -> !rvv.vector<i32, m2>
    %result2 = rvv.vlmul_trunc %src : !rvv.vector<i32, m2> -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins RVV_VectorType:$vs1);
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vlmul_trunc_v_<src_type>_<dst_type>
      // Example: __riscv_vlmul_trunc_v_i32m4_i32m2
      return "__riscv_vlmul_trunc_v" + 
             getVs1().getType().getVectorTypeString() +
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VMergeVVMOp : RVV_Op<"vmerge.vvm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-vector merge with mask";
  let description = [{
    Merges two vectors based on a mask. For each element position, if the mask
    bit is 1, the element is taken from vs1, otherwise from vs2.
    
    For each element i: vd[i] = mask[i] ? vs1[i] : vs2[i]
    
    Arguments:
    - `vs1`: First source vector (selected when mask bit is 1)
    - `vs2`: Second source vector (selected when mask bit is 0)
    - `mask`: Mask controlling the merge (must be provided)
    - `vl`: Vector length - number of elements to process
    
    Results:
    - `vd`: Result vector with merged elements
    
    Example:
    ```mlir
    %result = rvv.vmerge.vvm %vs1, %vs2, %mask, %vl : !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.mask<m1>, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$mask,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmerge_vvm_<type>
      // Example: __riscv_vmerge_vvm_i32m1
      return "__riscv_vmerge_vvm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VMergeVXMOp : RVV_Op<"vmerge.vxm", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-scalar merge with mask";
  let description = [{
    Merges a scalar value and a vector based on a mask. For each element position,
    if the mask bit is 1, the element is set to the scalar value, otherwise taken from vs2.
    
    For each element i: vd[i] = mask[i] ? rs1 : vs2[i]
    
    Arguments:
    - `rs1`: Scalar value (selected when mask bit is 1)
    - `vs2`: Source vector (selected when mask bit is 0)
    - `mask`: Mask controlling the merge (must be provided)
    - `vl`: Vector length - number of elements to process
    
    Results:
    - `vd`: Result vector with merged elements
    
    Example:
    ```mlir
    %result = rvv.vmerge.vxm %scalar, %vs2, %mask, %vl : i32, !rvv.vector<i32, m1>, !rvv.mask<m1>, i64
    ```
  }];
  let arguments = (ins
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs2,
    RVV_MaskType:$mask,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmerge_vxm_<type>
      // Example: __riscv_vmerge_vxm_i32m1
      return "__riscv_vmerge_vxm" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VMvVVOp : RVV_Op<"vmv.v.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector-to-vector move";
  let description = [{
    Copies elements from one vector register to another vector register.
    This is a simple vector copy operation.
    
    For each element i (where i < vl): vd[i] = vs1[i]
    
    Arguments:
    - `vs1`: Source vector to copy
    - `vl`: Vector length - number of elements to copy
    
    Results:
    - `vd`: Destination vector with copied elements
    
    Example:
    ```mlir
    %result = rvv.vmv.v.v %src, %vl : !rvv.vector<i32, m1>, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmv_v_v_<type>
      // Example: __riscv_vmv_v_v_i32m1
      return "__riscv_vmv_v_v" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VSetOp : RVV_Op<"vset", [Pure]> {
  let summary = "Set a sub-vector within a vector register group";
  let description = [{
    Sets (replaces) a portion of a vector register group with another vector.
    The index parameter specifies which portion of the register group to set.
    
    This is typically used with larger LMUL groups (m2, m4, m8) to set one
    of the constituent sub-vectors.
    
    Arguments:
    - `vd_in`: Input vector register group (destination)
    - `vs1`: Source vector to insert
    - `index`: Index indicating which sub-vector position to set (compile-time constant)
    
    Results:
    - `vd`: Result vector register group with the sub-vector updated
    
    Example:
    ```mlir
    // Set the second m1 vector within an m2 group
    %result = rvv.vset %vgroup, %vsub, 1 : !rvv.vector<i32, m2>, !rvv.vector<i32, m1>, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vd_in,
    RVV_VectorType:$vs1,
    I64Attr:$index
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vset_v_<sub_type>_<group_type>
      // Example: __riscv_vset_v_i32m1_i32m2
      return "__riscv_vset_v" + 
             getVs1().getType().getVectorTypeString() +
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VGetOp : RVV_Op<"vget", [Pure]> {
  let summary = "Get a sub-vector from a vector register group";
  let description = [{
    Extracts a portion from a vector register group. The index parameter
    specifies which portion of the register group to extract.
    
    This is typically used with larger LMUL groups (m2, m4, m8) to extract
    one of the constituent sub-vectors.
    
    Arguments:
    - `vs1`: Source vector register group
    - `index`: Index indicating which sub-vector position to get (compile-time constant)
    
    Results:
    - `vd`: Extracted sub-vector
    
    Example:
    ```mlir
    // Get the second m1 vector from an m2 group
    %result = rvv.vget %vgroup, 1 : !rvv.vector<i32, m2>, i64 -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_VectorType:$vs1,
    I64Attr:$index
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vget_v_<sub_type>_<group_type>
      // Example: __riscv_vget_v_i32m1_i32m2
      return "__riscv_vget_v" + 
             getVd().getType().getVectorTypeString() +
             getVs1().getType().getVectorTypeString();
    }
  }];
}

def VCreateOp : RVV_Op<"vcreate", [Pure]> {
  let summary = "Create a vector register group from sub-vectors";
  let description = [{
    Creates a vector register group by combining multiple smaller vectors.
    This is the inverse of vget - it assembles a larger LMUL group from
    individual sub-vectors.
    
    The number of operands depends on the LMUL ratio:
    - m2 from m1: 2 operands
    - m4 from m1: 4 operands
    - m8 from m1: 8 operands
    - m4 from m2: 2 operands
    - etc.
    
    Arguments:
    - `sources`: Variable number of source vectors to combine
    
    Results:
    - `vd`: Result vector register group
    
    Example:
    ```mlir
    // Create an m2 from two m1 vectors
    %result = rvv.vcreate %v0, %v1 : !rvv.vector<i32, m1>, !rvv.vector<i32, m1> -> !rvv.vector<i32, m2>
    
    // Create an m4 from four m1 vectors
    %result = rvv.vcreate %v0, %v1, %v2, %v3 : !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i32, m1>, !rvv.vector<i32, m1> -> !rvv.vector<i32, m4>
    ```
  }];
  let arguments = (ins
    Variadic<RVV_VectorType>:$sources
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vcreate_v_<sub_type>_<group_type>
      // Example: __riscv_vcreate_v_i32m1_i32m2
      if (getSources().size() > 0) {
        auto srcType = llvm::cast<VectorType>(getSources()[0].getType());
        return "__riscv_vcreate_v" + 
               srcType.getVectorTypeString() +
               getVd().getType().getVectorTypeString();
      }
      return "__riscv_vcreate_v";
    }
  }];
}

def VMvVXOp : RVV_Op<"vmv.v.x", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Move scalar to vector";
  let description = [{
    Moves (broadcasts) a scalar integer value to all elements of a vector register.
    Each element in the destination vector is set to the scalar value.
    
    For each element i (where i < vl): vd[i] = rs1
    
    Arguments:
    - `rs1`: Scalar integer value to broadcast
    - `vl`: Vector length - number of elements to set
    
    Results:
    - `vd`: Destination vector with all elements set to the scalar value
    
    Example:
    ```mlir
    %result = rvv.vmv.v.x %scalar, %vl : i32, i64 -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_ScalarType:$rs1,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmv_v_x_<type>
      // Example: __riscv_vmv_v_x_i32m1
      return "__riscv_vmv_v_x" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VIotaMOp : RVV_Op<"viota.m", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Vector index of active mask elements";
  let description = [{
    Generates an index vector where each element contains the count of active
    (set) mask bits before that position. This is essentially a parallel prefix
    sum (popcount) of the mask.
    
    For each element i: vd[i] = popcount(mask[0..i-1])
    
    Arguments:
    - `mask`: Source mask vector (required)
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector containing indices
    
    Example:
    ```mlir
    // mask = [1, 0, 1, 1, 0] -> result = [0, 1, 1, 2, 3]
    %result = rvv.viota.m %mask, %vl : !rvv.mask<m1>, i64 -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_MaskType:$mask,
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_viota_m_<type><policy>
      // Example: __riscv_viota_m_u32m1_tu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_viota_m" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VIdVOp : RVV_Op<"vid.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Vector element index";
  let description = [{
    Generates a vector where each element contains its own index position.
    This creates an identity vector [0, 1, 2, 3, ...].
    
    For each element i: vd[i] = i
    
    Arguments:
    - `vl`: Vector length - number of elements to process
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are set
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)
    
    Results:
    - `vd`: Result vector containing element indices
    
    Example:
    ```mlir
    // vl = 5 -> result = [0, 1, 2, 3, 4]
    %result = rvv.vid.v %vl : i64 -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    I64:$vl,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vid_v_<type><policy>
      // Example: __riscv_vid_v_u32m1_tu, __riscv_vid_v_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vid_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VMvSXOp : RVV_Op<"vmv.s.x", [
  Pure,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>
  ]> {
  let summary = "Move scalar to vector element 0";
  let description = [{
    Moves a scalar integer value to element 0 of a vector register.
    Other elements in the vector are preserved from the source vector.
    
    vd[0] = rs1
    vd[i] = vs2[i] for i > 0
    
    Arguments:
    - `rs1`: Scalar integer value to move into element 0
    - `vs2`: Source vector (elements 1..N-1 are preserved)
    - `vl`: Vector length (must be at least 1)
    
    Results:
    - `vd`: Destination vector with element 0 set to scalar value
    
    Example:
    ```mlir
    %result = rvv.vmv.s.x %scalar, %vec, %vl : i32, !rvv.vector<i32, m1>, i64 
      -> !rvv.vector<i32, m1>
    ```
  }];
  let arguments = (ins
    RVV_ScalarType:$rs1,
    RVV_VectorType:$vs2,
    I64:$vl
  );
  let results = (outs RVV_VectorType:$vd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vmv_s_x_<type>
      // Example: __riscv_vmv_s_x_i32m1
      return "__riscv_vmv_s_x" + 
             getVd().getType().getVectorTypeString();
    }
  }];
}

def VLenBOp : RVV_Op<"vlenb", [Pure]> {
  let summary = "Read VLENB CSR (vector register length in bytes)";
  let description = [{
    Reads the VLENB CSR, which contains the number of bytes in a vector register.
    This is a constant value determined by the hardware implementation.
    
    VLENB = VLEN / 8, where VLEN is the vector register length in bits.
    
    For example:
    - If VLEN = 128 bits, then VLENB = 16 bytes
    - If VLEN = 256 bits, then VLENB = 32 bytes
    - If VLEN = 512 bits, then VLENB = 64 bytes
    
    Results:
    - `rd`: Scalar result containing VLENB value (unsigned long)
    
    Example:
    ```mlir
    %vlenb = rvv.vlenb : i64
    ```
  }];
  let arguments = (ins);
  let results = (outs I64:$rd);
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName() {
      // Format: __riscv_vlenb
      return "__riscv_vlenb";
    }
  }];
}

def VWCvtXXVOp : RVV_Op<"vwcvt.x.x.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening signed integer conversion";
  let description = [{
    Performs element-wise widening of signed integers, doubling the element width.
    Each SEW-bit signed element is sign-extended to 2*SEW bits.
    
    This is a pseudo-instruction that maps to vwadd with a zero operand,
    effectively performing a widening sign-extension.
    
    - vs1: narrow signed vector (SEW bits)
    - vd: wide signed vector (2*SEW bits)
    
    Maps to __riscv_vwcvt_x_x_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vwcvt.x.x.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwcvt_x_x_v_<type><policy>
      // Example: __riscv_vwcvt_x_x_v_i64m2_tu, __riscv_vwcvt_x_x_v_i64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwcvt_x_x_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VWCvtUXXVOp : RVV_Op<"vwcvtu.x.x.v", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVWidenInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Widening unsigned integer conversion";
  let description = [{
    Performs element-wise widening of unsigned integers, doubling the element width.
    Each SEW-bit unsigned element is zero-extended to 2*SEW bits.
    
    This is a pseudo-instruction that maps to vwaddu with a zero operand,
    effectively performing a widening zero-extension.
    
    - vs1: narrow unsigned vector (SEW bits)
    - vd: wide unsigned vector (2*SEW bits)
    
    Maps to __riscv_vwcvtu_x_x_v intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vwcvtu.x.x.v %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i32, m1>, i64) -> !rvv.vector<i64, m2>
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
      // Format: __riscv_vwcvtu_x_x_v_<type><policy>
      // Example: __riscv_vwcvtu_x_x_v_u64m2_tu, __riscv_vwcvtu_x_x_v_u64m2_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vwcvtu_x_x_v" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}

def VNCvtXXWOp : RVV_Op<"vncvt.x.x.w", [
  Pure,
  DeclareOpInterfaceMethods<RVVPolicyInterface>,
  DeclareOpInterfaceMethods<RVVUnaryOpInterface>,
  DeclareOpInterfaceMethods<RVVVLConsumerInterface>,
  DeclareOpInterfaceMethods<RVVNarrowInterface>,
  AttrSizedOperandSegments
  ]> {
  let summary = "Narrowing integer conversion";
  let description = [{
    Performs element-wise narrowing (truncation) of integers, halving the element width.
    Each 2*SEW-bit element is truncated to SEW bits by discarding the upper bits.
    
    This is a pseudo-instruction that maps to vnsrl with a zero shift amount,
    effectively performing a narrowing truncation.
    
    - vs1: wide vector (2*SEW bits)
    - vd: narrow vector (SEW bits)
    
    Both signed and unsigned variants use the same operation since truncation
    is identical for both (just discards upper bits).
    
    Maps to __riscv_vncvt_x_x_w intrinsics.
    
    Example:
    ```mlir
    %result = rvv.vncvt.x.x.w %vs1, %vl {
      tail_policy = #rvv.tail_policy<tu>,
      mask_policy = #rvv.mask_policy<mu>
    } : (!rvv.vector<i64, m2>, i64) -> !rvv.vector<i32, m1>
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
      // Format: __riscv_vncvt_x_x_w_<type><policy>
      // Example: __riscv_vncvt_x_x_w_i32m1_tu, __riscv_vncvt_x_x_w_u32m1_tumu
      RVVPolicyInterface policyIf = *this;
      return "__riscv_vncvt_x_x_w" + 
             getVd().getType().getVectorTypeString() + 
             policyIf.getPolicySuffix();
    }
  }];
}


#endif // RVV_INTEGER_OPS