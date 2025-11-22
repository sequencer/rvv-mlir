#ifndef RVV_INTERFACES
#define RVV_INTERFACES

include "mlir/IR/OpBase.td"

def RVVPolicyInterface : OpInterface<"RVVPolicyInterface"> {
  let description = [{
    Interface for RVV operations that support masking and tail policies.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the mask operand",
      "::mlir::Value", "getMask", (ins)
    >,
    InterfaceMethod<
      "Get the tail policy attribute",
      "::mlir::rvv::TailPolicyAttr", "getTailPolicy", (ins),
      /*methodBody=*/[{
        return $_op.getTailPolicyAttr();
      }]
    >,
    InterfaceMethod<
      "Get the mask policy attribute",
      "::mlir::rvv::MaskPolicyAttr", "getMaskPolicy", (ins),
      /*methodBody=*/[{
        return $_op.getMaskPolicyAttr();
      }]
    >,
    InterfaceMethod<
      "Get the policy suffix string for intrinsic name generation",
      "std::string", "getPolicySuffix", (ins),
      /*methodBody=*/[{
        bool isMasked = static_cast<bool>($_op.getMask());
        auto tailPolicy = $_op.getTailPolicy();
        auto maskPolicy = $_op.getMaskPolicy();
        
        if (!isMasked) {
          if (tailPolicy == TailPolicy::Undisturbed)
            return "_tu";
          return "";
        }

        if (tailPolicy == TailPolicy::Undisturbed && maskPolicy == MaskPolicy::Undisturbed)
          return "_tumu";
        if (tailPolicy == TailPolicy::Undisturbed)
          return "_tum";
        if (maskPolicy == MaskPolicy::Undisturbed)
          return "_mu";

        return "_m";
      }]
    >
  ];

  let extraClassDeclaration = [{
  }];
}

def RVVVLProducerInterface : OpInterface<"RVVVLProducerInterface"> {
  let description = [{
    Interface for RVV operations that produce a vector length (vl) value.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the vector length result",
      "::mlir::Value", "getNewVl", (ins),
      /*methodBody=*/[{
        return $_op.getNewVl();
      }]
    >
  ];

  let verify = [{
    // Verify that the operation has at least one result of I64 type
    if ($_op->getNumResults() == 0) {
      return $_op->emitOpError("RVVVLProducerInterface requires at least one result");
    }
    auto vlResult = ::mlir::cast<ConcreteOp>($_op).getNewVl();
    if (!vlResult || !vlResult.getType().isInteger(64)) {
      return $_op->emitOpError("new vl result must be of I64 type");
    }
    return ::mlir::success();
  }];
}

def RVVVLConsumerInterface : OpInterface<"RVVVLConsumerInterface"> {
  let description = [{
    Interface for RVV operations that consume a vector length (vl) value.
    Most RVV operations take a vl operand to specify the active vector length.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the vector length operand",
      "::mlir::Value", "getVl", (ins),
      /*methodBody=*/[{
        return $_op.getVl();
      }]
    >
  ];

  let verify = [{
    // Verify that the operation has a vl operand of I64 type
    auto vlOperand = ::mlir::cast<ConcreteOp>($_op).getVl();
    if (!vlOperand) {
      return $_op->emitOpError("RVVVLConsumerInterface requires a vl operand");
    }
    if (!vlOperand.getType().isInteger(64)) {
      return $_op->emitOpError("vl operand must be of I64 type");
    }
    return ::mlir::success();
  }];
}

def RVVWidenInterface : OpInterface<"RVVWidenInterface"> {
  let description = [{
    Interface for RVV widening operations where the result is twice as wide
    as the operands.
  }];
  let cppNamespace = "::mlir::rvv";
}

def RVVNarrowInterface : OpInterface<"RVVNarrowInterface"> {
  let description = [{
    Interface for RVV narrowing operations where the result is half as wide
    as the source operands.
    Examples include vnsrl (narrowing shift right logical), vnsra (narrowing 
    shift right arithmetic), and other narrowing operations.
  }];
  let cppNamespace = "::mlir::rvv";
}

def RVVUnaryOpInterface : OpInterface<"RVVUnaryOpInterface"> {
  let description = [{
    Interface for RVV unary operations that have one source operand (vs1)
    and one destination operand (rd).
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the source operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (rd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >
  ];
}

def RVVVectorVectorOpInterface : OpInterface<"RVVVectorVectorOpInterface"> {
  let description = [{
    Interface for RVV binary operations that have two source operands (vs1, vs2)
    and one destination operand (rd).
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the first source operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the second source operand (vs2)",
      "::mlir::Value", "getVs2", (ins),
      /*methodBody=*/[{
        return $_op.getVs2();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >
  ];
}

def RVVVectorScalarOpInterface : OpInterface<"RVVVectorScalarOpInterface"> {
  let description = [{
    Interface for RVV vector-scalar operations that have a vector operand (vs1)
    and a scalar operand (rs1), and one destination operand (vd).
    Examples include vadd.vx, vsub.vx, vmul.vx, etc.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the vector operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the scalar operand (rs1)",
      "::mlir::Value", "getRs1", (ins),
      /*methodBody=*/[{
        return $_op.getRs1();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getVd();
      }]
    >
  ];
}

def RVVVectorVectorMaskOpInterface : OpInterface<"RVVVectorVectorMaskOpInterface"> {
  let description = [{
    Interface for RVV vector-vector operations that have two vector operands (vs1, vs2),
    a mask register as third operand (vs3), and one destination operand (vd).
    Examples include vadc.vv, vsbc.vv, vmadc.vv, vmsbc.vv, etc.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the first source operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the second source operand (vs2)",
      "::mlir::Value", "getVs2", (ins),
      /*methodBody=*/[{
        return $_op.getVs2();
      }]
    >,
    InterfaceMethod<
      "Get the mask operand (vs3)",
      "::mlir::Value", "getVs3", (ins),
      /*methodBody=*/[{
        return $_op.getVs3();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >
  ];
}

def RVVVectorScalarMaskOpInterface : OpInterface<"RVVVectorScalarMaskOpInterface"> {
  let description = [{
    Interface for RVV vector-scalar operations that have a vector operand (vs1),
    a scalar operand (rs1), a mask register as third operand (vs3), and one 
    destination operand (vd).
    Examples include vadc.vx, vsbc.vx, vmadc.vx, vmsbc.vx, etc.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the vector operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the scalar operand (rs1)",
      "::mlir::Value", "getRs1", (ins),
      /*methodBody=*/[{
        return $_op.getRs1();
      }]
    >,
    InterfaceMethod<
      "Get the mask operand (vs3)",
      "::mlir::Value", "getVs3", (ins),
      /*methodBody=*/[{
        return $_op.getVs3();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getVd();
      }]
    >
  ];
}

def RVVVectorVectorVectorOpInterface : OpInterface<"RVVVectorVectorVectorOpInterface"> {
  let description = [{
    Interface for RVV ternary operations that have three vector source operands 
    (vs1, vs2, vs3) and one destination operand (vd).
    Examples include vmacc.vv, vnmsac.vv, vmadd.vv, vnmsub.vv, etc.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the first source operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the second source operand (vs2)",
      "::mlir::Value", "getVs2", (ins),
      /*methodBody=*/[{
        return $_op.getVs2();
      }]
    >,
    InterfaceMethod<
      "Get the third source operand (vs3)",
      "::mlir::Value", "getVs3", (ins),
      /*methodBody=*/[{
        return $_op.getVs3();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >,
  ];
}

def RVVVectorScalarVectorOpInterface : OpInterface<"RVVVectorScalarVectorOpInterface"> {
  let description = [{
    Interface for RVV ternary operations that have vector-scalar-vector operands:
    - vs1: first vector operand
    - rs1: scalar operand (note: no vs2 to maintain naming consistency)
    - vs3: third vector operand
    - vd: destination operand
    Examples include vmacc.vx, vnmsac.vx, vmadd.vx, vnmsub.vx, etc.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the first vector operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the scalar operand (rs1)",
      "::mlir::Value", "getRs1", (ins),
      /*methodBody=*/[{
        return $_op.getRs1();
      }]
    >,
    InterfaceMethod<
      "Get the third vector operand (vs3)",
      "::mlir::Value", "getVs3", (ins),
      /*methodBody=*/[{
        return $_op.getVs3();
      }]
    >,
    InterfaceMethod<
      "Get the destination operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getVd();
      }]
    >
  ];
}

def RVVMaskResultInterface : OpInterface<"RVVMaskResultInterface"> {
  let description = [{
    Interface for RVV operations that produce a mask type result (vd).
    This interface is typically used by comparison operations (vmseq, vmslt, etc.)
    and other operations that generate mask values.
    
    The interface verifies that the result type is an RVV mask type and provides
    a method to retrieve the mask result.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the mask result (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >
  ];

  let verify = [{
    // Verify that the operation produces a mask type result
    auto vd = ::mlir::cast<ConcreteOp>($_op).getVd();
    if (!vd) {
      return $_op->emitOpError("RVVMaskResultInterface requires a vd result");
    }

    // Check that vd is an RVV mask type
    auto vdType = ::llvm::dyn_cast<::mlir::rvv::MaskType>(vd.getType());
    if (!vdType) {
      return $_op->emitOpError("RVVMaskResultInterface requires vd to be an RVV mask type");
    }

    return ::mlir::success();
  }];
}

def RVVMaskOpInterface : OpInterface<"RVVMaskOpInterface"> {
  let description = [{
    Interface for RVV unary mask operations that have one mask source operand (vs1)
    and one mask destination operand (vd).
    Examples include vmnot (bitwise NOT on mask register).
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the source mask operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the destination mask operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >
  ];

  let verify = [{
    auto concreteOp = ::mlir::cast<ConcreteOp>($_op);
    auto vs1 = concreteOp.getVs1();
    auto vd = concreteOp.getVd();

    if (!vs1 || !vd) {
      return $_op->emitOpError("RVVMaskOpInterface requires vs1 operand and vd result");
    }

    // Check that vs1 is an RVV mask type
    auto vs1Type = ::llvm::dyn_cast<::mlir::rvv::MaskType>(vs1.getType());
    if (!vs1Type) {
      return $_op->emitOpError("RVVMaskOpInterface requires vs1 to be an RVV mask type");
    }

    // Check that vd is an RVV mask type
    auto vdType = ::llvm::dyn_cast<::mlir::rvv::MaskType>(vd.getType());
    if (!vdType) {
      return $_op->emitOpError("RVVMaskOpInterface requires vd to be an RVV mask type");
    }

    return ::mlir::success();
  }];
}

def RVVMaskMaskOpInterface : OpInterface<"RVVMaskMaskOpInterface"> {
  let description = [{
    Interface for RVV binary mask operations that have two mask source operands 
    (vs1, vs2) and one mask destination operand (vd).
    Examples include vmand, vmandn, vmnand, vmnor, vmor, vmorn, vmxnor, vmxor.
  }];
  let cppNamespace = "::mlir::rvv";

  let methods = [
    InterfaceMethod<
      "Get the first source mask operand (vs1)",
      "::mlir::Value", "getVs1", (ins),
      /*methodBody=*/[{
        return $_op.getVs1();
      }]
    >,
    InterfaceMethod<
      "Get the second source mask operand (vs2)",
      "::mlir::Value", "getVs2", (ins),
      /*methodBody=*/[{
        return $_op.getVs2();
      }]
    >,
    InterfaceMethod<
      "Get the destination mask operand (vd)",
      "::mlir::Value", "getVd", (ins),
      /*methodBody=*/[{
        return $_op.getResult();
      }]
    >
  ];

  let verify = [{
    auto concreteOp = ::mlir::cast<ConcreteOp>($_op);
    auto vs1 = concreteOp.getVs1();
    auto vs2 = concreteOp.getVs2();
    auto vd = concreteOp.getVd();

    if (!vs1 || !vs2 || !vd) {
      return $_op->emitOpError("RVVMaskMaskOpInterface requires vs1, vs2 operands and vd result");
    }

    // Check that vs1 is an RVV mask type
    auto vs1Type = ::llvm::dyn_cast<::mlir::rvv::MaskType>(vs1.getType());
    if (!vs1Type) {
      return $_op->emitOpError("RVVMaskMaskOpInterface requires vs1 to be an RVV mask type");
    }

    // Check that vs2 is an RVV mask type
    auto vs2Type = ::llvm::dyn_cast<::mlir::rvv::MaskType>(vs2.getType());
    if (!vs2Type) {
      return $_op->emitOpError("RVVMaskMaskOpInterface requires vs2 to be an RVV mask type");
    }

    // Check that vd is an RVV mask type
    auto vdType = ::llvm::dyn_cast<::mlir::rvv::MaskType>(vd.getType());
    if (!vdType) {
      return $_op->emitOpError("RVVMaskMaskOpInterface requires vd to be an RVV mask type");
    }

    return ::mlir::success();
  }];
}



#endif // RVV_INTERFACES
