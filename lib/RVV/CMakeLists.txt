# Only TableGen generation for now - no C++ source files yet
add_mlir_dialect_library(MLIRRVV
  RVVDialect.cpp
  RVVAttributes.cpp
  RVVInterfaces.cpp
  RVVTypes.cpp
  RVVOps.cpp
  RVVToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/RVV

  DEPENDS
  RVVIncGen
  RVVPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIREmitCDialect
  MLIRPass
)

install(TARGETS obj.MLIRRVV LIBRARY DESTINATION lib)
