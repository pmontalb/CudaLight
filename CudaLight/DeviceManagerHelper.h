#pragma once

#include <Types.h>

#pragma region Macro Utilities

#define __CREATE_FUNCTION_0_ARG(NAME)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME();\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, TYPE0, ARG0)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0);\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1);\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
		}\
	}

#define __CREATE_FUNCTION_7_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6);\
		}\
	}

#define __CREATE_FUNCTION_8_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7);\
		}\
	}

#define __CREATE_FUNCTION_9_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7, TYPE8, ARG8)\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8);\
		}\
	}

#pragma endregion

// Device
__CREATE_FUNCTION_1_ARG(GetDevice, int&, dev);
__CREATE_FUNCTION_0_ARG(ThreadSynchronize);
__CREATE_FUNCTION_1_ARG(SetDevice, const int, dev);
__CREATE_FUNCTION_0_ARG(GetDeviceStatus);
__CREATE_FUNCTION_1_ARG(GetBestDevice, int&, dev);

__CREATE_FUNCTION_1_ARG(GetDeviceCount, int&, count);
__CREATE_FUNCTION_2_ARG(HostToHostCopy, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(HostToDeviceCopy, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(DeviceToDeviceCopy, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(AutoCopy, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_1_ARG(Alloc, MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(AllocHost, MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(Free, const MemoryBuffer, buf);
__CREATE_FUNCTION_1_ARG(FreeHost, const MemoryBuffer, buf);

// Initializer
__CREATE_FUNCTION_2_ARG(Initialize, MemoryBuffer, buf, const double, value);
__CREATE_FUNCTION_3_ARG(LinSpace, MemoryBuffer, buf, const double, x0, const double, x1);
__CREATE_FUNCTION_2_ARG(RandUniform, MemoryBuffer, buf, const unsigned, seed);
__CREATE_FUNCTION_2_ARG(RandNormal, MemoryBuffer, buf, const unsigned, seed);

// CuBlasWrappers
__CREATE_FUNCTION_4_ARG(Add, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha = 1.0);
__CREATE_FUNCTION_3_ARG(Subtract, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y);
__CREATE_FUNCTION_3_ARG(AddEqual, MemoryBuffer, z, const MemoryBuffer, x, const double, alpha = 1.0);
__CREATE_FUNCTION_6_ARG(AddEqualMatrix, MemoryTile, A, const MemoryTile, B, const MatrixOperation, aOperation = MatrixOperation::None, const MatrixOperation, bOperation = MatrixOperation::None, const double, alpha = 1.0, const double, beta = 1.0);
__CREATE_FUNCTION_2_ARG(SubtractEqual, MemoryBuffer, z, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(Scale, MemoryBuffer, z, const double, alpha);
__CREATE_FUNCTION_4_ARG(ElementwiseProduct, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha = 1.0);
__CREATE_FUNCTION_9_ARG(Multiply, MemoryTile, A, const MemoryTile, B, const MemoryTile, C, const unsigned, leadingDimensionB, const unsigned, leadingDimensionC, const MatrixOperation, bOperation = MatrixOperation::None, const MatrixOperation, cOperation = MatrixOperation::None, const double, alpha = 1.0, const double, beta = 0.0);
__CREATE_FUNCTION_6_ARG(Dot, MemoryBuffer, y, const MemoryTile, A, const MemoryBuffer, x, const MatrixOperation, aOperation = MatrixOperation::None, const double, alpha = 1.0, const double, beta = 0.0);
__CREATE_FUNCTION_1_ARG(CumulativeRowSum, MemoryTile, A);
__CREATE_FUNCTION_1_ARG(Eye, MemoryTile, A);
__CREATE_FUNCTION_3_ARG(Solve, const MemoryTile, A, MemoryTile, B, const MatrixOperation, aOperation = MatrixOperation::None);
__CREATE_FUNCTION_2_ARG(Invert, MemoryTile, A, const MatrixOperation, aOperation = MatrixOperation::None);
__CREATE_FUNCTION_4_ARG(KroneckerProduct, MemoryTile, A, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_2_ARG(ArgAbsMin, int&, argMin, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(ArgAbsMax, int&, argMax, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(AbsMin, double&, min, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(AbsMax, double&, max, const MemoryBuffer, x);

//CuSparseWrappers
__CREATE_FUNCTION_4_ARG(SparseAdd, MemoryBuffer, z, const SparseMemoryBuffer, x, const MemoryBuffer, y, const double, alpha = 1.0);
__CREATE_FUNCTION_3_ARG(SparseSubtract, MemoryBuffer, z, const SparseMemoryBuffer, x, const MemoryBuffer, y);
__CREATE_FUNCTION_5_ARG(SparseDot, MemoryBuffer, y, const SparseMemoryTile, A, const MemoryBuffer, x, const MatrixOperation, aOperation = MatrixOperation::None, const double, alpha = 1.0);
__CREATE_FUNCTION_7_ARG(SparseMultiply, MemoryTile, A, const SparseMemoryTile, B, const MemoryTile, C, const unsigned, leadingDimensionB, const unsigned, leadingDimensionC, const MatrixOperation, bOperation = MatrixOperation::None, const double, alpha = 1.0);

#pragma region Undef macros

#undef __CREATE_FUNCTION_0_ARG
#undef __CREATE_FUNCTION_1_ARG
#undef __CREATE_FUNCTION_2_ARG
#undef __CREATE_FUNCTION_3_ARG
#undef __CREATE_FUNCTION_4_ARG
#undef __CREATE_FUNCTION_5_ARG
#undef __CREATE_FUNCTION_6_ARG
#undef __CREATE_FUNCTION_7_ARG
#undef __CREATE_FUNCTION_8_ARG
#undef __CREATE_FUNCTION_9_ARG

#pragma endregion