#include <DeviceManagerHelper.h>
#include <Exception.h>

#pragma region

#define __CREATE_FUNCTION_0_ARG(NAME)\
	EXTERN_C int _##NAME();\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME()\
			{\
				int err = _##NAME();\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, TYPE0, ARG0)\
	EXTERN_C int _##NAME(TYPE0 ARG0);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0)\
			{\
				int err = _##NAME(ARG0);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1)\
			{\
				int err = _##NAME(ARG0, ARG1);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_7_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_8_ARG(NAME, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);\
				if (err != 0)\
					throw CudaKernelException(#NAME, err);\
			}\
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
__CREATE_FUNCTION_2_ARG(RandNormal, MemoryBuffer, buf, const  unsigned, seed);

// CuBlasWrapper
__CREATE_FUNCTION_4_ARG(Add, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_3_ARG(Subtract, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y);
__CREATE_FUNCTION_3_ARG(AddEqual, MemoryBuffer, z, const MemoryBuffer, x, const double, alpha);
__CREATE_FUNCTION_5_ARG(AddEqualMatrix, MemoryTile, A, const MemoryTile, B, const MatrixOperation, aOperation, const MatrixOperation, bOperation, const double, alpha);
__CREATE_FUNCTION_2_ARG(SubtractEqual, MemoryBuffer, z, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(Scale, MemoryBuffer, z, const double, alpha);
__CREATE_FUNCTION_4_ARG(ElementwiseProduct, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_8_ARG(Multiply, MemoryTile, A, const MemoryTile, B, const MemoryTile, C, const unsigned, leadingDimensionB, const unsigned, leadingDimensionC, const MatrixOperation, bOperation, const MatrixOperation, cOperation, const double, alpha);
__CREATE_FUNCTION_5_ARG(Dot, MemoryBuffer, y, const MemoryTile, A, const MemoryBuffer, x, const MatrixOperation, aOperation, const double, alpha);
__CREATE_FUNCTION_1_ARG(CumulativeRowSum, MemoryTile, A);
__CREATE_FUNCTION_3_ARG(Solve, const MemoryTile, A, MemoryTile, B, const MatrixOperation, aOperation);
__CREATE_FUNCTION_2_ARG(Invert, MemoryTile, A, const MatrixOperation, aOperation);

//CuSparseWrappers
__CREATE_FUNCTION_4_ARG(SparseAdd, MemoryBuffer, z, const SparseMemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_3_ARG(SparseSubtract, MemoryBuffer, z, const SparseMemoryBuffer, x, const MemoryBuffer, y);
__CREATE_FUNCTION_5_ARG(SparseDot, MemoryBuffer, y, const SparseMemoryTile, A, const MemoryBuffer, x, const MatrixOperation, aOperation, const double, alpha);
__CREATE_FUNCTION_7_ARG(SparseMultiply, MemoryTile, A, const SparseMemoryTile, B, const MemoryTile, C, const unsigned, leadingDimensionB, const unsigned, leadingDimensionC, const MatrixOperation, bOperation, const double, alpha);

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

#pragma endregion