#include <DeviceManagerHelper.h>
#include <CudaException.h>

#pragma region Macro helpers

#define __CREATE_FUNCTION_0_ARG(NAME, EXCEPTION)\
	EXTERN_C int _##NAME();\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME()\
			{\
				int err = _##NAME();\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_1_ARG(NAME, EXCEPTION, TYPE0, ARG0)\
	EXTERN_C int _##NAME(TYPE0 ARG0);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0)\
			{\
				int err = _##NAME(ARG0);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_2_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1)\
			{\
				int err = _##NAME(ARG0, ARG1);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_3_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_4_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_5_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_6_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_7_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_8_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#pragma endregion

// Device
__CREATE_FUNCTION_1_ARG(GetDevice, CudaKernelExceptionFactory,  int&, dev);
__CREATE_FUNCTION_0_ARG(ThreadSynchronize, CudaKernelExceptionFactory);
__CREATE_FUNCTION_1_ARG(SetDevice, CudaKernelExceptionFactory, const int, dev);
__CREATE_FUNCTION_0_ARG(GetDeviceStatus, CudaKernelExceptionFactory);
__CREATE_FUNCTION_1_ARG(GetBestDevice, CudaKernelExceptionFactory, int&, dev);
__CREATE_FUNCTION_1_ARG(GetDeviceCount, CudaKernelExceptionFactory, int&, count);
__CREATE_FUNCTION_2_ARG(HostToHostCopy, CudaKernelExceptionFactory, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(HostToDeviceCopy, CudaKernelExceptionFactory, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(DeviceToDeviceCopy, CudaKernelExceptionFactory, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_2_ARG(AutoCopy, CudaKernelExceptionFactory, MemoryBuffer, dest, const MemoryBuffer, source);
__CREATE_FUNCTION_1_ARG(Alloc, CudaKernelExceptionFactory, MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(AllocHost, CudaKernelExceptionFactory, MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(Free, CudaKernelExceptionFactory, const MemoryBuffer, buf);
__CREATE_FUNCTION_1_ARG(FreeHost, CudaKernelExceptionFactory, const MemoryBuffer, buf);


// Initializer
__CREATE_FUNCTION_2_ARG(Initialize, CudaKernelExceptionFactory, MemoryBuffer, buf, const double, value);
__CREATE_FUNCTION_3_ARG(LinSpace, CudaKernelExceptionFactory, MemoryBuffer, buf, const double, x0, const double, x1);
__CREATE_FUNCTION_2_ARG(RandUniform, CudaKernelExceptionFactory, MemoryBuffer, buf, const unsigned, seed);
__CREATE_FUNCTION_2_ARG(RandNormal, CudaKernelExceptionFactory, MemoryBuffer, buf, const  unsigned, seed);

// CuBlasWrapper
__CREATE_FUNCTION_4_ARG(Add, CuBlasKernelExceptionFactory, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_3_ARG(Subtract, CuBlasKernelExceptionFactory, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y);
__CREATE_FUNCTION_3_ARG(AddEqual, CuBlasKernelExceptionFactory, MemoryBuffer, z, const MemoryBuffer, x, const double, alpha);
__CREATE_FUNCTION_5_ARG(AddEqualMatrix, CuBlasKernelExceptionFactory, MemoryTile, A, const MemoryTile, B, const MatrixOperation, aOperation, const MatrixOperation, bOperation, const double, alpha);
__CREATE_FUNCTION_2_ARG(SubtractEqual, CuBlasKernelExceptionFactory, MemoryBuffer, z, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(Scale, CuBlasKernelExceptionFactory, MemoryBuffer, z, const double, alpha);
__CREATE_FUNCTION_4_ARG(ElementwiseProduct, CuBlasKernelExceptionFactory, MemoryBuffer, z, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_8_ARG(Multiply, CuBlasKernelExceptionFactory, MemoryTile, A, const MemoryTile, B, const MemoryTile, C, const unsigned, leadingDimensionB, const unsigned, leadingDimensionC, const MatrixOperation, bOperation, const MatrixOperation, cOperation, const double, alpha);
__CREATE_FUNCTION_5_ARG(Dot, CuBlasKernelExceptionFactory, MemoryBuffer, y, const MemoryTile, A, const MemoryBuffer, x, const MatrixOperation, aOperation, const double, alpha);
__CREATE_FUNCTION_1_ARG(CumulativeRowSum, CuBlasKernelExceptionFactory, MemoryTile, A);
__CREATE_FUNCTION_1_ARG(Eye, CuBlasKernelExceptionFactory, MemoryTile, A);
__CREATE_FUNCTION_3_ARG(Solve, CuBlasKernelExceptionFactory, const MemoryTile, A, MemoryTile, B, const MatrixOperation, aOperation);
__CREATE_FUNCTION_2_ARG(Invert, CuBlasKernelExceptionFactory, MemoryTile, A, const MatrixOperation, aOperation);
__CREATE_FUNCTION_4_ARG(KroneckerProduct, CuBlasKernelExceptionFactory, MemoryTile, A, const MemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_2_ARG(ArgAbsMin, CuBlasKernelExceptionFactory, int&, argMin, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(ArgAbsMax, CuBlasKernelExceptionFactory, int&, argMax, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(AbsMin, CuBlasKernelExceptionFactory, double&, min, const MemoryBuffer, x);
__CREATE_FUNCTION_2_ARG(AbsMax, CuBlasKernelExceptionFactory, double&, max, const MemoryBuffer, x);

//CuSparseWrappers
__CREATE_FUNCTION_4_ARG(SparseAdd, CuSparseKernelExceptionFactory, MemoryBuffer, z, const SparseMemoryBuffer, x, const MemoryBuffer, y, const double, alpha);
__CREATE_FUNCTION_3_ARG(SparseSubtract, CuSparseKernelExceptionFactory, MemoryBuffer, z, const SparseMemoryBuffer, x, const MemoryBuffer, y);
__CREATE_FUNCTION_5_ARG(SparseDot, CuSparseKernelExceptionFactory, MemoryBuffer, y, const SparseMemoryTile, A, const MemoryBuffer, x, const MatrixOperation, aOperation, const double, alpha);
__CREATE_FUNCTION_7_ARG(SparseMultiply, CuSparseKernelExceptionFactory, MemoryTile, A, const SparseMemoryTile, B, const MemoryTile, C, const unsigned, leadingDimensionB, const unsigned, leadingDimensionC, const MatrixOperation, bOperation, const double, alpha);

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