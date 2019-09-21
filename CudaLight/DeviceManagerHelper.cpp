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

#define __CREATE_FUNCTION_9_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7, TYPE8, ARG8)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_10_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7, TYPE8, ARG8, TYPE9, ARG9)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_11_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7, TYPE8, ARG8, TYPE9, ARG9, TYPE10, ARG10)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9, TYPE10 ARG10);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9, TYPE10 ARG10)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_12_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7, TYPE8, ARG8, TYPE9, ARG9, TYPE10, ARG10, TYPE11, ARG11)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9, TYPE10 ARG10, TYPE11 ARG11);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9, TYPE10 ARG10, TYPE11 ARG11)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}

#define __CREATE_FUNCTION_13_ARG(NAME, EXCEPTION, TYPE0, ARG0, TYPE1, ARG1, TYPE2, ARG2, TYPE3, ARG3, TYPE4, ARG4, TYPE5, ARG5, TYPE6, ARG6, TYPE7, ARG7, TYPE8, ARG8, TYPE9, ARG9, TYPE10, ARG10, TYPE11, ARG11, TYPE12, ARG12)\
	EXTERN_C int _##NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9, TYPE10 ARG10, TYPE11 ARG11, TYPE12 ARG12);\
	namespace dm\
	{\
		namespace detail\
		{\
			void NAME(TYPE0 ARG0, TYPE1 ARG1, TYPE2 ARG2, TYPE3 ARG3, TYPE4 ARG4, TYPE5 ARG5, TYPE6 ARG6, TYPE7 ARG7, TYPE8 ARG8, TYPE9 ARG9, TYPE10 ARG10, TYPE11 ARG11, TYPE12 ARG12)\
			{\
				int err = _##NAME(ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9, ARG10, ARG11, ARG12);\
				if (err != 0)\
					EXCEPTION::ThrowException(#NAME, err);\
			}\
		}\
	}
#pragma endregion

// Device
__CREATE_FUNCTION_1_ARG(GetDevice, CudaKernelExceptionFactory, int&, dev);
__CREATE_FUNCTION_0_ARG(ThreadSynchronize, CudaKernelExceptionFactory);
__CREATE_FUNCTION_1_ARG(SetDevice, CudaKernelExceptionFactory, const int, dev);
__CREATE_FUNCTION_0_ARG(GetDeviceStatus, CudaKernelExceptionFactory);
__CREATE_FUNCTION_1_ARG(GetBestDevice, CudaKernelExceptionFactory, int&, dev);
__CREATE_FUNCTION_1_ARG(GetDeviceCount, CudaKernelExceptionFactory, int&, count);
__CREATE_FUNCTION_2_ARG(HostToHostCopy, CudaKernelExceptionFactory, MemoryBuffer&, dest, const MemoryBuffer&, source);
__CREATE_FUNCTION_2_ARG(HostToDeviceCopy, CudaKernelExceptionFactory, MemoryBuffer&, dest, const MemoryBuffer&, source);
__CREATE_FUNCTION_2_ARG(DeviceToDeviceCopy, CudaKernelExceptionFactory, MemoryBuffer&, dest, const MemoryBuffer&, source);
__CREATE_FUNCTION_2_ARG(AutoCopy, CudaKernelExceptionFactory, MemoryBuffer&, dest, const MemoryBuffer&, source);
__CREATE_FUNCTION_1_ARG(Alloc, CudaKernelExceptionFactory, MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(AllocHost, CudaKernelExceptionFactory, MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(Free, CudaKernelExceptionFactory, const MemoryBuffer&, buf);
__CREATE_FUNCTION_1_ARG(FreeHost, CudaKernelExceptionFactory, const MemoryBuffer&, buf);


// Initializer
__CREATE_FUNCTION_1_ARG(Zero, CudaKernelExceptionFactory, MemoryBuffer&, buf);
__CREATE_FUNCTION_2_ARG(Initialize, CudaKernelExceptionFactory, MemoryBuffer&, buf, const double, value);
__CREATE_FUNCTION_3_ARG(LinSpace, CudaKernelExceptionFactory, MemoryBuffer&, buf, const double, x0, const double, x1);
__CREATE_FUNCTION_1_ARG(Reciprocal, CudaKernelExceptionFactory, MemoryBuffer&, buf);
__CREATE_FUNCTION_2_ARG(RandUniform, CudaKernelExceptionFactory, MemoryBuffer&, buf, const unsigned, seed);
__CREATE_FUNCTION_2_ARG(RandNormal, CudaKernelExceptionFactory, MemoryBuffer&, buf, const unsigned, seed);
__CREATE_FUNCTION_2_ARG(RandShuffle, CudaKernelExceptionFactory, MemoryBuffer&, buf, const unsigned, seed);
__CREATE_FUNCTION_3_ARG(RandShufflePair, CudaKernelExceptionFactory, MemoryBuffer&, buf1, MemoryBuffer&, buf2, const unsigned, seed);
__CREATE_FUNCTION_2_ARG(RandShuffleColumns, CudaKernelExceptionFactory, MemoryTile&, buf, const unsigned, seed);
__CREATE_FUNCTION_3_ARG(RandShuffleColumnsPair, CudaKernelExceptionFactory, MemoryTile&, buf1, MemoryTile&, buf2, const unsigned, seed);

// CuBlasWrapper
__CREATE_FUNCTION_4_ARG(Add, CuBlasKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x, const MemoryBuffer&, y, const double, alpha);
__CREATE_FUNCTION_3_ARG(Subtract, CuBlasKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x, const MemoryBuffer&, y);
__CREATE_FUNCTION_3_ARG(AddEqual, CuBlasKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x, const double, alpha);
__CREATE_FUNCTION_6_ARG(AddEqualMatrix, CuBlasKernelExceptionFactory, MemoryTile&, A, const MemoryTile&, B, const MatrixOperation, aOperation, const MatrixOperation, bOperation, const double, alpha, const double, beta);
__CREATE_FUNCTION_2_ARG(SubtractEqual, CuBlasKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(Scale, CuBlasKernelExceptionFactory, MemoryBuffer&, z, const double, alpha);
__CREATE_FUNCTION_2_ARG(ScaleColumns, CuBlasKernelExceptionFactory, MemoryTile&, z, const MemoryBuffer&, alpha);
__CREATE_FUNCTION_4_ARG(ElementwiseProduct, CuBlasKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x, const MemoryBuffer&, y, const double, alpha);
__CREATE_FUNCTION_7_ARG(Multiply, CuBlasKernelExceptionFactory, MemoryTile&, A, const MemoryTile&, B, const MemoryTile&, C, const MatrixOperation, bOperation, const MatrixOperation, cOperation, const double, alpha, const double, beta);
__CREATE_FUNCTION_10_ARG(SubMultiply, CuBlasKernelExceptionFactory, MemoryTile&, A, const MemoryTile&, B, const MemoryTile&, C, const unsigned, nRowsB, const unsigned, nColsB, const unsigned, nColsC, const MatrixOperation, bOperation, const MatrixOperation, cOperation, const double, alpha, const double, beta);
__CREATE_FUNCTION_9_ARG(BatchedMultiply, CuBlasKernelExceptionFactory, MemoryCube&, A, const MemoryCube&, B, const MemoryCube&, C, const unsigned, strideB, const unsigned, strideC, const MatrixOperation, bOperation, const MatrixOperation, cOperation, const double, alpha, const double, beta);
__CREATE_FUNCTION_6_ARG(Dot, CuBlasKernelExceptionFactory, MemoryBuffer&, y, const MemoryTile&, A, const MemoryBuffer&, x, const MatrixOperation, aOperation, const double, alpha, const double, beta);
__CREATE_FUNCTION_1_ARG(CumulativeRowSum, CuBlasKernelExceptionFactory, MemoryTile&, A);
__CREATE_FUNCTION_1_ARG(Eye, CuBlasKernelExceptionFactory, MemoryTile&, A);
__CREATE_FUNCTION_3_ARG(Solve, CuBlasKernelExceptionFactory, const MemoryTile&, A, MemoryTile&, B, const MatrixOperation, aOperation);
__CREATE_FUNCTION_2_ARG(Invert, CuBlasKernelExceptionFactory, MemoryTile&, A, const MatrixOperation, aOperation);
__CREATE_FUNCTION_4_ARG(KroneckerProduct, CuBlasKernelExceptionFactory, MemoryTile&, A, const MemoryBuffer&, x, const MemoryBuffer&, y, const double, alpha);
__CREATE_FUNCTION_4_ARG(BatchedTransposedKroneckerProduct, CuBlasKernelExceptionFactory, MemoryCube&, T, const MemoryTile&, x, const MemoryTile&, y, const double, alpha);
__CREATE_FUNCTION_4_ARG(RowWiseSum, CuBlasKernelExceptionFactory, MemoryBuffer&, x, const MemoryTile&, A, MemoryBuffer&, cache, const MatrixOperation, aOperation);
__CREATE_FUNCTION_4_ARG(CubeWiseSum, CuBlasKernelExceptionFactory, MemoryTile&, A, const MemoryCube&, T, MemoryCube&, cacheReshape, MemoryBuffer&, cacheOnes);
__CREATE_FUNCTION_2_ARG(ArgAbsMin, CuBlasKernelExceptionFactory, int&, argMin, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(ColumnWiseArgAbsMin, CuBlasKernelExceptionFactory, MemoryBuffer&, argMin, const MemoryTile&, A);
__CREATE_FUNCTION_2_ARG(ArgAbsMax, CuBlasKernelExceptionFactory, int&, argMax, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(ColumnWiseArgAbsMax, CuBlasKernelExceptionFactory, MemoryBuffer&, argMax, const MemoryTile&, A);
__CREATE_FUNCTION_2_ARG(IsNonZero, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(EuclideanNorm, CuBlasKernelExceptionFactory, double&, norm, const MemoryBuffer&, v);

//CuSparseWrappers
__CREATE_FUNCTION_4_ARG(SparseAdd, CuSparseKernelExceptionFactory, MemoryBuffer&, z, const SparseMemoryBuffer&, x, const MemoryBuffer&, y, const double, alpha);
__CREATE_FUNCTION_3_ARG(SparseSubtract, CuSparseKernelExceptionFactory, MemoryBuffer&, z, const SparseMemoryBuffer&, x, const MemoryBuffer&, y);
__CREATE_FUNCTION_5_ARG(SparseDot, CuSparseKernelExceptionFactory, MemoryBuffer&, y, const SparseMemoryTile&, A, const MemoryBuffer&, x, const MatrixOperation, aOperation, const double, alpha);
__CREATE_FUNCTION_5_ARG(SparseMultiply, CuSparseKernelExceptionFactory, MemoryTile&, A, const SparseMemoryTile&, B, const MemoryTile&, C, const MatrixOperation, bOperation, const double, alpha);

//CubWrappers
__CREATE_FUNCTION_2_ARG(Sum, CudaKernelExceptionFactory, double&, sum, const MemoryBuffer&, v);
__CREATE_FUNCTION_2_ARG(AbsMin, CuBlasKernelExceptionFactory, double&, min, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(AbsMax, CuBlasKernelExceptionFactory, double&, max, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(Min, CuBlasKernelExceptionFactory, double&, min, const MemoryBuffer&, x);
__CREATE_FUNCTION_2_ARG(Max, CuBlasKernelExceptionFactory, double&, max, const MemoryBuffer&, x);

// Forge Helpers
__CREATE_FUNCTION_3_ARG(MakePair, CudaKernelExceptionFactory, MemoryBuffer&, z, const MemoryBuffer&, x, const MemoryBuffer&, y);
__CREATE_FUNCTION_4_ARG(MakeTriple, CudaKernelExceptionFactory, MemoryBuffer&, v, const MemoryBuffer&, x, const MemoryBuffer&, y, const MemoryBuffer&, z);

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
