# CudaLight
C++ manager class for CudaLightKernel API. The low level calls are managed by the static class DeviceManager, whereas the high level infrastructure is delegated to the particular buffer type. Only contiguous memory data structures have been implemented, as this project aims to give a simplified version to the CUDA standard library. The implemented structures are:

- Vector
- Matrix (only column-wise)
- 3D Tensor

## Types
All data structures are templated where the arguments are the memory space and the math domain. The memory space indicates where the memory has to be allocated, i.e. host side (CPU) or device side (GPU). The math domain defines the type of the vector: integer, float or double.

## Restrictions
Dynamic buffers are not allowed. Size is needed in every constructor, and it's not possible to resize the given buffer

For convenience's sake the following typedefs have been defined:

- Vector:
```c++
	typedef Vector<MemorySpace::Device, MathDomain::Int> GpuIntegerVector;
	typedef Vector<MemorySpace::Device, MathDomain::Float> GpuSingleVector;
	typedef GpuSingleVector GpuFloatVector;
	typedef Vector<MemorySpace::Device, MathDomain::Double> GpuDoubleVector;

	typedef Vector<MemorySpace::Host, MathDomain::Int> CpuIntegerVector;
	typedef Vector<MemorySpace::Host, MathDomain::Float> CpuSingleVector;
	typedef CpuSingleVector CpuFloatVector;
	typedef Vector<MemorySpace::Host, MathDomain::Double> CpuDoubleVector;
	
	typedef GpuSingleVector vec;
	typedef GpuDoubleVector dvec;
	typedef GpuIntegerVector ivec;
```

- Matrix:
```c++
	typedef ColumnWiseMatrix<MemorySpace::Device, MathDomain::Int> GpuIntegerMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Device, MathDomain::Float> GpuSingleMatrix;
	typedef GpuSingleMatrix GpuFloatMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Device, MathDomain::Double> GpuDoubleMatrix;

	typedef ColumnWiseMatrix<MemorySpace::Host, MathDomain::Int> CpuIntegerMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Host, MathDomain::Float> CpuSingleMatrix;
	typedef CpuSingleVector CpuFloatMatrix;
	typedef ColumnWiseMatrix<MemorySpace::Host, MathDomain::Double> CpuDoubleMatrix;

	typedef GpuSingleMatrix mat;
	typedef GpuDoubleMatrix dmat;
	typedef GpuIntegerMatrix imat;
```

## Sample usage
- Alloc a vector of 10 floats on GPU:
```c++
  cl::GpuSingleVector gpuVector(10);
```

- Alloc a vector of n integers on CPU, initialised at -1:
```c++
  const unsigned nElements = 50;
  cl::CpuIntegerVector cpuVector(nElements, -1);
```

- Alloc a float vector of n integers on CPU, initialised with a linear space between -1 and 1:
```c++
  const unsigned nElements = 50;
  const float lowerBound = -1.0f;
  const float upperBound =  1.0f;
  cl::vec v = cl::LinSpace(lowerBound, upperBound, nElements);
```

- Add two vectors with cuBlas:
```c++
  const cl::vec a = cl::LinSpace(-1.0, 1.0, 100);
  const cl::vec b = cl::RandomUniform(v1.size());
  const cl::vec c = a + b;
```

- Element-wise product between vectors:
```c++
  const cl::vec a = cl::LinSpace(-1.0, 1.0, 100);
  const cl::vec b = cl::RandomUniform(v1.size());
  const cl::vec c = a % b;
```

- Dot product between matrices:
```c++
  const unsigned nRowsA = 10;
  const unsigned nColsA = 15;
  const unsigned nColsB = 20;
  const cl::mat A(nRowsA, nColsA, 2.7182f);
  const cl::mat B(nColsA, nColsB, 3.1415f);
  const cl::mat C = A * B;
```

- Dot product between a matrix and a vector:
```c++
  const unsigned nRowsA = 10;
  const unsigned nColsA = 15;
  const cl::mat A(nRowsA, nColsA, 2.7182f);
  const cl::vec x(nColsA, 3.1415f);
  const cl::vec y = A * x;
```

- Serialization (compatible with numpy.loadtxt):
```c++
  const unsigned nRows = 10;
  const unsigned nCols = 15;
  std::ofstream f("matrix.cl");
  cl::mat m(nRows, nCols);
  m.LinSpace(0.0f, 1.0f);
  f << m;
```

- Deserialization (compatible with numpy.savetxt):
```c++
  std::ofstream f1("matrix.cl");
  cl::mat m = cl::DeserializeMatrix(f1);
  
  std::ofstream f2("vector.cl");
  cl::vec v = cl::DeserializeMatrix(f2);
```
