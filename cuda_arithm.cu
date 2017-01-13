#include <cuda_runtime.h>

#include "gpumat.h"

using namespace gpumat;

namespace internal{

#define BLOCKSIZE	16

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
__global__ void add(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
__global__ void sub(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
__global__ void matmul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
__global__ void matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
__global__ void matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C)
{

}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
__global__ void mulval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{

}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
__global__ void addval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{

}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
__global__ void subval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{

}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
__global__ void subval(const GpuVal& value, const GpuMat& A, GpuMat& C)
{

}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
__global__ void biasPlus(GpuMat& A, const GpuMat& bias)
{

}

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
__global__ void elemiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
void cuda_add(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::add<<<dimGrid, dimBlock>>>(A, B, C);
}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::sub<<<dimGrid, dimBlock>>>(A, B, C);
}

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
void cuda_matmul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::matmul<<<dimGrid, dimBlock>>>(A, B, C);
}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
void cuda_matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C)
{
	int x1 = round((double)At.cols / BLOCKSIZE + 0.5);
	int x2 = round((double)At.rows / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::matmulT1<<<dimGrid, dimBlock>>>(At, B, C);
}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
void cuda_matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::matmulT2<<<dimGrid, dimBlock>>>(A, Bt, C);
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
void cuda_mulval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::mulval<<<dimGrid, dimBlock>>>(A, value, C);
}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
void cuda_addval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::addval<<<dimGrid, dimBlock>>>(A, value, C);
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
void cuda_subval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::subval<<<dimGrid, dimBlock>>>(A, value, C);
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
void cuda_subval(const GpuVal& value, const GpuMat& A, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::subval<<<dimGrid, dimBlock>>>(value, A, C);
}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
void cuda_biasPlus(GpuMat& A, const GpuMat& bias)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::biasPlus<<<dimGrid, dimBlock>>>(A, bias);
}

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
void cuda_elemiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = round((double)A.rows / BLOCKSIZE + 0.5);
	int x2 = round((double)A.cols / BLOCKSIZE + 0.5);

	if(!x1) x1 = 1;
	if(!x2) x2 = 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	internal::elemiseMul<<<dimGrid, dimBlock>>>(A, B, C);
}
