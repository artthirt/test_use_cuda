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
template< class T >
__global__ void add(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
template< class T >
__global__ void sub(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
template< class T >
__global__ void matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C)
{

}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
template< class T >
__global__ void matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C)
{

}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
template< class T >
__global__ void mulval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{

}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
template< class T >
__global__ void addval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{

}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
template< class T >
__global__ void subval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{

}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
template< class T >
__global__ void subval(const GpuVal& value, const GpuMat& A, GpuMat& C)
{

}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
template< class T >
__global__ void biasPlus(GpuMat& A, const GpuMat& bias)
{

}

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
template< class T >
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::add<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::add<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sub<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::sub<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmul<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::matmul<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
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

	switch (At.type) {
	case GPU_DOUBLE:
		internal::matmulT1<double> <<<dimGrid, dimBlock>>>(At, B, C);
		break;
	case GPU_FLOAT:
		internal::matmulT1<float> <<<dimGrid, dimBlock>>>(At, B, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::matmulT2<double> <<<dimGrid, dimBlock>>>(A, Bt, C);
		break;
	case GPU_FLOAT:
		internal::matmulT2<float> <<<dimGrid, dimBlock>>>(A, Bt, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::mulval<double> <<<dimGrid, dimBlock>>>(A, value, C);
		break;
	case GPU_FLOAT:
		internal::mulval<float> <<<dimGrid, dimBlock>>>(A, value, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::addval<double> <<<dimGrid, dimBlock>>>(A, value, C);
		break;
	case GPU_FLOAT:
		internal::addval<float> <<<dimGrid, dimBlock>>>(A, value, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>(A, value, C);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>(A, value, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::subval<double> <<<dimGrid, dimBlock>>>(value, A, C);
		break;
	case GPU_FLOAT:
		internal::subval<float> <<<dimGrid, dimBlock>>>(value, A, C);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::biasPlus<double> <<<dimGrid, dimBlock>>>(A, bias);
		break;
	case GPU_FLOAT:
		internal::biasPlus<float> <<<dimGrid, dimBlock>>>( A, bias);
		break;
	}
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

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemiseMul<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::elemiseMul<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}
