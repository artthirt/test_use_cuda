#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "gpumat.h"

using namespace gpumat;

///////// begin internal namespace ///////////////

namespace internal{

struct Mtx{
	int rows;
	int cols;
	u_char* data;

	Mtx(){
		rows = cols = 0;
		data = 0;
	}
	Mtx(int rows, int cols, void* data){
		this->rows = rows;
		this->cols = cols;
		this->data = (u_char*)data;
	}
	Mtx(const GpuMat& mat){
		rows = mat.rows;
		cols = mat.cols;
		data = mat.data;
	}
};

struct Vls{
	u_char* value;
	Vls(){
		value = 0;
	}
	Vls(void* val){
		this->value = (u_char*)val;
	}
	Vls(const GpuVal& val){
		value = val.value;
	}
};

#define BLOCKSIZE	16

/**
 * @brief memset
 * @param A = val
 * @param val

 */
template< class T >
__global__ void memset(Mtx A, T val)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] = val;
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
template< class T >
__global__ void add(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * A.cols + col] = dA[row * A.cols + col] + dB[row * B.cols + col];
}

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
template< class T >
__global__ void sub(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] - dB[row * B.cols + col];
}

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
template< class T >
__global__ void matmul(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	float sC = 0;

	if(row < A.rows && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[row * A.cols + i] * DB[i * B.cols + col];
		}
		DC[row * B.cols + col] = sC;
	}
}

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
template< class T >
__global__ void matmulT1(Mtx At, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)At.data;
	T* DB = (T*)B.data;
	T* DC = (T*)C.data;

	float sC = 0;

//	s += val1[j * At.cols + i]/*at(i, j)*/ * val2[j * B.cols + k];
	if(row < At.cols && col < B.cols){
		for(int i = 0; i < B.rows; i++){
			sC += DA[i * At.cols + row] * DB[i * B.cols + col];
		}
		DC[row * C.cols + col] = sC;
	}

}

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
template< class T >
__global__ void matmulT2(Mtx A, Mtx Bt, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* DA = (T*)A.data;
	T* DB = (T*)Bt.data;
	T* DC = (T*)C.data;

	float sC = 0;

//	s += val1[i * A.cols + j]/*at(i, j)*/ * val2[k * Bt.cols + j]/*at(j, k)*/;
	if(row < A.rows && col < Bt.rows){
		for(int i = 0; i < Bt.cols; i++){
//			sC += DA[row * B.rows + i] * DB[i * B.cols + col];
			sC += DA[row * A.cols + i] * DB[col * Bt.cols + i];
		}
		DC[row * C.cols + col] = sC;
	}
}

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
template< class T >
__global__ void mulval(Mtx A, Vls value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;
	T val = *(T*)value.value;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * val;
}

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
template< class T >
__global__ void addval(Mtx A, Vls value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T val = *(T*)value.value;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] + val;
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
template< class T >
__global__ void subval(Mtx A, Vls value, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T val = *(T*)value.value;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] - val;
}

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
template< class T >
__global__ void subval(Vls value, Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA =(T*) A.data;
	T val = *(T*)value.value;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = val - dA[row * A.cols + col];

}

/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
template< class T >
__global__ void biasPlus(Mtx A, const Mtx bias)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dBias = (T*)bias.data;

	if(row < A.rows && col < A.cols)
		dA[row * A.cols + col] += dBias[col];

}

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
template< class T >
__global__ void elemiseMul(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] * dB[row * B.cols + col];
}

/**
 * @brief elemiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
template< class T >
__global__ void elemiseDiv(Mtx A, Mtx B, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dB = (T*)B.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = dA[row * A.cols + col] / dB[row * B.cols + col];
}

/**
 * @brief transpose
 * @param A
 * @param C = A'
 */
template< class T >
__global__ void transpose(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[col * C.cols + row] = dA[row * A.cols + col];
}

/**
 * @brief sqrt
 * @param A
 * @param C = A'
 */
template< class T >
__global__ void sqrt(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = std::sqrt(dA[row * A.cols + col]);
}

/**
 * @brief reLu
 * @param A
 * @param C = reLu(A)
 */
template< class T >
__global__ void reLu(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = max(dA[row * A.cols + col], 0.);
}

/**
 * @brief deriv_reLu
 * @param A
 * @param C = reLu(A)
 */
template< class T >
__global__ void deriv_reLu(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols)
		dC[row * C.cols + col] = (dA[row * A.cols + col] > 0)? 1 : 0;
}

/**
 * @brief _exp
 * @param A
 * @param C = exp(A)
 */
template< class T >
__global__ void _exp(Mtx A, Mtx C)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dA = (T*)A.data;
	T* dC = (T*)C.data;

	if(row < A.rows && col < A.cols){
		T val = exp(dA[row * A.cols + col]);
		dC[row * C.cols + col] = val;
	}
}

/**
 * @brief sum_col
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_col(Mtx C, Mtx cols)
{
	//int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(col < C.cols){
		for(int i = 0; i < C.rows; i++){
			dZ[col] += dC[i * C.cols + col];
		}
	}
}
/**
 * @brief sum_row
 * @param A
 * @param C = exp(A)
 * @param rows = sum(C)
 */
template< class T >
__global__ void sum_row(Mtx C, Mtx rows)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	//int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)rows.data;

	if(row < C.rows){
		for(int i = 0; i < C.cols; i++){
			dZ[row] += dC[row * C.cols + i];
		}
	}
}

/**
 * @brief div_col
 * @param C -> in/out C[i, j] /=  cols[j]
 * @param cols -> in
 */
template< class T >
__global__ void div_col(Mtx C, Mtx cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)cols.data;

	if(row < C.rows && col < C.cols){
		dC[row * C.cols + col] = dC[row * C.cols + col] / dZ[col];
	}
}

/**
 * @brief div_row
 * @param C -> in/out C[i, j] /=  rows[i]
 * @param rows -> in
 */
template< class T >
__global__ void div_row(Mtx C, Mtx rows)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	T* dC = (T*)C.data;
	T* dZ = (T*)rows.data;

	if(row < C.rows && col < C.cols){
		dC[row * C.cols + col] = dC[row * C.cols + col] / dZ[row];
	}
}

}

//////// end namespace /////////////////

/**
 * @brief cuda_memset
 * @param A = val
 * @param val
 */
extern "C"
void cuda_memset(GpuMat& A, double val)
{
	if(A.empty())
		return;

	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::memset<double> <<<dimGrid, dimBlock>>>(A, val);
		break;
	case GPU_FLOAT:
		internal::memset<float> <<<dimGrid, dimBlock>>>(A, val);
		break;
	}
}

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_add(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_matmul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C)
{
	//	int r = At.cols;
	//	int c = B.cols;

	int x1 = At.rows / BLOCKSIZE + 1;
	int x2 = B.cols / BLOCKSIZE + 1;

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
extern "C"
void cuda_matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C)
{
	int x1 = A.rows / BLOCKSIZE + 1;
	int x2 = A.cols / BLOCKSIZE + 1;

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
extern "C"
void cuda_mulval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_addval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_subval_Aval(const GpuMat& A, const GpuVal& value, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_subval_valA(const GpuVal& value, const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_biasPlus(GpuMat& A, const GpuMat& bias)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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
extern "C"
void cuda_elemiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

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

/**
 * @brief elemiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
extern "C"
void cuda_elemiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::elemiseDiv<double> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	case GPU_FLOAT:
		internal::elemiseDiv<float> <<<dimGrid, dimBlock>>>(A, B, C);
		break;
	}
}

/**
 * @brief cuda_transpose
 * @param A
 * @param C = A'
 */
extern "C"
void cuda_transpose(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::transpose<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::transpose<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_elemiseSqrt
 * @param A
 * @param C = A'
 */
extern "C"
void cuda_elemiseSqrt(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::sqrt<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::sqrt<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::reLu<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::reLu<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_derivReLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_derivReLu(const GpuMat& A, GpuMat& C)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::deriv_reLu<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::deriv_reLu<float> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}
}

/**
 * @brief cuda_softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 */
extern "C"
void cuda_softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ)
{
	int x1 = A.cols / BLOCKSIZE + 1;
	int x2 = A.rows / BLOCKSIZE + 1;

	dim3 dimGrid(x1, x2), dimBlock(BLOCKSIZE, BLOCKSIZE);

	switch (A.type) {
	case GPU_DOUBLE:
		internal::_exp<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	case GPU_FLOAT:
		internal::_exp<double> <<<dimGrid, dimBlock>>>(A, C);
		break;
	}

	switch (axis) {
		case 0:
			{
				switch (A.type) {
				case GPU_DOUBLE:
						internal::sum_col<double> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
						internal::div_col<double> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				case GPU_FLOAT:
						internal::sum_col<float> <<<dim3(x1, 1), dim3(BLOCKSIZE, 1)>>>(C, partZ);
						internal::div_col<float> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				}
			}
			break;
		case 1:
			{
				switch (A.type) {
				case GPU_DOUBLE:
						internal::sum_row<double> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
						internal::div_row<double> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				case GPU_FLOAT:
						internal::sum_row<float> <<<dim3(1, x2), dim3(1, BLOCKSIZE)>>>(C, partZ);
						internal::div_row<float> <<<dimGrid, dimBlock>>>(C, partZ);
					break;
				}
			}
			break;
	}
}
