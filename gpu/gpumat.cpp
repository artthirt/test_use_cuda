#include "gpumat.h"

#include <sstream>

#include <cuda_runtime.h>

using namespace gpumat;

GpuMat::GpuMat()
{
	rows = 0;
	cols = 0;
	type = 0;
	data = 0;
}

GpuMat::GpuMat(int rows, int cols, int type)
{
	if(rows && cols){
		this->rows = rows;
		this->cols = cols;
		this->type = type;

		int size = rows * cols * depth();

		cudaError_t err = cudaMalloc(&data, size);
	}
}

GpuMat::GpuMat(int rows, int cols, int type, void *data)
{
	if(rows && cols && data){
		this->rows = rows;
		this->cols = cols;
		this->type = type;

		int size = rows * cols * depth();

		cudaError_t err = cudaMalloc(&this->data, size);

		if(data){
			cudaMemcpy(this->data, data, size, cudaMemcpyHostToDevice);
		}
	}
}

GpuMat::GpuMat(const GpuMat &mat)
{
	rows = mat.rows;
	cols = mat.cols;
	type = mat.type;

	if(mat.data){
		cudaError_t err = cudaMalloc((void**)&data, mat.size());
		err = cudaMemcpy(data, mat.data, mat.size(), cudaMemcpyDeviceToDevice);
	}
}

GpuMat::~GpuMat()
{
	release();
}

GpuMat &GpuMat::operator =(const GpuMat &mat)
{
	if(!mat.data)
		return *this;

	cudaError_t err = cudaSuccess;
	if(mat.rows != rows || mat.cols != cols || mat.type != mat.type){
		release();

		rows = mat.rows;
		cols = mat.cols;
		type = mat.type;

		err = cudaMalloc(&data, mat.size());
	}

	if(mat.data && err == cudaSuccess ){
		err = cudaMemcpy(data, mat.data, mat.size(), cudaMemcpyDeviceToDevice);
	}
	return *this;
}

GpuMat &GpuMat::ones()
{
	if(data){
		memset(*this, 1);
	}

	return *this;
}

GpuMat &GpuMat::zeros()
{
	if(data){
		cudaMemset(data, 0, size());
	}
	return *this;
}

int GpuMat::depth() const
{
	return SIZEOF_TYPE(type);
}

int GpuMat::size() const
{
	return rows * cols * depth();
}

int GpuMat::total() const
{
	return rows * cols;
}

bool GpuMat::empty() const
{
	return data == nullptr;
}

void GpuMat::resize(int rows, int cols, int type)
{
	int sz = rows * cols * SIZEOF_TYPE(type);

	if(sz == size())
		return;

	release();

	this->rows = rows;
	this->cols = cols;
	this->type = type;

	cudaError_t err = cudaMalloc(&data, size());
}

void GpuMat::resize(const GpuMat &mat)
{
	release();

	this->rows = mat.rows;
	this->cols = mat.cols;
	this->type = mat.type;

	cudaError_t err = cudaMalloc(&data, size());
}

void GpuMat::setData(void *data)
{
	if(!data || !this->data || !rows || !cols)
		return;

	cudaMemcpy(this->data, data, size(), cudaMemcpyHostToDevice);
}

void GpuMat::getData(void *data) const
{
	if(!this->data || !data || !rows || !cols)
		return;

	cudaMemcpy(data, this->data, size(), cudaMemcpyDeviceToHost);
}

void GpuMat::swap_dims()
{
	std::swap(rows, cols);
}

//************

template<typename T >
std::string getString(void* data, int rows, int cols)
{
	if(!rows || !cols)
		return "";
	std::vector<T> vec;
	vec.resize(rows * cols);

	int size = rows * cols * sizeof(T);

	cudaMemcpy(&vec[0], data, size, cudaMemcpyDeviceToHost);

	std::stringstream stream;

	stream << "[";
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			stream << vec[i * cols + j] << " ";
		}
		if(i != rows - 1)stream << ";\n ";
	}
	stream << "]\n";
	return stream.str();
}

//************

std::string GpuMat::operator()() const
{
	if(!data)
		return "";

	switch (type) {
		case GPU_FLOAT:
			return getString<float>(data, rows, cols);
		case GPU_DOUBLE:
			return getString<double>(data, rows, cols);
	}
	return "";
}

std::string GpuMat::print(int _rows) const
{
	if(!data)
		return "";

	if(_rows < 0)
		_rows = rows;
	if(_rows > rows)
		_rows = rows;

	switch (type) {
		case GPU_FLOAT:
			return getString<float>(data, _rows, cols);
		case GPU_DOUBLE:
			return getString<double>(data, _rows, cols);
	}
	return "";
}

void GpuMat::release()
{
	rows = cols = type = 0;
	if(data){
		cudaFree(data);
		data = 0;
	}
}

/////////////////////////////////////////////////

/**
 * @brief cuda_memset
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_memset(GpuMat& A, double val);

/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
extern "C"
void cuda_add(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief cuda_add_params
 * @param A
 * @param val1
 * @param B
 * @param val2
 * @param C = val1 * A + val2 * B
 */
extern "C"
void cuda_add_params(const GpuMat& A, const GpuMat& B, double val1, double val2, GpuMat& C);

/**
 * @brief cuda_add_paramsA
 * @param A -> A += val1 * B
 * @param val
 * @param B
 */
extern "C"
void cuda_add_paramsA(GpuMat& A, const GpuMat& B, double val1, double val2);

/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
extern "C"
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C, double valA, double valB);

/**
 * @brief cuda_subA
 * @param A = A * valA - B * valB
 * @param B
 * @param valA
 * @param valB
 */
extern "C"
void cuda_subA(GpuMat& A, const GpuMat& B, double valA, double valB);

/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
extern "C"
void cuda_matmul(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
extern "C"
void cuda_matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C);

/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
extern "C"
void cuda_matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C);

/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
extern "C"
void cuda_mulval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
extern "C"
void cuda_mulvalA(const GpuMat& A, double value);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addval(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addvalA(GpuMat& A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_AvaltoC(const GpuMat& A, double value, GpuMat& C);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valA(GpuMat& A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_Aval(GpuMat& A, double value);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valAtoC(double value, const GpuMat& A, GpuMat& C);
/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
extern "C"
void cuda_biasPlus(GpuMat& A, const GpuMat& bias);

/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
extern "C"
void cuda_elemiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemiseMul
 * @param A = A .* B
 * @param B
 */
extern "C"
void cuda_elemiseMulA(GpuMat& A, const GpuMat& B);

/**
 * @brief elemiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
extern "C"
void cuda_elemiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C);

/**
 * @brief elemiseSqrt
 * @param A
 * @param C - out C = sqrt(A)
 */
extern "C"
void cuda_elemiseSqrt(const GpuMat& A, GpuMat& C);

/**
 * @brief elemiseSqr
 * @param A
 * @param C - out C = A .* A
 */
extern "C"
void cuda_elemiseSqr(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_sumrows
 * @param A
 * @param C - out C[i] = val * sum(A[i, j])(j = [1..cols])
 */
extern "C"
void cuda_sumrows(const GpuMat& A, GpuMat& C, double val);

/**
 * @brief cuda_transpose
 * @param A
 * @param C = A'
 */
extern "C"
void cuda_transpose(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_reLu
 * @param A
 * @param C = reLu(A)
 */
extern "C"
void cuda_reLu(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_derivReLu
 * @param A
 * @param C = derivRelu(A)
 */
extern "C"
void cuda_derivReLu(const GpuMat& A, GpuMat& C);

/**
 * @brief cuda_softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 */
extern "C"
void cuda_softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ);

/**
 * @brief cuda_adamgrad
 * @param A = -alpha * (sb1 * mA / (sqrt(sb2 * vA) + eps)
 * @param mA
 * @param vA
 * @param alpha
 * @param sb1
 * @param sb2
 */
extern "C"
void cuda_adamgrad(GpuMat& A, const GpuMat& mA, const GpuMat& vA, double alpha, double sb1, double sb2);

/////////////////////////////////////////////////

namespace gpumat {

/**
 * @brief memset
 * @param A
 * @param val
 */
void memset(GpuMat& A, double val)
{
	if(A.empty())
		return;

	cuda_memset(A, val);
}

void add(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_add(A, B, C);
}


void add(const GpuMat &A, const GpuMat &B, GpuMat &C, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_add_params(A, B, valA, valB, C);
}

void add(GpuMat &A, const GpuMat &B, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	cuda_add_paramsA(A, B, valA, valB);
}

void sub(const GpuMat &A, const GpuMat &B, GpuMat &C, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_sub(A, B, C, valA, valB);
}


void sub(GpuMat &A, const GpuMat &B, double valA, double valB)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	cuda_subA(A, B, valA, valB);
}

void matmul(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.cols != B.rows || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != B.cols || C.type != A.type)
		C.resize(A.rows, B.cols, A.type);

	cuda_matmul(A, B, C);
}

void matmulT1(const GpuMat &At, const GpuMat &B, GpuMat &C)
{
	if(At.rows != B.rows || At.type != B.type)
		return;

	if(C.rows != At.cols || C.cols != B.cols || C.type != At.type)
		C.resize(At.cols, B.cols, At.type);

	cuda_matmulT1(At, B, C);
}

void matmulT2(const GpuMat &A, const GpuMat &Bt, GpuMat &C)
{
	if(A.cols != Bt.cols || A.type != Bt.type)
		return;

	if(C.rows != A.rows || C.cols != Bt.rows || C.type != A.type)
		C.resize(A.rows, Bt.rows, A.type);

	cuda_matmulT2(A, Bt, C);
}


void mulval(const GpuMat &A, double value, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_mulval(A, value, C);
}

void mulval(GpuMat& A, double value)
{
	cuda_mulvalA(A, value);
}

void addval(const GpuMat &A, double value, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_addval(A, value, C);
}

void addval(GpuMat& A, double value)
{
	cuda_addvalA(A, value);
}

void subval(const GpuMat &A, double value, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_subval_AvaltoC(A, value, C);
}

void subval(double value, const GpuMat &A, GpuMat &C)
{
	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_subval_valAtoC(value, A, C);
}

void subval(GpuMat &A, double value)
{
	cuda_subval_Aval(A, value);
}

void subval(double value, GpuMat &A)
{
	cuda_subval_valA(A, value);
}

void biasPlus(GpuMat &A, const GpuMat &bias)
{
	if(A.cols != bias.cols || A.cols != bias.rows)
		return;

	cuda_biasPlus(A, bias);
}

void elemiseMul(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemiseMul(A, B, C);
}


void elemiseMul(GpuMat &A, const GpuMat &B)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	cuda_elemiseMulA(A, B);
}

void elemiseDiv(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemiseDiv(A, B, C);
}


void transpose(const GpuMat &A, GpuMat &C)
{
	if(A.empty())
		return;

	if(C.rows != A.cols || C.cols != A.rows || C.type != A.type)
		C.resize(A.cols, A.rows, A.type);

	cuda_transpose(A, C);

}

void elemiseSqrt(const GpuMat &A, GpuMat &C)
{
	if(A.empty())
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemiseSqrt(A, C);
}

void elemiseSqr(const GpuMat &A, GpuMat &C)
{
	if(A.empty())
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_elemiseSqr(A, C);
}

void reLu(const GpuMat &A, GpuMat &C)
{
	if(A.empty())
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_reLu(A, C);
}

void deriv_reLu(const GpuMat &A, GpuMat &C)
{
	if(A.empty())
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_derivReLu(A, C);
}

void softmax(const GpuMat &A, int axis, GpuMat &C, GpuMat &partZ)
{
	if(A.empty())
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	if(axis == 0){
		if(partZ.cols != A.cols || partZ.rows != 1){
			partZ.resize(1, A.cols, A.type);
		}
	}
	if(axis == 1){
		if(partZ.rows != A.rows || partZ.cols != 1){
			partZ.resize(A.rows, 1, A.type);
		}
	}
	cuda_softmax(A, axis, C, partZ);
}

void sumRows(const GpuMat &A, GpuMat &C, double val)
{
	if(A.empty())
		return;

	if(A.rows != C.rows || C.cols != 1 || A.type != C.type){
		C.resize(1, A.cols, A.type);
	}

	cuda_sumrows(A, C, val);
}

void sub_adamGrad(GpuMat &A, const GpuMat &mA, const GpuMat &vA, double alpha, double sb1, double sb2)
{
	if(A.empty() || mA.empty() || vA.empty() ||
			A.type != mA.type || A.type != vA.type ||
			A.rows != mA.rows || A.cols != mA.cols ||
			A.rows != vA.rows || A.cols != vA.cols)
		return;

	cuda_adamgrad(A, mA, vA, alpha, sb1, sb2);
}

}
