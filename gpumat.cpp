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
	rows = this->rows;
	cols = this->cols;
	type = this->type;

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
	release();

	rows = this->rows;
	cols = this->cols;
	type = this->type;

	if(mat.data){
		cudaError_t err = cudaMalloc(&data, mat.size());
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
	switch (type) {
		case GPU_FLOAT:
			return sizeof(float);
		case GPU_DOUBLE:
			return sizeof(double);
		default:
			break;
	}
	return 0;
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

void GpuMat::getData(void *data)
{
	if(!this->data || !data || !rows || !cols)
		return;

	cudaMemcpy(data, this->data, size(), cudaMemcpyDeviceToHost);
}

//************

template<typename T >
std::string getString(void* data, int rows, int cols)
{
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

void GpuMat::release()
{
	rows = cols = type = 0;
	if(data){
		cudaFree(data);
		data = 0;
	}
}

/////////////////////////////////////////////////

template< typename T >
void toValue(const GpuVal& val, T& res)
{
	res = T(0);
	switch (val.type) {
		case GPU_FLOAT:
		{
			float tmp;
			cudaMemcpy(&tmp, val.value, sizeof(float), cudaMemcpyDeviceToHost);
			res = tmp;
		}
			break;
		case GPU_DOUBLE:
		{
			double tmp;
			cudaMemcpy(&tmp, val.value, sizeof(double), cudaMemcpyDeviceToHost);
			res = tmp;
		}
			break;
		default:
			break;
	}
}

///////////////////////////////

GpuVal::GpuVal()
{
	type = 0;
	value = 0;
}

GpuVal::GpuVal(float value)
{
	type = GPU_FLOAT;
	cudaError_t err = cudaMalloc(&this->value, size());
	err = cudaMemcpy(this->value, &value, sizeof(float), cudaMemcpyHostToDevice);
}

GpuVal::GpuVal(double value)
{
	type = GPU_DOUBLE;
	cudaError_t err = cudaMalloc(&this->value, size());
	err = cudaMemcpy(this->value, &value, sizeof(double), cudaMemcpyHostToDevice);
}

GpuVal::GpuVal(const GpuVal &val)
{
	type = val.type;

	cudaError_t err = cudaMalloc(&this->value, size());
	err = cudaMemcpy(this->value, &val.value, sizeof(double), cudaMemcpyDeviceToDevice);
}

GpuVal::~GpuVal()
{
	release();
}

GpuVal &GpuVal::operator=(const GpuVal &val)
{
	release();
	type = val.type;

	cudaError_t err = cudaMalloc(&this->value, size());
	err = cudaMemcpy(this->value, &val.value, sizeof(double), cudaMemcpyDeviceToDevice);
	return *this;
}

template<typename T>
void setVal(void* data, T val)
{
	cudaMemcpy(data, &val, sizeof(val), cudaMemcpyHostToDevice);
}

void GpuVal::setValue(double val)
{
	switch (this->type) {
		case GPU_FLOAT:
			setVal<float>(value, (float)val);
			break;
		case GPU_DOUBLE:
			setVal<double>(value, (double)val);
			break;
		default:
			break;
	}
}

double GpuVal::toDouble() const
{
	double res;
	toValue(*this, res);

	return res;
}

float GpuVal::toFloat() const
{
	float res;
	toValue(*this, res);

	return res;
}

int GpuVal::size()
{
	switch (type) {
		case GPU_FLOAT:
			return sizeof(float);
		case GPU_DOUBLE:
			return sizeof(double);
		default:
			break;
	}
}

void GpuVal::release()
{
	if(value){
		cudaFree(value);
		value = 0;
		type = 0;
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
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
extern "C"
void cuda_sub(const GpuMat& A, const GpuMat& B, GpuMat& C);

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
void cuda_mulval(const GpuMat& A, const GpuVal& value, GpuMat& C);

/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
extern "C"
void cuda_addval(const GpuMat& A, const GpuVal& value, GpuMat& C);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
extern "C"
void cuda_subval_Aval(const GpuMat& A, const GpuVal& value, GpuMat& C);

/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
extern "C"
void cuda_subval_valA(const GpuVal& value, const GpuMat& A, GpuMat& C);

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
 * @param B
 * @param C - out C = sqrt(A)
 */
extern "C"
void cuda_elemiseSqrt(const GpuMat& A, GpuMat& C);

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

void sub(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_sub(A, B, C);
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


void mulval(const GpuMat &A, const GpuVal &value, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_mulval(A, value, C);
}

void addval(const GpuMat &A, const GpuVal &value, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_addval(A, value, C);
}

void subval(const GpuMat &A, const GpuVal &value, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_subval_Aval(A, value, C);
}

void subval(const GpuVal &value, const GpuMat &A, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

	cuda_subval_valA(value, A, C);
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
		if(axis == 1 || partZ.rows != A.rows || partZ.cols != 1){
			partZ.resize(A.rows, 1, A.type);
		}
	}
	partZ.zeros();

	cuda_softmax(A, axis, C, partZ);
}

}
