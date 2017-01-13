#include "gpumat.h"

#include <cuda_runtime.h>

using namespace gpu;

GpuMat::GpuMat()
{
	rows = 0;
	cols = 0;
	type = 0;
	data = 0;
}

GpuMat::GpuMat(int rows, int cols, int type)
{
	this->rows = rows;
	this->cols = cols;
	this->type = type;

	int size = rows * cols * depth();

	cudaError_t err = cudaMalloc(&data, size);
}

GpuMat::GpuMat(const GpuMat &mat)
{
	rows = this->rows;
	cols = this->cols;
	type = this->type;

	cudaError_t err = cudaMalloc((void**)&data, mat.size());
	err = cudaMemcpy(data, mat.data, mat.size(), cudaMemcpyDeviceToDevice);
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

	cudaError_t err = cudaMalloc(&data, mat.size());
	err = cudaMemcpy(data, mat.data, mat.size(), cudaMemcpyDeviceToDevice);
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

void GpuMat::release()
{
	rows = cols = type = 0;
	if(data){
		cudaFree(data);
		data = 0;
	}
}

GpuMat::GpuMat(const ct::Mat_< float > &mat)
{
	rows = this->rows;
	cols = this->cols;
	type = GPU_FLOAT;

	cudaError_t err = cudaMalloc(&data, size());
	err = cudaMemcpy(data, mat.ptr(), size(), cudaMemcpyHostToDevice);
}

GpuMat::GpuMat(const ct::Mat_< double > &mat)
{
	rows = this->rows;
	cols = this->cols;
	type = GPU_DOUBLE;

	cudaError_t err = cudaMalloc(&data, size());
	err = cudaMemcpy(data, mat.ptr(), size(), cudaMemcpyHostToDevice);
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
}

double GpuVal::toDouble() const
{
	double res;
	toValue(*this, res);

	return res;
}

double GpuVal::toFloat() const
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

void add(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);
}

void sub(const GpuMat &A, const GpuMat &B, GpuMat &C)
{
	if(A.rows != B.rows || A.cols != B.cols || A.type != B.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);
}

void matmul(const GpuMat &At, const GpuMat &B, GpuMat &C)
{
	if(At.cols != B.rows || At.type != B.type)
		return;

	if(C.rows != At.cols || C.cols != B.cols || C.type != At.type)
		C.resize(At.cols, B.cols, At.type);

}

void matmulT1(const GpuMat &A, const GpuMat &Bt, GpuMat &C)
{
	if(A.rows != Bt.rows || A.type != Bt.type)
		return;

	if(C.rows != A.rows || C.cols != Bt.rows || C.type != A.type)
		C.resize(A.rows, Bt.rows, A.type);

}

void matmulT2(const GpuMat &A, const GpuMat &Bt, GpuMat &C)
{
	if(A.cols != Bt.cols || A.type != Bt.type)
		return;

	if(C.rows != A.rows || C.cols != Bt.rows || C.type != A.type)
		C.resize(A.rows, Bt.rows, A.type);

}


void mulval(const GpuMat &A, const GpuVal &value, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

}

void addval(const GpuMat &A, const GpuVal &value, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

}

void subval(const GpuMat &A, const GpuVal &value, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

}

void subval(const GpuVal &value, const GpuMat &A, GpuMat &C)
{
	if(A.type != value.type)
		return;

	if(C.rows != A.rows || C.cols != A.cols || C.type != A.type)
		C.resize(A);

}


void biasPlus(GpuMat &A, const GpuMat &bias)
{
	if(A.cols != bias.cols || A.cols != bias.rows)
		return;


}
