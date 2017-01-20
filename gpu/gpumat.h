#ifndef GPU_H
#define GPU_H

#include <iostream>
#include <vector>

#ifdef _MSC_VER
	typedef unsigned char u_char;
#endif

namespace gpumat{

enum{
	GPU_FLOAT = 0,
	GPU_DOUBLE
};

const int sizeof_enum[] = {
	sizeof(float),		/// GPU_FLOAT
	sizeof(double)		/// GPU_DOUBLE
};

#define SIZEOF_TYPE(type) (sizeof_enum[type])

class GpuMat{
public:
	int type;
	int rows;
	int cols;
	u_char* data;

	GpuMat();
	GpuMat(int rows, int cols, int type);
	GpuMat(int rows, int cols, int type, void* data);
	GpuMat(const GpuMat& mat);
	~GpuMat();

	GpuMat &operator =(const GpuMat& mat);

	////////////

	GpuMat& ones();
	GpuMat& zeros();

	////////////

	int depth() const;
	int size() const;
	int total() const;
	bool empty() const;

	void resize(int rows, int cols, int type);
	void resize(const GpuMat& mat);

	void setData(void* data);
	void getData(void* data);

	void swap_dims();

	std::string operator()() const;

	void release();

private:
};

/**
 * @brief memset
 * @param A
 * @param val
 */
void memset(GpuMat& A, double val);
/**
 * @brief add
 * @param A
 * @param B
 * @param C - out C = A .+ B
 */
void add(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief add
 * @param A
 * @param val1
 * @param B
 * @param val2
 * @param C
 */
void add(const GpuMat& A, double val1, const GpuMat& B, double val2, GpuMat& C);
/**
 * @brief add
 * @param A
 * @param val
 * @param B
 */
void add(GpuMat& A, double val, const GpuMat& B);
/**
 * @brief sub
 * @param A
 * @param B
 * @param C - out C = A .- B
 */
void sub(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief matmul
 * @param A
 * @param B
 * @param C - out C = A * B
 */
void matmul(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C - out C = A' * B
 */
void matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C);
/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C - out C = A * B'
 */
void matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C);
/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A * value
 */
void mulval(const GpuMat& A, double value, GpuMat& C);
/**
 * @brief mulval
 * @param A -> A *= value
 * @param value - mat 1x1
 */
void mulval(GpuMat& A, double value);
/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A + value
 */
void addval(const GpuMat& A, double value, GpuMat& C);
/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 */
void addval(GpuMat &A, double value);
/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = A - value
 */
void subval(const GpuMat& A, double value, GpuMat& C);
/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C - out C = value - C
 */
void subval(double value, const GpuMat& A, GpuMat& C);
/**
 * @brief subval
 * @param A - > A - value
 * @param value - mat 1x1
 */
void subval(GpuMat& A, double value);
/**
 * @brief subval
 * @param A -> value - A
 * @param value - mat 1x1
 */
void subval(double value, GpuMat& A);
/**
 * @brief biasPlus
 * @param A - out A[i] = A[i] + bias
 * @param bias
 */
void biasPlus(GpuMat& A, const GpuMat& bias);
/**
 * @brief elemiseMul
 * @param A
 * @param B
 * @param C - out C = A .* B
 */
void elemiseMul(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief elemiseDiv
 * @param A
 * @param B
 * @param C - out C = A ./ B
 */
void elemiseDiv(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief elemiseSqrt
 * @param A
 * @param B
 * @param C - out C = sqrt(A)
 */
void elemiseSqrt(const GpuMat& A, GpuMat& C);
/**
 * @brief sumRows
 * @param A
 * @param C - out C[i] = val * sum(A[i, j]) (j = [1, cols])
 */
void sumRows(const GpuMat& A, GpuMat& C, double val = 1.);
/**
 * @brief transpose
 * @param A
 * @param C - out C = A'
 */
void transpose(const GpuMat& A, GpuMat& C);
/**
 * @brief reLu
 * @param A
 * @param B
 * @param C - out C = reLu(A)
 */
void reLu(const GpuMat& A, GpuMat& C);
/**
 * @brief deriv_reLu
 * @param A
 * @param B
 * @param C - out C = deriv_reLu(A)
 */
void deriv_reLu(const GpuMat& A, GpuMat& C);
/**
 * @brief softmax
 * @param A
 * @param axis -> 0 - in row, 1 - in col
 * @param C = softmax(A)
 * @param partZ = sum(exp(A), axis)
 */
void softmax(const GpuMat& A, int axis, GpuMat& C, GpuMat& partZ);

}

#endif // GPU_H
