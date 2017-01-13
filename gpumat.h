#ifndef GPU_H
#define GPU_H

#include <iostream>
#include <vector>

#include <custom_types.h>

namespace gpu{

enum{
	GPU_FLOAT,
	GPU_DOUBLE
};


class GpuMat{
public:
	int type;
	int rows;
	int cols;
	u_char* data;

	GpuMat();
	GpuMat(int rows, int cols, int type);
	GpuMat(const GpuMat& mat);
	GpuMat(const ct::Mat_< float > & mat);
	GpuMat(const ct::Mat_< double > & mat);
	~GpuMat();

	GpuMat &operator =(const GpuMat& mat);

	int depth() const;
	int size() const;
	int total() const;

	void resize(int rows, int cols, int type);
	void resize(const GpuMat& mat);

	void release();

private:
};

class GpuVal{
public:
	int type;
	u_char* value;

	GpuVal();
	GpuVal(float value);
	GpuVal(double value);
	GpuVal(const GpuVal& val);
	~GpuVal();

	GpuVal &operator=(const GpuVal& val);

	double toDouble() const;
	double toFloat() const;

	int size();
	void release();

private:
};

/**
 * @brief add
 * @param A
 * @param B
 * @param C
 */
void add(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief sub
 * @param A
 * @param B
 * @param C
 */
void sub(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief matmul
 * @param A
 * @param B
 * @param C
 */
void matmul(const GpuMat& A, const GpuMat& B, GpuMat& C);
/**
 * @brief matmulT1
 * @param At - used as transposed matrix
 * @param B
 * @param C
 */
void matmulT1(const GpuMat& At, const GpuMat& B, GpuMat& C);
/**
 * @brief matmulT2
 * @param A
 * @param Bt - used as transposed matrix
 * @param C
 */
void matmulT2(const GpuMat& A, const GpuMat& Bt, GpuMat& C);
/**
 * @brief mulval
 * @param A
 * @param value - mat 1x1
 * @param C
 */
void mulval(const GpuMat& A, const GpuVal& value, GpuMat& C);
/**
 * @brief addval
 * @param A
 * @param value - mat 1x1
 * @param C
 */
void addval(const GpuMat& A, const GpuVal& value, GpuMat& C);
/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C
 */
void subval(const GpuMat& A, const GpuVal& value, GpuMat& C);
/**
 * @brief subval
 * @param A
 * @param value - mat 1x1
 * @param C
 */
void subval(const GpuVal& value, const GpuMat& A, GpuMat& C);
/**
 * @brief biasPlus
 * @param A
 * @param bias
 */
void biasPlus(GpuMat& A, const GpuMat& bias);

}

#endif // GPU_H
