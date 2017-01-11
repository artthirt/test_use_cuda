#ifndef MATS_H
#define MATS_H

#include <memory>

namespace mats{

struct Mat{
	Mat(){
		rows = 0;
		cols = 0;
		data = 0;
	}
	Mat(int rows, int cols){
		this->rows = rows;
		this->cols = cols;
		this->data = new double[rows * cols];
	}
	Mat(int rows, int cols, double* data){
		this->rows = rows;
		this->cols = cols;
		this->data = new double[rows * cols];
		memcpy(this->data, data, rows * cols * sizeof(double));
	}
	Mat(const Mat& m){
		rows = m.rows;
		cols = m.cols;

		this->data = new double[rows * cols];
		memcpy(this->data, m.data, rows * cols * sizeof(double));
	}

	~Mat(){
		if(data){
			delete []data;
		}
	}

	Mat& operator =(const Mat& m){
		if(data){
			delete [] data;
			data = 0;
		}
		rows = m.rows;
		cols = m.cols;

		this->data = new double[rows * cols];
		memcpy(this->data, m.data, rows * cols * sizeof(double));

		return *this;
	}

	size_t size() const{
		return rows * cols * sizeof(double);
	}

	inline double& at(int i0, int i1){
		return data[i0 * cols + i1];
	}
	inline double& at(int i0, int i1) const{
		return data[i0 * cols + i1];
	}

	inline double& operator[] (int index){
		return data[index];
	}
	inline double& operator[] (int index) const{
		return data[index];
	}

	double *data;
	int rows;
	int cols;
};

inline Mat matMult(const Mat& A, const Mat& B)
{
	if(A.cols != B.rows)
		return Mat();
	Mat res(A.rows, B.cols);

#pragma omp parallel for
	for(int i = 0; i < A.rows; ++i){
#pragma omp parallel for
		for(int j = 0; j < B.cols; ++j){
			res[i * B.cols + j] = 0;
			for(int k = 0; k < A.cols; ++k){
				res[i * B.cols + j] += A[i * A.cols + k] * B[k * B.cols + j];
			}
		}
	}
	return res;
}

}

#endif // MATS_H
