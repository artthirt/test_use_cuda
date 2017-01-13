#ifndef MATS_H
#define MATS_H

#include <memory>

#ifdef _MSCVER
#include <chrono>
#endif

namespace mats{

//////////////////////////////

#ifdef _MSCVER
/**
 * @brief getTick
 * @return
 */
inline int64_t getTick()
{
	using namespace std::chrono;
	milliseconds res = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
	return res.count();
}

#else

#include <time.h>
#include <sys/time.h>

inline int64_t getTick()
{
	timeval tv;
	gettimeofday(&tv, 0);
	int64_t ms = tv.tv_sec * 1000000 + tv.tv_usec / 1000;
	return ms;
}

#endif

//////////////////////////////


template< class T >
struct Mat{
	Mat(){
		rows = 0;
		cols = 0;
		data = 0;
	}
	Mat(int rows, int cols){
		this->rows = rows;
		this->cols = cols;
		this->data = new T[rows * cols];
	}
	Mat(int rows, int cols, T* data){
		this->rows = rows;
		this->cols = cols;
		this->data = new T[rows * cols];
		memcpy(this->data, data, rows * cols * sizeof(T));
	}
	Mat(const Mat& m){
		rows = m.rows;
		cols = m.cols;

		this->data = new T[rows * cols];
		memcpy(this->data, m.data, rows * cols * sizeof(T));
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

		this->data = new T[rows * cols];
		memcpy(this->data, m.data, rows * cols * sizeof(T));

		return *this;
	}

	size_t size() const{
		return rows * cols * sizeof(T);
	}

	inline T& at(int i0, int i1){
		return data[i0 * cols + i1];
	}
	inline T& at(int i0, int i1) const{
		return data[i0 * cols + i1];
	}

	inline T& operator[] (int index){
		return data[index];
	}
	inline T& operator[] (int index) const{
		return data[index];
	}

	T *data;
	int rows;
	int cols;
};

template< class T >
inline Mat<T> matMult(const Mat<T>& A, const Mat<T>& B)
{
	if(A.cols != B.rows)
		return Mat<T>();
	Mat<T> res(A.rows, B.cols);

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
