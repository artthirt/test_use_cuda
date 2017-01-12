#include <iostream>

#include <fstream>
#include <ctime>
#include <chrono>

#include <cuda_runtime.h>

#include "mats.h"

using namespace std;

extern "C"
cudaError_t cuda_main();

extern "C"
cudaError_t cuda_mult(mats::Mat<float>* a, mats::Mat<float>* b, mats::Mat<float> *c);

template<class T>
void print_mat(const mats::Mat<T>& mc)
{
	cout << "[ ";
	for(int i = 0; i < mc.rows; i++){
		for(int j = 0; j < mc.cols; j++){
			cout << mc.data[i * mc.cols + j] << "\t";
		}
		cout << ";\n  ";
	}
	cout << "]\n";
}

template<class T>
void print_mat(const mats::Mat<T>& mc, const std::string& caption, const std::string& fn)
{
	fstream fs;
	fs.open(fn, ios_base::in | ios_base::app);

	fs << "<<<<<< " << caption.c_str() << " >>>>>>>\n" <<"[ ";
	for(int i = 0; i < mc.rows; i++){
		for(int j = 0; j < mc.cols; j++){
			fs << mc.data[i * mc.cols + j] << "\t";
		}
		fs << ";\n  ";
	}
	fs << "]\n";
	fs.close();
}

int main(int argc, char *argv[])
{
	using namespace mats;

	// run your cuda application
	cudaError_t cuerr = cuda_main();
	// check for errors is always a good practice!
	if (cuerr != cudaSuccess) cout << "CUDA Error: " << cudaGetErrorString( cuerr ) << endl;

	double a[] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		2, 5, 7, 1,
		5, 8, 2, 9,
		4, 7, 6, 1
	};
	double b[] = {
		5, 2, 7, 3, 1, 6,
		7, 2, 3, 5, 1, 7,
		3, 4, 7, 2, 9, 1,
		4, 2, 8, 5, 6, 9
	};

//	Mat ma(5, 4, a), mb(4, 6, b), mc(5, 6);
	Mat<float> ma(32, 1000), mb(1000, 11), mc(32, 11);

	for(int i = 0; i < ma.rows; i++){
		for(int j = 0; j < ma.cols; j++){
			ma.at(i, j) = (float)i / ma.rows + (float)j / ma.cols;
		}
	}

	for(int i = 0; i < mb.rows; i++){
		for(int j = 0; j < mb.cols; j++){
			mb.at(i, j) = (float)(mb.rows - i) / mb.rows + (float)(mb.cols / 2. - j) / mb.cols;
		}
	}

	cudaError_t err = cudaSuccess;

	err = cuda_mult(&ma, &mb, &mc);

	print_mat(ma, "A", "mat.txt");
	print_mat(mb, "B", "mat.txt");
	print_mat(mc, "C", "mat.txt");

	cout << err << " " << "Hello World!" << endl;
	return 0;
}
