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
cudaError_t cuda_mult(mats::Mat* a, mats::Mat* b, mats::Mat *c);

//////////////////////////////

int64_t getTick()
{
	using namespace std::chrono;
	milliseconds res = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
	return res.count();
}

//////////////////////////////

void print_mat(const mats::Mat& mc)
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

void print_mat(const mats::Mat& mc, const std::string& caption, const std::string& fn)
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
	Mat ma(32, 1000), mb(1000, 11), mc(32, 11);

	for(int i = 0; i < ma.rows; i++){
		for(int j = 0; j < ma.cols; j++){
			ma.at(i, j) = (double)i / ma.rows + (double)j / ma.cols;
		}
	}

	for(int i = 0; i < mb.rows; i++){
		for(int j = 0; j < mb.cols; j++){
			mb.at(i, j) = (double)(mb.rows - i) / mb.rows + (double)(mb.cols / 2. - j) / mb.cols;
		}
	}

	const int test_count = 10000;

	cudaError_t err = cudaSuccess;

	int64_t  tick = getTick();
	for(int i = 0; i < test_count && err == cudaSuccess; ++i){
		err = cuda_mult(&ma, &mb, &mc);
	}
	cout << "time=" << getTick() - tick << endl;

	print_mat(ma, "A", "mat.txt");
	print_mat(mb, "B", "mat.txt");
	print_mat(mc, "C", "mat.txt");

	tick = getTick();
	for(int i = 0; i < test_count; ++i){
		mc = matMult(ma, mb);
	}
	print_mat(mc, "C", "mat1.txt");

	cout << "time=" << getTick() - tick << endl;


	cout << err << " " << "Hello World!" << endl;
	return 0;
}
