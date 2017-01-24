#include <iostream>

#include <fstream>
#include <ctime>
#include <chrono>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "mats.h"

#include "gpumat.h"

#include <custom_types.h>

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

#ifdef _MSC_VER
#else

double tick()
{
	struct timespec res;
	clock_gettime(CLOCK_MONOTONIC, &res);
	double dres = res.tv_nsec + res.tv_sec * 1e9;
	return (double)dres / 1000.;
}

#endif

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

void test_cuda()
{
#define TEST_VOID(type, result, _void_, caption) {	\
	type result;									\
	double tc = tick();								\
	_void_;											\
	tc = tick() - tc;								\
	std::string s = (std::string)result();			\
	std::cout << caption << " time: " << tc;		\
/*	std::cout << endl << s.c_str();*/				\
	std::cout << endl;								\
}

#define PRINT_MAT(result, caption)	{				\
	std::string s = (std::string)result();			\
	std::cout << caption;							\
	/*std::cout << endl << s.c_str();*/				\
	std::cout << endl;								\
}

#define CALC_MAT(_void_, result, caption, count){	\
	double tcc = 0;									\
	for(int i = 0; i < count; ++i){					\
		double tc = tick();							\
		_void_;										\
		tcc += tick() - tc;							\
	}												\
	tcc /= count;									\
	std::string s = (std::string)result();			\
	std::cout << caption << " time: " << tcc;		\
	/*std::cout << endl << s.c_str();*/				\
	std::cout << endl;								\
}

	ct::Matf A(2000, 180), B(180, 1400), C(2000, 180);

	for(int i = 0; i < A.total(); i++){
		A.ptr()[i] = i/100.;
		C.ptr()[i] = (11. + C.total() - i)/100.;
	}
	for(int i = 0; i < B.total(); i++){
		B.ptr()[i] = i/100.;
	}

//	A.randn(0, 1, 0);
//	B.randn(0, 1, 1);
//	C.randn(0, 1, 2);

	gpumat::GpuMat gA(A.rows, A.cols, gpumat::GPU_FLOAT, A.ptr()), g_tmp;
	gpumat::GpuMat gC(C.rows, C.cols, gpumat::GPU_FLOAT, C.ptr());
	gpumat::GpuMat gB(B.rows, B.cols, gpumat::GPU_FLOAT, B.ptr());
	gpumat::GpuMat R;

	double gv1(3.), gv2(0.001);

	PRINT_MAT(gA, "A");
	PRINT_MAT(gB, "B");
	PRINT_MAT(gC, "C");
	std::cout << "----\n";
	TEST_VOID(gpumat::GpuMat, R, gpumat::add(gA, gC, R, 1, 1), "A + C");
	TEST_VOID(gpumat::GpuMat, R, gpumat::sub(gA, gC, R), "A - C");
	TEST_VOID(gpumat::GpuMat, R, gpumat::addval(gA, gv1, R), "A + 3");
	TEST_VOID(gpumat::GpuMat, R, gpumat::subval(gA, gv1, R), "A - 3");
	TEST_VOID(gpumat::GpuMat, R, gpumat::subval(gv1, gA, R), "3 - A");
	TEST_VOID(gpumat::GpuMat, R, gpumat::mulval(gA, gv1, R), "A * 3");
	TEST_VOID(gpumat::GpuMat, R, gpumat::elemwiseMult(gA, gC, R), "A .* C");
	std::cout << "----\n";

	R.resize(1, gA.cols, gA.type);
	CALC_MAT(gpumat::sumRows(gA, R), R, "sumrows(A)", 10);
	CALC_MAT(gpumat::sumRows(gA, R), R, "sumrows(A)", 10);
	CALC_MAT(gpumat::sumRows_shared(gA, R), R, "sumrows(A) (shared)", 10);
	std::cout << "----\n";
	TEST_VOID(gpumat::GpuMat, T, gpumat::transpose(gA, T), "A'");

	g_tmp = gA;
	CALC_MAT(gpumat::addval(g_tmp, gv1), g_tmp, "Atmp + 3", 10);
	g_tmp = gA;
	CALC_MAT(gpumat::subval(g_tmp, gv1), g_tmp, "Atmp - 3", 10);
	g_tmp = gA;
	CALC_MAT(gpumat::subval(gv1, g_tmp), g_tmp, "3 - Atmp", 10);

	gpumat::GpuMat gAt, gBt, partZ;

	gpumat::transpose(gA, gAt);
	gpumat::transpose(gB, gBt);
	std::cout << "----\n";
	R.resize(gA.rows, gB.cols, gA.type);
	CALC_MAT(gpumat::matmul(gA, gB, R), R, "A * B", 100);
	CALC_MAT(gpumat::matmul_shared(gA, gB, R), R, "A * B (shared)", 100);
	R.resize(gAt.cols, gB.cols, gA.type);
	CALC_MAT(gpumat::matmulT1(gAt, gB, R), R, "At * B", 100);
	CALC_MAT(gpumat::matmulT1_shared(gAt, gB, R), R, "At * B (shared)", 100);
	R.resize(gA.rows, gBt.rows, gA.type);
	CALC_MAT(gpumat::matmulT2(gA, gBt, R), R, "A * Bt", 100);
	CALC_MAT(gpumat::matmulT2_shared(gA, gBt, R), R, "A * Bt (shared)", 100);
	std::cout << "----\n";
	PRINT_MAT(gA, "A");
	gB.resize(gA);
	CALC_MAT(gpumat::mulval(gA, gv2, gB), gB, "B", 10);
	R.resize(gB);
	partZ.resize(gB.rows, 1, gB.type);
	TEST_VOID(gpumat::GpuMat, R, gpumat::softmax(gB, 1, R, partZ), "softmax");
	PRINT_MAT(partZ, "partZ");

	gA.ones();
	gB.ones();

	TEST_VOID(gpumat::GpuMat, R, gpumat::matmul(gA, gB, R), "A(1) * B(1)");
}

int main(int argc, char *argv[])
{
	using namespace mats;

#if 0
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
	Mat<float> ma(32, 45), mb(45, 11), mc(32, 11);

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
#endif

	test_cuda();

	return 0;
}
