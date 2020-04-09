/*
 * BIG DATA / PARALLEL COMPUTING
 *
 * Lab 2: OMP
 * Example 2: Dot Product
 *
 * Leonard Mada
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define MAXTHRDS 124

// cd ".\Parallel"
// g++ VectorTest.c -o VectorTest.exe -fopenmp


// Timer
struct timer_st {
	// double start_time, total_time;
	double *const start_time = (double *const) malloc(sizeof(double));
	double *const total_time = (double *const) malloc(sizeof(double));
	timer_st() {}
	void start() const {
		// ( const_cast <timer_st*> (this) ) -> start_time = clock();
		*start_time = time();
	}
	void end() const {
		// ( const_cast <timer_st*> (this) ) -> total_time = clock() - start_time;
		*total_time = time() - *start_time;
	}
	void print() const {
		printf("Time = %f\n", (*total_time) / div());
	}
	void destruct() const {
		free(start_time);
		free(total_time);
	}
	double time() const {
		// return clock();
		return omp_get_wtime();
	}
	double div() const {
		// return CLOCKS_PER_SEC;
		return 1;
	}
};


// Test with sizes: 50 M, 80 M, 100 M;
// [4 threads]
int main(int argc, char *argv[])  {
	// !!! VERY SLOW !!!
	const bool TEST_LOCK = false;
	
	double *x, *y, dot_prod;
	int num_of_thrds, vec_len, i;
	
	const timer_st timer = timer_st();
	
	num_of_thrds = omp_get_num_procs();
	omp_set_num_threads(num_of_thrds);
	
	printf("Vector length = ");
	
	if(scanf("%d", &vec_len) < 1) {
		printf("Check input for vector length. Bye.\n");
		return -1;
	}
	
	x = (double*) malloc(vec_len*sizeof(double));
	y = (double*) malloc(vec_len*sizeof(double));
	
	for(i=0; i < vec_len; i++) {
		x[i] = 1. + i;
		y[i] = 1.;
	}
	for(i=0; i < vec_len; i++) {
		x[i]; y[i]; // "hot" memory
	}
	
	// Parallel with Reduction
	timer.start();
	dot_prod = 0.;
	#pragma omp parallel for reduction(+: dot_prod)
	for(i=0; i<vec_len; i++) {
		dot_prod += x[i]*y[i];
	}
	timer.end();
	timer.print();
	printf("Dot product with R = %f\n", dot_prod);
	
	// Parallel without Reduction
	timer.start();
	dot_prod = 0.;
	#pragma omp parallel for // reduction(+: dot_prod)
	for(i=0; i<vec_len; i++) {
		dot_prod += x[i]*y[i];
	}
	timer.end();
	timer.print();
	printf("Dot product w/o R  = %f\n", dot_prod);
	
	// Serial
	timer.start();
	dot_prod = 0.;
	for(i=0; i<vec_len; i++) {
		double temp = x[i]*y[i];
		dot_prod += temp;
	}
	timer.end();
	timer.print();
	printf("Dot product serial = %f\n", dot_prod);
	
	// Parallel with atomic operation
	timer.start();
	dot_prod = 0.;
	#pragma omp parallel for // reduction(+: dot_prod)
	for(i=0; i<vec_len; i++) {
		double temp = x[i]*y[i];
		#pragma omp atomic // update // update is slower
		dot_prod += temp;
	}
	timer.end();
	timer.print();
	printf("Dot product atomic = %f\n", dot_prod);
	
	// Parallel with Lock
	if(TEST_LOCK) {
		// !!! very slow !!!
	omp_lock_t writelock;
	omp_init_lock(&writelock);
	timer.start();
	dot_prod = 0.;
	#pragma omp parallel for // reduction(+: dot_prod)
	for(i=0; i<vec_len; i++) {
		double temp = x[i]*y[i];
		omp_set_lock(&writelock);
		dot_prod += temp;
		omp_unset_lock(&writelock);
	}
	timer.end();
	timer.print();
	printf("Dot product w lock = %f\n", dot_prod);
	omp_destroy_lock(&writelock);
	}
	
	// END
	free(x);
	free(y);
	free(timer);
} 


