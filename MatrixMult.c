
/*
 * BIG DATA / PARALLEL COMPUTING
 *
 * Lab 3: OMP
 * Matrix Multiplication
 *
 * Leonard Mada
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define MAXTHRDS 124

// cd ".\Parallel"
// g++ MatrixMult.c -o MatrixMult.exe -fopenmp

/*

sz = 80
m1 = matrix(1, ncol=sz, nrow=sz)
m2 = m1
# m2 = matrix(1:sz, ncol=sz, nrow=sz, byrow=T)
m1 %*% m2

*/


struct timer_st {
	// double start_time, total_time;
	double *const start_time = (double *const) malloc(sizeof(double));
	double *const total_time = (double *const) malloc(sizeof(double));
	timer_st() {}
	void start() const {
		// ( const_cast <timer_st*> (this) ) -> start_time = clock();
		*start_time = clock();
	}
	void end() const {
		// ( const_cast <timer_st*> (this) ) -> total_time = clock() - start_time;
		*total_time = clock() - *start_time;
	}
	void print() const {
		printf("Time = %f\n", (*total_time) / CLOCKS_PER_SEC);
	}
};

// Test with sizes: 2000 x 2000;
// [4 threads]
int main(int argc, char *argv[])  {
	int num_of_thrds;
	
	const timer_st timer = timer_st();
	
	int m_len, i;
	float **m1, **m2, **mR;
	int nRow, nCol, nRowCol;
	float rez;
	
	// Threads
	num_of_thrds = omp_get_num_procs();
	omp_set_num_threads(num_of_thrds);
	
	printf("Matrix size (square matrix) = ");
	
	if(scanf("%d", &m_len) < 1) {
		printf("Check input for vector length. Bye.\n");
		return -1;
	}
	
	m1 = (float**) malloc(m_len*sizeof(float*));
	m2 = (float**) malloc(m_len*sizeof(float*));
	mR = (float**) malloc(m_len*sizeof(float*));
	
	for(i=0; i < m_len; i++) {
		m1[i] = (float*) malloc(m_len*sizeof(float));
		m2[i] = (float*) malloc(m_len*sizeof(float));
		mR[i] = (float*) malloc(m_len*sizeof(float));
	}
	// initialize with values
	for(nRow=0; nRow < m_len; nRow++) {
		for(nCol=0; nCol < m_len; nCol++) {
			m1[nRow][nCol] = 1;
			m2[nRow][nCol] = 1;
			mR[nRow][nCol] = 0;
		}
	}
	
	// Parallel with private
	timer.start();
	
	#pragma omp parallel for private(nCol, nRowCol, rez) // collapse(2)
	for(nRow=0; nRow < m_len; nRow++) {
		for(nCol=0; nCol < m_len; nCol++) {
			rez = 0;
			for(nRowCol=0; nRowCol < m_len; nRowCol++) {
				rez += m1[nRow][nRowCol] * m2[nRowCol][nCol];
			}
			mR[nRow][nCol] = rez;
			// printf("R = %f\n", rez);
		}
	}
	timer.end();
	timer.print();
	printf("R = %f\nThreads = %d\n", 0, num_of_thrds);
	
	
	// END
	free(m1);
	free(m2);
	free(mR);
}
