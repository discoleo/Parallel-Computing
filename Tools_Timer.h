/*
 * BIG DATA / PARALLEL COMPUTING
 *
 * Tools: Timer
 *
 * Leonard Mada
*/

#ifndef TIMER_TOOLS_H
#define TIMER_TOOLS_H


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

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
	double print() const {
		const double dTime = (*total_time) / div();
		printf("Time = %f\n", dTime);
		return dTime;
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

#endif
