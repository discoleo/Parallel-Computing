/*
 * BIG DATA & PARALLEL COMPUTING
 *
 * Lab 1 BD: OMP
 * Computing Pi
 *
 * Leonard Mada
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <random>
#define MAXTHRDS 124

// cd "...\Parallel\src"
// g++ Pi_OMP.c -o Pi_OMP.exe -fopenmp

// Note: the code mixes some C++ into the C ;-)

// Problems:
// - cannot compute an accurate result
//   when using the random number generator std::mt19937;
// - uniform_real_distribution is marginally better than
//   uniform_int_distribution, but it is slower;
// TODO: investigate why it fails!

// Timer
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
	void destruct() const {
		free(start_time);
		free(total_time);
	}
};

// Test with sizes: 10 M, 40 M, 160 M;
// [4 threads]
int main(int argc, char *argv[])  {
	const double PI25DT = 3.141592653589793238462643;
	
	unsigned int npoints, i;
	unsigned int sum = 0;
	double rand_x, rand_y;
	// for RNG
	const unsigned int seed = 1234;
	
	const timer_st timer = timer_st();
	
	const unsigned int num_of_threads = omp_get_num_procs();
	omp_set_num_threads(num_of_threads);
	// num_of_threads = omp_get_num_threads();
	
	printf("Vector length = ");
	
	if(scanf("%d", &npoints) < 1) {
		printf("Check input for length of simulation. Bye.\n");
		return -1;
	}
	
	// Parallel with Reduction
	timer.start();
	sum = 0;
	#pragma omp parallel default(none) \
					private(rand_x, rand_y, i) \
					shared(npoints) reduction(+: sum)
	{
		const unsigned int sample_points_per_thread = npoints / num_of_threads;
		// Rng
		std::mt19937 rnd;
		const int idThread = omp_get_thread_num();
		rnd.seed(seed * idThread * idThread);
		std::uniform_int_distribution<uint32_t> uint_dist;
		const double MAX = uint_dist.max();
		// const double MIN = uint_dist.min();
		// printf("MAX = %f\n", MAX);
		
		for (i = 0; i < sample_points_per_thread; i++) {
			rand_x = ((double) uint_dist(rnd)) / MAX;
			rand_y = ((double) uint_dist(rnd)) / MAX;
			if (((rand_x - 0.5) * (rand_x - 0.5) +
				 (rand_y - 0.5) * (rand_y - 0.5)) <= 0.25) {
				sum ++;
			}
		}
		printf("Sum = %u\n", sum);
	}
	const double dPI = ((double) sum) * 4 / npoints;
	timer.end();
	timer.print();
	printf("Pi = %f\n", dPI);
	
	// Parallel with Reduction; Real RNG;
	timer.start();
	sum = 0;
	#pragma omp parallel default(none) private(rand_x, rand_y, i) shared(npoints) reduction(+: sum)
	{
		const unsigned int sample_points_per_thread = npoints / num_of_threads;
		// RNG
		std::mt19937 rnd;
		const int idThread = omp_get_thread_num();
		rnd.seed(seed * idThread * idThread);
		std::uniform_real_distribution<double> uint_dist(0, 1);
		
		for (i = 0; i < sample_points_per_thread; i++) {
			rand_x = ((double) uint_dist(rnd));
			rand_y = ((double) uint_dist(rnd));
			if (((rand_x - 0.5) * (rand_x - 0.5) +
				 (rand_y - 0.5) * (rand_y - 0.5)) <= 0.25) {
				sum ++;
			}
		}
		printf("Sum = %u\n", sum);
	}
	const double dPI2 = ((double) sum) * 4 / npoints;
	timer.end();
	timer.print();
	printf("Pi = %f\n", dPI2);
	
	// END
	timer.destruct();
}
