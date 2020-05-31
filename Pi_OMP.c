/*
 * BIG DATA / PARALLEL COMPUTING
 *
 * Lab 2: OMP
 * Example 1: Pi
 *
 * Leonard Mada
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <random>
#define MAXTHRDS 124

// cd "\Parallel\src"
// g++ Pi_OMP.c -o Pi_OMP.exe -fopenmp

// Note: the code mixes some C++ into the C ;-)

// Problems:
// - cannot compute an accurate result
//   when using the random number generator std::mt19937;
// - uniform_real_distribution is marginally better than
//   uniform_int_distribution, but it is slower;
// - balls in higher dimemsion do NOT offer any benefits;
// TODO:
// - investigate why it fails!
// - investigate balls in higher dimensions;

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
		// reutn CLOCKS_PER_SEC;
		return 1;
	}
};

// Test with sizes: 10 M, 40 M, 160 M;
// [4 threads]
int main(int argc, char *argv[])  {
	const double PI25DT = 3.141592653589793238462643;
	
	unsigned int npoints, i;
	unsigned int sum = 0;
	double rand_x, rand_y, rand_z, rand_z4; // z: for 3D & 4D variants
	// for RNG
	const unsigned int seed = 1234;
	
	const timer_st timer = timer_st();
	
	const unsigned int num_of_threads = omp_get_num_procs();
	omp_set_num_threads(num_of_threads);
	// num_of_threads = omp_get_num_threads();
	
	printf("N Iterations = ");
	
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
	printf("Pi [Real RND] = %f\n\n", dPI2);
	
	
	// ++++ Parallel 3D ++++
	timer.start();
	sum = 0;
	#pragma omp parallel default(none) \
					private(rand_x, rand_y, rand_z, i) \
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
			rand_z = ((double) uint_dist(rnd)) / MAX;
			if (((rand_x - 0.5) * (rand_x - 0.5) +
				 (rand_y - 0.5) * (rand_y - 0.5) +
				 (rand_z - 0.5) * (rand_z - 0.5)) <= 0.25) {
				sum ++;
			}
		}
		printf("Sum = %u\n", sum);
	}
	const double dPI3D = ((double) sum) * 3 * 2 / npoints;
	timer.end();
	timer.print();
	printf("Pi 3D = %f\n\n", dPI3D);
	
	
	// ++++ Parallel 4D ++++
	timer.start();
	sum = 0;
	#pragma omp parallel default(none) \
					private(rand_x, rand_y, rand_z, rand_z4, i) \
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
			rand_z = ((double) uint_dist(rnd)) / MAX;
			rand_z4 = ((double) uint_dist(rnd)) / MAX;
			if (((rand_x - 0.5) * (rand_x - 0.5) +
				 (rand_y - 0.5) * (rand_y - 0.5) +
				 (rand_z - 0.5) * (rand_z - 0.5) +
				 (rand_z4 - 0.5) * (rand_z4 - 0.5)) <= 0.25) {
				sum ++;
			}
		}
		printf("Sum = %u\n", sum);
	}
	const double dPI4D = sqrt(((double) sum) * 2 * 16 / npoints);
	timer.end();
	timer.print();
	printf("Pi 4D = %f\n\n", dPI4D);
	
	// =====================
	// ++++ Without RNG ++++
	
	// ++++ Parallel without RNG ++++
	{
	timer.start();
	const unsigned int sqLen = (unsigned int) sqrt(npoints);
	const unsigned int sample_points = sqLen * sqLen;
	// TODO: can loose a few pixels
	const unsigned int sample_points_per_thread = sample_points / num_of_threads;
	sum = 0;
	#pragma omp parallel default(none) private(rand_x, rand_y, i) reduction(+: sum)
	{
		const unsigned int idThread = omp_get_thread_num();
		const unsigned int lenMax = (idThread + 1) * sample_points_per_thread;
		
		for (i = idThread * sample_points_per_thread; i < lenMax; i++) {
			rand_x = ((double) ((int)(i / sqLen)) ) / sqLen;
			rand_y = ((double) (i % sqLen)) / sqLen;
			if (((rand_x - 0.5) * (rand_x - 0.5) +
				 (rand_y - 0.5) * (rand_y - 0.5)) <= 0.25) {
				sum ++;
			}
		}
		printf("Thread = %u, Sum = %u\n", idThread, sum);
	}
	const double dPI3 = ((double) sum) * 4 / sample_points;
	timer.end();
	timer.print();
	printf("Non-Random Pi = %f\n\n", dPI3);
	}
	
	// ++++ 3D without RNG ++++
	{
	timer.start();
	const int p3Len = (int) pow(npoints, 1.0d / 3);
	printf("R = %d\n", p3Len);
	const unsigned int sample_points = p3Len * p3Len * p3Len;
	// TODO: can loose a few pixels
	const unsigned int sample_points_per_thread = sample_points / num_of_threads;
	sum = 0;
	#pragma omp parallel default(none) private(rand_x, rand_y, rand_z, i) reduction(+: sum)
	{
		const unsigned int idThread = omp_get_thread_num();
		const unsigned int lenMax = (idThread + 1) * sample_points_per_thread;
		const int sqLen = p3Len * p3Len;
		
		for (i = idThread * sample_points_per_thread; i < lenMax; i++) {
			int x = (int)(i / sqLen);
			rand_x = ((double) x) / p3Len;
			int yz = i - x * sqLen; // %
			rand_y = ((double) ((int) (yz / p3Len)) ) / p3Len;
			rand_z = ((double) (yz % p3Len)) / p3Len;
			if (((rand_x - 0.5) * (rand_x - 0.5) +
				 (rand_y - 0.5) * (rand_y - 0.5) +
				 (rand_z - 0.5) * (rand_z - 0.5)) <= 0.25) {
				sum ++;
			}
		}
		printf("Thread = %u, Sum = %u\n", idThread, sum);
	}
	const double dPI3 = ((double) sum) * 6 / sample_points;
	timer.end();
	timer.print();
	// ~1/3 slower, BUT 4th digit accuracy does NOT converge!
	printf("Non-Random Pi 3D = %f\n", dPI3);
	}
	
	// ===============
	
	// END
	timer.destruct();
}
