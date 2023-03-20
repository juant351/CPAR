/*
 * Probabilistic approach to locate maximum heights
 * Hill Climbing + Montecarlo
 *
 	Código realizado por Héctor Toribio González y Juan Torres Viloria
 
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2021/2022
 *
 * v1.1
 *
 * (c) 2022 Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the CUDA assignment versions */
#include<cuda.h>







/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}


#define	PRECISION	10000

/*
 * Structure to represent a climbing searcher
 * 	This structure can be changed and/or optimized by the students
 */
typedef struct {
	int id;				// Searcher identifier
	int pos_row, pos_col;		// Position in the grid
	int steps;			// Steps count
	int follows;			// When it finds an explored trail, who searched that trail
} Searcher;




/*
 * Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]


/*
 * Function: Generate height for a given position
 * 	This function can be changed and/or optimized by the students
 */
__device__ int get_height( int x, int y, int rows, int columns, float x_min, float x_max, float y_min, float y_max  ) {
	/* Calculate the coordinates of the point in the ranges */
	float x_coord = x_min + ( (x_max - x_min) / rows ) * x;
	float y_coord = y_min + ( (y_max - y_min) / columns ) * y;
	/* Compute function value */
	float value = 2 * sin(x_coord) * cos(y_coord/2) + log( fabs(y_coord - M_PI_2) );
	/* Transform to fixed point precision */
	int fixed_point = (int)( PRECISION * value );
	return fixed_point;
}

/*
 * Function: Climbing step
 * 	This function can be changed and/or optimized by the students
 */
__device__ int climbing_step( int rows, int columns, Searcher *searchers, int search, int *heights, int *trails, int *tainted, float x_min, float x_max, float y_min, float y_max ) {
	int search_flag = 0;

	/* Annotate one step more, landing counts as the first step */
	searchers[ search ].steps ++;

	/* Get starting position */
	int pos_row = searchers[ search ].pos_row;
	int pos_col = searchers[ search ].pos_col;

	/* Stop if searcher finds another trail */
	int check;

	check = atomicCAS(&accessMat( tainted, pos_row, pos_col ), 0, 1);

	if ( check != 0 ) {
		search_flag = 1;
	}
	else {
		/* Annotate the trail */
		accessMat( trails, pos_row, pos_col ) = search;

		/* Compute the height */
		accessMat( heights, pos_row, pos_col ) = get_height( pos_row, pos_col, rows, columns, x_min, x_max, y_min, y_max );

		/* Locate the highest climbing direction */
		float local_max = accessMat( heights, pos_row, pos_col );
		int climbing_direction = 0;
		if ( pos_row > 0 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, pos_row-1, pos_col ) == INT_MIN )
				accessMat( heights, pos_row-1, pos_col ) = get_height( pos_row-1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, pos_row-1, pos_col ) > local_max ) {
				climbing_direction = 1;
				local_max = accessMat( heights, pos_row-1, pos_col );
			}
		}
		if ( pos_row < rows-1 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, pos_row+1, pos_col ) == INT_MIN )
				accessMat( heights, pos_row+1, pos_col ) = get_height( pos_row+1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, pos_row+1, pos_col ) > local_max ) {
				climbing_direction = 2;
				local_max = accessMat( heights, pos_row+1, pos_col );
			}
		}
		if ( pos_col > 0 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, pos_row, pos_col-1 ) == INT_MIN )
				accessMat( heights, pos_row, pos_col-1 ) = get_height( pos_row, pos_col-1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, pos_row, pos_col-1 ) > local_max ) {
				climbing_direction = 3;
				local_max = accessMat( heights, pos_row, pos_col-1 );
			}
		}
		if ( pos_col < columns-1 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, pos_row, pos_col+1 ) == INT_MIN )
				accessMat( heights, pos_row, pos_col+1 ) = get_height( pos_row, pos_col+1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, pos_row, pos_col+1 ) > local_max ) {
				climbing_direction = 4;
				local_max = accessMat( heights, pos_row, pos_col+1 );
			}
		}

		/* Stop if local maximum is reached */
		if ( climbing_direction == 0 ) {
			searchers[ search ].follows = search;
			search_flag = 1;
		}

		/* Move in the chosen direction: 0 does not change coordinates */
		switch( climbing_direction ) {
			case 1: pos_row--; break;
			case 2: pos_row++; break;
			case 3: pos_col--; break;
			case 4: pos_col++; break;
		}
		searchers[ search ].pos_row = pos_row;
		searchers[ search ].pos_col = pos_col;
	}

	/* Return a flag to indicate if search should stop */
	return search_flag;
}


#ifdef DEBUG
/*
 * Function: Print the current state of the simulation
 */
void print_heights( int rows, int columns, int *heights ) {
	/*
	 * You don't need to optimize this function, it is only for pretty
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Heights:\n");
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( heights, i, j ) != INT_MIN )
				printf(" %6d", accessMat( heights, i, j ) );
			else
				printf("       ");
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n\n");
}

void print_trails( int rows, int columns, int *trails ) {
	/*
	 * You don't need to optimize this function, it is only for pretty
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Trails:\n");
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( trails, i, j ) != -1 )
				printf("%7d", accessMat( trails, i, j ) );
			else
				printf("       ", accessMat( trails, i, j ) );
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("-------");
	printf("+\n\n");
}
#endif // DEBUG

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <x_min> <x_max> <y_min> <y_max> <searchers_density> <short_rnd1> <short_rnd2> <short_rnd3>\n");
	fprintf(stderr,"\n");
}


//KERNEL PARA INICIALIZAR LOS BUSCADORES (bucle 3.3)
__global__ void  kernel_InitSearchers(Searcher *searchersDevice, int *total_stepsDevice, int num_searchers){

	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	if ( globalPos < num_searchers ) {
		//searchersDevice[ globalPos ].id = globalPos;
		searchersDevice[ globalPos ].steps = 0;
		searchersDevice[ globalPos ].follows = -1;
		total_stepsDevice[ globalPos ] = 0;
	}
}

//KERNEL PARA INICIALIZAR EL TERRENO. (bucle 3.2)
/*
__global__ void kernel_InitTerrain(Searcher *searchersDevice, int *heightsDevice, int *trailsDevice, int *taintedDevice){
	arr[ (int)(exp1) * columns + (int)(exp2) ]
}
*/

//KERNEL PARA LA FUNCION CLIMBING_STEP (bucle 4)
__global__ void  kernel_climbStep(Searcher *searchersDevice, int *taintedDevice, int *heightsDevice,int *trailsDevice, int rows, int columns, float x_min, float x_max, float y_min, float y_max, int num_searchers){
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("posicion global: %d ", globalPos);
	int search_flag = 0;
	if(globalPos<num_searchers){
			while( ! search_flag ) {
				search_flag = climbing_step( rows, columns, searchersDevice, globalPos, heightsDevice, trailsDevice, taintedDevice, x_min, x_max, y_min, y_max );
			}
	}

}



//KERNEL PARA FOLLOWERS (bucle5)
__global__ void kernel_ComputeFollowers(Searcher *searchersDevice, int num_searchers){

	int globalPos = threadIdx.x + (blockIdx.x * blockDim.x);

	if ( globalPos < num_searchers ) {
		int search_flag = 0;
		int parent = globalPos;
		int follows_to = searchersDevice[ parent ].follows;
		while( ! search_flag ) {
			if ( follows_to == parent ) search_flag = 1;
			else {
				parent = follows_to;
				follows_to = searchersDevice[ parent ].follows;
			}
		}
		searchersDevice[ globalPos ].follows = follows_to;
	}
}

//KERNEL PARA STEPS (bucle 6)
__global__ void  kernel_Trail(Searcher *searchersDevice, int *total_stepsDevice, int num_searchers){
	int globalPos = threadIdx.x + (blockIdx.x * blockDim.x);

	if(globalPos<num_searchers){
		int pos_max = searchersDevice[ globalPos ].follows;
		atomicAdd(&total_stepsDevice[ pos_max ], searchersDevice[ globalPos ].steps);
	}
}


__global__ void  kernel_condicionCarrera(Searcher *searchersDevice, int *trailsDevice, int num_searchers, int columns){
	int globalPos = threadIdx.x + (blockIdx.x * blockDim.x);

	if(globalPos<num_searchers){
		searchersDevice[ globalPos ].follows = accessMat( trailsDevice, searchersDevice[ globalPos].pos_row, searchersDevice[ globalPos ].pos_col );
	}
}

__global__ void  kernel_SumaTainted (int *tainted, int num_searchers, int *suma){

	// Compute the global position of the thread in the grid
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	// Shared memory: One element per thread in the block
	// Call this kernel with the proper third launching parameter
	extern __shared__ int buffer[ ];

	// Load array values in the shared memory (0 if out of the array)
	if ( globalPos < num_searchers ) {
		buffer[ threadIdx.x ] = tainted[ globalPos ];
	}
	else buffer[ threadIdx.x ] = 0;

	// Wait for all the threads of the block to finish
	__syncthreads();

	// Reduction tree
	for( int step=blockDim.x/2; step>=1; step /= 2 ) {
		if ( threadIdx.x < step )
				buffer[ threadIdx.x ] += buffer[ threadIdx.x + step ];
		__syncthreads();
	}

	// The maximum value of this block is on the first position of buffer
	if ( threadIdx.x == 0 )

		atomicAdd( suma, buffer[0] );
}

__global__ void kernel_SumaHeights(int* array, int size, unsigned long long int *result){
    // Compute the global position of the thread in the grid
    int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory: One element per thread in the block
    // Call this kernel with the proper third launching parameter
    extern __shared__ int buffer[ ];

    // Load array values in the shared memory (0 if out of the array)
    if ( globalPos < size) { //and array[globalPos] != INT_MIN
        if( array[globalPos] != INT_MIN ) {
            buffer[ threadIdx.x ] = array[ globalPos ];
        } else {
            buffer[ threadIdx.x ] = 0;
        }
    }
    else buffer[ threadIdx.x ] = 0;

    // Wait for all the threads of the block to finish
    __syncthreads();

    // Reduction tree
    for( int step=blockDim.x/2; step>=1; step /= 2 ) {
        if ( threadIdx.x < step )
            buffer[ threadIdx.x ] += buffer[ threadIdx.x + step ];
        __syncthreads();
    }

    // The maximum value of this block is on the first position of buffer
    if ( threadIdx.x == 0 )
        atomicAdd( result, buffer[0] );
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	// This eliminates the buffer of stdout, forcing the messages to be printed immediately
	setbuf(stdout,NULL);

	int i,j, search;



	// Simulation data
	int rows, columns;		// Matrix sizes
	float x_min, x_max;		// Limits of the terrain x coordinates
	float y_min, y_max;		// Limits of the terrain y coordinates

	float searchers_density;	// Density of hill climbing searchers
	unsigned short random_seq[3];	// Status of the random sequence

	int *heights;			// Heights of the terrain points
	int *trails;			// Searchers trace and trails
	int *tainted;			// Position found in a search
	int num_searchers;		// Number of searchers
	Searcher *searchers;		// Searchers data
	int *total_steps;		// Annotate accumulated steps to local maximums

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc != 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	x_min = atof( argv[3] );
	x_max = atof( argv[4] );
	y_min = atof( argv[5] );
	y_max = atof( argv[6] );
	searchers_density = atof( argv[7] );

	/* 1.3. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		random_seq[i] = (unsigned short)atoi( argv[8+i] );
	}


#ifdef DEBUG
	/* 1.4. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
	printf("Arguments, x_range: ( %d, %d ), y_range( %d, %d )\n", x_min, x_max, y_min, y_max );
	printf("Arguments, searchers_density: %f\n", searchers_density );
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", random_seq[0], random_seq[1], random_seq[2]);
	printf("\n");
#endif // DEBUG



	/* 2. Start global timer */
	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	/* 3. Initialization */
	/* 3.1. Memory allocation */

	num_searchers = (int)( rows * columns * searchers_density );


	int NHILOS = 1024;
	int NHILOS_REDUCTION = 1024;
	int NBLOQUES = num_searchers/NHILOS;
	int NBLOQUES_REDUCTION = (rows *columns) / NHILOS_REDUCTION;

	if(num_searchers % NHILOS != 0){
		NBLOQUES ++;
	}
	
	if((rows *columns) % NHILOS_REDUCTION != 0){
		NBLOQUES_REDUCTION ++;
	}




	searchers = (Searcher *)malloc( sizeof(Searcher) * num_searchers );
	total_steps = (int *)malloc( sizeof(int) * num_searchers );
	heights = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	trails = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	tainted = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );

	Searcher *searchersDevice;
	int *trailsDevice;
	int *total_stepsDevice;
	int *heightsDevice;
	int *taintedDevice;


	cudaMalloc( (void**) &searchersDevice, sizeof(Searcher) * num_searchers);CHECK_CUDA_LAST();
	cudaMalloc( (void**) &total_stepsDevice, sizeof(int) * num_searchers);CHECK_CUDA_LAST();
	cudaMalloc( (void**) &trailsDevice, sizeof(int) * (size_t)rows * (size_t)columns );CHECK_CUDA_LAST();
	cudaMalloc( (void**) &heightsDevice, sizeof(int) * (size_t)rows * (size_t)columns );CHECK_CUDA_LAST();
	cudaMalloc( (void**) &taintedDevice, sizeof(int) * (size_t)rows * (size_t)columns );CHECK_CUDA_LAST();

	if ( searchers == NULL || total_steps == NULL ) {
		fprintf(stderr,"-- Error allocating searchers structures for size: %d\n", num_searchers );
		exit( EXIT_FAILURE );
	}


	if ( heights == NULL || trails == NULL || tainted == NULL ) {
		fprintf(stderr,"-- Error allocating terrain structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}


	/* 3.2. Terrain initialization */
	for( i=0; i<rows; i++ ) {
		for( j=0; j<columns; j++ ) {
			accessMat( heights, i, j ) = INT_MIN;
			accessMat( trails, i, j ) = -1;
			accessMat( tainted, i, j ) = 0;
		}
	}


	for( search = 0; search < num_searchers; search++ ) {
		searchers[ search ].pos_row = (int)( rows * erand48( random_seq ) );
		searchers[ search ].pos_col = (int)( columns * erand48( random_seq ) );
	}
	cudaMemcpy (taintedDevice, tainted, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice);
	cudaMemcpy (trailsDevice, trails, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice);
	cudaMemcpy (heightsDevice, heights, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice);
	cudaMemcpy (searchersDevice, searchers, sizeof(Searcher) * num_searchers, cudaMemcpyHostToDevice);
	cudaMemcpy (total_stepsDevice, total_steps, sizeof(int) * num_searchers, cudaMemcpyHostToDevice);
	/* 3.3. Searchers initialization */
	kernel_InitSearchers<<<NBLOQUES, NHILOS>>>(searchersDevice, total_stepsDevice, num_searchers);CHECK_CUDA_LAST();


	/* 4. Compute searchers climbing trails */
	kernel_climbStep<<<NBLOQUES, NHILOS>>>(searchersDevice, taintedDevice, heightsDevice,trailsDevice, rows, columns, x_min, x_max, y_min, y_max, num_searchers);CHECK_CUDA_LAST();


	kernel_condicionCarrera<<<NBLOQUES, NHILOS>>>(searchersDevice, trailsDevice, num_searchers, columns);CHECK_CUDA_LAST();


	/* 5. Compute the leading follower of each searcher */
	kernel_ComputeFollowers<<<NBLOQUES, NHILOS>>>(searchersDevice, num_searchers);CHECK_CUDA_LAST();

	/* 6. Compute accumulated trail steps to each maximum */

	kernel_Trail<<<NBLOQUES, NHILOS>>>(searchersDevice, total_stepsDevice, num_searchers);CHECK_CUDA_LAST();
	cudaMemcpy (total_steps, total_stepsDevice, sizeof(int) * num_searchers, cudaMemcpyDeviceToHost);
	cudaMemcpy (searchers, searchersDevice, sizeof(Searcher) * num_searchers, cudaMemcpyDeviceToHost);
	//cudaMemcpy (tainted, taintedDevice, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyDeviceToHost);
	cudaMemcpy (heights, heightsDevice, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyDeviceToHost);
	cudaMemcpy (trails, trailsDevice, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyDeviceToHost);

	/* 7. Compute statistical data */
	int num_local_max = 0;
	int max_height = INT_MIN;
	int max_accum_steps = INT_MIN;
	int *total_tainted;
	unsigned long long int total_heights = 0;

	for( search = 0; search < num_searchers; search++ ) {
		/* Maximum of accumulated trail steps to a local maximum */
		if ( max_accum_steps < total_steps[ search ] )
			max_accum_steps = total_steps[ search ];

		/* If this searcher found a maximum, check the maximum value */
		if ( searchers[ search ].follows == search ) {
			num_local_max++;
			int pos_row = searchers[ search ].pos_row;
			int pos_col = searchers[ search ].pos_col;
			if ( max_height < accessMat( heights, pos_row, pos_col ) )
				max_height = accessMat( heights, pos_row, pos_col );
		}
	}


	int *sumaTainted;
	unsigned long long int *sumaHeights;
	
	//Memoria compartida
	//int compartida = NHILOS* sizeof(int);
	int compartida_reduction = NHILOS_REDUCTION *sizeof(int);
	
	cudaMalloc( &sumaTainted, sizeof(int));CHECK_CUDA_LAST();
	cudaMemsetAsync(sumaTainted, 0, sizeof(int));CHECK_CUDA_LAST();
	
	kernel_SumaTainted<<<NBLOQUES_REDUCTION, NHILOS_REDUCTION, compartida_reduction>>>(taintedDevice, rows*columns, sumaTainted);CHECK_CUDA_LAST();
	cudaMemcpy(&total_tainted, sumaTainted, sizeof(int), cudaMemcpyDeviceToHost);CHECK_CUDA_LAST();

	cudaMalloc(&sumaHeights, sizeof(unsigned long long int));CHECK_CUDA_LAST();
	cudaMemsetAsync(sumaHeights, 0, sizeof(unsigned long long int));CHECK_CUDA_LAST();
	
	kernel_SumaHeights<<<NBLOQUES_REDUCTION, NHILOS_REDUCTION, compartida_reduction>>>(heightsDevice, rows*columns, sumaHeights);CHECK_CUDA_LAST();
	cudaMemcpy(&total_heights, sumaHeights, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);CHECK_CUDA_LAST();

	cudaFree(searchersDevice);
	cudaFree(total_stepsDevice);
	cudaFree(trailsDevice);
	cudaFree(heightsDevice);
	cudaFree(taintedDevice);





/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 6.2. Results: Statistics */
	printf("Result: %d, %d, %d, %d, %llu\n\n",
			num_local_max,
			max_height,
			max_accum_steps,
			total_tainted,
			total_heights );

	/* 7. Free resources */
	free( searchers );
	free( total_steps );
	free( heights );
	free( trails );
	free( tainted );

	/* 8. End */
	return 0;
}
