/*
 * PRACTICA 2 - MPI. 
 * Toribio Gonzalez, Hector.
 * Torres Viloria, Juan.
 *
 *
 * Probabilistic approach to locate maximum heights
 * Hill Climbing + Montecarlo
 *
 * MPI version
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
#include<omp.h>

/* Headers for the MPI assignment versions */
#include<mpi.h>                              
#include<stddef.h>                           

/* 
 * Global variables and macro to check errors in calls to MPI functions
 * The macro shows the provided message and the MPI string in case of error
 */
char mpi_error_string[ MPI_MAX_ERROR_STRING ];
int mpi_string_len;
#define MPI_CHECK( msg, mpi_call )	{ int check = mpi_call; if ( check != MPI_SUCCESS ) { MPI_Error_string( check, mpi_error_string, &mpi_string_len); fprintf(stderr,"MPI Error - %s - %s\n", msg, mpi_error_string ); MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE ); } }
//Si no ha habido exito en la llamada al sistema de MPI, se guarda el estado de error y se imprime por pantalla 

#define	PRECISION	10000

/* 
 * Structure to represent a climbing searcher 
 * 	This structure can be changed and/or optimized by the students
 */
 
 //OBJETO QUE ALMACENA CADA CAMINO, PUNTO DE INICIO, PUNTO DE FIN, PASOS Y A QUIEN SIGUE
typedef struct {
	int id;				// Searcher identifier
	int pos_row, pos_col;		// Position in the grid
	int steps;			// Steps count
	int follows;			// When it finds an explored trail, who searched that trail
} Searcher;


// Funcion implementada por nosotros, calcula el minimo entre dos valores.

int min (int x, int y){
	if (x>y) {
		return (y);
	}  else  {
		return (x);
	}
}

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
int get_height( int x, int y, int rows, int columns, float x_min, float x_max, float y_min, float y_max  ) {
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
int climbing_step(int begin_local, int size, int rows, int columns, Searcher *searchers, int search, int *heights, int *trails, int *tainted, float x_min, float x_max, float y_min, float y_max, int *almaceno_heights) {
	int search_flag = 0;   
	
	/* Annotate one step more, landing counts as the first step */
	searchers[ search ].steps ++;

	/* Get starting position */
	int pos_row = searchers[ search ].pos_row;
	int pos_col = searchers[ search ].pos_col;
	
	int row_local = pos_row - begin_local;

	/* Stop if searcher finds another trail */
	int check;
	check = accessMat( tainted, row_local, pos_col );
	accessMat( tainted, row_local, pos_col ) = 1;

	if ( check != 0 ) {
		searchers[ search ].follows = accessMat( trails, row_local, pos_col );
		search_flag = 1;
	}
	else {
		/* Annotate the trail */
		accessMat( trails, row_local, pos_col ) = searchers[search].id;

		/* Compute the height, le restamos begin_local porque pos_row es una posicion de nuestra matriz global y la matriz de heights tiene el tamaño de nuestro trozo local. Lo restamos para traducir la posicion global a la posicion local*/
		accessMat( heights, row_local, pos_col ) = get_height( pos_row , pos_col, rows, columns, x_min, x_max, y_min, y_max );

		/* Locate the highest climbing direction */
		float local_max = accessMat( heights, row_local, pos_col );
		int climbing_direction = 0;
		if ( pos_row > 0 ) {
			//Comprobacion de si se sale por abajo
			if(row_local != 0){
				/* Compute the height in the neighbor if needed */
				if ( accessMat( heights, row_local-1, pos_col ) == INT_MIN ) 
					accessMat( heights, row_local-1, pos_col ) = get_height( pos_row - 1 , pos_col, rows, columns, x_min, x_max, y_min, y_max );

				/* Annotate the travelling direction if higher */
				if ( accessMat( heights, row_local-1, pos_col ) > local_max ) {
					climbing_direction = 1;
					local_max = accessMat( heights, row_local-1, pos_col );
				}
			}else{
				//Si se sale por abajo actualizamos el array que almacena las alturas que se salen de la matriz
				if ( accessMat( almaceno_heights, 0, pos_col ) == INT_MIN ) 
					accessMat( almaceno_heights, 0, pos_col ) = get_height( pos_row-1, pos_col, rows, columns, x_min, x_max, y_min, y_max );
				if(accessMat( almaceno_heights, 0, pos_col ) > local_max){
					climbing_direction = 1;
					local_max = accessMat( almaceno_heights, 0, pos_col );
				}
			}
		}
		if ( pos_row < rows-1 ) {
			//Comprobacion de si se sale por arriba
			if(row_local != size-1){
				/* Compute the height in the neighbor if needed */
				if ( accessMat( heights, row_local+1, pos_col ) == INT_MIN )
					accessMat( heights, row_local+1, pos_col ) = get_height( pos_row+1, pos_col, rows, columns, x_min, x_max, y_min, y_max );

				/* Annotate the travelling direction if higher */
				if ( accessMat( heights, row_local+1, pos_col ) > local_max ) {
					climbing_direction = 2;
					local_max = accessMat( heights, row_local+1, pos_col );
				}
			}else{
				//Si se sale por abajo actualizamos el array que almacena las alturas que se salen de la matriz
				if ( accessMat( almaceno_heights, 1, pos_col ) == INT_MIN ) 
					accessMat( almaceno_heights, 1, pos_col ) = get_height( pos_row+1, pos_col, rows, columns, x_min, x_max, y_min, y_max );
				if(accessMat( almaceno_heights, 1, pos_col ) > local_max){
					climbing_direction = 2;
					local_max = accessMat( almaceno_heights, 1, pos_col );
				}
			}
		}
	
		if ( pos_col > 0 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, row_local, pos_col-1 ) == INT_MIN ) 
				accessMat( heights, row_local, pos_col-1 ) = get_height( pos_row, pos_col-1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, row_local, pos_col-1 ) > local_max ) {
				climbing_direction = 3;
				local_max = accessMat( heights, row_local, pos_col-1 );
			}
		}
		if ( pos_col < columns-1 ) {
			/* Compute the height in the neighbor if needed */
			if ( accessMat( heights, row_local, pos_col+1 ) == INT_MIN ) 
				accessMat( heights, row_local, pos_col+1 ) = get_height( pos_row, pos_col+1, rows, columns, x_min, x_max, y_min, y_max );

			/* Annotate the travelling direction if higher */
			if ( accessMat( heights, row_local, pos_col+1 ) > local_max ) {
				climbing_direction = 4;
				local_max = accessMat( heights, row_local, pos_col+1 );
			}
		}

		/* Stop if local maximum is reached */
		if ( climbing_direction == 0 ) {
			searchers[ search ].follows = searchers[ search ].id;
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
		if(pos_row< begin_local){
			search_flag = 2;
		}
		if(pos_row >= begin_local + size){
			search_flag = 3;
		}
		
		
		
		
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

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	// This eliminates the buffer of stdout, forcing the messages to be printed immediately
	setbuf(stdout,NULL);

	int i,j;

	// Simulation data
	int rows, columns;		// Matrix sizes
	float x_min, x_max;		// Limits of the terrain x coordinates
	float y_min, y_max;		// Limits of the terrain y coordinates

	float searchers_density;	// Density of hill climbing searchers
	unsigned short random_seq[3];	// Status of the random sequence

	int *heights = NULL;		// Heights of the terrain points
	int *almaceno_heights = NULL;
	int *trails = NULL;		// Searchers trace and trails
	int *tainted = NULL;		// Position found in a search
	int *follows = NULL;		// Compacted list of searchers "follows"
	int num_searchers;		// Number of searchers
	Searcher *searchers = NULL;	// Searchers data
	int *total_steps = NULL;	// Annotate accumulated steps to local maximums

	/* 0. Initialize MPI */
	MPI_Init( &argc, &argv );	//INICIALIZAMOS MPI, ARGUMENTOS
	int rank;			//GUARDAREMOS AQUÍ EL PROCESO POR EL QUE NOS LLEGAMOS
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc != 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
	if ( rank == 0 ) {
		printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
		printf("Arguments, x_range: ( %d, %d ), y_range( %d, %d )\n", x_min, x_max, y_min, y_max );
		printf("Arguments, searchers_density: %f\n", searchers_density );
		printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", random_seq[0], random_seq[1], random_seq[2]);
		printf("\n");
	}
#endif // DEBUG


	/* 2. Start global timer */
	MPI_CHECK( "Clock: Start-Barrier ", MPI_Barrier( MPI_COMM_WORLD ) );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	/* Statistical data */
	int num_local_max = 0;
	int max_accum_steps = INT_MIN;
	int total_tainted = 0;
	int local_tainted = 0;
	int max_height = INT_MIN;
	int local_max = INT_MIN;
	int num_procs;
	MPI_Comm_size ( MPI_COMM_WORLD , &num_procs ) ;
	


	int my_size_rows = rows/num_procs;

	int my_begin_rows = rank * my_size_rows ;

	
	int resto_rows=rows%num_procs;
	
	
	
	//my_size_rows = (rank<resto_rows)?++my_size_rows:my_size_rows;
	if(resto_rows!=0){
		if(rank<resto_rows){
			my_size_rows += 1;
		}
		my_begin_rows += min(rank, resto_rows);
	}
	

	
	


	//my_begin_rows=my_begin_rows + min(rank,resto_rows);
	
	//my_size_columns = (rank<resto_columns)?++my_size_columns:my_size_columns;
	//my_begin_columns=my_begin_columns + min(rank,resto_columns);

// MPI Version: Eliminate this conditional to start doing the work in parallel
                           

	
	/* 3. Initialization */
	/* 3.1. Memory allocation */
	num_searchers = (int)( rows * columns * searchers_density );
	searchers = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	total_steps = (int *)malloc( sizeof(int) * num_searchers ); 
	follows = (int *)malloc( sizeof(int) * num_searchers );


	Searcher *searchersMios = NULL;				//array para nuestros procesos locales
	Searcher *searchersFinished = NULL;
	Searcher *searchersSendUp = NULL;			// Searches que se pasan por arriba y que serán enviados al proceso anterior
	Searcher *searchersSendDown = NULL;			// Searchers que se pasan por debajo y serán enviados al proceso siguiente.

	searchersMios = (Searcher *)malloc( sizeof(Searcher) * num_searchers );
	searchersFinished = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	searchersSendUp = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	searchersSendDown = (Searcher *)malloc( sizeof(Searcher) * num_searchers ); 
	
	

	
	if ( searchers == NULL || total_steps == NULL || follows == NULL || searchersMios == NULL || searchersSendUp == NULL || searchersFinished == NULL ||searchersSendDown == NULL) {
		fprintf(stderr,"-- Error allocating searchers structures for size: %d\n", num_searchers );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	

	heights = (int *)malloc( sizeof(int) * (size_t)my_size_rows * (size_t)columns );
	almaceno_heights = (int *)malloc( sizeof(int) * (size_t)(2) * (size_t)columns );
	trails = (int *)malloc( sizeof(int) * (size_t)my_size_rows * (size_t)columns );
	tainted = (int *)malloc( sizeof(int) * (size_t)my_size_rows * (size_t)columns );
	if ( heights == NULL || trails == NULL || tainted == NULL ) {
		fprintf(stderr,"-- Error allocating terrain structures for size: %d x %d \n", rows, columns );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	// CREACION TIPO SEARCHER PARA LA COMUNICACION.
	int fields = 1;
	int array_of_blocklengths[] = { 5 };
	MPI_Aint array_of_displacements[] = {
		offsetof(Searcher, id),
		};
	MPI_Datatype array_of_types[] = { MPI_INT };
	MPI_Datatype MPI_Searcher;
	MPI_CHECK( "Creacion tipo de datos para comunicar los Searcher", MPI_Type_create_struct(fields, array_of_blocklengths, array_of_displacements, array_of_types, &MPI_Searcher) );
	MPI_CHECK( "Commit del MPI_Searcher", MPI_Type_commit(&MPI_Searcher) );
	


	
	/* 3.2. Terrain initialization */
	

	for( i=0; i<my_size_rows; i++ ) {
		for( j=0; j<columns; j++ ) {
			accessMat( heights, i, j) = INT_MIN;
			accessMat( trails, i, j  ) = -1;
			accessMat( tainted, i, j  ) = 0;
		}
	}


		for( j=0; j<columns; j++ ) {
			accessMat( almaceno_heights, 0, j) = INT_MIN;
			accessMat( almaceno_heights, 1, j) = INT_MIN;
		}
	
	/* 3.3. Searchers initialization */
	int search;
	int numero_buscadores = 0;
	for( search = 0; search < num_searchers; search++ ) {
		searchers[ search ].id = search;
		searchers[ search ].pos_row = (int)( rows * erand48( random_seq ) );
		searchers[ search ].pos_col = (int)( columns * erand48( random_seq ) );
		searchers[ search ].steps = 0;
		searchers[ search ].follows = -1;
		total_steps[ search ] = 0;
		
		// Nos quedamos con nuestros procesos
		if(( searchers[ search ].pos_row>=my_begin_rows )&&(searchers[ search ].pos_row<my_size_rows + my_begin_rows )){
			searchersMios[numero_buscadores] = searchers[search];
			numero_buscadores++;
		}
	}
	

	/* 4. Compute searchers climbing trails */
	int finished = 0;
	int up = 0;
	int down = 0;
	int terminados_fin = 0;
	int num_recived;
	int search_flag;
	while(!terminados_fin){
	int acaban = 0;
	for( search = 0; search < numero_buscadores; search++ ) {
		search_flag = 0;
		
		while( ! search_flag ) {
			
			search_flag = climbing_step( my_begin_rows, my_size_rows,rows, columns, searchersMios, search, heights, trails, tainted, x_min, x_max, y_min, y_max, almaceno_heights );

		}
		//Comprobamos el flag para saber si esta iteracion se termina, se sale por arriba o por abajo para guardar lo que corresponde en cada array
		if(search_flag == 1){
			searchersFinished[finished] = searchersMios[search];
			finished++;
		}
		
		if(search_flag == 3){
			searchersSendDown[down] = searchersMios[search];
			down++;
		}
		
		if(search_flag == 2){
			searchersSendUp[up] = searchersMios[search];
			up++;
		}
		

	}
	
	
	num_recived = 0;
	numero_buscadores = 0;
	MPI_Request envioUp, envioDown;

	//Envio de searchers que se salen por arriba y por abajo usando el objeto que hemos creado
	if(rank != 0){
		MPI_CHECK("Envio arriba", MPI_Isend(searchersSendUp, up, MPI_Searcher, rank-1, 0, MPI_COMM_WORLD, &envioUp));
	}
	
	if(rank != num_procs-1){
		MPI_CHECK("Envio abajo", MPI_Isend(searchersSendDown, down, MPI_Searcher, rank+1, 0, MPI_COMM_WORLD, &envioDown));
	}

	
	//Recibo de los searcherse que se salen
	if(rank != 0){
		MPI_Status status;
		MPI_CHECK("Recibo arriba", MPI_Recv(searchersMios, num_searchers, MPI_Searcher, rank-1, 0, MPI_COMM_WORLD, &status));
		MPI_Get_count(&status, MPI_Searcher, &num_recived);
		numero_buscadores+=num_recived;
	}
	
	if(rank != num_procs-1){
		MPI_Status status;
		MPI_CHECK("Recibo abajo", MPI_Recv(&searchersMios[numero_buscadores], num_searchers - numero_buscadores, MPI_Searcher, rank+1, 0, MPI_COMM_WORLD, &status)); 
		MPI_Get_count(&status, MPI_Searcher, &num_recived);
		numero_buscadores+=num_recived;
	}
	
	//Esperamos los envios
	if(rank != 0){
		MPI_CHECK("Esperamos envio de arriba", MPI_Wait(&envioUp, MPI_STATUS_IGNORE));
		up = 0;
	}
	
	if(rank != num_procs-1){
		MPI_CHECK("Esperamos envio de abajo", MPI_Wait(&envioDown, MPI_STATUS_IGNORE));
		down = 0;
	}
	
	
	
	
	//Reduccion para comprobar cuando tenemos que salir del while y terminar la busqueda
	MPI_CHECK( "All reduce para comprobar numero de searchers acabados", MPI_Allreduce(&finished, &acaban, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) );

		if(num_searchers == acaban){
			terminados_fin = 1;
		}
	}	
	

	
	
	



#ifdef DEBUG
/* Print computed heights at the end of the search.
 * You can ignore this debug feature.
 * If you want to use this functionality in parallel processes BEWARE: You should 
 * modify it to sincronize the processes to print in order, and each process should
 * prints only its part */
print_heights( rows, columns, heights );
#endif

	int tmp = finished;
	//Envio de numero de finalizados y searchers finalizados
	if(rank!=0){
		MPI_CHECK("Envio de contador de terminados al proceso 0", MPI_Send(&finished, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
		MPI_CHECK("Envio de array de terminados al proceso 0", MPI_Send(searchersFinished, finished, MPI_Searcher, 0, 0, MPI_COMM_WORLD));

	}else{
	
		//Llenamos searchers con los acabados de mi proceso 0
		Searcher *copia_end = (Searcher *)malloc( sizeof(Searcher) * num_searchers );
		for(i = 0; i<finished; i++){
			searchers[i] = searchersFinished[i];
		}
		
		int arrived = finished;
		
		//recibimos del resto de procesos que no son 0
		for(i = 1; i<num_procs; i++){
			MPI_Status statusPadres;
			MPI_CHECK("Recibimos el numero de acabados del proceso i", MPI_Recv(&finished, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &statusPadres));
			MPI_CHECK("Recibimos el numero de acabados del proceso i", MPI_Recv(copia_end, finished, MPI_Searcher, i, 0, MPI_COMM_WORLD, &statusPadres));
		
			for(j = 0; j<finished; j++){
				searchers[j + arrived] = copia_end[j];
			}
		
			arrived+=finished;
			

		}
	



	/* 5. Compute the leading follower of each searcher */
	for( search = 0; search < num_searchers; search++ ) {
		follows[ searchers[ search ].id ] = searchers[ search ].follows;
	}
	for( search = 0; search < num_searchers; search++ ) {
		int search_flag = 0;
		int parent = searchers[ search ].id;
		int follows_to = follows[ parent ];
		while( ! search_flag ) {
			if ( follows_to == parent ){
				search_flag = 1;//Aqui encuentra trail,guardamos en el array de finalizados los que ya han acabado
			}
			else {
				parent = follows_to;
				follows_to = follows[ parent ];
			}
		}
		searchers[ search ].follows = follows_to;
	}

	/* 6. Compute accumulated trail steps to each maximum */
	for( search = 0; search < num_searchers; search++ ) {
		int pos_max = searchers[ search ].follows;
		total_steps[ pos_max ] += searchers[ search ].steps;
	}

	/* 7. Compute statistical data */

	for( search = 0; search < num_searchers; search++ ) {
		/* Maximum of accumulated trail steps to a local maximum */
		if ( max_accum_steps < total_steps[ searchers[search].id]) 
			max_accum_steps = total_steps[ searchers[search].id];

		/* If this searcher found a maximum, check the maximum value */
		if ( searchers[ search ].follows == searchers[search].id ) {
			num_local_max++;

		}
	}
	free(copia_end);
	}	//Fin else
	
	finished = tmp;
	for(search = 0; search<finished; search++){
		if(searchersFinished[search].follows == searchersFinished[search].id){
		int pos_row = searchersFinished[ search ].pos_row - my_begin_rows;
		int pos_col = searchersFinished[ search ].pos_col;
		if ( local_max < accessMat( heights, pos_row, pos_col ) ) 
			local_max = accessMat( heights, pos_row, pos_col );
		}
	}
	
	printf("proceso %d con altura %d\n", rank, local_max);
	
	//Redudccion para obtener el maximo global calculando el maximo de los locales
	MPI_CHECK("Obtenemos maximo global", MPI_Reduce(&local_max, &max_height, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD));
	
	for( i=0; i<my_size_rows; i++ ) {
		for( j=0; j<columns; j++ ) {
			if ( accessMat( tainted, i, j ) == 1) 
				local_tainted++;
		}
	}
	
	//Reduccion para sumar todas las celdas manchadas y obtener las globales
	MPI_CHECK("Obtenemos tainted global", MPI_Reduce(&local_tainted, &total_tainted, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
	
	
	
	
	//Liberar recursos 
	MPI_Type_free(&MPI_Searcher);
	
	//Liberar memoria de los arrays de las comunicaciones.
	free(searchersMios);
	free(searchersFinished);
	free(searchersSendUp);
	free(searchersSendDown);
	free(almaceno_heights);
	

// MPI Version: Eliminate this conditional-end to start doing the work in parallel

	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	MPI_CHECK( "End-Barrier", MPI_Barrier( MPI_COMM_WORLD ) );
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	if ( rank == 0 ) { 
		printf("\n");
		/* 6.1. Total computation time */
		printf("Time: %lf\n", ttotal );

		/* 6.2. Results: Statistics */
		printf("Result: %d, %d, %d, %d\n\n", 
			num_local_max,
			max_height,
			max_accum_steps,
			total_tainted );
	}
			
	/* 7. Free resources */	
	free( searchers );
	
	free( total_steps );
	free( follows );
	free( heights );
	free( trails );
	free( tainted );

	/* 8. End */
	MPI_Finalize();
	return 0;
}
