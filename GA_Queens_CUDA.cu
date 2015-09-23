
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define Simple	0
#define Multi	1

#define Critical 1e-2
#define M_Critical 1e-3
#define debug
#define MAX_QUEENS 500		

int n ;
int goal ;					
clock_t start, end ;		
time_t	start_time, end_time ;

// queen stores the DNA of a individual 
// unitFitness is the fitness of the individual
// eachFitness is every gene’s fitness
typedef struct {
	int queen[MAX_QUEENS] ;
	int unitFitness	;
	int eachFitness[MAX_QUEENS] ;
} Population ; 

Population s_population, m_population[300] ;
int m_size ;
int m_totFitness ;    

void init ()
{
	srand (0) ;
	scanf ("%d", &n) ;
	goal = n * (n - 1) ;
	m_size = 300 ;
}

int compare(const void *a,const void *b)
{
  return ((Population *)b)->unitFitness - ((Population *)a)->unitFitness;
}  

int Aggressive(Population *p, int i, int j)
{
	return (abs(p->queen[i] - p->queen[j]) == abs(i - j)) ? 0 : 1 ;
}

void UpdateFitnessScore (Population *p) 
{
	int i, j;

	p->unitFitness = 0 ;
	for (i = 0 ; i < n ; i++)
	{
		p->eachFitness[i] = 0 ;
		for (j = 0 ; j < n ; j++)
			p->eachFitness[i] += Aggressive(p, i, j) ;

		p->unitFitness += p->eachFitness[i] ;     
	}

}


void CreateMultiStartPopulation ()
{	
	int loop, i, j ;
	int tmp[MAX_QUEENS] ;
	
	for (loop = 0 ; loop < m_size ; loop ++)	
	{
		for (i = 0 ; i < n ; i++)
				tmp[i] = i ;

			for (i = 0 ; i < n ; i++)
			{
				j = rand() % (n - i) ;
				m_population[loop].queen[i] = tmp[j] ;
				tmp[j] = tmp[n - i - 1] ;
			}
			UpdateFitnessScore(&m_population[loop]) ;
	}
}


void MultiMutate (Population* p)
{
	int i, j, swap ;
	int worst ;
	Population baby ;

	worst = 0 ;
	for (i = 0 ; i < n ; i++)
		if (p->eachFitness[i] < p->eachFitness[worst])
			worst = i ;

	baby = *p ;
	for (i = 0 ; i < n / 4 ; i++)
	{
		j = rand() % n ;	
			
		swap = baby.queen[worst] ;
		baby.queen[worst] = baby.queen[j] ;
		baby.queen[j] = swap ;

		UpdateFitnessScore(&baby) ;
		if (baby.unitFitness > p->unitFitness || (double)rand() / RAND_MAX < M_Critical)
		{
			*p = baby ;
			break ;
		}
	}
}

int RouletteWheelSelection()
{
	int selection = 0;
	int i ;

	double slice = (double)rand() / RAND_MAX;
	double addFitness = 0;
	for(i = 0; i < m_size ; i++)
	{
		addFitness +=  (double)m_population[i].unitFitness / m_totFitness ;
		if(addFitness > slice)
		{
			selection = i;
			break;
		}

	}
	return selection;
}


//calculate the crossover function in GPU with 256 threads and 1 block
__global__ static void cudaCross(int *pQueen, 
				 //int *pUnitFitness, 
				 //int *pEachFitness,
				 int *cQueen, 
				 int *cUnitFitness, 
				 int *cEachFitness,
				 int *mateSeq,
				 int *positionSeq,
				 int n,
				 int m_size,
				 int THREAD_NUM){
	int tid = threadIdx.x;
	int count;
	for (count = tid ; count < m_size - 2 ; count=count+THREAD_NUM)
	{
		int flag[500] ;
		int pos1, pos2, tmp, father, mother ;
		int i, j ;
		
	
		pos1 = positionSeq[count];
		pos2 = positionSeq[count+1];
		if (pos1 > pos2) { tmp = pos1 ; pos1 = pos2 ; pos2 = tmp; }
		
		father = mateSeq[(count)*2];
		mother = mateSeq[(count)*2+1];
		for (j = 0 ; j < n ; j ++)
			flag[j] = 0 ;
		for (j = pos1; j <= pos2; j++)
			flag[pQueen[father*n+j]] = 1 ;

		for(i = 0, j = 0 ; i < n ; i++)
		{
			if (i < pos1 || i > pos2) {
				while (flag[pQueen[mother*n+j]]) j++ ;
				cQueen[count*n + i] = pQueen[mother*n+j] ;
				j ++ ;
			}
			else cQueen[count*n + i] = pQueen[father*n+j] ;
		}

		cUnitFitness[count] = 0 ;
		for (i = 0 ; i < n ; i++)
		{
			cEachFitness[count*n + i] = 0 ;
			for (j = 0 ; j < n ; j++)
				cEachFitness[count*n + i] += (abs(cQueen[count*n + i] - cQueen[count*n + j]) == abs(i - j)) ? 0 : 1 ;

			cUnitFitness[count] += cEachFitness[count*n + i] ;     
		}
		//UpdateFitnessScore (baby) ;
		//CrossOverFM (m_population[father], m_population[mother], &p[count]) ; 
	}
}
//the crossover function is calculated on the GPU
void CrossOver () 
{
	int i,j,pos1,pos2;
	int h_parentSeq[m_size*2];
	int h_positionSeq[m_size*2];
	int h_pQueen[m_size*n];
	int h_cQueen[m_size*n];
	int h_cUnitFitness[m_size];
	int h_cEachFitness[m_size*n];

	int d_parentSeq[m_size*2];
	int d_positionSeq[m_size*2];
	int d_pQueen[m_size*n];
	int d_cQueen[m_size*n];
	int d_cUnitFitness[m_size];
	int d_cEachFitness[m_size*n];

	
	m_totFitness = 0 ;
	for (i = 0 ; i < m_size ; i++)
		m_totFitness += m_population[i].unitFitness ;

	for (i = 0 ; i < m_size*2 ; i=i+2){
		h_parentSeq[i] = RouletteWheelSelection () ;
		h_parentSeq[i+1] = RouletteWheelSelection () ;
		do {
		pos1 = rand() % n ;
		pos2 = rand() % n ;
		} while (pos1 == pos2) ;
		h_positionSeq[i] = pos1;
		h_positionSeq[i+1] = pos2;

	}

	for (i = 0 ; i < m_size ; i++)
		for (j = 0 ; j < n ; j++)
			h_pQueen[i*n+j] = m_population[i].queen[j];

	
	// this place should be the cuda crossover
	/***************************************************
	  1st Part: Allocation of memory on device memory  
	    ****************************************************/	    
	    
	    cudaMalloc((void**) &d_parentSeq, sizeof(int) * m_size * 2);
	    cudaMalloc((void**) &d_positionSeq, sizeof(int) * m_size * 2);
	    cudaMalloc((void**) &d_pQueen, sizeof(int) * m_size * n);
	    cudaMalloc((void**) &d_cQueen, sizeof(int) * m_size * n);
	    cudaMalloc((void**) &d_cUnitFitness, sizeof(int) * m_size);
	    cudaMalloc((void**) &d_cEachFitness, sizeof(int) * m_size * n);
	    //How to write Memcpy
	    cudaMemcpy(d_parentSeq, h_parentSeq, sizeof(int) * m_size * 2, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_pQueen, h_pQueen, sizeof(int) * m_size * n, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_positionSeq, h_positionSeq, sizeof(int) * m_size * 2, cudaMemcpyHostToDevice);
	    //cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);   
	 
	    
	    /***************************************************
	   2nd Part: Inovke kernel 
	    ****************************************************/
	    int THREAD_NUM = 256;
	    cudaCross<<<1, THREAD_NUM, 0>>>(d_pQueen, 
					 //int *pUnitFitness, 
					 //int *pEachFitness,
					 d_cQueen, 
					 d_cUnitFitness, 
					 d_cEachFitness,
					 d_parentSeq,
					 d_positionSeq,
					 n,
					 m_size,
					 THREAD_NUM);
	    
	    /***************************************************
	   3rd Part: Transfer result from device to host 
	    ****************************************************/
	    cudaMemcpy(h_cQueen, d_cQueen, sizeof(int) * m_size * n, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_cUnitFitness, d_cUnitFitness, sizeof(int) * m_size, cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_cEachFitness, d_cEachFitness, sizeof(int) * m_size * n, cudaMemcpyDeviceToHost);

	    cudaFree(d_parentSeq);
	    cudaFree(d_pQueen);
	    cudaFree(d_cQueen);
	    cudaFree(d_cUnitFitness);
	    cudaFree(d_cEachFitness);
	for (int count = 0 ; count < m_size - 2 ; count++){
		m_population[count+2].unitFitness = h_cUnitFitness[count] ;
		for (j = 0; j < n ; j++){
			m_population[count+2].queen[j] = h_cQueen[count*n+j];
			m_population[count+2].eachFitness[j] = h_cEachFitness[count*n+j];
		}
	}
}

void PrintQueens (Population p)
{
	double secs ;

	secs = (double)(end - start) / (CLOCKS_PER_SEC*20*8) ;
    printf("Calculations took %.3lf second%s.\n", secs, (secs < 1 ? "" : "s"));
	

}

int main ()
{
	
	/*
	freopen ("input.txt", "r", stdin) ;
	freopen ("output.txt", "w", stdout) ;
	*/

	while (1) 
	{
		
		init() ;	
		if (n == 0) break ;
		
		
		
		
		/*
		start = clock () ;
		time (&start_time) ;	
		printf("With Single Population : \nStart: \t %s", ctime(&start_time));

		CreateSimpleStartPopulation() ;
	
		do {
			SimpleMutate () ;
		} while (s_population.unitFitness < goal) ;

		end = clock () ;
		time (&end_time) ;
		PrintQueens(s_population) ;
		*/

	
	
		time (&start_time) ;	
		printf("With Multi Population : \nStart: \t %s", ctime(&start_time));

		CreateMultiStartPopulation() ;
		
		int iterationCount=0;

		for(int i=0;i<30;i++){
			start = clock () ;
			while (iterationCount<20){
				qsort(m_population, m_size, sizeof(Population), compare) ;
				MultiMutate (&m_population[0]) ;
				MultiMutate (&m_population[1]) ;
				if (m_population[0].unitFitness == goal || m_population[1].unitFitness ==  goal)
					break ;
				CrossOver () ;
				iterationCount++;
			} 
			end = clock () ;
			PrintQueens(m_population[0].unitFitness == goal ? m_population[0] : m_population[1]) ;
			iterationCount = 0;
		}

		time (&end_time) ;
  }
}