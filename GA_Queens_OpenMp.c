
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

#define Simple	0
#define Multi	1


#define Critical 1e-2
#define M_Critical 1e-3
#define debug
#define MAX_QUEENS 500		// MAX_QUEEN NUMBER IS 500

int n ;
int goal ;					
clock_t start, end ;		
time_t	start_time, end_time ;

//define the struct for one poplation
typedef struct {
	int queen[MAX_QUEENS] ;
	int unitFitness	;
	int eachFitness[MAX_QUEENS] ;
} Population ; 

Population s_population, m_population[10 + MAX_QUEENS / 10] ;
int m_size ;
int m_totFitness ;    

__inline static
void init ()
{
	srand (0) ;
	scanf ("%d", &n) ;
	goal = n * (n - 1) ;
	m_size = 30 + n / 10 ;
}


double wtime(void) 
{
 double          now_time;
 struct timeval  etstart;
 
 if (gettimeofday(&etstart, NULL) == -1)
   perror("Error: calling gettimeofday() not successful.\n");
 
 now_time = ((etstart.tv_sec) * 1000 +     
  etstart.tv_usec / 1000.0);  
 return now_time;
}

// interface for qsort
__inline static
int compare(const void *a,const void *b)
{
  return ((Population *)b)->unitFitness - ((Population *)a)->unitFitness;
}  


__inline static
int Aggressive(Population *p, int i, int j)
{
	return (abs(p->queen[i] - p->queen[j]) == abs(i - j)) ? 0 : 1 ;
}

// update p's fitness
__inline static
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

//randomly initialize the population
__inline static
void CreateSimpleStartPopulation ()
{
	int i, j ;
	int tmp[MAX_QUEENS] ;

	for (i = 0 ; i < n ; i++)
		tmp[i] = i ;

	for (i = 0 ; i < n ; i++)
	{
		j = rand() % (n - i) ;
		s_population.queen[i] = tmp[j] ;
		tmp[j] = tmp[n - i - 1] ;
	}
	UpdateFitnessScore(&s_population) ;

}


__inline static
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


__inline static
void SimpleMutate () 
{
	int i, j, swap ;
	int worst ;
	Population baby ;

	worst = 0 ;
	for (i = 0 ; i < n ; i++)
		if (s_population.eachFitness[i] < s_population.eachFitness[worst])
			worst = i ;
	
	do {
		j = rand() % n ;
	} while (worst == j) ;

	baby = s_population ;
	
	swap = baby.queen[worst] ;
	baby.queen[worst] = baby.queen[j] ;
	baby.queen[j] = swap ;

	UpdateFitnessScore(&baby) ;


	if (baby.unitFitness > s_population.unitFitness 
			|| (double)rand() / RAND_MAX < Critical)
			s_population = baby ;

}


__inline static
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

// RouletteWheel to select parents
__inline static
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

// crossover function
__inline static
void CrossOverFM (Population father, Population mother, Population *baby)
{
	int flag[MAX_QUEENS] ;
	int pos1, pos2, tmp ;
	int i, j ;

	do {
		pos1 = rand() % n ;
		pos2 = rand() % n ;
	} while (pos1 == pos2) ;
	if (pos1 > pos2) { tmp = pos1 ; pos1 = pos2 ; pos2 = tmp; }

	for (j = 0 ; j < n ; j ++)
		flag[j] = 0 ;
	for (j = pos1; j <= pos2; j++)
		flag[father.queen[j]] = 1 ;

	for(i = 0, j = 0 ; i < n ; i++)
	{
		if (i < pos1 || i > pos2) {
			while (flag[mother.queen[j]]) j++ ;
			baby->queen[i] = mother.queen[j] ;
			j ++ ;
		}
		else baby->queen[i] = father.queen[i] ;
	}
	UpdateFitnessScore (baby) ;
}


__inline static
void CrossOver () 
{
	int i ;
	int father, mother ;
	Population p[30 + MAX_QUEENS / 10];
	int count ;

	m_totFitness = 0 ;
	for (i = 0 ; i < m_size ; i++)
		m_totFitness += m_population[i].unitFitness ;
	// use openmp to parallel compute the process of crossover to speed up
	#pragma omp parallel for private(father, mother, count) firstprivate(m_size) shared(m_population, p) schedule(static) 
	for (count = 0 ; count < m_size - 2 ; count++)
	{
		father = RouletteWheelSelection () ;
		mother = RouletteWheelSelection () ;
		CrossOverFM (m_population[father], m_population[mother] , &p[count]) ; 
	}

	for (count = 0 ; count < m_size - 2 ; count++)
		m_population[count+2] = p[count] ;
}


__inline static
void PrintQueens (Population p)
{
	int i, j ;

	printf ("The Board size is %d, and \n ", n);



	
	// will print out the result if chess board is less than 30
	if (n <= 30) 
	{
		printf("One solution looks like : \n") ;
		for (i = 0 ; i < n ; i++)
		{
			for (j = 0 ; j < n ; j++)
				if (j == p.queen[i]) printf ("@ ");
				else printf ("# ") ;
			printf ("\n") ;
		}
	}

	printf ("\n\n\n") ;


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

	
		double start_time = 0.0f, end_time = 0.0f, took_time = 0.0f, start1 = 0.0f, end1 = 0.0f, took1 = 0.0f;
		start1 = wtime();


		CreateMultiStartPopulation() ;
			
		double secs ;
		while (1)
		{
			start_time = wtime();
			qsort(m_population, m_size, sizeof(Population), compare) ;
			MultiMutate (&m_population[0]) ;
			MultiMutate (&m_population[1]) ;
			if (m_population[0].unitFitness == goal || m_population[1].unitFitness ==  goal)
				break ;
			CrossOver () ;

			end_time = wtime();
			took_time = end_time - start_time;
			printf("iteration took time: %f milliseconds\n", took_time);

		} 

		end1 = wtime();
		took1 = end1 - start1;
		printf("total time: %f milliseconds\n", took1);


		PrintQueens(m_population[0].unitFitness == goal ? m_population[0] : m_population[1]) ;

  }
}