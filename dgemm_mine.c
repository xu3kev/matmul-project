#include<immintrin.h>
const char* dgemm_desc = "My awesome dgemm two-blocking with zero-padding.";

//BLOCK_SIZE need to have factor 8 !!!!
#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 8)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            double s = B[j*lda+k];
            for (i = 0; i < M; ++i) {
                double cij = C[j*lda+i];
                cij += A[k*lda+i] * s;
                C[j*lda+i] = cij;
            }
        }
    }
}

// pad to from M*N to K*K
void pad(const double * restrict s, double * restrict d, const int M, const int N, const int lda)
{
    int i,j;
    int K = BLOCK_SIZE;
    for(j = 0; j < N; ++j){
        for(i = 0; i < M; ++i){
            *(d + j*K + i) = *(s + j*lda + i);
        }
    }
}

// unpad back to the original result M*N
void depad(const double * restrict s, double * restrict d, const int M, const int N, const int lda)
{
    int i,j;
    int K = BLOCK_SIZE;
    for(j = 0; j < N; ++j){
        for(i = 0; i < M; ++i){
            *(d + j*lda + i)=*(s + j*K + i);
        }
    }
}

//register blocking
void dgemm_small(const double * restrict A, const double * restrict B, double * restrict C){
    const int M = SMALL_BLOCK_SIZE;
    const int lda = BLOCK_SIZE;
    int i,j,k;
    //for (j = 0; j < M; j++) {
    //    for (k = 0; k < M; k+=2) {
    //        double s1 = B[j*lda+k];
    //        double s2 = B[j*lda+k+1];
    //        for (i = 0; i < M; ++i) {
    //            C[j*lda+i] += A[k*lda+i] * s1;
    //            C[j*lda+i] += A[k*lda+lda+i] * s2;
    //        }
    //    }
    //}
    __builtin_assume_aligned(A, 32); 
    __builtin_assume_aligned(B, 32); 
    __builtin_assume_aligned(C, 32); 
    double cc[SMALL_BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    //_mm256d cc;
    for (j = 0; j < SMALL_BLOCK_SIZE; ++j) {
        for(i = 0;i < SMALL_BLOCK_SIZE; ++i)
            cc[i] = C[j*lda+i];
        for (k = 0; k < SMALL_BLOCK_SIZE; ++k) {
            double s = B[j*lda+k];
            for (i = 0; i < SMALL_BLOCK_SIZE; ++i) {
                cc[i] += A[k*lda+i] * s;
            }
        }
        for(i=0;i<SMALL_BLOCK_SIZE;++i)
            C[j*lda+i]=cc[i];
    }  
}

void basic_dgemm_square(const double * restrict A, const double * restrict B, double * restrict C)
{
    const int M = BLOCK_SIZE;
    const int N = SMALL_BLOCK_SIZE;
    const int n_blocks = BLOCK_SIZE/SMALL_BLOCK_SIZE;
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * SMALL_BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * SMALL_BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * SMALL_BLOCK_SIZE;
                //basic_dgemm(M, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE, A + i + k*M, B + k + j*M, C + i + j*M);
                dgemm_small(A+i+k*M, B+k+j*M, C+i+j*M);
                //kdgemm(A+i+k*M, B+k+j*M, C+i+j*M);
            }
        }
    }
}

void do_copy_square_in(const int M, const int lda, const double * restrict A,  double * restrict AA){
    int i, j;
    for(j = 0; j < M; ++j){
        for(i = 0; i < M; ++i){
            AA[j*M+i] = A[j*lda+i];
        }
    }
}

void do_copy_square_out(const int M, const int lda, double * restrict A, const double *restrict  AA){
    int i, j;
    for(j = 0; j < M; ++j){
        for(i = 0; i < M; ++i){
             A[j*lda+i] = AA[j*M+i];
        }
    }
}

void do_block_square(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              double * restrict AA, double * restrict BB, double * restrict CC,
              const int i, const int j, const int k)
{

    const int M = BLOCK_SIZE;
    do_copy_square_in(M, lda, A+i+k*lda, AA);
	//do_copy_square_in(BLOCK_SIZE, lda, B+k+j*lda, BB);
    do_copy_square_in(M, lda, C+i+j*lda, CC);

    basic_dgemm_square(AA, BB, CC);

    do_copy_square_out(M, lda, C+i+j*lda, CC);
}

void do_block_edge(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              double * restrict AA, double * restrict BB, double * restrict CC,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    if (M < BLOCK_SIZE/2 | N < BLOCK_SIZE/2 | K < BLOCK_SIZE/2){
        basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
    }
    else{
        int a;
    for (a = 0; a < BLOCK_SIZE*BLOCK_SIZE; ++a ){
       AA[a] = 0;
       BB[a] = 0;
       CC[a] = 0;
    }
    pad(A + i + k*lda, AA, M, K, lda);
    pad(B + k + j*lda, BB, K, N, lda);
    pad(C + i + j*lda, CC, M, N, lda);

    basic_dgemm_square(AA, BB, CC);

    depad(CC, C + i + j*lda, M, N, lda);
    }
    
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE;
    int bi, bj, bk;
    
    const int leftover = M % BLOCK_SIZE ? 1:0;
    double  AA[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    double  BB[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    double  CC[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
	
	if (leftover){
		for (bi = 0; bi < n_blocks; ++bi) {
			const int i = bi * BLOCK_SIZE;
			for (bj = 0; bj < n_blocks; ++bj) {
				const int j = bj * BLOCK_SIZE;
				for (bk = 0; bk < n_blocks; ++bk) {
					const int k = bk * BLOCK_SIZE;
					do_copy_square_in(BLOCK_SIZE, M, B+k+j*M, BB);
					do_block_square(M, A, B, C, AA, BB, CC, i, j, k);
					//do_block(M, A, B, C, i, j, k);
				}
				const int k = n_blocks * BLOCK_SIZE;
				do_block_edge(M, A, B, C, AA, BB, CC, i, j, k);
			}
			const int j = n_blocks * BLOCK_SIZE;
			for (bk = 0; bk < n_blocks + leftover; ++bk) {
				const int k = bk * BLOCK_SIZE;
				do_block_edge(M, A, B, C, AA, BB, CC, i, j, k);
			}
		}
		const int i = n_blocks * BLOCK_SIZE;
		for (bj = 0; bj < n_blocks + leftover; ++bj) {
			const int j = bj * BLOCK_SIZE;
			for (bk = 0; bk < n_blocks + leftover; ++bk) {
				const int k = bk * BLOCK_SIZE;
				do_block_edge(M, A, B, C, AA, BB, CC, i, j, k);
			}
		}
	}
	else{
		for (bj = 0; bj < n_blocks; ++bj) {
			const int j = bj * BLOCK_SIZE;
			for (bk = 0; bk < n_blocks; ++bk) {
				const int k = bk * BLOCK_SIZE;
				do_copy_square_in(BLOCK_SIZE, M, B+k+j*M, BB);
				for (bi = 0; bi < n_blocks; ++bi) {
					const int i = bi * BLOCK_SIZE;
					do_block_square(M, A, B, C, AA, BB, CC, i, j, k);
					//do_block(M, A, B, C, i, j, k);
				}
			}
		}
	}
}

