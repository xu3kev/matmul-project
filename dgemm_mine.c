#include<immintrin.h>
const char* dgemm_desc = "My awesome dgemm two-blocking with zero-padding.";

//BLOCK_SIZE need to have factor 8 !!!!
#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 16)
#endif

#ifndef SMALL_BLOCK_SIZE_X
#define SMALL_BLOCK_SIZE_X ((int) 8)
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
void dgemm_smaller(const double * restrict A, const double * restrict B, double * restrict C){
    const int M = 8;
    const int lda = BLOCK_SIZE;
    int i,j,k;
    double cc[8] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));

    __m256d cc1,cc2;
    __m256d s;
    __m256d s_per;
    __m256d ss;
    __m256d aa1;
    __m256d aa2;
    #define regA 2
    #define regB 2
    __m256d c[regA][regB];
    int ci,cj;
    for(ci=0;ci<8;ci+=regA*4){
        for(cj=0;cj<8;cj+=regB){

            for(j=0;j<regB;++j){
                for(i=0;i<regA;++i){
                    c[i][j] = _mm256_load_pd(C+(ci+i*4)+(cj+j)*lda);
                }
            }
            
            for(k=0;k<8;++k){
                for(j=0;j<regB;++j){
                    //double bb = B[k+(cj+j)*lda];
                    __m256d bb = _mm256_broadcast_sd(B+k+(cj+j)*lda);
                    for(i=0;i<regA;++i){
                        __m256d aa = _mm256_load_pd(A+(ci+i*4)+k*lda);
                        //c[i][j]+=aa*bb;
                        c[i][j] = _mm256_fmadd_pd(aa,bb,c[i][j]);
                    }
                }
            }

            for(j=0;j<regB;++j){
                for(i=0;i<regA;++i){
                    //C[(ci+i)+(cj+j)*lda] = c[i][j];
                    _mm256_store_pd(C+(ci+i*4)+(cj+j)*lda, c[i][j]);

                }
            }
             
        }
    }
}


//register blocking
void dgemm_smaller2(const double * restrict A, const double * restrict B, double * restrict C){
    const int M = 8;
    const int lda = BLOCK_SIZE;
    int i,j,k;
    double cc[8] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    //for(j=0;j<8;++j){
    //    for( i = 0;i<8;++i){
    //        C[j*lda+i] = B[j*lda+i]+A[j*lda+i];
    //    }
    //}
    //for(j=0;j<8;++j){
    //    for( i = 0;i<8;++i){
    //        C[j*lda+i] = B[j*lda+i]+A[j*lda+i];
    //    }
    //}

    __m256d cc1,cc2;
    __m256d s;
    __m256d s_per;
    __m256d ss;
    __m256d aa1;
    __m256d aa2;
    for (j = 0; j < 8; ++j) {
        //for(i = 0;i < 8; ++i)
        //    cc[i] = C[j*lda+i];
        cc1 = _mm256_load_pd(C+j*lda);
        cc2 = _mm256_load_pd(C+j*lda+4);

        //double s = B[j*lda+k];
        
        //s = _mm256_castsi256_pd(_mm256_permute2x128_si256(_mm256_castpd_si256(s),_mm256_castpd_si256(s),0));
        //for (k = 0; k < 8; ++k) {


#define MUL(kk,SHUF) ss = _mm256_shuffle_pd(s_per,s_per,SHUF);\
        aa1 = _mm256_load_pd(A+kk*lda+i);\
        aa2 = _mm256_load_pd(A+kk*lda+i+4);\
        cc1 = _mm256_fmadd_pd(aa1,ss,cc1);\
        cc2 = _mm256_fmadd_pd(aa2,ss,cc2);

        //s = _mm256_load_pd(B+j*lda);
        //s_per = _mm256_castsi256_pd(_mm256_permute2x128_si256(_mm256_castpd_si256(s),_mm256_castpd_si256(s),0));
        //MUL(0,0);
        //MUL(1,15);

        //s_per = _mm256_castsi256_pd(_mm256_permute2x128_si256(_mm256_castpd_si256(s),_mm256_castpd_si256(s),17));
        //MUL(2,0);
        //MUL(3,15);

        //s = _mm256_load_pd(B+j*lda+4);
        //s_per = _mm256_castsi256_pd(_mm256_permute2x128_si256(_mm256_castpd_si256(s),_mm256_castpd_si256(s),0));
        //MUL(4,0);
        //MUL(5,15);

        //s_per = _mm256_castsi256_pd(_mm256_permute2x128_si256(_mm256_castpd_si256(s),_mm256_castpd_si256(s),17));
        //MUL(6,0);
        //MUL(7,15);

#define MUL0(kk) ss = _mm256_broadcast_sd(B+j*lda+kk);\
            aa1 = _mm256_load_pd(A+kk*lda+i);\
            aa2 = _mm256_load_pd(A+kk*lda+i+4);\
            cc1 = _mm256_fmadd_pd(aa1,ss,cc1);\
            cc2 = _mm256_fmadd_pd(aa2,ss,cc2);

        MUL0(0);
        MUL0(1);
        MUL0(2);
        MUL0(3);
        MUL0(4);
        MUL0(5);
        MUL0(6);
        MUL0(7);

        //}
        //for(i=0;i<8;++i)
        //    C[j*lda+i]=cc[i];
        _mm256_store_pd(C+j*lda, cc1);
        _mm256_store_pd(C+j*lda+4, cc2);
    }

    //for (j = 0; j < 8; ++j) {
    //    for(i = 0;i < 8; ++i)
    //        cc[i] = C[j*lda+i];
    //    for (k = 0; k < 8; ++k) {
    //        double s = B[j*lda+k];
    //        for (i = 0; i < 8; ++i) {
    //            cc[i] += A[k*lda+i] * s;
    //        }
    //    }
    //    for(i=0;i<8;++i)
    //        C[j*lda+i]=cc[i];
    //}
}
void dgemm_small(const double * restrict A, const double * restrict B, double * restrict C){


    //const int M = 8;
    //const int lda = BLOCK_SIZE;
    //int i,j,k;
    //double cc[8] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    //for (j = 0; j < 8; ++j) {
    //    for(i = 0;i < 8; ++i)
    //        cc[i] = C[j*lda+i];
    //    for (k = 0; k < 8; ++k) {
    //        double s = B[j*lda+k];
    //        for (i = 0; i < 8; ++i) {
    //            cc[i] += A[k*lda+i] * s;
    //        }
    //    }
    //    for(i=0;i<8;++i)
    //        C[j*lda+i]=cc[i];
    //}  


    const int M = SMALL_BLOCK_SIZE;
    const int lda = BLOCK_SIZE;
    int i,j,k;
    i=0;k=8;j=0;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=0;k=8;j=8;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=0;k=0;j=8;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=0;k=0;j=0;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=8;k=0;j=0;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=8;k=0;j=8;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=8;k=8;j=8;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
    i=8;k=8;j=0;
    dgemm_smaller(A+i+lda*k,B+k+lda*j,C+i+lda*j);
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
                //dgemm_smaller(A+i+k*M, B+k+j*M, C+i+j*M);
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
void do_copy_square_in_transpose(const int M, const int lda, const double * restrict A,  double * restrict AA){
    int i, j;
    for(j = 0; j < M; ++j){
        for(i = 0; i < M; ++i){
            AA[j*M+i] = A[i*lda+j];
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
	do_copy_square_in(M, lda, B+k+j*lda, BB);
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
					//do_copy_square_in(BLOCK_SIZE, M, B+k+j*M, BB);
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
				//do_copy_square_in(BLOCK_SIZE, M, B+k+j*M, BB);
				for (bi = 0; bi < n_blocks; ++bi) {
					const int i = bi * BLOCK_SIZE;
					do_block_square(M, A, B, C, AA, BB, CC, i, j, k);
					//do_block(M, A, B, C, i, j, k);
				}
			}
		}
	}
}

