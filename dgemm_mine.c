const char* dgemm_desc = "Simple blocked dgemm.";

//BLOCK_SIZE need to have factor 8 !!!!
#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 4)
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
//void basic_dgemm(const int lda, const int M, const int N, const int K,
//                 const double *A, const double *B, double *C)
//{
//    int i, j, k;
//    for (i = 0; i < M; ++i) {
//        for (j = 0; j < N; ++j) {
//            double cij = C[j*lda+i];
//            for (k = 0; k < K; ++k) {
//                cij += A[k*lda+i] * B[j*lda+k];
//            }
//            C[j*lda+i] = cij;
//        }
//    }
//}
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

//register blocking

void dgemm_small(const double * restrict A, const double * restrict B, double * restrict C){
    const int M = SMALL_BLOCK_SIZE;
    int i,j,k;
    for (j = 0; j < M*M; j+=M) {
        for (k = 0; k < M; ++k) {
            double s = B[j+k];
            for (i = 0; i < M; ++i) {
                double cij = C[j+i];
                cij += A[k*M+i] * s;
                C[j+i] = cij;
            }
        }
    }
}


void basic_dgemm_square(const double * restrict A, const double * restrict B, double * restrict C)
{
    //int i, j, k;
    //for (j = 0; j < M; ++j) {
    //    for (k = 0; k < M; ++k) {
    //        double s = B[j*M+k];
    //        for (i = 0; i < M; ++i) {
    //            double cij = C[j*M+i];
    //            cij += A[k*M+i] * s;
    //            C[j+i] = cij;
    //        }
    //    }
    //}
    const int M = BLOCK_SIZE;
    const int n_blocks = BLOCK_SIZE/SMALL_BLOCK_SIZE;
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * SMALL_BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * SMALL_BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * SMALL_BLOCK_SIZE;
                basic_dgemm(M, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE, A + i + k*M, B + k + j*M, C + i + j*M);
            }
        }
    }
}

void do_copy_square_in(const int lda, const double * restrict A,  double * restrict AA){
    int i, j, k;
    const int M = BLOCK_SIZE;
    for(i = 0; i < M; ++i){
        for(j = 0; j < M; ++j){
            AA[i*M+j] = A[i*lda+j];
        }
    }
}

void do_copy_square_out(const int lda, double * restrict A, const double *restrict  AA){
    int i, j, k;
    const int M = BLOCK_SIZE;
    for(i = 0; i < M; ++i){
        for(j = 0; j < M; ++j){
             A[i*lda+j] = AA[i*M+j];
        }
    }
}

//void basic_dgemm_square(const int lda,
//                 const double * restrict A, const double * restrict B, double * restrict C)
//{
//    const int M = BLOCK_SIZE;
//    int i, j, k;
//    for (i = 0; i < M; ++i) {
//        for (j = 0; j < M; ++j) {
//            double cij = C[j*lda+i];
//            for (k = 0; k < M; ++k) {
//                cij += A[k*lda+i] * B[j*lda+k];
//            }
//            C[j*lda+i] = cij;
//        }
//    }
//}


void do_block_square(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              double * restrict AA, double * restrict BB, double * restrict CC,
              const int i, const int j, const int k)
{

    const int M = BLOCK_SIZE;
    do_copy_square_in(lda, A+i+k*lda, AA);
    do_copy_square_in(lda, B+k+j*lda, BB);
    do_copy_square_in(lda, C+i+j*lda, CC);
    //memset(CC, 0, sizeof(double)*M*M);

    //printf("Aij, %f\n", A[i+k*lda]);
    //printf("AA , %f\n", AA[0]);
    //printf("Bij, %f\n", B[k+j*lda]);
    //printf("BB , %f\n", BB[0]);

    basic_dgemm_square(AA, BB, CC);

    do_copy_square_out(lda, C+i+j*lda, CC);
    //printf("Cij, %f\n", C[i+j*lda]);
    //printf("CC , %f\n", CC[0]);
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    //printf("A C, %f\n", A[i+k*lda]);
    //printf("B C, %f\n", B[k+j*lda]);

    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
    //printf("C C, %f\n", C[i+j*lda]);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE;
    int bi, bj, bk;
    
    const int leftover = M % BLOCK_SIZE ? 1:0;
    double  AA[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    double  BB[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    double  CC[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));

    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block_square(M, A, B, C, AA, BB, CC, i, j, k);
                //do_block(M, A, B, C, i, j, k);
            }
            if (leftover){
                const int k = n_blocks * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
        if (leftover){
            const int j = n_blocks * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks + leftover; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
    if (leftover){
        const int i = n_blocks * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks + leftover; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks + leftover; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}

