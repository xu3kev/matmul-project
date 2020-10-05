const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void basic_dgemm_square(const int lda,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    const int M = BLOCK_SIZE;
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < M; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void do_block_square(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j, const int k)
{

    const int M = BLOCK_SIZE;
    const double *a = A + i + k*lda;
    const double *b = B + k + j*lda;
    const double *c = C + i + j*lda;
    basic_dgemm_square(lda, A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE;
    int bi, bj, bk;
    
    const int leftover = M % BLOCK_SIZE ? 1:0;

    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block_square(M, A, B, C, i, j, k);
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

