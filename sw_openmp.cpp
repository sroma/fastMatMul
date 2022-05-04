#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <omp.h>

/*  Strassen-Winograd Algorithm
    n - dimension of square matrices A, B, C.
        Variable n has to be a power of two;
    i, j - start row and col of matrices;
    s - stride of matrices;
    limit - size to call classic MatMul.

    g++ sw_omp.cpp -fopenmp --std=c++11 -o sw.omp.exe
    ./sw.omp.exe 1024 1
*/

template <typename T>
void MatMul(unsigned n, T* A, unsigned ia, unsigned ja, unsigned sa,
                        T* B, unsigned ib, unsigned jb, unsigned sb,
                        T* C, unsigned ic, unsigned jc, unsigned sc, unsigned limit = 0) {

    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < n; j++) {
            auto index = (i + ic) * sc + j + jc;
            C[index] = 0;
            for (auto k = 0; k < n; k++)
                    C[index] +=
                    A[(i + ia) * sa + k + ja] *
                    B[(k + ib) * sb + j + jb];
            }
};

template <typename T>
void MatAdd(unsigned n, T* A, unsigned ia, unsigned ja, unsigned sa,
                        T* B, unsigned ib, unsigned jb, unsigned sb,
                        T* C, unsigned ic, unsigned jc, unsigned sc) {
    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < n; j++)
        C[(i + ic) * sc + j + jc] =
        A[(i + ia) * sa + j + ja] +
        B[(i + ib) * sb + j + jb];
};

template <typename T>
void MatSub(unsigned n, T* A, unsigned ia, unsigned ja, unsigned sa,
                        T* B, unsigned ib, unsigned jb, unsigned sb,
                        T* C, unsigned ic, unsigned jc, unsigned sc) {
    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < n; j++)
        C[(i + ic) * sc + j + jc] =
        A[(i + ia) * sa + j + ja] -
        B[(i + ib) * sb + j + jb];
};

template <typename T>
inline void MatrixZero (T* M, unsigned n) {
    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < n; j++)
            M[i * n + j] = 0;
}

template <typename T>
inline void MatrixE (T* M, unsigned n) {
    MatrixZero(M, n);
    for (auto i = 0; i < n; i++)
        M[i * n + i] = 1;
}

template <typename T>
inline void MatrixABC (T* M, unsigned n) {
    for (auto i = 0; i < n; i++)
            for (auto j = 0; j < n; j++) {
            auto index = i * n + j;
            M[index] = index;
        }
}

template <typename T>
inline void PrintMatrix (T* M, unsigned n) {
    for (auto i = 0; i < n; i++) {
            for (auto j = 0; j < n; j++)
                std::cout << std::setw(5) << M[i * n + j];
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
inline void PrintMatrixStride (T* M, unsigned n,
 unsigned im, unsigned jm, unsigned sm) {
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < n; j++)
            //std::cout << std::setw(5) << (i + im) * sm + j + jm;  // print index (debug)
            std::cout << std::setw(5) << M[(i + im) * sm + j + jm]; // print value
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// SW alg. with openMP. Best performance at 4 threads (4 matmuls' calls in parallel)
// Additional improvment may be apply.
// I.e., merge secions of 2 and 3 and add to this new section calc of P2;
// I guess sections of 6-7 may be rewrite in simplier way.
template <typename T>
void MatMulSV (unsigned n, T* A, unsigned ia, unsigned ja, unsigned sa,
                           T* B, unsigned ib, unsigned jb, unsigned sb,
                           T* C, unsigned ic, unsigned jc, unsigned sc,
                            unsigned limit) {

    if (n <= limit) {
        MatMul(n, A, ia, ja, sa, B, ib, jb, sb, C, ic, jc, sc);
        return;
    }

    unsigned n2 = n >> 1; // n / 2

    // use 8 additional buffers of size n * n / 4
    T *S1 = new T [2 * n * n];
    T *S2 = S1 + n2 * n2;
    T *S3 = S2 + n2 * n2;
    T *S4 = S3 + n2 * n2;
    T *S5 = S4 + n2 * n2;
    T *S6 = S5 + n2 * n2;
    T *S7 = S6 + n2 * n2;
    T *S8 = S7 + n2 * n2;

    // Task section #1
    #pragma omp task
    MatAdd  (n2, A, ia+n2, ja, sa, A, ia+n2, ja+n2, sa, S1, 0, 0, n2);
    #pragma omp task
    MatSub  (n2, A, ia, ja, sa, A, ia+n2, ja, sa, S3, 0, 0, n2);
    #pragma omp task
    MatSub  (n2, B, ib, jb+n2, sb, B, ib, jb, sb, S5, 0, 0, n2);
    #pragma omp task
    MatSub  (n2, B, ib+n2, jb+n2, sb, B, ib, jb+n2, sb, S7, 0, 0, n2);
    #pragma omp taskwait

    // Task section #2
    #pragma omp task
    MatSub  (n2, S1, 0, 0, n2, A, ia, ja, sa, S2,  0,  0, n2);
    #pragma omp task
    MatSub  (n2, B, ib+n2, jb+n2, sb, S5, 0, 0, n2, S6, 0, 0, n2);
    #pragma omp taskwait

    // Task section #3
    #pragma omp task
    MatSub  (n2, S6, 0, 0, n2, B, ib+n2, jb, sb, S8, 0, 0, n2);
    #pragma omp task
    MatSub  (n2, A, ia, ja+n2, sa, S2, 0, 0, n2, S4, 0, 0, n2);
    #pragma omp taskwait

    // Task section #4
    #pragma omp task
    MatMulSV(n2, S2, 0, 0, n2, S6,  0, 0, n2, C, ic, jc+n2, sc, limit);    // P1
    #pragma omp task
    MatMulSV(n2, A, ia, ja, sa, B, ib, jb, sb, C, ic, jc, sc, limit);      // P2
    #pragma omp task
    MatMulSV(n2, A, ia+n2, ja+n2, sa, S8, 0, 0, n2, C, ic+n2, jc, sc, limit);   // P7
    #pragma omp task
    MatMulSV(n2, S1, 0, 0, n2, S5, 0, 0, n2, C, ic+n2, jc+n2, sc, limit);  // P5
    #pragma omp taskwait

    // Task section #5
    #pragma omp task
    MatMulSV(n2, A, ia, ja+n2, sa, B, ib+n2, jb, sb, S1, 0, 0, n2, limit); // P3
    #pragma omp task
    MatMulSV(n2, S3, 0, 0, n2, S7, 0, 0, n2, S5, 0, 0, n2, limit);         // P4
    #pragma omp task
    MatMulSV(n2, S4, 0, 0, n2, B, ib+n2, jb+n2, sb, S2, 0, 0, n2, limit);  // P6
    #pragma omp task
    MatAdd  (n2, C, ic, jc+n2, sc, C, ic, jc, sc, C,  ic, jc+n2, sc);      // T1
    #pragma omp taskwait

    // Task section #6
    #pragma omp task
    MatAdd  (n2, C, ic, jc, sc, S1, 0, 0, n2, C, ic, jc, sc);              // C11
    #pragma omp task
    MatAdd  (n2, C, ic+n2, jc+n2, sc, S2, 0, 0, n2, S6, 0, 0, n2);         // P5 + P6
    #pragma omp task
    MatAdd  (n2, C, ic, jc+n2, sc, S5, 0, 0, n2, S3, 0, 0, n2);            // T2
    #pragma omp taskwait

    // Task section #7
    #pragma omp task
    MatAdd  (n2, C, ic, jc+n2, sc, S6, 0, 0, n2,  C, ic, jc+n2, sc);      // C12
    #pragma omp task
    MatSub  (n2, S3, 0, 0, n2, C, ic+n2, jc, sc, C, ic+n2,  jc, sc);      // C21
    #pragma omp task
    MatAdd  (n2, C, ic+n2, jc+n2, sc, S3, 0, 0, n2,  C, ic+n2, jc+n2, sc);// C22
    #pragma omp taskwait

    delete[] S1;
};

int main (const int argc, const char * argv[]) {
    unsigned n  = (argc > 1) ? atol(argv[1]) : 32; // matrix dim size
    unsigned threads = (argc > 2) ? atoi(argv[2]) : 1;
    unsigned limit = 32; // at this limit use classic matmul

    if (!((n > 0) && ((n & (n - 1)) == 0))) {
      std::cout << "Dimension 'n' has to be a power of 2." << std::endl;
      return 1;
    }

    std::cout << " Matrix size: " << n << " x " << n <<
      ". Threads: " << threads << ". Limit: " << limit << std::endl;

    int* A = new int [n * n];
    int* B = new int [n * n];
    int* C = new int [n * n];
    int* D = new int [n * n];
    int* E = new int [n * n];

    MatrixABC (A, n);
    MatrixE (B, n);

    double tm_sv = omp_get_wtime();
    #pragma omp parallel num_threads(threads) shared(n, limit, A, B, C)
    {
          #pragma omp single nowait
          MatMulSV(n, A, 0, 0, n, B, 0, 0, n, C, 0, 0, n, limit);
    }
    tm_sv = omp_get_wtime() - tm_sv;

    double tm_n3 = omp_get_wtime();
    #pragma omp parallel for num_threads(threads)
    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < n; j++){
            auto index = i * n + j;
            D[index] = 0;
            for (auto k = 0; k < n; k++)
                D[index] += A[i * n + k] * B[k * n + j];
        }
    tm_n3 = omp_get_wtime() - tm_n3;

    MatSub(n, C, 0, 0, n, D, 0, 0, n, E, 0, 0, n);
    if (n <= 32) {
        std::cout << std::endl << "A" << std::endl;
        PrintMatrix(A, n);
        std::cout << std::endl << "B" << std::endl;
        PrintMatrix(B, n);
        std::cout << std::endl << "C (Strassen-Winograd)" << std::endl;
        PrintMatrix(C, n);
        std::cout << std::endl << "D (Classic MatMul)" << std::endl;
        PrintMatrix(D, n);
        std::cout << std::endl << "E (Difference)" << std::endl;
        PrintMatrix(E, n);
    }
    else {
        std::cout << "Control print:\t";
        std::cout << "MIN, MAX of Difference matrix: " << *std::min_element(E, E + n * n) << " " << *std::max_element(E, E + n * n) << std::endl;
    }

    std::cout << "Strassen-Winograd time, s: " << tm_sv << std::endl;
    std::cout << "Classic    MatMul time, s: " << tm_n3 << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    return 0;
}
