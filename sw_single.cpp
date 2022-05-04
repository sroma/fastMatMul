#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <chrono>

/*  Strassen-Winograd Algorithm
    n - dimension of square matrices A, B, C.
        Variable n has to be a power of two;
    i, j - start row and col of matrices;
    s - stride of matrices;
    limit - size to call classic MatMul.

    OpenMP uses only for wtime

    g++ sw_single.cpp --std=c++11 -o sw.exe
    ./sw.exe 1024
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

// Strassen-Vinograd alg. BUF - auxilary buffer of n * n
template <typename T>
void MatMulSV (unsigned n, T* A, unsigned ia, unsigned ja, unsigned sa,
                           T* B, unsigned ib, unsigned jb, unsigned sb,
                           T* C, unsigned ic, unsigned jc, unsigned sc,
                           T* BUF, unsigned limit) {

    if (n <= limit) {
        MatMul(n, A, ia, ja, sa, B, ib, jb, sb, C, ic, jc, sc);
        return;
    }

    unsigned n2 = n >> 1; // n / 2

    // to store auxilary calculations only 4 marices of [n/2] x [n/2] need
    T *X = BUF;
    T *Y = X + n2 * n2;
    T *Z = Y + n2 * n2;
    T *W = Z + n2 * n2;

    // S3 = A11 - A21 = X
    MatSub  (n2, A, ia, ja, sa, A, ia+n2, ja, sa, X, 0, 0, n2);
    // S7 = B22 - B12 = Y
    MatSub  (n2, B, ib+n2, jb+n2, sb, B, ib, jb+n2, sb, Y, 0, 0, n2);
    // P4 = S3 * S7 = X * Y = C21
    MatMulSV(n2, X, 0, 0, n2, Y, 0, 0, n2, C, ic+n2, jc, sc, Z, limit);
    // S1 = A21 + A22 = X
    MatAdd  (n2, A, ia+n2, ja, sa, A, ia+n2, ja+n2, sa, X, 0, 0, n2);
    // S5 = B12 - B11 = Y
    MatSub  (n2, B, ib, jb+n2, sb, B, ib, jb, sb, Y, 0, 0, n2);
    // P5 = S1 * S5 = X * Y = C22
    MatMulSV(n2, X, 0, 0, n2, Y, 0, 0, n2, C, ic+n2, jc+n2, sc, Z, limit);
    // S2 = S1 - A11 = X - A11 = X
    MatSub  (n2, X, 0, 0, n2, A, ia, ja, sa, X,  0,  0, n2);
    // S6 = B22 - S5 = B22 - Y = Y
    MatSub  (n2, B, ib+n2, jb+n2, sb, Y, 0, 0, n2, Y, 0, 0, n2);
    // P1 = S2 * S6 = X * Y = C12
    MatMulSV(n2, X, 0, 0, n2, Y,  0, 0, n2, C, ic, jc+n2, sc, Z, limit);
    // S4 = A12 - S2 = A12 - X = X
    MatSub  (n2, A, ia, ja+n2, sa, X, 0, 0, n2, X, 0, 0, n2);
    // S8 = S6 - B21 = Y - B21 = Y
    MatSub  (n2, Y, 0, 0, n2, B, ib+n2, jb, sb, Y, 0, 0, n2);
    // P2 = A11 * B11 = C11
    MatMulSV(n2, A, ia, ja, sa, B, ib, jb, sb, C, ic, jc, sc, Z, limit);
    // T1 = P2 + P1 = C11 + C12 = C12
    MatAdd  (n2, C, ic, jc, sc, C,  ic, jc+n2, sc, C, ic, jc+n2, sc);
    // T2 = T1 + P4 = C12 + C21 = C21
    MatAdd  (n2, C, ic, jc+n2, sc, C, ic+n2, jc, sc, C, ic+n2, jc, sc);
    // C12 = T1 + P5 = C12 + C22
    MatAdd  (n2, C, ic, jc+n2, sc, C, ic+n2, jc+n2, sc, C,  ic, jc+n2, sc);
    // C22 = P5 + T2 = C22 + C21
    MatAdd  (n2, C, ic+n2, jc+n2, sc, C, ic+n2, jc, sc, C, ic+n2, jc+n2,  sc);
    // P3 = A12 * B21 = Z
    MatMulSV(n2, A, ia, ja+n2, sa, B, ib+n2, jb, sb, Z, 0, 0, n2, W, limit);
    // C11 = P2 + P3 = C11 + Z
    MatAdd  (n2, C, ic, jc, sc, Z, 0, 0, n2, C, ic, jc, sc);
    // P6 = S4 * B22 = X * B22 = Z
    MatMulSV(n2, X, 0, 0, n2, B, ib+n2, jb+n2, sb, Z, 0, 0, n2, W, limit);
    // C12 = C12 + P6 = C12 + Z
    MatAdd  (n2, C, ic, jc+n2, sc, Z, 0, 0, n2, C, ic, jc+n2, sc);
    // P7 = A22 * S8 = A22 * Y = Z
    MatMulSV(n2, A, ia+n2, ja+n2, sa, Y, 0, 0, n2, Z, 0, 0, n2, W, limit);
    // C21 = T2 - P7 = C21 - Z
    MatSub  (n2, C, ic+n2,  jc, sc, Z, 0, 0, n2, C, ic+n2, jc, sc);
};

int main (const int argc, const char * argv[]) {
    unsigned n  = (argc > 1) ? atol(argv[1]) : 32;
    unsigned limit = 32; // at this limit use classic matmul

    if (!((n > 0) && ((n & (n - 1)) == 0))) {
      std::cout << "Dimension 'n' has to be a power of 2." << std::endl;
        return 1;
    }

    std::cout << " Matrix size: " << n << " x " << n << ". Limit: " << limit << std::endl;

    int* A = new int [n * n];
    int* B = new int [n * n];
    int* C = new int [n * n];
    int* D = new int [n * n];
    int* E = new int [n * n];

    MatrixABC (A, n);
    MatrixE (B, n);

    // test of Strassen-Vinograd alg.
    auto tm0 = std::chrono::steady_clock::now();
    int *W = new int[n * n]; // auxilary buffer for inner calculations
    MatMulSV(n, A, 0, 0, n, B, 0, 0, n, C, 0, 0, n, W, limit);

    // test of classic matmul
    auto tm1 = std::chrono::steady_clock::now();
    for (auto i = 0; i < n; i++)
        for (auto j = 0; j < n; j++){
            auto index = i * n + j;
            D[index] = 0;
            for (auto k = 0; k < n; k++)
                D[index] += A[i * n + k] * B[k * n + j];
        }
    auto tm2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> sec_sw = tm1 - tm0;
    std::chrono::duration<double> sec_n3 = tm2 - tm1;

    MatSub(n, C, 0, 0, n, D, 0, 0, n, E, 0, 0, n);
    if (n <= 32) {
        std::cout << std::endl << "A" << std::endl;
        PrintMatrix(A, n);
        std::cout << std::endl << "B" << std::endl;
        PrintMatrix(B, n);
        std::cout << std::endl << "C (Strassen-Vinograd)" << std::endl;
        PrintMatrix(C, n);
        std::cout << std::endl << "D (Classic MatMul)" << std::endl;
        PrintMatrix(D, n);
        std::cout << std::endl << "E (Difference)" << std::endl;
        PrintMatrix(E, n);
    }
    else {
        std::cout << "Control print:\t";
        std::cout << "MIN, MAX of Difference matrix: " << *std::min_element(E, E + n * n)
         << " " << *std::max_element(E, E + n * n) << std::endl;
    }

    std::cout << "Strassen-Vinograd time, s: " << sec_sw.count() << std::endl;
    std::cout << "Classic    MatMul time, s: " << sec_n3.count() << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    delete[] W;
    return 0;
}
