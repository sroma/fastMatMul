# fastMatMul
Strassen-Winograd Algorithm (single and OpenMP)

sw_single.cpp
/*  
    Strassen-Winograd Algorithm
    n - dimension of square matrices A, B, C.
        Variable n has to be a power of two;
    i, j - start row and col of matrices;
    s - stride of matrices;
    limit - size to call classic MatMul.

    g++ sw_single.cpp --std=c++11 -o sw.exe
    ./sw.exe 1024
    
    Use auxilary buffer of n * n
*/

sw_openmp.cpp
/*  
    Strassen-Winograd Algorithm
    n - dimension of square matrices A, B, C.
        Variable n has to be a power of two;
    i, j - start row and col of matrices;
    s - stride of matrices;
    limit - size to call classic MatMul.

    g++ sw_omp.cpp -fopenmp --std=c++11 -o sw.omp.exe
    ./sw.omp.exe 1024 1
    
    Use auxilary buffer of 2 * n * n
    // SW alg. with openMP. Best performance at 4 threads (4 matmuls' calls in parallel)
    // Additional improvment may be apply.
    // I.e., merge secions of 2 and 3 and add to this new section calc of P2;
    // I guess sections of 6-7 may be rewrite in simplier way.
*/
