#include <iostream>
#include<random>
#include<vector>
#include<omp.h>
#include<iomanip>
/*2024BCS0012 Dhanvanth Balaji*/
/*series*/
std::vector<std::vector<double>> covariance_matrix_calc(std::vector<std::vector<double>>& finan_matrix, int M, int N);
int LU_Decomposition(std::vector<std::vector<double>>& covar_mat, std::vector<std::vector<double>>& L_mat, std::vector<std::vector<double>>& U_mat, std::vector<double> &B,int M);
std::vector<double> forward_sub(std::vector<std::vector<double>>& Left, std::vector<double>& Right, int M);
std::vector<double> backward_sub(std::vector<std::vector<double>>& Left, std::vector<double>& Right, int M);
void normalization(std::vector<double>& unnorm_vector, int M);
/*parallel versions*/
std::vector<std::vector<double>> covariance_matrix_calc_par(std::vector<std::vector<double>>& finan_matrix, int M, int N);
int LU_Decomposition_par(std::vector<std::vector<double>>& covar_mat, std::vector<std::vector<double>>& L_mat, std::vector<std::vector<double>>& U_mat,std::vector<double>& B_par, int M);
/*--------------------------------------------------------MATRIX CREATION---------------------------------------------------------------------------------------------------*/
void define_random_finan_mat(std::vector<std::vector<double>>& finan_matrix, int M, int N)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::student_t_distribution<double> dist(4.0);
    /*realistic market simulated scale factor and mean drift*/
    double scale_factor = 0.015;
    double mean_drift = 0.0003;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            finan_matrix[i][j] = (mean_drift + (dist(engine) * scale_factor));
        }
    }
}
void identity_creation(std::vector<std::vector<double>>& L_mat, int M)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            if (i == j)
            {
                L_mat[i][j] = 1.0;
            }
            else
            {
                L_mat[i][j] = 0.0;
            }
        }
    }
}
void B_matrix(std::vector<double> &new_vector, int M)
{
    for (int i = 0; i < M; i++)
    {
        new_vector[i] = 1;
    }
}
void thread_benchmarks(int no_of_threads, int M, int N, std::vector<std::vector<double>> &Mat_par, std::vector<std::vector<double>> &L_mat_par, std::vector<std::vector<double>> &U_mat_par, std::vector<double> &B_par, double serial_taken)
{
    omp_set_num_threads(no_of_threads);
    double start = omp_get_wtime();
    auto covar_par = covariance_matrix_calc_par(Mat_par, M, N);
    LU_Decomposition_par(covar_par, L_mat_par, U_mat_par,B_par,M);
    auto Y_par = forward_sub(L_mat_par, B_par, M);
    auto X_par = backward_sub(U_mat_par, Y_par, M);
    double end = omp_get_wtime();
    double parallel_taken = end - start;
    std::cout << std::left
    << std::setw(12) << no_of_threads
    << std::setw(18) << parallel_taken
    << std::setw(15) << serial_taken / parallel_taken
    << std::setw(15)<< (serial_taken / (no_of_threads * parallel_taken)*100) << "\n";
}
void portfolio_optimization_bechmarks(int M, int N)
{
    std::vector<std::vector<double>> Mat_O(M, std::vector<double>(N));
    std::vector<std::vector<double>> L_mat_O(M, std::vector<double>(M));
    std::vector<std::vector<double>> U_mat_O(M, std::vector<double>(M));
    std::vector<double> B_O(M, 0.0);
    B_matrix(B_O, M);
    identity_creation(L_mat_O, M);
    define_random_finan_mat(Mat_O, M, N);
    auto Mat = Mat_O;
    auto L_mat = L_mat_O;
    auto U_mat = U_mat_O;
    auto B = B_O;
    double start = omp_get_wtime();
    auto covar = covariance_matrix_calc(Mat, M, N);
    LU_Decomposition(covar, L_mat, U_mat, B,M);
    auto Y = forward_sub(L_mat, B, M);
    auto X = backward_sub(U_mat, Y, M);
    double end = omp_get_wtime();
    double serial_taken = end - start;
    std::cout << std::left
    << std::setw(12) << "threadnum"
    << std::setw(18) << "Time taken"
    << std::setw(15) << "Speedup"
    << std::setw(15) << "Efficiency" << "\n";
    

    std::cout << std::left
    << std::setw(12) << "Serial"
    << std::setw(18) << (end - start)
    << std::setw(15) << 1.0
    << std::setw(15) << "100.0" << "\n";
        
    std::vector<int> thread_array = { 2,4,8,16 };
    for (int i = 0; i<thread_array.size(); i++)
    {
        auto Mat_par = Mat_O;
        auto L_mat_par = L_mat_O;
        auto U_mat_par = U_mat_O;
        auto B_par = B_O;
        thread_benchmarks(thread_array[i], M, N, Mat_par, L_mat_par, U_mat_par, B_par, serial_taken);
    }
}
int main()
{
    std::cout << "==================================================================\n";
    std::cout << " Student Name: Dhanvanth Balaji | Roll No: 2024BCS0012\n";
    std::cout << " PDC Assignment 1 - Accelerating Big-Data Quantitative Workloads using Shared-Memory Parallelism (Markowitz)\n";
    std::cout << "==================================================================\n";
    std::vector<int> size_array_1_2 = { 100, 300, 500, 1000, 1500 };
    int N = 1000;
    for (size_t i = 0; i < size_array_1_2.size(); i++)
    {
        std::cout << "\n==================================================================\n";
        std::cout << " Matrix Size (M x N): " << size_array_1_2[i] << " x " << N << "\n";
        std::cout << "==================================================================\n";
        portfolio_optimization_bechmarks(size_array_1_2[i], N);
    }
    return 0;
}


