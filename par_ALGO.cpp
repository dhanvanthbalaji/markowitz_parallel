#include <iostream>
#include<random>
#include<vector>
#include<omp.h>
/*2024BCS0012 Dhanvanth Balaji*/
void average_matrix_calc(std::vector<std::vector<double>>& finan_matrix, std::vector<double>& average_x_mat, int M, int N);
std::vector<std::vector<double>> covariance_matrix_calc_par(std::vector<std::vector<double>>& finan_matrix, int M, int N)
{
    std::vector<double> mean_xi(M, 0.0);
    std::vector<std::vector<double>> covar_i_j(M, std::vector<double>(M));
    average_matrix_calc(finan_matrix, mean_xi, M, N);
    for (int i = 0; i < M; i++)
    {
        #pragma omp parallel for schedule(guided)
        for (int j = 0; j < M; j++)
        {
            double covariance = 0.0;
            for (int k = 0; k < N;k++)
            {
                double x_deviation = finan_matrix[i][k] - mean_xi[i];
                double y_deviation = finan_matrix[j][k] - mean_xi[j];
                double co_var_dev = x_deviation * y_deviation;
                covariance += co_var_dev;
            }
            covar_i_j[i][j] = covariance / (N - 1);
        }
    }
    return covar_i_j;
}
/*-----------------------------------------------------LU FACTORIZATION-------------------------------------------------------------------------------------------------*/

int LU_Decomposition_par(std::vector<std::vector<double>>& covar_mat, std::vector<std::vector<double>>& L_mat, std::vector<std::vector<double>>& U_mat, std::vector<double> &B, int M)
{
    for (int i = 0; i < M; i++)
    {
        double max_value = std::abs(covar_mat[i][i]);
        int max_index = i;
        for (int j = i + 1; j < M; j++)
        {
            if (max_value < std::abs(covar_mat[j][i]))
            {
                max_index = j;
                max_value = std::abs(covar_mat[j][i]);
            }
        }
        for (int k = 0; k < M; k++)
        {
            double swapper = covar_mat[max_index][k];
            covar_mat[max_index][k] = covar_mat[i][k];
            covar_mat[i][k] = swapper;
        }
        for (int m = 0; m < i; m++)
        {
            double swapper_1 = L_mat[max_index][m];
            L_mat[max_index][m] = L_mat[i][m];
            L_mat[i][m] = swapper_1;
        }
        double swapper_2 = B[max_index];
        B[max_index] = B[i];
        B[i] = swapper_2;
        if ((std::abs(covar_mat[i][i]) <= 1e-9))return -1;
        #pragma omp parallel for schedule(guided)
        for (int l = i + 1; l < M; l++)
        {
            double multiplier = covar_mat[l][i] / covar_mat[i][i];
            for (int k = 0; k < M; k++)
            {
                covar_mat[l][k] -= (covar_mat[i][k] * multiplier);
            }
            L_mat[l][i] = multiplier;
        }

    }
    U_mat = covar_mat;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j< i; j++)
        {
            U_mat[i][j] = 0.0;
        }
    }
    return 1;
}
