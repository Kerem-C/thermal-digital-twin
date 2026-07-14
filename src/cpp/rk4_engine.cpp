#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

std::vector<double> calculate_derivative(
    const double* T, const double* C, const double* P,
    const double* K_data, const int* K_indices, const int* K_indptr,
    int num_nodes)
{
    std::vector<double> dT(num_nodes, 0.0);

    for (int i = 0; i < num_nodes; ++i) {
        double dot_product = 0.0;

        for (int j = K_indptr[i]; j < K_indptr[i+1]; ++j) {
            dot_product += K_data[j] * T[K_indices[j]];
        }
        dT[i] = (dot_product + P[i]) / C[i];
    }

    return dT;
}

py::array_t<double> step_rk4(
    py::array_t<double> T_current, double dt,
    py::array_t<double> C_arr, py::array_t<double> P_arr,
    py::array_t<double> K_data_arr, py::array_t<int> K_indices_arr, py::array_t<int> K_indptr_arr) 
{
    auto buf_T = T_current.request();
    auto buf_C = C_arr.request();
    auto buf_P = P_arr.request();
    auto buf_K_data = K_data_arr.request();
    auto buf_K_indices = K_indices_arr.request();
    auto buf_K_indptr = K_indptr_arr.request();

    int num_nodes = buf_T.shape[0];
    const double* T = static_cast<double*>(buf_T.ptr);
    const double* C = static_cast<double*>(buf_C.ptr);
    const double* P = static_cast<double*>(buf_P.ptr);
    const double* K_data = static_cast<double*>(buf_K_data.ptr);
    const int* K_indices = static_cast<int*>(buf_K_indices.ptr);
    const int* K_indptr = static_cast<int*>(buf_K_indptr.ptr);

    std::vector<double> k1 = calculate_derivative(T, C, P, K_data, K_indices, K_indptr, num_nodes);

    std::vector<double> T_k2(num_nodes);
    for(int i=0; i<num_nodes; ++i) T_k2[i] = T[i] + 0.5 * dt * k1[i];
    std::vector<double> k2 = calculate_derivative(T_k2.data(), C, P, K_data, K_indices, K_indptr, num_nodes);

    std::vector<double> T_k3(num_nodes);
    for(int i=0; i<num_nodes; ++i) T_k3[i] = T[i] + 0.5 * dt * k2[i];
    std::vector<double> k3 = calculate_derivative(T_k3.data(), C, P, K_data, K_indices, K_indptr, num_nodes);

    std::vector<double> T_k4(num_nodes);
    for(int i=0; i<num_nodes; ++i) T_k4[i] = T[i] + dt * k3[i];
    std::vector<double> k4 = calculate_derivative(T_k4.data(), C, P, K_data, K_indices, K_indptr, num_nodes);

    auto result = py::array_t<double>(num_nodes);
    auto buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    for (int i = 0; i < num_nodes; ++i) {
        ptr_result[i] = T[i] + (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }

    return result;
}

PYBIND11_MODULE(rk4_backend, m) {
    m.doc() = "C++ Accelerated RK4 Integration Engine";
    m.def("step_rk4", &step_rk4, "Perform one RK4 step using CSR sparse arrays");
}
