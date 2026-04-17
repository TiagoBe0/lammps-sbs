#include "PCA.h"
#include <Eigen/Dense>
#include <stdexcept>

PCA::PCA(int n_components) : n_components_(n_components) {}

void PCA::fit(const std::vector<std::vector<double>>& dvs) {
    if (dvs.empty()) return;
    const int N = static_cast<int>(dvs.size());
    const int D = static_cast<int>(dvs[0].size());

    // Compute mean
    mean_.assign(D, 0.0);
    for (const auto& dv : dvs)
        for (int j = 0; j < D; ++j) mean_[j] += dv[j];
    for (auto& v : mean_) v /= N;

    // Build centred data matrix X [N x D]
    Eigen::MatrixXd X(N, D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            X(i, j) = dvs[i][j] - mean_[j];

    // Covariance matrix C = X^T X / (N-1)
    Eigen::MatrixXd C = (X.transpose() * X) / (N > 1 ? N - 1 : 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("PCA eigendecomposition failed");

    // Eigenvalues are sorted ascending; take last n_components (largest)
    const int k = std::min(n_components_, D);
    components_.resize(D, k);
    for (int c = 0; c < k; ++c)
        components_.col(c) = es.eigenvectors().col(D - 1 - c);

    fitted_ = true;
}

std::vector<std::vector<double>> PCA::transform(
        const std::vector<std::vector<double>>& dvs) const {
    if (!fitted_) throw std::runtime_error("PCA not fitted");
    const int N = static_cast<int>(dvs.size());
    const int D = static_cast<int>(mean_.size());
    const int k = static_cast<int>(components_.cols());

    std::vector<std::vector<double>> out(N, std::vector<double>(k));
    for (int i = 0; i < N; ++i) {
        Eigen::VectorXd v(D);
        for (int j = 0; j < D; ++j) v(j) = dvs[i][j] - mean_[j];
        Eigen::VectorXd z = components_.transpose() * v;
        for (int c = 0; c < k; ++c) out[i][c] = z(c);
    }
    return out;
}
