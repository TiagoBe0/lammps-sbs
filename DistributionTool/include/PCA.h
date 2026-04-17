#pragma once
#include <vector>
#include <Eigen/Dense>

class PCA {
public:
    explicit PCA(int n_components = 2);

    // Fit principal axes on reference DVs (zero-centered around their mean)
    void fit(const std::vector<std::vector<double>>& dvs);

    // Project DVs onto the first n_components principal axes
    // Returns matrix of shape [dvs.size()][n_components]
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& dvs) const;

    bool fitted() const { return fitted_; }
    int  nComponents() const { return n_components_; }

private:
    int  n_components_;
    bool fitted_ = false;
    std::vector<double> mean_;
    Eigen::MatrixXd     components_; // shape [D, n_components]
};
