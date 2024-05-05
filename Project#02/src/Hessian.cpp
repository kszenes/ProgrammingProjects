#include "Hessian.h"
#include "Eigen/Eigenvalues"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include <cmath>
#include <fstream>
#include <iostream>

Hessian::Hessian(const std::string &fname) { build_matrix(parse_file(fname)); }

HessCols Hessian::parse_file(const std::string &fname) const {
  std::ifstream file(fname);
  if (!file.is_open()) {
    fmt::print("Hessian file {} could not be opened.", fname);
    exit(1);
  }
  int n_atoms;
  file >> n_atoms;

  int file_rows = 3 * n_atoms * n_atoms;

  std::vector<double> xs(file_rows), ys(file_rows), zs(file_rows);

  for (int i = 0; i < file_rows; ++i) {
    file >> xs[i] >> ys[i] >> zs[i];
  }

  return HessCols{xs, ys, zs};
}

// Structure of Hessian
// x1x1 x1y1 x1z1 x1x2 ... x1zN
// y1x1
// ...
// zNx1
void Hessian::build_matrix(const HessCols &hess_cols) {
  auto &[xs, ys, zs] = hess_cols;

  // Number of rows in hessian file = 3 * N^2
  int mat_rows = sqrt(3 * xs.size()); // sqrt(9 * N^2)
  int n_atoms = sqrt(xs.size() / 3);  // N

  data = Eigen::MatrixXd::Zero(mat_rows, mat_rows);

  for (int row = 0; row < mat_rows; ++row) {
    for (int j = 0; j < n_atoms; ++j) {
      int col_idx = j * n_atoms;

      data(row, col_idx) = xs[row * n_atoms + j];
      data(row, col_idx + 1) = ys[row * n_atoms + j];
      data(row, col_idx + 2) = zs[row * n_atoms + j];
    }
  }
}

void Hessian::mass_reweigth(const Molecule &mol) {
  int n_rows = this->data.rows();
  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_rows; ++col) {
      int row_atom = row / 3;
      int col_atom = col / 3;
      double m_row = mol.get_mass(row_atom);
      double m_col = mol.get_mass(col_atom);

      this->data(row, col) /= sqrt(m_row * m_col);
    }
  }
}

void Hessian::print() const { std::cout << "Hessian:\n" << this->data << '\n'; }

void Hessian::check_sym() const {
  bool is_sym = this->data.isApprox(this->data.transpose());

  if (is_sym) {
    fmt::println("Hessian is SYMMETRIC");
  } else {
    fmt::println("Hessian is NOT symmetric");
  }
}

void Hessian::get_freq() const {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(this->data);

  auto vals = solver.eigenvalues();

  // Conversion factor from Ha/Bohr^2 -> cm^-1
  double factor = 5140.4845318177295;

  fmt::println("eigvals = {}", vals);
  fmt::println("we = {}\n", factor * vals.cwiseAbs().cwiseSqrt());

}
