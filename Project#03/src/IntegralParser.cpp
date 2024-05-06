#include "IntegralParser.h"
#include <fmt/core.h>
#include <fstream>
#include <iostream>

IntegralParser::IntegralParser(const std::string &dir_name) {
  parse_e_nuc(dir_name + "/enuc.dat");
  S = one_elec_parser(dir_name + "/s.dat");
  std::cout << "Overlap\n" << S << "\n\n";
  T = one_elec_parser(dir_name + "/t.dat");
  std::cout << "Kinetic\n" << T << "\n\n";
  V = one_elec_parser(dir_name + "/v.dat");
  std::cout << "Nuclear\n" << V << "\n\n";

  build_core_hamiltonian();
  std::cout << "Core Hamiltonian\n" << H_core << "\n\n";

}

void IntegralParser::parse_e_nuc(const std::string &fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    fmt::println("Failed to open enuc file: {}", fname);
    exit(1);
  }

  file >> this->v_nn;
  fmt::println("Nuclear energy: {}\n", this->v_nn);
}

// Matrix(mu_i, nu_j)
Eigen::MatrixXd
IntegralParser::one_elec_parser(const std::string &fname) const {
  std::ifstream file(fname);
  if (!file.is_open()) {
    fmt::println("Failed to open overlap file: {}", fname);
    exit(1);
  }

  // Parse file
  std::vector<int> mus, nus;
  std::vector<double> vals;

  int mu, nu;
  double val;
  while (!file.eof()) {
    file >> mu >> nu >> val;
    mus.push_back(mu);
    nus.push_back(nu);
    vals.push_back(val);
  }

  // Find number of orbitals
  int max_mu = *std::max_element(mus.begin(), mus.end());
  int max_nu = *std::max_element(nus.begin(), nus.end());
  int n = std::max(max_mu, max_nu);

  // Fill matrix
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(n, n);
  for (size_t i = 0; i < vals.size(); ++i) {
    // change to zero indexed
    mat(mus[i] - 1, nus[i] - 1) = vals[i];
  }

  // Symmetrise
  Eigen::MatrixXd mat_trans = mat.transpose();
  mat = mat + mat.transpose().eval();
  mat.diagonal() /= 2;

  return mat;
}

void IntegralParser::build_core_hamiltonian() {
  H_core = T + V;
}
