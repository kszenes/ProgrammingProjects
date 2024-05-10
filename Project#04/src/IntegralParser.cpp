#include "IntegralParser.h"
#include "Eigen/Eigenvalues"
#include "Indexing.h"
#include <fmt/core.h>
#include <fstream>
#include <iostream>

IntegralParser::IntegralParser(const std::string &dir_name) {
  get_nelec(dir_name + "/geom.dat");
  parse_e_nuc(dir_name + "/enuc.dat");
  S = one_elec_parser(dir_name + "/s.dat");
  std::cout << "Overlap\n" << S << "\n\n";
  T = one_elec_parser(dir_name + "/t.dat");
  std::cout << "Kinetic\n" << T << "\n\n";
  V = one_elec_parser(dir_name + "/v.dat");
  std::cout << "Nuclear\n" << V << "\n\n";
  eri = two_elec_parser(dir_name + "/eri.dat");
  // std::cout << "ERI\n" << eri << "\n\n";

  orthogonalize_S();
  std::cout << "Orthogonalized S\n" << S << "\n\n";

  mu.x = one_elec_parser(dir_name + "/mux.dat");
  mu.y = one_elec_parser(dir_name + "/muy.dat");
  mu.z = one_elec_parser(dir_name + "/muz.dat");
}

int IntegralParser::get_nelec() const { return n_elec; }
double IntegralParser::get_vnn() const { return v_nn; }
Eigen::MatrixXd IntegralParser::get_S() const { return S; }
Eigen::MatrixXd IntegralParser::get_T() const { return T; }
Eigen::MatrixXd IntegralParser::get_V() const { return V; }
Eigen::VectorXd IntegralParser::get_eri() const { return eri; }
DipoleIntegrals IntegralParser::get_mu() const { return mu; }

std::ifstream get_file(const std::string &fname) {
  std::ifstream file(fname);
  if (!file.is_open()) {
    fmt::println("Failed to open overlap file: {}", fname);
    exit(1);
  }
  return file;
}

void IntegralParser::get_nelec(const std::string &fname) {
  auto file = get_file(fname);

  int n_nuc;
  file >> n_nuc;

  double charge;
  double x, y, z;

  for (int i = 0; i < n_nuc; ++i) {
    file >> charge >> x >> y >> z;
    n_elec += static_cast<int>(charge);
  }

  fmt::println("Number of electrons: {}\n", n_elec);
}

void IntegralParser::parse_e_nuc(const std::string &fname) {
  auto file = get_file(fname);

  file >> this->v_nn;
  fmt::println("Nuclear energy: {}\n", this->v_nn);
}

// Matrix(mu_i, nu_j)
Eigen::MatrixXd
IntegralParser::one_elec_parser(const std::string &fname) const {
  auto file = get_file(fname);

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

int get_num_ao(const std::string &fname) {
  auto file = get_file(fname);
  std::string line;
  int mu, nu, lambda, sigma;
  double val;
  int cur = 0;
  while (!file.eof()) {
    file >> mu >> nu >> lambda >> sigma >> val;
    cur = std::max(mu, cur);
  }
  fmt::println("Num Atomic Orbitals: {}", cur);
  return cur;
}

// Matrix(mu_i, nu_j, lambda_k, sigma_l)
Eigen::VectorXd
IntegralParser::two_elec_parser(const std::string &fname) const {
  int n_ao = get_num_ao(fname);

  int M = n_ao * (n_ao + 1) / 2;
  int N = M * (M + 1) / 2;
  Eigen::VectorXd eri = Eigen::VectorXd::Zero(N);

  auto file = get_file(fname);

  int mu, nu, lambda, sigma;
  double val;

  int index;

  while (!file.eof()) {
    file >> mu >> nu >> lambda >> sigma >> val;
    index = get_4index(mu - 1, nu - 1, lambda - 1, sigma - 1);
    eri(index) = val;
  }
  return eri;
}

void IntegralParser::orthogonalize_S() {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(this->S);
  Eigen::VectorXd vals = solver.eigenvalues();
  Eigen::MatrixXd vecs = solver.eigenvectors();

  S = vecs * vals.cwiseInverse().cwiseSqrt().asDiagonal() * vecs.transpose();
}
