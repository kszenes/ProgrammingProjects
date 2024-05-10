#include "SCF.h"
#include "Eigen/Eigenvalues"
#include "Indexing.h"
#include "fmt/core.h"
#include <iostream>

SCF::SCF(IntegralParser &integrals)
    : n_ao(integrals.get_S().rows()), n_occ(integrals.get_nelec() / 2),
      integrals(integrals) {
  build_core_hamiltonian();
  Fock = H_core;
  build_density();
  energies.push_back(energy);
}

void SCF::build_core_hamiltonian() {
  H_core = integrals.get_T() + integrals.get_V();
  std::cout << "Core Hamiltonian\n" << H_core << "\n\n";
}

void SCF::build_coeffs() {
  Eigen::MatrixXd Fock_transformed =
      integrals.get_S().transpose() * Fock * integrals.get_S();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Fock_transformed);
  Coeffs = integrals.get_S() * solver.eigenvectors();
}

// Computes guess density and initial Fock corresponding to H_core
void SCF::build_density() {
  build_coeffs();

  prev_Density = Density;
  Density = Coeffs.leftCols(n_occ) * Coeffs.leftCols(n_occ).transpose();
}

void SCF::build_fock() {
  Fock = H_core;
  for (int mu = 0; mu < n_ao; ++mu) {
    for (int nu = 0; nu < n_ao; ++nu) {
      double val = 0.0;
      for (int lambda = 0; lambda < n_ao; ++lambda) {
        for (int sigma = 0; sigma < n_ao; ++sigma) {
          val += Density(lambda, sigma) *
                 (2 * integrals.get_eri()(get_4index(mu, nu, lambda, sigma)) -
                  integrals.get_eri()(get_4index(mu, lambda, nu, sigma)));
        }
      }
      Fock(mu, nu) += val;
    }
  }
}

void SCF::compute_energy() {
  energy = Density.cwiseProduct(H_core + Fock).sum();
}

Errors SCF::get_errors() const {
  double e_error = energies.back() - energies.at(energies.size() - 2);
  double d_error = (prev_Density - Density).norm();

  return {e_error, d_error};
}

void SCF::run() {
  fmt::println("\n=== Starting SCF ===\n");
  fmt::println("{:>4} {:>15} {:>15} {:>15} {:>15}", "iter", "E_elec", "E_tot",
               "delta_E", "delta_D");
  for (int iter = 0; iter < n_iter; ++iter) {
    build_fock();
    build_density();
    compute_energy();
    energies.push_back(energy);
    auto [e_error, d_error] = get_errors();
    fmt::println("{:4} {: 15.9f} {: 15.9f} {: 15.9f} {: 15.9f}", iter,
                 energies.back(), energies.back() + integrals.get_vnn(),
                 e_error, d_error);
    if (abs(e_error) < tol || abs(d_error) < tol) {
      fmt::println("\n === SCF Converged === \n");
      return;
    }
  }
  throw std::runtime_error("SCF FAILED to converge");
}
