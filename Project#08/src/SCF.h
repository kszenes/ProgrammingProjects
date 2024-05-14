#ifndef SCF_H
#define SCF_H

#include "Eigen/Core"
#include "IntegralParser.h"
#include "torch/torch.h"

struct Errors {
  double energy;
  double density;
};

struct SCF {
public:
  SCF(IntegralParser &integrals);
  void run();

  void build_core_hamiltonian();
  void build_guess();
  void compute_energy();
  Errors get_errors() const;
  void build_fock();
  void build_coeffs();
  void build_density();
  Eigen::MatrixXd get_fock_mo() const;
  double get_eelec() const;
  double get_etot() const;
  torch::Tensor get_mo_energies() const;

  const int n_ao = 0;
  const int n_occ = 0;
  const double tol = 1e-9;
  const int n_iter = 100;
  double energy = 0.0;
  std::vector<double> energies;
  IntegralParser integrals;
  torch::Tensor H_core;
  torch::Tensor Fock;
  torch::Tensor Coeffs;
  torch::Tensor Density;
  torch::Tensor prev_Density;
};

#endif // SCF_H
