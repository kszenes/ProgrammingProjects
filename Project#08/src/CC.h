#ifndef CC_H
#define CC_H

#include "Indexing.h"
#include "IntegralParser.h"
#include "SCF.h"
#include "unsupported/Eigen/CXX11/Tensor"

class CC {
public:
  CC(const IntegralParser &ints, const SCF &scf)
      : n_ao(2 * scf.n_ao), n_occ(2 * scf.n_occ),
        n_virtual(2 * (scf.n_ao - scf.n_occ)), ints_(ints), scf_(scf) {}
  void run();

private:
  Eigen::VectorXd get_eri_mo() const;
  Eigen::MatrixXd spatial2spin_1b(const Eigen::MatrixXd &mat_in) const;
  Eigen::Tensor<double, 4>
  spatial2spin_2b(const Eigen::VectorXd &tensor_in) const;
  Eigen::Tensor<double, 4>
  antisymmetrise(const Eigen::Tensor<double, 4> &tensor_in) const;
  Eigen::MatrixXd get_fock_spin() const;
  void init_amplitudes();
  double compute_mp2_energy() const;
  void build_taus();

  void build_intermediates();
  void build_F();
  void build_W();

  void build_denominators();
  void build_D1();
  void build_D2();

  void update_amplitudes();
  Eigen::MatrixXd compute_t1() const;
  Eigen::Tensor<double, 4> compute_t2() const;

  double get_cc_energy();

  void prepare();

  // in spin orbitals
  const int n_ao;
  const int n_occ;
  const int n_virtual;

  IntegralParser ints_;
  SCF scf_;
  Eigen::Tensor<double, 4> eri_anti; // <pq||rs> antsimmetrised in spin basis
  Eigen::MatrixXd fock;              // in spin basis

  // Ampllitudes
  Eigen::MatrixXd t1;
  Eigen::Tensor<double, 4> t2;
  // Effective two particle excitation operators
  Eigen::Tensor<double, 4> tau;
  Eigen::Tensor<double, 4> tau_tilde;
  // Intermediates
  Eigen::MatrixXd F;
  Eigen::Tensor<double, 4> W;
  // Denominator Arrays
  Eigen::MatrixXd D1;
  Eigen::Tensor<double, 4> D2;

  double energy;
};

#endif // CC_H
