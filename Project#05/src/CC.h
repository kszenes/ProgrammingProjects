#ifndef CC_H
#define CC_H

#include "Indexing.h"
#include "IntegralParser.h"
#include "SCF.h"
#include "unsupported/Eigen/CXX11/Tensor"

class CC {
public:
  CC(const IntegralParser &ints, const SCF &scf) : ints_(ints), scf_(scf) {}
  void run();

private:
  Eigen::VectorXd get_eri_mo(const bool fast_algo = true) const;
  Eigen::MatrixXd spatial2spin_1b(const Eigen::MatrixXd &mat_in) const;
  Eigen::Tensor<double, 4>
  spatial2spin_2b(const Eigen::VectorXd &tensor_in) const;
  Eigen::MatrixXd get_fock_spin() const;
  void init_amplitudes();
  void compute_mp2_e() const;

  void prepare();

  IntegralParser ints_;
  SCF scf_;
  Eigen::Tensor<double, 4> eri; // in spin basis
  Eigen::MatrixXd fock;         // in spin basis
  Eigen::MatrixXd t1;
  Eigen::Tensor<double, 4> t2;
};

#endif // CC_H
