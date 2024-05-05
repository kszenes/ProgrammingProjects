#ifndef HESSIAN_H
#define HESSIAN_H

#include <string>
#include <vector>
#include "Eigen/Core"
#include "Molecule.h"

struct HessCols {
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> zs;
};

class Hessian {
public:
  Hessian(const std::string &fname);
  void mass_reweigth(const Molecule& mol);
  void print() const;
  void check_sym() const;
  void get_freq() const;

private:
  HessCols parse_file(const std::string &fname) const;
  void build_matrix(const HessCols& hess_cols);

  Eigen::MatrixXd data;
};

#endif // HESSIAN_H
