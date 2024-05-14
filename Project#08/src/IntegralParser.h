#ifndef INTEGRAL_PARSER_H
#define INTEGRAL_PARSER_H

#include "Eigen/Core"
#include <string>
#include "torch/torch.h"

struct DipoleIntegrals {
  torch::Tensor x;
  torch::Tensor y;
  torch::Tensor z;
};

class IntegralParser {
public:
  IntegralParser(const std::string &dir_name);
  int get_nelec() const;
  double get_vnn() const;
  torch::Tensor get_S() const;
  torch::Tensor get_T() const;
  torch::Tensor get_V() const;
  Eigen::VectorXd get_eri() const;
  DipoleIntegrals get_mu() const;

private:
  void parse_e_nuc(const std::string &fname);
  void get_nelec(const std::string &fname);
  torch::Tensor one_elec_parser(const std::string &fname) const;
  Eigen::VectorXd two_elec_parser(const std::string &fname) const;

  void orthogonalize_S();

  int n_elec = 0;
  double v_nn;
  torch::Tensor S;
  torch::Tensor T;
  torch::Tensor V;
  torch::Tensor H_core;

  Eigen::VectorXd eri;

  DipoleIntegrals mu;
};

#endif // INTEGRAL_PARSER_H
