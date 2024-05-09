#ifndef INTEGRAL_PARSER_H
#define INTEGRAL_PARSER_H

#include "Eigen/Core"
#include <string>

class IntegralParser {
public:
  IntegralParser(const std::string &dir_name);
  int get_nelec() const;
  double get_vnn() const;
  Eigen::MatrixXd get_S() const;
  Eigen::MatrixXd get_T() const;
  Eigen::MatrixXd get_V() const;
  Eigen::VectorXd get_eri() const;

private:
  void parse_e_nuc(const std::string &fname);
  void get_nelec(const std::string &fname);
  Eigen::MatrixXd one_elec_parser(const std::string &fname) const;
  Eigen::VectorXd two_elec_parser(const std::string &fname) const;

  void orthogonalize_S();

  int n_elec = 0;
  double v_nn;
  Eigen::MatrixXd S;
  Eigen::MatrixXd T;
  Eigen::MatrixXd V;
  Eigen::MatrixXd H_core;

  Eigen::VectorXd eri;
};

#endif // INTEGRAL_PARSER_H
