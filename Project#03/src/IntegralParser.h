#ifndef INTEGRAL_PARSER_H
#define INTEGRAL_PARSER_H

#include <string>
#include "Eigen/Core"

class IntegralParser {
public:
  IntegralParser(const std::string &dir_name);

private:
  void parse_e_nuc(const std::string& fname);
  Eigen::MatrixXd one_elec_parser(const std::string& fname) const;
  void build_core_hamiltonian();

  double v_nn;
  Eigen::MatrixXd S;
  Eigen::MatrixXd T;
  Eigen::MatrixXd V;
  Eigen::MatrixXd H_core;
};

#endif // INTEGRAL_PARSER_H
