#ifndef SCFANALYSER_H
#define SCFANALYSER_H

#include "Eigen/Core"
#include "IntegralParser.h"
#include "SCF.h"
#include <vector>

class Analyser {
public:
  Analyser(const std::string &dir_name, const IntegralParser &integrals,
           const SCF &scf);
  void analyze();

private:
  void parse_geometry(const std::string &dir_name);
  void print_fock_mo() const;
  Eigen::Vector3d get_com() const;
  void translate(const Eigen::Vector3d &vec);

  Eigen::Vector3d compute_dipole_nuc() const;
  Eigen::Vector3d compute_dipole_elec() const;
  void analyze_dipole() const;

  IntegralParser integrals;
  SCF scf;

  std::vector<double> charges;
  std::vector<Eigen::Vector3d> atoms;
};

#endif // SCFANALYSER_H
