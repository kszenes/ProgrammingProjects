#ifndef MOLECULE_H
#define MOLECULE_H

#include "Eigen/Core"
#include <string>
#include <vector>

struct Molecule {
  Molecule(const std::string &fname);

  void print() const;
  int size() const;
  double get_dist(int i, int j) const;
  double get_angle(int i, int center, int j) const;
  double get_oop_angle(int i, int j, int center, int l) const;
  double get_dihedral(int i, int j, int k, int l) const;
  Eigen::Vector3d get_com() const;
  void translate(const Eigen::Vector3d &vec);
  Eigen::Vector3d prime_mom_intertia() const;
  void print_mol_rot_type() const;

  std::vector<int> charges;
  std::vector<Eigen::Vector3d> atoms;
};

#endif // MOLECULE_H
