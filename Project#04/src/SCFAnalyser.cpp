#include "SCFAnalyser.h"
#include "fmt/core.h"
#include "masses.h"
#include <fstream>
#include <iostream>

Analyser::Analyser(const std::string &dir_name, const IntegralParser &integrals,
                   const SCF &scf)
    : integrals(integrals), scf(scf) {
  parse_geometry(dir_name);
}

void Analyser::parse_geometry(const std::string &dir_name) {
  std::string fname = dir_name + "/geom.dat";
  std::ifstream file(fname);
  if (!file.is_open()) {
    fmt::print("File {} could not be opened", fname);
    exit(1);
  }

  int n_atoms;
  file >> n_atoms;

  double charge;
  double x;
  double y;
  double z;
  for (int i = 0; i < n_atoms; ++i) {
    file >> charge >> x >> y >> z;
    charges.push_back(charge);
    atoms.push_back(Eigen::Vector3d(x, y, z));
  }
}

void Analyser::print_fock_mo() const {
  Eigen::MatrixXd Fock_mo = scf.Coeffs.transpose() * scf.Fock * scf.Coeffs;
  std::cout << "Fock in MO basis\n" << Fock_mo << "\n\n";
}

// Center-of-mass
Eigen::Vector3d Analyser::get_com() const {

  double tot_mass = 0.0;
  for (auto charge : charges) {
    tot_mass += masses[charge];
  }

  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < charges.size(); ++i) {
    com += masses[charges[i]] * atoms[i];
  }
  return com / tot_mass;
}

// Translate atomic coordinates
void Analyser::translate(const Eigen::Vector3d &vec) {
  for (auto &atom : atoms) {
    atom += vec;
  }
}

Eigen::Vector3d Analyser::compute_dipole_nuc() const {
  Eigen::Vector3d dipole = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < charges.size(); ++i) {
    dipole += charges[i] * atoms[i];
  }
  return dipole;
}

Eigen::Vector3d Analyser::compute_dipole_elec() const {
  Eigen::Vector3d dipole = Eigen::Vector3d::Zero();

  double mux = (scf.Density.transpose() * integrals.get_mu().x).trace();
  double muy = (scf.Density.transpose() * integrals.get_mu().y).trace();
  double muz = (scf.Density.transpose() * integrals.get_mu().z).trace();

  dipole << mux, muy, muz;

  return dipole;
}

void Analyser::analyze_dipole() const {
  Eigen::Vector3d dipole_elec = compute_dipole_elec();
  Eigen::Vector3d dipole_nuc = compute_dipole_nuc();
  Eigen::Vector3d dipole_tot = dipole_nuc + 2 * dipole_elec;

  std::cout << "Dipole\n" << dipole_tot << "\n\n";
}

void Analyser::analyze() {
  fmt::println("\n=== Analyzing ===\n\n");
  print_fock_mo();

  translate(-get_com());
  analyze_dipole();
}
