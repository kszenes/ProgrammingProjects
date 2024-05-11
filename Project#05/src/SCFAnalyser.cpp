#include "SCFAnalyser.h"
#include "Indexing.h"
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

void Analyser::print_mo_coeffs() const {
  std::cout << "MO Coeffs\n" << scf.Coeffs << "\n\n";
}

Eigen::VectorXd Analyser::get_mo_energies() const {
  Eigen::VectorXd mo_energies =
      (scf.Coeffs.transpose() * scf.Fock * scf.Coeffs).diagonal();

  std::cout << "MO energies\n" << mo_energies << "\n\n";
  return mo_energies;
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

Eigen::VectorXd Analyser::get_eri_mo(const bool fast_algo) const {
  Eigen::VectorXd eri_mo = Eigen::VectorXd::Zero(integrals.get_eri().size());
  const Eigen::MatrixXd &coeffs = scf.Coeffs;

  const int n_ao = scf.n_ao;
  double val;
  if (fast_algo) {
#define INDEX(mu, nu, lambda, sigma)                                           \
  mu + nu *n_ao + lambda *n_ao *n_ao + sigma *n_ao *n_ao *n_ao

    std::vector<double> tmp_s(std::pow(n_ao, 4), 0);
    std::vector<double> tmp_r(std::pow(n_ao, 4), 0);
    std::vector<double> tmp_q(std::pow(n_ao, 4), 0);
    double val = 0.0;

    // (mu, nu, lambda, sigma) -> (mu, nu, lambda, s)
    for (int p = 0; p < n_ao; ++p) {
      for (int q = 0; q < n_ao; ++q) {
        for (int r = 0; r < n_ao; ++r) {
          for (int s = 0; s < n_ao; ++s) {
            val = 0.0;
            for (int sigma = 0; sigma < n_ao; ++sigma) {
              val += coeffs(sigma, s) *
                     integrals.get_eri()(get_4index(p, q, r, sigma));
            }
            tmp_s[INDEX(p, q, r, s)] = val;
          }
        }
      }
    }

    // (mu, nu, lambda, s) -> (mu, nu, r, s)
    for (int p = 0; p < n_ao; ++p) {
      for (int q = 0; q < n_ao; ++q) {
        for (int r = 0; r < n_ao; ++r) {
          for (int s = 0; s < n_ao; ++s) {
            val = 0.0;
            for (int lambda = 0; lambda < n_ao; ++lambda) {
              val += coeffs(lambda, r) * tmp_s[INDEX(p, q, lambda, s)];
            }
            tmp_r[INDEX(p, q, r, s)] = val;
          }
        }
      }
    }

    // (mu, nu, r, s) -> (mu, q, r, s)
    for (int p = 0; p < n_ao; ++p) {
      for (int q = 0; q < n_ao; ++q) {
        for (int r = 0; r < n_ao; ++r) {
          for (int s = 0; s < n_ao; ++s) {
            val = 0.0;
            for (int nu = 0; nu < n_ao; ++nu) {
              val += coeffs(nu, q) * tmp_r[INDEX(p, nu, r, s)];
            }
            tmp_q[INDEX(p, q, r, s)] = val;
          }
        }
      }
    }

    // (mu, q, r, s) -> (p, q, r, s)
    for (int p = 0; p < n_ao; ++p) {
      for (int q = 0; q < n_ao; ++q) {
        for (int r = 0; r < n_ao; ++r) {
          for (int s = 0; s < n_ao; ++s) {
            val = 0.0;
            for (int mu = 0; mu < n_ao; ++mu) {
              val += coeffs(mu, p) * tmp_q[INDEX(mu, q, r, s)];
            }
            eri_mo(get_4index(p, q, r, s)) = val;
          }
        }
      }
    }

  } else {
    // MO indices
    for (int p = 0; p < n_ao; ++p) {
      for (int q = 0; q < n_ao; ++q) {
        for (int r = 0; r < n_ao; ++r) {
          for (int s = 0; s < n_ao; ++s) {
            // AO indices
            val = 0;
            for (int mu = 0; mu < n_ao; ++mu) {
              for (int nu = 0; nu < n_ao; ++nu) {
                for (int lambda = 0; lambda < n_ao; ++lambda) {
                  for (int sigma = 0; sigma < n_ao; ++sigma) {
                    val +=
                        coeffs(mu, p) * coeffs(nu, q) *
                        integrals.get_eri()(get_4index(mu, nu, lambda, sigma)) *
                        coeffs(lambda, r) * coeffs(sigma, s);
                  }
                }
              }
            }
            eri_mo(get_4index(p, q, r, s)) = val;
          }
        }
      }
    }
  }

  return eri_mo;
}

double Analyser::get_mp2_correction() const {
  const int n_ao = scf.n_ao;
  const int n_occ = scf.n_occ;
  double mp2_e = 0.0;

  Eigen::VectorXd mo_e = get_mo_energies();
  Eigen::VectorXd eri_mo = get_eri_mo();

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = n_occ; a < n_ao; ++a) {
        for (int b = n_occ; b < n_ao; ++b) {
          double numer = eri_mo(get_4index(i, a, j, b)) *
                         (2 * eri_mo(get_4index(i, a, j, b)) -
                          eri_mo(get_4index(i, b, j, a)));
          double denom = mo_e(i) + mo_e(j) - mo_e(a) - mo_e(b);
          mp2_e += numer / denom;
        }
      }
    }
  }
  return mp2_e;
}

void Analyser::analyze() {
  fmt::println("\n === Analyzing ===\n");
  print_mo_coeffs();
  double mp2_e = get_mp2_correction();
  fmt::println("SCF Energy     = {: 17.12f}", scf.get_etot());
  fmt::println("MP2 Correction = {: 17.12f}", mp2_e);
  fmt::println("MP2 Energy     = {: 17.12f}", scf.get_etot() + mp2_e);
}
