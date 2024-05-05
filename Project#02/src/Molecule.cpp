#include "Molecule.h"
#include "Eigen/Eigenvalues"
#include "fmt/core.h"
#include "fmt/ranges.h"
#include "masses.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

auto rad2deg = [](double angRad) { return 360 * angRad / (2 * M_PI); };
auto deg2rad = [](double angDeg) { return 2 * M_PI * angDeg / 360; };

Molecule::Molecule(const std::string &fname) {
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

void Molecule::print() const {
  fmt::println("Geometry");
  for (int i = 0; i < this->size(); ++i) {
    fmt::println("{:2} {: .6f} {: .6f} {: .6f}", charges[i], atoms[i][0],
                 atoms[i][1], atoms[i][2]);
  }
  fmt::println("");
}

int Molecule::size() const { return atoms.size(); }

double Molecule::get_dist(int i, int j) const {
  return (atoms[i] - atoms[j]).norm();
}

// NOTE: Angle returned in degrees
double Molecule::get_angle(int i, int center, int j) const {
  if (i == j) {
    fmt::println("Warning: Attempted computation of angle between the same "
                 "atoms ({}- {}- {})",
                 i, center, j);
    return 0.0;
  }
  Eigen::Vector3d e_i = (atoms[i] - atoms[center]) / get_dist(i, center);
  Eigen::Vector3d e_j = (atoms[j] - atoms[center]) / get_dist(j, center);
  double angRadian = acos(e_i.dot(e_j));

  return rad2deg(angRadian);
}

// Clip value of x to range [lower, upper]
double clip(double x, double lower, double upper) {
  return std::max(lower, std::min(x, upper));
}

// Out-of-plane angle of i from plane j - center - l
double Molecule::get_oop_angle(int i, int j, int center, int l) const {
  // Perfom checks
  bool are_same =
      (i == j || i == center || i == l || j == center || center == l);
  if (are_same) {
    fmt::println("Warning: Attempted computation of OOP angle between the same "
                 "atoms ({}- {}- {}- {})",
                 i, j, center, l);
    return 0.0;
  }
  double angle_jcl = deg2rad(get_angle(j, center, l));
  double tol = 1e-4;
  if (abs(asin(angle_jcl)) < tol) {
    fmt::println("Computing OOP angle failed. The 2 vectors defining the plane "
                 "are LINEAR");
    return 0.0;
  }

  // Compute
  Eigen::Vector3d e_j = (atoms[center] - atoms[j]) / get_dist(j, center);
  Eigen::Vector3d e_l = (atoms[center] - atoms[l]) / get_dist(l, center);
  Eigen::Vector3d e_i = (atoms[center] - atoms[i]) / get_dist(i, center);

  double theta = (e_j.cross(e_l)).dot(e_i) / sin(angle_jcl);

  double angRadian = asin(clip(theta, -1, 1));
  return rad2deg(angRadian);
}

double Molecule::get_dihedral(int i, int j, int k, int l) const {
  bool are_same = (i == j || i == k || i == l || j == k || k == l);
  if (are_same) {
    fmt::println(
        "Warning: Attempted computation of dihedral angle between the same "
        "atoms ({}- {}- {}- {})",
        i, j, k, l);
    return 0.0;
  }

  Eigen::Vector3d e_ij = (atoms[j] - atoms[i]) / get_dist(j, i);
  Eigen::Vector3d e_jk = (atoms[k] - atoms[j]) / get_dist(k, j);
  Eigen::Vector3d e_kl = (atoms[l] - atoms[k]) / get_dist(l, k);

  double sin_ijk = sin(deg2rad(get_angle(i, j, k)));
  double sin_jkl = sin(deg2rad(get_angle(j, k, l)));

  // TODO: Compute sign of dihedral
  // Sign of dihedral is neg if vector k-l to the left of plane i-j-k when
  // viewed along j-k

  double theta =
      ((e_ij).cross(e_jk)).dot((e_jk.cross(e_kl))) / (sin_ijk * sin_jkl);
  double angRadian = acos(clip(theta, -1, 1));
  return rad2deg(angRadian);
}

// Center-of-mass
Eigen::Vector3d Molecule::get_com() const {

  double tot_mass = 0.0;
  for (auto charge : charges) {
    tot_mass += masses[charge];
  }

  Eigen::Vector3d com = Eigen::Vector3d::Zero();
  for (int i = 0; i < this->size(); ++i) {
    com += masses[charges[i]] * atoms[i];
  }
  return com / tot_mass;
}

// Translate atomic coordinates
void Molecule::translate(const Eigen::Vector3d &vec) {
  for (auto &atom : atoms) {
    atom += vec;
  }
}

// auto printMat = [](const Eigen::MatrixXd &mat) {
//   for (int j = 0; j < mat.cols(); ++j) {
//     for (int i = 0; i < mat.rows(); ++i) {
//       fmt::print("{} ", mat(i, j));
//     }
//     fmt::print("\n");
//   }
// };

Eigen::Vector3d Molecule::prime_mom_intertia() const {
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();

  int n_atoms = this->size();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double val = 0.0;
      if (i == j) {
        int first_idx = (i + 1) % 3;
        int second_idx = (i + 2) % 3;
        for (int idx = 0; idx < n_atoms; ++idx) {
          Eigen::Vector3d vec_sq = atoms[idx].cwiseProduct(atoms[idx]);
          val +=
              masses[charges[idx]] * (vec_sq[first_idx] + vec_sq[second_idx]);
        }
      } else {
        for (int idx = 0; idx < n_atoms; ++idx) {
          val += masses[charges[idx]] * atoms[idx][i] * atoms[idx][j];
        }
      }
      inertia(i, j) = val;
    }
  }
  std::cout << inertia << '\n';

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(inertia);
  Eigen::Vector3d vals = es.eigenvalues();

  return vals;
}

void Molecule::print_mol_rot_type() const {
  auto moments = this->prime_mom_intertia();
  double a = moments[0];
  double b = moments[1];
  double c = moments[2];

  fmt::print("Molecule is a ");
  if (a == b == c) {
    fmt::println("SPHERICAL top");
  } else if (a < b && b == c) {
    fmt::println("LINEAR molecule");
  } else if (a == b && b == c) {
    fmt::println("SYMMETRIC top");
  } else {
    fmt::println("ASYMMETRIC top");
  }
}

double Molecule::get_mass(int i) const {
  return masses[charges[i]];
}
