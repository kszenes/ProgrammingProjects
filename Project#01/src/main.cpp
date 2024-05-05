#include "Molecule.h"
#include "fmt/core.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

void printAllDists(const Molecule &mol) {
  fmt::println("Interactomic Distances [Bohr]:");
  int n_atoms = mol.size();
  for (int i = 1; i < n_atoms; ++i) {
    for (int j = 0; j < i; ++j) {
      auto dist = mol.get_dist(i, j);
      fmt::println("{:<2} {:<2}  {:.5f}", i, j, dist);
    }
  }
  fmt::println("");
}

void printAllAngles(const Molecule &mol, double dist_cutoff = 3.3) {
  fmt::println("Bond Angles:");
  int n_atoms = mol.size();
  for (int i = 0; i < n_atoms; ++i) {
    for (int center = 0; center < n_atoms; ++center) {
      for (int j = i + 1; j < n_atoms; ++j) {
        bool is_too_far = (mol.get_dist(i, center) > dist_cutoff) ||
                          (mol.get_dist(j, center) > dist_cutoff);
        if ((i == center || j == center) || is_too_far) {
          continue;
        }
        auto angle = mol.get_angle(i, center, j);
        fmt::println("{}- {}- {} {:3.6f}", i, center, j, angle);
      }
    }
  }
  fmt::println("");
}

void printAllOOPAngles(const Molecule &mol, double dist_cutoff = 4.0) {
  fmt::println("OOP Angles:");
  int n_atoms = mol.size();
  for (int j = 0; j < n_atoms; ++j) {
    for (int center = 0; center < n_atoms; ++center) {
      for (int l = j + 1; l < n_atoms; ++l) {
        for (int i = 0; i < n_atoms; ++i) {
          bool is_too_far = (mol.get_dist(i, center) > dist_cutoff) ||
                            (mol.get_dist(j, center) > dist_cutoff) ||
                            (mol.get_dist(l, center) > dist_cutoff);
          if ((i == j || i == l || i == center || j == center || l == center) ||
              is_too_far) {
            continue;
          }
          auto angle = mol.get_oop_angle(i, j, center, l);
          fmt::println("{}- {}- {}- {} {:3.6f}", i, j, center, l, angle);
        }
      }
    }
  }
  fmt::println("");
}

void printAllDihedrals(const Molecule &mol, double dist_cutoff = 4.0) {
  fmt::println("Dihedral Angles:");
  int n_atoms = mol.size();
  for (int i = 0; i < n_atoms; ++i) {
    for (int j = i + 1; j < n_atoms; ++j) {
      for (int k = j + 1; k < n_atoms; ++k) {
        for (int l = k + 1; l < n_atoms; ++l) {
          bool is_too_far = (mol.get_dist(i, j) > dist_cutoff) ||
                            (mol.get_dist(j, k) > dist_cutoff) ||
                            (mol.get_dist(k, l) > dist_cutoff);
          if (is_too_far) {
            continue;
          }
          auto angle = mol.get_dihedral(i, j, k, l);
          fmt::println("{}- {}- {}- {} {:3.6f}", i, j, k, l, angle);
        }
      }
    }
  }
  fmt::println("");
}

int main() {
  Molecule mol("../input/acetaldehyde.dat");
  mol.print();
  printAllDists(mol);
  printAllAngles(mol);
  printAllOOPAngles(mol);
  printAllDihedrals(mol);

  auto com = mol.get_com();
  fmt::println("Center of mass: {}", com);
  mol.translate(-com);

  auto moments = mol.prime_mom_intertia();
  fmt::println("Principal moments: {}\n", moments);
  mol.print_mol_rot_type();

  // TODO: Compute Rotational Constats
}
