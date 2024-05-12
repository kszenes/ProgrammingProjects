#include "CC.h"
#include "fmt/core.h"
#include <cmath>
#include <iostream>

Eigen::VectorXd CC::get_eri_mo() const {
  Eigen::VectorXd eri_mo = Eigen::VectorXd::Zero(ints_.get_eri().size());
  const Eigen::MatrixXd &coeffs = scf_.Coeffs;

#define INDEX(mu, nu, lambda, sigma)                                           \
  mu + nu *scf_.n_ao + lambda *scf_.n_ao *scf_.n_ao +                          \
      sigma *scf_.n_ao *scf_.n_ao *scf_.n_ao

  std::vector<double> tmp_s(std::pow(scf_.n_ao, 4), 0);
  std::vector<double> tmp_r(std::pow(scf_.n_ao, 4), 0);
  std::vector<double> tmp_q(std::pow(scf_.n_ao, 4), 0);
  double val = 0.0;

  // (mu, nu, lambda, sigma) -> (mu, nu, lambda, s)
  for (int p = 0; p < scf_.n_ao; ++p) {
    for (int q = 0; q < scf_.n_ao; ++q) {
      for (int r = 0; r < scf_.n_ao; ++r) {
        for (int s = 0; s < scf_.n_ao; ++s) {
          val = 0.0;
          for (int sigma = 0; sigma < scf_.n_ao; ++sigma) {
            val +=
                coeffs(sigma, s) * ints_.get_eri()(get_4index(p, q, r, sigma));
          }
          tmp_s[INDEX(p, q, r, s)] = val;
        }
      }
    }
  }

  // (mu, nu, lambda, s) -> (mu, nu, r, s)
  for (int p = 0; p < scf_.n_ao; ++p) {
    for (int q = 0; q < scf_.n_ao; ++q) {
      for (int r = 0; r < scf_.n_ao; ++r) {
        for (int s = 0; s < scf_.n_ao; ++s) {
          val = 0.0;
          for (int lambda = 0; lambda < scf_.n_ao; ++lambda) {
            val += coeffs(lambda, r) * tmp_s[INDEX(p, q, lambda, s)];
          }
          tmp_r[INDEX(p, q, r, s)] = val;
        }
      }
    }
  }

  // (mu, nu, r, s) -> (mu, q, r, s)
  for (int p = 0; p < scf_.n_ao; ++p) {
    for (int q = 0; q < scf_.n_ao; ++q) {
      for (int r = 0; r < scf_.n_ao; ++r) {
        for (int s = 0; s < scf_.n_ao; ++s) {
          val = 0.0;
          for (int nu = 0; nu < scf_.n_ao; ++nu) {
            val += coeffs(nu, q) * tmp_r[INDEX(p, nu, r, s)];
          }
          tmp_q[INDEX(p, q, r, s)] = val;
        }
      }
    }
  }

  // (mu, q, r, s) -> (p, q, r, s)
  for (int p = 0; p < scf_.n_ao; ++p) {
    for (int q = 0; q < scf_.n_ao; ++q) {
      for (int r = 0; r < scf_.n_ao; ++r) {
        for (int s = 0; s < scf_.n_ao; ++s) {
          val = 0.0;
          for (int mu = 0; mu < scf_.n_ao; ++mu) {
            val += coeffs(mu, p) * tmp_q[INDEX(mu, q, r, s)];
          }
          eri_mo(get_4index(p, q, r, s)) = val;
        }
      }
    }
  }

  return eri_mo;
}

Eigen::MatrixXd CC::spatial2spin_1b(const Eigen::MatrixXd &mat_in) const {
  Eigen::MatrixXd mat_out(n_ao, n_ao);

  for (int p = 0; p < scf_.n_ao; ++p) {
    for (int q = 0; q < scf_.n_ao; ++q) {
      mat_out(2 * p, 2 * q) = mat_in(p, q);
      mat_out(2 * p + 1, 2 * q + 1) = mat_in(p, q);
    }
  }
  return mat_out;
}

Eigen::Tensor<double, 4>
CC::spatial2spin_2b(const Eigen::VectorXd &tensor_in) const {
  // Spin basis
  Eigen::Tensor<double, 4> tensor_out(n_ao, n_ao, n_ao, n_ao);
  for (int p = 0; p < scf_.n_ao; ++p) {
    for (int q = 0; q < scf_.n_ao; ++q) {
      for (int r = 0; r < scf_.n_ao; ++r) {
        for (int s = 0; s < scf_.n_ao; ++s) {
          // Mulliken (p, r, q, s) -> <p, q, r, s> Physics

          // <up, up, up, up>
          tensor_out(2 * p, 2 * q, 2 * r, 2 * s) =
              tensor_in(get_4index(p, r, q, s));
          // <down, up, down, up>
          tensor_out(2 * p + 1, 2 * q, 2 * r + 1, 2 * s) =
              tensor_in(get_4index(p, r, q, s));
          // <up, down, up, down>
          tensor_out(2 * p, 2 * q + 1, 2 * r, 2 * s + 1) =
              tensor_in(get_4index(p, r, q, s));
          // <down, down, down, down>
          tensor_out(2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1) =
              tensor_in(get_4index(p, r, q, s));
        }
      }
    }
  }
  return tensor_out;
}

Eigen::Tensor<double, 4>
CC::antisymmetrise(const Eigen::Tensor<double, 4> &tensor_in) const {
  Eigen::Tensor<double, 4> tensor_out(n_ao, n_ao, n_ao, n_ao);
  for (int p = 0; p < n_ao; ++p) {
    for (int q = 0; q < n_ao; ++q) {
      for (int r = 0; r < n_ao; ++r) {
        for (int s = 0; s < n_ao; ++s) {
          tensor_out(p, q, r, s) =
              tensor_in(p, q, r, s) - tensor_in(p, q, s, r);
        }
      }
    }
  }
  return tensor_out;
}

Eigen::MatrixXd CC::get_fock_spin() const {
  Eigen::MatrixXd fock_spin = Eigen::MatrixXd::Zero(n_ao, n_ao);
  Eigen::VectorXd mo_e_space = scf_.get_mo_energies();

  for (int p = 0; p < mo_e_space.size(); ++p) {
    fock_spin(2 * p, 2 * p) = mo_e_space(p);
    fock_spin(2 * p + 1, 2 * p + 1) = mo_e_space(p);
  }

  return fock_spin;
}

void CC::init_amplitudes() {
  Eigen::VectorXd mo_e_space = scf_.get_mo_energies();
  Eigen::VectorXd mo_e_spin = Eigen::VectorXd::Zero(n_ao);
  for (int i = 0; i < n_ao / 2; ++i) {
    mo_e_spin(2 * i) = mo_e_space(i);
    mo_e_spin(2 * i + 1) = mo_e_space(i);
  }

  t1 = Eigen::MatrixXd::Zero(n_occ, n_virtual);
  t2 = Eigen::Tensor<double, 4>(n_occ, n_occ, n_virtual, n_virtual);

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          double numer = eri_anti(i, j, a + n_occ, b + n_occ);
          double denom = mo_e_spin(i) + mo_e_spin(j) - mo_e_spin(a + n_occ) -
                         mo_e_spin(b + n_occ);
          t2(i, j, a, b) = numer / denom;
        }
      }
    }
  }
}

double CC::compute_mp2_energy() const {
  double mp2_energy = 0.0;
  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          mp2_energy += (eri_anti(i, j, a + n_occ, b + n_occ)) * t2(i, j, a, b);
        }
      }
    }
  }
  mp2_energy = mp2_energy / 4.0;
  return mp2_energy;
}

void CC::build_taus() {
  tau.resize(n_occ, n_occ, n_virtual, n_virtual);
  tau_tilde.resize(n_occ, n_occ, n_virtual, n_virtual);

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          tau_tilde(i, j, a, b) = t2(i, j, a, b) + 0.5 * (t1(i, a) * t1(j, b) -
                                                          t1(i, b) * t1(j, a));
          tau(i, j, a, b) =
              t2(i, j, a, b) + t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a);
        }
      }
    }
  }
}

void CC::build_intermediates() {
  build_F();
  build_W();
}

void CC::build_F() {
  F = Eigen::MatrixXd::Zero(n_ao, n_ao);
  double first = 0.0;
  double second = 0.0;
  double third = 0.0;
  double fourth = 0.0;

  // Occupied - Occupied
  for (int m = 0; m < n_occ; ++m) {
    for (int i = 0; i < n_occ; ++i) {
      first = 0.0;
      second = 0.0;
      third = 0.0;
      fourth = 0.0;

      first = (m == i) ? 0.0 : fock(m, i);

      for (int e = 0; e < n_virtual; ++e) {
        second += t1(i, e) * fock(m, n_occ + e);
      }

      for (int e = 0; e < n_virtual; ++e) {
        for (int n = 0; n < n_occ; ++n) {
          third += t1(n, e) * eri_anti(m, n, i, e + n_occ);
        }
      }

      for (int e = 0; e < n_virtual; ++e) {
        for (int f = 0; f < n_virtual; ++f) {
          for (int n = 0; n < n_occ; ++n) {
            fourth +=
                tau_tilde(i, n, e, f) * eri_anti(m, n, e + n_occ, f + n_occ);
          }
        }
      }

      F(m, i) = first + 0.5 * second + third + 0.5 * fourth;
    }
  }

  // Virtual - Virtual
  for (int a = 0; a < n_virtual; ++a) {
    for (int e = 0; e < n_virtual; ++e) {
      first = 0.0;
      second = 0.0;
      third = 0.0;
      fourth = 0.0;

      first = (a == e) ? 0.0 : fock(a + n_occ, e + n_occ);

      for (int m = 0; m < n_occ; ++m) {
        second += t1(m, a) * fock(m, e + n_occ);
      }

      for (int m = 0; m < n_occ; ++m) {
        for (int f = 0; f < n_virtual; ++f) {
          third += t1(m, f) * eri_anti(m, a + n_occ, f + n_occ, e + n_occ);
        }
      }

      for (int m = 0; m < n_occ; ++m) {
        for (int n = 0; n < n_occ; ++n) {
          for (int f = 0; f < n_virtual; ++f) {
            fourth +=
                tau_tilde(m, n, a, f) * eri_anti(m, n, e + n_occ, f + n_occ);
          }
        }
      }
      F(a + n_occ, e + n_occ) = first - 0.5 * second + third - 0.5 * fourth;
    }
  }

  // Occupied - Virtual | Virtual - Occupied
  for (int m = 0; m < n_occ; ++m) {
    for (int e = 0; e < n_virtual; ++e) {
      first = 0.0;
      second = 0.0;

      first = fock(m, e + n_occ);

      for (int n = 0; n < n_occ; ++n) {
        for (int f = 0; f < n_virtual; ++f) {
          second += t1(n, f) * eri_anti(m, n, e + n_occ, f + n_occ);
        }
      }

      F(m, e + n_occ) = first + second;
      F(e + n_occ, m) = first + second;
    }
  }
}

void CC::build_W() {
  W.resize(n_ao, n_ao, n_ao, n_ao);
  W.setZero();
  double first = 0.0;
  double second = 0.0;
  double third = 0.0;
  double fourth = 0.0;

  // Occupied - Occupied - Occupied - Occupied
  for (int m = 0; m < n_occ; ++m) {
    for (int n = 0; n < n_occ; ++n) {
      for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
          first = 0.0;
          second = 0.0;
          third = 0.0;
          fourth = 0.0;

          first = eri_anti(m, n, i, j);

          for (int e = 0; e < n_virtual; ++e) {
            second += t1(j, e) * eri_anti(m, n, i, e + n_occ);
            second -= t1(i, e) * eri_anti(m, n, j, e + n_occ);
          }

          for (int e = 0; e < n_virtual; ++e) {
            for (int f = 0; f < n_virtual; ++f) {
              third += tau(i, j, e, f) * eri_anti(m, n, e + n_occ, f + n_occ);
            }
          }

          W(m, n, i, j) = first + second + 0.25 * third;
        }
      }
    }
  }

  // Virtual - Virtual - Virtual - Virtual
  for (int a = 0; a < n_virtual; ++a) {
    for (int b = 0; b < n_virtual; ++b) {
      for (int e = 0; e < n_virtual; ++e) {
        for (int f = 0; f < n_virtual; ++f) {
          first = 0.0;
          second = 0.0;
          third = 0.0;
          fourth = 0.0;

          first = eri_anti(a + n_occ, b + n_occ, e + n_occ, f + n_occ);

          for (int m = 0; m < n_occ; ++m) {
            second += t1(m, b) * eri_anti(a + n_occ, m, e + n_occ, f + n_occ);
            second -= t1(m, a) * eri_anti(b + n_occ, m, e + n_occ, f + n_occ);
          }

          for (int m = 0; m < n_occ; ++m) {
            for (int n = 0; n < n_occ; ++n) {
              third += tau(m, n, a, b) * eri_anti(m, n, e + n_occ, f + n_occ);
            }
          }

          W(a + n_occ, b + n_occ, e + n_occ, f + n_occ) =
              first - second + 0.25 * third;
        }
      }
    }
  }

  // Occupied - Virtual - Virtual - Occupied
  for (int m = 0; m < n_occ; ++m) {
    for (int b = 0; b < n_virtual; ++b) {
      for (int e = 0; e < n_virtual; ++e) {
        for (int j = 0; j < n_occ; ++j) {
          first = 0.0;
          second = 0.0;
          third = 0.0;
          fourth = 0.0;

          first = eri_anti(m, b + n_occ, e + n_occ, j);

          for (int f = 0; f < n_virtual; ++f) {
            second += t1(j, f) * eri_anti(m, b + n_occ, e + n_occ, f + n_occ);
          }

          for (int n = 0; n < n_occ; ++n) {
            third += t1(n, b) * eri_anti(m, n, e + n_occ, j);
          }

          for (int n = 0; n < n_occ; ++n) {
            for (int f = 0; f < n_virtual; ++f) {
              fourth += (0.5 * t2(j, n, f, b) + t1(j, f) * t1(n, b)) *
                        eri_anti(m, n, e + n_occ, f + n_occ);
            }
          }

          W(m, b + n_occ, e + n_occ, j) = first + second - third - fourth;
        }
      }
    }
  }
}

void CC::build_denominators() {
  build_D1();
  build_D2();
}

void CC::build_D1() {
  D1 = Eigen::MatrixXd::Zero(n_occ, n_virtual);

  for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virtual; ++a) {
      D1(i, a) = fock(i, i) - fock(a + n_occ, a + n_occ);
    }
  }
}

void CC::build_D2() {
  D2.resize(n_occ, n_occ, n_virtual, n_virtual);

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          D2(i, j, a, b) = fock(i, i) + fock(j, j) -
                           fock(a + n_occ, a + n_occ) -
                           fock(b + n_occ, b + n_occ);
        }
      }
    }
  }
}

void CC::update_amplitudes() {
  Eigen::MatrixXd t1_new = compute_t1();
  Eigen::Tensor<double, 4> t2_new = compute_t2();
  t1 = t1_new;
  t2 = t2_new;
}

Eigen::MatrixXd CC::compute_t1() const {
  Eigen::MatrixXd t1_new = Eigen::MatrixXd::Zero(n_occ, n_virtual);
  double first = 0.0;
  double second = 0.0;
  double third = 0.0;
  double fourth = 0.0;
  double fifth = 0.0;
  double sixth = 0.0;
  double seventh = 0.0;
  for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virtual; ++a) {
      first = 0.0;
      second = 0.0;
      third = 0.0;
      fourth = 0.0;
      fifth = 0.0;
      sixth = 0.0;
      seventh = 0.0;

      first = fock(i, a + n_occ);

      for (int e = 0; e < n_virtual; ++e) {
        second += t1(i, e) * F(a + n_occ, e + n_occ);
      }

      for (int m = 0; m < n_occ; ++m) {
        third += t1(m, a) * F(m, i);
      }

      for (int m = 0; m < n_occ; ++m) {
        for (int e = 0; e < n_virtual; ++e) {
          fourth += t2(i, m, a, e) * F(m, e + n_occ);
        }
      }

      for (int n = 0; n < n_occ; ++n) {
        for (int f = 0; f < n_virtual; ++f) {
          fifth += t1(n, f) * eri_anti(n, a + n_occ, i, f + n_occ);
        }
      }

      for (int m = 0; m < n_occ; ++m) {
        for (int e = 0; e < n_virtual; ++e) {
          for (int f = 0; f < n_virtual; ++f) {
            sixth +=
                t2(i, m, e, f) * eri_anti(m, a + n_occ, e + n_occ, f + n_occ);
          }
        }
      }

      for (int m = 0; m < n_occ; ++m) {
        for (int e = 0; e < n_virtual; ++e) {
          for (int n = 0; n < n_occ; ++n) {
            seventh += t2(m, n, a, e) * eri_anti(n, m, e + n_occ, i);
          }
        }
      }

      double numer =
          first + second - third + fourth - fifth - 0.5 * sixth - 0.5 * seventh;
      t1_new(i, a) = numer / D1(i, a);
    }
  }
  return t1_new;
}

Eigen::Tensor<double, 4> CC::compute_t2() const {
  Eigen::Tensor<double, 4> t2_new(n_occ, n_occ, n_virtual, n_virtual);
  double first = 0.0;
  double second = 0.0;
  double third = 0.0;
  double fourth = 0.0;
  double fifth = 0.0;
  double sixth = 0.0;
  double seventh = 0.0;
  double eighth = 0.0;

  double inner = 0.0;

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          first = 0.0;
          second = 0.0;
          third = 0.0;
          fourth = 0.0;
          fifth = 0.0;
          sixth = 0.0;
          seventh = 0.0;
          eighth = 0.0;

          // first
          first = eri_anti(i, j, a + n_occ, b + n_occ);

          // second
          for (int e = 0; e < n_virtual; ++e) {
            // original
            inner = 0.0;
            for (int m = 0; m < n_occ; ++m) {
              inner += t1(m, b) * F(m, e + n_occ);
            }
            second += t2(i, j, a, e) * (F(b + n_occ, e + n_occ) - 0.5 * inner);

            // permuted
            inner = 0.0;
            for (int m = 0; m < n_occ; ++m) {
              inner += t1(m, a) * F(m, e + n_occ);
            }
            second -= t2(i, j, b, e) * (F(a + n_occ, e + n_occ) - 0.5 * inner);
          }

          // third
          for (int m = 0; m < n_occ; ++m) {
            // original
            inner = 0.0;
            for (int e = 0; e < n_virtual; ++e) {
              inner += t1(j, e) * F(m, e + n_occ);
            }
            third += t2(i, m, a, b) * (F(m, j) + 0.5 * inner);

            // permuted
            inner = 0.0;
            for (int e = 0; e < n_virtual; ++e) {
              inner += t1(i, e) * F(m, e + n_occ);
            }
            third -= t2(j, m, a, b) * (F(m, i) + 0.5 * inner);
          }

          // fourth
          for (int m = 0; m < n_occ; ++m) {
            for (int n = 0; n < n_occ; ++n) {
              fourth += tau(m, n, a, b) * W(m, n, i, j);
            }
          }

          // fifth
          for (int e = 0; e < n_virtual; ++e) {
            for (int f = 0; f < n_virtual; ++f) {
              fifth += tau(i, j, e, f) *
                       W(a + n_occ, b + n_occ, e + n_occ, f + n_occ);
            }
          }

          // sixth
          for (int m = 0; m < n_occ; ++m) {
            for (int e = 0; e < n_virtual; ++e) {
              // (ij,ab)
              sixth +=
                  t2(i, m, a, e) * W(m, b + n_occ, e + n_occ, j) -
                  t1(i, e) * t1(m, a) * eri_anti(m, b + n_occ, e + n_occ, j);
              // (ij,ba)
              sixth -=
                  t2(i, m, b, e) * W(m, a + n_occ, e + n_occ, j) -
                  t1(i, e) * t1(m, b) * eri_anti(m, a + n_occ, e + n_occ, j);
              // (ji,ab)
              sixth -=
                  t2(j, m, a, e) * W(m, b + n_occ, e + n_occ, i) -
                  t1(j, e) * t1(m, a) * eri_anti(m, b + n_occ, e + n_occ, i);
              // (ji,ba)
              sixth +=
                  t2(j, m, b, e) * W(m, a + n_occ, e + n_occ, i) -
                  t1(j, e) * t1(m, b) * eri_anti(m, a + n_occ, e + n_occ, i);
            }
          }

          // seventh
          for (int e = 0; e < n_virtual; ++e) {
            seventh += t1(i, e) * eri_anti(a + n_occ, b + n_occ, e + n_occ, j);
            seventh -= t1(j, e) * eri_anti(a + n_occ, b + n_occ, e + n_occ, i);
          }

          // eighth
          for (int m = 0; m < n_occ; ++m) {
            eighth += t1(m, a) * eri_anti(m, b + n_occ, i, j);
            eighth -= t1(m, b) * eri_anti(m, a + n_occ, i, j);
          }

          double numer = first + second - third + 0.5 * fourth + 0.5 * fifth +
                         sixth + seventh - eighth;
          t2_new(i, j, a, b) = numer / D2(i, j, a, b);
        }
      }
    }
  }
  return t2_new;
}

double CC::get_cc_energy() {
  double energy = 0.0;

  double first = 0.0;
  double second = 0.0;
  double third = 0.0;

  for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virtual; ++a) {
      first += t1(i, a) * fock(i, a + n_occ);
    }
  }

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          second += t2(i, j, a, b) * eri_anti(i, j, a + n_occ, b + n_occ);
        }
      }
    }
  }

  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = 0; a < n_virtual; ++a) {
        for (int b = 0; b < n_virtual; ++b) {
          third += t1(i, a) * t1(j, b) * eri_anti(i, j, a + n_occ, b + n_occ);
        }
      }
    }
  }

  energy = first + 0.25 * second + 0.5 * third;
  return energy;
}

void CC::prepare() {
  Eigen::VectorXd eri_spatial = get_eri_mo();
  eri_anti = antisymmetrise(spatial2spin_2b(eri_spatial));
  fock = get_fock_spin();

  init_amplitudes();
  energy = compute_mp2_energy();

  build_denominators();
}

void CC::run() {
  fmt::println("\n\n === Starting Coupled Cluster ===\n");
  prepare();

  const int n_iter = 100;
  const double energy_tol = 1e-9;
  const double t1_tol = 1e-9;
  const double t2_tol = 1e-9;
  fmt::println("{:>4} {:>15} {:>15} {:>15} {:>15}", "iter", "E_CC", "delta_E",
               "rms(t1)", "rms(t2)");

  for (int iter = 0; iter < n_iter; ++iter) {
    build_taus();
    build_intermediates();

    Eigen::MatrixXd t1_new = compute_t1();
    Eigen::Tensor<double, 4> t2_new = compute_t2();

    double t1_rms = (t1 - t1_new).norm();
    // Eigen tensor does not automatically convert to double
    Eigen::Tensor<double, 0> t2_rms = ((t2 - t2_new).square().sum().sqrt());
    double t2_rms_val = t2_rms(0);

    t1 = t1_new;
    t2 = t2_new;

    double energy_new = get_cc_energy();
    double e_error = energy_new - energy;
    energy = energy_new;

    fmt::println("{:4} {: 15.9f} {: 15.9f} {: 15.9f} {: 15.9f}", iter,
                 energy_new, e_error, t1_rms, t2_rms_val);

    // check convergence
    if (abs(e_error) < energy_tol || t1_rms < t1_tol || t2_rms_val < t2_tol) {
      fmt::println("\n === CC Converged === \n");
      fmt::println("SCF E_tot: {:> 20.14f}", scf_.get_etot());
      fmt::println("CC Energy: {:> 20.9f}", energy);
      fmt::println("CC E_tot:  {:> 20.9f}", energy + scf_.get_etot());
      return;
    }
  }
  fmt::println("CC FAILED to converge");
}
