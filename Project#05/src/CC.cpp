#include "CC.h"
#include "fmt/core.h"
#include <iostream>

Eigen::VectorXd CC::get_eri_mo(const bool fast_algo) const {
  Eigen::VectorXd eri_mo = Eigen::VectorXd::Zero(ints_.get_eri().size());
  const Eigen::MatrixXd &coeffs = scf_.Coeffs;

  const int n_ao = scf_.n_ao;
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
                     ints_.get_eri()(get_4index(p, q, r, sigma));
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
                    val += coeffs(mu, p) * coeffs(nu, q) *
                           ints_.get_eri()(get_4index(mu, nu, lambda, sigma)) *
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

Eigen::MatrixXd CC::spatial2spin_1b(const Eigen::MatrixXd &mat_in) const {
  const int n_ao = scf_.n_ao;
  Eigen::MatrixXd mat_out(2 * n_ao, 2 * n_ao);

  for (int p = 0; p < n_ao; ++p) {
    for (int q = 0; q < n_ao; ++q) {
      mat_out(2 * p, 2 * q) = mat_in(p, q);
      mat_out(2 * p + 1, 2 * q + 1) = mat_in(p, q);
    }
  }
  return mat_out;
}

Eigen::Tensor<double, 4>
CC::spatial2spin_2b(const Eigen::VectorXd &tensor_in) const {
  const int n_ao = scf_.n_ao;
  // Spin basis
  Eigen::Tensor<double, 4> tensor_out(2 * n_ao, 2 * n_ao, 2 * n_ao, 2 * n_ao);
  for (int p = 0; p < n_ao; ++p) {
    for (int q = 0; q < n_ao; ++q) {
      for (int r = 0; r < n_ao; ++r) {
        for (int s = 0; s < n_ao; ++s) {
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

Eigen::MatrixXd CC::get_fock_spin() const {
  int n_ao = 2 * scf_.n_ao;
  int n_occ = 2 * scf_.n_occ;
  Eigen::MatrixXd h_pq = spatial2spin_1b(ints_.get_T() + ints_.get_V());
  Eigen::MatrixXd fock_spin = Eigen::MatrixXd::Zero(n_ao, n_ao);

  double val = 0.0;
  for (int p = 0; p < n_ao; ++p) {
    for (int q = 0; q < n_ao; ++q) {
      val = 0.0;
      for (int m = 0; m < n_occ; ++m) {
        val += eri(p, m, q, m) - eri(p, m, m, q);
      }
      fock_spin(p, q) = val;
    }
  }
  return fock_spin;
}

void CC::init_amplitudes() {
  int n_ao = 2 * scf_.n_ao;
  int n_occ = 2 * scf_.n_occ;
  int n_virtual = n_ao - n_occ;

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
      for (int a = n_occ; a < n_ao; ++a) {
        for (int b = n_occ; b < n_ao; ++b) {
          double numer = eri(i, j, a, b) - eri(i, j, b, a);
          double denom =
              mo_e_spin(i) + mo_e_spin(j) - mo_e_spin(a) - mo_e_spin(b);
          t2(i, j, a, b) = numer / denom;
        }
      }
    }
  }
}

void CC::compute_mp2_e() const {
  int n_ao = 2 * scf_.n_ao;
  int n_occ = 2 * scf_.n_occ;

  double mp2_e = 0.0;
  for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
      for (int a = n_occ; a < n_ao; ++a) {
        for (int b = n_occ; b < n_ao; ++b) {
          mp2_e += (eri(i, j, a, b) - eri(i, j, b, a)) * t2(i, j, a, b);
        }
      }
    }
  }
  mp2_e = mp2_e / 4.0;
  fmt::println("MP2 Energy: {}\n", mp2_e);
}

void CC::prepare() {
  Eigen::VectorXd eri_spatial = get_eri_mo();
  eri = spatial2spin_2b(eri_spatial);
  fock = get_fock_spin();
  init_amplitudes();

  compute_mp2_e();
}

void CC::run() {
  fmt::println("\n\n=== Starting Coupled Cluster ===\n");
  prepare();
}
