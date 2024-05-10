#ifndef INDEXING_H
#define INDEXING_H

constexpr inline int get_2index(int i, int j) {
  return i > j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
}

constexpr inline int get_4index(int mu, int nu, int lambda, int sigma) {
  int mu_nu = get_2index(mu, nu);
  int lambda_sigma = get_2index(lambda, sigma);

  return get_2index(mu_nu, lambda_sigma);
}

#endif // INDEXING_H
