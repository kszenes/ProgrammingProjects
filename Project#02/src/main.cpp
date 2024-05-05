#include "Molecule.h"
#include "Hessian.h"
#include "fmt/core.h"
#include "fmt/ranges.h"


void run_example(const std::string &root_name) {
  fmt::println("Running {}", root_name);
  std::string geom_fname = root_name + "_geom.txt";
  std::string hess_fname = root_name + "_hessian.txt";

  Molecule mol(geom_fname);

  Hessian hess(hess_fname);
  hess.check_sym();

  hess.mass_reweigth(mol);
  hess.check_sym();

  hess.get_freq();
  fmt::println("");
}

int main() {
  run_example("../input/h2o");
  run_example("../input/3c1b");
  run_example("../input/benzene");
}
