#include "CC.h"
#include "IntegralParser.h"
#include "SCF.h"
#include "SCFAnalyser.h"
#include "fmt/core.h"
#include "fmt/ranges.h"

void run_example(const std::string &dir_name) {
  IntegralParser integrals(dir_name);
  SCF scf(integrals);
  scf.run();
  Analyser analyser(dir_name, integrals, scf);
  analyser.analyze();

  CC cc(integrals, scf);
  cc.run();
}

int main() { run_example("../input/h2o/STO-3G"); }
