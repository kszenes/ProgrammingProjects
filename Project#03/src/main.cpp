#include "fmt/core.h"
#include "fmt/ranges.h"
#include "IntegralParser.h"

void run_example(const std::string &dir_name) {
  IntegralParser integrals(dir_name);

}

int main() { run_example("../input/h2o/STO-3G"); }
