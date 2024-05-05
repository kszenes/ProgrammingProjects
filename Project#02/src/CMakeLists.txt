set(CMAKE_TOOLCHAIN_FILE $ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake)
cmake_minimum_required(VERSION 3.21)

project(fmt_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt CONFIG REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)

add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)


add_executable(main main.cpp Molecule.cpp Hessian.cpp)
target_link_libraries(main PRIVATE fmt::fmt Eigen3::Eigen)
target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic -Werror)
install(TARGETS main DESTINATION ${CMAKE_BINARY_DIR})
