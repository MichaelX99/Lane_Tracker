#ifndef __TOOLS__
#define __TOOLS__

#include <string>

#include <Eigen/Dense>

bool is_Float(std::string someString);

//Eigen::MatrixXd Populate_Transform(std::string fpath);
Eigen::MatrixXd Populate_Transform(std::string fpath, std::string to_get);

std::vector<std::vector<float> > Populate_Weights(std::string fpath);

std::vector<float> Populate_Biases(std::string fpath);

float* Columate_Matrix(std::vector<std::vector<float> > input);
float* Vectorize(std::vector<float> input);

#endif
