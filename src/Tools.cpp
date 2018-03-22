#include <boost/lexical_cast.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <boost/foreach.hpp>

#include <iostream>
#include <fstream>

#include "Tools.h"

bool is_Float(std::string someString)
{
/*
  Input: A string
  Output: A boolean
  Function: Check if the string is a float value or not

  https://stackoverflow.com/questions/447206/c-isfloat-function
*/
  using boost::lexical_cast;
  using boost::bad_lexical_cast;

  try
  {
    boost::lexical_cast<float>(someString);
  }
  catch (bad_lexical_cast &)
  {
    return false;
  }

  return true;
}

Eigen::MatrixXd Populate_Transform(std::string fpath, std::string to_get)
{
/*
  Input: The path to the input text file, the transform within the text file to retrieve
  Output: An Eigen Matrix object of the transform
  Function: Populate an Eigen Matrix object with a transform from the provided text file
*/
  std::ifstream file_obj;
  std::string line, transform;

  std::vector<float> float_transform;

  file_obj.open(fpath);
  if (file_obj.is_open())
  {
    while (!file_obj.eof())
    {
      getline(file_obj, line);
      //if (line.find("Tr_velo_to_cam: ") != std::string::npos)
      if (line.find(to_get) != std::string::npos)
      {
        transform = line;
      }
    }
  }

  // https://stackoverflow.com/questions/5888022/split-string-by-single-spaces
  typedef std::vector<std::string> Tokens;
  Tokens tokens;
  boost::split( tokens, transform, boost::is_any_of(" ") );

  BOOST_FOREACH( const std::string& i, tokens )
  {
    if (is_Float(i))
    {
      float_transform.push_back(std::stof(i));
    }
  }

  if (to_get == "Tr_velo_to_cam: ")
  {
    Eigen::MatrixXd output(4,4);

    output(0,0) = float_transform[0];
    output(0,1) = float_transform[1];
    output(0,2) = float_transform[2];
    output(0,3) = float_transform[3];

    output(1,0) = float_transform[4];
    output(1,1) = float_transform[5];
    output(1,2) = float_transform[6];
    output(1,3) = float_transform[7];

    output(2,0) = float_transform[8];
    output(2,1) = float_transform[9];
    output(2,2) = float_transform[10];
    output(2,3) = float_transform[11];

    output(3,0) = 0;
    output(3,1) = 0;
    output(3,2) = 0;
    output(3,3) = 1;

    return output;
  }
  else if (to_get == "R0_rect: ")
  {
    Eigen::MatrixXd output(4,4);

    output(0,0) = float_transform[0];
    output(0,1) = float_transform[1];
    output(0,2) = float_transform[2];
    output(0,3) = 0;

    output(1,0) = float_transform[3];
    output(1,1) = float_transform[4];
    output(1,2) = float_transform[5];
    output(1,3) = 0;

    output(2,0) = float_transform[6];
    output(2,1) = float_transform[7];
    output(2,2) = float_transform[8];
    output(2,3) = 0;

    output(3,0) = 0;
    output(3,1) = 0;
    output(3,2) = 0;
    output(3,3) = 1;

    return output;
  }
  else
  {
    Eigen::MatrixXd output(3,4);

    output(0,0) = float_transform[0];
    output(0,1) = float_transform[1];
    output(0,2) = float_transform[2];
    output(0,3) = float_transform[3];

    output(1,0) = float_transform[4];
    output(1,1) = float_transform[5];
    output(1,2) = float_transform[6];
    output(1,3) = float_transform[7];

    output(2,0) = float_transform[8];
    output(2,1) = float_transform[9];
    output(2,2) = float_transform[10];
    output(2,3) = float_transform[11];

    return output;
  }
}

std::vector<std::vector<float> > Populate_Weights(std::string fpath)
{
/*
  Input: Filepath to the text file
  Output: Matrix in 2D vector form
  Function: Populate a 2D vector with floats
*/
  std::vector<std::vector<float> > output, transpose_output;

  std::ifstream file_obj;
  std::string line;

  std::vector<std::string> string_file;

  file_obj.open(fpath);
  if (file_obj.is_open())
  {
    while (!file_obj.eof())
    {
      getline(file_obj, line);
      string_file.push_back(line);
    }
  }

  // https://stackoverflow.com/questions/5888022/split-string-by-single-spaces
  std::vector<float> temp;
  for (size_t i = 0; i < string_file.size(); i++)
  {
    typedef std::vector<std::string> Tokens;
    Tokens tokens;
    boost::split( tokens, string_file[i], boost::is_any_of(" ") );

    BOOST_FOREACH( const std::string& j, tokens )
    {
      if (is_Float(j))
      {
        temp.push_back(boost::lexical_cast<float>(j));
      }
    }
    if (temp.size() != 0)
    {
      transpose_output.push_back(temp);
      temp.clear();
    }
  }

  temp.clear();
  for (int i = 0; i < transpose_output[0].size(); i++)
  {
    for (int j = 0; j < transpose_output.size(); j++)
    {
      temp.push_back(transpose_output[j][i]);
    }
    output.push_back(temp);
    temp.clear();
  }

  return output;
}

std::vector<float> Populate_Biases(std::string fpath)
{
/*
  Input: Filepath to the text file
  Output: Vector of values
  Function: Populate a vector with floats
*/
  std::vector<float> output;

  std::ifstream file_obj;
  std::string line;

  std::vector<std::string> string_file;

  file_obj.open(fpath);
  if (file_obj.is_open())
  {
    while (!file_obj.eof())
    {
      getline(file_obj, line);
      if (is_Float(line))
      {
        output.push_back(boost::lexical_cast<float>(line));
      }
    }
  }

  return output;
}



float* Columate_Matrix(std::vector<std::vector<float> > input)
{
/*
  Input: A 2D weight matrix
  Output: A pointer to an array
  Function: Populate an array in the mandatory column ordered form for CUBLAS
*/
  const int m = input.size();
  const int n = input[0].size();

  float *output = (float*)malloc(m * n * sizeof(float));

  int ind;
  for (int j = 0; j < n; j++)
  {
    for (int i = 0; i < m; i++)
    {
      ind = i + (m * j);
      output[ind] = input[i][j];
    }
  }

  return output;
}

float* Vectorize(std::vector<float> input)
{
/*
  Input: A bias vector
  Output: A pointer to an array
  Function: Populate an arry for CUBLAS
*/
  const int m = input.size();

  float *output = (float*)malloc(m * sizeof(float));

  for (int i = 0; i < m; i++)
  {
    output[i] = input[i];
  }

  return output;
}
