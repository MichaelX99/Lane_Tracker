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

float* Populate_Transform(std::string fpath)
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

  float* float_output = Columate_Matrix(output);

  return float_output;
}

std::vector<float> Populate_States(std::string fpath)
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
