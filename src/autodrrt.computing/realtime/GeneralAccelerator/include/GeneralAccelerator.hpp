#ifndef GENERAL_ACCELERATOR_HPP_
#define GENERAL_ACCELERATOR_HPP_
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
const int MAX_READ_BUFFER_SIZE = 512;
const int TRTEXEC_NOT_FOUND = 0;
class ParseParameters
{
public:    
    ParseParameters();
    double GetCommdResult(const std::string &cmd);
    std::string parse_parameters(int argc, char *argv[],int iter);
    bool get_parameter(int argc, char *argv[], const std::string &parameter_name, std::string &parameter_value);
    void run_model_opt(const double & speed_no_dla, const double & speed_dla);
    std::string engine_path;  
private:
    bool if_has_path = false;
    bool if_has_shape = false;
    const int RESULT_NOTHING_FOUND_CODE = 0;
    const int RESULT_MODEL_PATH_ERROR_CODE = 1;
    const int RESULT_MODEL_SHAPE_ERROR_CODE = 2;
    const int RESULT_MODEL_SPTEED_CODE = 3;

    const std::string ENVIRONMENT_OPT_PARAMETER = "--env-opt=";  
    const std::string MODEL_PATH_PARAMETER = "--model-path=";   
    const std::string MODEL_SHAPE_PARAMETER = "--inputShapes=";   


    const std::string RESULT_MODEL_PATH_ERROR = "Could not open file";
    const std::string RESULT_MODEL_SHAPE_ERROR = "Dynamic dimensions required for input";
    const std::string RESULT_MODEL_SPTEED = "Throughput:";

    
    


    
    std::string model_path;    
    std::string model_shape;  
    double model_speed_with_dla;
    double model_speed = 0;  
private:
    int check_output(const std::string &readline);

};
#endif  // GENERAL_ACCELERATOR_HPP_
