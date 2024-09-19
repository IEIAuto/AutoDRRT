#include "GeneralAccelerator.hpp"
#include <assert.h>
int main(int argc, char *argv[])
{
    
    double speed_no_dla = 0;
    double speed_dla = 0;
    ParseParameters p;
    std::string last_cmd;
    std::string auto_select_parameter = p.parse_parameters(argc, argv, 1);
    // std::cout << auto_select_parameter;
    if(auto_select_parameter.find("ModelAccelerator") != std::string::npos)
    {
        std::string init_cmd = auto_select_parameter +   "2>&1";
        speed_no_dla = p.GetCommdResult(init_cmd);
        // std::cout << std::to_string(speed_no_dla);
        auto_select_parameter = p.parse_parameters(argc, argv, 2);
        std::string second_cmd =  auto_select_parameter + " --allowGPUFallback --useDLACore=0" + " 2>&1";
        speed_dla = p.GetCommdResult(second_cmd);     
        // std::cout << "start to run_model_opt, please wait ...";  
        p.run_model_opt(speed_no_dla,speed_dla);
    }else if (auto_select_parameter.find("jetson_clocks") != std::string::npos)
    {
        p.GetCommdResult(auto_select_parameter); 
        std::cout << "-- THE ENVIRONMENT HAS BEEN OPTIMIZED FOR MAXIMUM PERFORMANCE! --" << std::endl;
    }
    
    return 0;
}

