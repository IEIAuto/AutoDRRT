#include "GeneralAccelerator.hpp"

ParseParameters::ParseParameters()
{

}

double ParseParameters::GetCommdResult(const std::string &cmd)
{
    FILE *fp=nullptr;
    if((fp=popen(cmd.c_str(),"r"))==nullptr)
    {
        std::cout << "unknow error" << std::endl;
        exit(1);
    }
    char read_str[MAX_READ_BUFFER_SIZE]="";
    while(fgets(read_str,sizeof(read_str),fp))
    {
        std::string result(read_str);
        int readline_result = this->check_output(result);
        if(readline_result!=this->RESULT_NOTHING_FOUND_CODE&&readline_result!=this->RESULT_MODEL_SPTEED_CODE)
        {
            pclose(fp);
            // exit(1);
        }else if (readline_result==this->RESULT_MODEL_SPTEED_CODE)
        {
            pclose(fp);
            return this->model_speed;
        }
    }
    
    pclose(fp);
    return 0;
}

void replaceAll(std::string& str, const std::string& searchStr, const std::string& replaceStr) {
    std::size_t pos = 0;
    while ((pos = str.find(searchStr, pos)) != std::string::npos) {
        // std::cout << str << std::endl;
        str.replace(pos, searchStr.length(), replaceStr);
        pos += replaceStr.length();
    }
    // return str;
}

std::string filter_output(std::string readline)
{
    std::string searchs_str_1 = "TensorRT";
    std::string searchs_str_2 = "trtexec";
    std::string searchs_str_3 = "TRT";
    std::string replace_str_1 = "ModelAccelerator";
    std::string replace_str_2 = "ModelAccelerator";
    std::string replace_str_3 = "MA";

    replaceAll(readline,searchs_str_1,replace_str_1);
    replaceAll(readline,searchs_str_2,replace_str_2);
    replaceAll(readline,searchs_str_3,replace_str_3);
    return readline; 

    // std::size_t found_1 = readline.find(searchs_str_1);
    // if(found_1 != std::string::npos)
    // {
    //     std::cout << readline << "||" << std::to_string(searchs_str_1.length()) << std::endl;
    //     readline.replace(found_1,searchs_str_1.length(),replace_str_1);
    // }
    //     std::cout << readline << std::endl;
    // std::size_t found_2 = readline.find(searchs_str_2);
    // if(found_2 != std::string::npos)
    // {
    //     readline.replace(found_2,searchs_str_2.length(),replace_str_2);
    // }

    // return readline;
}

int ParseParameters::check_output(const std::string &readline)
{
    size_t index = readline.find(this->RESULT_MODEL_PATH_ERROR);
    if (index < MAX_READ_BUFFER_SIZE)//说明没找到
    {
        // std::cout << readline << "!!!!" << std::endl;
        return this->RESULT_MODEL_PATH_ERROR_CODE;
    }
    index = readline.find(this->RESULT_MODEL_SHAPE_ERROR);
    if (index > MAX_READ_BUFFER_SIZE)//说明没找到
    {
        // std::cout << readline << std::endl;
        // std::cout << "." ;
        // std::cout << readline<< std::endl;
        std::cout << filter_output(readline) << std::endl;
    }else{
        std::string::size_type pos = readline.find_first_of(".");
        std::cout << readline.substr(0,pos) << ". it should be configured. For example --inputShapes=input0:1x3x256x256,input1:1x3x128x128 " << std::endl;
        return this->RESULT_MODEL_SHAPE_ERROR_CODE;
    }
    index = readline.find(this->RESULT_MODEL_SPTEED);
    if (index < MAX_READ_BUFFER_SIZE)//说明找到了
    {
        // size_t end = readline.find("qps");
        // std::cout << "QOS+++" << index + this->RESULT_MODEL_SPTEED.length()+1 << std::endl;
        // std::cout << readline.substr(index + this->RESULT_MODEL_SPTEED.length()+1 ,7)  << std::endl;
        this->model_speed = std::stod(readline.substr(index + this->RESULT_MODEL_SPTEED.length()+1 ,7));
        return this->RESULT_MODEL_SPTEED_CODE;
    }
    return this->RESULT_NOTHING_FOUND_CODE;
}


std::string ParseParameters::parse_parameters(int argc, char *argv[],int iter)
{   
    std::string shape_param = "";
    std::string engine_path_name;
    if(argc < 2)
    {
        std::cout << " Too few input arguements, type GeneralAccelerator -h for help" << std::endl;
        exit(1);
    }
    std::string environment_opt;
    //开启环境优化选项
    if(this->get_parameter(argc,argv,this->ENVIRONMENT_OPT_PARAMETER,environment_opt))
    {
         //组装命令行参数
        return "sudo sysctl -w net.core.rmem_max=2147483647 && xhost + && sudo sysctl net.ipv4.ipfrag_time=3 && sudo sysctl net.ipv4.ipfrag_high_thresh=134217728 && \
        sudo ip link set lo multicast on && sudo jetson_clocks ";
    }

    //获取模型路径参数
    this->if_has_path = this->get_parameter(argc,argv,this->MODEL_PATH_PARAMETER,this->model_path);
    if(this->if_has_path)
    {
        size_t index = this->model_path.find(".onnx");
        this->engine_path = this->model_path.substr(0,index) + ".engine";
        engine_path_name = this->model_path.substr(0,index) + ".engine" + std::to_string(iter);
    }else{
         std::cout << "Model path mast be given ! for example: --model-path=absolutepath/modelname.onnx" << std::endl;
         exit(-1);     

    }
    //获取模型输入形状
    if(this->get_parameter(argc,argv,this->MODEL_SHAPE_PARAMETER,this->model_shape))
    {
        shape_param = "--optShapes="+this->model_shape;
    }
    //组装命令行参数
    return "./ModelAccelerator --onnx=" + this->model_path + " --saveEngine=" + engine_path_name + " --best " + shape_param + " --sparsity=enable ";
   
}

//获取命令行中是否有对应的参数值
bool ParseParameters::get_parameter(int argc, char *argv[], const std::string &parameter_name, std::string &parameter_value)
{
    bool parameter_flag = false;
    for(int i = 0; i < argc; i++)
    {
        // std::cout << "the argc is :" << argc << std::endl;
        std::string arg_string(argv[i]);
        {
           
            size_t index = arg_string.find(parameter_name);
            if (index > MAX_READ_BUFFER_SIZE)//说明没找到
            {
                // std::cout << "Model path mast be given like --model-path=absolutepath/modelname.onnx" << std::endl;                
            }else
            {
                parameter_value = arg_string.substr(index+parameter_name.length());
                parameter_flag = true;
            }
        }
    }
    return parameter_flag;
}


//执行模型最优化
void ParseParameters::run_model_opt(const double & speed_no_dla, const double & speed_dla)
{
    FILE *fp=nullptr;
    if(speed_no_dla > speed_dla)
    {
        double speed_max = speed_no_dla;
        std::string last_cmd = "rm " +  this->engine_path + std::to_string(2) + "&&" + "mv " +  this->engine_path + std::to_string(1) + " " + this->engine_path;
        
        if((fp=popen(last_cmd.c_str(),"r"))==nullptr)
        {
            std::cout << "unknow error" << std::endl;
            exit(1);
        }else{
            std::cout << "Successful ,the optimized model is " << this->engine_path << " .Performance is " << std::to_string(speed_max) <<" qps"<< std::endl;
            pclose(fp);
        }

    }else
    {
        double speed_max = speed_dla;
        std::string last_cmd = "rm " +  this->engine_path + std::to_string(1) + "&&" + "mv " +  this->engine_path + std::to_string(2) + " " + this->engine_path;
         
        if((fp=popen(last_cmd.c_str(),"r"))==nullptr)
        {
            std::cout << "unknow error" << std::endl;
            exit(1);
        }else{
            std::cout << "Successful ,the optimized model is " << this->engine_path << " .Performance is " << std::to_string(speed_max) <<" qps"<< std::endl;
            pclose(fp);
        }
    }
}
