#include "system_monitor/system_monitor.hpp"

SystemMonitor::SystemMonitor(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options),
network_card_id_(declare_parameter("network_card_id", "eth0"))
{
    sys_pub_ = this->create_publisher<std_msgs::msg::String>("system_stats", 10);
    diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics",10);
    timer_ = this->create_wall_timer(std::chrono::seconds(1),std::bind(&SystemMonitor::publsih_stats, this));
    RCLCPP_INFO(this->get_logger(),"System Monitor Node has started.");
}

SystemMonitor::~SystemMonitor(){}

void SystemMonitor::publsih_stats(){

    CpuStatus cpu_status = get_cpu_status(); //cpu
    GpuStatus gpu_status = get_gpu_status(); //gpu
    MemoryStats mem_status = get_memory_status(); //内存
    NetworkStats net_stats = get_network_usage(network_card_id_); //net

    std_msgs::msg::String sys_msg;
    std::stringstream ss;
    ss<<"Cpu_load: "<<cpu_status.load <<"%, Cpu_temperature: "<<cpu_status.temperature<<"℃, Cpu_frequency: "<<cpu_status.frequency
    <<"Mhz Gpu_load: "<<gpu_status.load<<"%, Memory: "<<mem_status.load;
    sys_msg.data = ss.str();
    // sys_msg.data = "CPU: " + std::to_string(cpu_usage) + "%, "
    //              + "GPU: " + std::to_string(gpu_usage) + "%, "
    //              + "Mem: " + std::to_string(mem_usage) + "%";

    sys_pub_->publish(sys_msg);
    RCLCPP_INFO(this->get_logger(), "Published: %s", sys_msg.data.c_str());

    // 诊断消息
    diagnostic_msgs::msg::DiagnosticArray diag_array;
    diag_array.header.stamp = this->now();
    //CPU diagnostics
    diagnostic_msgs::msg::DiagnosticStatus diag_cpu_status;
    diag_cpu_status.name = "Cpu Status";
    diag_cpu_status.hardware_id = "Jetson/Orin";
    if(cpu_status.load < 50.0){
        diag_cpu_status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_cpu_status.message = "CPU Load Normal";
    }else if(cpu_status.load < 80.0){
        diag_cpu_status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        diag_cpu_status.message = "CPU Load High";
    }else{
        diag_cpu_status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        diag_cpu_status.message = "CPU Load Critical";
    }
    diagnostic_msgs::msg::KeyValue diag_cpu_kv;
    diag_cpu_kv.key = "Load";
    diag_cpu_kv.value = std::to_string(cpu_status.load) + "%";
    diag_cpu_status.values.push_back(diag_cpu_kv);
    diag_cpu_kv.key = "Temperature";
    diag_cpu_kv.value = std::to_string(cpu_status.temperature) + "℃";
    diag_cpu_status.values.push_back(diag_cpu_kv);
    diag_cpu_kv.key = "Frequency";
    diag_cpu_kv.value = std::to_string(cpu_status.frequency) + "Mhz";
    diag_cpu_status.values.push_back(diag_cpu_kv);

    

    //GPU Diagnostics
    diagnostic_msgs::msg::DiagnosticStatus diag_gpu_status;
    diag_gpu_status.name = "GPU Status";
    diag_gpu_status.hardware_id = "Jetson/Orin";
    if(gpu_status.load < 50.0){
        diag_gpu_status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_gpu_status.message = "GPU Load Normal";
    }else if(gpu_status.load < 80.0){
        diag_gpu_status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        diag_gpu_status.message = "CPU Load High";
    }else{
        diag_gpu_status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR ;
        diag_gpu_status.message = "CPU Load Critical";
    }
    diagnostic_msgs::msg::KeyValue diag_gpu_kv;
    diag_gpu_kv.key = "Load";
    diag_gpu_kv.value = std::to_string(gpu_status.load) + "%";
    diag_gpu_status.values.push_back(diag_gpu_kv);
    diag_gpu_kv.key = "Temperature";
    diag_gpu_kv.value = std::to_string(cpu_status.temperature) + "℃";
    diag_gpu_status.values.push_back(diag_gpu_kv);
    diag_gpu_kv.key = "Frequency";
    diag_gpu_kv.value = std::to_string(cpu_status.frequency) + "Mhz";
    diag_gpu_status.values.push_back(diag_gpu_kv);

    //Mem
    diagnostic_msgs::msg::DiagnosticStatus diag_mem_status;
    diag_mem_status.name = "Mem Status";
    diag_mem_status.hardware_id = "Jetson/Orin";
    if(mem_status.load < 50.0){
        diag_mem_status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        diag_mem_status.message = "Mem Load Normal";
    }else if(mem_status.load < 80.0){
        diag_mem_status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        diag_mem_status.message = "Mem Load High";
    }else{
        diag_mem_status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        diag_mem_status.message = "Mem Load Critical";
    }
    diagnostic_msgs::msg::KeyValue diag_mem_kv;
    diag_mem_kv.key = "Load";
    diag_mem_kv.value = std::to_string(mem_status.load) + "%";
    diag_mem_status.values.push_back(diag_mem_kv);

    diag_mem_kv.key = "Temperature";
    diag_mem_kv.value = std::to_string(mem_status.temperature_celsius) + "℃";
    diag_mem_status.values.push_back(diag_mem_kv);

    diag_mem_kv.key = "Uptime";
    diag_mem_kv.value = mem_status.temperature_celsius;
    diag_mem_status.values.push_back(diag_mem_kv);

    diag_mem_kv.key = "Read_mb";
    diag_mem_kv.value = mem_status.read_mb;
    diag_mem_status.values.push_back(diag_mem_kv);

    diag_mem_kv.key = "Write_mb";
    diag_mem_kv.value = mem_status.write_mb;
    diag_mem_status.values.push_back(diag_mem_kv);



    diagnostic_msgs::msg::DiagnosticStatus net_status;
    net_status.name = "Network contition";
    net_status.hardware_id = "Jetson/Orin";
    net_status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    net_status.message = "Normal";
    diagnostic_msgs::msg::KeyValue rx_kv, tx_kv;
    rx_kv.key = "Download Speed";
    rx_kv.value = std::to_string(net_stats.rx_rate) + "KB/s";
    tx_kv.key = "Upload Speed";
    tx_kv.value = std::to_string(net_stats.tx_rate) + "KB/s";

    net_status.values.push_back(rx_kv);
    net_status.values.push_back(tx_kv);


    diag_array.status.push_back(diag_cpu_status);
    diag_array.status.push_back(diag_gpu_status);
    diag_array.status.push_back(diag_mem_status);
    diag_array.status.push_back(net_status);
    
    diag_pub_->publish(diag_array);

}

// double SystemMonitor::get_cpu_status(){
//     // std::ifstream file("/proc/stat");  //ifstram输入文件流类，用于读取文件，创建file，并读取/proc/stat
//     // std::string line;
//     // std::getline(file, line);        //用getline读取第一行，再次调用getline会读取下一行
//     std::ifstream file("/proc/stat"); 
//     if(!file.is_open()){
//         // std::cerr<<"Error: Unable to open /proc/stat"<<std::endl;
//         RCLCPP_ERROR(this->get_logger(),"Error: Unable to open /proc/stat");
//         return -1.0;
//     }
//     std::string line;
//     if(std::getline(file, line)){
//         std::istringstream iss(line);
//         std::string cpu;
//         long user, nice, system, idle;
//         if(iss >> cpu >> user >> nice >> system >>idle){
//             return(user+nice+system)*100.0 / (user + nice + system +idle);
//         }else{
//             RCLCPP_ERROR(this->get_logger(),"Error: Unable to parse CPU data");
//         }   
//     }else{
//             std::cerr<<"Error: Failed to read from /proc/stat"<<std::endl;
//         }
    
//     file.close();
// }

CpuStatus SystemMonitor::get_cpu_status(){
    double cpu_load=-1.0;
    double cpu_temp=-1.0;
    //CPU Load
    std::ifstream file("/proc/stat"); 
    if(!file.is_open()){
        // std::cerr<<"Error: Unable to open /proc/stat"<<std::endl;
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to open /proc/stat");
    }

    std::string line;
    std::getline(file, line);
    file.close();

    std::istringstream iss(line);
    std::string cpu_label;
    long user, nice, system, idle;
    if(iss >> cpu_label >> user >> nice >> system >>idle){
        cpu_load = (user+nice+system)*100.0 / (user + nice + system +idle);
    }else{
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to parse CPU data");
    }

    // CPU tempreture
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if(temp_file.is_open()){
        int temp_millideg;
        temp_file >> temp_millideg;
        cpu_temp = temp_millideg / 1000.0;
        temp_file.close(); 
    }else{
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to open cpu tempreture file");
    }

    // CPU frequency
    std::vector<double> cpu_frequencies;
    for(int i = 0; i < 12; ++i){
        std::ifstream freq_file("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq");
        if(freq_file.is_open()){
            double freq_khz;
            freq_file >> freq_khz;
            cpu_frequencies.push_back(freq_khz / 1000.0); //转换为Mhz
            freq_file.close();
        }else{
            break; //停止读取（某些系统可能CPU核心编号不连续）
        }
    }
    double avg_freq = (!cpu_frequencies.empty())? std::accumulate(cpu_frequencies.begin(), cpu_frequencies.end(), 0.0) / cpu_frequencies.size() : -1.0;

    return {cpu_load, cpu_temp, avg_freq};
}

GpuStatus SystemMonitor::get_gpu_status(){
    double gpu_load = -1.0;
    double gpu_freq = -1.0;
    double temp_millideg = -1.0;
    //Gpu Load
    std::ifstream load_file("/sys/devices/gpu.0/load");
    if(load_file.is_open()){
        load_file >> gpu_load;
        load_file.close();
        gpu_load = gpu_load/10.0;
    }else{
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to open GPU Load File!");
    }
    
    //Gpu Frequency
    std::ifstream freq_file("/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq");
    if(freq_file.is_open()){
        freq_file >> gpu_freq;
        freq_file.close();
        gpu_freq = gpu_freq / 1000000.0;
    }else{
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to open GPU Frequency File!");
    }

    //Gpu Temperature
    std::ifstream temp_file("/sys/class/thermal/thermal_zone1/temp");
    if(temp_file.is_open()){
        temp_file >> temp_millideg;
        temp_file.close();
        temp_millideg = temp_millideg / 1000.0;
    }

    return {gpu_load, gpu_freq, temp_millideg};
}

// memory 监测
MemoryStats SystemMonitor::get_memory_status(){
    double load = -1.0;
    double  temperature_celsius = -1.0;
    std::string uptime = "NO";
    double read_mb = -1.0, write_mb = -1.0;
    uint64_t prev_read_sectors_ = 0;
    uint64_t prev_write_sectors_ = 0;

    int interval_ms = 1000;

    //内存使用情况
    struct sysinfo memInfo;
    if(sysinfo(&memInfo) != 0){
        RCLCPP_ERROR(this->get_logger(), "Failed to get memory info");
    }
    // return 100.0 *(1.0-(double)mem_info.freeram/mem_info.totalram);
    long total_memory_mb = memInfo.totalram * memInfo.mem_unit / (1024*1024);
    long free_mem = memInfo.freeram * memInfo.mem_unit /(1024*1024);
    load = (total_memory_mb - free_mem) / total_memory_mb;

    //获取内存温度
    std::ifstream temp_file("/sys/class/thermal/thermal_zone2/temp");
    if(temp_file.is_open()){
        temp_file >> temperature_celsius;
        temp_file.close();
        temperature_celsius /=1000;
    }else{
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to open Memory Temp File!");
    }

    //获取系统开机时间
    std::ifstream uptime_file("/proc/uptime");
    if(uptime_file.is_open()){
        double uptime_seconds;
        uptime_file >> uptime_seconds;
        uptime_file.close();
        int hours = static_cast<int>(uptime_seconds)/3600;
        int minutes = (static_cast<int>(uptime_seconds)%3600)/60;
        uptime = std::to_string(hours) + "h " + std::to_string(minutes) + "m";
    }else{
        RCLCPP_ERROR(this->get_logger(),"Error: Unable to open UpTime File!");
    }

    //读取恢复错误 不支持
    //读取内存读写速率
    uint64_t read_sectors = 0, write_sectors = 0;
    std::ifstream disk_file("/proc/diskstats");
    if(disk_file.is_open()){
        std::string line, device;
        int read_sectors = 0, write_sectros = 0;
        while(std::getline(disk_file, line)){
            std::istringstream iss(line);
            uint64_t major, minor;
            std::string dev;
            uint64_t reads, reads_merged, read_sectors_tmp, read_time;
            uint64_t writes, writes_merged, write_sectors_tmp, write_time;
            iss >> major >> minor >> dev
                >> reads >> reads_merged >> read_sectors_tmp >> read_time 
                >> writes >> writes_merged >> write_sectors_tmp >> write_time;
            
            if(dev == "mmcblk0"){
                read_sectors = read_sectors_tmp;
                write_sectors = write_sectors_tmp;
                break;
            }

        }
        disk_file.close();
    }
    if(prev_read_sectors_ == 0 || prev_write_sectors_ == 0){
        prev_read_sectors_ = read_sectors;
        prev_write_sectors_ = write_sectors;
    }

    double time_interval = interval_ms / 1000.0;
    double read_rate = (read_sectors - prev_read_sectors_)*512.0 / (1021*1024) /time_interval;
    double write_rate = (write_sectors - prev_write_sectors_)*512.0 / (1021*1024) /time_interval;

    prev_read_sectors_ = read_sectors;
    prev_write_sectors_ = write_sectors;

    read_mb = read_rate;
    write_mb = write_rate;

    return {load, temperature_celsius, uptime, read_mb, write_mb};
    
}

NetworkStats SystemMonitor::get_network_usage(const std::string &interface){
    NetworkStats stats = {0, 0};

    std::ifstream net_dev("/proc/net/dev");
    if(!net_dev.is_open()){
        RCLCPP_ERROR(this->get_logger(),"Failed to open /proc/net/dev");
        return stats;
    }

    std::string line;

    while(std::getline(net_dev, line)){
        if(line.find(interface) != std::string::npos){
            std::istringstream iss(line);  //将字符串line绑定到输入流对象iss上，后续可通过>>操作符从iss中提取数据；
            std::string iface;
            long rx_bytes, tx_bytes;
            iss>> iface >> rx_bytes; //读取接收字节，>>会跳过开头的空白符，读取连续的非空白符，直到遇到下一个空白符，跳过空白符，读取数值
            for(int i=0; i< 7;++i){//读取发送字节
                iss>>tx_bytes;
            }

            static long last_rx =0,last_tx = 0;
            static auto last_time = std::chrono::steady_clock::now();

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - last_time).count();

            if(last_rx > 0 && last_tx > 0){
                stats.rx_rate = (rx_bytes - last_rx)/elapsed/1024.0; //kb/s
                stats.tx_rate = (tx_bytes - last_tx)/elapsed/1024.0; //kb/s
            }
            last_rx = rx_bytes;
            last_tx = tx_bytes;
            last_time = now;
            break;
        }
    }

    return stats;
}