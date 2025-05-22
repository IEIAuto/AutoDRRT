#include <chrono>
#include "bevdet_ros2.h"
//#include "highgui.hpp"

using Label = autoware_auto_perception_msgs::msg::ObjectClassification;

std::map< int, std::vector<int>> colormap { 
    {0, {0, 0, 255}},  // dodger blue 
    {1, {0, 201, 87}},   // 青色
    {2, {0, 201, 87}},
    {3, {160, 32, 240}},
    {4, {3, 168, 158}},
    {5, {255, 0, 0}},
    {6, {255, 97, 0}},
    {7, {30,  0, 255}},
    {8, {255, 0, 0}},
    {9, {0, 0, 255}},
    {10, {0, 0, 0}}
};

void Getinfo(void) {
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device label: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  Shared memory in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);
    }
    printf("\n");
}

void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel=false) {
    std::ofstream out_file;
    out_file.open(file_name, std::ios::out);
    if (out_file.is_open()) {
        for (const auto &box : boxes) {
            out_file << box.x << " ";
            out_file << box.y << " ";
            out_file << box.z << " ";
            out_file << box.l << " ";
            out_file << box.w << " ";
            out_file << box.h << " ";
            out_file << box.r << " ";
            if (with_vel) {
                out_file << box.vx << " ";
                out_file << box.vy << " ";
            }
            out_file << box.score << " ";
            out_file << box.label << "\n";
        }
    }
    out_file.close();
    return;
};


Eigen::VectorXd Box2Eigen(const Box& box) {
    Eigen::VectorXd vec(7);
    vec << box.x, box.y, box.z + box.h * 0.5, box.l, box.w, box.h, box.r;
    return vec;
}

std::vector<Eigen::VectorXd> Boxes2VecXd(const std::vector<Box> &boxes) {
    std::vector<Eigen::VectorXd> eigen_boxes;
    eigen_boxes.reserve(boxes.size());

    for (const auto& box: boxes) {
        eigen_boxes.emplace_back(Box2Eigen(box)); 
    }
    return eigen_boxes;
}


void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, std::vector<Box> &lidar_boxes,
                     const Eigen::Quaternion<float> &lidar2ego_rot, const Eigen::Translation3f &lidar2ego_trans) { 
    for (size_t i = 0; i < ego_boxes.size(); i++) {
        Box b = ego_boxes[i];
        Eigen::Vector3f center(b.x, b.y, b.z);
        center -= lidar2ego_trans.translation();         
        center = lidar2ego_rot.inverse().matrix() * center;
        b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
        b.x = center.x();
        b.y = center.y();
        b.z = center.z();
        lidar_boxes.push_back(b);
    }
}


void box3DToDetectedObject(const std::vector<Box>& ego_boxes, 
                           autoware_auto_perception_msgs::msg::DetectedObjects & output_msg) {
    for (auto & box3d: ego_boxes) {
        autoware_auto_perception_msgs::msg::DetectedObject obj;
        obj.existence_probability = box3d.score;

        // classification
        autoware_auto_perception_msgs::msg::ObjectClassification classification;
        classification.probability = 1.0f;
    
        if (box3d.label == 0) {
            classification.label = Label::CAR;
        } else if (box3d.label == 1) {
            classification.label = Label::BICYCLE;
        } else if (box3d.label == 2) {
            classification.label = Label::UNKNOWN;
        } else if (box3d.label == 3) {
            classification.label = Label::TRUCK;
        } else if (box3d.label == 4) {
            classification.label = Label::BUS;
        //} else if (box3d.label == 5) {
        //    classification.label = Label::CONSTRUCTION_VEHICLE;
        } else if (box3d.label == 6) {
            classification.label = Label::MOTORCYCLE;
        }

        obj.classification.emplace_back(classification);

        // float yaw = -box3d.r - tier4_autoware_utils::pi / 2;
        float yaw = -box3d.r;
        obj.kinematics.pose_with_covariance.pose.position =
            tier4_autoware_utils::createPoint(box3d.x, box3d.y, (box3d.z + box3d.h * 0.5));
        obj.kinematics.pose_with_covariance.pose.orientation =
            tier4_autoware_utils::createQuaternionFromYaw(yaw);
        obj.shape.type = autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX;
        obj.shape.dimensions =
            tier4_autoware_utils::createTranslation(box3d.l, box3d.w, box3d.h);

        // twist
        float vel_x = box3d.vx;
        float vel_y = box3d.vy;
        geometry_msgs::msg::Twist twist;
        twist.linear.x = std::sqrt(std::pow(vel_x, 2) + std::pow(vel_y, 2));
        twist.angular.z = 2 * (std::atan2(vel_y, vel_x) - yaw);
        obj.kinematics.twist_with_covariance.twist = twist;
        obj.kinematics.has_twist = "true";
        output_msg.objects.emplace_back(obj);
    }
}

double quaternionToYaw(const geometry_msgs::msg::Quaternion& quat) {
    return std::atan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 
                      1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z));
}

void DetectedObjectTobox3D(autoware_auto_perception_msgs::msg::DetectedObjects output_msg,
                           std::vector<Box>& ego_boxes) {

    for (auto & obj: output_msg.objects) {
        Box box3d;

        if (obj.classification[0].label == Label::CAR) {
            box3d.label = 0;
        } else if (obj.classification[0].label == Label::BICYCLE) {
            box3d.label = 1;
        } else if (obj.classification[0].label == Label::UNKNOWN) {
            box3d.label = 2;
        } else if (obj.classification[0].label == Label::TRUCK) {
            box3d.label = 3;
        } else if (obj.classification[0].label == Label::BUS) {
            box3d.label = 4;
        //} else if (obj.classification[0].label == Label::CONSTRUCTION_VEHICLE) {
        //    box3d.label = 5;
        } else if (obj.classification[0].label == Label::MOTORCYCLE) {
            box3d.label = 6;
        }

        box3d.score = obj.classification[0].probability;
        box3d.vx = 0;
        box3d.vy = 0;
        box3d.l =  obj.shape.dimensions.x;
        box3d.w =  obj.shape.dimensions.y;
        box3d.h =  obj.shape.dimensions.z;

        box3d.x = obj.kinematics.pose_with_covariance.pose.position.x; 
        box3d.y = obj.kinematics.pose_with_covariance.pose.position.y; 
        box3d.z = obj.kinematics.pose_with_covariance.pose.position.z - box3d.h * 0.5;
        box3d.r = quaternionToYaw(obj.kinematics.pose_with_covariance.pose.orientation);
        
        if (sqrt(pow(box3d.x, 2) + pow(box3d.y, 2)) < 80) {
            ego_boxes.emplace_back(box3d);
       }
    }
}


std::string vectorToString(const std::vector<Eigen::VectorXd>& vec) {
    std::stringstream ss;
    ss << "Vector contains " << vec.size() << " elements:\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << "  [" << vec[i].transpose() << "]";
        if (i != vec.size() - 1) ss << "\n";
    }
    return ss.str();
}

// 常量定义
const std::vector<std::string> CAM_TYPES = {"CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_BACK_LEFT", "CAM_BACK"};

const std::map<std::string, Matrix4d> CAM_TRANSFORMS = {
    {"CAM_FRONT_RIGHT", (Matrix4d() <<
        -0.76604,  0.64279,  0.0,  1.48544,
             0.0,      0.0, -1.0, -0.9,
        -0.64279, -0.76604,  0.0, -0.05898,
             0.0,      0.0,  0.0,  1.0).finished()},
    {"CAM_FRONT_LEFT", (Matrix4d() <<
        0.76604,  0.64279,  0.0,  -1.54972,
            0.0,      0.0, -1.0,  -0.9,
       -0.64279,  0.76604,  0.0,  -0.13558,
            0.0,      0.0,  0.0,   1.0).finished()},
    // 其他相机类型变换矩阵...
};

const Matrix3d INTRINSICS_1 = (Matrix3d() <<
    1144.08,     0.0, 960,
        0.0, 1144.08, 540,
        0.0,     0.0, 1.0).finished();

// 读取边界框数据
MatrixXd read_boxes(const std::string& path) {
    MatrixXd boxes;
    std::ifstream fin(path);
    // 读取并解析数据...
    return boxes;
}

void inference_result(std::vector<Eigen::VectorXd> eigen_boxes, std::vector<cv::Mat> & vec_Mat) { 
    int i;
    for (const auto& cam_type : CAM_TYPES) {
        Eigen::Matrix4d transformation_matrix;
        if (cam_type == "CAM_FRONT_RIGHT") {
            transformation_matrix << -0.76604,  0.64279,  0.00000,  1.48544,
                                      0.00000,  0.00000, -1.00000, -0.90000,
                                     -0.64279, -0.76604,  0.00000, -0.05898,
                                      0.00000,  0.00000,  0.00000,  1.00000;
            i = 0;
        } else if (cam_type == "CAM_FRONT_LEFT") {
            transformation_matrix << 0.76604,  0.64279,  0.00000, -1.54972,
                                     0.00000,  0.00000, -1.00000, -0.90000,
                                    -0.64279,  0.76604,  0.00000, -0.13558,
                                     0.00000,  0.00000,  0.00000,  1.00000;
            i = 1;
        } else if (cam_type == "CAM_BACK_RIGHT") {
            transformation_matrix << -0.70711, -0.70711,  0.00000, -1.27279,
                                      0.00000,  0.00000, -1.00000, -0.90000,
                                      0.70711, -0.70711,  0.00000, -0.14142,
                                      0.00000,  0.00000,  0.00000,  1.00000;
            i = 2;
        } else if (cam_type == "CAM_FRONT") {
            transformation_matrix << 0.00000, -1.00000,  0.00000,  0.00000,
                                     0.00000,  0.00000, -1.00000, -0.90000,
                                     1.00000,  0.00000,  0.00000, -2.90000,
                                     0.00000,  0.00000,  0.00000,  1.00000;
            i = 3;
        } else if (cam_type == "CAM_BACK_LEFT") {
            transformation_matrix << 0.70711, -0.70711,  0.00000,  1.27279,
                                     0.00000,  0.00000, -1.00000, -0.90000,
                                     0.70711,  0.70711,  0.00000, -0.14142,
                                     0.00000,  0.00000,  0.00000,  1.00000;
            i = 4;
        } else if (cam_type == "CAM_BACK") {
            transformation_matrix << -0.00000,  1.00000,  0.00000, -0.00000,
                                      0.00000,  0.00000, -1.00000, -0.90000,
                                     -1.00000, -0.00000,  0.00000, -2.05000,
                                      0.00000,  0.00000,  0.00000,  1.00000;
            i = 5;
        }

        Eigen::Matrix4d cam2img = Eigen::Matrix4d::Identity();
        cam2img.block<3, 3>(0, 0) = INTRINSICS_1;

        Eigen::Matrix4d transform = cam2img * transformation_matrix;

        vec_Mat[i] = visualize_camera(vec_Mat[i], eigen_boxes, transform);

        // 保存结果
        //std::string output_dir = "/home/orin/disk/ros2_bevdet/bevdet_tools/result/" + cam_type +".png";
        //imwrite(output_dir, vec_Mat[i]);
    }
}


std::vector<cv::Mat> shuffle(std::vector<cv::Mat> vec_Mat) {
    std::vector<int> new_order = {4, 3, 2, 1, 5, 0};
    std::vector<cv::Mat> rearranged(vec_Mat.size());
    for (size_t i = 0; i < vec_Mat.size(); i++) {
         rearranged[i] = vec_Mat[new_order[i]];
    }
/*
    for (size_t i = 0; i < 6; i++) {
        if (i == 0) {
            cv::imwrite("/home/orin/disk/ros2_bevdet/bevdet_tools/CAM_FRONT_RIGHT.png", rearranged[i]);
        } else if (i == 1) {
            cv::imwrite("/home/orin/disk/ros2_bevdet/bevdet_tools/CAM_FRONT_LEFT.png", rearranged[i]);
        } else if (i == 2) {
            cv::imwrite("/home/orin/disk/ros2_bevdet/bevdet_tools/CAM_BACK_RIGHT.png", rearranged[i]);
        } else if (i == 3) {
            cv::imwrite("/home/orin/disk/ros2_bevdet/bevdet_tools/CAM_FRONT.png", rearranged[i]);
        } else if (i == 4) {
            cv::imwrite("/home/orin/disk/ros2_bevdet/bevdet_tools/CAM_BACK_LEFT.png", rearranged[i]);
        } else if (i == 5) {
            cv::imwrite("/home/orin/disk/ros2_bevdet/bevdet_tools/CAM_BACK.png", rearranged[i]);
        }
    }
*/
    return rearranged;
}


cv::Mat Six2One(std::vector<cv::Mat> & vec_Mat) {
    // 验证图像尺寸一致性
    cv::Size base_size;
    for (int i = 0; i < 6; i++) {
        if (vec_Mat[i].empty()) {
            std::cerr << "Failed to load: " << i << std::endl;
            // return -1;
        }
        if (i == 0) {
            base_size = vec_Mat[i].size();
        } else if (base_size != vec_Mat[i].size()) {
            std::cerr << "Size mismatch at: " << i << std::endl;
            // return -1;
        }
    }

    // 步骤2：创建拼接画布（2行3列）
    const int cols = 3, rows = 2;
    cv::Mat panorama(
        base_size.height * rows,  // 总高度
        base_size.width * cols,   // 总宽度
        CV_8UC3,                  // 数据类型
        cv::Scalar(0, 0, 0)       // 黑色背景
    );

    // 步骤3：矩阵块拼接（无间隙）
    for (int i = 0; i < 6; i++) {
        // 计算当前图像位置
        int row = i / cols;
        int col = i % cols;

        // 定义目标ROI区域
        cv::Rect roi(
            col * base_size.width,  // x起始
            row * base_size.height, // y起始
            base_size.width,        // 区块宽度
            base_size.height        // 区块高度
        );

        // 执行深拷贝（保持原始图像数据）
        cv::Mat target_roi = panorama(roi);
        vec_Mat[i].copyTo(target_roi);
    }

    return panorama;
}


sensor_msgs::msg::Image Mat2RosImage(const cv::Mat& image, const std::string& encoding = "bgr8") {
    sensor_msgs::msg::Image ros_image;
    cv_bridge::CvImage cv_image;

    cv_image.header.stamp = rclcpp::Clock().now();
    cv_image.header.frame_id = "camera";
    cv_image.encoding = encoding;
    cv_image.image = image;

    cv_image.toImageMsg(ros_image);
    return ros_image;
}


BevDetNode::BevDetNode(): Node("ros2_bevdet") {
    
    pkg_path_ = ament_index_cpp::get_package_share_directory("ros2_bevdet");

    const auto img_N_ = static_cast<size_t>(this->declare_parameter<int64_t>("N"));
    const auto img_w_ = static_cast<size_t>(this->declare_parameter<int64_t>("W"));
    const auto img_h_ = static_cast<size_t>(this->declare_parameter<int64_t>("H"));

    uchar_images = new unsigned char*[img_N_];

    for (size_t i = 0; i < img_N_; i++) {
        uchar_images[i] = new unsigned char[img_w_ * img_h_ * 3];
    }

    const std::string model_config_= pkg_path_ + "/" + this->declare_parameter<std::string>("ModelConfig");
    const std::string imgstage_file_= pkg_path_ + "/" + this->declare_parameter<std::string>("ImgStageEngine");
    const std::string bevstage_file_= pkg_path_ + "/" + this->declare_parameter<std::string>("BEVStageEngine");
    
    const std::string output_lidarbox_= pkg_path_ + "/" + this->declare_parameter<std::string>("OutputLidarBox");
     
    camconfig_ = YAML::LoadFile(pkg_path_ + "/" + this->declare_parameter<std::string>("CamConfig")); 

    sample_= this->declare_parameter<std::vector<std::string>>("cams");

    for(auto file : sample_) {
        // imgs_file_.push_back(pkg_path_ +"/"+ file.second.as<std::string>());
        imgs_name_.push_back(file.as<std::string>()); 
    }

    // 读取图像参数
    sampleData_.param = camParams(camconfig_, img_N_, imgs_name_);

    // 模型配置文件，图像数量，cam内参，cam2ego的旋转和平移，模型权重文件
    bevdet_ = std::make_shared<BEVDet>(model_config_, img_N_, sampleData_.param.cams_intrin, sampleData_.param.cams2ego_rot, 
                                      sampleData_.param.cams2ego_trans, imgstage_file_, bevstage_file_);
    
    // gpu分配内参， cuda上分配6张图的大小 每个变量sizeof(uchar)个字节，并用imgs_dev指向该gpu上内存, sizeof(uchar) =1
    CHECK_CUDA(cudaMalloc((void**)&imgs_dev_, img_N_ * 3 * img_w_ * img_h_ * sizeof(uchar)));
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    using std::placeholders::_5;
    using std::placeholders::_6;
    using std::placeholders::_7;
        
    rmw_qos_profile_t image_rmw_qos(rclcpp::SensorDataQoS().get_rmw_qos_profile());
    image_rmw_qos.depth = 7;
    image_rmw_qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;

    //Front_Right
    image_sub_sync[0].subscribe(this, "/sensing/camera/camera3/image3", image_rmw_qos);
    //Front_Left
    image_sub_sync[1].subscribe(this, "/sensing/camera/camera4/image4", image_rmw_qos);
    //Back_Right
    image_sub_sync[2].subscribe(this, "/sensing/camera/camera0/image0", image_rmw_qos);
    //Front
    image_sub_sync[3].subscribe(this, "/sensing/camera/camera5/image5", image_rmw_qos);
    //Back_Left
    image_sub_sync[4].subscribe(this, "/sensing/camera/camera1/image1", image_rmw_qos);
    //Back
    image_sub_sync[5].subscribe(this, "/sensing/camera/camera2/image2", image_rmw_qos);
    //pcl
    pcl_sub_sync.subscribe(this, "/perception/object_recognition/detection/centerpoint/objects", image_rmw_qos);

    sync_stream1.reset(new message_filters::Synchronizer<approximate_policy_stream1>(approximate_policy_stream1(300), image_sub_sync[0], image_sub_sync[1], image_sub_sync[2], image_sub_sync[3], image_sub_sync[4], image_sub_sync[5], pcl_sub_sync));

    sync_stream1->registerCallback(std::bind(&BevDetNode::sync_stream_callback1, this, _1, _2, _3, _4, _5, _6, _7, img_N_, img_w_, img_h_));

    //objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/perception/object_recognition/detection/centerpoint/objects", rclcpp::QoS(1));
    //objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/perception/object_recognition/detection/centerpoint/validation/objects", rclcpp::QoS(1));
    //objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/perception/object_recognition/detection/objects", rclcpp::QoS(1));

    objects_pub_total = this->create_publisher<sensor_msgs::msg::Image>("Image_total", rclcpp::QoS(1));
}

void BevDetNode::image_trans_to_uncharptr_to_gpu(const std::vector<sensor_msgs::msg::Image> & msg_total, uchar* out_imgs, std::vector<cv::Mat> & vec_Mat, size_t img_N_, size_t img_w_, size_t img_h_) {

    cv::Mat cv_image[img_N_];
    // unsigned char* image_ptr[msg_num] = {(unsigned char*) "0"};
    // 定义一个变量指向gpu
    uchar* temp_gpu = nullptr;
    uchar* temp = new uchar[img_w_ * img_h_ * 3];
    // gpu分配内存 存储1张图像的大小 sizeof(uchar) = 1
    CHECK_CUDA(cudaMalloc(&temp_gpu, img_h_ * img_w_ * 3));

    vec_Mat.clear();

    for (size_t i = 0; i < img_N_; i++) {
        cv_image[i] = cv_bridge::toCvCopy(msg_total[i], sensor_msgs::image_encodings::BGR8)->image;
        cv::resize(cv_image[i], cv_image[i], cv::Size(img_w_, img_h_));
        std::memcpy(uchar_images[i], cv_image[i].data, img_w_ * img_h_ * 3 * sizeof(uchar));
        cv::Mat test_img(img_h_, img_w_, CV_8UC3, uchar_images[i]);
        vec_Mat.emplace_back(test_img);

        CHECK_CUDA(cudaMemcpy(temp_gpu, uchar_images[i], img_w_ * img_h_ * 3, cudaMemcpyHostToDevice));
        convert_RGBHWC_to_BGRCHW(temp_gpu, out_imgs + i * img_w_ * img_h_ * 3, 3, img_h_, img_w_);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(temp_gpu));
    delete[] temp;
}


void BevDetNode::sync_stream_callback1(const sensor_msgs::msg::Image::ConstSharedPtr &msg0, const sensor_msgs::msg::Image::ConstSharedPtr &msg1, const sensor_msgs::msg::Image::ConstSharedPtr &msg2, const sensor_msgs::msg::Image::ConstSharedPtr &msg3, const sensor_msgs::msg::Image::ConstSharedPtr &msg4, const sensor_msgs::msg::Image::ConstSharedPtr &msg5, const autoware_auto_perception_msgs::msg::DetectedObjects::ConstSharedPtr &msg6, size_t img_N_, size_t img_w_, size_t img_h_) {

    std::cout << "I get sth. from Camera" << std::endl;
    std::cout << "msg0->header = " << msg0->header.frame_id << " " << msg0->header.stamp.sec << "\n";

    std::vector<sensor_msgs::msg::Image> msg_total;
    msg_total.emplace_back(*msg0);
    msg_total.emplace_back(*msg1);
    msg_total.emplace_back(*msg2);
    msg_total.emplace_back(*msg3);
    msg_total.emplace_back(*msg4);
    msg_total.emplace_back(*msg5);

    std::vector<Box> ego_boxes;
    ego_boxes.clear();
 
    DetectedObjectTobox3D(*msg6, ego_boxes);
    std::vector<Eigen::VectorXd> eigen_boxes = Boxes2VecXd(ego_boxes);

    std::vector<cv::Mat> vec_Mat;
    image_trans_to_uncharptr_to_gpu(msg_total, imgs_dev_, vec_Mat, img_N_, img_w_, img_h_);

    inference_result(eigen_boxes, vec_Mat);
   
    auto rearranged_Mat = shuffle(vec_Mat);
    cv::Mat Picx6 = Six2One(rearranged_Mat);

    cv::Mat small_one; 
    cv::resize(Picx6, small_one, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    //cv::resize(Picx6, small_one, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
/*
    std::string output_dir = "/home/orin/disk/ros2_bevdet/bevdet_tools/result/all.png";
    cv::imshow(output_dir, small_one);
*/
    sensor_msgs::msg::Image image_total = Mat2RosImage(Picx6, "bgr8"); 
    //sensor_msgs::msg::Image image_total = Mat2RosImage(small_one, "bgr8"); 
    objects_pub_total->publish(image_total);

/*
    autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
    box3DToDetectedObject(ego_boxes, output_msg);
    //std::cout << "msg0->header =" << msg0->header.frame_id << " " << msg0->header.stamp.sec << "\n";
    output_msg.header = msg0->header;
    output_msg.header.frame_id = "velodyne_top_base_link";
    objects_pub_->publish(output_msg);
    //exit(0);
*/
}


BevDetNode::~BevDetNode() {
    delete imgs_dev_;
    for (size_t i = 0; i < img_N_; i++) {
        delete uchar_images[i];
    }
    delete uchar_images;
}

int main(int argc, char **argv) { 
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BevDetNode>());
    rclcpp::shutdown();
    return 0;
}
