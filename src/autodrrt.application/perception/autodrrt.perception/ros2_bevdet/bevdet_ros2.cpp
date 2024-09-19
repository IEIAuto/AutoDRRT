#include "bevdet_ros2.h"
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

void Getinfo(void) 
{
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


void Boxes2Txt(const std::vector<Box> &boxes, std::string file_name, bool with_vel=false) 
{
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
      if(with_vel)
      {
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


void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
                                        std::vector<Box> &lidar_boxes,
                                        const Eigen::Quaternion<float> &lidar2ego_rot,
                                        const Eigen::Translation3f &lidar2ego_trans)
{
    
    for(size_t i = 0; i < ego_boxes.size(); i++)
    {
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


void box3DToDetectedObject(
    const std::vector<Box>& ego_boxes, 
    autoware_auto_perception_msgs::msg::DetectedObjects & output_msg)
{
    for(auto & box3d: ego_boxes)
    {
        autoware_auto_perception_msgs::msg::DetectedObject obj;
        obj.existence_probability = box3d.score;

        // classification
        autoware_auto_perception_msgs::msg::ObjectClassification classification;
        classification.probability = 1.0f;
    
        if (box3d.label == 0) {
            classification.label = Label::CAR;
        } else if (box3d.label == 1) {
            classification.label = Label::TRUCK;
        } else if (box3d.label == 3) {
            classification.label = Label::BUS;
        } else if (box3d.label == 4) {
            classification.label = Label::TRAILER;
        } else if (box3d.label == 7) {
            classification.label = Label::BICYCLE;
        } else if (box3d.label == 6) {
            classification.label = Label::MOTORCYCLE;
        } else if (box3d.label == 8) {
            classification.label = Label::PEDESTRIAN;
        } else {
            classification.label = Label::UNKNOWN;
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
// void Egobox2Lidarbox(const std::vector<Box>& ego_boxes, 
//                                         jsk_recognition_msgs::BoundingBoxArrayPtr lidar_boxes,
//                                         const Eigen::Quaternion<float> &lidar2ego_rot,
//                                         const Eigen::Translation3f &lidar2ego_trans, float score_thre = 0.2)
// {
    
//     for(auto b : ego_boxes)
//     {   
//         if(b.score < score_thre)
//             continue;
//         jsk_recognition_msgs::BoundingBox box;

//         Eigen::Vector3f center(b.x, b.y, b.z + b.h/2.);
//         // Eigen::Vector3f center(b.x, b.y, b.z);

//         center -= lidar2ego_trans.translation();         
//         center = lidar2ego_rot.inverse().matrix() * center;
        
//         b.r -= lidar2ego_rot.matrix().eulerAngles(0, 1, 2).z();
//         Eigen::Quaterniond q(Eigen::AngleAxisd(b.r, Eigen::Vector3d(0, 0, 1)));

//         box.pose.position.x = center.x();
//         box.pose.position.y = center.y();
//         box.pose.position.z = center.z();
//         box.pose.orientation.x = q.x();
//         box.pose.orientation.y = q.y();
//         box.pose.orientation.z = q.z();
//         box.pose.orientation.w = q.w();

//         // 长宽高不变
//         box.dimensions.x = b.l;
//         box.dimensions.y = b.w;
//         box.dimensions.z = b.h;
        
//         box.label = b.label;
//         box.header.frame_id = "map";
//         box.header.stamp = ros::Time::now();

//         lidar_boxes->boxes.emplace_back(box);
//     }
// }

BevDetNode::BevDetNode(): Node("ros2_bevdet")
{
    
    pkg_path_ = ament_index_cpp::get_package_share_directory("ros2_bevdet");

    const auto img_N_ = static_cast<size_t>(this->declare_parameter<int64_t>("N"));
    const auto img_w_ = static_cast<size_t>(this->declare_parameter<int64_t>("W"));
    const auto img_h_ = static_cast<size_t>(this->declare_parameter<int64_t>("H"));

    const std::string front = this->declare_parameter<std::string>("/front/image_raw_time");
    const std::string front_left = this->declare_parameter<std::string>("/front_left/image_raw_time");
    const std::string front_right = this->declare_parameter<std::string>("/front_right/image_raw_time");
    const std::string rear = this->declare_parameter<std::string>("/rear/image_raw_time");
    const std::string rear_left = this->declare_parameter<std::string>("/rear_left/image_raw_time");
    const std::string rear_right = this->declare_parameter<std::string>("/rear_right/image_raw_time");

    
    uchar_images = new unsigned char*[img_N_];

    for(size_t i = 0; i < img_N_; i++)
    {
        uchar_images[i] = new unsigned char[img_w_ * img_h_ * 3];
    }

    const std::string model_config_= pkg_path_ + "/" + this->declare_parameter<std::string>("ModelConfig");
    const std::string imgstage_file_= pkg_path_ + "/" + this->declare_parameter<std::string>("ImgStageEngine");
    const std::string bevstage_file_= pkg_path_ + "/" + this->declare_parameter<std::string>("BEVStageEngine");
    
    const std::string output_lidarbox_= pkg_path_ + "/" + this->declare_parameter<std::string>("OutputLidarBox");
     
    camconfig_ = YAML::LoadFile(pkg_path_ + "/" + this->declare_parameter<std::string>("CamConfig")); 

    sample_= this->declare_parameter<std::vector<std::string>>("cams");

    for(auto file : sample_)
    {
        // imgs_file_.push_back(pkg_path_ +"/"+ file.second.as<std::string>());
        imgs_name_.push_back(file.as<std::string>()); 
    }

    // 读取图像参数
    sampleData_.param = camParams(camconfig_, img_N_, imgs_name_);


    // 模型配置文件，图像数量，cam内参，cam2ego的旋转和平移，模型权重文件
    bevdet_ = std::make_shared<BEVDet>(model_config_, img_N_, sampleData_.param.cams_intrin, 
                sampleData_.param.cams2ego_rot, sampleData_.param.cams2ego_trans, 
                                                    imgstage_file_, bevstage_file_);
    
    
    // gpu分配内参， cuda上分配6张图的大小 每个变量sizeof(uchar)个字节，并用imgs_dev指向该gpu上内存, sizeof(uchar) =1
    CHECK_CUDA(cudaMalloc((void**)&imgs_dev_, img_N_ * 3 * img_w_ * img_h_ * sizeof(uchar)));
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    using std::placeholders::_4;
    using std::placeholders::_5;
    using std::placeholders::_6;
    // using std::placeholders::_7;
    
    
    // pub_cloud_ = n_.advertise<sensor_msgs::PointCloud2>("/points_raw", 10); 
    // pub_boxes_ = n_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/boxes", 10);   
    rmw_qos_profile_t image_rmw_qos(rclcpp::SensorDataQoS().get_rmw_qos_profile());
    image_rmw_qos.depth = 5;
    // image_rmw_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
    image_rmw_qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;


    image_sub_sync[0].subscribe(this, front, image_rmw_qos);
    image_sub_sync[1].subscribe(this, front_left,image_rmw_qos);
    image_sub_sync[2].subscribe(this, front_right,image_rmw_qos);
    image_sub_sync[3].subscribe(this, rear,image_rmw_qos);
    image_sub_sync[4].subscribe(this, rear_left,image_rmw_qos);
    image_sub_sync[5].subscribe(this, rear_right,image_rmw_qos);

    // pcl_sub_sync.subscribe(this, "/sensing/lidar/top/outlier_filtered/pointcloud", rclcpp::SensorDataQoS().get_rmw_qos_profile());

    sync_stream.reset(new message_filters::Synchronizer<approximate_policy_stream>(approximate_policy_stream(100), image_sub_sync[0], image_sub_sync[1], image_sub_sync[2], image_sub_sync[3], image_sub_sync[4], image_sub_sync[5]));

    sync_stream->registerCallback(std::bind(&BevDetNode::sync_stream_callback, this, _1, _2, _3, _4, _5, _6));

    // sync_->registerCallback(boost::bind(&BevDetNode::callback,this, _1, _2, _3, _4, _5, _6,_7)); // 绑定回调函数
    objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/bevdet/boundingbox", rclcpp::QoS(1));




    // input_topics_.push_back("/front/image_raw_time");
    // filters_.resize(input_topics_.size());
    // rclcpp::QoS qos(rclcpp::KeepLast(1));
    // qos.reliable();
    // for (size_t d = 0; d < input_topics_.size(); ++d) {

    //     // CAN'T use auto type here.
    //     std::function<void(const sensor_msgs::msg::Image::ConstSharedPtr msg)> cb = std::bind(
    //         &BevDetNode::async_stream_callback, this,
    //         std::placeholders::_1, input_topics_[d]);
    //     // rclcpp::SensorDataQoS().keep_last(20)
    //     filters_[d].reset();
    //         filters_[d] = this->create_subscription<sensor_msgs::msg::Image>(
    //         input_topics_[d], qos, cb);
    // }
}
void BevDetNode::async_stream_callback(const sensor_msgs::msg::Image::ConstSharedPtr & input_ptr, const std::string & topic_name)
    {   
        std::cout << "I get sth. from Camera" << std::endl;
        std::vector<sensor_msgs::msg::Image> msg_total;
        // if(!is_camerainfo_subcried)
        // {
        //     return;
        // }
        msg_total.emplace_back(*input_ptr);
        // msg_total.emplace_back(*msg1);
        // msg_total.emplace_back(*msg2);
        // msg_total.emplace_back(*msg3);
        // msg_total.emplace_back(*msg4);
        // msg_total.emplace_back(*msg5);
        
        // cv::Mat test_img(new_height,new_width,CV_8UC3,uchar_images[0]);
        // cv::imwrite("/home/orin/bev_fusion/src/ROS2-CUDA-BEVFusion/dump/mat0.jpg",cv_image[0]);
        // cv::imwrite("/home/orin/bev_fusion/src/ROS2-CUDA-BEVFusion/dump/test_img.jpg",test_img);
        image_trans_to_uncharptr_to_gpu(msg_total,imgs_dev_);
        sampleData_.imgs_dev = imgs_dev_;

        std::vector<Box> ego_boxes;
        ego_boxes.clear();
        float time = 0.f;
        // 测试推理  图像数据, boxes，时间
        bevdet_->DoInfer(sampleData_, ego_boxes, time);
        std::cout << "DoInfer  ok" << std::endl;
        std::cout << "ego_boxes size is" << ego_boxes.size() << std::endl;
        autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
        for(size_t box_index = 0; box_index <  ego_boxes.size(); box_index++)
        {
            std::cout << "x =" << ego_boxes[box_index].x << "\n" << \
            "y =" << ego_boxes[box_index].y << "\n" << \
            "z =" << ego_boxes[box_index].z << "\n" << \
            "l =" << ego_boxes[box_index].l << "\n" << \
            "w =" << ego_boxes[box_index].w << "\n" << \
            "h =" << ego_boxes[box_index].h << "\n" << \
            "r =" << ego_boxes[box_index].r << "\n" << \
            "label =" << ego_boxes[box_index].label << "\n" << std::endl;
        }
        box3DToDetectedObject(ego_boxes,output_msg);
        output_msg.header = input_ptr->header;
        output_msg.header.frame_id = "velodyne_top_base_link";
        objects_pub_->publish(output_msg);
    }
void BevDetNode::image_trans_to_uncharptr_to_gpu( const std::vector<sensor_msgs::msg::Image>  & msg_total, uchar* out_imgs) {
    // size_t new_width = 2048;
    // size_t new_height = 1280;
    // // std::vector<unsigned char*> images;
    // // unsigned char** uchar_images;
    // size_t msg_num = msg_total.size();

    cv::Mat cv_image[img_N_];
    // unsigned char* image_ptr[msg_num] = {(unsigned char*) "0"};
        // 定义一个变量指向gpu
    uchar* temp_gpu = nullptr;
    uchar* temp = new uchar[img_w_ * img_h_ * 3];
    // gpu分配内存 存储1张图像的大小 sizeof(uchar) = 1
    CHECK_CUDA(cudaMalloc(&temp_gpu, img_h_ * img_w_ * 3));

    

    for (size_t i = 0; i < img_N_; i++) {
        cv_image[i] = cv_bridge::toCvCopy(msg_total[i], sensor_msgs::image_encodings::BGR8)->image;
        cv::resize(cv_image[i], cv_image[i], cv::Size(img_w_, img_h_));
        
        // image_ptr[i] = cv_image[i].data;
        std::memcpy(uchar_images[i], cv_image[i].data, img_w_ * img_h_ * 3 * sizeof(uchar));
        cv::Mat test_img(img_h_,img_w_,CV_8UC3,uchar_images[i]);
        // cv::imwrite("/data/source/bevnet/bevdet_yutong/bevdet-tensorrt-cpp-master/mat0.jpg",cv_image[0]);
        // cv::imwrite("/data/source/bevnet/bevdet_yutong/bevdet-tensorrt-cpp-master/test_img.jpg",test_img);
        // std::cout << "camera_count is :" << img_N_ <<" w: " <<img_w_ <<" h: " << img_h_ <<std::endl;
        // images.emplace_back(image_ptr[i]);
        CHECK_CUDA(cudaMemcpy(temp_gpu, uchar_images[i], img_w_ * img_h_ * 3, cudaMemcpyHostToDevice));
        convert_RGBHWC_to_BGRCHW(temp_gpu, out_imgs + i * img_w_ * img_h_ * 3, 3, img_h_, img_w_);
        // CHECK_CUDA(cudaMemcpy(out_imgs + img_w_ * img_h_ * 3 * i, uchar_images[i], img_w_ * img_h_ * 3 * sizeof(uchar), cudaMemcpyHostToDevice));
    }

    // for (size_t i = 0; i < img_N_; i++) {

       
    //     CHECK_CUDA(cudaMemcpy(temp, out_imgs + img_w_ * img_h_ * 3 * i , img_w_ * img_h_ * 3 * sizeof(uchar), cudaMemcpyDeviceToHost));
    //     cv::Mat test_img(img_h_,img_w_,CV_8UC3,temp);
    //     cv::imwrite("/data/source/bevnet/bevdet_yutong/bevdet-tensorrt-cpp-master/after.jpg",test_img);
    //     // std::cout << "camera_count is :" << img_N_ <<" w: " <<img_w_ <<" h: " << img_h_ <<std::endl;
    //     // images.emplace_back(image_ptr[i]);
    //     // CHECK_CUDA(cudaMemcpy(out_imgs + img_w_ * img_h_ * 3 * i, uchar_images[i], img_w_ * img_h_ * 3 * img_N_ * sizeof(uchar), cudaMemcpyHostToDevice));
    // }


    CHECK_CUDA(cudaDeviceSynchronize());
    // cv::Mat test_img(new_height,new_width,CV_8UC3,uchar_images[0]);
    // cv::imwrite("/home/orin/bev_fusion/src/ROS2-CUDA-BEVFusion/dump/mat0.jpg",cv_image[0]);
    // cv::imwrite("/home/orin/bev_fusion/src/ROS2-CUDA-BEVFusion/dump/test_img.jpg",test_img);
    // std::cout << "/home/orin/bev_fusion/src/ROS2-CUDA-BEVFusion/dump/mat0.jpg" << std::endl;
    //     std::cout << "/home/orin/bev_fusion/src/ROS2-CUDA-BEVFusion/dump/test_img.jpg" << std::endl;
    // return uchar_images;
    CHECK_CUDA(cudaFree(temp_gpu));
    delete[] temp;
}

void BevDetNode::sync_stream_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg0, const sensor_msgs::msg::Image::ConstSharedPtr &msg1, const sensor_msgs::msg::Image::ConstSharedPtr &msg2, const sensor_msgs::msg::Image::ConstSharedPtr &msg3, const sensor_msgs::msg::Image::ConstSharedPtr &msg4, const sensor_msgs::msg::Image::ConstSharedPtr &msg5) {
        // cv::Mat img_fl, img_f, img_fr, img_bl, img_b, img_br;//camera orders
        std::cout << "I get sth. from Camera" << std::endl;
        std::vector<sensor_msgs::msg::Image> msg_total;
        // if(!is_camerainfo_subcried)
        // {
        //     return;
        // }

        std::cout << "msg0->header =" << msg0->header.frame_id << " " << msg0->header.stamp.sec << "\n";
        msg_total.emplace_back(*msg0);
        msg_total.emplace_back(*msg1);
        msg_total.emplace_back(*msg2);
        msg_total.emplace_back(*msg3);
        msg_total.emplace_back(*msg4);
        msg_total.emplace_back(*msg5);
        
  
        image_trans_to_uncharptr_to_gpu(msg_total,imgs_dev_);
        sampleData_.imgs_dev = imgs_dev_;

        std::vector<Box> ego_boxes;
        ego_boxes.clear();
        float time = 0.f;
        // 测试推理  图像数据, boxes，时间
        bevdet_->DoInfer(sampleData_, ego_boxes, time);
        std::cout << "DoInfer  ok" << std::endl;
        std::cout << "ego_boxes size is" << ego_boxes.size() << std::endl;
        autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
        for(size_t box_index = 0; box_index <  ego_boxes.size(); box_index++)
        {
            std::cout << "x =" << ego_boxes[box_index].x << "\n" << \
            "y =" << ego_boxes[box_index].y << "\n" << \
            "z =" << ego_boxes[box_index].z << "\n" << \
            "l =" << ego_boxes[box_index].l << "\n" << \
            "w =" << ego_boxes[box_index].w << "\n" << \
            "h =" << ego_boxes[box_index].h << "\n" << \
            "r =" << ego_boxes[box_index].r << "\n" << \
            "label =" << ego_boxes[box_index].label << "\n" << std::endl;
        }
        box3DToDetectedObject(ego_boxes,output_msg);
        std::cout << "msg0->header =" << msg0->header.frame_id << " " << msg0->header.stamp.sec << "\n";
        output_msg.header = msg0->header;
        output_msg.header.frame_id = "velodyne_top_base_link";
        objects_pub_->publish(output_msg);
    }

BevDetNode::~BevDetNode()
{
    delete imgs_dev_;
    for(size_t i = 0; i < img_N_; i++)
    {
    delete uchar_images[i];
    }
    delete uchar_images;
}


int main(int argc, char **argv)
{   
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BevDetNode>());
    rclcpp::shutdown();
    return 0;
}