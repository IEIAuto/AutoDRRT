/*
 *  Copyright (C) 2007 Austin Robot Technology, Patrick Beeson
 *  Copyright (C) 2009, 2010 Austin Robot Technology, Jack O'Quin
 *  Copyright (C) 2015, Jack O'Quin
 *
 *  License: Modified BSD Software License Agreement
 *
 *  $Id$
 */

/** \file
 *
 *  Input classes for the Velodyne HDL-64E 3D LIDAR:
 *
 *     Input -- base class used to access the data independently of
 *              its source
 *
 *     InputSocket -- derived class reads live data from the device
 *              via a UDP socket
 *
 *     InputPCAP -- derived class provides a similar interface from a
 *              PCAP dump
 */

#include <unistd.h>
#include <string>
#include <sstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <poll.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <velodyne_driver/input.h>
#include <velodyne_driver/time_conversion.hpp>

namespace velodyne_driver
{
  static const size_t packet_size =
    sizeof(velodyne_msgs::msg::VelodynePacket().data);

  ////////////////////////////////////////////////////////////////////////
  // Input base class implementation
  ////////////////////////////////////////////////////////////////////////

  /** @brief constructor
   *
   *  @param node_ptr ROS node handle for calling node.
   *  @param port UDP port number.
   */
  Input::Input(rclcpp::Node * node_ptr, uint16_t port):
    node_ptr_(node_ptr),
    port_(port)
  {
    devip_str_ = node_ptr_->declare_parameter("device_ip", std::string(""));
    sensor_timestamp_ = node_ptr_->declare_parameter("sensor_timestamp", false);
    if (!devip_str_.empty())
      RCLCPP_INFO_STREAM(node_ptr_->get_logger(), "Only accepting packets from IP address: "
                      << devip_str_);
  }

  ////////////////////////////////////////////////////////////////////////
  // InputSocket class implementation
  ////////////////////////////////////////////////////////////////////////

  /** @brief constructor
   *
   *  @param private_nh ROS private handle for calling node.
   *  @param port UDP port number
   */
  InputSocket::InputSocket(rclcpp::Node * node_ptr, uint16_t port):
    Input(node_ptr, port)
  {
    sockfd_ = -1;

    if (!devip_str_.empty()) {
      inet_aton(devip_str_.c_str(),&devip_);
    }

    // connect to Velodyne UDP port
    RCLCPP_INFO_STREAM(node_ptr_->get_logger(), "Opening UDP socket: port " << port);
    sockfd_ = socket(PF_INET, SOCK_DGRAM, 0);
    if (sockfd_ == -1)
      {
        perror("socket");               // TODO: RCLCPP_ERROR errno
        return;
      }

    sockaddr_in my_addr;                     // my address information
    memset(&my_addr, 0, sizeof(my_addr));    // initialize to zeros
    my_addr.sin_family = AF_INET;            // host byte order
    my_addr.sin_port = htons(port);          // port in network byte order
    my_addr.sin_addr.s_addr = INADDR_ANY;    // automatically fill in my IP

    if (bind(sockfd_, (sockaddr *)&my_addr, sizeof(sockaddr)) == -1)
      {
        perror("bind");                 // TODO: RCLCPP_ERROR errno
        return;
      }

    if (fcntl(sockfd_,F_SETFL, O_NONBLOCK|FASYNC) < 0)
      {
        perror("non-block");
        return;
      }

    RCLCPP_DEBUG(node_ptr_->get_logger(), "Velodyne socket fd is %d\n", sockfd_);
  }

  /** @brief destructor */
  InputSocket::~InputSocket(void)
  {
    (void) close(sockfd_);
  }

  /** @brief Get one velodyne packet. */
  int InputSocket::getPacket(velodyne_msgs::msg::VelodynePacket *pkt, const double time_offset)
  {
    double time1 = node_ptr_->now().seconds();

    struct pollfd fds[1];
    fds[0].fd = sockfd_;
    fds[0].events = POLLIN;
    static const int POLL_TIMEOUT = 1000; // one second (in msec)

    sockaddr_in sender_address;
    socklen_t sender_address_len = sizeof(sender_address);

    while (rclcpp::ok())
      {
        // Unfortunately, the Linux kernel recvfrom() implementation
        // uses a non-interruptible sleep() when waiting for data,
        // which would cause this method to hang if the device is not
        // providing data.  We poll() the device first to make sure
        // the recvfrom() will not block.
        //
        // Note, however, that there is a known Linux kernel bug:
        //
        //   Under Linux, select() may report a socket file descriptor
        //   as "ready for reading", while nevertheless a subsequent
        //   read blocks.  This could for example happen when data has
        //   arrived but upon examination has wrong checksum and is
        //   discarded.  There may be other circumstances in which a
        //   file descriptor is spuriously reported as ready.  Thus it
        //   may be safer to use O_NONBLOCK on sockets that should not
        //   block.

        // poll() until input available
        do
          {
            if(!rclcpp::ok())
            {
              RCLCPP_ERROR(node_ptr_->get_logger(), "poll() error: shutdown requested");
              return -1;
            }
            int retval = poll(fds, 1, POLL_TIMEOUT);
            if (retval < 0)             // poll() error?
              {
                if (errno != EINTR)
                  RCLCPP_ERROR(node_ptr_->get_logger(), "poll() error: %s", strerror(errno));
                return -1;
              }
            if (retval == 0)            // poll() timeout?
              {
                RCLCPP_WARN(node_ptr_->get_logger(), "Velodyne poll() timeout");
                return 0;
              }
            if ((fds[0].revents & POLLERR)
                || (fds[0].revents & POLLHUP)
                || (fds[0].revents & POLLNVAL)) // device error?
              {
                RCLCPP_ERROR(node_ptr_->get_logger(), "poll() reports Velodyne error");
                return -1;
              }
          } while ((fds[0].revents & POLLIN) == 0);

        // Receive packets that should now be available from the
        // socket using a blocking read.
        ssize_t nbytes = recvfrom(sockfd_, &pkt->data[0],
                                  packet_size,  0,
                                  (sockaddr*) &sender_address,
                                  &sender_address_len);

        if (nbytes < 0)
          {
            if (errno != EWOULDBLOCK)
              {
                perror("recvfail");
                RCLCPP_INFO(node_ptr_->get_logger(), "recvfail");
                return -1;
              }
          }
        else if ((size_t) nbytes == packet_size)
          {
            // read successful,
            // if packet is not from the lidar scanner we selected by IP,
            // continue otherwise we are done
            if(devip_str_ != ""
               && sender_address.sin_addr.s_addr != devip_.s_addr)
              continue;
            else
              break; //done
          }

        RCLCPP_DEBUG_STREAM(node_ptr_->get_logger(), "incomplete Velodyne packet read: "
                         << nbytes << " bytes");
      }

      if (!sensor_timestamp_) {
        // Packet stamp from when read began. Add the time offset.
        auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(time1 + time_offset)).count();

        pkt->stamp = rclcpp::Time(time_ns);
      } else {
        // Time for each packet is a 4 byte uint located starting at offset 1200 in
        // the data packet
        auto ros_time_now = node_ptr_->now();
        pkt->stamp = rosTimeFromGpsTimestamp(ros_time_now, &(pkt->data[1200]));
      }


    return 1;
  }

  ////////////////////////////////////////////////////////////////////////
  // InputPCAP class implementation
  ////////////////////////////////////////////////////////////////////////

  /** @brief constructor
   *
   *  @param node_ptr ROS node_ptr for calling node.
   *  @param port UDP port number
   *  @param filename PCAP dump file name
   */
  InputPCAP::InputPCAP(rclcpp::Node * node_ptr, uint16_t port,
                       std::string filename, bool read_once,
                       bool read_fast, double repeat_delay):
    Input(node_ptr, port),
    last_packet_receive_time_(rclcpp::Time(0.0, RCL_ROS_TIME)),
    last_packet_stamp_(rclcpp::Time(0.0, RCL_ROS_TIME)),
    filename_(filename)
  {
    (void)read_once;
    (void)read_fast;
    (void)repeat_delay;

    pcap_ = NULL;
    empty_ = true;

    // get parameters using private node handle
    read_once_ = node_ptr_->declare_parameter("read_once", false);
    read_fast_ = node_ptr_->declare_parameter("read_fast", false);
    repeat_delay_ = node_ptr_->declare_parameter("repeat_delay", 0.0);

    if (read_once_)
      RCLCPP_INFO(node_ptr_->get_logger(), "Read input file only once.");
    if (read_fast_)
      RCLCPP_INFO(node_ptr_->get_logger(), "Read input file as quickly as possible.");
    if (repeat_delay_ > 0.0)
      RCLCPP_INFO(node_ptr_->get_logger(), "Delay %.3f seconds before repeating input file.",
               repeat_delay_);

    // Open the PCAP dump file
    RCLCPP_INFO(node_ptr_->get_logger(), "Opening PCAP file \"%s\"", filename_.c_str());
    if ((pcap_ = pcap_open_offline(filename_.c_str(), errbuf_) ) == NULL)
      {
        RCLCPP_FATAL(node_ptr_->get_logger(), "Error opening Velodyne socket dump file.");
        return;
      }

    std::stringstream filter;
    if( devip_str_ != "" )              // using specific IP?
      {
        filter << "src host " << devip_str_ << " && ";
      }
    filter << "udp dst port " << port;
    pcap_compile(pcap_, &pcap_packet_filter_,
                 filter.str().c_str(), 1, PCAP_NETMASK_UNKNOWN);
  }

  /** destructor */
  InputPCAP::~InputPCAP(void)
  {
    pcap_close(pcap_);
  }

  /** @brief Get one velodyne packet. */
  int InputPCAP::getPacket(velodyne_msgs::msg::VelodynePacket *pkt, const double time_offset)
  {
    (void)time_offset;

    struct pcap_pkthdr *header;
    const u_char *pkt_data;

    while (true)
      {
        int res;
        if ((res = pcap_next_ex(pcap_, &header, &pkt_data)) >= 0)
          {
            // Skip packets not for the correct port and from the
            // selected IP address.
            if ( /* !devip_str_.empty() &&  Let the filter take care of skipping bad packets ... not dependent on device IP setting*/
                (0 == pcap_offline_filter(&pcap_packet_filter_,
                                          header, pkt_data)))
            {
              continue;
            }

            memcpy(&pkt->data[0], pkt_data+BLOCK_LENGTH, packet_size);
            rclcpp::Time t=rclcpp::Clock{RCL_ROS_TIME}.now();
            pkt->stamp = rosTimeFromGpsTimestamp(t,&(pkt->data[TIMESTAMP_BYTE])); // time_offset not considered here, as no synchronization required
            empty_ = false;

            // Keep the reader from blowing through the file.
            if (read_fast_ == false)
            {
              if (last_packet_stamp_ != rclcpp::Time(0.0, RCL_ROS_TIME) && last_packet_receive_time_ != rclcpp::Time(0.0, RCL_ROS_TIME))
              {
                rclcpp::Time current_packet_stamp = pkt->stamp;
                rclcpp::Duration expected_cycle_time = current_packet_stamp - last_packet_stamp_;
                rclcpp::Time expected_end = last_packet_receive_time_ + expected_cycle_time;
                rclcpp::Time actual_end = rclcpp::Clock{RCL_ROS_TIME}.now();

                // Detect backward jumps in time
                if (actual_end < last_packet_receive_time_)
                {
                  expected_end = actual_end + expected_cycle_time;
                }

                // Calculate the time we'll sleep for.
                rclcpp::Duration sleep_time = expected_end - actual_end;


                // Make sure to reset our start time.
                last_packet_receive_time_ = expected_end;
                last_packet_stamp_ = current_packet_stamp;
                
                // If we've taken too much time we won't sleep.
                if(sleep_time <= rclcpp::Duration::from_seconds(0.0))
                {
                  // If we've jumped forward in time, or the loop has taken more than a full extra
                  // cycle, reset our cycle.
                  if (actual_end > expected_end + expected_cycle_time)
                  {
                    last_packet_receive_time_ = actual_end;
                  }                
                }
                else
                {
                  uint64_t count = sleep_time.nanoseconds();
                  rclcpp::sleep_for(std::chrono::nanoseconds(count));
                }
              }
              else
              {
                last_packet_stamp_ = pkt->stamp;
                last_packet_receive_time_ = rclcpp::Clock{RCL_ROS_TIME}.now();
              }              
            }
            return 1;                   // success
          }

        if (empty_)                 // no data in file?
          {
            RCLCPP_WARN(node_ptr_->get_logger(), "Error %d reading Velodyne packet: %s",
                     res, pcap_geterr(pcap_));
            return -1;
          }

        if (read_once_)
          {
            RCLCPP_INFO(node_ptr_->get_logger(), "end of file reached -- done reading.");
            return -1;
          }

        if (repeat_delay_ > 0.0)
          {
            RCLCPP_INFO(node_ptr_->get_logger(), "end of file reached -- delaying %.3f seconds.",
                     repeat_delay_);
            usleep(rint(repeat_delay_ * 1000000.0));
          }

        RCLCPP_DEBUG(node_ptr_->get_logger(), "replaying Velodyne dump file");

        // I can't figure out how to rewind the file, because it
        // starts with some kind of header.  So, close the file
        // and reopen it with pcap.
        pcap_close(pcap_);
        pcap_ = pcap_open_offline(filename_.c_str(), errbuf_);
        empty_ = true;              // maybe the file disappeared?
      } // loop back and try again
  }

} // velodyne_driver namespace
