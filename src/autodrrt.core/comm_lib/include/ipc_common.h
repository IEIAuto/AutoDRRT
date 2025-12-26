/**
 * @file ipc_common.h
 * @brief 共享内存 IPC 通信的公共定义和数据结构
 * 
 * 本文件定义了生产者和消费者之间共享内存通信所需的：
 * - 共享内存名称常量
 * - 图像数据尺寸参数
 * - 元数据结构（用于同步和数据描述）
 */

#ifndef IPC_COMMON_H
#define IPC_COMMON_H

#include <cstddef>
#include <cstdint>

// ============================================================================
// 共享内存名称定义
// ============================================================================

/** @brief 元数据共享内存的名称（POSIX 共享内存） */
#define META_SHM_NAME "/cuda_ipc_meta"

// ============================================================================
// 图像数据参数定义
// ============================================================================

/** @brief 图像宽度（像素） */
#define IMAGE_WIDTH 4096

/** @brief 图像高度（像素） */
#define IMAGE_HEIGHT 8536

/** @brief 图像通道数（RGB = 3） */
#define IMAGE_CHANNELS 3

/** @brief 图像数据总大小（字节）= 宽度 × 高度 × 通道数 */
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)

// ============================================================================
// 元数据结构定义
// ============================================================================

/**
 * @struct shm_meta
 * @brief 共享内存元数据结构，用于生产者和消费者之间的同步和数据描述
 * 
 * 该结构存储在独立的共享内存区域（META_SHM_NAME）中，包含：
 * - 同步标志（ready, ack）
 * - 数据共享内存的名称
 * - 图像数据的尺寸信息
 * - 时间戳（用于延迟统计）
 */
struct shm_meta {
    /** @brief 数据就绪标志：1=生产者已写入数据，0=数据未就绪 */
    int ready;
    
    /** @brief 确认标志：1=消费者已处理完成，0=未处理 */
    int ack;
    
    /** @brief 数据共享内存的名称（用于消费者打开对应的共享内存） */
    char shm_name[64];
    
    /** @brief 图像宽度（像素） */
    size_t width;
    
    /** @brief 图像高度（像素） */
    size_t height;
    
    /** @brief 图像通道数（通常为 3，表示 RGB） */
    size_t channels;
    
    /** @brief 数据总大小（字节）= width × height × channels */
    size_t data_size;
    
    /** @brief 时间戳（纳秒），由生产者设置，用于计算端到端延迟 */
    uint64_t timestamp_ns;
    
    /** @brief 发布索引（从 0 开始递增），由生产者设置，用于追踪发布次数和消息对应 */
    uint64_t publish_index;

    /** @brief 发布者写入共享内存阶段耗时（微秒） */
    int64_t shm_write_us;

    /** @brief 发布者更新元数据耗时（微秒） */
    int64_t metadata_us;

    /** @brief 发布者 ROS2 publish 耗时（微秒） */
    int64_t ros_pub_us;
};

#endif
