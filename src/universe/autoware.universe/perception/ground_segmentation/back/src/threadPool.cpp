#include "ground_segmentation/threadPool.hpp"

ThreadPool::ThreadPool():
m_threads(std::vector<std::thread>(10)), m_shutdown(false), busyBus(0)
{
    for (int i = 0; i < m_threads.size(); ++i) {

        m_threads[i] = std::thread(ThreadWorker(this, i));  //分配工作线程

    } 
}

ThreadPool::ThreadPool(const int n_threads):
m_threads(std::vector<std::thread>(n_threads)), m_shutdown(false)
{
    for (int i = 0; i < m_threads.size(); ++i) {

        m_threads[i] = std::thread(ThreadWorker(this, i));  //分配工作线程

    }
}

void ThreadPool::init() {
    for (int i = 0; i < m_threads.size(); ++i) {

        m_threads[i] = std::thread(ThreadWorker(this, i));  //分配工作线程

    }
}

void ThreadPool::shutdown() {
    m_shutdown = true;
    m_conditional_lock.notify_all();   //通知 唤醒所有工作线程
    
    for (int i = 0; i < m_threads.size(); ++i) {

        if (m_threads[i].joinable()) {  //判断线程是否正在等待

            m_threads[i].join();  //将线程加入等待队列

        }
    }
}
