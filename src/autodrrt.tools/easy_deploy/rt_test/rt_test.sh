#!/bin/bash

# 测试时长（秒）
DURATION=60

# 日志保存路径
mkdir -p results
LOG_FILE="results/cyclictest_result_$(date +%Y%m%d_%H%M%S).log"

# 检查依赖
check_dependencies() {
    for cmd in cyclictest stress; do
        if ! command -v $cmd &>/dev/null; then
            echo "缺少命令: $cmd，请先执行：sudo apt install rt-tests stress"
            exit 1
        fi
    done
}

# 启动压力测试
start_stress() {
    echo "启动 stress 压力负载..."
    stress --cpu 6 --io 4 --vm 2 --vm-bytes 128M --timeout ${DURATION}s &
    STRESS_PID=$!
    echo "stress 已启动 (PID=$STRESS_PID)"
}

# 启动 cyclictest，并保存结果
run_cyclictest() {
    echo "正在运行 cyclictest...（${DURATION}s）"
    sudo cyclictest --smp --priority=99 --interval=1000 --duration=${DURATION}s | tee "$LOG_FILE"
    echo "测试日志已保存到 $LOG_FILE"
}

# 清理子进程
cleanup() {
    if [[ -n "$STRESS_PID" ]]; then
        echo "正在清理 stress (PID=$STRESS_PID)..."
        kill -9 $STRESS_PID &>/dev/null
    fi
}

# 分析 Max 延迟
analyze_result() {
    echo "正在分析最大延迟..."
    MAX_LATENCY=$(grep "Max:" "$LOG_FILE" | awk '{print $NF}' | sort -nr | head -1)

    if [[ -z "$MAX_LATENCY" ]]; then
        echo "无法解析 Max 延迟。"
        return
    fi

    echo "最大延迟：${MAX_LATENCY} μs"

    if (( MAX_LATENCY < 100 )); then
        echo "实时性非常优秀（<100μs）"
    elif (( MAX_LATENCY < 1000 )); then
        echo "实时性尚可（<1ms）"
    elif (( MAX_LATENCY < 5000 )); then
        echo "实时性较差（<5ms），建议优化系统或考虑 RT 内核"
    else
        echo "实时性非常差（>5ms），不适合硬实时控制系统"
    fi
}

# 主流程
main() {
    check_dependencies

    echo "测试将运行 ${DURATION}s。"
    read -p "是否启用 stress 负载进行并发压力测试？[y/N]: " use_stress
    if [[ "$use_stress" == "y" || "$use_stress" == "Y" ]]; then
        start_stress
    fi

    trap cleanup EXIT
    run_cyclictest
    analyze_result
}

main

