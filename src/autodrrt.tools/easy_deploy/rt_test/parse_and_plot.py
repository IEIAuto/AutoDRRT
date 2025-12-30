import sys
import os
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    threads = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("T:"):
                parts = line.split()
                thread_id = parts[1].strip('()')
                max_latency = int(parts[-1])
                avg_latency = int(parts[-3])
                threads[thread_id] = {
                    'max': max_latency,
                    'avg': avg_latency
                }
    return threads

def plot_latencies(data, title="Cyclictest Latencies", save_path=None):
    thread_ids = list(data.keys())
    max_vals = [data[tid]['max'] for tid in thread_ids]
    avg_vals = [data[tid]['avg'] for tid in thread_ids]

    x = range(len(thread_ids))
    plt.figure(figsize=(10, 5))
    plt.bar(x, max_vals, color='red', alpha=0.6, label='Max Latency (us)')
    plt.bar(x, avg_vals, color='blue', alpha=0.6, label='Avg Latency (us)')
    plt.xticks(x, thread_ids)
    plt.xlabel("Thread ID")
    plt.ylabel("Latency (us)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ 图表已保存至: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python3 parse_and_plot.py <cyclictest_log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    if not os.path.isfile(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        sys.exit(1)

    data = parse_log_file(log_file)
    basename = os.path.basename(log_file).replace('.log', '')
    output_path = f"plot/{basename}_plot.png"
    plot_latencies(data, title=f"Cyclictest Result: {basename}", save_path=output_path)

