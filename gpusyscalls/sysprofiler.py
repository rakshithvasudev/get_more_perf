import argparse
import psutil
import GPUtil
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor

def collect_cpu_metrics():
    return psutil.cpu_percent(interval=1)

def collect_memory_metrics():
    return psutil.virtual_memory().percent

def collect_gpu_metrics():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming we're using the first GPU
        return {
            'gpu_util': gpu.load * 100,
            'gpu_memory': gpu.memoryUtil * 100,
            'gpu_power': gpu.powerDraw if hasattr(gpu, 'powerDraw') else 0
        }
    return {'gpu_util': 0, 'gpu_memory': 0, 'gpu_power': 0}

def collect_metrics(stop_event):
    data = []
    start_time = time.time()
    
    while not stop_event.is_set():
        cpu = collect_cpu_metrics()
        memory = collect_memory_metrics()
        gpu_metrics = collect_gpu_metrics()
        
        data.append({
            'timestamp': time.time() - start_time,
            'cpu': cpu,
            'memory': memory,
            **gpu_metrics
        })
        
        time.sleep(1)  # Collect metrics every second
    
    return pd.DataFrame(data)

def run_program(command):
    process = subprocess.Popen(command, shell=True)
    return process

def profile(command, output_file):
    stop_event = threading.Event()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        metrics_future = executor.submit(collect_metrics, stop_event)
        program_future = executor.submit(run_program, command)
        
        program_future.result()  # Wait for the program to finish
        stop_event.set()  # Stop collecting metrics
        
        df = metrics_future.result()
    
    df.to_csv(output_file, index=False)
    return df

def plot_metrics(df, output_file):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    ax1.plot(df['timestamp'], df['cpu'], label='CPU')
    ax1.plot(df['timestamp'], df['memory'], label='Memory')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_title('CPU and Memory Utilization')
    ax1.legend()
    
    ax2.plot(df['timestamp'], df['gpu_util'], label='GPU Utilization')
    ax2.plot(df['timestamp'], df['gpu_memory'], label='GPU Memory')
    ax2.set_ylabel('Utilization (%)')
    ax2.set_title('GPU Utilization and Memory')
    ax2.legend()
    
    ax3.plot(df['timestamp'], df['gpu_power'], label='GPU Power')
    ax3.set_ylabel('Power (W)')
    ax3.set_title('GPU Power Consumption')
    ax3.legend()
    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_file)

def main():
    parser = argparse.ArgumentParser(description='System Profiler')
    parser.add_argument('command', help='Command to run and profile')
    parser.add_argument('--output', default='metrics.csv', help='Output file for metrics (CSV)')
    parser.add_argument('--plot', default='metrics_plot.png', help='Output file for plot (PNG)')
    args = parser.parse_args()
    
    print(f"Profiling command: {args.command}")
    df = profile(args.command, args.output)
    
    print(f"Metrics saved to: {args.output}")
    plot_metrics(df, args.plot)
    print(f"Plot saved to: {args.plot}")
    
    print("\nSummary:")
    print(f"Average CPU Utilization: {df['cpu'].mean():.2f}%")
    print(f"Average Memory Utilization: {df['memory'].mean():.2f}%")
    print(f"Average GPU Utilization: {df['gpu_util'].mean():.2f}%")
    print(f"Average GPU Memory Utilization: {df['gpu_memory'].mean():.2f}%")
    print(f"Average GPU Power Consumption: {df['gpu_power'].mean():.2f}W")
    print(f"Total Runtime: {df['timestamp'].max():.2f}s")

if __name__ == "__main__":
    main()
