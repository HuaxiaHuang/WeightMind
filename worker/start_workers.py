"""
Celery Worker 动态启动器 - 终极融合版
融合了 Windows 最佳实践 (多路独立 solo) 与 动态硬件感知
"""
import os
import subprocess
import time
import sys
import psutil
from loguru import logger

# ==========================================
# ⚙️ 资源调优配置
# ==========================================
TARGET_RESOURCE_RATIO = 0.5  # 目标使用资源的百分比 (本地 0.5，云端可改 0.9)

def calculate_optimal_workers() -> int:
    """动态计算最安全的 Worker 并发数量"""
    logger.info("🔍 正在侦测系统硬件资源...")
    
    # 1. 物理 CPU 核心数
    physical_cores = psutil.cpu_count(logical=False)
    
    # 2. 可用物理内存 (GB)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    logger.info(f"💻 硬件侦测: {physical_cores} 物理核心, 剩余可用内存: {available_ram_gb:.1f} GB")
    
    # ── 核心计算逻辑 ──
    target_cores = max(1, int(physical_cores * TARGET_RESOURCE_RATIO))
    
    # 内存防爆: Marker 模型极度吃内存，保守预留单 Worker 3-4GB
    safe_workers_by_ram = max(1, int(available_ram_gb / 4.0))
    
    # 取两者中的极小值，绝对保证系统安全不蓝屏！
    final_count = min(target_cores, safe_workers_by_ram)
    
    logger.info(f"⚖️ 算力评估: CPU允许 {target_cores} 个, 内存允许 {safe_workers_by_ram} 个")
    return final_count


def main():
    # 1. 动态获取你应该开几个 Worker
    worker_count = calculate_optimal_workers()
    print(f"\n🚀 正在启动 {worker_count} 个独立的 Celery Solo Worker...\n")
    
    processes = []
    try:
        # 2. 沿用你优秀的 Windows 多进程并发方案
        for i in range(worker_count):
            # 自动生成独立的名字，使用 solo 模式，保留了你的队列设置
            cmd = f"{sys.executable} -m celery -A worker.celery_app worker -n worker_{i}@%h --loglevel=info -Q parsing,indexing -P solo"
            
            # 启动子进程
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)
            print(f"✅ Worker {i} 启动指令已发送 [PID: {p.pid}]")
            
            # 错开启动时间，防止模型同时加载瞬间吸干内存，或者抢占 Redis 连接
            time.sleep(3) 

        print("\n🎯 所有 Worker 已就绪！它们正在后台抢单。")
        print("🛑 按 Ctrl+C 可一键安全关闭所有任务。\n")
        
        # 保持主进程运行，监控子进程
        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 收到关闭指令，正在安全清理所有 Worker...")
        for p in processes:
            p.terminate()
        sys.exit(0)

if __name__ == "__main__":
    # 确保 Python 能找到你的 core 和 worker 目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    os.environ["PYTHONPATH"] = project_root
    
    main()