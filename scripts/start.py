"""
一键启动脚本（Windows PowerShell 辅助）
用法：python scripts/start.py [service]
service: all | backend | worker | frontend | infra
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: str, cwd: Path = ROOT, new_window: bool = False):
    """执行命令"""
    if new_window and sys.platform == "win32":
        # Windows: 在新的 PowerShell 窗口中运行
        full_cmd = f'start "Sci_RAG" powershell -NoExit -Command "cd \'{cwd}\'; {cmd}"'
        subprocess.Popen(full_cmd, shell=True, cwd=cwd)
    else:
        subprocess.Popen(cmd, shell=True, cwd=cwd)


def check_env():
    """检查 .env 文件是否存在"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        print("⚠️  未找到 .env 文件，正在从 .env.example 复制...")
        import shutil
        shutil.copy(ROOT / ".env.example", env_file)
        print("✅ 已创建 .env，请编辑其中的 LLM_API_KEY 后重新运行")
        return False
    return True


def main():
    service = sys.argv[1] if len(sys.argv) > 1 else "all"

    if not check_env():
        return

    print(f"\n🚀 启动 Sci_RAG — 服务: {service}\n")

    if service in ("all", "infra"):
        print("📦 启动基础设施 (Docker Compose)...")
        run("docker-compose up -d", new_window=False)
        time.sleep(3)

    if service in ("all", "backend"):
        print("⚡ 启动 FastAPI 后端...")
        run("python main.py", new_window=True)
        time.sleep(2)

    if service in ("all", "worker"):
        print("⚙️  启动 Celery Worker...")
        run(
            "celery -A worker.celery_app worker --loglevel=info -Q parsing,indexing --concurrency=2",
            new_window=True,
        )
        time.sleep(1)

    if service in ("all", "frontend"):
        print("🌐 启动 Streamlit 前端...")
        run("streamlit run frontend/app.py --server.port=8501", new_window=True)

    if service == "all":
        print("\n✅ 所有服务已启动！")
        print("   前端: http://localhost:8501")
        print("   API:  http://localhost:8000")
        print("   API文档: http://localhost:8000/docs")
        print("   Celery监控: celery -A worker.celery_app flower")


if __name__ == "__main__":
    main()
