@echo off
chcp 65001 >nul
echo ===================================================
echo      🚀 Sci_RAG 知识库系统 一键启动程序 🚀
echo ===================================================
echo.

:: 1. 启动主后端 API
echo [1/3] 正在启动 FastAPI 主后端服务...
start "Sci_RAG - 后端 API" cmd /k "call Env\Scripts\activate.bat && python main.py"

:: 停顿 2 秒，给后端一点启动时间
timeout /t 2 >nul

:: 2. 启动后台包工头 (Celery Workers)
echo [2/3] 正在启动 Celery 后台解析引擎...
start "Sci_RAG - Celery Worker" cmd /k "call Env\Scripts\activate.bat && set PYTHONPATH=. && python worker\start_workers.py"

:: 停顿 2 秒
timeout /t 2 >nul

:: 3. 启动前端界面 (已修复路径激活逻辑)
echo [3/3] 正在启动前端用户界面...
start "Sci_RAG - 前端 Web" cmd /k "call Env\Scripts\activate.bat && cd frontend && streamlit run app.py"

echo.
echo ✅ 启动指令已全部发送！
echo 👉 请查看弹出的三个独立控制台窗口日志。
echo 🛑 想要关闭系统时，直接关闭那三个黑色窗口即可。
pause