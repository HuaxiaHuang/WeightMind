"""
Celery 应用配置
"""
from celery import Celery

from core.config import settings

celery_app = Celery(
    "sci_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # 时区
    timezone="Asia/Shanghai",
    enable_utc=True,
    
    # 任务超时
    task_soft_time_limit=600,    # 10 分钟软超时
    task_time_limit=900,         # 15 分钟硬超时
    
    # 重试策略
    task_max_retries=3,
    task_default_retry_delay=30,  # 重试间隔 30 秒
    
    # Worker 配置
    worker_prefetch_multiplier=1,   # 每次只取一个任务（解析任务很重）
    worker_max_tasks_per_child=50,  # 执行 50 个任务后重启 Worker（防内存泄漏）
    
    # 结果过期时间
    result_expires=86400,  # 24 小时
    
    # 任务路由
    task_routes={
        "worker.tasks.process_paper": {"queue": "parsing"},
        "worker.tasks.index_paper": {"queue": "indexing"},
    },
)

# 自动发现任务
celery_app.autodiscover_tasks(["worker"])
