"""Sample scenarios demonstrating the forethought-reflection loop."""
from __future__ import annotations


def run_demo(app):  # type: ignore[override]
    print("==========================================")
    print("🚀 Scenario 1: Agent encounters dangerous operation for the first time")
    print("==========================================")
    app.invoke({"query": "我想清空当前目录下的所有 git 修改，用什么命令？", "retry_count": 0})

    print("\n\n==========================================")
    print("🚀 Scenario 2: Agent encounters similar problem the second time")
    print("==========================================")
    app.invoke({"query": "只要是没提交的文件我都想删了，怎么弄？", "retry_count": 0})
