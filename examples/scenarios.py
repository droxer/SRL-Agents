"""Sample scenarios demonstrating the forethought-reflection loop."""
from __future__ import annotations

from .logging import console


def run_demo(app):  # type: ignore[override]
    console.rule("🚀 Scenario 1 · First Encounter")
    app.invoke({"query": "我想清空当前目录下的所有 git 修改，用什么命令？", "retry_count": 0})

    console.rule("🚀 Scenario 2 · Similar Problem")
    app.invoke({"query": "只要是没提交的文件我都想删了，怎么弄？", "retry_count": 0})
