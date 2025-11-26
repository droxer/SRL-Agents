-- 1. 启用向量扩展 (只需执行一次)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 创建带有向量字段的表
-- 这里的 1536 是 OpenAI text-embedding-3-small 的维度
CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(50),
    insight TEXT NOT NULL,           -- 具体的经验规则
    reasoning TEXT,                  -- 反思的逻辑（为什么会有这条规则）
    embedding vector(1536),          -- 【核心】语义向量
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 创建索引以加速检索 (IVFFlat 或 HNSW)
CREATE INDEX ON agent_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);