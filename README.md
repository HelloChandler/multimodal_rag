# Multimodal RAG (Text + Image)

企业级多模态检索增强生成（RAG）系统，支持 Markdown 文档及内嵌图片的解析、向量化、检索与问答。系统封装火山引擎 Doubao 多模态模型，并可扩展至 OpenAI、DeepSeek、Moonshot 等任意模型。

## 🚀 功能亮点
- Markdown + 图片混合解析，自动抽取/复制图片并转 base64
- 自定义多模态 Embedding（豆包 `doubao-embedding-vision-250615`）
- FAISS 本地向量库 + LangChain 检索链
- Doubao 多模态 LLM（`doubao-seed-1-6-vision-250815`），可热插拔替换
- CLI & FastAPI 双入口，支持纯文本或图片查询

## 📦 快速开始
```bash
cd multimodal_rag
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```

环境变量配置于 `.env`（见示例）。核心参数：
```
ARK_API_KEY=xxx
ARK_API_SECRET=xxx
DOUBAO_LLM_MODEL=doubao-seed-1-6-vision-250815
DOUBAO_EMBED_MODEL=doubao-embedding-vision-250615
```

## 🗂️ 数据准备
将 Markdown 与图片放入 `data/raw/`。`file_utils` 会自动解析 inline 图片（`![desc](path)`），并复制图片到该目录。处理后的文本块写入 `data/processed/`，向量库落盘 `data/index/faiss.index`。

## 🔧 构建向量库
```bash
python -m src.pipelines.build_index --source data/raw --chunk-size 500 --overlap 100
```
`--source` 默认为 `.env` 中配置的 `RAW_DATA_DIR`（未配置时为 `data/raw`），可按需指向任意 Markdown 数据目录。
该脚本会：
1. 遍历 Markdown/文本
2. 抽取文本与图片
3. 清理 + 切片
4. 生成多模态向量并写入 FAISS

## 🤖 运行 RAG
### CLI
```bash
python -m src.app.main --query "图中是什么？" --image data/raw/example.png
```

### FastAPI
```bash
uvicorn src.app.api:app --reload
```
POST `/rag`：
```json
{
  "query": "文档里解释了什么？",
  "image_base64": "..."
}
```

## ✅ 测试
```bash
pytest
```

## 🧱 目录结构
```
multimodal_rag/
├── data/
├── src/
├── tests/
└── ...
```

## 🔐 生产建议
- 配置 API Key 到安全密钥管理
- 在 CI/CD 中预构建向量库
- 根据场景扩展 `LLMBase` 与 `MultimodalEmbeddings` 实现，统一接口即可替换模型

Enjoy building multimodal RAG!

