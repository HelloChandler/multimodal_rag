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
├── configs/          # 配置文件目录
│   └── models.yaml   # 模型配置文件
├── data/            # 数据目录
├── src/             # 源码目录
├── tests/           # 测试目录
└── ...
```

## 🎯 模型配置系统
本项目实现了灵活的模型接入系统，支持通过配置文件新增模型接入能力，无需修改核心代码。

### 配置文件格式
模型配置文件位于 `configs/models.yaml`，支持以下结构：

```yaml
# 通用配置
general:
  default_timeout: 30  # 默认请求超时时间（秒）
  max_retries: 2       # 默认最大重试次数
  enable_async: true   # 是否启用异步调用

# 模型配置列表
models:
  # 示例：豆包多模态模型
  doubao:
    type: llm  # 模型类型：llm（大语言模型）或 embedding（嵌入模型）
    name: doubao-seed-1-6-vision-250815
    api_key: ${ARK_API_KEY}  # 从环境变量获取
    api_secret: ${ARK_API_SECRET}  # 从环境变量获取
    base_url: https://ark.cn-beijing.volces.com/api/v3  # 可选，模型API基础URL
    interface_type: openai  # 接口类型：openai（默认）、anthropic、google等
    parameter_mapping:  # 参数映射关系，将统一参数映射到模型特定参数
      temperature: temperature
      max_tokens: max_tokens
    response_format:  # 响应格式映射，将模型响应映射到统一格式
      content: choices[0].message.content
      model: model
    parameters:  # 可选参数
      temperature: 0.7
      max_tokens: 2000

# 回退策略配置
fallback:
  llm:
    order: [doubao, deepseek]  # 调用顺序，按优先级从高到低
    strategy: sequential  # 回退策略：sequential（顺序）或 random（随机）
    retry_on_failure: true  # 失败时是否重试
    max_retries_per_model: 1  # 每个模型的最大重试次数

# 缓存配置
caching:
  enabled: true  # 是否启用缓存
  type: memory  # 缓存类型：memory（内存）或 redis
  ttl: 3600  # 缓存过期时间（秒）
  max_size: 1000  # 最大缓存条目数

# 日志配置
logging:
  level: INFO  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/model_integration.log  # 日志文件路径
  rotate: true  # 是否启用日志轮转
  max_bytes: 10485760  # 单个日志文件最大大小（10MB）
  backup_count: 5  # 保留的日志文件数量
```

### 配置字段说明

#### 通用配置 (general)
- `default_timeout`: 默认请求超时时间（秒）
- `max_retries`: 默认最大重试次数
- `enable_async`: 是否启用异步调用

#### 模型配置 (models)
- `type`: 模型类型，可选值：`llm`（大语言模型）或 `embedding`（嵌入模型）
- `name`: 模型名称
- `api_key`: API密钥，可以从环境变量获取（格式：`${ENV_VAR_NAME}`）
- `api_secret`: 可选，API密钥（部分模型需要）
- `base_url`: 可选，模型API基础URL
- `region`: 可选，模型API区域（部分模型需要）
- `interface_type`: 接口类型，默认值：`openai`
- `parameter_mapping`: 可选，参数映射关系，将统一参数映射到模型特定参数
- `response_format`: 可选，响应格式映射，将模型响应映射到统一格式
- `parameters`: 可选，模型默认参数

#### 参数映射 (parameter_mapping)
参数映射用于将统一的参数名称映射到模型特定的参数名称，例如：

```yaml
parameter_mapping:
  temperature: temp  # 将统一参数temperature映射到模型参数temp
  max_tokens: max_length  # 将统一参数max_tokens映射到模型参数max_length
```

#### 响应格式映射 (response_format)
响应格式映射用于从模型响应中提取所需的字段，支持JSONPath表达式：

```yaml
response_format:
  content: choices[0].message.content  # 提取响应内容
  model: model  # 提取模型名称
  created: created  # 提取创建时间
  finish_reason: choices[0].finish_reason  # 提取结束原因
```

### 通过配置文件新增模型接入

要新增一个模型接入，只需在 `configs/models.yaml` 的 `models` 部分添加一个新的模型配置即可。例如，新增一个OpenAI兼容的自定义模型：

```yaml
custom-model:
  type: llm
  name: custom-gpt-model
  api_key: ${CUSTOM_API_KEY}
  base_url: https://api.custom.com/v1/chat/completions
  interface_type: openai
  parameter_mapping:
    temperature: temp
    max_tokens: max_length
  response_format:
    content: data.response.content
    model: data.model
  parameters:
    temperature: 0.7
    max_tokens: 2000
```

### 扩展接口类型

如果需要接入不兼容OpenAI接口的模型，可以通过以下步骤扩展接口类型：

1. 创建自定义的模型适配器类，继承自 `UnifiedLLM` 或 `UnifiedEmbeddings` 抽象基类
2. 实现必要的方法（如 `generate`, `chat` 或 `embed_query`, `embed_documents`）
3. 使用 `ModelFactory.register_interface_type` 方法注册新的接口类型

示例：

```python
from src.model.model_factory import ModelFactory
from src.model.unified_llm import UnifiedLLM

# 创建自定义的LLM类
class AnthropicLLM(UnifiedLLM):
    def __init__(self, config):
        super().__init__(config)
        # 初始化Anthropic客户端

    def generate(self, prompt, **kwargs):
        # 实现生成文本的逻辑
        pass

    def chat(self, messages, **kwargs):
        # 实现对话的逻辑
        pass

# 注册新的接口类型
ModelFactory.register_interface_type("anthropic", AnthropicLLM)
```

## 🔐 生产建议
- 配置 API Key 到安全密钥管理
- 在 CI/CD 中预构建向量库
- 使用配置文件管理模型接入，避免硬编码API密钥和模型参数
- 对于不兼容OpenAI接口的模型，通过扩展接口类型实现自定义接入

Enjoy building multimodal RAG!

