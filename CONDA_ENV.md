# Conda 虚拟环境使用文档

本文档详细介绍了如何使用 Conda 管理本项目的虚拟环境，包括环境创建、激活、更新和删除等操作。

## 前提条件

确保你的系统已安装 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

## 环境创建

### 使用 environment.yml 创建环境

项目根目录下已提供 `environment.yml` 文件，包含所有必要的依赖。执行以下命令创建环境：

```bash
conda env create -f environment.yml
```

环境创建成功后，你将拥有一个名为 `multimodal_rag` 的虚拟环境，Python 版本为 3.10。

### 手动创建环境（可选）

如果你需要手动创建环境，可以执行以下命令：

```bash
# 创建环境
conda create -n multimodal_rag python=3.10

# 激活环境
conda activate multimodal_rag

# 安装依赖
pip install -r requirements.txt
```

## 环境激活

### Windows

```bash
conda activate multimodal_rag
```

### macOS/Linux

```bash
conda activate multimodal_rag
```

## 环境更新

### 更新特定依赖

```bash
# 使用 pip 更新
pip install --upgrade 包名

# 或使用 conda 更新
conda update 包名
```

### 更新所有依赖

```bash
# 使用 requirements.txt 更新
pip install -r requirements.txt --upgrade

# 或重新创建环境
conda env update -f environment.yml
```

## 环境删除

如果你不再需要这个环境，可以执行以下命令删除：

```bash
conda remove -n multimodal_rag --all
```

## 项目运行

### 构建索引

在激活环境后，你可以使用以下命令从 Markdown 文档构建 ChromaDB 索引：

```bash
python -m src.pipelines.build_index
```

你也可以指定自定义参数：

```bash
python -m src.pipelines.build_index --source data/raw --chunk-size 500 --overlap 100
```

### 启动服务

使用以下命令启动 FastAPI 服务：

```bash
uvicorn src.app.main:app --reload
```

服务启动后，你可以通过 `http://localhost:8000` 访问 API。

## 测试

使用以下命令运行项目测试：

```bash
python -m pytest
```

或者运行特定测试：

```bash
python -m pytest tests/test_retriever.py -v
```

## 环境信息

### 查看已安装的包

```bash
# 使用 pip
pip list

# 使用 conda
conda list
```

### 导出环境配置

如果你需要导出当前环境的配置，可以执行以下命令：

```bash
conda env export > environment.yml
```

## 常见问题

### 权限问题

如果在安装包时遇到权限问题，可以尝试使用 `--user` 参数：

```bash
pip install --user 包名
```

### 网络问题

如果在安装依赖时遇到网络问题，可以尝试更换镜像源：

```bash
# 临时使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 环境冲突

如果你遇到环境冲突问题，可以尝试创建一个全新的环境：

```bash
# 删除旧环境
conda remove -n multimodal_rag --all

# 创建新环境
conda env create -f environment.yml
```
