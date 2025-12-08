# Accelerated Inference Project

本项目旨在通过 KV Cache 压缩技术加速 Pythia-70M 模型的推理过程。项目包含自定义的 `kvpress` 库用于实现各种压缩算法（Presses），以及用于评估模型困惑度（PPL）、推理速度和内存占用的脚本。

## 目录结构

- **`kvpress/`**: 核心库，包含 KV Cache 压缩的实现。
  - `base_press.py`: 定义了压缩算法的基类 `BasePress` 和 Hook 机制。
  - `presses/`: 具体的压缩算法实现，例如 `KnormPress`。
- **`evaluate/`**: 评估脚本和 Notebook。
  - `ppl.py` / `ppl.ipynb`: 用于在 Wikitext-2 and PG19 数据集上计算 Pythia-70M 的困惑度。
  - `speed_amd_memory.py`: 用于测试不同上下文长度下的推理速度和内存峰值。
- **`attention/`**: 包含对注意力机制的特定补丁或修改。
- **`dataset/`**: 用于存放下载的数据集（如 Wikitext-2, PG19）。

## 环境准备

请确保安装了以下 Python 依赖库：

```bash
pip install torch transformers datasets tqdm matplotlib
```

此外，你需要下载 `wikitext-2-raw-v1` 和 `pg19` 数据集并放置在 `dataset/` 目录下，或者调整评估脚本中的路径。

## 使用说明

### 1. 评估模型困惑度 (PPL)

运行 `evaluate/ppl.py` 脚本来评估原始模型的性能：

```bash
python evaluate/ppl.py
```

或者使用 `evaluate/ppl.ipynb` Notebook 进行交互式运行和分析。

### 2. 速度与内存基准测试

运行 `evaluate/speed_amd_memory.py` 脚本来测试推理性能：

```bash
python evaluate/speed_amd_memory.py
```

测试结果（图表）将保存在 `plots/` 目录下。

### 3. 使用 KVPress 进行加速

`kvpress` 库通过 Hook 机制集成到 Hugging Face Transformers 模型中。基本用法如下：

```python
from transformers import AutoModelForCausalLM
from kvpress import KnormPress

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

# 初始化压缩器
press = KnormPress()

# 使用上下文管理器应用压缩
with press(model):
    # 在此处进行的模型推理将自动应用 KV Cache 压缩
    output = model.generate(...)
```

要实现新的压缩算法，请继承 `kvpress.base_press.BasePress` 并实现 `compress` 方法。
