# RAG-Anything 动手复现项目

[按顺序动手复现](../动手复现步骤.md) · [前一轮方案与历史核验](../RAG-Anything_工程复现与研究改进指南.md)

当前组合：DeepSeek 官方 `deepseek-flash` 处理文字与图片；本机 Ollama `qwen3-embedding:latest` 输出 4096 维向量。原论文实验模型与本轮选型不同。

已通过：环境与依赖检查、现有解析完整性检查、19 项离线测试及 1 项框架集成测试、真实文字/关键词/向量/图片接口探针、四类真实论文内容的小样本建库与重启查询。视觉问答实际传入 2 张图片。完整论文重新解析、正式建库及 20 题问答评测由你按步骤执行。详见 [本轮核验记录](../verification.handson.json)。

## 从这里开始

```bash
cd "/Users/drw/PycharmProjects/rag-anything-reproduction"
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip check
python reproduce.py check
python reproduce.py inspect --doc-id rag-anything-v1
```

确认终端与 PyCharm 都使用当前项目的 `.venv/bin/python`。本轮 `.env` 已配置；Key 只存放在本地。接下来按动手指南逐步使用 `rag-anything-handson-v1` 和 `handson-qwen4096-v1` 完成全文流程。保留已有解析与知识库。

## 命令说明


| 命令                   | 作用                                       | 外部模型 API                  |
| ---------------------- | ------------------------------------------ | ----------------------------- |
| `check`                | 检查包版本和解析命令                       | 不调用                        |
| `probe`                | 验证文字、关键词、embedding，可选择图片；支持 `--role` | 文字/视觉调用，向量在本地 |
| `sample`               | 从论文现有解析中选取四类内容               | 不调用                        |
| `smoke`                | 真实模型小样本建库、重启、检索与视觉回答   | 调用                          |
| `parse`                | 本地解析 PDF，首次可能下载解析模型         | 不调用已配置的文字/视觉 API   |
| `inspect`              | 核对解析内容、源 PDF 和图片哈希            | 不调用                        |
| `ingest`               | 从已检查的解析结果建库                     | 调用                          |
| `query --context-only` | 结构化检索，不生成答案                     | 关键词与 embedding 仍可能调用 |
| `query`                | 诊断检索与文字生成                         | 调用                          |
| `query --vlm`          | 诊断检索与视觉增强生成                     | 调用                          |
| `batch`                | 串行运行问题文件，保存逐题结果，异常时停止 | 调用                          |

初始方案关闭重排序，使用小规模预算和本地存储。模型和依赖与原文的差异在指南中单列。

```bash
python reproduce.py --help
python -m unittest -v test_offline.py
python -m unittest -v test_integration.py
```

测试不调用模型 API，临时测试记录保留在 `runs/offline-tests/`，不递归删除文件或目录。
