# RAG-Anything 工程复现启动项目

[完整实施指南](../../../RAG-Anything_工程复现与研究改进指南.md)

状态：依赖安装及一致性检查、13 项离线检查、1 项真实框架集成检查、原论文 CPU 解析已通过。集成使用模型替身；真实模型接口、正式建库和问答质量待验证。详见 [现存核验汇总](../../../verification.handson.json)。（早期 `verification.json` 已清理；此链接提供后续核验汇总。）

## 从这里开始

```bash
cd "/Users/drw/VscodeProjects/rag-anything-reproduction"
source .venv/bin/activate
python -m pip check
python reproduce.py check
python reproduce.py inspect --doc-id rag-anything-v1
```

本机环境与解析已准备。按指南第 4 节填写本地 `.env`，验证文字、embedding 及真实图片接口，再进行入库和查询。新机器首次安装步骤见指南。

## 命令说明


| 命令                   | 作用                                       | 外部模型 API                  |
| ---------------------- | ------------------------------------------ | ----------------------------- |
| `check`                | 检查包版本和解析命令                       | 不调用                        |
| `probe`                | 验证文字、embedding，可选择图片探针        | 调用                          |
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
