# RAG-Anything 复现

历史测试：

全文解析（2026-10-09，对应论文 Figure 1 左侧的“多模态知识统一”）：已通过 `reproduce.py parse` 调用 MinerU，在 CPU 上解析 18 页论文，将内容整理为带有类型和页码的结构化列表，共 149 个内容块（正文 122、图片 15、表格 6、公式 6）；提取了正文、图片裁剪及图注、表格结构与内容、公式的 LaTeX 表达，并将图表类对象统一交给图片处理流程。原始解析列表、用于建库的规范化列表、全文 Markdown 和 27 张图片／表格／公式裁剪图保存在 `parsed/rag-anything-handson-v1/`；随后通过 `inspect` 核对源 PDF、内容列表和裁剪图的路径及校验值，检查通过，完成建库前的材料准备。详见 [全文解析记录](parsed/rag-anything-handson-v1/manifest.json)。

全文建库（2026-10-09，对应 Figure 1 中部的“双图构建”和“向量数据库”）：已通过 `reproduce.py ingest` 读取上述解析列表，使用 DeepSeek `deepseek-flash` 从正文中抽取实体及关系，并对图片、表格和公式生成描述、提取相关知识，将正文知识与跨模态知识按实体对齐合并为同一知识图谱；同时使用本机 Ollama `qwen3-embedding:latest` 将内容块、实体和关系编码为 4096 维向量，保存到 `storage/handson-qwen4096-v1/`。建库后共保存 37 个检索内容块（正文重新分块后 10 个、多模态描述 27 个）、644 个图谱节点和 1271 条边；27 个非文本对象全部入库，无缺失对象或失败模型调用，知识库状态为 `ready`。详见 [建库状态](https://github.com/Dairenwen/RAG-Anything-reproduction/blob/2dd2aa92e023879864f2d638b265b43f17209532/storage/handson-qwen4096-v1/pilot_manifest.json) 和 [入库覆盖检查](https://github.com/Dairenwen/RAG-Anything-reproduction/blob/2dd2aa92e023879864f2d638b265b43f17209532/storage/handson-qwen4096-v1/rag-anything-handson-v1-ingest-audit.json)。

全文检索（2026-10-09，对应 Figure 1 右侧的“结构化知识导航”“语义相似度匹配”及答案生成）：已通过 `reproduce.py query` 重新读取 `handson-qwen4096-v1`，先使用 `--mode mix --context-only` 检查检索证据，最终上下文包含 34 个实体、89 条关系和 3 个内容块；随后去掉 `--context-only`，由 DeepSeek 根据证据生成回答。在“原论文实验使用哪种嵌入模型及维度”的同题对比中，[naive](runs/20261009_212549_75e63c97/result.json) 仅检索内容块，最终上下文缺少答案，因证据不足拒答；[hybrid](runs/20261009_212554_53fc1f07/result.json) 和 [mix](runs/20261009_171042_011bbe28/result.json) 均通过图谱证据正确回答 `text-embedding-3-large`、3072 维，三种模式均正常运行。另通过 `--mode mix --vlm` 对 Figure 1 提问，实际向视觉模型传入 1 张经校验确认的原图，正确识别合并前的“文本知识图”和“跨模态知识图”，详见 [视觉问答结果](runs/20261009_211747_4fd92f8d/result.json)。

全文测试结果（2026-10-09）已完成三种模式各 20 题的文字批量问答、4 道视觉题及两次 q04 检索诊断，均正常结束，没有记录到失败的模型调用。知识库为 `handson-qwen4096-v1`，文档为 `rag-anything-handson-v1`；文字与视觉使用 DeepSeek `deepseek-flash`，向量使用本机 Ollama `qwen3-embedding:latest`、4096 维。

三轮使用相同问题文件、模型、知识库和检索参数：`TOP_K=20`、`CHUNK_TOP_K=10`、`MAX_ENTITY_TOKENS=4000`、`MAX_RELATION_TOKENS=4000`、`MAX_TOTAL_TOKENS=12000`，重排序关闭，未启用 `--vlm`。


| 模式与运行记录                                            | 正常完成 | 初核正确     | 累计耗时 | 平均每题耗时 | 记录的总 tokens | 未答出的可回答题   |
| --------------------------------------------------------- | -------- | ------------ | -------- | ------------ | --------------- | ------------------ |
| [naive](runs/20261009_215205_261e8f02/run-manifest.json)  | 20/20    | 16/20（80%） | 22.51 秒 | 1.13 秒      | 206350          | q01、q03、q04、q10 |
| [hybrid](runs/20261009_223219_236062b3/run-manifest.json) | 20/20    | 18/20（90%） | 64.86 秒 | 3.24 秒      | 252807          | q04、q10           |
| [mix](runs/20261009_214201_2266b64a/run-manifest.json)    | 20/20    | 18/20（90%） | 63.79 秒 | 3.19 秒      | 253821          | q04、q10           |

在这套问题中，图谱证据帮助 `hybrid` 和 `mix` 多答出了 q01（原论文嵌入模型及维度）和 q03（主模型及重排模型）；`mix` 没有增加正确题数。共同遗漏的是 q04 的实验设置正文和 q10 的 Figure 1 节点图例。q19、q20 询问论文未给出的具体参数，拒答符合参考答案。

**视觉问答：4 道 Figure 1 问题**

[视觉批量运行记录](runs/20261009_214919_8b48a998/run-manifest.json)：使用 `mix --vlm`，4/4 正常完成，无失败模型调用；累计耗时 17.79 秒，平均每题 4.45 秒，记录的总 tokens 为 53042。


| 题目与逐题结果                                                                          | 图片输入数 | 实际传入图片         | 初核答案       | 证据情况                                |
| --------------------------------------------------------------------------------------- | ---------- | -------------------- | -------------- | --------------------------------------- |
| [q09：合并前的两个知识图](runs/20261009_214919_8b48a998/q09-result.json)                | 1          | Figure 1             | 正确           | 目标图片命中                            |
| [q10：ParentNode、ChildNode、MergedNode](runs/20261009_214919_8b48a998/q10-result.json) | 2          | Figure 3、Figure 4   | 证据不足，拒答 | 未选中 Figure 1，未改善文字问答中的失败 |
| [q11：两种检索机制](runs/20261009_214919_8b48a998/q11-result.json)                      | 1          | 第 17 页的失败案例图 | 正确           | 文字证据已包含答案，目标图片未命中      |
| [q12：合并前的两个向量库](runs/20261009_214919_8b48a998/q12-result.json)                | 1          | Figure 1             | 正确           | 目标图片命中                            |

答案初核为 3/4（75%）；按题目计算，目标图片命中为 2/4（50%）。`--vlm` 会发送检索选中的可用图片

**q04：检索参数调整记录**

原配置为候选 10 块，总预算 12000 tokens，使用相同问题进行两次 `--context-only` 诊断：


| 检索记录                                                      | 候选内容块数 | 总 token 上限 | 实体／关系各自 token 上限 | 最终内容块数 | 答案证据                   |
| ------------------------------------------------------------- | ------------ | ------------- | ------------------------- | ------------ | -------------------------- |
| [扩大候选与上下文](runs/20261009_223704_b870b941/result.json) | 20           | 24000         | 4000／4000                | 13           | 未保留答案正文             |
| [缩减图谱占用](runs/20261009_224310_b8e7e9ba/result.json)     | 20           | 24000         | 1000／1000                | 19           | 第 16 个内容块包含完整答案 |

- 第一次：CHUNK_TOP_K 从 10 增到 20，让排名第 14 的答案正文进入候选；总预算增到 24000，但最终只留下 13 块，答案仍被裁掉。
- 第二次：实体、关系预算各从 4000 降到 1000，让图谱材料占用减少。最终留下 19 块，答案正文保留在第 16 块。

**重排对照测试（2026-10-10）**

本轮补上原论文使用的 `BAAI/bge-reranker-v2-m3`，在本机 Apple GPU（MPS）上运行。重排的作用是：把问题和每个候选内容块放在一起打分，把更相关的材料排到前面，再按数量和长度限制保留材料。文字与视觉仍使用 DeepSeek，向量仍使用 Ollama。

文字组使用 [14 道补充题](question/questions.supplement.v1.jsonl)，包含 q04、q10 两道旧失败题，以及正文、公式、图表细节、跨表比较和不可回答题；视觉组使用其中 [6 道题](question/questions.supplement.v1.visual.jsonl)。关闭和开启重排各运行一次，共 40 次问答、14 道不同题目，均正常完成，没有失败的模型调用。参考答案见 [补充题核对材料](question/evaluation.supplement.v1.jsonl)，仍为待人工确认的草稿。

两组均使用 `mix`，参数统一为 `TOP_K=20`、`CHUNK_TOP_K=20`、`MAX_ENTITY_TOKENS=4000`、`MAX_RELATION_TOKENS=4000`、`MAX_TOTAL_TOKENS=24000`，同一知识库、同一模型和同一问题文件。本轮重排只调整顺序，没有按分数额外筛除候选；BGE 使用 float32、每批 1 对文本、输入上限 8192 tokens。本轮的问题和预算与 2026-10-09 的 20 题测试不同，应在下面的开启／关闭组之间比较。


| 条件与运行记录                                                    | 正常完成 | 核心答案初核正确 | 累计耗时  | 平均每题耗时 | 记录的总 tokens |
| ----------------------------------------------------------------- | -------- | ---------------- | --------- | ------------ | --------------- |
| [文字，关闭重排](runs/20261010_101958_b3bea029/run-manifest.json) | 14/14    | 9/14（64.3%）    | 53.05 秒  | 3.79 秒      | 341500          |
| [文字，开启重排](runs/20261010_102052_ee917d5e/run-manifest.json) | 14/14    | 10/14（71.4%）   | 130.67 秒 | 9.33 秒      | 341152          |
| [视觉，关闭重排](runs/20261010_102308_33f856c8/run-manifest.json) | 6/6      | 4/6（66.7%）     | 25.08 秒  | 4.18 秒      | 157873          |
| [视觉，开启重排](runs/20261010_102334_bad72aae/run-manifest.json) | 6/6      | 4/6（66.7%）     | 62.34 秒  | 10.39 秒     | 156879          |

“核心答案正确”指题目要求的信息都答全、答对，部分正确或前后矛盾不算正确。开启重排的文字 q23 虽然答对了两个数值，却额外错误地说 RAGAnything 排第二；若附加判断出错也扣分，本轮文字两组均为 **9/14**。因此，重排救回了一题，但完整答案的正确数没有净增加，视觉题的正确数也没有增加。

当前脚本每题先做一次证据检查，再为生成答案检索一次，因此开启组每题实际重排两次：文字组共 28 次、视觉组共 12 次。上述耗时包含两次处理，不含运行开始时加载模型的时间，也不能直接作为框架生产环境的单次查询延迟。tokens 是程序记录的模型用量，不是 DeepSeek 账单。本轮没有发现 BGE 输入超过 8192 tokens 而被截断的情况。

## 确认环境

```bash
cd "/Users/drw/PycharmProjects/RAG-Anything-reproduction"
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip check
python reproduce.py check
```


|  |  |  |
| - | - | - |
|  |  |  |

## 项目目录

RAG-Anything-reproduction/
├── README.md                              # 当前目录、现有库及日常使用说明
├── reproduce.py                           # 解析／建库／检索入口，含视觉图片路径兼容修复
├── api.py                                 # 保留的本地检索 API，源码未改
├── requirements.txt                       # 固定官方组件版本及提交
├── requirements-service.txt               # 可选 API 依赖
├── requirements.lock.macos-arm64-py312.txt  # 本机依赖版本记录
├── LICENSE
├── .gitignore
├── .env                                   # 本地模型地址与密钥，保留且不公开
├── .venv/                                 # Python 环境、官方核心包和运行依赖
│   └── lib/python3.12/site-packages/
│       ├── raganything/                   # 官方 RAG-Anything 核心
│       ├── lightrag/                      # 官方 LightRAG 核心
│       └── mineru/                        # 官方 MinerU 核心
├── .idea/                                 # 保留的 PyCharm 项目设置
├── 动手复现步骤.md                         # 现有知识库检索与后续建库步骤
├── data/                                  # 两份原始输入，均保留
│   ├── RAG-Anything_arXiv_2025.pdf
│   └── MCP服务.pdf
├── question/                              # 题目及对应参考答案，统一保存 JSONL 文件
├── parsed/                                # 正式解析资源，检索仍依赖这些路径
│   ├── rag-anything-handson-v1/
│   └── rag-anything-handson-v2/
├── storage/                               # 两份可继续检索的完整知识库
│   ├── handson-qwen4096-v1/
│   └── handson-qwen4096-v2/
├── models/
│   └── bge-reranker-v2-m3/                 # 本地重排模型，保留供后续检索使用
├── reports/                               # 保留的报告全部为 Markdown
│   ├── 建库与解析测试/                     # 耗时、全流程对比、调用配置等报告
│   ├── 解析输出MD/                        # 三次测试解析留下的 Markdown；图片引用正式资源
│   ├── 历史文档快照/                       # 原测试留下的 Markdown 文档快照
│   ├── 历史检索测试报告.md                 # 清理前 README 的历史检索结果
│   └── 清理核验/
│       ├── 官方核心代码核验.md
│       ├── 清理与迁移记录.md
│       └── 已删除文件清单.md
└── runs/                                  # 当前为空；后续检索会自动生成新结果

每个正式解析目录包含 `manifest.json`、`content-list.json`、`parser-content-list.json` 和 MinerU 输出。源 PDF、清单及 `mineru/.../images/` 是后续校验和视觉检索使用的资源，应保持原路径。

每个知识库包含 `pilot_manifest.json`、`graph_chunk_entity_relation.graphml`、三个 `vdb_*.json`、`kv_store_*.json`、入库覆盖检查和模型用量记录。

## 命令

### 1. 环境准备与查看帮助

日常使用先进入项目并激活现有环境：

```bash
cd "/Users/drw/PycharmProjects/RAG-Anything-reproduction"
source .venv/bin/activate
python -m pip check
python reproduce.py check
```

`pip check` 检查依赖冲突；`check` 检查组件版本、固定源码提交和 MinerU 命令。这两项检查不调用模型，也不建库。

仅在新机器、尚未创建环境时使用以下安装命令；本机现有环境无需重复创建：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

查看全部阶段及某个阶段的选项：

```bash
python reproduce.py --help
python reproduce.py parse --help
python reproduce.py ingest --help
python reproduce.py query --help
python reproduce.py batch --help
```

`--help` 只显示帮助，不执行解析、建库或模型请求。退出虚拟环境使用 `deactivate`。

### 2. 模型配置与连通性检查

模型配置从项目根目录的 `.env` 读取。也可以在命令前临时指定环境变量；命令前的值优先于 `.env`，只对这次命令生效。


| 配置项                                                       | 作用                                                                                   |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `TEXT_BASE_URL`、`TEXT_MODEL`、`TEXT_API_KEY`                | 文本模型的接口地址、模型名和密钥，用于实体关系提取、关键词提取及文字回答               |
| `VISION_BASE_URL`、`VISION_MODEL`、`VISION_API_KEY`          | 视觉模型的接口地址、模型名和密钥，用于图像描述及启用视觉增强后的图片问答               |
| `EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_API_KEY` | 嵌入模型的接口地址、模型名和密钥，用于建库及查询向量；本机 Ollama 的密钥可填写`ollama` |
| `EMBEDDING_DIM`                                              | 向量维度；当前两份库均为`4096`，查询必须保持一致                                       |
| `EMBEDDING_SEND_DIMENSIONS`                                  | 是否向嵌入接口发送`dimensions` 参数，脚本默认 `false`；查询须与建库记录一致            |

视觉接口地址、密钥留空时会使用文本接口地址、密钥；视觉模型名应按实际服务单独配置。文本与视觉使用同一模型时填写相同配置，使用不同模型时分别填写。模型名必须是对应服务实际提供的名称。

分别检查文本和嵌入接口：

```bash
python reproduce.py probe --role text
python reproduce.py probe --role embedding
```

检查视觉接口时必须提供真实图片。下面使用第一篇论文现有的 Figure 1 裁剪图：

```bash
PAPER_IMAGE="$PWD/parsed/rag-anything-handson-v1/mineru/RAG-Anything_arXiv_2025_4ae5f35e/RAG-Anything_arXiv_2025/auto/images/d960511e87edf887a6ee67caba7d4f7cb9da80f3fe7cdea264ab856390831d14.jpg"
python reproduce.py probe --role vision --image "$PAPER_IMAGE"

# 在已设置 PAPER_IMAGE 的同一终端中，检查三类接口。
python reproduce.py probe --role all --image "$PAPER_IMAGE"
```


| 选项               | 解释                                                         |
| ------------------ | ------------------------------------------------------------ |
| `--role text`      | 检查文字生成；针对官方 DeepSeek 地址还会检查关键词 JSON 输出 |
| `--role embedding` | 检查两段文本的向量形状、数值及维度是否符合配置               |
| `--role vision`    | 检查图片输入与视觉回答，必须同时指定`--image`                |
| `--role all`       | 默认值，检查三类接口；未提供图片时视觉检查会被跳过           |
| `--image`          | 待测试图片的路径，支持 PNG、JPEG、WebP、GIF                  |

`probe` 会实际调用模型，并将结果保存到 `runs/probes/`，不会建立知识库。

### 3. 解析新文档

将新 PDF 放入 `data/`，再执行解析。以下 `新文档.pdf` 是示例文件名，需要替换成实际文件；新文档标识也应选择尚未使用的名称。

```bash
python reproduce.py parse \
  --pdf "data/新文档.pdf" \
  --doc-id new-paper-v1 \
  --device cpu \
  --timeout 1800
```


| 选项        | 解释                                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| `--pdf`     | 必填，源 PDF 路径；相对路径按当前终端所在目录解析                                                                  |
| `--doc-id`  | 必填，解析文档标识，输出到`parsed/<doc-id>/`；仅使用字母、数字、下划线或连字符，首字符为字母或数字，最长 80 个字符 |
| `--device`  | MinerU 解析设备，默认`cpu`；脚本还接受 `mps`、`cuda:0`，实际能否使用取决于本机环境与解析后端                       |
| `--timeout` | MinerU 解析超时，单位秒，默认`1800`                                                                                |

### 4. 检查解析材料

新文档解析后先检查，再建库：

```bash
python reproduce.py inspect --doc-id new-paper-v1
```

### 5. 四类内容的小样本检查（可选）

`sample` 专门从 RAG-Anything 论文中选取实验设置正文、Figure 1、Table 1、Equation 1；它不是任意文档的通用抽样工具。准备独立样本后，`smoke` 检查小样本建库、关闭并重新读取库，以及多种检索和问答。

```bash
python reproduce.py sample \
  --source-doc-id rag-anything-handson-v1 \
  --doc-id rag-anything-sample-demo

python reproduce.py smoke \
  --doc-id rag-anything-sample-demo \
  --kb rag-anything-sample-demo-kb \
  --timeout 900
```


| 命令／选项               | 解释                                                                              |
| ------------------------ | --------------------------------------------------------------------------------- |
| `sample --source-doc-id` | 原论文的完整解析标识；脚本默认`rag-anything-v1`，当前项目需显式使用上面的正式标识 |
| `sample --doc-id`        | 新的小样本解析标识；脚本默认`rag-anything-smoke-v1`                               |
| `smoke --doc-id`         | 已由`sample` 准备的小样本标识；脚本默认 `rag-anything-smoke-v1`                   |
| `smoke --kb`             | 独立小样本库名称；脚本默认`smoke-qwen4096-v1`                                     |
| `smoke --timeout`        | 小样本建库及每项查询的超时，单位秒，默认`900`                                     |

`sample` 不调用模型；`smoke` 会调用文本、视觉、嵌入模型并创建独立测试库。日常查询已有完整库可以跳过本阶段。

### 6. 构建新知识库

解析检查通过后，将新文档加入新库：

```bash
python reproduce.py ingest \
  --doc-id new-paper-v1 \
  --kb new-paper-kb-v1 \
  --timeout 3600
```


| 选项        | 解释                                                                                    |
| ----------- | --------------------------------------------------------------------------------------- |
| `--doc-id`  | 必填，选择已经检查通过的`parsed/<doc-id>/`                                              |
| `--kb`      | 知识库名称，输出到`storage/<kb>/`；默认 `pilot-v1`，建议显式指定；命名规则同 `--doc-id` |
| `--timeout` | 本次内容入库等待时间，单位秒，默认`3600`；不是单次模型请求的超时                        |

`ingest` 使用文本模型提取知识、视觉模型处理需要图片输入的对象、嵌入模型生成向量，由官方框架建立图谱和索引；不会再次解析 PDF。脚本同时检查多模态对象覆盖及失败调用，成功后库状态为 `ready`。

### 7. 单题检索与问答

只查看第一篇论文的检索证据：

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 python reproduce.py query \
  --kb handson-qwen4096-v1 \
  --mode mix \
  --context-only \
  --question "原论文实验使用哪种嵌入模型及维度？"
```

生成文字回答：

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 python reproduce.py query \
  --kb handson-qwen4096-v1 \
  --mode mix \
  --question "RAG-Anything 如何结合知识图谱与向量检索？"
```

启用视觉增强，查看检索选中的图片：

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 python reproduce.py query \
  --kb handson-qwen4096-v1 \
  --mode mix \
  --vlm \
  --question "Figure 1 中合并前的两个知识图分别是什么？"
```

2026-10-11 修复了 macOS 目录名大小写不同导致原图被安全检查拦截的问题：项目入口从知识库清单读取图片目录，确认图片位于本项目解析目录且校验值一致后，通过官方 `extra_safe_dirs` 参数传入原图目录的已存路径拼写。无需重建库。若旧运行出现 `Blocking image path outside safe directories` 且图片输入数为 0，该次视觉问答实际退回了文字查询，应在修复后重新运行。

启用本地 BGE 重排：

```bash
python reproduce.py rerank-check --device mps

EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 python reproduce.py query \
  --kb handson-qwen4096-v1 \
  --mode mix \
  --rerank \
  --question "论文的实验设置有哪些？"
```

`rerank-check` 只用本地模型做相关性检查，不读取知识库或调用模型 API；`--device` 可填 `auto`、`mps`、`cpu`，默认按配置选择。该选项只作用于这次检查；正式查询的重排设备由 `RERANK_DEVICE` 配置。

查询第二份文档时显式使用它记录的地址：

```bash
EMBEDDING_BASE_URL=http://localhost:11434/v1 python reproduce.py query \
  --kb handson-qwen4096-v2 \
  --mode mix \
  --question "这份文档介绍了什么？"
```


| 选项             | 解释                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `--kb`           | 选择`storage/<kb>/`，默认 `pilot-v1`；查询当前两份库需显式指定                                          |
| `--question`     | 必填，问题文字，包含空格时用引号包住                                                                    |
| `--mode naive`   | 主要通过内容块向量检索                                                                                  |
| `--mode hybrid`  | 结合实体及关系的图谱检索                                                                                |
| `--mode mix`     | 结合图谱检索与内容块向量检索，默认模式                                                                  |
| `--context-only` | 返回结构化检索证据，不生成最终回答；检索仍可能调用关键词及嵌入模型                                      |
| `--vlm`          | 生成答案时允许使用检索选中的可用图片，不保证一定命中目标图；与`--context-only` 同用时不执行视觉答案生成 |
| `--rerank`       | 为本次查询开启本地 BGE 重排，需要现有`models/bge-reranker-v2-m3/` 权重                                  |
| `--no-rerank`    | 为本次查询关闭重排，优先于`.env` 中的 `ENABLE_RERANK`                                                   |
| `--timeout`      | 每题等待时间，单位秒，默认`300`                                                                         |

不指定 `--rerank` 或 `--no-rerank` 时遵循 `.env` 配置。每次查询写入新的 `runs/<时间_标识>/`：`result.json` 保存答案、证据、计时及调用记录，`run-manifest.json` 保存运行配置；失败时另存 `error.json`。

### 8. 批量问答

用已补回的 14 道文字题查询第一篇现有库：

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 python reproduce.py batch \
  --kb handson-qwen4096-v1 \
  --questions question/questions.supplement.v1.jsonl \
  --mode mix \
  --no-rerank \
  --timeout 300
```

用 6 道视觉题执行视觉问答：

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 python reproduce.py batch \
  --kb handson-qwen4096-v1 \
  --questions question/questions.supplement.v1.visual.jsonl \
  --mode mix \
  --vlm \
  --no-rerank
```

`batch` 用 `--questions` 指定 JSONL 问题文件，其余查询选项与 `query` 相同，`--timeout` 按每题计算。问题按文件顺序逐题执行，不是整批并发。

自建问题文件统一保存到 `question/`，例如 `question/my-questions.jsonl`。每行一个 JSON 对象，例如：

```jsonl
{"id":"q01","doc_id":"rag-anything-handson-v1","question":"论文研究的问题是什么？"}
{"id":"q02","doc_id":"rag-anything-handson-v1","question":"框架支持哪些内容类型？"}
```

`id` 必须唯一并符合标识命名规则；`doc_id` 必须属于所选知识库；`question` 不能为空。批量输出位于新的 `runs/<时间_标识>/`，包括每题的 `<id>-result.json` 和整批 `run-manifest.json`。批量执行不会自动按参考答案评分。

### 9. 调整常用参数

下面是脚本未被 `.env` 或命令前环境变量覆盖时的默认值。模型配置和这些参数均通过环境变量读取，不是 `--top-k` 一类命令行选项。


| 环境变量              | 脚本默认值                   | 解释                                                           |
| --------------------- | ---------------------------- | -------------------------------------------------------------- |
| `CHUNK_SIZE`          | `1200`                       | 建库正文分块的 token 长度                                      |
| `CHUNK_OVERLAP`       | `100`                        | 相邻正文块重叠的 token 数，必须小于块长                        |
| `SUMMARY_LANGUAGE`    | `English`                    | 建库知识摘要语言                                               |
| `LLM_CONCURRENCY`     | `2`                          | 传给 LightRAG 的文本模型异步任务上限；不代表配置了两个不同模型 |
| `TOP_K`               | `20`                         | 图谱检索的实体、关系候选数量参数                               |
| `CHUNK_TOP_K`         | `10`                         | 内容块候选数量参数                                             |
| `MAX_ENTITY_TOKENS`   | `4000`                       | 检索上下文中实体材料的 token 预算                              |
| `MAX_RELATION_TOKENS` | `4000`                       | 检索上下文中关系材料的 token 预算                              |
| `MAX_TOTAL_TOKENS`    | `12000`                      | 检索上下文的总 token 预算，不是模型输出长度上限                |
| `ENABLE_RERANK`       | `false`                      | 默认是否启用重排，可被`--rerank`、`--no-rerank` 覆盖           |
| `RERANK_DEVICE`       | `auto`                       | 重排设备；可用`auto`、`mps`、`cpu`                             |
| `RERANK_MAX_LENGTH`   | `8192`                       | 重排模型每对“问题＋材料”的最大输入 token 长度                |
| `RERANK_BATCH_SIZE`   | `1`                          | 本地重排每批处理的文本对数量                                   |
| `RERANK_MODEL_PATH`   | `models/bge-reranker-v2-m3/` | 本地重排权重目录                                               |

扩大内容候选及总预算，同时减少实体、关系各自占用的诊断示例：

```bash
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 \
CHUNK_TOP_K=20 \
MAX_TOTAL_TOKENS=24000 \
MAX_ENTITY_TOKENS=1000 \
MAX_RELATION_TOKENS=1000 \
python reproduce.py query \
  --kb handson-qwen4096-v1 \
  --mode mix \
  --context-only \
  --question "论文的实验设置有哪些？"
```

查询参数可以按题调整；嵌入模型、地址、维度及 `EMBEDDING_SEND_DIMENSIONS` 必须保持与所选库一致。建库分块与模型配置变化后，不应继续向原库混入新文档。

### 10. 启动本地查询 API（可选）

需要通过网页或其他程序调用时，安装接口依赖并启动服务：

```bash
python -m pip install -r requirements-service.txt

PILOT_KB=handson-qwen4096-v1 \
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1 \
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```


| 参数／配置         | 解释                                                                   |
| ------------------ | ---------------------------------------------------------------------- |
| `PILOT_KB`         | API 启动时加载的库名，默认`pilot-v1`；第一篇使用 `handson-qwen4096-v1` |
| `--host 127.0.0.1` | 只监听本机                                                             |
| `--port 8000`      | 接口端口，可改为其他空闲端口                                           |
| `api:app`          | 加载`api.py` 中的 FastAPI 应用                                         |

保持服务终端运行，在另一个终端检查状态并提问：

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"论文研究的问题是什么？","mode":"mix","context_only":false,"vlm":false}'
```

请求中的 `question` 必填，长度为 1～2000 个字符且不能全为空白；`mode` 可填 `naive`、`hybrid`、`mix`，默认 `mix`；`context_only`、`vlm` 均默认 `false`，含义与命令行对应选项一致。API 当前没有请求级重排开关，需在启动服务前配置 `ENABLE_RERANK`。每次请求固定等待上限为 300 秒，并在 `runs/api_<请求标识>/` 保存记录。

服务运行后可以在浏览器打开 `http://127.0.0.1:8000/docs` 查看交互式接口文档。当前服务按单进程串行处理查询，不加 `--workers`；停止服务使用 `Ctrl+C`。切换知识库或模型配置后，需要重新启动服务。
