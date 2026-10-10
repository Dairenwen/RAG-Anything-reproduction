# RAG-Anything 工程复现方案与验证记录

**本轮操作入口：** [动手复现步骤](/Users/drw/PycharmProjects/rag-anything-reproduction/动手复现步骤.md)。当前选用 DeepSeek 官方文字/视觉接口与本机 Ollama Qwen3 4096 维向量，已修复迁移后的环境与解析路径；下文保留前一轮方案及历史核验，最新运行证据以本轮核验目录为准。

本文以论文 Figure 1 为主线，将 RAG-Anything 落地为可检查、可重启、可评测的本地系统。第一轮使用一篇公开 PDF 和 20 道诊断题，先验证解析、建库、检索与视觉回引，再开展研究改进。

**完成边界：依赖安装、真实 PDF 解析和使用替身模型的工程集成已通过；真实模型接口、正式入库与问答质量尚未验证。** 本文是工程复现方案及阶段性验证记录，后续需以真实运行记录补全结果。

## Material Passport


| 字段                | 内容                                                        |
| ------------------- | ----------------------------------------------------------- |
| Origin Skill / Mode | academic-research-suite / experiment-agent；plan + 工程核验 |
| Origin Date         | 2026-10-08，Asia/Shanghai                                   |
| Verification Status | UNVERIFIED：真实模型端到端实验未完成；已验证项目见第 10 节  |
| Version Label       | engineering_v2                                              |
| 原文与代码          | arXiv:2510.12323v1；官方固定提交`8664e8b…`                 |
| 本次贡献            | 启动器、兼容性修正、产物校验、运行记录、诊断题及评测方案    |

## 目录

1. [复现目标与论文框架](#1-复现目标与论文框架)
2. [环境、版本与实验差异](#2-环境版本与实验差异)
3. [工程目录与执行约定](#3-工程目录与执行约定)
4. [环境及模型接口验收](#4-环境及模型接口验收)
5. [PDF 解析与证据验收](#5-pdf-解析与证据验收)
6. [多模态索引构建](#6-多模态索引构建)
7. [检索及视觉生成验证](#7-检索及视觉生成验证)
8. [批量评测与演示](#8-批量评测与演示)
9. [本地 API 与故障处理](#9-本地-api-与故障处理)
10. [本次核验记录与交付状态](#10-本次核验记录与交付状态)
11. [研究改进与论文结果复现](#11-研究改进与论文结果复现)
12. [参考来源](#12-参考来源)

## 1. 复现目标与论文框架

### 1.1 范围及完成条件


| 层次                   | 目标                         | 完成条件                                   |
| ---------------------- | ---------------------------- | ------------------------------------------ |
| 工程复现，本轮重点     | 从 PDF 到带证据的回答        | 真实模型联调、完整入库、重启查询及演示通过 |
| 机制验证，本轮准备     | 理解图、向量和原图各自的作用 | 同题对照、实际输入记录、错误归因齐全       |
| 论文结果复现，后续阶段 | 对齐原文基准结果             | 数据、模型、预算、重排序与判分协议对齐     |

第一轮不要求得到论文总体准确率，也不预设 `mix` 必须优于其他模式。

### 1.2 Figure 1 与工程步骤的对应

```text
原始 PDF
  → MinerU 解析：正文、图像、表格、公式及页码
  → 规范化内容列表，人工核对必要证据
  → 正文抽取实体关系；非文本生成描述与父对象摘要
  → 非文本内部实体通过 belongs_to 连接其父对象
  → 对齐实体，形成统一知识图谱 G
  → 实体、关系、内容块的文字表示编码，形成向量索引 T
  → 查询：图导航 + 语义检索，组装上下文
  → 文字生成；或回引有效原图并进行视觉增强生成
  → 保存来源、实际模型输入、回答、用量与评测结果
```

非文本主要通过**文字代理**进入统一检索空间，原图在生成阶段提供视觉细节。知识图谱保留连接和归属关系；向量索引负责语义匹配，两者共同组成 `I=(G,T)`。当前代码的图入口也使用实体/关系向量检索，因此图通路本身可能调用 embedding。[S1][S2][S3]

论文描述的双图可以融合到同一存储；无需为“两张图”额外部署两个图数据库。第一轮使用 LightRAG 默认体系的本地 JSON、NetworkX 与 NanoVectorDB 存储。

### 1.3 按阶段验收


| 阶段          | 必须留下的证据                                  |
| ------------- | ----------------------------------------------- |
| M0 环境       | 安装版本、官方提交、依赖一致性及测试记录        |
| M1 模型接口   | 文字返回、实际 embedding 维度、真实图片探针     |
| M2 解析       | 内容列表、图片、PDF/解析/图片哈希及人工抽查     |
| M3 索引       | 底层处理状态、多模态覆盖检查、成功落盘与`ready` |
| M4 检索       | 三种模式逐题证据，保留空结果和遗漏              |
| M5 生成       | 实际模型输入、图片数量、回答及重启查询          |
| M6 评测与演示 | 冻结问题集、逐题评分、失败分析及本地 API        |

## 2. 环境、版本与实验差异

### 2.1 已核验环境与依赖

本机：macOS 27.0.1、Apple Silicon `arm64`、48 GiB 内存、Python 3.12.13。解析设备为 CPU。首次解析可能下载模型权重；其他架构及 GPU 部署应独立验收。


| 组件            | 固定版本 / 标识                                              |
| --------------- | ------------------------------------------------------------ |
| RAG-Anything    | `8664e8b318a3ed651fe62dfcfddfb1f1d633d5be`，安装包版本 1.4.2 |
| LightRAG        | `lightrag-hku==1.4.16`                                       |
| MinerU          | `mineru[core]==3.4.1`                                        |
| PDF 兼容依赖    | `pypdfium2==4.30.0`、`pdftext==0.6.3`                        |
| 模型 SDK / 配置 | `openai==2.54.0`、`python-dotenv==1.2.4`                     |
| 可选服务        | `fastapi==0.142.4`、`uvicorn==0.54.0`                        |

必须同时固定 PDF 兼容依赖：MinerU 3.4.1 使用的 `get_pos()` 在 pypdfium2 5.x 被移除，较新 pdftext 又要求 5.x。本次真实运行遇到该错误后，将二者固定到兼容组合，随后 `pip check` 和真实解析通过。[S4][S5]

完整传递依赖保存在 [本机版本快照](/Users/drw/PycharmProjects/rag-anything-reproduction/requirements.lock.macos-arm64-py312.txt)。该快照对应 macOS arm64 / Python 3.12；不能视为跨平台兼容证明。当前 MinerU 4.x 的新命令与本文固定版不同，按本项目依赖和命令执行。

### 2.2 与原文设置的差异


| 项目            | 论文 §3.1                      | 本轮方案                                             |
| --------------- | ------------------------------- | ---------------------------------------------------- |
| 文字 / 视觉模型 | GPT-4o-mini                     | 同名初始候选；需账户探针确认                         |
| Embedding       | text-embedding-3-large，3072 维 | 默认相同；以实际返回维度验收                         |
| 重排序          | bge-reranker-v2-m3              | 关闭，后续单独接入                                   |
| 实体 / 关系预算 | 合计 20,000 tokens              | 各 4,000 tokens 上限                                 |
| 内容块预算      | 12,000 tokens                   | LightRAG 总上下文上限 12,000；含提示词等开销，不等价 |
| 数据 / 评测     | 两个公开基准；模型裁判          | 单篇论文；20 道人工复核的诊断题                      |
| 运行代码        | 原文实验版本待确认              | 固定当前官方提交，并加本地启动器                     |

这些差异进入运行记录，结果应标为“工程基线”，不能直接与原文准确率作同条件比较。

## 3. 工程目录与执行约定

项目：[rag-anything-reproduction](/Users/drw/PycharmProjects/rag-anything-reproduction/README.md)。以下命令均在该目录执行。

```text
rag-anything-reproduction/
├── requirements.txt / requirements-service.txt   固定直接依赖
├── requirements.lock.macos-arm64-py312.txt        本机完整版本快照
├── .env.example / .env                           配置模板 / 本地密钥
├── reproduce.py / api.py                        命令行 / 可选本地服务
├── test_offline.py / test_integration.py          行为 / 工程集成检查
├── questions.pilot.jsonl / questions.visual.jsonl 20 题 / 4 道视觉题
├── evaluation.template.csv                      逐题评测模板
├── parsed/<doc_id>/                             解析及原始图片
├── storage/<kb>/                                持久化索引及建库审计
└── runs/<run_id>/                               配置、问题、结果与错误记录
```

`doc-id` 标识解析版本，`kb` 标识索引版本，`run_id` 标识一次查询或批次。PDF、解析、图片、模型或分块设置改变时，使用新的标识。启动器校验哈希，拒绝重复入库和半完成知识库；CLI 与 API 共用文件锁，防止本项目进程同时占用同一库。

文件锁适用于 macOS/Linux；其他工具若绕过锁访问底层存储，仍可能造成冲突。路径迁移需整体保留解析、图片和索引，并重新校验绝对引用；第一轮更稳妥的做法是在目标位置重新解析建库。

## 4. 环境及模型接口验收

### 4.1 安装或复核环境

本机 `.venv` 已建立。继续操作时执行：

```bash
cd "/Users/drw/PycharmProjects/rag-anything-reproduction"
source .venv/bin/activate
python -m pip check
python reproduce.py check
```

新机器首次安装时执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python reproduce.py check
```

确认 Python 为 3.12。相同平台重建已核验环境时，可改用版本快照安装；安装后仍需运行 `check`。

`check` 不调用模型，核对依赖、`mineru` 命令及安装来源提交。依赖一致性由 `pip check` 验证。集成测试需要先安装服务依赖：

```bash
python -m pip install -r requirements-service.txt
python -m unittest -v test_offline.py test_integration.py
```

集成检查使用固定的模型替身，验证框架接线与持久化；不评价真实模型理解能力。测试产物保留于 `runs/`。

### 4.2 配置三个模型角色

仅首次创建配置文件：

```bash
cp -n .env.example .env
```

在本地编辑 `.env`，填写模型 ID、端点和密钥：


| 角色      | 必填配置                                                                   | 用途                           |
| --------- | -------------------------------------------------------------------------- | ------------------------------ |
| 文字      | `TEXT_API_KEY / TEXT_BASE_URL / TEXT_MODEL`                                | 抽取实体关系、关键词与文字回答 |
| 视觉      | `VISION_API_KEY / VISION_BASE_URL / VISION_MODEL`                          | 图像描述及原图增强回答         |
| Embedding | `EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL / EMBEDDING_DIM` | 编码文档对象和问题             |

视觉与 embedding 的密钥、端点留空时继承文字配置；前提是同一服务确实提供三类能力。密钥不进入笔记或版本库。当前适配 OpenAI-compatible 接口；其他协议、Azure 参数或特殊模型限制需另行适配。

初始值：分块 1,200 tokens、重叠 100、`top_k=20`、`chunk_top_k=10`、LLM 并发 2、重排序关闭。索引语言显式传入底层；实际配置进入 manifest。

### 4.3 运行接口探针

```bash
python reproduce.py probe
```

验收文字返回正常、embedding 实际维度一致。若供应商不支持 `dimensions`，设置 `EMBEDDING_SEND_DIMENSIONS=false`，并填写实际维度。索引与查询必须使用同一 embedding 模型、端点和维度。

该探针不测视觉；第 5 节取得真实图片后再测。所有真实模型请求均产生服务用量，TokenTracker 仅记录可获得的 usage，费用以服务商账单为准。

## 5. PDF 解析与证据验收

### 5.1 解析输入及命令

输入是用户提供的 18 页公开论文，SHA-256：

```text
3e278ab46a87681b4abc1bccf9c78cdf8721c3318ca0d251d620bfc85c54523b
```

从头执行时：

```bash
python reproduce.py parse \
  --pdf "/Users/drw/PycharmProjects/rag-anything-reproduction/data/RAG-Anything_arXiv_2025.pdf" \
  --doc-id rag-anything-v1 --device cpu --timeout 1800
```

**本机该文档已解析完成，不必重复运行。** 本次验证采用 600 秒上限；上面的首次复现命令放宽为 1,800 秒。直接校验现有产物：

```bash
python reproduce.py inspect --doc-id rag-anything-v1
```

重做解析时换新的 `doc-id`。解析使用本地 `pipeline`，不调用配置中的文字/视觉 API；超时是运行上限，不是预计耗时。

### 5.2 产物及规范化

```text
parsed/rag-anything-v1/
├── parser-content-list.json   官方解析器返回列表
├── content-list.json          本地规范化后的入库列表
├── manifest.json              输入、内容、图片哈希及版本
└── mineru/                    原始结果、表格、公式和图片
```

本次返回 122 个正文块、11 个 `image`、4 个 `chart`、6 个表格和 6 个公式，共 149 项。启动器将 `chart` 映射为 `image`，保留 `_pilot_original_type` 和原始列表，使图表进入视觉处理器。入库列表因此包含 **122 个正文、15 个图像、6 个表格、6 个公式**。这是有记录的工程适配。

### 5.3 人工证据检查


| 对象     | PDF 页序 | 核验重点                                       |
| -------- | -------: | ---------------------------------------------- |
| Figure 1 |        4 | 完整流程、双图、VDB 和查询箭头可辨认           |
| Table 2  |        8 | 方法行、Txt./Mm./Una. 与 Overall 列对应正确    |
| Table 4  |        9 | 60.0、62.4、63.4 的 Overall 数值和方法对应正确 |
| 核心公式 |     3–5 | 符号、锚点和`belongs_to` 方向与原文一致        |

本次已检查 Figure 1 裁剪及 Table 2/4 的结构化内容；不能由少量抽查推断全部解析无误。运行评测前，仍需按问题集逐题确认必要证据。`page_idx` 从 0 开始；问题文件的 `pdf_pages_1based` 从 1 开始。[S2]

### 5.4 验证真实视觉接口

从 Figure 1 对应解析块取得 `img_path`，将下面占位符替换为真实路径：

```bash
python reproduce.py probe --image "/替换为解析图片的绝对路径.jpg"
```

检查模型确实描述该图，而非仅返回一段通用文字。通过后再建库。图片文件需保留，后续视觉回引依赖这些路径。

## 6. 多模态索引构建

### 6.1 入库

解析证据及三类接口通过后执行：

```bash
python reproduce.py ingest --doc-id rag-anything-v1 --kb pilot-v1 --timeout 3600
```

启动器先核对 PDF、内容列表及图片哈希，再使用官方 `insert_content_list()`。正文分块抽取实体关系；非文本生成描述与父对象摘要，抽取内部实体并添加归属边，最终写入图和实体/关系/内容块向量索引。[S2]

### 6.2 成功判定与审计文件

```text
storage/pilot-v1/
├── pilot_manifest.json                  schema、状态、配置与文档身份
├── rag-anything-v1-ingest-audit.json     底层状态与多模态覆盖检查
├── rag-anything-v1-ingest-usage.json     建库 usage
└── 图、向量、KV、来源及缓存存储
```

状态按 `indexing → ready` 变化；异常记为 `failed`。只有底层处理状态完整、多模态覆盖检查通过、显式持久化和正常结束完成后，才写入 `ready`。

官方代码可能跳过失败的多模态描述。启动器额外核对每种模态在每页的预期对象数与实际块数，并检查被捕获的模型调用错误。**数量吻合仍不证明描述正确**，必须抽查图表摘要、实体和来源。相同内容去重也可能触发覆盖告警，应核实后通过新的解析/索引版本处理，不直接忽略。

### 6.3 最低结构检查

选择一个图表父对象及其内部实体，核对描述、来源块、归属边和向量对象 ID；再检查一个正文实体是否在图中正确关联。重启后查询同一知识库，确认数据确实来自持久化存储。

更换模型、分块或描述策略时，建立 `pilot-v2` 等新库；失败库保留。第一轮不做复杂增量修复，也不直接复用无 manifest 的旧存储。

## 7. 检索及视觉生成验证

### 7.1 三种模式及可解释范围


| 模式     | LightRAG 1.4.16 行为                | 诊断目标       |
| -------- | ----------------------------------- | -------------- |
| `naive`  | 内容块向量检索                      | 语义相似度召回 |
| `hybrid` | 合并 local 实体与 global 关系图检索 | 图证据召回     |
| `mix`    | 图检索加内容块向量召回              | 组合召回       |

三者使用同一份含多模态文字代理的索引。`naive` 不能直接冒充论文中“跳过双图构建”的独立 Chunk-only 消融，`hybrid` 也不等同于论文概念层的全部混合检索机制。[S3]

同题先检索、再生成：

```bash
python reproduce.py query --kb pilot-v1 --mode mix --context-only \
  --question "According to Table 4, what is the overall accuracy gain of RAGAnything over Chunk-only, in percentage points?"
```

将 `mix` 分别换为 `naive`、`hybrid`。查看 `result.json` 中的实体、关系、内容块及 references，按人工证据判断召回完整性。只检索仍可能调用关键词模型和 embedding。

结构化接口返回最终处理后的证据，不是融合前的全部候选；研究候选贡献或路径权重时，需要增加内部观测。

### 7.2 文字与原图增强生成

```bash
python reproduce.py query --kb pilot-v1 --mode mix \
  --question "According to Table 2, what is RAGAnything's accuracy for multimodal questions?"

python reproduce.py query --kb pilot-v1 --mode mix --vlm \
  --question "In Figure 1, which two knowledge graphs are built before the merged knowledge graph?"
```

`--vlm` 从检索提示词中回引有效图片并交给视觉模型。找不到有效图片时，官方实现回退普通查询；请求视觉模式不等于实际发送了图片。[S2]

每次记录应检查：


| 字段                               | 判读方式                                       |
| ---------------------------------- | ---------------------------------------------- |
| `status / answer`                  | 成功、空检索与回答是否对应                     |
| `retrieval`                        | 诊断证据及来源                                 |
| `vlm_requested`                    | 是否请求视觉增强                               |
| `images_in_generation_adapter`     | 适配器观察到的图片数                           |
| `events`                           | 实际提示词、模型请求状态与输出；图片负载已脱敏 |
| `reported_usage / elapsed_seconds` | 本次诊断和生成开销                             |

至少一个视觉题须满足：正确原图进入适配器、调用成功返回、答案与原图一致。图片数量大于零本身不代表模型理解正确。

### 7.3 诊断与生成的边界

启动器先调用一次 `aquery_data()` 诊断，再调用正式查询；视觉回退还可能增加检索。两次结果可能不同，因此 `retrieval` 不能自动视为生成证据。结果明确标记 `diagnostic_retrieval_is_separate_from_generation`，实际输入以 `events` 为准。

空诊断证据会直接记录 `empty_retrieval`，不调用生成模型。返回的“证据不足”是工程保护行为，应与模型自主拒答分开评测。耗时和 usage 含诊断开销，也应与生产查询口径区分。

## 8. 批量评测与演示

### 8.1 问题及判分协议

[20 题诊断集](/Users/drw/PycharmProjects/rag-anything-reproduction/questions.pilot.jsonl) 包含正文 4、表格 4、图像 4、公式 3、跨模态 3、无答案 2。每题有参考答案、必要证据和 PDF 页序；标注根据原文建立，仍需研究者逐题签认后冻结。

参考答案只用于评测，不传入模型。运行记录保存问题文件哈希。冻结后若修订答案或问题，另存版本并说明原因，保留旧预测。

评分规则：数值及单位须正确；比较题按“百分点”判断；解释题覆盖参考答案的必要事实即可，不要求逐字一致；无答案题正确说明原文未提供信息。请求失败与空检索保留，不能从分母中悄悄删除。

### 8.2 执行矩阵

先运行 R1–R3，再运行 G1；真实视觉接口通过后运行 G2。每个批次串行执行，单题默认超时 300 秒，遇异常停止并保留已完成题目，不自动重跑。

```bash
python reproduce.py batch --kb pilot-v1 --questions questions.pilot.jsonl --mode naive --context-only
python reproduce.py batch --kb pilot-v1 --questions questions.pilot.jsonl --mode hybrid --context-only
python reproduce.py batch --kb pilot-v1 --questions questions.pilot.jsonl --mode mix --context-only
python reproduce.py batch --kb pilot-v1 --questions questions.pilot.jsonl --mode mix
python reproduce.py batch --kb pilot-v1 --questions questions.visual.jsonl --mode mix --vlm
```


| 条件         | 问题数 | 用途                 |
| ------------ | -----: | -------------------- |
| R1 / R2 / R3 |  各 20 | 三种模式检索证据对照 |
| G1           |     20 | 文字生成工程基线     |
| G2           |      4 | 视觉回引及回答检查   |

批次保存 `run-manifest.json` 与 `qXX-result.json`；错误记录包含失败题 ID。将实际 run_id 和逐题判断填入 [评测模板](/Users/drw/PycharmProjects/rag-anything-reproduction/evaluation.template.csv)。模板的空白分数不代表任何实验结果。

### 8.3 指标与错误归因


| 指标               | 定义 / 口径                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| 必要证据完整召回率 | 可回答题中，全部必要证据进入指定证据集合的比例；注明诊断集合或实际上下文 |
| 回答准确率         | 按冻结规则判为正确的题数 / 全部题数，另列请求失败数                      |
| 无答案处理         | 正确拒答、错误作答、空检索保护、请求失败的具体计数                       |
| 用量与耗时         | 建库和查询分开；记录输入规模、图片数、缓存及额外检索                     |

错误标签：`P` 解析、`D` 描述、`K` 知识构建、`R` 检索、`C` 上下文截断、`V` 视觉回引、`A` 回答、`E` 评测。允许多标签；根据实际证据归因。

单篇 20 题用于诊断，每道题使总体准确率变化 5 个百分点；不能据此推出跨文档性能结论。演示按“原 PDF → 解析 → 图与向量对象 → 同题检索 → 原图输入 → 回答 → 失败定位”展开。

## 9. 本地 API 与故障处理

### 9.1 服务运行

真实命令行入库及查询通过后，再演示本地服务：

```bash
python -m pip install -r requirements-service.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

默认库为 `pilot-v1`。其他库可在启动前设置 `PILOT_KB`。打开 `/docs` 或 `/health`；查询请求：

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Which embedding model is used?","mode":"mix","context_only":true,"vlm":false}'
```

响应包含状态、答案、诊断证据、图片数量、usage 和 request_id；完整记录进入 `runs/api_<request_id>/`。非法模式、空白问题返回 422；执行超时返回 504；运行异常返回 500。服务只有一个进程和一条在途查询，文件锁覆盖整个服务生命周期。

这是本地单用户查询接口。新增文档时先正常停止服务，再运行 CLI 入库并重启；上传、任务队列、账号权限和生产部署另行设计。[S8]

### 9.2 故障处理顺序


| 现象              | 检查 / 处理                                      |
| ----------------- | ------------------------------------------------ |
| 安装或导入失败    | 当前`.venv`、固定版本、`pip check`、完整安装日志 |
| `get_pos` 错误    | pypdfium2/pdftext 兼容组合，避免盲目升级         |
| 不支持的内容类型  | 检查原始列表；chart 已转换；其他类型先定义映射   |
| 401 / 404 / 429   | 对应角色的权限、模型 ID、端点及限流；先修探针    |
| 维度或哈希不一致  | 新建解析/索引版本，保留旧产物                    |
| 多模态覆盖告警    | 对照审计、模型请求及块记录，排查遗漏与去重       |
| 开启 VLM 但零图片 | 检索引用、绝对路径、图片有效性及视觉回退         |
| 知识库被占用      | 正常停止占用进程；不并行操作同一库               |
| 候选存在但答错    | 查看实际生成上下文、图表描述、截断及回答计算     |

启动器不自动重跑失败实验；依赖可能有内部网络重试。超时后已有请求或部分写入可能已发生，检查现有状态后再决定恢复。

不批量删除文件或目录。旧解析、失败库和运行记录保留；新实验使用新标识。确需删除时只处理一个明确文件，批量清理由研究者手动执行。

## 10. 本次核验记录与交付状态

### 10.1 已执行验证


| 项目                             | 本次结果                     | 证据                                                  |
| -------------------------------- | ---------------------------- | ----------------------------------------------------- |
| 官方提交及接口                   | 已核对并验证安装来源         | `check`、固定源码                                     |
| 真实依赖安装                     | 通过；兼容依赖修正后无冲突   | 本机版本快照、安装与依赖日志                          |
| 离线行为检查                     | 13 项通过                    | 配置、锁、哈希、空检索、chart 及调用记录              |
| 真实框架集成                     | 1 项综合测试通过，模型为替身 | 四模态入库、三模式检索、视觉输入、重启查询、ASGI 接口 |
| 原论文 CPU 解析                  | 通过，149 项                 | 内容列表、图片及 manifest                             |
| 重点证据抽查                     | Figure 1、Table 2/4 已检查   | 图像裁剪、表格结构和关键数值                          |
| 真实文字 / 视觉 / embedding 接口 | 待验证                       | 尚未填写可调用配置                                    |
| 正式知识库及 20 题评测           | 待执行                       | 不存在已验证准确率                                    |
| 真模型本地服务                   | 待验证                       | ASGI 集成通过不等于实际服务已部署                     |

详细证据及文件哈希见 [工程核验记录](/Users/drw/PycharmProjects/rag-anything-reproduction/verification.json)。框架集成使用替身模型验证工程链路，不能替代真实模型端到端验收，也不能保证其他机器、供应商和文档零错误。

### 10.2 当前下一步

继续使用已建立的环境和 `parsed/rag-anything-v1/`：填写本地 `.env` → 文字/embedding 探针 → Figure 1 图片探针 → 逐题复核解析与答案 → 建立 `pilot-v1` → 三模式检索及生成 → 批量评测 → 本地演示。

项目已迁至 `/Users/drw/PycharmProjects/rag-anything-reproduction`。源 PDF 已纳入 `data/`；文档链接、解析图片路径和虚拟环境启动路径已更新，迁移后再次检查依赖、解析哈希及核心导入。历史日志保留原执行位置，迁移记录见 `verification/migration.json`。失败解析、重复测试与研发临时材料集中于 `_待手动清理/`，按使用者禁止批量删除的规则留待手动清理。

### 10.3 实验记录模板

```markdown
### Run：编号
- 日期 / 时区：
- 官方提交 / 启动器哈希 / 完整依赖版本：
- OS / Python / 架构 / 内存 / 解析设备：
- 模型供应商 / ID / 可得快照 / 端点：
- PDF / 解析 / 问题集哈希：
- doc-id / kb / run_id：
- 修改变量 / 保持一致的条件：
- 产物 / 实际用量 / 耗时：
- 逐题验收 / 失败阶段与证据：
```

最终交付包需包含环境、文档及解析、索引审计、冻结题集、完整预测、人工评分、失败分析和演示记录。本 Markdown 可作为研究产出中的方法与实施记录；结果部分应在真实实验完成后据实补充。

## 11. 研究改进与论文结果复现

### 11.1 冻结基线后研究一个变量


| 研究问题               | 修改变量                   | 关键控制条件                                 |
| ---------------------- | -------------------------- | -------------------------------------------- |
| 原图是否改善视觉回答   | 是否加入检索到的原图       | 同一问题、冻结上下文、同一视觉模型及生成参数 |
| 图信息是否改善证据召回 | 检索策略或真正的图构建消融 | 同一文档、描述模型、评测题集与明确预算       |
| 重排序是否降低干扰     | reranker 关闭 / 开启       | 固定候选、生成模型及上下文预算               |

严格原图消融只检索一次，保存上下文和图片列表；同一视觉模型分别读取“文字”与“同一文字 + 原图”，随机运行顺序并逐题盲判。当前普通 `query` 会再次检索，不实现冻结上下文的严格消融，G1/G2 的差异只能用于工程诊断。

从高频失败选择一个改进方向，例如表头层级表示、实体消歧、跨页来源约束或任务相关图像描述。不要同时更换解析器、模型和检索策略后将变化归因于单一组件。

正式评测按文档划分开发/测试集，避免用测试题指导建库或调参；报告各题型、长度、无答案比例、解析失败、成本和延迟。若做区间估计或重采样，考虑同一文档内问题的相关性；样本量依据希望检测的效果与开发集波动确定。

### 11.2 对齐论文结果前必须解决的事项

原文配置为 DocBench 229 篇 / 1,102 题，MMLongBench 135 篇 / 1,082 题，GPT-4o-mini、3072 维 embedding、bge-reranker-v2-m3、实体关系合计 20k 与内容块 12k 预算、一句话回答及附录 A.4 判分。直接视觉基线还使用最多 50 页、144 dpi。[S1]


| 未确定项                         | 复现要求                                           |
| -------------------------------- | -------------------------------------------------- |
| 原始代码提交、解析版本、模型快照 | 查明可得版本；替代项明确报告                       |
| MMLongBench-Doc 数据清单         | 当前仓库介绍 1,091 题，需解释与原文 1,082 题的差异 |
| 图扩展跳数、融合权重等           | 区分原文概念、固定代码实现与自行设置               |
| 原文独立 token 预算              | 核对上下文组装与截断，不能只填写相似数字           |
| 本轮模式对照与原文消融           | 单独实现并验证真正的 Chunk-only 等条件             |

原文总体准确率为 DocBench 63.4%、MMLongBench 42.8%。仅在条件对齐后用于比较；条件不同则报告为工程变体或重新评估。[S1][S6][S7]

## 12. 参考来源

以下为事实与接口依据；目录、验收规则和小规模评测设计是本次工程方案。

- **[S1] 原文**：[RAG-Anything，arXiv:2510.12323v1](https://arxiv.org/abs/2510.12323v1)，重点为 Figure 1、§§2.2–2.4、§3.1、Tables 1–4、附录 A.4；依据用户提供的完整 PDF。
- **[S2] 框架源码**：[官方固定提交](https://github.com/HKUDS/RAG-Anything/tree/8664e8b318a3ed651fe62dfcfddfb1f1d633d5be)，核对 `parser.py`、`processor.py`、`modalprocessors.py`、`query.py`、`config.py` 与复现脚本。
- **[S3] 检索后端**：[LightRAG 官方说明](https://github.com/HKUDS/LightRAG#selecting-query-modes)，[1.4.16 发布包](https://pypi.org/project/lightrag-hku/1.4.16/)；同时核对该发布包接口及实际集成行为。
- **[S4] 解析依赖**：[MinerU 3.4.1](https://pypi.org/project/mineru/3.4.1/)，[pdftext 0.6.3 发布元数据](https://pypi.org/pypi/pdftext/0.6.3/json)。
- **[S5] PDF API 变更**：[pypdfium2 官方 changelog](https://pypdfium2-team.github.io/pypdfium2/changelog.html)，[MinerU 对应问题记录](https://github.com/opendatalab/MinerU/issues/5227)；本机另以失败日志和修正后解析验证。
- **[S6] DocBench**：[官方仓库](https://github.com/Anni-Zou/DocBench)。
- **[S7] MMLongBench-Doc**：[Ma et al. 对应官方仓库](https://github.com/mayubo2333/MMLongBench-Doc)，避免与其他近似同名基准混淆。
- **[S8] 服务生命周期**：[FastAPI 官方 Lifespan 文档](https://fastapi.tiangolo.com/advanced/events/)；本次检查启动、查询、校验与关闭的 ASGI 行为。

核对日期：2026-10-08。固定依赖与实际验证记录优先于持续变化的主线文档。
