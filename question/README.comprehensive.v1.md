# v1 综合测试题集

使用现有 `handson-qwen4096-v1`，文档为 `rag-anything-handson-v1`。
共 48 题、10 类，其中 42 题可回答、6 题应明确拒答。参考答案依据本地论文正文、结构化表格、公式及已查看的原始裁剪图编写；未运行模型问答或自动评分。

- `questions.comprehensive.v1.jsonl`：唯一的运行输入，不含参考答案。
- `evaluation.comprehensive.v1.jsonl`：同 ID 的参考答案、必需事实、证据索引、PDF 页码、目标图路径及校验值、计算结果和评分注意事项。
- PDF 页码从 1 开始；证据索引从 0 开始，对应 `parsed/rag-anything-handson-v1/content-list.json`。


| 类型                    | 覆盖内容                           | 题数 |
| ----------------------- | ---------------------------------- | ---: |
| `text_fact`             | 原论文事实及历史失败回归           |    2 |
| `text_reasoning`        | 机制比较和失败分析                 |    2 |
| `table_reasoning`       | 表格定位、排序、并列和计算         |    5 |
| `equation_reasoning`    | 公式与正文语义对齐                 |    4 |
| `visual_grounding`      | 原图图例、子图和空间关系           |    4 |
| `cross_modal_reasoning` | 正文/原图/表格/公式组合推理        |   16 |
| `cross_table_reasoning` | 跨页多表关联、聚合和比率           |    3 |
| `evidence_reasoning`    | 证据边界、错误前提和结论核查       |    4 |
| `unanswerable`          | 缺失参数、年份、指标及库外信息拒答 |    6 |
| `robustness`            | 意译和中英同题一致性               |    2 |

`type`、`modalities`、`difficulty` 是分析标签，现有 `batch` 不会据此按题切换模式或自动评分。`modalities` 反映证据类型，其中嵌在原图内的表格需要先取得原图。多表题单独分类，避免把所有复杂问题都说成跨模态题。

图内可能有被引用文档自己的编号，例如本论文 Figure 3 内的源图写着 Figure 2；题目使用当前论文的编号。Figures 3–6 和 11–12 含原始源证据与模型回答框，读取数值时应核对源证据，不能把失败答案当真值。

**问题文件 `questions.comprehensive.v1.jsonl`**


| 字段                | 含义                                                                         |
| ------------------- | ---------------------------------------------------------------------------- |
| `id`                | 题目唯一编号，例如`cv1-q21`                                                  |
| `doc_id`            | 题目针对的文档，当前为`rag-anything-handson-v1`                              |
| `question`          | 实际提交给模型的问题                                                         |
| `type`              | 测试类别，例如表格推理、公式理解、跨模态推理、拒答测试                       |
| `difficulty`        | 标注的预期难度：简单、中等、困难；                                           |
| `modalities`        | 涉及的证据类型：正文、图片、表格、公式。图内表格可能同时标为`image`、`table` |
| `consistency_group` | 部分题目使用，用于关联同义改写或中英同题，检查回答一致性                     |

现有批量程序使用 `id`、`doc_id`、`question`。其余字段用于分析

**参考文件 `evaluation.comprehensive.v1.jsonl`**


| 字段                            | 含义                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `question_id`                   | 对应问题文件的`id`                                                                         |
| `doc_id`、`type`                | 对应文档和测试类别                                                                         |
| `answerable`                    | 论文是否提供足够信息；`false`表示预期应拒答                                                |
| `reference_answer`              | 整理的参考答案                                                                             |
| `required_facts`                | 判断答案完整正确所需的关键事实                                                             |
| `evidence_block_indices_0based` | 证据在解析文件`content-list.json`中的位置，**从 0 开始**                                   |
| `pdf_pages_1based`              | 证据所在 PDF 页码，**从 1 开始**                                                           |
| `evidence_objects`              | 证据名称，例如 Table 2、Figure 4、某节正文                                                 |
| `target_assets`                 | 需要核对的目标原图信息，具体见下表                                                         |
| `visual_evidence_policy`        | 核对要求：文本证据即可，或还应检查目标原图是否命中；拒答题也可能需要检查原图中缺失的年份等 |
| `annotation_status`             | 标明参考答案依据源材料整理，尚未经过模型结果评分                                           |
| `grading_notes`                 | 评分注意事项，例如不能混淆年份、单位或图中的失败回答                                       |
| `calculations`                  | 部分计算题的表达式和预期结果                                                               |
| `consistency_group`             | 同题一致性分组，与问题文件对应                                                             |
| `absence_check_scope`           | 拒答题中，判断信息未提供时所检查的材料范围                                                 |
| `source_content_sha256`         | 原解析内容文件的指纹，用于确认参考答案对应哪个版本的材料                                   |
| `answer_correct`                | 实际答案是否正确，暂为`null`，运行后填写                                                   |
| `necessary_evidence_complete`   | 检索证据是否完整，暂为`null`                                                               |
| `target_assets_hit`             | 生成回答时是否传入正确目标原图，暂为`null`                                                 |
| `failure_stage`                 | 失败发生在哪个环节，例如漏检、选错图片、读图错误、计算错误，暂为`null`                     |

`target_assets` 内每个对象的字段：


| 字段                 | 含义                                 |
| -------------------- | ------------------------------------ |
| `block_index_0based` | 图片对应的解析块索引                 |
| `asset_basename`     | 图片文件名                           |
| `asset_path`         | 图片相对项目目录的路径               |
| `sha256`             | 图片内容指纹，用于核对是否为同一张图 |

---

## 2026-10-11

### 1. 运行配置

运行依据：[run-manifest.json](../runs/20261011_110345_7173b7cc/run-manifest.json)。判断本轮用了什么模型，应看这份运行快照；


| 项目              | 本轮值与含义                                                                            |
| ----------------- | --------------------------------------------------------------------------------------- |
| 知识库            | `handson-qwen4096-v1`，沿用已建立的 v1 库                                               |
| 文档              | `rag-anything-handson-v1`，RAG-Anything 论文                                            |
| 开始时间          | 2026-10-11 11:03:45，Asia/Shanghai                                                      |
| 完成证据          | 最后一份答案文件写于约 11:11:44；48 个题号均有结果                                      |
| 检索方式          | `mix`，结合图检索与向量检索                                                             |
| 生成设置          | `vlm_requested=true`，`context_only=false`                                              |
| 本轮文本/视觉请求 | 均为`deepseek-flash`，地址 `https://api.deepseek.com`；保存的事件确认该模型名被实际请求 |
| 向量模型          | 本地`qwen3-embedding:latest`，4096 维，`http://127.0.0.1:11434/v1`                      |
| 重排              | 本地`BAAI/bge-reranker-v2-m3`，实际设备 `mps`，`float32`，最大输入 8192，批大小 1       |
| 检索与上下文预算  | `top_k=20`、`chunk_top_k=20`；实体/关系预算各 4000，总预算 24000                        |
| 块参数            | `chunk_size=1200`、`chunk_overlap=100`，沿用库的索引配置                                |
| 并发设置          | `concurrency=2` 是模型调用配置；当前批量入口仍按题逐个运行                              |
| 依赖快照          | `raganything 1.4.2`、`lightrag-hku 1.4.16`、`mineru 3.4.1`                              |

**查询模型与建库模型要分开看。** `settings` 记录本轮 DeepSeek 查询配置；`index_manifest.index_identity` 记录此前建库时的本地文本模型 `benchmark-312-qwen25-7b-ctx16k`、视觉模型 `benchmark-312-qwen38-27b-ctx16k`。本轮使用 DeepSeek 生成答案，索引里的图片描述、表格分析和图实体仍来自此前建库过程。

### 2. `runs/`

每次命令行查询/批量测试通常产生一个独立子目录，名称为 `日期_时间_随机后缀`；这里的时间使用 Asia/Shanghai。知识库本体在 `storage/`，解析材料和原图在 `parsed/`，`runs/` 保存测试记录。

本次完整目录共 49 个文件：

```text
runs/
└── 20261011_110345_7173b7cc/      # 本节分析的完整运行
    ├── run-manifest.json         # 配置、版本、索引与输入指纹
    ├── cv1-q01-result.json       # 每题一份实际结果
    ├── cv1-q02-result.json
    ├── ...
    └── cv1-q48-result.json
```


| 文件/目录                | 含义                                                                    |
| ------------------------ | ----------------------------------------------------------------------- |
| `run-manifest.json`      | 本次运行的配置快照、计划题数、库版本、依赖版本和校验信息                |
| `<题号>-result.json`     | `batch` 逐题输出；本轮为 `cv1-q01-result.json` 到 `cv1-q48-result.json` |
| `result.json`            | 单题`query` 命令的输出命名                                              |
| `error.json`（可能出现） | 命令捕获异常时保存错误类型、时间、当前题号及已有事件；本轮没有此文件    |
|                          | 知识库锁文件`.pilot.lock` 位于 `storage/<库名>/`，并非答案文件。        |

#### `run-manifest.json` 字段


| 字段               | 含义                                                                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `created_at`       | 运行目录创建时间，含时区                                                                                                                           |
| `kb`               | 使用的知识库名称                                                                                                                                   |
| `question_count`   | 输入计划运行多少题                                                                                                                                 |
| `mode`             | 全局检索模式；本轮`mix`                                                                                                                            |
| `vlm_requested`    | 是否请求视觉增强生成                                                                                                                               |
| `context_only`     | 是否只获取上下文，不生成答案                                                                                                                       |
| `settings`         | 本轮实际读取的公开配置；API 密钥不会写入这里                                                                                                       |
| `packages`         | 本轮相关依赖包版本                                                                                                                                 |
| `index_manifest`   | 已建立知识库的清单，包含索引身份、状态、文档、源文件与原图校验信息；它描述建库来源                                                                 |
| `starter_sha256`   | 查询入口`reproduce.py` 的文件指纹                                                                                                                  |
| `questions_sha256` | 批量输入 JSONL 的文件指纹；单题查询通常为`null`                                                                                                    |
| `reranker`         | 实际加载的重排器记录，含模型、固定版本、路径、实际设备、精度、长度、批大小、权重大小、依赖版本、加载时间等；需与`settings.rerank_device=auto` 区分 |

### 3. 答案 JSON 字段

#### 3.1 顶层字段


| 字段                                               | 含义与读取注意事项                                           |
| -------------------------------------------------- | ------------------------------------------------------------ |
| `question_id`                                      | 题号，与问题文件`id`、参考文件 `question_id` 对应            |
| `question`                                         | 本题实际提交的完整问题                                       |
| `status`                                           | 执行状态；本轮全为`success`，表示流程成功，不表示答案正确。入口还可能返回 `empty_retrieval` |
| `mode`                                             | 本题检索模式；本轮均为`mix`                                  |
| `context_only`                                     | `true` 时只取上下文，通常没有生成答案；本轮 `false`          |
| `vlm_requested`                                    | 本题是否启用视觉增强请求；本轮全部`true`，标签 `type` 不会自动切换该设置 |
| `images_in_generation_adapter`                     | 生成适配器收到的图片数量，是`vision_request.image_count` 的合计；不是正确图片数，也不表示模型使用了每张图 |
| `retrieval`                                        | 独立诊断检索返回的结构化结果，详见下文                       |
| `answer`                                           | 模型生成的答案全文，通常是 Markdown，包含答案及 References；不是标准答案 |
| `events`                                           | 请求与上下文审计记录，可核对实际关键词、重排、生成提示和送入的图片 |
| `reported_usage`                                   | 已追踪请求返回的 token 用量汇总                              |
| `elapsed_seconds`                                  | 该题从诊断检索到生成完成的耗时，包括两次检索、重排、向量/模型调用等；不包括程序启动、重排器加载和结果写盘等外围步骤 |
| `diagnostic_retrieval_is_separate_from_generation` | 本轮为`true`：`retrieval` 是先做的一次诊断检索，生成时又执行一次实际检索，两者不能直接视为完全相同 |
| `rerank_enabled`                                   | 是否启用重排；本轮`true`                                     |
| `rerank_calls`                                     | 本题实际重排回调次数；本轮每题 2 次，对应诊断及实际生成检索  |
| `rerank_elapsed_seconds`                           | 本题重排耗时之和，已包含在`elapsed_seconds` 内，不能再重复相加 |

#### 3.2 `retrieval`：检索的内容

顶层有 `status`、`message`、`data`、`metadata`。`status/message` 描述诊断检索的运行状态，不描述最终回答对错。

`data` 包含四组对象：


| 对象            | 主要字段与含义                                                                                                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entities`      | `entity_name` 实体名称；`entity_type` 类型；`description` 建库生成的描述；`source_id` 来源块 ID，多个来源可用 `<SEP>` 连接；`file_path` 源文件路径；`created_at` 建库记录的 Unix 时间戳 |
| `relationships` | `src_id/tgt_id` 边的源/目标实体；`description` 关系描述；`keywords` 关系关键词；`weight` 图边权重；`source_id/file_path/created_at` 描述来源。`weight` 不是正确率、概率或重排分数       |
| `chunks`        | `chunk_id` 检索块 ID；`reference_id` 引用编号；`content` 块文本，可能混合正文、HTML 表格、建库生成的视觉描述和 `Image Path`；`file_path` 源文件路径                                     |
| `references`    | `reference_id` 与 `file_path` 的对应表，供答案引用使用；本轮源文档是同一 PDF                                                                                                            |

`chunk_id` 是检索存储中的块 ID，不能直接当成参考标注的 `evidence_block_indices_0based`。References 中的 `[1]` 仅指向源 PDF；没有逐项证明答案正确，也未必定位到页码和具体单元格。

`metadata` 包含：


| 字段                                         | 含义                                           |
| -------------------------------------------- | ---------------------------------------------- |
| `query_mode`                                 | 诊断检索使用的模式                             |
| `keywords.high_level`                        | 抽取的高层主题关键词                           |
| `keywords.low_level`                         | 抽取的具体实体、名称、年份等关键词             |
| `processing_info.total_entities_found`       | 检索阶段找到的实体数量                         |
| `processing_info.total_relations_found`      | 检索阶段找到的关系数量                         |
| `processing_info.entities_after_truncation`  | 预算截断后保留的实体数量                       |
| `processing_info.relations_after_truncation` | 预算截断后保留的关系数量                       |
| `processing_info.merged_chunks_count`        | 合并后的候选块数量                             |
| `processing_info.final_chunks_count`         | 最终保留的块数量，受重排、过滤及上下文预算约束 |

例如 q21 诊断阶段找到 42 个实体、122 条关系，保留 42 个实体、86 条关系，30 个合并候选块最终保留 13 块。`chunk_top_k=20` 是上限配置，不保证每题最终得到 20 块。本轮诊断结果每题为 12–18 块、29–50 个实体、45–89 条关系。

#### 3.3 `events`：发生事件


| `kind`                    | 字段及用途                                                                                                                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text_request`            | `model/status` 是实际文本请求模型与状态；`prompt/system_prompt/history_messages` 是输入；`response` 是返回文本。本轮每题两次，用于两次检索的关键词抽取                                                                 |
| `rerank_request`          | `model/device/status` 是本地重排模型、设备及状态；`candidate_count` 是候选数；`top_n` 是请求保留数；`max_length` 是输入长度上限；`ranked_candidates` 是排序审计；`elapsed_seconds` 是重排耗时，失败时可有 `error_type` |
| `actual_retrieval_prompt` | `prompt` 记录用于实际生成的检索提示。它比单独的 `retrieval` 更适合核对最终文字上下文；是否送入原图还需检查 `vision_request`                                                                                            |
| `vision_request`          | `model/status/image_count` 记录视觉适配器的生成调用；`messages` 记录 system/user 输入及图文部件；`response` 是实际生成答案                                                                                             |

`ranked_candidates` 中每项包含：`index` 是重排前候选列表的索引，列表顺序才是排序结果；`relevance_score` 是模型 sigmoid 后的相关性分数，不是答案真实概率；`content_sha256` 是候选文本指纹；`input_tokens` 是 query+候选的截断前 token 长度；`truncated` 表示是否超过重排长度上限。本轮这类长度截断为 0，但后续总上下文预算仍会裁剪证据。

`vision_request.messages` 中每条消息有 `role` 与 `content`。`content` 可以是字符串，也可以是含 `text`、`image_url` 的部件数组。为避免把大段图片数据写入结果，`image_url.url` 保存为 `[image payload omitted]`，但保留 `payload_sha256`。

**两个图片 SHA256 的计算对象不同：**参考标注 `target_assets.sha256` 校验原图文件字节；事件 `payload_sha256` 校验整个图片 data URL 字符串。不能直接比较二者。本次核对时，先验证原图文件，再按实际编码方式构造 `data:image/jpeg;base64,` 加原文件的 Base64，计算该字符串指纹，与真实生成请求逐张匹配。

#### 3.4 `reported_usage`：token 与调用数


| 字段                | 含义                                                                           |
| ------------------- | ------------------------------------------------------------------------------ |
| `prompt_tokens`     | 追踪器累计的输入 token；包含被追踪的文本/视觉模型请求及本地 embedding 返回用量 |
| `completion_tokens` | 追踪器累计的生成 token                                                         |
| `total_tokens`      | 各接口返回的总 token 累计                                                      |
| `call_count`        | 成功向追踪器报告用量的请求次数，不等于重排次数或事件总数                       |

本轮每题 `call_count=5`：两次文本关键词请求、一次视觉生成，以及两次本地 embedding 用量记录。embedding 没有单独写成 `events` 事件；本地 BGE 重排也不会增加这个 API token 调用数。因此，不能把总 token 全部按 DeepSeek 费用计算，不能把视觉请求记录当作提供方确实利用了每张图的证明。

### 4. 回答结果

- **核心通过**：问题要求的关键结论、数值、对应关系与证据边界正确且足够完整。多问缺少一问、错误排名、错误取数、错误拒答均不能通过。
- **完整通过**：满足核心要求，且本次核对未发现附加的事实细节错误。本轮 q24、q26 核心内容正确，但分别多写错了颜色和节点名称。
- **执行成功**、**目标原图送入**、**回答正确**分别统计，互不替代。这里没有给所有题的 `necessary_evidence_complete` 自动赋值，目标图命中也不能说明所有必要文字证据齐全。


| 指标                   |                                 结果 |
| ---------------------- | -----------------------------------: |
| 计划题数 / 实际结果数  |                              48 / 48 |
| 执行成功               |                          48/48，100% |
| 核心通过               |                        42/48，87.50% |
| 完整通过               |                        40/48，83.33% |
| 核心错误或漏答         |   6 题：q04、q07、q22、q25、q27、q37 |
| 核心正确但附加细节错误 |                       2 题：q24、q26 |
| 可回答题核心通过       |                        36/42，85.71% |
| 可回答题完整通过       |                        34/42，80.95% |
| 应拒答题正确拒答       |                            6/6，100% |
| 标注要求目标原图的题数 |                                21 题 |
| 这些题全部目标图送入   | 21/21；按题累计目标图 22/22 次均命中 |
| 同义/中英一致性        |          q06/q47、q21/q48 两组均一致 |

q25 不计入正确拒答：它是**原图有答案却说没有信息**的可回答题，属于漏答。6/6 正确拒答只针对 q41–q46：


| 类别                    | 题数 | 核心通过 | 完整通过 |
| ----------------------- | ---: | -------: | -------: |
| `text_fact`             |    2 |      2/2 |      2/2 |
| `text_reasoning`        |    2 |      1/2 |      1/2 |
| `table_reasoning`       |    5 |      4/5 |      4/5 |
| `equation_reasoning`    |    4 |      4/4 |      4/4 |
| `visual_grounding`      |    4 |      4/4 |      4/4 |
| `cross_modal_reasoning` |   16 |    13/16 |    11/16 |
| `cross_table_reasoning` |    3 |      3/3 |      3/3 |
| `evidence_reasoning`    |    4 |      3/4 |      3/4 |
| `unanswerable`          |    6 |      6/6 |      6/6 |
| `robustness`            |    2 |      2/2 |      2/2 |

表格计算、公式语义、基础空间定位、跨表关联和限定范围的拒答表现较好。跨模态组合题的错误更多，核心通过 13/16，完整通过 11/16；**这一分布说明复杂证据对齐是本题集暴露的主要薄弱处**。不过类别样本数小、难度不同，不能据此估计真实任务分布的准确率。

### 逐题分析

点击题号可打开对应实际结果。“原图数”是本题全部送入生成适配器的图片数；“目标命中”仅检查参考文件要求的目标图，`—` 表示没有预先指定目标图，不表示未传图。


| 题号                                                            | 类型                    | 结论               | 实际回答与核对结果                                                                         | 原图数 | 目标命中 |
| --------------------------------------------------------------- | ----------------------- | ------------------ | ------------------------------------------------------------------------------------------ | -----: | -------: |
| [cv1-q01](../runs/20261011_110345_7173b7cc/cv1-q01-result.json) | `text_fact`             | 通过               | 正确区分论文与本地配置：text-embedding-3-large，3072 维。                                  |      8 |       — |
| [cv1-q02](../runs/20261011_110345_7173b7cc/cv1-q02-result.json) | `text_fact`             | 通过               | 最多 50 页、144 dpi、一句话回答，三项均正确。                                              |     11 |       — |
| [cv1-q03](../runs/20261011_110345_7173b7cc/cv1-q03-result.json) | `text_reasoning`        | 通过               | 显式关系走结构知识导航；无连接但语义相近走向量匹配。                                       |      9 |       — |
| [cv1-q04](../runs/20261011_110345_7173b7cc/cv1-q04-result.json) | `text_reasoning`        | 部分正确，未通过   | 两类失败模式识别正确；把自适应空间推理单独配给文本检索偏差，改进对象对应不准确。           |      8 |       — |
| [cv1-q05](../runs/20261011_110345_7173b7cc/cv1-q05-result.json) | `table_reasoning`       | 通过               | DocBench 229/1102/66；MMLongBench 135/1082/47.5，数据集对应正确。                          |     11 |       — |
| [cv1-q06](../runs/20261011_110345_7173b7cc/cv1-q06-result.json) | `table_reasoning`       | 通过               | 76.3% 对 59.7%，增加 16.6 个百分点，正确。                                                 |     12 |       — |
| [cv1-q07](../runs/20261011_110345_7173b7cc/cv1-q07-result.json) | `table_reasoning`       | 部分正确，未通过   | 第一名 MMGraphRAG 46.9% 正确；RAGAnything 45.7% 应排第三，回答第二。                       |      9 |       — |
| [cv1-q08](../runs/20261011_110345_7173b7cc/cv1-q08-result.json) | `table_reasoning`       | 通过               | News 减少 2.3 个百分点，Overall 增加 1.0 个百分点，正确。                                  |      8 |       — |
| [cv1-q09](../runs/20261011_110345_7173b7cc/cv1-q09-result.json) | `table_reasoning`       | 通过               | 正确指出 Txt. 并列、Mm. 第一、Una. 不第一，未把整体优势推广到所有题型。                    |     11 |       — |
| [cv1-q10](../runs/20261011_110345_7173b7cc/cv1-q10-result.json) | `equation_reasoning`    | 通过               | 正确解释模态类型与内容，并说明保留图注、表头及上下文。                                     |      7 |       — |
| [cv1-q11](../runs/20261011_110345_7173b7cc/cv1-q11-result.json) | `equation_reasoning`    | 通过               | 详细描述 d_j^chunk 经 R 抽取；内部实体以 belongs_to 关联锚点，正确。                       |      3 |       — |
| [cv1-q12](../runs/20261011_110345_7173b7cc/cv1-q12-result.json) | `equation_reasoning`    | 通过               | 正确覆盖实体、关系、跨模态内容块，拒绝仅嵌入正文的解释。                                   |      7 |       — |
| [cv1-q13](../runs/20261011_110345_7173b7cc/cv1-q13-result.json) | `equation_reasoning`    | 通过               | q、P(q)、V*(q) 三项正确，并指出恢复原始视觉内容。                                          |      5 |       — |
| [cv1-q14](../runs/20261011_110345_7173b7cc/cv1-q14-result.json) | `visual_grounding`      | 通过               | 正确区分多模态父锚点、内部子实体与跨图对齐后的合并节点。                                   |      8 |      1/1 |
| [cv1-q15](../runs/20261011_110345_7173b7cc/cv1-q15-result.json) | `visual_grounding`      | 通过               | 蓝色 RAGAnything、橙色 MMGraphRAG；大差距在 101–200 和 200+ 页，正确。                    |     11 |      1/1 |
| [cv1-q16](../runs/20261011_110345_7173b7cc/cv1-q16-result.json) | `visual_grounding`      | 通过               | DAE 上、VAE 下；Style 左、Content 右，空间定位正确。                                       |      9 |      1/1 |
| [cv1-q17](../runs/20261011_110345_7173b7cc/cv1-q17-result.json) | `visual_grounding`      | 通过               | 2019 年工资为 DKK 25,335 million，年份、数值、单位正确。                                   |     13 |      1/1 |
| [cv1-q18](../runs/20261011_110345_7173b7cc/cv1-q18-result.json) | `cross_modal_reasoning` | 通过               | 非文本图与文本图分工、实体名称对齐键，均正确。                                             |      9 |      1/1 |
| [cv1-q19](../runs/20261011_110345_7173b7cc/cv1-q19-result.json) | `cross_modal_reasoning` | 通过               | Text VDB、Multi-modal VDB 及实体/关系/跨模态内容块覆盖范围正确。                           |      7 |      1/1 |
| [cv1-q20](../runs/20261011_110345_7173b7cc/cv1-q20-result.json) | `cross_modal_reasoning` | 通过               | 正确选择 DAE Style 的左上子图，并解释用布局图约束面板；编号表述见下文。                    |     11 |      1/1 |
| [cv1-q21](../runs/20261011_110345_7173b7cc/cv1-q21-result.json) | `cross_modal_reasoning` | 通过               | 增加 DKK 1,443 million，同比增长 5.70%，计算与四舍五入正确。                               |     10 |      1/1 |
| [cv1-q22](../runs/20261011_110345_7173b7cc/cv1-q22-result.json) | `cross_modal_reasoning` | 部分正确，未通过   | 26,778 和 823 正确；三个维度应为行标题、年份列、单位，回答把单位换成数据单元格。           |     13 |      1/1 |
| [cv1-q23](../runs/20261011_110345_7173b7cc/cv1-q23-result.json) | `cross_modal_reasoning` | 通过               | Twitter15 最低 -S-A，移除源推文嵌入与双重共注意力，正确。                                  |     11 |      1/1 |
| [cv1-q24](../runs/20261011_110345_7173b7cc/cv1-q24-result.json) | `cross_modal_reasoning` | 核心通过，细节错误 | Twitter16 最低 -S-A、第二低 -A 及组件解释正确；额外把 -A 说成紫色，实际绿色。              |     13 |      1/1 |
| [cv1-q25](../runs/20261011_110345_7173b7cc/cv1-q25-result.json) | `cross_modal_reasoning` | 部分正确，未通过   | Evidence Inference 最高组合及 0.506 正确；漏掉同图 BoolQ 的 0.525 与差值 0.019，错误拒答。 |      9 |      1/1 |
| [cv1-q26](../runs/20261011_110345_7173b7cc/cv1-q26-result.json) | `cross_modal_reasoning` | 核心通过，细节错误 | 八步处理顺序及自下而上方向正确；额外把 Word CRF 写成 Word CRF-F。                          |     11 |      1/1 |
| [cv1-q27](../runs/20261011_110345_7173b7cc/cv1-q27-result.json) | `cross_modal_reasoning` | 部分正确，未通过   | GEM 54.31 与布局陷阱正确；基线应 44.24，答成 54.21，增益应 10.07 而非 0.10 个百分点。      |     10 |      1/1 |
| [cv1-q28](../runs/20261011_110345_7173b7cc/cv1-q28-result.json) | `cross_modal_reasoning` | 通过               | 拒答匹配时 accuracy=1；JSON 键 accuracy/reasoning，论文评估模型 GPT-4o-mini，正确。        |     13 |      1/1 |
| [cv1-q29](../runs/20261011_110345_7173b7cc/cv1-q29-result.json) | `cross_modal_reasoning` | 通过               | Finance 平均 192 页、67.0%，比 LightRAG 高 10.8 个百分点，跨表关联正确。                   |     11 |       — |
| [cv1-q30](../runs/20261011_110345_7173b7cc/cv1-q30-result.json) | `cross_modal_reasoning` | 通过               | Finance 87 页/43.6%，Guidebooks 78 页/43.9%；正确反驳越长必然越准。                        |     11 |       — |
| [cv1-q31](../runs/20261011_110345_7173b7cc/cv1-q31-result.json) | `cross_table_reasoning` | 通过               | 领域问题数加总 1102、1082，均与 Table 1 一致。                                             |     11 |       — |
| [cv1-q32](../runs/20261011_110345_7173b7cc/cv1-q32-result.json) | `cross_table_reasoning` | 通过               | 每文档问题数 7.20 与 10.64，后者更高，正确。                                               |     11 |       — |
| [cv1-q33](../runs/20261011_110345_7173b7cc/cv1-q33-result.json) | `cross_table_reasoning` | 通过               | 两个数据集增益 5.0 与 3.9 个百分点，DocBench 更大，正确。                                  |     12 |       — |
| [cv1-q34](../runs/20261011_110345_7173b7cc/cv1-q34-result.json) | `evidence_reasoning`    | 通过               | 正确给出 Txt. 持平、Mm. 上升、Una. 下降，拒绝推广到不可回答题；用前后数值表达变化。        |     12 |       — |
| [cv1-q35](../runs/20261011_110345_7173b7cc/cv1-q35-result.json) | `cross_modal_reasoning` | 通过               | 192 页归入 101–200；区间差距 13.6、总体差距 2.4 个百分点，正确。                          |     10 |      1/1 |
| [cv1-q36](../runs/20261011_110345_7173b7cc/cv1-q36-result.json) | `evidence_reasoning`    | 通过               | 完整模型对 Chunk-only 优势 3.4、对无重排优势 1.0，论文设计解释正确。                       |     11 |       — |
| [cv1-q37](../runs/20261011_110345_7173b7cc/cv1-q37-result.json) | `evidence_reasoning`    | 部分正确，未通过   | 否定每领域第一，Tut./Admin. 反例正确；Broch. 误判第一，实际 LightRAG 41.0% 高于 34.0%。    |     10 |       — |
| [cv1-q38](../runs/20261011_110345_7173b7cc/cv1-q38-result.json) | `cross_modal_reasoning` | 通过               | 正确否定问题已全部解决，分别指出图文粒度错位/文本偏置与表格边界歧义。                      |     10 |      2/2 |
| [cv1-q39](../runs/20261011_110345_7173b7cc/cv1-q39-result.json) | `evidence_reasoning`    | 通过               | 正确拒绝把本地 Qwen/DeepSeek 配置当作论文设置，并给出论文模型。                            |      4 |       — |
| [cv1-q40](../runs/20261011_110345_7173b7cc/cv1-q40-result.json) | `cross_modal_reasoning` | 通过               | entity_type=equation、summary 最多 100 words，说明语义解释优于仅转录 LaTeX，正确。         |      8 |      1/1 |
| [cv1-q41](../runs/20261011_110345_7173b7cc/cv1-q41-result.json) | `unanswerable`          | 通过               | 正确指出论文未报告 95% 置信区间，没有自行估计。                                            |     11 |       — |
| [cv1-q42](../runs/20261011_110345_7173b7cc/cv1-q42-result.json) | `unanswerable`          | 通过               | 正确指出 δ 的具体实验取值未报告。                                                         |      4 |       — |
| [cv1-q43](../runs/20261011_110345_7173b7cc/cv1-q43-result.json) | `unanswerable`          | 通过               | 正确指出三项融合信号的精确权重未报告，没有编造公式。                                       |      9 |       — |
| [cv1-q44](../runs/20261011_110345_7173b7cc/cv1-q44-result.json) | `unanswerable`          | 通过               | 正确指出原表仅有 2020/2019/2018，无 2021 工资数据。                                        |     12 |      1/1 |
| [cv1-q45](../runs/20261011_110345_7173b7cc/cv1-q45-result.json) | `unanswerable`          | 通过               | 正确区分视觉判断与定量指标，未编造 silhouette score。                                      |      9 |      1/1 |
| [cv1-q46](../runs/20261011_110345_7173b7cc/cv1-q46-result.json) | `unanswerable`          | 通过               | 正确拒绝从论文推断本机耗时与通过率。                                                       |      9 |       — |
| [cv1-q47](../runs/20261011_110345_7173b7cc/cv1-q47-result.json) | `robustness`            | 通过               | 同义改写仍得到 76.3%、59.7%、16.6 个百分点，与 q06 一致。                                  |     10 |       — |
| [cv1-q48](../runs/20261011_110345_7173b7cc/cv1-q48-result.json) | `robustness`            | 通过               | 英文题仍得到 DKK 1,443 million、5.70%，与中文 q21 一致。                                   |     12 |      1/1 |

以下两点按语义接受，避免用参考文本的字面形式过度扣分：q34 给出了 59.7→76.3、46.8→46.0 及变化方向，虽然没有单独写出 16.6 和 -0.8；q38 已否定“全部解决”，并分别说明两幅反例的文本偏置/粒度错位和表格歧义，不要求逐字重复参考中的所有表述。

q20 所说的“Figure 2 左上角”指 Figure 3 内被引用源图自己的编号；目标 DAE/Style 位置和解释正确，本次不按目标错误扣分，表达时最好明确写“论文 Figure 3 中嵌入的源 Figure 2”。q12/q19 的公式转写中也带有解析文本的符号噪声，三类索引对象的语义正确；本题评价不要求精确 LaTeX 转录。

### 6. 错误依据与可能发生的环节

#### q04：失败模式正确，改进对象对应不准确

答案识别了文本中心检索偏差与结构处理困难，也给出两个改进名称；但把“自适应空间推理”单独归给文本检索偏差。A.5 明确把文本偏差关联到跨模态注意力不足，把默认扫描无法适应非线性结构关联到空间推理。由于题目要求“分别需要怎样改进”，这里按对应关系不准确判为部分正确。更合适的回答是：改善视觉取证和跨模态注意力，减少文本偏置；用自适应空间推理与布局感知解析处理复杂布局。

这是语义归纳与映射问题；仅凭记录不能证明哪一条上下文具体诱发了错误。

#### q07、q37：Table 3 的原始表已提供，仍出现排序错误

q07 的 Admin. 排名应为 MMGraphRAG 46.9%、GPT-4o-mini 46.8%、RAGAnything 45.7%，所以 RAGAnything 排第三。q37 的 Broch. 列应为 LightRAG 41.0% 高于 RAGAnything 34.0%，不能把 RAGAnything 说成第一。

这两题的诊断检索和实际生成提示中都包含正确的 Table 3 HTML，Table 3 原图也实际送入。错误不能归结为表格未检到或图片被拦截。与此同时，建库生成的表格附加分析存在领域列与数值错配，并出现 RAGAnything 各领域都优于基线的过度概括；正确表格与错误概括同时进入上下文。这是可能干扰排序的因素，但记录不能证明模型内部究竟依赖了哪段信息。

建议把原始表单元格作为排名计算依据，逐列比较全部方法；表格自动分析应与原始数据分开核验。

#### q22：数据读对，行—列—单位约束答错

2020 年工资 26,778、股权支付 823 均正确。问题进一步要求结合第 3.4 节说明定位维度，原文强调 row header、column label、tabular units。答案以 data cell 代替单位，虽然数值旁写了 DKK million，仍没有正确回答这项机制问题。

这是跨模态解释中的结构语义错误，不能只按数字相同就判全对。

#### q24、q26：核心答案正确，附加说明引入错误

q24 的两个标签与移除组件都正确；原图中 -A 是绿色，紫色对应 -R，答案却额外说 -A 是紫色。q26 的处理阶段和自下而上方向正确；原图节点名是 Word CRF，答案增加了没有依据的 `-F` 后缀。两题的目标原图都已送入。

这两题单独计入“核心通过”，完整通过时扣除，便于区分主问题失败与细节不可靠。

#### q25：正确原图已送入，但跨列信息漏答

Figure 6 原始表的 GloVe + LSTM – Attention 对应 Evidence Inference AUPRC 0.506、BoolQ AUPRC 0.525，差值 0.019。答案只完成前半题，错误声称没有 BoolQ 数据。

保存的建库图片描述主要复述案例的 Evidence Inference 结论，没有提取 BoolQ 的 0.525；实际检索提示里也没有这一数值。但正确 Figure 6 原图已经进入视觉生成请求。可确认的是：文字代理未覆盖扩展问题所需的跨列事实，生成环节又未从原图补回，形成错误拒答；不能说目标图未命中。

#### q27：图片描述中已有取数错误，生成答案继续沿用

原始 Figure 12 的 Train/Joint 列中，Training 1% New Domain 是 **44.24**，GEM 是 **54.31**，增益 **10.07 个百分点**。答案把基线写成 54.21，导致虽然减法本身成立，最终增益 0.10 仍错误。

核对发现，建库生成的 Figure 12 描述已经把该基线读成 54.21；实际生成提示包含该错误数值，没有正确的 44.24。目标原图已送入，而且原图中的目标值可辨认。这里有明确的上游描述错误与下游答案重复该错误的链条，生成阶段未完成原图纠正。至于模型是否优先相信文字代理，属于合理推测，无法单凭日志证明。

因此，后续最值得优先验证的是**图片描述的数值保真、单元格对应关系，以及答案生成时对原图的复核**。错误输入上的正确算术不能弥补取数错误。

### 7. 原图命中、性能与用量

#### 原图与证据利用

本轮 48 题全部使用视觉增强，即使是纯正文或公式语义题，也可能送入图片。累计发送 464 张次，每题平均 9.67 张，最少 3 张、最多 13 张。

参考标注预先指定原图的 21 题共有 22 次目标图要求（q38 同时要求两幅图）；用原始文件及实际 data URL 指纹逐题匹配，22/22 均已送入，未发现无法对应项目原图的已记录 payload。因此，本轮这些目标的路径读取与图片传入已通过核对。

这个结果只能支持“目标图进入生成适配器”，不能支持“模型准确读取、充分使用全部目标证据”，也不能推出检索精度为 100%。每题还有较多额外图片，可能增加噪声与输入成本；本次没有逐张标注相关性，所以没有计算图片检索 precision。

#### 性能和记录用量


| 指标                     |                                        数值 |
| ------------------------ | ------------------------------------------: |
| 单题计时合计             |                 475.91 秒，约 7 分 55.91 秒 |
| 目录创建到最后结果写盘   | 约 478.99 秒，7 分 58.99 秒；由文件时间估算 |
| 平均 / 中位单题耗时      |                              9.91 / 9.91 秒 |
| 最快 / 最慢单题          |                q06：8.12 秒 / q01：12.25 秒 |
| 本地重排回调             |                    96 次，均成功；每题 2 次 |
| 重排耗时合计             |          287.25 秒，占单题计时合计约 60.36% |
| 重排器加载时间           |                  约 0.66 秒，未计入各题耗时 |
| 成功文本关键词事件       |                                       96 次 |
| 成功视觉生成事件         |                                       48 次 |
| 实际生成检索提示记录     |                                       48 份 |
| 已记录请求失败           |                     0；本轮没有`error.json` |
| `prompt_tokens` 合计     |                                   1,258,515 |
| `completion_tokens` 合计 |                                      11,936 |
| `total_tokens` 合计      |                                   1,270,451 |
| `call_count` 合计        |        240，其中包含本地 embedding 用量记录 |

由于入口先执行诊断检索，再执行实际生成检索，同一道题做了两轮关键词抽取、向量查询和重排。上述耗时、调用数与 token 是这一审计流程的实际成本，不能直接当作只检索一次的生产服务性能。若后续优化，需保留必要的实际生成证据审计，并另测去掉重复诊断检索后的结果，不能直接把本轮时间除以二。

