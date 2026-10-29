# RAG-Anything 图检索算法


RAG-Anything 负责多模态内容与图片恢复，LightRAG 负责关键词提取、图检索、证据整理和回答生成。

## 1. 检索准备
检索开始前，建库阶段已经准备好以下数据：

| 数据 | 保存的内容 | 检索用途 |
|---|---|---|
| 知识图谱 | 实体节点、关系边、描述、来源编号 | 查询实体之间的连接，读取节点和关系属性 |
| 实体向量库 `entities_vdb` | 实体名称与实体描述的向量 | 找到与具体对象相关的实体 |
| 关系向量库 `relationships_vdb` | 关系关键词、两端实体名称、关系描述的向量 | 找到与问题主题相关的关系 |
| 内容块向量库 `chunks_vdb` | 正文块与多模态描述块的向量 | 按整句问题寻找相关内容 |
| 内容块存储 `text_chunks` | 块正文、文档编号、来源路径 | 根据块编号读取完整证据 |

图片、表格和公式进入索引时，会形成对象节点、详细描述块及细粒度实体。细粒度实体通过 `belongs_to` 关系连接到所属对象。例如，“文本编码器”可以关联到“Figure 1 系统架构图”，并通过同一个来源块找到该图的描述与图片路径。

**关键函数：**

- [`BaseModalProcessor._create_entity_and_chunk()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/raganything/modalprocessors.py:534)：写入多模态描述块、内容块向量和对象实体。
- [`BaseModalProcessor._process_chunk_for_extraction()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/raganything/modalprocessors.py:792)：从描述中抽取实体与关系，添加 `belongs_to` 连接。

## 2. 查询入口：选择实体、关系和内容块的组合

[`QueryMixin.aquery()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/raganything/query.py:128) 接收问题、检索模式和查询参数，将请求交给 LightRAG。启用视觉增强时，请求进入 `aquery_vlm_enhanced()`。

LightRAG 的 [`aquery_llm()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/lightrag.py:2884) 根据模式选择处理路线：

| 模式 | 检索路线 | 证据组成 |
|---|---|---|
| `local` | 从具体实体进入图谱 | 命中实体、相邻关系、关联内容块 |
| `global` | 从主题关系进入图谱 | 命中关系、两端实体、关联内容块 |
| `hybrid` | 合并 `local` 与 `global` | 两路图谱证据及其关联内容块 |
| `mix` | `hybrid` 加整句问题的内容块检索 | 图谱证据、关联内容块、直接召回的内容块 |
| `naive` | 按整句问题检索内容块 | 直接召回的内容块 |
| `bypass` | 由语言模型直接回答 | 问题与对话信息 |

`local` 侧重“对象”，`global` 侧重“联系”，`mix` 同时利用对象、联系和完整内容。

图检索模式统一进入 [`kg_query()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3164)。该函数组织关键词提取、检索上下文和回答生成。

## 3. 问题分析：提取两类关键词，分别生成向量

### 3.1 两类关键词各有用途

[`get_keywords_from_query()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3374) 优先使用查询参数中的关键词；自动提取时调用 [`extract_keywords_only()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3406)，由语言模型返回两组词：

| 字段 | 表达的内容 | 主要检索对象 |
|---|---|---|
| `low_level_keywords` | 实体名称、具体术语、对象细节 | 实体向量库 |
| `high_level_keywords` | 主题、关系、概括性概念 | 关系向量库 |

例如，对于“Figure 1 中，文本编码器如何参与检索？”，可用以下词组理解两路分工：

- 具体对象词：`Figure 1`、`Text Encoder`。
- 主题关系词：`retrieval`、`semantic matching`。


### 3.2 为三条检索路线准备查询向量

[`_perform_kg_search()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3573) 根据模式，将需要的文本批量交给嵌入函数：

- 具体对象词向量：查询实体。
- 主题关系词向量：查询关系。
- 完整问题向量：查询内容块，并为图关联块的相似度筛选提供依据。

[`NanoVectorDBStorage.query()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/nano_vector_db_impl.py:146) 接收查询向量，调用底层向量库计算余弦相似度，保留达到阈值的 Top-K 结果。

## 4. 实体路线：向量命中实体，到 GraphML 图谱中读取关联证据

实体路线分为三个环节：**用向量找到实体、按名称定位图节点、沿邻接边读取关系**。入口是 [`_get_node_data()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4359)。

### 4.1 加载 GraphML：把文件转换为内存中的图

当前知识库的图结构保存在 [graph_chunk_entity_relation.graphml](/Users/drw/PycharmProjects/RAG-Anything-reproduction/storage/handson-qwen4096-v1/graph_chunk_entity_relation.graphml) 中。

GraphML 文件主要保存两类对象：

| 对象 | 标识方式 | 主要属性 |
|---|---|---|
| 节点 | 实体名称，如 `Text Encoder` | 实体类型、描述、来源块编号 |
| 边 | 两端实体名称，如 `Text Encoder` 与 `Vector Database` | 关系描述、关键词、权重、来源块编号 |

初始化图存储时，[`NetworkXStorage.__post_init__()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:40) 确定 GraphML 文件路径，再调用 `load_nx_graph()`。该函数通过 `nx.read_graphml()` 解析文件，建立内存中的 NetworkX 图对象。

加载后，程序可以通过以下结构访问数据：

- 节点属性：`graph.nodes[实体名称]`。
- 边属性：`graph.edges[实体A, 实体B]`。
- 节点的邻接关系：`graph.edges(实体名称)`。

查询阶段复用这个内存图。[`_get_graph()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:80) 负责取得图对象

### 4.2 找到入口：实体向量库返回名称，名称对应图节点

`_get_node_data()` 首先调用：

`entities_vdb.query(query, top_k=top_k, query_embedding=query_embedding)`

这里的 `query` 是具体对象关键词，例如 `Text Encoder`。查询向量与实体向量进行余弦相似度比较，返回达到阈值的 Top-K 实体记录。

每条记录携带 `entity_name`。程序提取这些名称，形成种子实体列表，例如：

`Text Encoder`、`Vector Database`。

这一步完成了两种数据之间的连接：

| 数据层 | 使用的标识 | 作用 |
|---|---|---|
| 实体向量库 | 向量记录 ID，以及记录中的 `entity_name` | 找到语义相关的实体 |
| GraphML 图谱 | 节点 ID，即实体名称 | 精确定位对应节点 |

例如，向量结果携带 `entity_name="Text Encoder"`，程序随后用这个名称读取 GraphML 加载后的同名节点。

### 4.3 读取节点：取得实体属性和连接数量

获得种子实体名称后，`_get_node_data()` 通过 `asyncio.gather()` 同时调度两组读取：

1. `get_nodes_batch(node_ids)`：读取节点属性。
2. `node_degrees_batch(node_ids)`：读取节点度数。

随后，程序将结果整理为实体记录，保留：

- `entity_name`：实体名称。
- `entity_type`：实体类型。
- `description`：实体描述。
- `source_id`：来源块编号。
- `rank`：该节点的度数。

**实体列表继续保持向量相似度顺序。** 度数作为附加属性写入记录。

### 4.4 读取邻接边：围绕种子实体提取一跳连接

接下来，`_get_node_data()` 调用 [`_find_most_related_edges_from_entities()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4419)。

该函数提取种子实体名称，再调用 `get_nodes_edges_batch(node_names)`。当前实现逐个调用 [`NetworkXStorage.get_node_edges()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:129)，核心操作是：

`list(graph.edges(source_node_id))`

这一步利用内存图的邻接结构，取得所有与指定节点直接相连的边。

本轮扩展起点固定为种子实体列表。返回的实体列表由种子节点组成，邻接对象的名称通过关系端点进入证据。

### 4.5 整理关系：按端点对去重，补齐边属性

多个种子实体可能共享同一条边。例如，`Text Encoder` 和 `Vector Database` 同时成为种子时，两次邻接读取都可能取得它们之间的连接。

程序将每条边的两个端点排序，形成统一的端点对，再用集合记录已收集的关系。同一实体对保留一条候选关系。

随后，程序同时读取两类信息：

1. `get_edges_batch()`：取得关系描述、关键词、权重和来源编号。
2. `edge_degrees_batch()`：计算关系两端节点的度数之和。

底层 [`get_edge()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:123) 使用 `graph.edges.get((source_node_id, target_node_id))` 读取边属性。

完成后，每条候选关系包含：

| 字段 | 含义 |
|---|---|
| `src_tgt` | 两端实体名称 |
| `description` | 这两个实体之间的联系 |
| `keywords` | 关系关键词 |
| `weight` | 图谱中保存的关系权重 |
| `source_id` | 支撑该关系的内容块编号 |
| `rank` | 两端节点的度数之和 |

### 4.6 排列关系：先比较连接度，再比较权重

[`NetworkXStorage.edge_degree()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:117) 按以下方式计算关系的 `rank`：

> 关系连接度 = 端点 A 的度数 + 端点 B 的度数。

`_find_most_related_edges_from_entities()` 随后按 `(rank, weight)` 降序排列候选关系：

1. 连接度较高的关系排在前面。
2. 连接度相同时，权重较高的关系排在前面。

假设取得以下三条关系：

| 关系 | 端点度数之和 | 权重 | 排序位置 |
|---|---:|---:|---:|
| 编码器与向量库 | 10 | 1 | 1 |
| 编码器与架构图 | 6 | 10 | 2 |
| 编码器与文本信息 | 6 | 2 | 3 |


### 4.7 返回证据：保留实体、关系和来源线索

`_get_node_data()` 最终返回两个列表：

- `node_datas`：种子实体的属性与描述。
- `use_relations`：去重、补齐属性并排序后的相邻关系。

后续流程先合并各路图检索结果，再按实体与关系预算保留候选。来源回查函数随后读取保留记录中的 `source_id`：

- `_find_related_text_unit_from_entities()`：根据实体找到来源块。
- `_find_related_text_unit_from_relations()`：根据关系找到来源块。

例如，“文本编码器属于 Figure 1”这条关系携带一个来源块编号。程序用该编号取回图片描述块，获得架构说明与原图路径。

由此，一次实体检索最终提供三个层次的证据：**实体解释对象，关系补充联系，来源块提供完整细节。**

## 5. 关系路线：从主题关系进入图谱，补齐参与实体和来源证据

关系路线先根据问题主题找到相关关系，再利用关系两端的实体名称读取图节点，形成**关系说明、参与对象、来源内容**三类证据。总入口是 [`_get_edge_data()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4634)。

### 5.1 关系索引：将一条关系表达为可检索的文本

建库阶段，每条关系会形成一段用于嵌入的文本，主要包含：

- 关系关键词，如 `semantic matching`。
- 两端实体名称，如 `Vector Database` 和 `Semantic Matching`。
- 关系描述，如“向量数据库为语义匹配提供内容向量”。

源码将这些信息拼接为：

`关系关键词 + 端点A名称 + 端点B名称 + 关系描述`

随后，嵌入模型将这段文本转换为向量，存入 `relationships_vdb`。

关系向量记录还携带以下定位信息：

| 字段 | 作用 |
|---|---|
| `id` | 标识向量库中的关系记录 |
| `src_id` | 关系的一个端点实体名称 |
| `tgt_id` | 关系的另一个端点实体名称 |
| `source_id` | 关系对应的来源块编号 |

其中，`src_id` 和 `tgt_id` 连接向量检索结果与 GraphML 图谱中的边。

### 5.2 主题召回：用高层关键词寻找相关关系

问题分析阶段，`extract_keywords_only()` 提取 `high_level_keywords`，表达问题关注的主题和联系。

例如，问题是“结构检索与语义匹配如何配合”，主题关键词可以包含：

`structural navigation`、`semantic matching`、`hybrid retrieval`。

这些词组用于说明流程，实际内容由当次关键词提取结果决定。

`_get_edge_data()` 随后调用：

`relationships_vdb.query(keywords, top_k=top_k, query_embedding=query_embedding)`

[`NanoVectorDBStorage.query()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/nano_vector_db_impl.py:146) 使用主题关键词向量，与关系向量计算余弦相似度，返回达到阈值的 Top-K 关系记录。


### 5.3 GraphML 定位：根据两个端点读取完整关系

GraphML 加载后，节点和边已经保存在内存中的 NetworkX 图对象中。

收到向量结果后，`_get_edge_data()` 为每条关系提取两个端点，整理为：

`{"src": 实体A名称, "tgt": 实体B名称}`

随后调用 `get_edges_batch()` 读取这些实体对之间的边。

当前 `NetworkXStorage` 使用 [`BaseGraphStorage.get_edges_batch()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/base.py:535) 的实现，逐个调用底层 `get_edge(src_id, tgt_id)`。

[`NetworkXStorage.get_edge()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:123) 的核心操作是：

`graph.edges.get((source_node_id, target_node_id))`

例如，向量结果携带：

- `src_id="Knowledge Graph"`。
- `tgt_id="Structural Navigation"`。

程序便使用这两个名称定位对应边，并读取该边的属性：

| 属性 | 提供的信息 |
|---|---|
| `description` | 两个实体之间的具体联系 |
| `keywords` | 概括关系的关键词 |
| `weight` | 建库时保存的关系权重 |
| `source_id` | 支撑关系的内容块编号 |
| `file_path` | 关系对应的来源文件路径 |

**向量检索负责判断主题相关性，图存储负责提供完整的关系属性。**

### 5.4 保留顺序：关系沿语义相似度排列

取得边属性后，`_get_edge_data()` 按向量结果的原有顺序，重新构建 `edge_datas` 列表。

每条记录组合两类信息：

- 向量结果中的 `src_id`、`tgt_id` 等定位信息。
- 图谱中的关系描述、关键词、权重和来源信息。

关系继续保持主题向量召回的相似度顺序。`weight` 随关系属性一起保留，供后续记录和处理使用。

### 5.5 提取端点：从命中关系中收集参与对象

关系读取完成后，程序调用 [`_find_most_related_entities_from_relationships()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4693)。

该函数依次遍历 `edge_datas`，对每条关系执行：

1. 读取 `src_id`，将首次出现的名称加入实体列表。
2. 读取 `tgt_id`，将首次出现的名称加入实体列表。
3. 使用集合记录已收集的名称，完成实体去重。

沿用上面的三条示例关系，实体收集过程如下：

| 处理位置 | 读到的端点 | 新增实体 |
|---|---|---|
| 第 1 条关系 | `Knowledge Graph`、`Structural Navigation` | 两个实体均加入 |
| 第 2 条关系 | `Semantic Matching`、`Vector Database` | 两个实体均加入 |
| 第 3 条关系 | `Semantic Matching`、`Structural Navigation` | 两个实体均已收集 |

最终实体顺序为：

`Knowledge Graph`、`Structural Navigation`、`Semantic Matching`、`Vector Database`。

### 5.6 补齐节点：取得每个参与实体的属性

收集端点名称后，函数调用：

`knowledge_graph_inst.get_nodes_batch(entity_names)`

当前实现逐个调用 [`NetworkXStorage.get_node()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/kg/networkx_impl.py:107)，通过以下操作读取节点：

`graph.nodes.get(node_id)`

程序取得节点的类型、描述、来源编号等属性，再补入 `entity_name`，构建实体列表 `node_datas`。


### 5.7 返回结果：关系与实体共同进入证据整理

`_get_edge_data()` 最终返回两个列表：

| 返回值 | 内容 | 顺序依据 |
|---|---|---|
| `edge_datas` | 命中关系及其完整属性 | 关系向量的相似度顺序 |
| `use_entities` | 关系两端实体及其属性 | 端点名称首次出现的顺序 |

后续流程将它们与实体路线的结果合并，再按照实体和关系的长度预算保留候选。


## 6. 候选融合：补充内容块，轮流合并图谱结果

### 6.1 `mix` 加入完整问题的内容块召回

在 `mix` 模式下，`_perform_kg_search()` 还会调用 [`_get_vector_context()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3516)，使用完整问题查询 `chunks_vdb`。

这一条路线可以直接命中包含实验数值的正文、完整表格描述或图片结构说明。候选内容块先保留，随后参与统一合并。

### 6.2 两路实体和关系分别交替合并

`_perform_kg_search()` 采用轮流取结果的方式：

- 实体：实体路线第 1 个、关系路线第 1 个、实体路线第 2 个、关系路线第 2 个……
- 关系：实体路线第 1 条、关系路线第 1 条、实体路线第 2 条、关系路线第 2 条……

实体以名称去重，关系以规范化后的端点对去重。这样可以保留两条路线各自靠前的证据。

### 6.3 先为图谱证据分配长度预算

[`_apply_token_truncation()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3783) 分别按照 `max_entity_tokens` 和 `max_relation_tokens`，从合并结果前部保留能装入预算的实体与关系。

后续来源块回查使用这批保留结果。因此，实体和关系预算也会影响进入下一阶段的来源块范围。

## 7. 来源回查：把图谱知识还原成正文和多模态描述

图节点和关系中的 `source_id` 保存来源块编号。多个编号通过 `<SEP>` 连接。系统沿这些编号读取完整内容，得到支撑图谱知识的证据。

**关键函数及工作：**

- [`_find_related_text_unit_from_entities()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4475)：收集保留实体的来源块。
- [`_find_related_text_unit_from_relations()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4726)：收集保留关系的来源块，并与已选实体来源块去重。
- `text_chunks_db.get_by_ids()`：按选中的块编号批量读取正文与元数据。

例如，“文本编码器属于 Figure 1”这条关系指向一个图片描述块；该块包含图的详细说明和 `Image Path:` 路径。检索由此从概念联系到图像证据。

### 7.1 `VECTOR`：在图关联块中按问题相似度筛选

默认策略是 `VECTOR`。它调用 [`pick_by_vector_similarity()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/utils.py:2491)：

1. 汇总图谱结果关联的块编号。
2. 读取这些块的已有向量。
3. 计算各块与完整问题的余弦相似度。
4. 根据关联对象数量和 `related_chunk_number` 计算配额，选择相似度靠前的块。

图连接负责限定候选范围，向量相似度负责在这个范围内排序。

### 7.2 `WEIGHT`：按对象顺序和共同引用次数分配证据

配置为 `WEIGHT` 时，系统调用 [`pick_by_weighted_polling()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/utils.py:2411)：

1. 统计每个块被候选实体或关系共同引用的次数。
2. 将共同引用次数较高的块放在各自列表前部。
3. 按候选对象顺序分配线性递减的块配额。
4. 将剩余配额分配给仍有候选块的对象。

例如，5 个对象、最高配额 5、最低配额 1 时，初始配额依次为 5、4、3、2、1。[^3]

## 8. 内容块整理：三路合并，再按问题相关性重排

### 8.1 三路内容块轮流进入候选列表

[`_merge_all_chunks()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:3954) 接收三类内容块：

| 来源 | 获取方式 |
|---|---|
| 直接召回块 | 完整问题查询内容块向量库 |
| 实体关联块 | 根据实体的 `source_id` 回查 |
| 关系关联块 | 根据关系的 `source_id` 回查 |

合并顺序为：直接召回第 1 块、实体关联第 1 块、关系关联第 1 块，然后继续取第 2 块。相同 `chunk_id` 保留首次出现的位置。

### 8.2 重排模型对候选块重新打分

[`process_chunks_unified()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/utils.py:2755) 负责最后一轮块筛选。启用重排且已配置模型时，它通过 [`apply_rerank_if_enabled()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/utils.py:2671) 将“问题＋候选块”交给重排模型，按相关性分数重新排序。

后续依次执行重排分数阈值筛选、`chunk_top_k` 数量限制和 token 长度限制。重排作用于合并后的**内容块列表**；

## 9. 上下文组装

[`_build_context_str()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/operate.py:4056) 将检索结果整理为四部分：

| 部分 | 提供的信息 |
|---|---|
| 实体 | 对象名称、类型与描述 |
| 关系 | 对象之间的联系与关系描述 |
| 内容块 | 正文段落、图片说明、表格和公式描述 |
| 参考来源 | 内容块所属文件及引用编号 |


[`generate_reference_list_from_chunks()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/lightrag/utils.py:3341) 按文件路径组织参考来源，为内容块分配引用编号。随后，`kg_query()` 将上下文填入 `PROMPTS["rag_response"]`，调用语言模型生成回答。


## 10. 视觉增强

视觉增强从已经选中的检索证据继续处理，核心入口是 [`aquery_vlm_enhanced()`](/Users/drw/PycharmProjects/RAG-Anything-reproduction/.venv/lib/python3.12/site-packages/raganything/query.py:384)。[^2]

| 顺序 | 官方函数 | 完成的工作 |
|---|---|---|
| 1 | `aquery_vlm_enhanced()` | 以 `only_need_prompt=True` 获取检索提示词 |
| 2 | `_process_image_paths_for_vlm()` | 提取 `Image Path:` 图片路径，校验文件和允许目录，编码图片 |
| 3 | `_build_vlm_messages_with_images()` | 将图片与对应文字片段放入同一条多模态消息 |
| 4 | `_call_vlm_with_multimodal_content()` | 调用视觉模型，结合问题、文字证据和原图生成回答 |


