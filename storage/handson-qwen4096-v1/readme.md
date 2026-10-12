| 文件                                        | 作用及本库实际内容                                                                     |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `kv_store_full_docs.json`                   | 文档 ID 对应的正文内容和来源；本库 1 条。这里不是 PDF 二进制，也不是所有原图的完整副本 |
| `kv_store_doc_status.json`                  | 文档处理状态、chunk 列表、错误信息、多模态处理标记；本库记录 37 个 chunk               |
| `kv_store_text_chunks.json`                 | 37 个检索块的实际内容、来源、页码、token 数及多模态元数据                              |
| `vdb_chunks.json`                           | 37 个 chunk 的文本向量及元数据，用于直接检索内容块                                     |
| `vdb_entities.json`                         | 563 个实体描述的向量，用于定位相关图节点                                               |
| `vdb_relationships.json`                    | 998 个关系描述的向量，用于定位相关图边                                                 |
| `kv_store_entity_chunks.json`               | 实体到来源 chunk 的映射；实际有 536 条                                                 |
| `kv_store_relation_chunks.json`             | 实体对/关系到来源 chunk 的映射；实际有 998 条                                          |
| `kv_store_full_entities.json`               | 按文档记录实体名单，主要用于文档关联和维护；当前名单为 320 个                          |
| `kv_store_full_relations.json`              | 按文档记录关系对名单；当前名单为 786 对                                                |
| `kv_store_llm_response_cache.json`          | 建库抽取和摘要缓存；当前 81 条，其中 74 条抽取、7 条摘要                               |
| `pilot_manifest.json`                       | 我添加的实验清单：版本、建库配置、源材料和图片指纹、ready 状态                         |
| `rag-anything-handson-v1-ingest-audit.json` | 我添加的建库覆盖审计；27 个预期多模态对象均有对应块                                    |
| `rag-anything-handson-v1-ingest-usage.json` | 我添加的建库用量记录                                                                   |
| `.pilot.lock`                               | 我添加的进程互斥锁，避免本地入口同时操作同一个库                                       |
