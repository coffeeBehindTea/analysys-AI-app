# Day 3–4 文档摄取日志

- Embedding 模型：`embedding-3`
- Chroma Collection：`robot_knowledge_v4`
- Chroma 持久化目录：`chroma_data`
- Chunk 参数：`800/120`
- 文件总数：5；成功 0；跳过 5；失败 0

## 逐文件结果

| 文件 | 结果 | Document ID | 位置数 | Chunk 数 | 说明 |
|---|---|---|---:|---:|---|
| ID_reader_COGNEX_Reference_Manual_DM260.pdf | skipped | `-` | - | - | 文档已经存在：ID_reader_COGNEX_Reference_Manual_DM260.pdf |
| victron energy_Manual_Lithium_Smart_Battery_en.pdf | skipped | `-` | - | - | 文档已经存在：victron energy_Manual_Lithium_Smart_Battery_en.pdf |
| 京东无人仓场景-仓储机器人测试规程与判定标准.md | skipped | `-` | - | - | 文档已经存在：京东无人仓场景-仓储机器人测试规程与判定标准.md |
| 仓储机器人故障说明.txt | skipped | `-` | - | - | 文档已经存在：仓储机器人故障说明.txt |
| 工业移动机器人安全标准.pdf | skipped | `-` | - | - | 文档已经存在：工业移动机器人安全标准.pdf |


## 当前 V4 Collection 实际状态

> 本日志在首次成功摄取后，被第二次重复摄取执行结果覆盖。
> 上方“成功 0、跳过 5”描述的是第二次执行。
> 以下数据通过 `ChromaVectorStore.list_documents()` 从已经保存的
> Chunk 元数据中恢复，不会再次调用 Embedding 服务。

- Collection：`robot_knowledge_v4`
- 文档数：5
- Chunk 总数：355
- 文档状态：全部为 `ready`

| 文件 | Document ID | 位置数 | Chunk 数 | 状态 |
|---|---|---:|---:|---|
| ID_reader_COGNEX_Reference_Manual_DM260.pdf | `e71c6c5a3c4e97ab6954cfe747cd887187432b42fab6c471b432257adab2b481` | 59 | 98 | ready |
| victron energy_Manual_Lithium_Smart_Battery_en.pdf | `190a3d6b0f28bc063e7786e70841997513949bd2e8d8cecbaf8ba7895fe5e2d0` | 45 | 165 | ready |
| 京东无人仓场景-仓储机器人测试规程与判定标准.md | `5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80` | 35 | 35 | ready |
| 仓储机器人故障说明.txt | `dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652` | 4 | 4 | ready |
| 工业移动机器人安全标准.pdf | `6115fd3e165aa9899787bd0296aa8f6c6cc547163d0354f225c6d5e9e25732f8` | 34 | 53 | ready |


## 说明

- `ingested`：本次完成解析、切分、Embedding 和 Chroma 写入。
- `skipped`：相同文件内容已经存在，未再次调用 Embedding。
- `failed`：该文件未完成摄取，具体原因见表格。
- 日志只记录文件级统计和标识，不记录 Chunk 正文、向量或 API Key。
