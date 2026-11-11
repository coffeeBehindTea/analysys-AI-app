# Day 3–4 Document Ingestion Log

- Embedding Model: `embedding-3`
- Chroma Collection：`robot_knowledge_v4`
- Chroma Durable Directory: `chroma_data`
- Chunk parameter: `800/120`
- Total number of files: 5; success 0; skip 5; failure 0

## Document-by-document result

| Documentation | Result | Document ID | Number of places | Chuck Number | Annotations |
|---|---|---|---:|---:|---|
| ID_reader_COGNEX_Reference_Manual_DM260.pdf | skipped | `-` | - | - | Document already exists: ID_reader_COGNEX_Reference_Manual_DM260.pdf |
| victron energy_Manual_Lithium_Smart_Battery_en.pdf | skipped | `-` | - | - | Document already exists: victron energy_Manual_Lithium_Smart_Battery_en.pdf |
| Kyodong Unmanned Fields View - Warehousing Robot Test Protocol and Decision Criteria.md | skipped | `-` | - | - | Document already exists: JD.com Unmanned Site - Warehousing Robot Test Program and Decision Criteria.md |
| Warehousing robot failure statement.txt | skipped | `-` | - | - | Document already exists: description of storage robot failure. txt |
| Industrial mobile robot safety standards.pdf | skipped | `-` | - | - | Document already exists: Industrial Mobile Robot Safety Standard.pdf |


## Current V4 Collaction Actual Status

> This log is covered by the second repeated intake implementation after the first successful uptake.
> The above description of “success 0, skip 5” is the second execution.
> The following data was used from the stored data of `ChromaVectorStore.list_documents()`.
> Restores in the Chunk metadata without calling the Embedding service again.

- Collection：`robot_knowledge_v4`
- Number of documents: 5
- Chunk Total: 355
- Document status: All `ready`

| Documentation | Document ID | Number of places | Chuck Number | Status |
|---|---|---:|---:|---|
| ID_reader_COGNEX_Reference_Manual_DM260.pdf | `e71c6c5a3c4e97ab6954cfe747cd887187432b42fab6c471b432257adab2b481` | 59 | 98 | ready |
| victron energy_Manual_Lithium_Smart_Battery_en.pdf | `190a3d6b0f28bc063e7786e70841997513949bd2e8d8cecbaf8ba7895fe5e2d0` | 45 | 165 | ready |
| Kyodong Unmanned Fields View - Warehousing Robot Test Protocol and Decision Criteria.md | `5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80` | 35 | 35 | ready |
| Warehousing robot failure statement.txt | `dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652` | 4 | 4 | ready |
| Industrial mobile robot safety standards.pdf | `6115fd3e165aa9899787bd0296aa8f6c6cc547163d0354f225c6d5e9e25732f8` | 34 | 53 | ready |


## Annotations

- `ingested`: This is the final analysis, splits, Embedding and Chroma.
- `skipped`: The content of the same document already exists, and Embedding is not called again.
- `failed`: This document was not fully ingested for the reasons given in the table.
- Logs only record file-level statistics and identification, and do not record Chunk text, vector or API Key.
