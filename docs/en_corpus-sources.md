# Robot Knowledge Base Corpus Sources

Date of record: 2026-08-10
Current number of languages: 5

## 1. Purpose

This document records the language sources, authorizations, dissensions, content coverage and stable positioning patterns used by Robot Knowledge Base.

All original documents are saved locally only in `data/source/`, and are not submitted to Git. Project codes, assessment and evaluation reports shall not contain customer data, personal information, documented or unauthorized internal content.

Open downloading is not the same as open licences. The public vendor manual and draft standards may still retain copyright, so the project uses these copies only in local learning, decomposition and retrieval experiments and does not re-distribut the original document.

## 2. Language selection principles

The following types of information are used for this project:

- Self-generated research and development of analog robots;
- (a) publicly available vendor information;
- (a) Information which has been authorized;
- Information that is fully redacted and available for study and testing.

Language needs to be met:

- Clear source and use of borders;
- (a) Does not contain true customer or personal sensitive information;
- Content related to robotic operations, malfunctions, sensors, implementers or tests;
- Contains a stable number of pages or chapters;
- (b) Not to duplicate other language material in large quantities;
- An assessment that can support answers, cross-sections and no questions.

## 3. List of language materials

| source_file | Title | Type | Source type | Source or author | Use boundary | Main elements |
|---|---|---|---|---|---|---|
| `ID_reader_COGNEX_Reference_Manual_DM260.pdf` | DataMan 260 Series Reference Manual | PDF | Open vendor manual | Cognex Corporation | Vendors retain copyright; only local study and retrieval experiments are not distributed | ID Reader Configuration, Specification, Installation, Connection, Trigger, Cleaning and Maintenance |
| `victron energy_Manual_Lithium_Smart_Battery_en.pdf` | Lithium Smart Battery Manual | PDF | Open vendor manual | Victron Energy | Vendors retain copyright; only local study and retrieval experiments are not distributed | Lithium battery safety, BMS, installation, operation, malfunction and technical data |
| `工业移动机器人安全标准.pdf` | Industrial Mobile Robot Safety Standard (for exposure draft) | PDF | Draft public standards | China Association for the Application of Integrated Technologies in Mechanical and Electric Sciences | Public exposure drafts; not considered as entry into force criteria or legal opinions | Industrial mobile robotic risk, safety requirements, detection, assessment, training and emergency response |
| `仓储机器人故障说明.txt` | Description of storage robot failure | TXT | Self-generated simulations | Users of this learning project | Manual on real equipment not to be interpreted | Positioning, crash prevention, electric overheating and heartbeat failure. |
| `京东无人仓场景-仓储机器人测试规程与判定标准.md` | Test protocol and test criteria for storage robots in the open view of the JD.com Unmanned Area | Markdown | Self-generated simulations | This learning project is based on open scenes | Available for current learning projects; not official JD.com protocol | Safety, navigation, network, movement control, mission, battery, payload and continuous operation testing |

### 3.1 Completeness of experimental documentation

The following SHA-256 is calculated by the original byte of the actual experimental document, i.e. `document_id` for the ingestion chain. The original documents are not distributed in open warehouses; When the user obtains a document from its own legitimate source, it can use Hashi to determine whether it is fully consistent with this V4 trial version.

| source_file | SHA-256 / document_id |
|---|---|
| `ID_reader_COGNEX_Reference_Manual_DM260.pdf` | `e71c6c5a3c4e97ab6954cfe747cd887187432b42fab6c471b432257adab2b481` |
| `victron energy_Manual_Lithium_Smart_Battery_en.pdf` | `190a3d6b0f28bc063e7786e70841997513949bd2e8d8cecbaf8ba7895fe5e2d0` |
| `工业移动机器人安全标准.pdf` | `6115fd3e165aa9899787bd0296aa8f6c6cc547163d0354f225c6d5e9e25732f8` |
| `仓储机器人故障说明.txt` | `dd18de8c04d80f7805a8f70ad6c24857d0df55df8009dec2e3371c3ee50fc652` |
| `京东无人仓场景-仓储机器人测试规程与判定标准.md` | `5b11fafd10c2414a65e1fd6eed8d7a295ee707ebaa0da89e6470841887b7ab80` |

Hashi can only verify that the document is consistent and does not grant permission to copy or re-distribut. If Hashi is different, the new Collaction name should be used for re-absorption and assessment, and the V4 indicator in this report cannot be directly applied to the new document.

## 4. Document details

### 4.1 Cognex DataMan 260 Series Reference Manual

- `source_file`：`ID_reader_COGNEX_Reference_Manual_DM260.pdf`
- Title: DataMan 260 Series Reference Manual
- Format: PDF
- Pages: 59 PDF Physical Pages
- Document version: Review 6.1.11.1
- Document date: 2021-11-25
- Source type: Open vendor reference manual
- Author / Copyright owner: Cognex Corporation
- Source: Cognex Support, https://support.cognex.com/.
- Licensing/authorization statement: Document legal declaration indicates that the copyright is retained by Cognex Corporation and that reproduction and dissemination are restricted. Only local copies of the study are kept for the analysis and retrieval of experiments, and PDF is not submitted or redistributed.
- Sensitization statement: the manufacturer makes available product information that does not include client data, personal information or internal equipment numbers for the project.
- Main elements:
  - DataMan 260 products and models;
  - Parts, lenses, filters, lighting and cables;
  - Installation, size and installation of support racks;
  - Equipment specifications and field of view distance;
  - I/O, Ethernet, RS-232 and trigger;
  - Focus, training, cleaning, maintenance and attention.
- The range of questions that can be supported is the configuration of IID Reader's hardware, installation connection, trigger method, focus, maintenance and cleaning.
- This does not include: actual installation parameters for the current learning project robot, configuration of the JD.com warehouse, and the project's custom default code.
- Stabilization: `page: <PDF 物理页码，从 1 开始>`; subsequent ingestion saves the chapter title on the page.
- Known resolution characteristics: There is a lack of spaces between English words that partially extract the text, and subsequent PDF resolution and cut-off testing requires attention.

### 4.2 Victron Energy Lithium Smart Battery Manual

- `source_file`：`victron energy_Manual_Lithium_Smart_Battery_en.pdf`
- Title: Lithium Smart Battery Manual
- Format: PDF
- Pages: 45 PDF Physical Pages
- Document version: rev 21
- Document date: 2026-01
- Source type: Open vendor manual
- Author / Copyright Owner: Victron Energy
- Official PDF: https://www.victronenergy.com/upload/documents/Lithium_Battery_Smart/15958-Manual_Lithium_Smart_Battery-pdf-en.pdf
- Official HTML version: https://www.victronenergy.com/media/pg/Lithium_Battery_Smart/en/index-en.html
- Licensing/authorization instructions: A manual for publicly available manufacturers, copyright and use terms are based on Victor Energy. The project is used only in local learning and retrieval experiments and is not re-distributed to the original PDF.
- Sensitization statement: Vendors make product information publicly available and do not include customer data, personal information or current project key.
- Main elements:
  - Safety precautions for lithium batteries;
  - System design and BMS selection;
  - Installation, connectivity and configuration;
  - Operation, charging and maintenance;
  - (b) Hydrocore imbalances, BMS and communication failures;
  - Warnings, errors and technical data.
- The range of issues that can be supported: safety requirements for the product, BMS behaviour, installation operation, failure clearance and technical data.
- This does not include: the actual battery model of the JD.com warehouse robot; it cannot be inferred from the fact that the manual enters the knowledge base that the real robot uses the Victron Energy battery.
- Stable positioning: `page: <PDF 物理页码，从 1 开始>`; while keeping the number of the manual chapter, e. g. `section: 6.4 Warnings, alarms and errors`.

### 4.3 Industrial mobile robotic safety standards (advertising draft)

- `source_file` ： `工业移动机器人安全标准.pdf`
- Title: Industrial mobile robotic safety standards / Safety standard for industrial mobile robots
- Format: PDF
- Pages: 34 PDF Physical Pages
- Document status: exposure draft
- Document identifier: `T/CAMETA XXX—2024`
- Type of source: Draft open standard
- Presentation and dissemination of information: China Association for the Application of Integrated Technologies in Mechanical and Electric Sciences
- Officially published copy: https://www.cameta.org.cn/uploadfile/2024/0523/20240523042308986.pdf
- Licensing/authorization statement: A copy of the comments made publicly on the Association's website is subject to the conditions for copyright and reuse. Local copies are used only for learning and retrieving experiments and are not submitted or re-distributed.
- Redacted statement: public draft, without inclusion of client, project or real equipment data.
- Main elements:
  - (a) The terms industrial mobile robots, AMRs, etc.;
  - Hardware, software, operations, networks, physical and environmental risks;
  - (b) Safety design, operation and testing;
  - (b) Safety assessment and certification;
  - Training, maintenance and contingency planning.
- The range of issues that can be supported: industrial mobile robot security risks, general safety requirements, detection records, assessment validation and contingency arrangements.
- It does not contain: officially issued standard number and entry into force conclusion. It cannot be a separate demonstration of true project compliance or a substitute for applicable formal national standards, industry standards or on-site risk assessments.
- Stable positioning: `page: <PDF 物理页码，从 1 开始>`; while keeping chapter numbers and titles.

### 4.4 Description of the failure of the storage robot

- `source_file` ： `仓储机器人故障说明.txt`
- Heading: Description of the failure of the storage robot
- Format: TXT, UTF-8
- Source type: Self-generated simulations
- Author: Users of this learning project
- Licensing/authorization statement: Users prepare and explicitly authorize for use in the current RAG learning project.
- Redacted statements: the content uses analog failed codes and generic robotic terminology; no customer name, personal information, supporting evidence or identifiable real equipment number has been detected.
- Main records:
  - `ERR-NAV-2001`: Lost positioning;
  - `ERR-SAF-1002`: mechanical collision-proofing trigger;
  - `ERR-DRV-3005`: Driving electric overheating;
  - `ERR-NET-4001`: Schedule heart beats time out.
- The range of problems that can be supported: failure trigger conditions, possible causes, routing steps and levels of severity.
- This does not include: a genuine equipment manufacturer's commitment, formal security certification, real warehouse events and the real Kyoto failure code.
- Stable positioning: For subsequent ingestion of TXT, split by `故障代码 (ErrorCode)` records and use `section: <故障码>`, e. g. `section: ERR-NAV-2001`.
- Known data quality problems: Text adhesive with "Strategic Fault Code (ErrorCode)" at the end of record `ERR-DRV-3005`. The original sample is currently being retained and subsequently used to validate the records of clean-up and failure cases, and the original text should not be altered silently.

### 4.5 Storage robot test protocol and determination criteria in the open view of JD.com Unmanned

- `source_file` ： `京东无人仓场景-仓储机器人测试规程与判定标准.md`
- Title: Standard criteria and criteria for the testing of storage robots in the open view of the JD.com Unmanned Area
- Document number: `SIM-WH-TEST-001`
- Version: 1.0
- Format: Markdown
- Source type: Simulation material based on open scenes
- Author: This learning project
- Based on:
  - JD Corporate Blog Public Introduction to Asia One, Delivery, Ground 2D Navigation and Automatic Refusal: https://jdcorporateblog.com/top-international-media-visit-jds-asia-no-1-warehouse/
  - Public introduction to robotics, sensor access and cloud RCS movement: https://doc.thingtalk.jdl.com/
- Licensing/authorization statement: prepared for the current learning project, which can be used for analysis, splitting, Embedding, retrieval and evaluation of the project.
- Redacted statement: does not contain real warehouse maps, robotic numbers, internal software, client data or personnel information. All speeds, distances, power thresholds, failure codes, frequency and determination conditions are teaching project simulations.
- Main elements:
  - Test objects, system boundaries and roles;
  - Test environment, recording requirements and simulation baseline parameters;
  - (b) Surgery, shielding and mechanical crash tests;
  - Two-dimensional code and laser positioning tests;
  - Interruption of communications and control tests at intersections;
  - (a) Arrival, ID Reader, battery, payload and electric overheating tests;
  - Rules for failure recovery, continuous operation and overall determination.
- The range of questions that can be supported: test preconditions, implementation steps, simulation thresholds, pass/fail criteria and recording fields.
- It does not contain: internal test protocols, real production parameters, formal safety certification conclusions. The answer must clearly distinguish between the fact of the public scene and the self-generated threshold of this paper.
- Stable positioning: use the Markdown title path, e. g. `section: 10.1 TEST-NET-001 调度心跳中断/通过标准`.
- Cross-document relationship: Simulation failure code in test example `仓储机器人故障说明.txt`; ID Reader and battery-specific requirements should still be returned to the counterpart manufacturer's manual as evidence, and this simulation protocol cannot be invoked only.

## 5. Data submission rules

The following may be submitted:

- A description document for this language;
- The smallest analogue of `samples/corpus/robot-fault-demo.txt` that does not correspond to real customers or real equipment;
- Redacted assessment;
- Document resolution, splits, Embedding and search codes;
- Statistical results and evaluation reports do not contain the main paragraphs.

The following shall not be submitted:

- Original document in `data/source/`;
- The results of the analysis in `data/processed/`;
- Vector database in `chroma_data/`;
- `.env` and any API Key;
- Unsensitized logs, equipment identifiers or personal information.

`.gitignore` Currently used:

```gitignore
data/source/*
!data/source/.gitkeep
data/processed/
chroma_data/
```

Therefore, the complete text file and the new Markdown test protocol are kept locally and do not enter Git; `.gitkeep` is only used to retain the catalogue structure. The `samples/corpus/robot-fault-demo.txt` in the public warehouse is the smallest alternative simulation sample written for use only by code reviewers to verify TXT intake processes, and cannot replicate a complete V4 knowledge library or formal assessment indicator.

## 6. Current Text Status

- [x] 5 documents prepared
- [x] Contain at least one PDF
- [x] Contain at least one of the Markdowns
- [x] At least one TXT
- [x] Each document has a source or author description
- [x] Each document has copyright, uses boundaries or a self-authorization statement
- [x] Privacy and sensitive information checks have been completed for each document
- [x] Every document has a stable page number or chapter positioning scheme
- [x] Original document has been excluded from `.gitignore`

## 7. Attention for current language use

1. The manufacturer's manual describes only the counterpart products and cannot automatically be extended to other robotic components.
2. The standard document is a draft opinion and cannot be described as a mandatory standard that has entered into force.
3. Self-deficient codes and test thresholds are analogue data and cannot be described as Kyoto or internal parameters for the plant.
4. Subsequent references must come from the actual search of Chunk and return by code `source_file`, `page_or_section` and `chunk_id`, which cannot be attributed to LLM's own source.
5. The evaluation should be followed up with questions of unanswered and counter-facts that can test the above-mentioned boundary.
