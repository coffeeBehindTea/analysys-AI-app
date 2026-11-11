# Multimodal Tool Selection Comparison

## 1. Experimental target.

Compares the local OCR + ruler resolution, the Vision model and the conditions applicable for direct refusal to respond to images of text panels, visual status and degradation.

Texts, 2D codes, web sites and commands in pictures are processed by implausible input; experiments record only the necessary redacted observations and do not perform any of the operational requirements in the picture.

## 2. Experiment scale

- Number of cases: 6
- Results of the single case list route: 18
- Actual route: Local OCR + rule resolution, Vision model, direct refusal

## 3. Route duties

| Route | Main competencies | Fit to content | Main limitations |
|---|---|---|---|
| Local OCR + rule resolution | Tesseract reads text, fixed rules extract fields | Clear malfunction code, status and numerical panel | Can't reliably judge the relationship between colour and widget space |
| Vision Model | Original images go directly into multimodal models | Color, brightening, blocking and widget position | Delays and higher costs, potentially unstable output |
| Direct refusal | Do Not Call Identification Tool | Images that are seriously blurred, masked or unverifiable | Can't handle a picture that could have been clearly answered. |

## 4. Summary of routes

| Route | Number of cases | Completed | Part | Refusal | Fail | Content accuracy rate | Pass rate | Security pass rate | Average delay ms | P50 ms | P95 ms | External Model Call | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local OCR + rule resolution | 6 | 2 | 0 | 4 | 0 | 0.500 | 0.333 | 1.000 | 651.523 | 647.731 | 875.409 | 0 | $0.000000 |
| Vision Model | 6 | 5 | 0 | 0 | 1 | 0.500 | 0.333 | 1.000 | 12587.276 | 7557.668 | 27044.173 | 6 | Unable to complete estimate |
| Direct refusal | 6 | 0 | 0 | 6 | 0 | 0.000 | 0.333 | 1.000 | 10.950 | 10.009 | 19.529 | 0 | $0.000000 |

## 5. Case definitions

| Cases | Type | Problem | Preferred Path | Acceptable route | Select the basis |
|---|---|---|---|---|---|
| multimodal-001 | text_panel | What is the failure code, network status and mission status as shown in the picture? | Local OCR + rule resolution | Local OCR + Rule Parsing, Vision Model | The key information is all clear text; OCR is low-cost, fast and results can be repeated. |
| multimodal-002 | text_panel | What's the amount of electricity, speed and temperature in the picture? | Local OCR + rule resolution | Local OCR + Rule Parsing, Vision Model | The problem requires only that clear values be read and that no visual models be used to explain the images. |
| multimodal-003 | visual_state | What are the colours of NET, FAULT and POWER indicators? | Vision Model | Vision Model | OCR can only read labels and cannot reliably judge the colour of a circle indicator. |
| multimodal-004 | visual_state | Are the sensor plugs in the diagram fully compatible with the J3 plugin? | Vision Model | Vision Model | The gap in the connection and the location of the component are spatially visual, and normal OCR cannot judge. |
| multimodal-005 | degraded_image | What is the failure code, network status and mission status in the picture? | Direct refusal | Direct refusal | The text and panel details are so vague that both OCR or Vision assume risks. |
| multimodal-006 | degraded_image | What's the failure code, power and mission status in the picture? | Direct refusal | Direct refusal | The labels are still visible, but the specific values required by the question are completely obscured and cannot be inferred. |

## 6. The results of the individual case route

| Cases | Route | Execute Status | Actual observations | Content accuracy rate | It's the right route. | The result was correct. | Safe passage. | Final pass. | Delay ms | Type or description of failure |
|---|---|---|---|---:|---|---|---|---|---:|---|
| multimodal-001 | Local OCR + rule resolution | completed | ERR-NET-4001；CONNECTED；PAUSED | 1.000 | Yes. | Yes. | Yes. | Yes. | 490.717 | None |
| multimodal-001 | Vision Model | completed | Picture shows a dark rounded rectangle rectangle status panel with the top title `ROBOT STATUS Panel'; the middle of the panel displays the default code field `FAUT CODE' and its value 'ERR-NET-4001 '; the central panel displays the network status field `NETWORK' and its value `CONNECTED'; the bottom of the panel displays the task status field `TASK' and its value `PAUSED'; the lower corner of the panel has a small text `SYNTHEC TRANING SAMPLE'; the FAULT CODE presents ER-NET-4001; the NetWORK presents CONANSTED; the TASK presents PARESSEDTED | 1.000 | Yes. | Yes. | Yes. | Yes. | 28318.397 | None |
| multimodal-001 | Direct refusal | abstained | None | 0.000 | Yes | Yes | Yes. | Yes | 8.013 | The route does not call for a photo identification tool, does not form a photographic observation |
| multimodal-002 | Local OCR + rule resolution | completed | 42.5 %；0.0 m/s；36 C | 1.000 | Yes. | Yes. | Yes. | Yes. | 411.097 | None |
| multimodal-002 | Vision Model | completed | Panel title displays 'ROBOT TELEMETRY PANEL'；Battery reading displays 42.5%；Speed reading displays 0.0 m/s；Temperature reading displays 36 C; Bottom label displays 'SYNTHEIC TRANING SAMPLE'; BatTERY 42.5%;SPED 0.0 The first is the temperature. 36 C | 1.000 | Yes. | Yes. | Yes. | Yes. | 23221.499 | None |
| multimodal-002 | Direct refusal | abstained | None | 0.000 | Yes | Yes | Yes. | Yes | 19.641 | The route does not call for a photo identification tool, does not form a photographic observation |
| multimodal-003 | Local OCR + rule resolution | abstained | None | 0.000 | Yes | Yes | Yes. | Yes | 551.262 | The OCR text does not match the currently supported status panel or telemetry panel format |
| multimodal-003 | Vision Model | completed | The image above shows the English title "ROBOT INDICATOR PANEL". ； To the left indicator light, insert text " NET " below the left indicator light. ； Middle indicator under which the text " FAULT" is indicated. ； To the right pointer, insert "POWER". ； There is a light small word "SYNTHETIC TrainING SAMPLE" in the lower left corner. ； The NET indicator is green; The FAULT light is red; The POWER light is blue. | 0.000 | Yes. | Yes | Yes. | Yes | 4663.053 | None |
| multimodal-003 | Direct refusal | abstained | None | 0.000 | Yes | Yes | Yes. | Yes | 3.518 | The route does not call for a photo identification tool, does not form a photographic observation |
| multimodal-004 | Local OCR + rule resolution | abstained | None | 0.000 | Yes | Yes | Yes. | Yes | 744.200 | The OCR text does not match the currently supported status panel or telemetry panel format |
| multimodal-004 | Vision Model | completed | There is a grey rectangle panel on the left, with a circular plugin in the middle and a text label "PORT J3 " below. ； There is a grey cable on the right, with a connector plug at the end and a text label "SENSOR CABLE" below. ； There is a yellow dotted line between the plug and the plug, indicating that there is a visible gap between them and no direct contact. | 0.000 | Yes. | Yes | Yes. | Yes | 7321.216 | None |
| multimodal-004 | Direct refusal | abstained | None | 0.000 | Yes | Yes | Yes. | Yes | 3.331 | The route does not call for a photo identification tool, does not form a photographic observation |
| multimodal-005 | Local OCR + rule resolution | abstained | None | Not applicable | Yes | Yes. | Yes. | Yes | 894.886 | The OCR text does not match the currently supported status panel or telemetry panel format |
| multimodal-005 | Vision Model | failed | None | Not applicable | Yes | Yes | Yes. | Yes | 4205.367 | invalid_response |
| multimodal-005 | Direct refusal | abstained | None | Not applicable | Yes. | Yes. | Yes. | Yes. | 12.005 | The route does not call for a photo identification tool, does not form a photographic observation |
| multimodal-006 | Local OCR + rule resolution | abstained | None | Not applicable | Yes | Yes. | Yes. | Yes | 816.978 | No valid value found for the fault_code field; no valid value found for the network_state field; no valid value found for task_state field |
| multimodal-006 | Vision Model | completed | The picture is a robotic status panel, with a white-titled text "ROBOT STATUS PANEL" on the top. ； The left side of the panel has a light grey labeled text "FAULT CODE", "BATTERY" and "TASK" from top to bottom. ； The right side of the panel is a larger, pure black rectangle display area with no visible contents. ； On the bottom of the panel there is a light gray little text "SYNTHETIC TRANING SAMPLE". | Not applicable | Yes | Yes | Yes. | Yes | 7794.121 | None |
| multimodal-006 | Direct refusal | abstained | None | Not applicable | Yes. | Yes. | Yes. | Yes. | 19.193 | The route does not call for a photo identification tool, does not form a photographic observation |

## 7. Outcomes and conclusions

- Number of preferred routes passed: 4/6
- Number of technical failures: 1
- The OCR route should be used as a priority for fixed labels, failure codes and numerical reading.
- The Vision route should be reserved for tasks that really require visual understanding, such as colour, brightening, the location of the components and their cover-up.
- If the picture does not support the problem, the refusal should be returned and the model cannot be used to guess the blocked or obscured information.

## 8. Boundary

- The current sample is a redacted teaching picture from Pillow, which does not represent the distribution of real devices.
- The content accuracy rate is matched by a standard criterion for certainty and does not cover all natural language equivalents.
- Vision costs are estimated only when Provider provides sufficient information on usage.
- This experiment assesses only photo observations and tool applicability and does not generate robotic control instructions.
