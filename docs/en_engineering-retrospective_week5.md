# Week 5 Engineering Retrospective

#### View Configuration

Two approaches have been deployed in the mapping component: a local OCR and a visual model. OCR can only recognize text, and the Vision model returns the description of the picture. The model can only choose whether to call the vision model, and OCR is used only for matching different scenarios.

At the same time, there are many check items when images are uploaded, such as image resolution, file size, empty or not.

#### Return result processing

The description of the VISION model is unstable, so that accuracy is judged by rules that match, for example, keywords and values, and one by one. Each image calling the vision model is recorded as a ref, which avoids repetition of calls and ultimately serves as a reference.





#### Harder.

The biggest difficulty remains that the score points, for example, are accurate. The vision model may give numerous descriptions of the equivalents, and therefore return values are subject to constant adjustments based on the results of the review by means of fixed rules.
