# Week 3 Engineering Retrospective

#### Recall Optimization

1. Inserts the word homogenization to avoid failure to match keyword error codes or models because of different case sizes or formats.
2. Adds a search based on keyword matching, which scores according to the number of key/word matches between the chunk text and the problem text, with different weights for different types of word matching.
3. Rescoring with RRF combined vectors and keyword matching fractions.

However, it was found that there were unrelated chunk scores of quantities matched by normal words exceeding the chunk that should be in Top3 and adding reranking components that would match keywords by checking the fraction composition on RRF basis but which were squeezed out of Top3 and moved to Top3.



#### Legacy issues

What needs to be optimized now is focused on evidence gate and LLM. The doorman rejected the answerable questions because conservative judgements sometimes called back the evidence correctly, but the scores fell short of the target. LLM, even if it provides all the required grounds, may cite only some of them or wrongly consider them unresponsive, and the answer may be less extensive than expected.



#### Harder.

The operation of “ticking patches” in optimizing the recall and answer generation process to improve accuracy is, in fact, more like a proposed combination that turns into an east wall to the west wall. The changes became issue-specific and eventually led to a reduction in the overall accuracy rate. The code was then returned to the best consolidated version.
