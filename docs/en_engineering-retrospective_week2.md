# Week 2 Engineering Retrospective

### Database preparation

The document is processed in three levels of quantification

1. The original content of each page (for pdf) and the number of pages from which it comes are recorded as sourceTextSegment in loader.
2. Use chunker to split text according to parameters (chunk_size, overlap) and generate its identity information for each chunk, e. g. chunk_id, content_hash. Also record the source of the document.
3. Put the chunk on Embedding, get the vector.
4. Saves the Quantified chunk.

Also saves the full document's Hash value to prevent repeated documents from being processed repeatedly in a confused database.

### Call database

3 of the most relevant users were quantified and sent together as information support. In addition to replying, LLM indicates what information was cited in the response and is finally listed as a Citations for users.



### Harder.

There is a lot of additions this week, and if the time and energy of still reading codes are not allowed. With the availability of a number of methods of achieving the code that are common to some codes (e.g. building a schema), AI is required to describe the objective of the new content, the method of achieving it and the chain of call. In the case of a test component, the test target for each test function, the test method, the expected behaviour of the tested module, and the expected results are described. Put more energy on the logical chain than on the code itself.
