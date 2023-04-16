(Work-in-progress documentation; the tool is not yet ready)

# Semantra

Semantra is a multipurpose tool for semantically searching documents. Query by meaning rather than just by matching text.

The tool, made to run on the command line, analyzes specified text and PDF files on your computer and launches a local web search application for interactively querying them. The purpose of Semantra is to make running a specialized semantic search engine easy, friendly, configurable, and private/secure.

Semantra is built for individuals seeking needles in haystacks — journalists sifting through leaked documents on deadline, researchers seeking insights within papers, students engaging with literature by querying themes, historians connecting events across books, and so forth.

## Questions

### Can it use ChatGPT?

No, and this is by design.

Semantra does not use any generative models like ChatGPT. It is built only to query text semantically without any layers on top to attempt explaining, summarizing, or synthesizing results. Generative language models occasionally produce outwardly plausible but ultimately incorrect information, placing the burden of verification on the user. Semantra treats primary source material as the only source of truth and endeavors to show that a human-in-the-loop search experience on top of simpler embedding models is more serviceable to users.

### What are embeddings

Assume a mystical machine exists that converts text you give it to a number. “Dog” → 352. “Cat” → 361. “Toasters are cool” → 6,723,412. The closer the meaning of the text you give it, the closer the output number will be. Here, “Dog” and “Cat” are really close at only 9 apart, while the text “Toasters are cool” is off by millions.

Well, these mystical machines are essentially machine learning models. These models are fed vast amounts of text and “learn” to output numbers that correspond to meaning by setting up internal networks of weights and adjusting them slightly over many millions of iterations based on if text fits expected patterns or not. In a rudimentary example, “dog” and “cat” may be relatively interchangeable in sentences talking about pets and so the model infers they are similar.

The mystical machine above only outputs one number, but in practice more are needed to meaningfully encode text semantically. Two numbers enables placing text in two-dimensional space. Three numbers gives you a three-dimensional representation, which provides more directions along which placed text can be close together (x ↔ y, y ↔ z, z ↔ x).

These output numbers are called “embeddings” and are treated as vectors which can be pictured as arrows pointing in a direction. To convert text to numbers is to “embed” the text into the "vector space." Multiple passages of text are semantically similar according to embedding models if their embedding vectors point in similar directions. Though it’s hard to picture, popular embedding models typically output embeddings that are hundreds or even thousands of dimensions!
