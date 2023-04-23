# Lesson 2

## Sculpting semantic searches: exploring U.S. presidential inaugural speeches

In this lesson, we'll work with a multi-document collection and learn how to utilize some of Semantra's more advanced features to refine our query.

### Step 1: prepare the documents

Download the following collection of U.S. presidential inaugural speeches:

- [us_inaugural_speeches.zip (334 KB)](https://github.com/freedmand/semantra/raw/main/docs/example_docs/us_inaugural_speeches.pdf)

### Step 5: a brief aside on embeddings

You'll need a base level of understanding about _embeddings_ to effectively work with Semantra. At its core, embeddings are numerical representations of meaning. And embedding models are machine learning models that take in text (or other media) and output these numerical embeddings.

Text embedding models are typically trained on terabytes of text encompassing many genres. From that data they become skilled at encoding text into meaning by inferring patterns.

It is helpful to imagine working with embeddings in Semantra as _sculpting_ in the domain of meaning.

Embeddings can be hundreds or even thousands of dimensions long, which sounds pretty confusing. But it can be helpful to visualize them as bar charts, where each dimension is a bar.

Since embeddings are entirely numerical, you can do mathematical operations on them, with some pretty fun results. For example, if you take the embedding for `queen`, `king`, `woman`, and `man`, you can do arithmetic like `king - man + woman` and end up with a result that's pretty close to the embedding for `queen`.

![king - man + woman ~= queen diagram](img/queen_king_example.png)

This kind of analogizing mathematically opens up powerful implications for semantic search. Semantra allows performing basic arithmetic in the search bar and also adding/subtracting search results to finely shape your query and find specific results.
