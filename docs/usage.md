# Quick start

Welcome to Semantra, a tool to query documents based on semantic meaning. Semantra is a powerful local search engine but is different from most other search engines, so it might take a little getting used to. Some key concepts are needed to understand what it means to query semantically and how to best utilize the tool.

## Installation

Semantra is a Python command-line application. To use Semantra, you must have [Python 3.6+ installed](https://www.python.org/downloads/).

Install Semantra via:

```sh
pip install semantra
```

## Command-line usage

Semantra operates on collections of documents — currently text or PDF files — stored on your local computer.

At its simplest, you can run Semantra over a single document by running:

```sh
semantra doc.pdf
```

You can run Semantra over multiple documents, too:

```sh
semantra report.pdf book.txt manuscript.pdf
```

Semantra will take some time to process the input documents. This is a one-time operation per document (subsequent runs over the same document collection will be near instantaneous).

Once processing is complete, Semantra will launch a local webserver, by default at [localhost:8080](http://localhost:8080). On this web page, you can interactively query the passed in documents semantically.

**Notes:**

When you first run Semantra, it may take several minutes and several hundred megabytes of hard disk space to download a local machine learning model that can process the document you passed in. The model used can be customized, but the default one is a great mix of being fast, lean, and effective.

If you want to process documents quickly without using your own computational resources and don't mind paying or sharing data with external services, you can use OpenAI's embedding model.

## Web usage

When you first navigate to Semantra's web interface, you will see a screen like this:

![](img/initial_screen.jpg)

### Querying semantically

Type in a concept to being querying the documents semantically.

## Concepts

- **machine learning models**: often called _artificial intelligence_ or "AI," these models are often seen as magic boxes that take in various inputs and output something that resembles human intelligence. Many different kinds of models exist that can do everything from classifying objects in images, generating text and pictures, and semantically encoding text. Magic they are not. These models often involve massive "neural networks" composed of millions of computational connections that strengthen and weaken in response to "training data," the data fed in to _teach_ the model. As tempting as it is to anthropomorphize these models and endow them with human faculties, it is more practical to view them as large-scale statistical machines that have gleaned insights across terabytes of training data.

- **embeddings**: Models that encode semantic meaning are called "embedding models." These models are input text, pictures, or other pieces information and output a long chain of numbers called an "embedding." The models are tuned so that similar content produces similar embeddings.

  Mathematically, an embedding is a high-dimensional vector, and embeddings are considered similar if the angle between them is small (that is, they are pointing in similar directions in their high-dimensional spaces).
