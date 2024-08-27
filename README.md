# Semantra

https://user-images.githubusercontent.com/306095/233867821-601db8b0-19c6-4bae-8e93-720b324dc199.mov

Semantra is a multipurpose tool for semantically searching documents. Query by meaning rather than just by matching text.

The tool, made to run on the command line, analyzes specified text and PDF files on your computer and launches a local web search application for interactively querying them. The purpose of Semantra is to make running a specialized semantic search engine easy, friendly, configurable, and private/secure.

Semantra is built for individuals seeking needles in haystacks — journalists sifting through leaked documents on deadline, researchers seeking insights within papers, students engaging with literature by querying themes, historians connecting events across books, and so forth.

## Resources

- [Tutorial](docs/tutorial.md): a gentle introduction to getting started with Semantra — everything from installing the tool to hands-on examples of analyzing documents with it
- [Guides](docs/guides.md): practical guides on how to do more with Semantra
- [Concepts](docs/concepts.md): Explainers on some concepts to better understand how Semantra works
- [Using the web interface](docs/help.md): A reference on how to use the Semantra web app

This page gives a high-level overview of Semantra and a reference of its features. It's also available in other languages: [Semantra en español](docs/README_es.md), [Semantra 中文说明](docs/README_zh-CN.md)

## Installation

Ensure you have [Python >= 3.9](https://www.python.org/downloads/).

The easiest way to install Semantra is via `pipx`. If you do not have `pipx` installed, run:

```sh
python3 -m pip install --user pipx
```

Or, if you have [Homebrew](https://brew.sh/) installed, you can run `brew install pipx`.

Once `pipx` is installed, run:

```sh
python3 -m pipx ensurepath
```

Open a new terminal window for the new path settings `pipx` sets to go into effect. Then run:

```sh
python3 -m pipx install semantra
```

This will install Semantra on your path. You should be able to run `semantra` in the terminal and see output.

Note: if the above steps don't work or you'd like a more granular installation, you can install Semantra in a virtual environment (though note it will only be accessible while the virtual environment is activated):

```sh
python3 -m venv venv
source venv/bin/activate
pip install semantra
```

## Usage

Semantra operates on collections of documents — text or PDF files — stored on your local computer.

At its simplest, you can run Semantra over a single document by running:

```sh
semantra doc.pdf
```

You can run Semantra over multiple documents, too:

```sh
semantra report.pdf book.txt
```

Semantra will take some time to process the input documents. This is a one-time operation per document (subsequent runs over the same document collection will be near instantaneous).

Once processing is complete, Semantra will launch a local webserver, by default at [localhost:8080](http://localhost:8080). On this web page, you can interactively query the passed in documents semantically.

**Quick notes:**

When you first run Semantra, it may take several minutes and several hundred megabytes of hard disk space to download a local machine learning model that can process the document you passed in. [The model used can be customized](docs/guide_models.md), but the default one is a great mix of being fast, lean, and effective.

If you want to process documents quickly without using your own computational resources and don't mind paying or sharing data with external services, you can use [OpenAI's embedding model](docs/guide_openai.md).

## Quick tour of the web app

When you first navigate to the Semantra web interface, you will see a screen like this:

![Semantra web interface](docs/img/initial_screen.png)

Type in something in the search box to start querying semantically. Hit <kbd>Enter</kbd> or click the search icon to execute the query.

Search results will appear in the left pane ordered by most relevant documents:

![Semantra search results](docs/img/resultspane.png)

The yellow scores show relevance from 0-1.00. Anything in the 0.50 range indicates a strong match. Lighter brown highlights will stream in over the search results explaining the most relevant portions to your query.

Clicking on a search result's text will navigate to the relevant section of the associated document.

![Highlighted search result in document](docs/img/contentwindow_pdf.png)

Clicking on the plus/minus buttons associated with a search result will positively/negatively tag those results. Re-running the query will cause these additional query parameters to go into effect.

![Positively/negatively tagging search results](docs/img/inaugural_speeches_healthcare_plus_minus.png)

Finally, text queries can be added and subtracted with plus/minus signs in the query text to sculpt a precise semantic meaning.

![Adding and subtracting text queries](docs/img/inaugural_speeches_economic_capitalism_war.png)

For a more in-depth walkthrough of the web app, check out the [tutorial](docs/tutorial.md) or [the web app reference](docs/help.md).

## Quick concepts

Using a semantic search engine is fundamentally different than an exact text matching algorithm.

For starters, there will _always_ be search results for a given query, no matter how irrelevant it is. The scores may be really low, but the results will never disappear entirely. This is because semantic searching with query arithmetic often reveals useful results amid very minor score differences. The results will always be sorted by relevance and only the top 10 results per document are shown so the lower scoring results are cut off automatically.

Another difference is that Semantra will not necessarily find exact text matches if you query something that directly appears in the document. At a high level, this is because words can mean different things in different contexts, e.g. the word "leaves" can refer to the leaves on trees or to someone _leaving_. The embedding models that Semantra uses convert all the text and queries you enter into long sequences of numbers that can be mathematically compared, and an exact substring match is not always significant in this sense. See [the embeddings concept doc](docs/concept_embeddings.md) for more information on embeddings.

## Command-line reference

```sh
semantra [OPTIONS] [FILENAME(S)]...
```

## Options

- `--model [openai|minilm|mpnet|sgpt|sgpt-1.3B]`: Preset model to use for embedding. See [the models guide](docs/guide_models.md) for more info (default: mpnet)
- `--transformer-model TEXT`: Custom Huggingface transformers model name to use for embedding (only one of `--model` and `--transformer-model` should be specified). See [the models guide](docs/guide_models.md) for more info
- `--windows TEXT`: Embedding windows to extract. A comma-separated list of the format "size[\_offset=0][_rewind=0]. A window with size 128, offset 0, and rewind of 16 (128_0_16) will embed the document in chunks of 128 tokens which partially overlap by 16. Only the first window is used for search. See the [windows concept doc](docs/concept_windows.md) for more information (default: 128_0_16)
- `--encoding`: Encoding to use for reading text files [default: utf-8]
- `--no-server`: Do not start the UI server (only process)
- `--port INTEGER`: Port to use for embedding server (default: 8080)
- `--host TEXT`: Host to use for embedding server (default: 127.0.0.1)
- `--pool-size INTEGER`: Max number of embedding tokens to pool together in requests
- `--pool-count INTEGER`: Max number of embeddings to pool together in requests
- `--doc-token-pre TEXT`: Token to prepend to each document in transformer models (default: None)
- `--doc-token-post TEXT`: Token to append to each document in transformer models (default: None)
- `--query-token-pre TEXT`: Token to prepend to each query in transformer models (default: None)
- `--query-token-post TEXT`: Token to append to each query in transformer models (default: None)
- `--num-results INTEGER`: Number of results (neighbors) to retrieve per file for queries (default: 10)
- `--annoy`: Use approximate kNN via Annoy for queries (faster querying at a slight cost of accuracy); if false, use exact exhaustive kNN (default: True)
- `--num-annoy-trees INTEGER`: Number of trees to use for approximate kNN via Annoy (default: 100)
- `--svm`: Use SVM instead of any kind of kNN for queries (slower and only works on symmetric models)
- `--svm-c FLOAT`: SVM regularization parameter; higher values penalize mispredictions more (default: 1.0)
- `--explain-split-count INTEGER`: Number of splits on a given window to use for explaining a query (default: 9)
- `--explain-split-divide INTEGER`: Factor to divide the window size by to get each split length for explaining a query (default: 6)
- `--num-explain-highlights INTEGER`: Number of split results to highlight for explaining a query (default: 2)
- `--force`: Force process even if cached
- `--silent`: Do not print progress information
- `--no-confirm`: Do not show cost and ask for confirmation before processing with OpenAI
- `--version`: Print version and exit
- `--list-models`: List preset models and exit
- `--show-semantra-dir`: Print the directory semantra will use to store processed files and exit
- `--semantra-dir PATH`: Directory to store semantra files in
- `--help`: Show this message and exit

## Frequently asked questions

### Can it use ChatGPT?

No, and this is by design.

Semantra does not use any generative models like ChatGPT. It is built only to query text semantically without any layers on top to attempt explaining, summarizing, or synthesizing results. Generative language models occasionally produce outwardly plausible but ultimately incorrect information, placing the burden of verification on the user. Semantra treats primary source material as the only source of truth and endeavors to show that a human-in-the-loop search experience on top of simpler embedding models is more serviceable to users.

## Development

The Python app is in `src/semantra/semantra.py` and is managed as a standard Python command-line project with `pyproject.toml`.

The local web app is written in [Svelte](https://svelte.dev/) and managed as a standard npm application.

To develop for the web app `cd` into `client` and then run `npm install`.

To build the web app, run `npm run build`. To build the web app in watch mode and rebuild when there's changes, run `npm run build:watch`.

## Contributions

The app is still in early stages, but contributions are welcome. Please feel free to submit an issue for any bugs or feature requests.
