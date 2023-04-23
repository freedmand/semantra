import json
import os
from tqdm import tqdm
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import click
from models import models, BaseModel, TransformerModel, as_numpy
import io
from pdf import get_pdf_content
import math
import hashlib
from util import (
    file_md5,
    get_tokens_filename,
    get_config_filename,
    get_embeddings_filename,
    get_num_embeddings,
    get_offsets,
    read_embeddings_file,
    write_embedding,
    join_text_chunks,
    get_annoy_filename,
    get_num_annoy_embeddings,
    write_annoy_db,
    load_annoy_db,
    sort_results,
    HASH_LENGTH,
)
import pkg_resources

with open(pkg_resources.resource_filename("semantra", "VERSION"), "r") as f:
    VERSION = f.read().strip()

package_directory = os.path.dirname(os.path.abspath(__file__))


class Content:
    def __init__(self, rawtext, filename):
        self.rawtext = rawtext
        self.filename = filename
        self.filetype = "text"


def get_text_content(md5, filename, semantra_dir, force, silent):
    if filename.endswith(".pdf"):
        return get_pdf_content(md5, filename, semantra_dir, force, silent)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        rawtext = f.read()
        return Content(rawtext, filename)


TRANSFORMER_POOL_DEFAULT = 15000


class Document:
    def __init__(
        self,
        filename,
        md5,
        semantra_dir,
        base_filename,
        config,
        embeddings_filenames,
        use_annoy,
        annoy_filenames,
        windows,
        offsets,
        tokens_filename,
        num_dimensions,
    ):
        self.filename = filename
        self.md5 = md5
        self.semantra_dir = semantra_dir
        self.base_filename = base_filename
        self.config = config
        self.embeddings_filenames = embeddings_filenames
        self.use_annoy = use_annoy
        self.annoy_filenames = annoy_filenames
        self.windows = windows
        self.offsets = offsets
        self.tokens_filename = tokens_filename
        self.num_dimensions = num_dimensions

    @property
    def content(self):
        return get_text_content(self.md5, self.filename, self.semantra_dir, False, True)

    @property
    def text_chunks(self):
        with open(self.tokens_filename, "r") as f:
            return json.loads(f.read())

    @property
    def num_embeddings(self):
        return len(self.offsets[0])

    @property
    def embedding_db(self):
        if not self.use_annoy:
            raise ValueError("Embeddings are not stored in Annoy database")
        return load_annoy_db(self.annoy_filenames[0], self.num_dimensions)

    @property
    def embeddings(self):
        results, embedding_count = read_embeddings_file(
            self.embeddings_filenames[0],
            self.num_dimensions,
            self.num_embeddings,
        )
        assert embedding_count == self.num_embeddings
        return results


def process(
    filename,
    semantra_dir,
    model,
    num_dimensions,
    use_annoy,
    num_annoy_trees,
    windows,
    cost_per_token,
    pool_count,
    pool_size,
    force,
    silent,
    no_confirm,
):
    # Check if semantra dir exists
    if not os.path.exists(semantra_dir):
        os.makedirs(semantra_dir)

    # Get the md5 and config
    md5 = file_md5(filename)
    base_filename = os.path.basename(filename)
    config = model.get_config()
    config_hash = hashlib.shake_256(json.dumps(config).encode()).hexdigest(HASH_LENGTH)

    # File names
    tokens_filename = os.path.join(semantra_dir, get_tokens_filename(md5, config_hash))
    config_filename = os.path.join(semantra_dir, get_config_filename(md5, config_hash))

    should_calculate_tokens = True
    if force or not os.path.exists(tokens_filename):
        # Calculate tokens to get text chunks
        content = get_text_content(md5, filename, semantra_dir, force, silent)
        text = content.rawtext
        tokens = model.get_tokens(text)
        should_calculate_tokens = False
        text_chunks = model.get_text_chunks(text, tokens)
        with open(tokens_filename, "w") as f:
            f.write(json.dumps(text_chunks))
    else:
        with open(tokens_filename, "r") as f:
            text_chunks = json.loads(f.read())
    num_tokens = len(text_chunks)

    # Get embedding offsets based on config parameters
    (
        offsets,
        num_embedding_tokens,
    ) = get_offsets(num_tokens, windows)

    # Full config contains additional details
    full_config = {
        **config,
        "filename": filename,
        "md5": md5,
        "base_filename": base_filename,
        "num_dimensions": num_dimensions,
        "cost_per_token": cost_per_token,
        "windows": windows,
        "num_tokens": num_tokens,
        "num_embeddings": len(offsets),
        "num_embedding_tokens": num_embedding_tokens,
        "use_annoy": use_annoy,
        "num_annoy_trees": num_annoy_trees,
        "semantra_version": VERSION,
    }

    if force or not os.path.exists(config_filename):
        if cost_per_token is not None and not no_confirm:
            click.confirm(
                f"Tokens will cost ${num_embedding_tokens * cost_per_token:.2f}. Proceed?",
                abort=True,
            )

    # Write out the config every time
    with open(config_filename, "w") as f:
        f.write(json.dumps(full_config))

    embeddings_filenames = []
    annoy_filenames = []
    with tqdm(
        total=num_embedding_tokens,
        desc="Calculating embeddings",
        leave=False,
        disable=silent,
    ) as pbar:
        for (size, offset, rewind), sub_offsets in zip(windows, offsets):
            embeddings_filename = os.path.join(
                semantra_dir,
                get_embeddings_filename(md5, config_hash, size, offset, rewind),
            )
            annoy_filename = os.path.join(
                semantra_dir,
                get_annoy_filename(
                    md5, config_hash, size, offset, rewind, num_annoy_trees
                ),
            )
            embeddings_filenames.append(embeddings_filename)
            annoy_filenames.append(annoy_filename)

            if os.path.exists(embeddings_filename) and (
                not use_annoy or os.path.exists(annoy_filename)
            ):
                num_embeddings = get_num_embeddings(embeddings_filename, num_dimensions)
                if use_annoy:
                    num_annoy_embeddings = get_num_annoy_embeddings(
                        annoy_filename, num_dimensions
                    )

                if (
                    not force
                    and num_embeddings == len(sub_offsets)
                    and (not use_annoy or num_annoy_embeddings == len(sub_offsets))
                ):
                    # Embedding is fully calculated
                    continue

            if should_calculate_tokens:
                tokens = model.get_tokens(join_text_chunks(text_chunks))
                should_calculate_tokens = False

            # Read embeddings if they exist
            embedding_index = 0
            if not force and os.path.exists(embeddings_filename):
                embeddings, embedding_index = read_embeddings_file(
                    embeddings_filename, num_dimensions, len(sub_offsets)
                )
            else:
                embeddings = np.empty(
                    (len(sub_offsets), num_dimensions), dtype=np.float32
                )
                embedding_index = 0

            num_skip = embedding_index
            iteration = 0

            # Write embeddings
            pool = []
            pool_token_count = 0

            with open(embeddings_filename, "ab") as f:

                def flush_pool():
                    nonlocal pool, pool_token_count, embeddings, embedding_index, f

                    if len(pool) > 0:
                        embedding_results = model.embed(tokens, pool)
                        # Call .cpu if embedding_results contains it
                        if hasattr(embedding_results, "cpu"):
                            embedding_results = embedding_results.cpu()
                        embeddings[
                            embedding_index : embedding_index + len(pool)
                        ] = embedding_results
                        for embedding in embedding_results:
                            write_embedding(f, embedding, num_dimensions)
                        embedding_index += len(pool)
                        pool = []
                        pool_token_count = 0

                for offset in sub_offsets:
                    size = offset[1] - offset[0]

                    # Skip if already calculated
                    if iteration < num_skip:
                        iteration += 1
                        pbar.update(size)
                        continue

                    window_text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                    if len(window_text) == 0:
                        pbar.update(size)
                        continue

                    pool.append(offset)
                    pool_token_count += size
                    if (
                        pool_count is not None and len(pool) >= pool_count
                    ) or pool_token_count >= pool_size:
                        flush_pool()
                    pbar.update(size)

                flush_pool()

            # Write embeddings db
            if use_annoy:
                write_annoy_db(
                    filename=annoy_filename,
                    num_dimensions=num_dimensions,
                    embeddings=embeddings,
                    num_trees=num_annoy_trees,
                )

    return Document(
        filename=filename,
        md5=md5,
        semantra_dir=semantra_dir,
        base_filename=base_filename,
        config=full_config,
        embeddings_filenames=embeddings_filenames,
        use_annoy=use_annoy,
        annoy_filenames=annoy_filenames,
        windows=windows,
        offsets=offsets,
        tokens_filename=tokens_filename,
        num_dimensions=num_dimensions,
    )


def process_windows(windows: str) -> "list[tuple[int, int, int]]":
    for window in windows.split(","):
        if "_" in window:
            # One or two occurrences?
            if window.count("_") == 1:
                size, offset = window.split("_")
                rewind = 0
            else:
                size, offset, rewind = window.split("_")
            yield int(size), int(offset), int(rewind)
        else:
            yield int(window), 0, 0


@click.command()
@click.argument("filename", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--model",
    type=click.Choice(models.keys(), case_sensitive=True),
    default="mpnet",
    show_default=True,
    help="Preset model to use for embedding",
)
@click.option(
    "--transformer-model",
    type=str,
    help="Custom Huggingface transformers model name to use for embedding",
)
@click.option(
    "--windows",
    type=str,
    default="128_0_16",
    show_default=True,
    help='Embedding windows to extract. A comma-separated list of the format "size[_offset=0][_rewind=0]. A window with size 128, offset 0, and rewind of 16 (128_0_16) will embed the document in chunks of 128 tokens which partially overlap by 16. Only the first window is used for search.',
)
@click.option(
    "--no-server",
    is_flag=True,
    default=False,
    show_default=True,
    help="Do not start the UI server (only process)",
)
@click.option(
    "--port",
    type=int,
    default=8080,
    show_default=True,
    help="Port to use for embedding server",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    show_default=True,
    help="Host to use for embedding server",
)
@click.option(
    "--pool-size",
    type=int,
    default=None,
    help="Max number of embedding tokens to pool together in requests",
)
@click.option(
    "--pool-count",
    type=int,
    default=None,
    help="Max number of embeddings to pool together in requests",
)
@click.option(
    "--doc-token-pre",
    type=str,
    default=None,
    help="Token to prepend to each document in transformer models (default: None)",
)
@click.option(
    "--doc-token-post",
    type=str,
    default=None,
    help="Token to append to each document in transformer models (default: None)",
)
@click.option(
    "--query-token-pre",
    type=str,
    default=None,
    help="Token to prepend to each query in transformer models (default: None)",
)
@click.option(
    "--query-token-post",
    type=str,
    default=None,
    help="Token to append to each query in transformer models (default: None)",
)
@click.option(
    "--num-results",
    type=int,
    default=10,
    show_default=True,
    help="Number of results (neighbors) to retrieve per file for queries",
)
@click.option(
    "--annoy",
    is_flag=True,
    default=True,
    show_default=True,
    help="Use approximate kNN via Annoy for queries (faster querying at a slight cost of accuracy); if false, use exact exhaustive kNN",
)
@click.option(
    "--num-annoy-trees",
    type=int,
    default=100,
    show_default=True,
    help="Number of trees to use for approximate kNN via Annoy",
)
@click.option(
    "--svm",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use SVM instead of any kind of kNN for queries (slower and only works on symmetric models)",
)
@click.option(
    "--svm-c",
    type=float,
    default=1.0,
    show_default=True,
    help="SVM regularization parameter; higher values penalize mispredictions more",
)
@click.option(
    "--explain-split-count",
    type=int,
    default=9,
    show_default=True,
    help="Number of splits on a given window to use for explaining a query",
)
@click.option(
    "--explain-split-divide",
    type=int,
    default=6,
    show_default=True,
    help="Factor to divide the window size by to get each split length for explaining a query",
)
@click.option(
    "--num-explain-highlights",
    type=int,
    default=2,
    show_default=True,
    help="Number of split results to highlight for explaining a query",
)
@click.option(
    "--force", is_flag=True, default=False, help="Force process even if cached"
)
@click.option(
    "--silent",
    is_flag=True,
    default=False,
    help="Do not print progress information",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    default=False,
    help="Do not show cost and ask for confirmation before processing with OpenAI",
)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Print version and exit",
)
@click.option(
    "--list-models",
    is_flag=True,
    default=False,
    help="List preset models and exit",
)
@click.option(
    "--show-semantra-dir",
    is_flag=True,
    default=False,
    help="Print the directory semantra will use to store processed files and exit",
)
@click.option(
    "--semantra-dir",
    type=click.Path(exists=False),
    default=None,
    help="Directory to store semantra files in",
)
def main(
    filename,
    windows="128_0_16",
    no_server=False,
    port=8080,
    host="0.0.0.0",
    pool_size=None,
    pool_count=None,
    doc_token_pre=None,
    doc_token_post=None,
    query_token_pre=None,
    query_token_post=None,
    model="mpnet",
    transformer_model=None,
    num_annoy_trees=100,
    num_results=10,
    annoy=True,
    svm=False,
    svm_c=1.0,
    explain_split_count=9,
    explain_split_divide=6,
    num_explain_highlights=2,
    force=False,
    silent=False,
    no_confirm=False,
    version=False,
    list_models=False,
    show_semantra_dir=False,
    semantra_dir=None,  # auto
):
    if version:
        print(VERSION)
        return

    if list_models:
        for model_name in models:
            print(f"- {model_name}")
        return

    if semantra_dir is None:
        semantra_dir = click.get_app_dir("Semantra")

    if show_semantra_dir:
        print(semantra_dir)
        return

    if filename is None or len(filename) == 0:
        raise click.UsageError("Must provide a filename to process/query")

    processed_windows = list(process_windows(windows))

    if transformer_model is not None:
        # Handle custom transformers model
        if pool_size is None:
            pool_size = TRANSFORMER_POOL_DEFAULT

        cost_per_token = None
        model = TransformerModel(
            transformer_model,
            doc_token_pre=doc_token_pre,
            doc_token_post=doc_token_post,
            query_token_pre=query_token_pre,
            query_token_post=query_token_post,
        )
    else:
        # Pull preset model
        model_config = models[model]
        cost_per_token = model_config["cost_per_token"]
        if pool_size is None:
            pool_size = model_config["pool_size"]
        if pool_count is None:
            pool_count = model_config.get("pool_count", None)
        model: BaseModel = model_config["get_model"]()

    # Check if model is compatible
    if svm and model.is_asymmetric():
        raise ValueError(
            "SVM is not compatible with asymmetric models. "
            "Please use a symmetric model or kNN."
        )

    documents = {}
    pbar = tqdm(filename, disable=silent)
    for fn in pbar:
        pbar.set_description(f"{os.path.basename(fn)}")
        documents[fn] = process(
            filename=fn,
            semantra_dir=semantra_dir,
            model=model,
            num_dimensions=model.get_num_dimensions(),
            use_annoy=annoy,
            num_annoy_trees=num_annoy_trees,
            windows=processed_windows,
            cost_per_token=cost_per_token,
            pool_count=pool_count,
            pool_size=pool_size,
            force=force,
            silent=silent,
            no_confirm=no_confirm,
        )

    cached_content = None
    cached_content_filename = None

    def get_content(filename):
        nonlocal cached_content, cached_content_filename
        # Check if we can pull from cache
        if filename == cached_content_filename:
            return cached_content
        # If not, grab content
        content = documents[filename].content
        # Cache the content
        cached_content_filename = filename
        cached_content = content
        # Return the now-cached content
        return content

    # Start a Flask server
    app = Flask(__name__)

    @app.route("/")
    def base():
        return send_from_directory(
            pkg_resources.resource_filename("semantra", "client/public"), "index.html"
        )

    # Path for all the static files (compiled JS/CSS, etc.)
    @app.route("/<path:path>")
    def home(path):
        return send_from_directory(
            pkg_resources.resource_filename("semantra", "client/public"), path
        )

    @app.route("/api/files", methods=["GET"])
    def files():
        return jsonify(
            [
                {
                    "basename": doc.base_filename,
                    "filename": doc.filename,
                    "filetype": doc.content.filetype,
                }
                for doc in documents.values()
            ]
        )

    @app.route("/api/query", methods=["POST"])
    def query():
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        if svm:
            return querysvm()
        if annoy:
            return queryann()

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            # Get kNN with cosine similarity
            distances = np.dot(embeddings, embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
            )
            sorted_ix = np.argsort(-distances)

            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = float(distances[index])
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(sort_results(results, True))

    @app.route("/api/querysvm", methods=["POST"])
    def querysvm():
        from sklearn import svm

        queries = request.json["queries"]
        preferences = request.json["preferences"]

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)
        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            x = np.concatenate([embeddings, embedding[None, ...]])
            y = np.zeros(len(embeddings) + 1)
            y[-1] = 1

            # Train the svm
            clf = svm.LinearSVC(
                class_weight="balanced",
                verbose=False,
                max_iter=10000,
                tol=1e-6,
                C=svm_c,
            )
            clf.fit(x, y)

            # Infer similarities
            similarities = clf.decision_function(x)[: len(embeddings)]
            sorted_ix = np.argsort(-similarities)

            text_chunks = doc.text_chunks
            offsets = doc.offsets
            sub_results = []
            for index in sorted_ix[:num_results]:
                distance = similarities[index]
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])

        return jsonify(sort_results(results, True))

    @app.route("/api/queryann", methods=["POST"])
    def queryann():
        queries = request.json["queries"]
        preferences = request.json["preferences"]

        # Get combined query and preference embedding
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        results = []
        for doc in documents.values():
            embedding_db = doc.embedding_db
            text_chunks = doc.text_chunks
            offsets = doc.offsets[0]
            sub_results = []
            for [index, distance] in zip(
                *embedding_db.get_nns_by_vector(embedding, num_results, -1, True)
            ):
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        # Convert distance from Euclidean distance of normalized vectors to cosine
                        "distance": 1 - distance**2.0 / 2.0,
                        "offset": offset,
                        "index": int(index),
                        "filename": doc.filename,
                        "queries": queries,
                        "preferences": preferences,
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(sort_results(results, True))

    @app.route("/api/explain", methods=["POST"])
    def explain():
        filename = request.json["filename"]
        offset = request.json["offset"]
        tokens = documents[filename].text_chunks[offset[0] : offset[1]]
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        embedding = model.embed_queries_and_preferences(queries, preferences, documents)

        # Find hot-spots within the result tokens
        def get_splits(divide_factor=2, num_splits=3, start=0, end=len(tokens)):
            window_length = math.ceil((end - start) / divide_factor)
            split_length = math.ceil((end - start) / num_splits)
            splits = []
            for i in range(num_splits):
                splits.append(
                    (
                        start + i * split_length,
                        min(end, start + i * split_length + window_length),
                    )
                )
            return splits

        def exclude_window(start, end):
            nonlocal tokens
            return join_text_chunks(tokens[:start] + tokens[end:])

        def get_highest_ranked_split(splits):
            nonlocal tokens, embedding
            split_queries = [exclude_window(start, end) for start, end in splits]
            split_windows = np.array(
                [
                    as_numpy(model.embed_document(split_query))
                    for split_query in split_queries
                ]
            )
            distances = split_windows.dot(embedding) / (
                np.linalg.norm(split_windows, axis=1) * np.linalg.norm(embedding)
            )
            # Return the splits in order of highest to lowest ranked
            return sorted(zip(splits, distances), key=lambda x: x[1], reverse=False)

        def as_tokens(splits):
            nonlocal tokens
            indices = sorted([split[0] for split in splits], key=lambda x: x[0])
            last_index = 0
            chunks = []

            def append(start, end, type):
                if start >= end:
                    return
                nonlocal chunks, tokens
                chunks.append(
                    {
                        "text": join_text_chunks(tokens[start:end]),
                        "type": type,
                    }
                )

            for index in indices:
                append(last_index, index[0], "normal")
                append(max(index[0], last_index), index[1], "highlight")
                last_index = index[1]

            append(last_index, len(tokens), "normal")
            return chunks

        splits = get_splits(
            divide_factor=explain_split_divide,
            num_splits=explain_split_count,
            start=0,
            end=len(tokens),
        )
        top_splits = get_highest_ranked_split(splits)[:num_explain_highlights]
        return jsonify(as_tokens(top_splits))

    @app.route("/api/getfile", methods=["GET"])
    def getfile():
        filename = request.args.get("filename")
        content = get_content(filename)
        filename = content.filename
        return send_file(filename)

    @app.route("/api/pdfpositions", methods=["GET"])
    def pdfpositions():
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype == "pdf":
            return jsonify(content.positions)
        else:
            return jsonify([])

    @app.route("/api/pdfpage", methods=["GET"])
    def pdfpage():
        filename = request.args.get("filename")
        content = get_content(filename)
        page = request.args.get("page")
        scale = request.args.get("scale")
        if content.filetype == "pdf":
            pil_image = content.get_page_image_pil(int(page), float(scale))
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            response = make_response(img_byte_arr.getvalue())
            response.headers.set("Content-Type", "image/png")
            return response

    @app.route("/api/pdfchars", methods=["GET"])
    def pdfchars():
        filename = request.args.get("filename")
        content = get_content(filename)
        if content.filetype != "pdf":
            return jsonify([])
        page = request.args.get("page")
        return jsonify(content.get_page_chars(int(page)))

    @app.route("/api/text", methods=["GET"])
    def text():
        filename = request.args.get("filename")
        return jsonify(documents[filename].text_chunks)

    if not no_server:
        app.run(host=host, port=port)


if __name__ == "__main__":
    main()
