import hashlib
import json
import os
import struct
import tqdm
import numpy as np
from annoy import AnnoyIndex
from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import click
from models import models, BaseModel, TransformerModel
import io
from pdf import get_pdf_content


def join_text_chunks(chunks):
    return "".join(chunks)


def get_config_filename(key, filename):
    return filename + f".{key}.config.json"


def get_tokens_filename(key, filename):
    return filename + f".{key}.tokens.json"


def get_embeddings_filename(key, filename):
    return filename + f".{key}.embeddings"


def get_annoy_filename(key, subkey, filename):
    return filename + f".{key}.{subkey}.annoy"


def write_embedding(file, embedding, num_dimensions):
    # Write float-encoded embeddings
    for i in range(num_dimensions):
        file.write(struct.pack("f", embedding[i]))
    file.flush()


def read_embedding(chunk, num_dimensions):
    # Read float-encoded embeddings
    embedding = []
    for i in range(num_dimensions):
        embedding.append(struct.unpack("f", chunk[i * 4 : (i + 1) * 4])[0])
    return embedding


def safe_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


class Content:
    def __init__(self, rawtext, filename):
        self.rawtext = rawtext
        self.filename = filename
        self.filetype = "text"


def get_text_content(filename, semantra_dir, base_filename):
    if filename.endswith(".pdf"):
        return get_pdf_content(filename, semantra_dir, base_filename)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        rawtext = f.read()
        return Content(rawtext, filename)


def get_embeddings_dbs(filenames, num_dimensions, windows, window_indices, embeddings):
    dbs = []
    for i, (filename, _, window_start) in enumerate(
        zip(filenames, windows, window_indices)
    ):
        sub_embeddings = (
            embeddings[window_start : window_indices[i + 1]]
            if i < len(window_indices) - 1
            else embeddings[window_start:]
        )

        db = AnnoyIndex(num_dimensions, "angular")
        for i, embedding in enumerate(sub_embeddings):
            db.add_item(i, embedding)
        db.build(10)
        db.save(filename)
        dbs.append(db)

    return dbs


def load_saved_embeddings_dbs(filenames, num_dimensions):
    dbs = []
    for filename in filenames:
        db = AnnoyIndex(num_dimensions, "angular")
        db.load(filename)
        dbs.append(db)
    return dbs


def get_binary_embedding_offsets(
    doc_size, max_window_tokens, min_window_tokens, divide_factor, use_offset
):
    num_tokens = 0
    size = max_window_tokens
    prev_size = 0

    offsets = []
    windows = []
    window_indices = []

    while size >= min_window_tokens:
        windows.append(size)
        window_indices.append(len(offsets))
        x = 0
        if use_offset and prev_size != 0 and size % 2 == 0:
            size2 = int(size / 2)
            offsets.append([0, size2])
            num_tokens += size2
            x = size2

        while x < doc_size:
            offsets.append([x, min(x + size, doc_size)])
            num_tokens += min(x + size, doc_size) - x
            x += size

        prev_size = size
        size = int(size / divide_factor)

    return offsets, windows, window_indices, num_tokens


def read_embeddings_file_old(embeddings_filename, num_dimensions):
    embeddings = []
    with open(embeddings_filename, "rb") as f:
        while True:
            chunk = f.read(num_dimensions * 4)
            if not chunk:
                break
            embeddings.append(read_embedding(chunk, num_dimensions))
    return embeddings


def read_embeddings_file(embeddings_filename, num_dimensions, capacity):
    # Get the file size
    with open(embeddings_filename, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    # Calculate the number of embeddings
    num_embeddings = file_size // (num_dimensions * 4)

    # Change the file size to the expected size
    with open(embeddings_filename, "ab") as f:
        f.truncate(num_embeddings * num_dimensions * 4)

    if num_embeddings == 0:
        return np.zeros((capacity, num_dimensions), dtype="float32"), 0

    # Memory map the file
    read_embeddings = np.memmap(
        embeddings_filename,
        dtype="float32",
        mode="r",
        shape=(num_embeddings, num_dimensions),
    )

    # Create an array with shape (capacity, num_dimensions) filled with 0s
    embeddings = np.zeros((capacity, num_dimensions), dtype="float32")

    # Copy the original embeddings into the new array
    embeddings[:num_embeddings] = read_embeddings[:num_embeddings]

    return embeddings, num_embeddings


TRANSFORMER_POOL_DEFAULT = 15000


class Document:
    def __init__(
        self,
        filename,
        semantra_dir,
        base_filename,
        config,
        embeddings_filename,
        database_filenames,
        windows,
        window_indices,
        offsets,
        tokens_filename,
    ):
        self.filename = filename
        self.semantra_dir = semantra_dir
        self.base_filename = base_filename
        self.config = config
        self.embeddings_filename = embeddings_filename
        self.database_filenames = database_filenames
        self.windows = windows
        self.window_indices = window_indices
        self.offsets = offsets
        self.tokens_filename = tokens_filename

    @property
    def content(self):
        return get_text_content(self.filename, self.semantra_dir, self.base_filename)

    @property
    def text_chunks(self):
        with open(self.tokens_filename, "r") as f:
            return json.loads(f.read())

    @property
    def num_dimensions(self):
        return self.config["num_dimensions"]

    @property
    def num_embeddings(self):
        return len(self.offsets)

    @property
    def embeddings_dbs(self):
        return load_saved_embeddings_dbs(self.database_filenames, self.num_dimensions)

    @property
    def embeddings(self):
        results, embedding_count = read_embeddings_file(
            self.embeddings_filename,
            self.num_dimensions,
            self.num_embeddings,
        )
        assert embedding_count == self.num_embeddings
        return results


def process(
    filename,
    semantra_dir,
    model,
    model_params,
    num_dimensions,
    max_window_tokens,
    min_window_tokens,
    divide_factor,
    use_offset,
    cost_per_token,
    pool_count,
    pool_size,
    doc_token_pre,
    doc_token_post,
    query_token_pre,
    query_token_post,
):
    print("Processing", filename)
    if semantra_dir is None:
        semantra_dir = os.path.join(os.path.dirname(filename), ".semantra")

    # Check if semantra dir exists
    if not os.path.exists(semantra_dir):
        os.makedirs(semantra_dir)

    # Load the text of the file
    base_filename = os.path.basename(filename)
    content = get_text_content(filename, semantra_dir, base_filename)
    text = content.rawtext

    # All the parameters that affect the output of the embeddings
    config = {
        "filename": os.path.abspath(filename),
        "model_params": model_params,
        "num_dimensions": num_dimensions,
        "max_window_tokens": max_window_tokens,
        "min_window_tokens": min_window_tokens,
        "divide_factor": divide_factor,
        "use_offset": use_offset,
        "doc_token_pre": doc_token_pre,
        "doc_token_post": doc_token_post,
        "query_token_pre": query_token_pre,
        "query_token_post": query_token_post,
        "md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
    }

    hashable_config_contents = json.dumps(config)
    config_key = hashlib.shake_256(hashable_config_contents.encode()).hexdigest(10)
    tokens_filename = os.path.join(
        semantra_dir, get_tokens_filename(config_key, base_filename)
    )
    config_filename = os.path.join(
        semantra_dir, get_config_filename(config_key, base_filename)
    )
    embeddings_filename = os.path.join(
        semantra_dir, get_embeddings_filename(config_key, base_filename)
    )

    print("Loading text chunks...")
    should_calculate_tokens = True
    if not os.path.exists(tokens_filename):
        # Calculate tokens to get text chunks
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
        windows,
        window_indices,
        num_embedding_tokens,
    ) = get_binary_embedding_offsets(
        num_tokens, max_window_tokens, min_window_tokens, divide_factor, use_offset
    )

    # Get database filenames for each window size
    database_filenames = [
        os.path.join(
            semantra_dir, get_annoy_filename(config_key, f"{window}", base_filename)
        )
        for window in windows
    ]

    # Full config contains additional details
    full_config = {
        **config,
        "cost_per_token": cost_per_token,
        "windows": windows,
        "window_indices": window_indices,
        "num_tokens": num_tokens,
        "num_embeddings": len(offsets),
        "num_embedding_tokens": num_embedding_tokens,
    }

    print(config_key)
    print(full_config)

    # Check if config does not exist or is different
    try:
        with open(config_filename, "r") as f:
            old_config = json.loads(f.read())
            if not (all(config[key] == old_config[key] for key in config)):
                # Config is different
                # Remove embeddings file if it exists
                safe_remove(embeddings_filename)
                for database_filename in database_filenames:
                    safe_remove(database_filename)
            if full_config != old_config:
                # If new details in the full config are different, update the config without updating everything
                # (this is to avoid re-embedding if the config is the same)
                with open(config_filename, "w") as f:
                    f.write(json.dumps(full_config))
    except FileNotFoundError:
        # Config does not exist
        # Remove embeddings file if it exists
        if cost_per_token is not None:
            print(
                f"Tokens will cost ${num_embedding_tokens * cost_per_token:.2f}. Proceed? y/n"
            )
            if input() != "y":
                return

        safe_remove(embeddings_filename)
        for database_filename in database_filenames:
            safe_remove(database_filename)

    if not all(
        os.path.exists(database_filename) for database_filename in database_filenames
    ):
        with open(config_filename, "w") as f:
            f.write(json.dumps(config))

        if should_calculate_tokens:
            tokens = model.get_tokens(text)

        # Read embeddings if they exist
        embeddings = np.empty((len(offsets), num_dimensions), dtype=np.float32)
        embedding_index = 0
        if os.path.exists(embeddings_filename):
            embeddings, embedding_index = read_embeddings_file(
                embeddings_filename, num_dimensions, len(offsets)
            )

        num_skip = embedding_index
        iteration = 0

        # Write embeddings
        pool = []
        pool_token_count = 0

        def flush_pool():
            nonlocal pool, pool_token_count, embeddings, embedding_index, f

            if len(pool) > 0:
                embedding_results = model.embed(tokens, pool)
                embeddings[
                    embedding_index : embedding_index + len(pool)
                ] = embedding_results
                for embedding in embedding_results:
                    write_embedding(f, embedding, num_dimensions)
                embedding_index += len(pool)
                pool = []
                pool_token_count = 0

        with open(embeddings_filename, "ab") as f:
            with tqdm.tqdm(total=num_embedding_tokens) as pbar:
                for offset in offsets:
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
        get_embeddings_dbs(
            filenames=database_filenames,
            num_dimensions=num_dimensions,
            windows=windows,
            window_indices=window_indices,
            embeddings=embeddings,
        )

    return Document(
        filename=filename,
        semantra_dir=semantra_dir,
        base_filename=base_filename,
        config=full_config,
        embeddings_filename=embeddings_filename,
        database_filenames=database_filenames,
        windows=windows,
        window_indices=window_indices,
        offsets=offsets,
        tokens_filename=tokens_filename,
    )


@click.command()
@click.argument("filename", type=click.Path(exists=True), required=True, nargs=-1)
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
    "--max-window-tokens",
    type=int,
    default=128,
    show_default=True,
    help="Maximum window size for embedding tokens",
)
@click.option(
    "--min-window-tokens",
    type=int,
    default=128,
    show_default=True,
    help="Minimum window size for embedding tokens",
)
@click.option(
    "--divide-factor",
    type=int,
    default=4,
    show_default=True,
    help="Recursive factor to divide window size by",
)
@click.option(
    "--use-offset",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to use an offsetted window when embedding",
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
    "--svm",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use SVM instead of kNN",
)
@click.option(
    "--svm-c",
    type=float,
    default=1.0,
    show_default=True,
    help="SVM regularization parameter; higher values penalize mispredictions more",
)
@click.option(
    "--force", is_flag=True, default=False, help="Force process even if cached"
)
@click.option(
    "--semantra-dir",
    type=click.Path(exists=False),
    default=None,
    help="Directory to store semantra files in",
)
def get_embeddings(
    filename,
    max_window_tokens=128,
    min_window_tokens=128,
    divide_factor=4,
    use_offset=True,
    pool_size=None,
    pool_count=None,
    doc_token_pre=None,
    doc_token_post=None,
    query_token_pre=None,
    query_token_post=None,
    model="mpnet",
    transformer_model=None,
    svm=False,
    svm_c=1.0,
    force=False,
    semantra_dir=None,  # auto
):
    if transformer_model is not None:
        # Handle custom transformers model
        if pool_size is None:
            pool_size = TRANSFORMER_POOL_DEFAULT

        model_params = {"type": "transformers", "model_name": transformer_model}
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
        model_params = model_config["params"]
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

    documents = {
        fn: process(
            filename=fn,
            semantra_dir=semantra_dir,
            model=model,
            model_params=model_params,
            num_dimensions=model.get_num_dimensions(),
            max_window_tokens=max_window_tokens,
            min_window_tokens=min_window_tokens,
            divide_factor=divide_factor,
            use_offset=use_offset,
            cost_per_token=cost_per_token,
            pool_count=pool_count,
            pool_size=pool_size,
            doc_token_pre=doc_token_pre,
            doc_token_post=doc_token_post,
            query_token_pre=query_token_pre,
            query_token_post=query_token_post,
        )
        for fn in filename
    }

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
        return send_from_directory("client/public", "index.html")

    # Path for all the static files (compiled JS/CSS, etc.)
    @app.route("/<path:path>")
    def home(path):
        return send_from_directory("client/public", path)

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

    @app.route("/api/querysvm", methods=["POST"])
    def querysvm():
        from sklearn import svm

        queries = request.json["queries"]
        preferences = request.json["preferences"]

        query_embedding = model.embed_queries(queries) if len(queries) > 0 else None
        results = []
        for doc in documents.values():
            embeddings = doc.embeddings

            # Add preferences to embeddings
            preference_embedding = np.sum(
                [
                    *([query_embedding] if query_embedding is not None else []),
                    *[
                        documents[pref["filename"]].embeddings[pref["index"]]
                        * pref["weight"]
                        for pref in preferences
                    ],
                ],
                axis=0,
            )

            x = np.concatenate([embeddings, preference_embedding[None, ...]])
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
            for i, index in enumerate(sorted_ix[:10]):
                distance = similarities[index]
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                    }
                )
            results.append([doc.filename, sub_results])

        return jsonify(results)

    @app.route("/api/query", methods=["POST"])
    def query():
        queries = request.json["queries"]
        preferences = request.json["preferences"]
        if svm:
            return querysvm()
        query_embedding = model.embed_queries(queries) if len(queries) > 0 else None
        results = []
        print(queries)
        print(preferences)
        print("---")
        for doc in documents.values():
            # Add preferences to embeddings
            preference_embedding = np.sum(
                [
                    *([query_embedding] if query_embedding is not None else []),
                    *[
                        documents[pref["filename"]].embeddings[pref["index"]]
                        * pref["weight"]
                        for pref in preferences
                    ],
                ],
                axis=0,
            )

            embeddings_dbs = doc.embeddings_dbs
            text_chunks = doc.text_chunks
            offsets = doc.offsets
            sub_results = []
            for i, [index, distance] in enumerate(
                zip(
                    *embeddings_dbs[0].get_nns_by_vector(
                        preference_embedding, 10, -1, True
                    )
                )
            ):
                offset = offsets[index]
                text = join_text_chunks(text_chunks[offset[0] : offset[1]])
                sub_results.append(
                    {
                        "text": text,
                        "distance": distance,
                        "offset": offset,
                        "index": int(index),
                    }
                )
            results.append([doc.filename, sub_results])
        return jsonify(results)

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

    print("Running at port 8080")
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    get_embeddings()
