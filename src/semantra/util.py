import struct
import hashlib
import os
import numpy as np

HASH_LENGTH = 24


def file_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:HASH_LENGTH]


def join_text_chunks(chunks):
    return "".join(chunks)


# Filenames for generated files
def get_converted_pdf_txt_filename(md5):
    return f"{md5}.pdf.txt"


def get_pdf_positions_filename(md5):
    return f"{md5}.pdf.positions.json"


def get_tokens_filename(md5, config_hash):
    return f"{md5}.{config_hash}.tokens.json"


def get_embeddings_filename(md5, config_hash, size, offset, rewind):
    return f"{md5}.{config_hash}.{size}_{offset}_{rewind}.embeddings"


def get_annoy_filename(md5, config_hash, size, offset, rewind, num_trees):
    return f"{md5}.{config_hash}.{size}_{offset}_{rewind}.{num_trees}t.annoy"


def get_config_filename(md5, config_hash):
    return f"{md5}.{config_hash}.config.json"


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


def write_annoy_db(filename, num_dimensions, embeddings, num_trees):
    # Import annoy here so that it's not required for the CLI
    from annoy import AnnoyIndex

    dbs = []
    db = AnnoyIndex(num_dimensions, "angular")
    for i, embedding in enumerate(embeddings):
        db.add_item(i, embedding)
    db.build(num_trees)
    db.save(filename)
    dbs.append(db)

    return dbs


def load_annoy_db(filename, num_dimensions):
    # Import annoy here so that it's not required for the CLI
    from annoy import AnnoyIndex

    db = AnnoyIndex(num_dimensions, "angular")
    db.load(filename)
    return db


def get_num_annoy_embeddings(annoy_filename, num_dimensions):
    return load_annoy_db(annoy_filename, num_dimensions).get_n_items()


def safe_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def get_num_embeddings(embeddings_filename, num_dimensions):
    # Get the file size
    with open(embeddings_filename, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    # Calculate the number of embeddings
    return file_size // (num_dimensions * 4)


def read_embeddings_file(embeddings_filename, num_dimensions, capacity):
    # Calculate the number of embeddings
    num_embeddings = min(
        get_num_embeddings(embeddings_filename, num_dimensions), capacity
    )

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


def get_offsets(doc_size, windows):
    num_tokens = 0

    offsets = []

    for size, offset, rewind in windows:
        sub_offsets = []
        x = 0
        if offset > 0:
            sub_offsets.append([0, offset])
            num_tokens += offset
            x = offset
        else:
            x = rewind

        while x < doc_size:
            x -= rewind
            sub_offsets.append([x, min(x + size, doc_size)])
            num_tokens += min(x + size, doc_size) - x
            x += size

        offsets.append(sub_offsets)

    return offsets, num_tokens


def sort_results(results, reverse):
    # Get average distance per result
    avg_distances = []
    for result in results:
        avg_distances.append(np.mean([item["distance"] for item in result[1]]))

    # Sort results by average distance
    return {
        "results": [x for _, x in sorted(zip(avg_distances, results), reverse=reverse)],
        "sort": "desc" if reverse else "asc",
    }
