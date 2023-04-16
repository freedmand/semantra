import pypdfium2 as pdfium
from multiprocessing import Lock
import json
import os
import hashlib
import tqdm


class PDFContent:
    def __init__(self, rawtext, filename, positions, pdf):
        self.rawtext = rawtext
        self.filename = filename
        self.positions = positions
        self.pdf = pdf
        self.pdfium = pdfium.PdfDocument(filename)
        self.mutex = Lock()
        self.filetype = "pdf"

    def __del__(self):
        self.pdf.close()

    def get_page_image_pil(self, page_number, scale):
        with self.mutex:
            page = self.pdfium[page_number]
            bitmap = page.render(scale=scale)
            return bitmap.to_pil()

    def get_page_chars(self, page_number):
        textmap = self.pdf.pages[page_number].get_textmap()
        return textmap.tuples


def file_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:10]


def get_pdf_content(filename, semantra_dir, base_filename):
    hash = file_md5(filename)
    converted_txt = os.path.join(semantra_dir, base_filename + f".{hash}.txt")
    position_index = os.path.join(
        semantra_dir, base_filename + f".{hash}.positions.json"
    )

    import pdfplumber

    pdf = pdfplumber.open(filename)

    if not os.path.exists(converted_txt) or not os.path.exists(position_index):
        positions = []
        position = 0
        with open(converted_txt, "w", encoding="utf-8", errors="ignore") as f:
            for page in tqdm.tqdm(pdf.pages, desc="Extracting PDF contents"):
                page_width = page.width
                page_height = page.height
                textmap = page.get_textmap()
                pagetext = "".join([tuple[0] for tuple in textmap.tuples])
                positions.append(
                    {
                        "char_index": position,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                )
                position += f.write(pagetext)
                position += f.write("\f")
        with open(position_index, "w") as f:
            json.dump(positions, f)
        with open(converted_txt, "r") as f:
            rawtext = f.read()
        return PDFContent(rawtext, filename, positions, pdf)
    else:
        with open(converted_txt, "r", encoding="utf-8", errors="ignore") as f:
            rawtext = f.read()
        with open(position_index, "r") as f:
            positions = json.load(f)

        return PDFContent(rawtext, filename, positions, pdf)
