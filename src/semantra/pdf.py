import pypdfium2 as pdfium
from threading import Lock
import json
import os
from tqdm import tqdm
from .util import get_converted_pdf_txt_filename, get_pdf_positions_filename

mutexes = {}


def get_mutex(filename):
    # Ensure that only one thread is accessing a PDF at a time
    if filename not in mutexes:
        mutexes[filename] = Lock()
    return mutexes[filename]


class PDFContent:
    def __init__(self, rawtext, filename, positions):
        self.rawtext = rawtext
        self.filename = filename
        self.positions = positions
        self.pdfium = pdfium.PdfDocument(filename)
        self.mutex = get_mutex(filename)
        self.filetype = "pdf"

    def get_page_image_pil(self, page_number, scale):
        with self.mutex:
            page = self.pdfium[page_number]
            bitmap = page.render(scale=scale)
            return bitmap.to_pil()

    def get_page_chars(self, page_number):
        with self.mutex:
            page = self.pdfium[page_number]
            textpage = page.get_textpage()
            num_chars = textpage.count_chars()
            char_boxes = [textpage.get_charbox(i) for i in range(num_chars)]
            chars = [
                textpage.get_text_range(index=i, count=1) for i in range(num_chars)
            ]
            return [(c, b) for c, b in list(zip(chars, char_boxes))]


# Page separator character
LINE_FEED = "\f"


def get_pdf_content(md5, filename, semantra_dir, force, silent):
    converted_txt = os.path.join(semantra_dir, get_converted_pdf_txt_filename(md5))
    position_index = os.path.join(semantra_dir, get_pdf_positions_filename(md5))

    pdf = pdfium.PdfDocument(filename)
    n_pages = len(pdf)

    if force or not os.path.exists(converted_txt) or not os.path.exists(position_index):
        positions = []
        position = 0
        # newline="" ensures pdfium's \r is preserved
        with open(converted_txt, "w", newline="") as f:
            for page_index in tqdm(
                range(n_pages),
                desc="Extracting PDF contents",
                leave=False,
                disable=silent,
            ):
                page = pdf[page_index]
                page_width, page_height = page.get_size()
                textpage = page.get_textpage()
                pagetext = textpage.get_text_range()

                positions.append(
                    {
                        "char_index": position,
                        "page_width": page_width,
                        "page_height": page_height,
                    }
                )
                position += f.write(pagetext)
                position += f.write(LINE_FEED)
        with open(position_index, "w") as f:
            json.dump(positions, f)
        with open(converted_txt, "r", newline="") as f:
            rawtext = f.read()
        return PDFContent(rawtext, filename, positions)
    else:
        with open(converted_txt, "r", newline="") as f:
            rawtext = f.read()
        with open(position_index, "r") as f:
            positions = json.load(f)

        return PDFContent(rawtext, filename, positions)
