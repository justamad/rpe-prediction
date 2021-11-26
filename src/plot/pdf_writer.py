from PyPDF2 import PdfFileWriter, PdfFileReader
from matplotlib.backends.backend_pdf import PdfPages

import io


class PDFWriter(object):

    def __init__(self, file_name: str):
        self.__pdf_buffer = io.BytesIO()
        self.__pp = PdfPages(self.__pdf_buffer)
        self.__bookmarks = []
        self.__file_name = file_name

    def save_figure(self):
        # TODO: Add optional bookmark parameters to here!
        self.__pp.savefig()

    def add_booklet(self, subject_name: str, set_nr: int, joints: list):
        for joint in joints:
            self.__bookmarks.append((subject_name, set_nr, joint))

    def close_and_save_file(self, add_bookmarks: bool = True):
        self.__pp.close()
        output_file = PdfFileWriter()
        input_file = PdfFileReader(self.__pdf_buffer)

        bookmark_cache = {}
        for nr_page in range(input_file.getNumPages()):
            output_file.addPage(input_file.getPage(nr_page))

            if add_bookmarks:
                # Determine the parent bookmark
                subject_name, nr_set, joint = self.__bookmarks[nr_page]
                if subject_name not in bookmark_cache:
                    bookmark_cache[subject_name] = output_file.addBookmark(subject_name, nr_page, parent=None)
                subject_bookmark = bookmark_cache[subject_name]

                # Determine set bookmark
                key = subject_name + str(nr_set)
                if key not in bookmark_cache:
                    bookmark_cache[key] = output_file.addBookmark(f"Set {nr_set}", nr_page, parent=subject_bookmark)
                set_bookmark = bookmark_cache[key]

                output_file.addBookmark(joint, nr_page, parent=set_bookmark)

        output_stream = open(self.__file_name, 'wb')
        output_file.write(output_stream)
        output_stream.close()
