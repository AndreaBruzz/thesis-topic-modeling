from parsers import TipsterParser

import os
import utils

if __name__ == "__main__":
    leaves = utils.get_leaf_directories('tipster/corpora_unzipped')
    output_base = 'docs'

    for leaf in leaves:
        print(f'DEBUG: Processing leaf directory: {leaf}')
        parser = TipsterParser(leaf)
        parsed_documents = parser.process_all_documents()

        output_dir = utils.create_output_directory(leaf, output_base)

        for doc in parsed_documents:
            output_file_path = os.path.join(output_dir, f"{doc['DOCNO']}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(f"TEXT:\n{doc['TEXT']}")
