from parsers import TipsterParser

import utils
import es_helpers

if __name__ == "__main__":
    index = 'tipster'

    es = es_helpers.connect_elasticsearch()
    es_helpers.create_index(es, index)

    leaves = utils.get_leaf_directories('tipster/corpora_unzipped')

    for leaf in leaves:
        print(f'DEBUG: Processing leaf directory: {leaf}')
        parser = TipsterParser(leaf)
        parsed_documents = parser.process_all_documents()

        es_helpers.index_documents(es, index, parsed_documents)

