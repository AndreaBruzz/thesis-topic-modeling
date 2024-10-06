from bs4 import BeautifulSoup
import os

class TipsterParser:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def parse_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        documents = soup.find_all('doc')

        parsed_documents = []
        for doc in documents:
            doc_data = self.extract_document_data(doc)
            if doc_data:
                parsed_documents.append(doc_data)

        return parsed_documents

    def extract_document_data(self, doc):
        docno = doc.find('docno').get_text(strip=True) if doc.find('docno') else None
        text = doc.find('text').decode_contents() if doc.find('text') else None
        text = self.delete_tags(text)

        if docno and text:
            return {'DOCNO': docno, 'TEXT': text}
        else:
            return None

    def delete_tags(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator="\n", strip=True)
        return clean_text

    def process_all_documents(self):
        all_documents = []
        for file in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, file)
            if os.path.isfile(file_path) and file.endswith('.txt'):
                file_documents = self.parse_document(file_path)
                all_documents.extend(file_documents)
        return all_documents
