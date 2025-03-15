from bs4 import BeautifulSoup

import os
import random
import re

class TipsterParser:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def parse_document(self, file_path):
        with open(file_path, 'r', encoding='latin-1') as f: #con utf-8 avevo alcuni problemi
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
        if text: 
            text = self.delete_tags(text)

        title = None
        possible_title_tags = ['title', 'head', 'hl', 'ttl', 'dochead', 'caption', 'subject', 'descriptor']

        for tag in possible_title_tags:
            title_tag = doc.find(tag)
            if title_tag:
                title = title_tag.get_text(strip=True)
                break

        if docno and text:
            return {'DOCNO': docno, 'TITLE': title, 'TEXT': text}
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
                try:
                    file_documents = self.parse_document(file_path)
                    all_documents.extend(file_documents)
                except Exception as e:
                    print(f'Failed to parse file {file_path}')
                    with open('storage/logs/failures.txt', 'a', encoding='utf-8') as f:
                        f.write(f'{file_path} \n{e} \n')

        return all_documents

class QueryParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_queries(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {self.file_path}")
            return {}
        except Exception as e:
            print(f"Error reading file: {self.file_path} - {e}")
            return {}

        soup = BeautifulSoup(content, 'html.parser')
        tops = soup.find_all('top')

        parsed_queries = {}
        for top in tops:
            query_string = str(top)
            query_string = self.clean_extra_closing_tags(query_string)
            query_data = self.extract_query_data(query_string)

            if query_data and query_data['NUM'] != 'N/A':
                parsed_queries[query_data['NUM']] = query_data

        return parsed_queries
    
    def extract_query_data(self, query):
        parts = re.split(r'<(num|title|desc|narr)>', query)

        query_data = {'NUM': 'N/A', 'TITLE': 'N/A', 'DESC': 'N/A', 'NARR': 'N/A'}

        for i in range(1, len(parts), 2):
            tag = parts[i].strip()
            content = parts[i + 1].strip()

            if tag == 'num':
                query_data['NUM'] = self.clean_label(content, 'Number:')
            elif tag == 'title':
                query_data['TITLE'] = content.strip()
            elif tag == 'desc':
                query_data['DESC'] = self.clean_label(content, 'Description:')
            elif tag == 'narr':
                query_data['NARR'] = self.clean_label(content, 'Narrative:')

        return query_data
    
    def clean_label(self, text, label):
        return text.replace(label, '').strip()
    
    def clean_extra_closing_tags(self, text):
        tags_to_remove = ['</num>', '</title>', '</desc>', '</narr>', '</top>']

        for tag in tags_to_remove:
            text = text.replace(tag, '')

        return text.strip()

class QrelsParser:
    def __init__(self, qrels_file):
        self.qrels_file = qrels_file

    def parse_qrels(self, query_id=None):
        qrels = {}

        # If not asked for a specific query return all relevant qrels
        if query_id is None:
            with open(self.qrels_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        q_id = parts[0]
                        doc_id = parts[2]
                        relevance = int(parts[3])

                        if q_id not in qrels:
                            qrels[q_id] = []

                        if relevance == 1:
                            qrels[q_id].append(doc_id)
            return qrels
        # If asked for a specific query return all relevance feedback documents split
        # between relevant and not relevant
        else:
            relevant_docs = []
            non_relevant_docs = []

            with open(self.qrels_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 4 and parts[0] == query_id:
                        doc_id = parts[2]
                        relevance = int(parts[3])
                        if relevance == 1:
                            relevant_docs.append(doc_id)
                        else:
                            non_relevant_docs.append(doc_id)

            return {"relevant": relevant_docs, "non_relevant": non_relevant_docs}

class OracleParser:
    def __init__(self, oracle_file):
        self.oracle_file = oracle_file
        os.makedirs(os.path.dirname(oracle_file), exist_ok=True)

    def parse_oracle(self, query_id):
        oracle_result = []

        with open(self.oracle_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0] == query_id:
                    doc_id = parts[1]
                    oracle_result.append(doc_id)

        return oracle_result
    
    def create_oracle(self, qrels_file):
        qrels = {}

        with open(qrels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 4:
                    q_id = parts[0]
                    doc_id = parts[2]
                    relevance = int(parts[3])

                    if q_id not in qrels:
                        qrels[q_id] = []

                    if relevance == 1:
                        qrels[q_id].append(doc_id)
        
        with open(self.oracle_file, 'w') as file:
            for q_id, docs in qrels.items():
                if len(docs) > 10:
                    selected_docs = random.sample(docs, 10)
                else:
                    selected_docs = docs
                
                for doc in selected_docs:
                    file.write(q_id + " " + doc + " 1" "\n")
