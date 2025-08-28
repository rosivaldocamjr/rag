import os
import sys
import types
import runpy
import pytest


def test_missing_json(monkeypatch):
    sys.path.insert(0, os.getcwd())
    dummy_modules = {
        'langchain': types.ModuleType('langchain'),
        'langchain.text_splitter': types.ModuleType('langchain.text_splitter'),
        'langchain_community': types.ModuleType('langchain_community'),
        'langchain_community.embeddings': types.ModuleType('langchain_community.embeddings'),
        'langchain_community.vectorstores': types.ModuleType('langchain_community.vectorstores'),
        'langchain_experimental': types.ModuleType('langchain_experimental'),
        'langchain_experimental.text_splitter': types.ModuleType('langchain_experimental.text_splitter'),
        'langchain_core': types.ModuleType('langchain_core'),
        'langchain_core.documents': types.ModuleType('langchain_core.documents'),
    }

    dummy_modules['langchain.text_splitter'].RecursiveCharacterTextSplitter = object
    dummy_modules['langchain_community.embeddings'].SentenceTransformerEmbeddings = object
    dummy_modules['langchain_community.vectorstores'].FAISS = object
    dummy_modules['langchain_experimental.text_splitter'].SemanticChunker = object
    dummy_modules['langchain_core.documents'].Document = object

    for name, module in dummy_modules.items():
        sys.modules[name] = module

    original_exists = os.path.exists

    def fake_exists(path):
        if path == 'parsed_data.json':
            return False
        return original_exists(path)

    monkeypatch.setattr(os.path, 'exists', fake_exists)

    with pytest.raises(FileNotFoundError):
        runpy.run_path('ingestion.py', run_name='__main__')
