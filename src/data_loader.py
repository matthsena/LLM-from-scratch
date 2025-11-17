"""
Data Loader para SLM
Carrega e preprocessa todos os arquivos .md da pasta files
"""

import os
import glob
from typing import List, Tuple
import numpy as np


class TextDataLoader:
    def __init__(self, data_dir: str = "files"):
        """
        Inicializa o data loader

        Args:
            data_dir: Diretório contendo os arquivos .md
        """
        self.data_dir = data_dir
        self.text = ""
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def load_data(self) -> str:
        """
        Carrega todos os arquivos .md do diretório

        Returns:
            Texto concatenado de todos os arquivos
        """
        md_files = glob.glob(os.path.join(self.data_dir, "*.md"))

        if not md_files:
            raise ValueError(f"Nenhum arquivo .md encontrado em {self.data_dir}")

        texts = []
        print(f"Carregando {len(md_files)} arquivo(s) .md...")

        for file_path in md_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
                print(f"  - {os.path.basename(file_path)}: {len(content)} caracteres")

        self.text = "\n\n".join(texts)
        print(f"\nTotal de caracteres: {len(self.text)}")

        return self.text

    def build_vocab(self):
        """
        Constrói o vocabulário a partir do texto
        """
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Vocabulário construído: {self.vocab_size} caracteres únicos")

    def encode(self, text: str) -> List[int]:
        """
        Converte texto em sequência de índices

        Args:
            text: Texto para codificar

        Returns:
            Lista de índices
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: List[int]) -> str:
        """
        Converte sequência de índices em texto

        Args:
            indices: Lista de índices

        Returns:
            Texto decodificado
        """
        return ''.join([self.idx_to_char[idx] for idx in indices])

    def create_sequences(self, seq_length: int = 100, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências de treinamento

        Args:
            seq_length: Comprimento de cada sequência
            stride: Passo entre sequências

        Returns:
            Tupla (X, y) com dados de entrada e saída
        """
        encoded_text = self.encode(self.text)

        X = []
        y = []

        for i in range(0, len(encoded_text) - seq_length, stride):
            X.append(encoded_text[i:i + seq_length])
            y.append(encoded_text[i + 1:i + seq_length + 1])

        X = np.array(X)
        y = np.array(y)

        print(f"Sequências criadas: {len(X)} sequências de comprimento {seq_length}")

        return X, y

    def prepare_data(self, seq_length: int = 100, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara todos os dados para treinamento

        Args:
            seq_length: Comprimento de cada sequência
            stride: Passo entre sequências

        Returns:
            Tupla (X, y) com dados de entrada e saída
        """
        self.load_data()
        self.build_vocab()
        return self.create_sequences(seq_length, stride)


if __name__ == "__main__":
    # Teste do data loader
    loader = TextDataLoader("files")
    X, y = loader.prepare_data(seq_length=50)

    print(f"\nShape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"\nExemplo de sequência:")
    print(f"Input:  {loader.decode(X[0].tolist())}")
    print(f"Target: {loader.decode(y[0].tolist())}")
