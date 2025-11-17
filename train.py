"""
Script de treinamento para a Small Language Model (SLM)
Treina o modelo usando arquivos .md da pasta files
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import json

from src.data_loader import TextDataLoader
from src.model import SmallLanguageModel


class TextDataset(Dataset):
    """Dataset personalizado para sequências de texto"""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    """Classe para treinar a SLM"""

    def __init__(
        self,
        model,
        train_loader,
        data_loader,
        device,
        learning_rate=0.001,
        checkpoint_dir="checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.data_loader = data_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X)

            # Calcula loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Log do progresso
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Batch [{batch_idx + 1}/{num_batches}] - Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch, loss, config):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': config,
            'vocab_size': self.data_loader.vocab_size,
            'char_to_idx': self.data_loader.char_to_idx,
            'idx_to_char': self.data_loader.idx_to_char
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint salvo: {checkpoint_path}")

        # Salva também como best_model.pt
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)

    def train(self, num_epochs, config):
        """Loop principal de treinamento"""
        print(f"\nIniciando treinamento por {num_epochs} épocas...")
        print("=" * 70)

        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"\nÉpoca {epoch}/{num_epochs}")
            print("-" * 70)

            avg_loss = self.train_epoch(epoch)

            print(f"\nÉpoca {epoch} - Loss médio: {avg_loss:.4f}")

            # Salva checkpoint se for o melhor modelo
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss, config)
                print(f"Novo melhor modelo! Loss: {avg_loss:.4f}")

            # Gera texto de exemplo
            if epoch % 5 == 0 or epoch == num_epochs:
                self.generate_sample()

        print("\n" + "=" * 70)
        print("Treinamento concluído!")

    def generate_sample(self):
        """Gera um texto de exemplo"""
        self.model.eval()
        with torch.no_grad():
            # Começa com alguns caracteres
            start_text = "A inteligência"
            start_idx = self.data_loader.encode(start_text)
            start_tensor = torch.tensor([start_idx], dtype=torch.long).to(self.device)

            # Gera novos tokens
            generated = self.model.generate(
                start_tensor,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40
            )

            # Decodifica
            generated_text = self.data_loader.decode(generated[0].tolist())

            print(f"\nTexto gerado:")
            print("-" * 70)
            print(generated_text)
            print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Treinar Small Language Model")
    parser.add_argument("--data_dir", type=str, default="files", help="Diretório com arquivos .md")
    parser.add_argument("--seq_length", type=int, default=100, help="Comprimento da sequência")
    parser.add_argument("--stride", type=int, default=10, help="Passo entre sequências")
    parser.add_argument("--d_model", type=int, default=256, help="Dimensão do modelo")
    parser.add_argument("--num_heads", type=int, default=4, help="Número de cabeças de atenção")
    parser.add_argument("--num_layers", type=int, default=4, help="Número de camadas transformer")
    parser.add_argument("--d_ff", type=int, default=512, help="Dimensão feed-forward")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Taxa de dropout")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Diretório para checkpoints")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Carrega dados
    print("\n" + "=" * 70)
    print("CARREGANDO DADOS")
    print("=" * 70)
    data_loader = TextDataLoader(args.data_dir)
    X, y = data_loader.prepare_data(seq_length=args.seq_length, stride=args.stride)

    # Cria dataset e dataloader
    dataset = TextDataset(X, y)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Cria modelo
    print("\n" + "=" * 70)
    print("CRIANDO MODELO")
    print("=" * 70)

    config = {
        'vocab_size': data_loader.vocab_size,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_seq_length': args.seq_length,
        'dropout': args.dropout
    }

    model = SmallLanguageModel(**config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Vocabulário: {data_loader.vocab_size} caracteres")
    print(f"Parâmetros totais: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"Tamanho do dataset: {len(dataset)} sequências")
    print(f"Batches por época: {len(train_loader)}")

    # Salva configuração
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Treina
    print("\n" + "=" * 70)
    print("TREINAMENTO")
    print("=" * 70)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        data_loader=data_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer.train(args.epochs, config)


if __name__ == "__main__":
    main()
