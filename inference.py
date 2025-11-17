"""
Script de inferência para a Small Language Model (SLM)
Gera texto usando o modelo treinado
"""

import torch
import argparse
import os

from src.model import SmallLanguageModel


class TextGenerator:
    """Classe para gerar texto com o modelo treinado"""

    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Inicializa o gerador de texto

        Args:
            checkpoint_path: Caminho para o checkpoint do modelo
            device: Device para usar (cuda/cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Usando device: {self.device}")
        print(f"Carregando modelo de: {checkpoint_path}")

        # Carrega checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extrai configuração e vocabulário
        self.config = checkpoint['config']
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = checkpoint['vocab_size']

        # Cria modelo
        self.model = SmallLanguageModel(**self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Modelo carregado com sucesso!")
        print(f"Época do checkpoint: {checkpoint['epoch']}")
        print(f"Loss: {checkpoint['loss']:.4f}")
        print(f"Vocabulário: {self.vocab_size} caracteres")

    def encode(self, text: str):
        """Codifica texto em índices"""
        unknown_chars = [ch for ch in text if ch not in self.char_to_idx]
        if unknown_chars:
            raise ValueError(
                f"Caracteres desconhecidos no prompt: {unknown_chars}. "
                "Por favor, use apenas caracteres presentes no vocabulário."
            )
        return [self.char_to_idx[ch] for ch in text]
    def decode(self, indices):
        """Decodifica índices em texto"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])

    def generate(
        self,
        prompt: str = "",
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 40
    ):
        """
        Gera texto

        Args:
            prompt: Texto inicial (prompt)
            max_length: Comprimento máximo do texto gerado
            temperature: Temperatura para sampling (maior = mais criativo)
            top_k: Número de tokens mais prováveis a considerar

        Returns:
            Texto gerado
        """
        self.model.eval()

        # Se não há prompt, começa com um caractere aleatório
        if not prompt:
            prompt = " "

        # Codifica prompt
        encoded_prompt = self.encode(prompt)
        idx = torch.tensor([encoded_prompt], dtype=torch.long).to(self.device)

        print(f"\nPrompt: '{prompt}'")
        print(f"Gerando {max_length} caracteres...")
        print("-" * 70)

        # Gera texto
        with torch.no_grad():
            generated = self.model.generate(
                idx,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )

        # Decodifica
        generated_text = self.decode(generated[0].tolist())

        return generated_text

    def interactive_mode(self):
        """Modo interativo para gerar texto"""
        print("\n" + "=" * 70)
        print("MODO INTERATIVO")
        print("=" * 70)
        print("Digite um prompt e pressione Enter para gerar texto.")
        print("Digite 'sair' para encerrar.")
        print("=" * 70 + "\n")

        while True:
            try:
                prompt = input("Prompt: ")

                if prompt.lower() in ['sair', 'exit', 'quit']:
                    print("Encerrando...")
                    break

                # Parâmetros padrão
                max_length = 200
                temperature = 0.8
                top_k = 40

                # Permite configurar parâmetros
                config = input("Usar configurações padrão? (s/n): ").lower()
                if config == 'n':
                    try:
                        max_length = int(input(f"Comprimento máximo [{max_length}]: ") or max_length)
                        temperature = float(input(f"Temperature [{temperature}]: ") or temperature)
                        top_k = int(input(f"Top-k [{top_k}]: ") or top_k)
                    except ValueError:
                        print("Valores inválidos, usando padrão...")

                # Gera texto
                generated_text = self.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )

                print("\nTexto gerado:")
                print("-" * 70)
                print(generated_text)
                print("-" * 70 + "\n")

            except KeyboardInterrupt:
                print("\n\nEncerrando...")
                break
            except Exception as e:
                print(f"\nErro: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Gerar texto com SLM treinada")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Caminho para o checkpoint do modelo"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Texto inicial para geração"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Comprimento máximo do texto gerado"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperatura para sampling (0.1-2.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Número de tokens mais prováveis"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interativo"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device para usar (cuda/cpu)"
    )

    args = parser.parse_args()

    # Verifica se checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"Erro: Checkpoint não encontrado em {args.checkpoint}")
        print("Treine o modelo primeiro usando train.py")
        return

    # Cria gerador
    generator = TextGenerator(args.checkpoint, args.device)

    # Modo interativo ou geração única
    if args.interactive:
        generator.interactive_mode()
    else:
        generated_text = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )

        print("\nTexto gerado:")
        print("=" * 70)
        print(generated_text)
        print("=" * 70)


if __name__ == "__main__":
    main()
