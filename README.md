# Small Language Model (SLM) - Treinamento Local

Uma implementação completa de uma Small Language Model (SLM) em Python usando PyTorch. Este modelo aprende a gerar texto baseado em arquivos Markdown (.md) fornecidos na pasta `files`.

## Características

- **Arquitetura Transformer**: Implementação from-scratch de um modelo transformer com multi-head attention
- **Treinamento Local**: Treina completamente offline usando seus próprios dados
- **Flexível**: Suporta múltiplos arquivos .md como fonte de treinamento
- **Geração de Texto**: Gera texto coerente baseado no conteúdo aprendido
- **Leve**: Modelo pequeno otimizado para rodar em CPUs e GPUs modestas

## Estrutura do Projeto

```
LLM-from-scratch/
├── files/                  # Pasta para arquivos .md de treinamento
│   └── exemplo.md         # Arquivo de exemplo
├── src/
│   ├── data_loader.py     # Carregamento e preprocessamento de dados
│   └── model.py           # Arquitetura da SLM (Transformer)
├── checkpoints/           # Modelos treinados salvos aqui
├── train.py              # Script de treinamento
├── inference.py          # Script de geração de texto
└── requirements.txt      # Dependências do projeto
```

## Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd LLM-from-scratch
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

### 1. Preparar os Dados

Coloque todos os seus arquivos `.md` na pasta `files/`. O modelo usará todos os arquivos dessa pasta para treinamento.

```bash
# Exemplo: adicionar seus arquivos
cp seu-arquivo.md files/
cp outro-arquivo.md files/
```

### 2. Treinar o Modelo

Execute o script de treinamento:

```bash
# Treinamento básico (configurações padrão)
python train.py

# Treinamento personalizado
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --d_model 256 \
  --num_layers 4 \
  --lr 0.001
```

#### Parâmetros de Treinamento

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--data_dir` | `files` | Diretório com arquivos .md |
| `--seq_length` | `100` | Comprimento da sequência |
| `--stride` | `10` | Passo entre sequências |
| `--d_model` | `256` | Dimensão do modelo |
| `--num_heads` | `4` | Número de cabeças de atenção |
| `--num_layers` | `4` | Número de camadas transformer |
| `--d_ff` | `512` | Dimensão da rede feed-forward |
| `--batch_size` | `32` | Tamanho do batch |
| `--epochs` | `50` | Número de épocas |
| `--lr` | `0.001` | Taxa de aprendizado |
| `--dropout` | `0.1` | Taxa de dropout |

### 3. Gerar Texto

Após treinar o modelo, use o script de inferência para gerar texto:

```bash
# Modo interativo
python inference.py --interactive

# Geração única com prompt
python inference.py --prompt "A inteligência artificial" --max_length 300

# Ajustar criatividade (temperature)
python inference.py \
  --prompt "Machine Learning é" \
  --temperature 1.2 \
  --max_length 200 \
  --top_k 50
```

#### Parâmetros de Geração

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--checkpoint` | `checkpoints/best_model.pt` | Caminho do modelo |
| `--prompt` | `""` | Texto inicial |
| `--max_length` | `200` | Comprimento máximo gerado |
| `--temperature` | `0.8` | Criatividade (0.1-2.0) |
| `--top_k` | `40` | Top-k sampling |
| `--interactive` | `False` | Modo interativo |

**Dica sobre Temperature:**
- `0.1-0.5`: Mais conservador e repetitivo
- `0.8-1.0`: Equilibrado
- `1.0-2.0`: Mais criativo e aleatório

## Arquitetura do Modelo

### Transformer Components

1. **Token Embedding**: Converte caracteres em vetores densos
2. **Positional Embedding**: Adiciona informação de posição
3. **Multi-Head Attention**: Mecanismo de atenção com múltiplas cabeças
4. **Feed-Forward Networks**: Redes neurais densas
5. **Layer Normalization**: Normalização entre camadas
6. **Residual Connections**: Conexões residuais para melhor gradiente

### Configuração Padrão

- **Vocabulário**: Baseado em caracteres (todos os caracteres únicos dos textos)
- **Dimensão do modelo (d_model)**: 256
- **Cabeças de atenção**: 4
- **Camadas**: 4
- **Parâmetros**: ~1-2M (dependendo do vocabulário)

## Exemplos

### Exemplo 1: Treinar com textos técnicos

```bash
# Adicione seus documentos técnicos
cp documentacao/*.md files/

# Treine com mais épocas para melhor qualidade
python train.py --epochs 100 --batch_size 16

# Gere texto técnico
python inference.py --prompt "O sistema funciona" --temperature 0.7
```

### Exemplo 2: Modo Interativo

```bash
python inference.py --interactive

# No prompt interativo:
Prompt: Deep Learning é
Usar configurações padrão? (s/n): s

# O modelo gerará continuação do texto
```

### Exemplo 3: Ajustar tamanho do modelo

```bash
# Modelo menor (mais rápido, menos parâmetros)
python train.py --d_model 128 --num_layers 3 --num_heads 4

# Modelo maior (melhor qualidade, mais lento)
python train.py --d_model 512 --num_layers 6 --num_heads 8
```

## Monitoramento do Treinamento

Durante o treinamento, você verá:

```
Época 1/50
----------------------------------------------------------------------
  Batch [10/100] - Loss: 3.2451
  Batch [20/100] - Loss: 2.8932
  ...

Época 1 - Loss médio: 2.5432
Novo melhor modelo! Loss: 2.5432
Checkpoint salvo: checkpoints/model_epoch_1.pt

Texto gerado:
----------------------------------------------------------------------
A inteligência artificial permite que sistemas...
----------------------------------------------------------------------
```

## Dicas e Melhores Práticas

1. **Quantidade de Dados**: Quanto mais texto, melhor. Idealmente 100KB+ de texto
2. **Épocas**: Comece com 50-100 épocas e monitore o overfitting
3. **Seq_length**: Sequências maiores capturam mais contexto, mas usam mais memória
4. **Learning Rate**: Se a loss não diminuir, tente reduzir o learning rate
5. **Temperature**: Para texto mais consistente, use temperature baixa (0.5-0.7)

## Requisitos do Sistema

- **CPU**: Qualquer processador moderno
- **RAM**: 4GB+ recomendado
- **GPU**: Opcional, mas acelera o treinamento significativamente
- **Python**: 3.8+
- **Espaço em disco**: ~500MB para checkpoints

## Solução de Problemas

### O modelo não aprende (loss não diminui)
- Reduza o learning rate: `--lr 0.0001`
- Aumente o número de épocas
- Verifique se há dados suficientes

### Out of Memory
- Reduza `--batch_size`
- Reduza `--seq_length`
- Reduza `--d_model` ou `--num_layers`

### Texto gerado não faz sentido
- Treine por mais épocas
- Adicione mais dados de treinamento
- Ajuste a temperature na inferência

## Próximos Passos

Possíveis melhorias:

- [ ] Implementar tokenização BPE para vocabulário maior
- [ ] Adicionar validação set para evitar overfitting
- [ ] Implementar beam search para melhor geração
- [ ] Adicionar suporte para múltiplos idiomas
- [ ] Criar interface web para geração de texto
- [ ] Implementar fine-tuning para tarefas específicas

## Licença

MIT License

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Referências

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Paper original do Transformer
- [PyTorch Documentation](https://pytorch.org/docs/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
