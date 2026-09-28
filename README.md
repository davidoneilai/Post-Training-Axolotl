# SFT com Axolotl (B200)

Pipeline Docker para fine-tuning com Axolotl em GPUs Blackwell (cu130).

> **Nota:** este repositório contém apenas o pipeline (Docker, scripts e configs).
> A pasta `axolotl/` **não faz parte do Git** — clone o upstream separadamente (passo 1).

## 1. Clone

```bash
git clone https://github.com/SEU-USUARIO/Post-Training-Axolotl.git
cd teste-axo

git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl && git checkout 3c4ff59f && cd ..
```

## 2. Build

```bash
cp .env.example .env
# Preencha HF_TOKEN (datasets/modelos gated) e, se quiser, WANDB_API_KEY
docker compose build
```

## 3. Run

```bash
docker compose up -d
```

Se alterar `docker-compose.yml` ou `.env`, recrie o container:

```bash
docker compose up -d --force-recreate
```

### Treino (recomendado)

Use sempre `run-sft.sh` — ele configura o ambiente cu130 antes de chamar o Axolotl e grava o log em `outputs/logs/` (visível no host):

```bash
docker compose exec axolotl-sft bash /workspace/pipeline/scripts/run-sft.sh \
  /workspace/pipeline/configs/sft/template-lora-b200.yml
```

Ou entre no container e rode de lá:

```bash
docker compose exec axolotl-sft bash

bash /workspace/pipeline/scripts/run-sft.sh \
  /workspace/pipeline/configs/qwen/3.5-4b-only-raft.yaml
```

Em background (`nohup`), **não** redirecione para `/workspace/pipeline/*.log` — essa pasta raiz não é montada no host. O script já faz `tee` para `outputs/logs/`:

```bash
nohup bash /workspace/pipeline/scripts/run-sft.sh \
  /workspace/pipeline/configs/qwen/3.5-4b-only-raft.yaml >/dev/null 2>&1 &
```

No host: `tail -f outputs/logs/<config>-*.log`

Checkpoints e logs em `outputs/` (pasta local, ignorada pelo Git).

## Notas (B200 / cu130)

**cuBLAS:** na imagem cu130, o PyTorch precisa do `libcublas` do pacote `nvidia.cu13`. Sem isso, o treino pode falhar com `CUBLAS_STATUS_INVALID_VALUE` no RoPE. O `docker-compose.yml` já define `LD_LIBRARY_PATH` e o `run-sft.sh` faz `source` do `cuda13_env.sh`.

Se rodar `axolotl train` **direto** (sem `run-sft.sh`), carregue o env antes:

```bash
source /workspace/axolotl/scripts/cuda13_env.sh
```

**Datasets no config:**

| Origem | Exemplo no YAML |
|--------|-----------------|
| Hugging Face Hub | `path: org/dataset` + `split: train` |
| Arquivo local (montado em `./data`) | `path: /workspace/pipeline/data/...` |

Paths relativos como `data/...` **não funcionam** — o Axolotl roda a partir de `/workspace/axolotl`.

**Dados locais:** coloque em `./data/` no host (ex.: `./data/meu-dataset/train.jsonl`). Apenas `data/exemplo.jsonl` é versionado como referência.

## Licença

MIT — ver [LICENSE](LICENSE).
