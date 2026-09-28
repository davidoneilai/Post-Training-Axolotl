#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-/workspace/axolotl}"

CONFIG="${1:-${PIPELINE_ROOT}/configs/sft/template-lora-b200.yml}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Erro: config não encontrada: ${CONFIG}" >&2
  echo "Uso: $0 [caminho/para/config.yml]" >&2
  exit 1
fi

if [[ ! -d "${AXOLOTL_ROOT}" ]]; then
  echo "Erro: Axolotl não encontrado em ${AXOLOTL_ROOT}" >&2
  echo "Clone o repositório em ./axolotl ou ajuste AXOLOTL_ROOT." >&2
  exit 1
fi

if [[ -f "${AXOLOTL_ROOT}/scripts/cuda13_env.sh" ]]; then
  # Necessário no cu130: sem isso, exec/bash cai no libcublas do host e quebra GEMM (CUBLAS_STATUS_INVALID_VALUE).
  # shellcheck source=/dev/null
  source "${AXOLOTL_ROOT}/scripts/cuda13_env.sh"
fi

# outputs/ é bind-mount do host (./outputs). /workspace/pipeline raiz NÃO é montada,
# então redirecionar para lá deixa o .log só dentro do container.
LOG_DIR="${PIPELINE_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"
CONFIG_STEM="$(basename "${CONFIG}")"
CONFIG_STEM="${CONFIG_STEM%.*}"
LOG_FILE="${LOG_DIR}/${CONFIG_STEM}-$(date +%Y%m%d-%H%M%S).log"

echo "==> Config: ${CONFIG}"
echo "==> Axolotl: ${AXOLOTL_ROOT}"
echo "==> Log (host: ./outputs/logs/): ${LOG_FILE}"
echo "==> Iniciando SFT..."

cd "${AXOLOTL_ROOT}"
set +e
axolotl train "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
exit_code=${PIPESTATUS[0]}
set -e
echo "==> Log salvo em: ${LOG_FILE}"
exit "${exit_code}"
