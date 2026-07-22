# Pipeline SFT para NVIDIA B200 (Blackwell) — baseado na imagem oficial Axolotl.
# Não reconstruímos PyTorch/CUDA; usamos a stack cu130 já validada pela AxolotlAI.
FROM axolotlai/axolotl-uv:main-py3.12-cu130-2.10.0

LABEL org.opencontainers.image.title="axolotl-sft-b200"
LABEL org.opencontainers.image.description="Pipeline de SFT com Axolotl otimizado para GPU B200"

RUN apt-get update && \
    apt-get install -y --no-install-recommends vim curl jq nano git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pipeline

COPY configs ./configs
COPY scripts ./scripts

RUN chmod +x scripts/*.sh && \
    install -m 755 scripts/entrypoint.sh /usr/local/bin/pipeline-entrypoint.sh

ENV AXOLOTL_ROOT=/workspace/axolotl
ENV PIPELINE_ROOT=/workspace/pipeline

ENTRYPOINT ["/usr/local/bin/pipeline-entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
