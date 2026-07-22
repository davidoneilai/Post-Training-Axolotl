#!/usr/bin/env bash
set -euo pipefail

echo "=== Verificação de GPU (B200 / cu130) ==="
echo

python - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    print(f"ERRO: PyTorch não instalado: {e}")
    sys.exit(1)

print(f"PyTorch:     {torch.__version__}")
print(f"CUDA avail:  {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERRO: CUDA não disponível. Verifique --gpus e o driver NVIDIA.")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cc = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"GPU {i}:       {name}")
    print(f"  CC:        sm_{cc[0]}{cc[1]}")
    print(f"  VRAM:      {mem:.1f} GB")

cc0 = torch.cuda.get_device_capability(0)
if cc0[0] >= 10:
    print("\nOK: Blackwell detectado (sm_100+). Stack cu130 adequada.")
elif cc0[0] >= 9:
    print("\nOK: Hopper detectado (sm_90+).")
else:
    print(f"\nAVISO: GPU sm_{cc0[0]}{cc0[1]} — confira se a imagem cu130 é a desejada.")
PY

echo
echo "=== Teste rápido de tensor na GPU ==="
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); print('Matmul OK:', (x@x).shape)"

echo
echo "Verificação concluída."
