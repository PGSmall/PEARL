#!/bin/bash

ATTN=${1:-"pearl"}
TLP=${2:-"on"}
NGPUS=${3:-1}
LOGFILE=${4:-"results.txt"}

DATASETS=(
  voc21
  context60
  coco_object
  voc20
  context59
  coco_stuff164k
  city_scapes
  ade20k
)

: > "${LOGFILE}"

for BENCHMARK in "${DATASETS[@]}"; do
  {
    printf "\n=== DATASET: %s | Attn:%s | TLP:%s ===\n\n" \
      "${BENCHMARK}" "${ATTN}" "${TLP}"
  } >> "${LOGFILE}"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=1 \
  TORCH_CUDNN_SDPA_ENABLED=1 \
  torchrun --nproc_per_node="${NGPUS}" --master_port=29500 \
    eval.py \
    --launcher pytorch \
    --config "./configs/cfg_${BENCHMARK}.py" \
    --attn "${ATTN}" \
    --prop "${TLP}" \
    |& tee -a "${LOGFILE}"

  echo "----------" >> "${LOGFILE}"
done

python3 summarize_seg_metrics.py "${LOGFILE}"
