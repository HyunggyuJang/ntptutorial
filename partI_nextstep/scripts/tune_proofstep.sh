REPO_DIR=".."
TRAIN_FILE=${REPO_DIR}/data/processed/proofstep-train.jsonl
VALID_FILE=${REPO_DIR}/data/processed/proofstep-val.jsonl
MODEL=EleutherAI/pythia-2.8b-deduped
CONFIG=${REPO_DIR}/scripts/ds_config.json

OUTDIR=${REPO_DIR}/model/${MODEL}

deepspeed --include localhost:0,1,2,3,4,5,6,7  ${REPO_DIR}/ntp_python/tune.py \
    --deepspeed ${CONFIG} \
    --model_name_or_path ${MODEL} \
    --train_data_path ${TRAIN_FILE} \
    --valid_data_path ${VALID_FILE} \
    --fp16 \
    --output_dir ${OUTDIR} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --load_best_model_at_end 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir "$OUTDIR" \
    --report_to="tensorboard"
