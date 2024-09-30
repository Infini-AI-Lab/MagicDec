ROOT_DIR="./Data/Ruler/benchmark_root"

MODEL_NAME=meta-llama/Meta-Llama-3.1-8B
MODEL_PATH="/scratch/checkpoints/meta-llama/Meta-Llama-3.1-8B"
MODEL_TEMPLATE_TYPE="meta-chat"
MODEL_FRAMEWORK="hf"

TOKENIZER_TYPE="hf"
TOKENIZER_PATH=$MODEL_PATH

rm -rf ${ROOT_DIR}/${MODEL_NAME} 
DATA_DIR="${ROOT_DIR}/${MODEL_NAME}/data"
PRED_DIR="${ROOT_DIR}/${MODEL_NAME}/pred"

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
NUM_SAMPLES=100
MAX_SEQ_LENGTH=$1
BENCHMARK=synthetic
STOP_WORDS=""
REMOVE_NEWLINE_TAB=""

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

for TASK in "${TASKS[@]}"; do
python Data/Ruler/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type ${MODEL_TEMPLATE_TYPE} \
    --num_samples ${NUM_SAMPLES} \
    ${REMOVE_NEWLINE_TAB}
done