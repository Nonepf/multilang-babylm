#!/bin/bash
# Fix vocabulary: Qwen2.5 actual vocab is 151665, but our models were trained with 151643.
# 1. Download original Qwen2.5 tokenizer
# 2. Restore tokenizer files to every checkpoint
# 3. Resize model embeddings (preserves trained weights, adds 22 random-init rows)
set -e

DIR="${1:-checkpoints}"
echo "=== Step 1: Download original Qwen2.5 tokenizer ==="

VOCAB_TMP_DIR="/tmp/qwen25_tok_$$"
python3 -c "
import os, warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
os.makedirs('$VOCAB_TMP_DIR', exist_ok=True)
t.save_pretrained('$VOCAB_TMP_DIR')
print(f'Downloaded: vocab_size={t.vocab_size}, len={len(t)}, eos={t.eos_token_id}')
# Print saved files
for f in sorted(os.listdir('$VOCAB_TMP_DIR')):
    print(f'  {f}')
"

echo ""
echo "=== Step 2: Restore tokenizer + resize embedding for each checkpoint ==="

python3 -c "
import os, sys, warnings, glob, shutil
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from transformers import GPT2LMHeadModel, AutoTokenizer

SRC = '$VOCAB_TMP_DIR'
TARGET_VOCAB = 151665
EOS_ID = 151643
count = 0

for root, dirs, files in os.walk('$DIR'):
    if 'model.safetensors' not in files:
        continue

    model_path = root
    print(f'\n--- {model_path} ---')

    # 2a. Restore original tokenizer files
    for f in glob.glob(f'{SRC}/*'):
        dst = os.path.join(model_path, os.path.basename(f))
        shutil.copy2(f, dst)
        print(f'  restored: {os.path.basename(f)}')

    # 2b. Load model + resize embedding
    model = GPT2LMHeadModel.from_pretrained(model_path)
    old = model.config.vocab_size
    print(f'  model vocab: {old}')

    if old < TARGET_VOCAB:
        model.resize_token_embeddings(TARGET_VOCAB)
        model.config.eos_token_id = EOS_ID
        model.save_pretrained(model_path)
        print(f'  resized: {old} -> {TARGET_VOCAB}')
    else:
        print(f'  already correct size, skipping')

    count += 1

print(f'\nProcessed {count} checkpoints.')
"

# Cleanup
rm -rf "$VOCAB_TMP_DIR"

echo ""
echo "=== Step 3: Verify ==="
python3 -c "
import os, warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from transformers import AutoTokenizer, GPT2LMHeadModel

ok = 0; bad = 0
for root, dirs, files in os.walk('$DIR'):
    if 'model.safetensors' not in files:
        continue
    m = GPT2LMHeadModel.from_pretrained(root)
    t = AutoTokenizer.from_pretrained(root)
    vs = m.config.vocab_size
    eos_m = m.config.eos_token_id
    eos_t = t.eos_token_id
    if vs == 151665 and eos_t == 151643 and eos_m == 151643:
        ok += 1
    else:
        print(f'  MISMATCH: {root}  vocab={vs}  eos_model={eos_m}  eos_tok={eos_t}')
        bad += 1
print(f'\nOK: {ok}, MISMATCH: {bad}')
" 2>/dev/null

echo ""
echo "Done."
