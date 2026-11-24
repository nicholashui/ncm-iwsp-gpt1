# NCM-iWSP-GPT1

這個倉庫是 **NCM-iWSP-GPT1**：用程式寫下一概念模型（NCM）的權重，再交給固定的 evaluator 計分。資料與 NCM-GPT1 相同，是 Hugging Face [`lucadiliello/bookcorpusopen`](https://huggingface.co/datasets/lucadiliello/bookcorpusopen) 加上 GPT-2 BPE。它**不是** NCM-GPT1。

**NCM-GPT1** 這個名字只留給 [nicholashui/ncm-gpt1](https://github.com/nicholashui/ncm-gpt1) 裡那個用梯度訓練的 BookCorpus 模型。本倉庫是 [nicholashui/ncm-iwsp-gpt1](https://github.com/nicholashui/ncm-iwsp-gpt1)。系統名稱是 **NCM-iWSP-GPT1**，Python 套件是 `ncm_iwsp_gpt1`。不要把兩個系統當成同一個模型，也不要去改 `ncm-gpt1`、`ncm-iwsp` 或 `HazyResearch/ncm`。

This repository is **NCM-iWSP-GPT1**: an NCM whose weights are written in Python and scored by a fixed evaluator. The data is the GPT-1-scale corpus used by NCM-GPT1 (`lucadiliello/bookcorpusopen`, GPT-2 BPE). The gradient BookCorpus model keeps the name **NCM-GPT1**. This GitHub repository is [nicholashui/ncm-iwsp-gpt1](https://github.com/nicholashui/ncm-iwsp-gpt1). The Python package is `ncm_iwsp_gpt1`.

`python -m ncm_iwsp_gpt1 finetune` is a separate gradient run of this untied generator, starting from the converted v10 init. The writer loop does not call it. See [`docs/finetune_kaggle.md`](docs/finetune_kaggle.md).

The method is interpretable weight-space programming ([arXiv:2611.18114](https://arxiv.org/abs/2611.18114), code [csinva/weight-space-programming-autoresearch](https://github.com/csinva/weight-space-programming-autoresearch)). The network is the NCM hierarchy from [HazyResearch/ncm](https://github.com/HazyResearch/ncm) ([arXiv:2610.14162](https://arxiv.org/html/2610.14162v1)). Level 0 is not a transformer target. The evaluator owns that unigram. The loop shape follows the read-only pilot [nicholashui/ncm-iwsp](https://github.com/nicholashui/ncm-iwsp) (NCM-MiniMind-iWSP-pilot, commit `6b91c9f`) and is vendored here so this repo stands alone. That pilot is MiniMind data with an ASCII/byte pretok. This repo's primary path is BookCorpus and GPT-2 BPE.

iWSP papers mostly hand-write much smaller nets. A stub cannot be assumed to program ~100M parameters in one shot. The runnable config is the size the dry loop actually builds. The target config is the full architecture, and it does instantiate on CPU for the parameter count.

## 不使用反向傳播

Writer 只能用閉式、由 seed 決定的方式覆寫張量。預設的 iteration 0 什麼都不寫，留下隨機初始化，那就是 baseline。

Writer 的迴圈裡沒有 `loss.backward`、Adam、AdamW，也沒有任何梯度優化器。`finetune` 是另外一條指令，只更新已經展開成 untied head 的 generator，見下文。Audit 看的是 writer 程式本身（註解和字串不算），會拒絕：

- `loss.backward`、`torch.optim`、Adam / AdamW、`sklearn` 的 `.fit`
- `train_tokenizer` / `train_generator`、codebook EMA（forward 不會更新碼本）
- `from_pretrained`、`torch.hub`、`torch.load` 這類權重下載
- `write_weights` 收下資料、標籤或隱藏集
- 沒有寫清楚權重是什麼（`MODEL_DESCRIPTION` 至少 80 字）

`pytest` 也會掃這個套件有沒有同樣的訓練呼叫。掃描不是證明。Writer 自己的 Python 仍然可以想辦法作弊，這和 iWSP 論文裡的事後檢查是同一類限制。

Writer 迴圈不花 NCM-GPT1 的 Kaggle T4 額度。`finetune` 會用 T4，但只在 Nicholas 確認路徑之後，由他自己提交一份新的 notebook。本倉庫不代為提交 Kaggle job，也不呼叫付費 API。

## 授權

本倉庫的 harness 是 MIT（Copyright (c) 2026 Nicholas Hui）。`ncm_iwsp_gpt1/models/` 改編自 HazyResearch/ncm commit `c702742`（MIT, Copyright (c) 2025 hermann kumbong），並參考 ncm-iwsp commit `6b91c9f` 的推論版。沒有複製 codebook EMA、k-means restart，也沒有複製官方 `train_tokenizer.py` / `train_generator.py`。詳見 `NOTICE`。

GPT-2 BPE 來自 `tiktoken` 內建的 `gpt2` encoding，不是一份下載的模型權重。

`lucadiliello/bookcorpusopen` **沒有**放進這個 git repo。使用前請自己讀 dataset card。本倉庫不授予語料的使用權，也不對授權做法律判斷。不要把 BookCorpus、packed window、MiniMind 語料提交進 git。

## 參數數量

在 CPU 上實例化模組後用 `count_params` 量到的。碼本算進去（即使 `requires_grad` 是 false）。非嵌入數字會扣掉名稱裡含 `wte`、`wpe`，以及 generator 的 `wle`。這些數字不是訓練結果。

| Config | Context | VQ levels | Codebook | Embed | Generator | Tokenizer | Generator params |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `configs/gpt1_data.yaml` | 32 | 6: `[1, 4, 9, 16, 25, 32]`（87 cells） | 64 | 32 | 2 × 32 × 4 | **3,284,816**（3.285M）；非嵌入 1,674,064 | **37,280**（0.037M）；非嵌入 36,064 |
| `configs/target.yaml` | 512 | 23: `[1, 4, 9, …, 484, 512]`（4307 cells） | 4096 | 384 | 12 × 768 × 12 | **92,276,480**（92.28M）；非嵌入 72,763,136 | **154,666,240**（154.67M）；非嵌入 154,353,664 |

兩份 config 的詞表都是 GPT-2 BPE：50257 個原始 id，`<|endoftext|>` 是 50256，`<|pad|>` 是 50257，嵌入表補到 **50304**。這和 NCM-GPT1 的 `in_vocab_size` 相同。

Tokenizer 的 92,276,480 與 NCM-GPT1 量到的 tokenizer 總數一致。NCM-GPT1 的 generator 是 **101,688,854**（101.69M），因為輸出頭綁在碼本上，而且有 `post_upsample_conv`。這裡的 generator 頭沒有綁定，也沒有那個 conv，所以是 154.67M。這不是 2018 GPT-1 的權重複製品。閉式轉換可以把一份已完成的 gradient checkpoint 填進這兩個模組，見下一節；本倉庫不附帶那份 checkpoint。

POSIX，使用安裝一節建立的 `.venv`：

```bash
.venv/bin/python -m ncm_iwsp_gpt1 count --config configs/gpt1_data.yaml
.venv/bin/python -m ncm_iwsp_gpt1 count --config configs/target.yaml            # estimate, same integers
.venv/bin/python -m ncm_iwsp_gpt1 count --config configs/target.yaml --instantiate
```

Windows 把 `.venv/bin/python` 換成 `.venv\Scripts\python.exe`。

`configs/gpt1_data.yaml` 是乾跑用的保守尺寸。`configs/target.yaml` 才是要對齊的完整 NCM-iWSP-GPT1：12 層 / 768 維 / 12 頭、上下文 512、23 層平方 schedule（`1²..22²` 然後用 512 而不是 `23²`）、每層 codebook 4096。層內的碼一次並行預測。`predict_level0` 維持 false。

## 從已完成的 gradient checkpoint 做閉式轉換

轉換只做兩件事：同名張量原樣複製，以及把綁定碼本的輸出頭展開成這個倉庫的 `Linear(768, 4096, bias=True)`。公式、post-upsample 不能吸收的原因、以及 NicNicNic / Kaggle 指令寫在 [`docs/convert_from_gradient_checkpoint.md`](docs/convert_from_gradient_checkpoint.md)。

Post-upsample conv 在每個 level 的 latent 上混鄰近位置（`chunk + Conv1d(chunk)`，發生在 `wte` 之前）。它不能寫進逐位置的 Linear，這個架構也不會加上那層。那些權重是剩下來用不到的。它們在初始化時是 0（恆等），step 100,000 時可能已經不是 0，所以整段 generator forward 只是近似。同一個 hidden 上，body 和展開後的頭是精確的。

來源檔只讀。`--out` 若落在來源目錄或 `/kaggle/input` 裡，指令會拒絕寫入。不要把結果寫回 `E:\Project\ncm-gpt1-kaggle-100k`。Kaggle 用一份新的 notebook（`kaggle/convert_iwsp.ipynb`），不要去改 v10 training kernel。

NicNicNic 上、之後才跑的指令（這裡不要執行）：

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 convert ^
  --config configs\target.yaml ^
  --tokenizer E:\Project\ncm-gpt1-kaggle-100k\tokenizer.pt ^
  --generator E:\Project\ncm-gpt1-kaggle-100k\generator_step100000.pt ^
  --out E:\Project\ncm-iwsp-gpt1\artifacts\ncm_iwsp_gpt1_from_100k.pt
```

`generator.pt` 和 `latest.pt` 也可以當作 `--generator`。成功之後 tokenizer + generator 是 246,942,720 個參數。`loop` / `eval` 的 `--init-checkpoint` 會在 `write_weights` 之前載入這份檔；stub 的 iteration 0 不會覆寫它。隨機初始化 baseline 仍然是 baseline。Config 必須和轉換時同一份。

The same converter is `python -m ncm_iwsp_gpt1 convert`. Math, the post-upsample caveat, and the Kaggle notebook are in [`docs/convert_from_gradient_checkpoint.md`](docs/convert_from_gradient_checkpoint.md). Source paths are read-only. `--out` inside the export directory or under `/kaggle/input` is refused.

## 從已轉換的 v10 init 做梯度微調

這一步更新的是 **NCM-iWSP-GPT1** 的 generator：untied `Linear(768, 4096)`，沒有 post-upsample conv，tokenizer 載入後凍結。起點是 NicNicNic 上已經轉好的 `E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt`（92,276,480）和 `generator.pt`（154,666,240），格式是 `{model, step, converted_from, ...}`。這份微調留在這個架構裡，不會把權重送回 NCM-GPT1 的 trainer。

把 `E:\Project\ncm-iwsp-gpt1-init` 做成一份新的 Kaggle dataset。不要寫進 `E:\Project\ncm-gpt1-kaggle-100k`，也不要改 kernel `nicholashuihk/ncm-gpt-1-train`。Packed windows 用 v10 已經打好的 `artifacts/packed_windows.pt`（在 export 的 artifacts 樹裡，不一定跟原始 tokenizer 放在一起）。語料是 `lucadiliello/bookcorpusopen`，GPT-2 BPE，context 512。不要重訓 tokenizer。

預設是每個 GPU batch 1、gradient checkpointing、按 level 切開的 block attention、T4 上 fp16。v10 的 tied-head generator 用過 batch 2；這裡是 154.67M，所以從 batch 1 開始。Loss 預設 alpha 0、mid_aux 0，每個 code cell 一樣重。`val_ce_first`、`val_ce_mid`、`val_ce_last` 分開記。v10 後期的 loss 含 alpha 0.5 和 mid_aux 0.5，那個純量不能拿來跟 alpha 0 的曲線比。

`kaggle/finetune_iwsp.ipynb` 的 `NICHOLAS_CONFIRMED` 預設是 False。Nicholas 先看路徑，再自己提交。檔名、schema 和指令在 [`docs/finetune_kaggle.md`](docs/finetune_kaggle.md)。

```bash
python -m ncm_iwsp_gpt1 finetune --config configs/target.yaml --dry-plan
```

`finetune` updates this generator from those two converted files. The tokenizer stays frozen. Writes go to `/kaggle/working` or `--out`, never under `/kaggle/input` or into the init directory. Default loss is alpha 0 and mid_aux 0. Validation records `val_ce_first`, `val_ce_mid`, and `val_ce_last` separately. Per-GPU batch defaults to 1 with checkpointing.

## 文字對話

`python -m ncm_iwsp_gpt1 chat` 把短提示接成一個 window 的延續。它不是 causal LM，也不是聊天機器人，也不是 NCM-GPT1。`predict_level0` 維持 false，level 0 不是 generator 的目標。`prefix_len` 維持 1，那個位置仍是 `<|endoftext|>`，用來對上現有權重。提示詞不放進這個前綴。

預設要一個前綴 level-0 頭（`--prefix-head`）。頭讀真實提示的 `encoder.wte`，抽出一個 level-0，不再把短句重複填滿 context。細碼仍由凍結的 generator 抽出再解碼。stderr 印 `level0_source=prefix_head`。沒有這個 checkpoint 就停下來，不會退回重複打包。對照組要寫明：`--level0-source packed`，或 `--level0-source unigram --packed packed_windows.pt`。

這是延續，不是問答。「How are you?」不是要回「I'm fine」。訓練指令是 `python -m ncm_iwsp_gpt1 train-prefix-level0`，只更新這個頭。`packed_windows.pt` 不在 repo 裡，這裡不啟動長訓練，也不提交 Kaggle。細節在 [`docs/chat.md`](docs/chat.md)。

Kaggle kernel [nicholashuihk/ncm-iwsp-gpt1-prefix-level0](https://www.kaggle.com/code/nicholashuihk/ncm-iwsp-gpt1-prefix-level0) version 2 跑完 20,000 step。前綴頭、指標和 `val_ce_vs_unigram.svg` 在 [`artifacts/kaggle-prefix-level0-v2/`](artifacts/kaggle-prefix-level0-v2/)。v3 generator 權重仍然不在 git 裡。下面的 tokenizer 和 fine-tune 路徑只是例子。

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat --help

.venv\Scripts\python.exe -m ncm_iwsp_gpt1 train-prefix-level0 --config configs\target.yaml --dry-plan

.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --finetune E:\Project\ncm-iwsp-gpt1-finetune-v3\latest.pt ^
  --prefix-head artifacts\kaggle-prefix-level0-v2\prefix_level0.pt ^
  --prompt "The cat sat"
```

`--prompt` prints the decoded window on stdout. Omit it for the stdin loop. The generator prefix stays `<|endoftext|>`. See [`docs/chat.md`](docs/chat.md). The v3 generator is not in git. The prefix head, its metrics, and the val-CE plot are in [`artifacts/kaggle-prefix-level0-v2/`](artifacts/kaggle-prefix-level0-v2/).

## 安裝

在專案根目錄建一個本地 `.venv`。它已經在 `.gitignore` 裡，不要提交。先用這個 venv 的 Python 裝 PyTorch，再裝 `requirements.txt`。下面的 CPU index 是通用文件。有 NVIDIA GPU 的機器請改裝與驅動相符的 CUDA wheel（見 [pytorch.org](https://pytorch.org)），然後同樣裝 `requirements.txt`。

CPU 就夠跑 runnable config 的乾迴圈和煙霧測試。有 CUDA 時，`eval` 和 `loop` 的預設裝置是 GPU（`--device auto`）。`--device cuda` 會在沒有 CUDA 時直接失敗。`--device cpu` 強制用 CPU。

Windows，checkout 在 `e:\Project\ncm-iwsp-gpt1`。沒有 `py` 啟動器時，把 `py -3` 換成 `python`。

```bat
cd /d e:\Project\ncm-iwsp-gpt1
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

POSIX：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
```

`--download` 才需要額外的 `datasets` 套件，預設不安裝、也不會上網抓語料。裝進同一個 venv：`.venv\Scripts\python.exe -m pip install datasets`，或 POSIX 的 `.venv/bin/python -m pip install datasets`。

## 打包 BookCorpus

下載預設關閉，CI 不會抓語料。有本地 jsonl、arrow / parquet，或 Hugging Face cache 時再打包。短段落會接在同一個 token stream 裡，只有最後一個 window 的尾巴會 pad。一個文件先用 SHA-256 整篇分到 `visible_train` / `visible` / `hidden`，writer 看不到 hidden。同一篇裡的段落不會被拆到不同的 split。

Runnable config 的 window 是 32。Target config 的 window 是 NCM-GPT1 用的 512。32 的 cache 不能拿去餵 512 的模型，反過來也不行。打包本身不會建立那 2.5 億參數的模組。

合成段落（不是 BookCorpus）的 `pack` 寫在下一節，和乾跑用同一支 venv Python。

本地 jsonl，欄位 `text` 或 `content`。不要把這個檔提交進 git。Windows 把 `.venv/bin/python` 換成 `.venv\Scripts\python.exe`。

```bash
.venv/bin/python -m ncm_iwsp_gpt1 pack \
  --config configs/target.yaml \
  --jsonl /path/to/bookcorpusopen.jsonl \
  --max-docs 2000 \
  --out cache/bookcorpus.pt
```

本地 arrow、parquet，或 HF datasets cache 目錄：

```bash
.venv/bin/python -m ncm_iwsp_gpt1 pack \
  --config configs/target.yaml \
  --arrow /path/to/bookcorpus.arrow \
  --max-docs 2000 \
  --out cache/bookcorpus.pt

.venv/bin/python -m ncm_iwsp_gpt1 pack \
  --config configs/target.yaml \
  --hf-cache ~/.cache/huggingface/datasets/lucadiliello___bookcorpusopen \
  --max-docs 2000 \
  --out cache/bookcorpus.pt
```

真的要下載時才加 `--download`。這不是預設，CI 沒有這一步。先把 `datasets` 裝進 `.venv`，再打包：

```bash
.venv/bin/python -m pip install datasets
.venv/bin/python -m ncm_iwsp_gpt1 pack \
  --config configs/target.yaml \
  --hf-dataset lucadiliello/bookcorpusopen \
  --download \
  --max-docs 2000 \
  --out cache/bookcorpus.pt
```

Cache 是一個 torch dict：`visible_train`、`visible`、`hidden`、`meta`，格式沿用 `packed_windows.pt` 的想法。`.gitignore` 排除了 `*.pt`、`*.jsonl`、`*.arrow`、`*.parquet`。

## 乾跑迴圈

Stub writer 不呼叫付費 API。Iteration 0 是隨機初始化。之後的 iteration 把 GPT-2 token embedding 寫成正弦、碼本寫成軸對齊的原型、generator 的輸入投影寫成縮放過的單位矩陣。那是閉式，不是結果。

Windows：

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 pack --demo --config configs/gpt1_data.yaml --out cache/demo.pt
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 eval --config configs/gpt1_data.yaml --packed cache/demo.pt --iteration 0 --device cpu
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 loop --config configs/gpt1_data.yaml --packed cache/demo.pt --iters 3 --out runs/demo --device cpu
```

POSIX：

```bash
.venv/bin/python -m ncm_iwsp_gpt1 pack --demo --config configs/gpt1_data.yaml --out cache/demo.pt
.venv/bin/python -m ncm_iwsp_gpt1 eval --config configs/gpt1_data.yaml --packed cache/demo.pt --iteration 0 --device cpu
.venv/bin/python -m ncm_iwsp_gpt1 loop --config configs/gpt1_data.yaml --packed cache/demo.pt --iters 3 --out runs/demo --device cpu
```

有 GPU 時把 `--device cpu` 拿掉，或寫 `--device cuda`。預設 `auto` 會在有 CUDA 時用 CUDA。

每一輪都從 `seed` 重建兩個模組，呼叫 `write_weights(tokenizer, generator, iteration=0)`，審計 writer 檔，再在 **visible** window 上計分。Agent 看的數字是 `results.csv` 裡的 `mean_level_ce`：第 1 層到最後一層的交叉熵，不加權平均。Level 0 不是 generator 的目標。`length_weighted_ce` 是官方 alpha=0 的權重，每個碼位置一樣重。

Hidden split 的交叉熵只寫進 `hidden_metrics.jsonl`。`results.csv` 沒有 hidden 欄。不要把 hidden 檔貼進 agent 的 prompt。

成功的定義不是 NCM-GPT1 的驗證 CE。Audit 要通過，而且在 **hidden** split 上，`mean_level_ce` 要同時嚴格低於隨機初始化、也嚴格低於 evaluator 裡的 unigram。Stub 不保證做得到。乾淨的失敗可以是結果：兩篇權重空間的論文都說，深的殘差表示目前還不是這類 writer 寫得出來的東西。

## 以後接上 Claude 或 Opus

這個倉庫現在不呼叫付費 API。之後可以沿用 csinva `launch_claude.py` 的形狀，但不要把它指去 NCM-GPT1：

1. 把 `ncm_iwsp_gpt1/writers/stub.py` 複製到 run 目錄。乾跑已經會把 writer 快照存在 metrics 旁邊。
2. Agent 只能改那個檔案：`write_weights`、`MODEL_SHORTHAND`、`MODEL_DESCRIPTION`（至少 80 字，要說明碼本和 embedding 是什麼）。
3. 不可以改 `evaluate.py`、`audit.py`、`packing.py`，也不可以讀 `hidden_metrics.jsonl`。
4. 給 agent 的數字只有 `results.csv` 的 `mean_level_ce`。
5. `write_weights` 開頭保留 `torch.manual_seed`。同一個 iteration 呼叫兩次，張量必須一樣。Audit 會檢查這件事。
6. 用迴圈、閉式、小表來**產生**張量。不要貼上幾百萬個浮點數，也不要對那些表做一步梯度。
7. 論文裡那種 200 字的 describe-and-rebuild 審計是之後由人來跑的，這裡沒有自動化。

## 測試

煙霧測試就是安裝一節裡的 `pytest`：Windows 用 `.venv\Scripts\python.exe -m pytest`，POSIX 用 `.venv/bin/python -m pytest`。GitHub Actions 在乾淨的 runner 上直接裝 CPU 版 PyTorch，不使用這份本地 `.venv`。

涵蓋：GPT-2 BPE 把類似 BookCorpus 的短文打成 window（含長度 512）、文件級 hash split 的隔離、兩份 config 載入、上面的參數數量（runnable 與 target 都會在 CPU 上實例化）、audit 拒絕 backward / Adam / AdamW / 權重下載 / hidden 洩漏、合成 packed window 上的 forward，以及 prefix level-0 頭的形狀、一步訓練和 chat 接線。GitHub Actions 用 CPU 版 PyTorch 跑同一套。測試只用合成句子，不包含真的 BookCorpus。

## 目錄

| Path | Role |
| --- | --- |
| `configs/gpt1_data.yaml` | BookCorpus + GPT-2 BPE，CPU 乾跑尺寸，context 32 |
| `configs/target.yaml` | 完整 NCM-iWSP-GPT1：12/768/12，23 層，codebook 4096，context 512 |
| `ncm_iwsp_gpt1/pretok.py` | 凍結的 GPT-2 BPE，pad 到 50304 |
| `ncm_iwsp_gpt1/packing.py` | 文件 hash split，段落打進固定 window |
| `ncm_iwsp_gpt1/models/` | VQ-VAE 與 untied generator；finetune 會對 generator 開 checkpoint |
| `ncm_iwsp_gpt1/writers/stub.py` | Writer 應該只改這個檔的複本 |
| `ncm_iwsp_gpt1/evaluate.py` | 固定的指標 |
| `ncm_iwsp_gpt1/prefix_level0.py` | 前綴 level-0 頭；只訓練這個頭，tokenizer 與 generator 凍結 |
| `ncm_iwsp_gpt1/chat.py` | 用真實提示抽 level 0；packed 與 unigram 是明示對照 |
| `ncm_iwsp_gpt1/audit.py` | 訓練、下載、hidden 洩漏 |
| `ncm_iwsp_gpt1/loop.py` | propose / write / eval / audit |
| `ncm_iwsp_gpt1/cli.py` | `pack`、`count`、`eval`、`loop`、`convert`、`finetune`、`chat`、`train-prefix-level0` |
| `ncm_iwsp_gpt1/convert.py` | 閉式複製 tokenizer / body，展開 tied head，丟棄 post-upsample |
| `ncm_iwsp_gpt1/finetune.py` | 從已轉換 init 梯度更新 untied generator；tokenizer 凍結 |
| `docs/convert_from_gradient_checkpoint.md` | 公式、caveat、NicNicNic 與 Kaggle 指令 |
| `docs/finetune_kaggle.md` | 上傳 init、接上 `artifacts/packed_windows.pt`、T4 預設、loss 怎麼比 |
| `docs/chat.md` | 前綴頭延續、必要的 checkpoint、以及 packed / unigram 對照 |
| `kaggle/train_prefix_level0.ipynb` | 前綴頭的 Kaggle notebook。`NICHOLAS_CONFIRMED` 維持 False。子程序帶 `PYTHONPATH`，cwd 是 repo |
| `patches/time-budget-max-hours.patch` | Kaggle 資料集在 `f35e352` 上加的時間預算。本倉庫的訓練程式沒有套用 |
| `artifacts/kaggle-prefix-level0-v2/` | version 2 的 `prefix_level0.pt`、指標，和 `val_ce_vs_unigram.svg` |
| `kaggle/convert_iwsp.py` | 新的 Kaggle notebook 用的轉換腳本，寫入 `/kaggle/working` |
| `kaggle/convert_iwsp.ipynb` | 同一支腳本的 notebook 版，不提交 Kaggle job |
| `kaggle/finetune_iwsp.py` | 新的微調腳本，只寫 `/kaggle/working`，需要 `--confirm` |
| `kaggle/finetune_iwsp.ipynb` | 同一支微調腳本的 notebook，`NICHOLAS_CONFIRMED` 預設關閉 |

## 還沒解決的風險

- 沒有公開證據說 iWSP 寫得出深的殘差碼階層。這個 harness 可以記下失敗。
- Audit 是原始碼掃描加上確定性檢查。它讀不到被藏在別的模組裡的訓練步驟。
- 這裡的 generator 是 154.67M，不是 NCM-GPT1 的 101.69M。差在沒有綁定碼本的輸出頭，也沒有 `post_upsample_conv`。Tokenizer 的參數總數則與 NCM-GPT1 量到的 92.28M 相同。閉式轉換會把 tied head 展開成那 154.67M，並丟棄 post-upsample。整段 forward 因此只在那些 conv 仍為 0 時與來源一致。
- Target 的一次 forward（序列長度約 4307）比數參數重得多。CI 會實例化它來核對參數數量，但不會用它跑語料。
- Level-0 和各層 unigram 是 evaluator 在凍結的 visible-train window 上做的 add-one 計數，不是權重。Visible 很少的時候這個 baseline 會很吵。
- 這次環境沒有下載或打包真正的 `lucadiliello/bookcorpusopen`。整本書可能很大；先用 `--max-docs`。一行一本書時，hash 仍是整篇文件，段落只用於打包。
- 不要把 NCM-GPT1 的 Kaggle T4 額度拿來跑這個搜尋，除非 Nicholas 明確把額度撥過來。
