# AHEAD：節點度數校正與動態圖異常偵測

本專案以 [AHEAD](https://doi.org/10.1145/3770855.3817610) 的 GraphMixer 模型為基礎，探討節點歷史鄰居數量對異常分數校正的影響。研究比較六種校正方法，使用 Digg、Bitcoin-Alpha、Wikipedia 與 MOOC 四個資料集，以及兩種模型選擇策略和五個隨機種子。

在已用於探索的實驗區間 D 中，相較於原始校正方法（Original），Fixed 1.1 在兩種模型選擇策略下的平均 AUROC 均於 Digg 與 Wikipedia 較高，在 Bitcoin-Alpha 與 MOOC 較低。結果顯示固定度數校正倍率的適用性因資料集而異，並非獨立測試集的驗證結果。

## 快速開始

下載專案並在專案根目錄執行：

```bash
git clone https://github.com/tyTaco/AHEAD-Degree-Calibration.git
cd AHEAD-Degree-Calibration
python3 -B scripts/show_results.py
```

Windows 可使用 `python` 或 `py -3` 取代 `python3`。結果展示工具只需要 Python 標準函式庫，無須下載資料集、安裝 PyTorch 或使用 GPU。程式讀取已保存的各次實驗指標，計算 144 組方法平均值與樣本標準差，以及 120 組 Fixed 1.1 相對其他方法的配對差值，並核對已保存的配對摘要。

## 研究方法

令 `s` 為原始異常分數，`d` 為來源節點取樣所得的有效歷史鄰居數量（sampled historical degree），並非完整圖中不重複的鄰居數量。

| 方法 | 校正方式 | 實作位置 |
|---|---|---|
| Original | d < 3 時乘以 0.9，否則維持原分數 | 模型內部 |
| None | 保留原始分數 s | 模型內部 |
| Continuous | s × (1 − 0.1 exp(−d/3)) | 模型內部 |
| Fixed 1.1 | d < 3 時乘以 1.1，否則維持原分數 | 後處理 |
| C-Pairwise | 依較早 C 區間的標籤及 AUROC，從 1、0.9、1.1 選擇倍率 | 後處理 |
| C-Guarded | 在倍率選擇時加入時間區塊 Bootstrap 條件 | 後處理 |

Original 為原作者的校正方式。None 與 Continuous 透過 `AHEAD_DEGREE_CALIB_MODE` 設定為 `none`、`continuous`，原始模式為 `original`。Fixed 1.1 與兩種 C 方法使用已保存的原始分數進行後處理；兩種 C 方法均使用歷史標籤，並非無監督校正。

## 資料集

| 資料集 | 資料來源、格式與標籤 | 原論文資料集 |
|---|---|---|
| Digg | [KONECT Digg reply 頁及 TSV 下載](https://www.konect.cc/networks/munmun_digg_reply/)；時間性回覆網路、TSV 壓縮檔。研究版合成替換邊為正類。 | 是 |
| Bitcoin-Alpha | [SNAP 說明](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)、[CSV gzip](https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz)，欄位 source、target、rating、time；本研究負信任評分為正類代理，非已證實詐欺。 | 是；本研究正類數與論文不同 |
| Wikipedia | [JODIE 資料頁](https://snap.stanford.edu/jodie/)、[CSV](https://snap.stanford.edu/jodie/wikipedia.csv)，使用者、項目、時間、state label 與邊特徵；封禁相關的最後互動是正類代理。 | 是 |
| MOOC | [JODIE 資料頁](https://snap.stanford.edu/jodie/)、[CSV](https://snap.stanford.edu/jodie/mooc.csv)，相同 JODIE 格式；退課相關的最後互動是正類代理。 | **否，不在原論文六個實驗資料集內** |

MOOC 是額外納入的時間二分圖資料集，可用於比較教育領域與其他標籤定義下的校正效果。[SNAP 資料說明](https://snap.stanford.edu/data/act-mooc.html)記載 7,047 位使用者、97 種活動、411,749 筆互動、4,066 個正標籤及每事件 4 維特徵。退課相關的最後互動在本研究中作為異常評分的代理標籤，不代表欺詐或一般異常。

## 實驗設定

| 參數 | 設定 |
|---|---|
| Random seeds | 0–4 |
| Epochs | 100 |
| Batch size | 200 |
| Number of neighbors | 20 |
| Neighbor sampling | recent |
| Time gap | 2000 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| val_ratio（程式參數） | 0.05 |
| test_ratio（程式參數） | 0.5 |

每個隨機種子使用相同訓練軌跡比較兩種模型選擇策略：Validation AUROC 選擇驗證集 AUROC 最高的 checkpoint；Train Loss 選擇訓練損失最低的 checkpoint。後者不以驗證標籤選模，但共用訓練流程仍計算 Validation AUROC。

## 實驗結果

以下為探索區間 D 中 Fixed 1.1 減去 Original 的五個隨機種子平均 AUROC 差值，正值表示改善，負值表示下降。兩欄代表不同模型選擇策略。

| 資料集 | Validation AUROC | Train Loss |
|---|---:|---:|
| Digg | +0.063857 | +0.126480 |
| Bitcoin-Alpha | −0.043053 | −0.020316 |
| Wikipedia | +0.034676 | +0.040315 |
| MOOC | −0.015060 | −0.002207 |

兩種模型選擇策略呈現相同的變化方向。完整六種方法的 AUROC、Average Precision（AP）與梯形積分 PR-AUC 可透過快速開始的指令取得；AP 與梯形 PR-AUC 為不同指標。

### 度數與標籤分布

下表為探索性開發區間 D 的事件分布。比例為各組正類事件數除以該組事件總數；兩種模型選擇策略與五個隨機種子共用事件分組。

| 資料集、D 列位置（左閉右開） | 低 d<3 事件／正類／比例 | 高 d≥3 事件／正類／比例 |
|---|---:|---:|
| Digg `[52576,61339)` | 2,885／245／8.4922% | 5,878／148／2.5179% |
| Bitcoin-Alpha `[14532,16940)` | 471／12／2.5478% | 1,937／213／10.9964% |
| Wikipedia `[94484,110232)` | 1,288／13／1.0093% | 14,460／8／0.0553% |
| MOOC `[247049,288224)` | 865／0／0% | 40,310／389／0.9650% |

MOOC 在此 D 區間的低度數組沒有正類事件。對低度數事件套用 1.1 倍率後，部分低度數負類與高度數正類的相對排序發生不利變化。Fixed 1.1 相對 None 的平均 AUROC 差值分別為 −0.004885（Validation AUROC）及 −0.000789（Train Loss）。此結果反映既有分數的排序變化，而非因果關係或其他資料集的普遍規律。

結果來源：[校正比較](src/research_runs/fixed_degree_up_exploration_20261123/report.json)、[度數分布](src/research_runs/unlabeled_median_exploration_20261123/report.json)。

## 專案結構

| 路徑 | 用途 |
|---|---|
| `scripts/show_results.py` | 實驗統計摘要 |
| `scripts/verify_scores.py` | 評估指標與檔案一致性驗證 |
| `src/models/AHEADGraphMixer.py` | 模型與度數校正 |
| `src/run_fixed_degree_up_development.py` | Fixed 1.1 比較 |
| `src/retrospective_pairwise_development.py` | C-Pairwise 倍率選擇 |
| `src/conservative_pairwise_development.py` | C-Guarded 倍率選擇 |
| `src/train_shared_research.py` | Digg、Bitcoin-Alpha、Wikipedia 訓練 |
| `src/train_segmented_research.py` | MOOC 訓練 |
| `src/prepare_reproducible.py`、`src/preprocessing_profiles.json` | Historical／Custom 前處理及 manifest 驗證 |
| `src/train_preprocessed_research.py` | 資料驗證及隔離訓練入口 |
| `src/score_preprocessed_research.py`、`src/evaluate_preprocessed_calibration.py` | checkpoint 出分、六方法評估與重讀 |
| `src/summarize_calibration_results.py` | 新 run 與歷史結果統計 |
| `src/prepare_jodie_research.py` | JODIE 格式資料轉換 |
| `src/research_runs/` | 實驗結果與版本、模型選擇紀錄 |
| `tests/` | 結果展示與驗證工具測試 |
| `src/run_degree_development.py`、`src/run_segmented_development.py` | 開發區間推論與結果輸出 |
| `src/evaluate_shared_research.py` | 測試集評估與結果輸出，與 Level 2 指標驗證不同 |
| `src/tests/` | 校正方法與補充研究測試 |
| `src/saved_results/degree_ablation/digg/` | 早期 Digg 消融實驗結果，須留意舊 Test 選模限制 |

`src/` 亦包含標籤來源查核及補充校正研究程式（如 `audit_*`、`prototype` 與 `development` 相關檔案），部分為主要方法的相依程式。

`src/README.md` 保留 AHEAD 原始專案的說明文件，其中的訓練指令不適用於本專案封裝。請以本頁提供的執行指令為準。

## 實驗環境與安裝

原實驗使用以下環境：

| 項目 | 版本與規格 |
|---|---|
| OS | Ubuntu 24.04.5 LTS，Linux |
| CPU／RAM | Intel Core i7-13700K，16 核、24 執行緒；記憶體顯示約 31 GiB |
| GPU／顯存 | NVIDIA GeForce RTX 3090，24,576 MiB |
| NVIDIA driver | 595.99.02；`nvidia-smi` 顯示 driver 支援 CUDA 13.2，這不是 PyTorch build 版本 |
| Python／PyTorch | Python 3.12.3；PyTorch 2.6.0+cu126，PyTorch CUDA build **12.6** |
| 套件 | NumPy 1.26.4、SciPy 1.13.1、pandas 2.2.3、scikit-learn 1.5.2；其餘參考 `requirements.txt` |

如需執行模型或評估指標驗證，請在 Linux 專案根目錄建立虛擬環境。以下安裝指令依據本研究使用的套件版本整理；全新環境的安裝與完整執行仍待驗證：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 依 https://pytorch.org/get-started/locally/ 選擇適合平台的 torch；例如 Linux 的 CUDA 12.6 build：
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

CPU-only 可改用下列指令**取代**上方的 CUDA PyTorch 安裝命令，再執行相同的 `pip install -r requirements.txt`；新環境安裝與 CPU-only build 本身尚未驗證，歷史 seal 的精確 PyTorch 版本也可能不同：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

其他平台請參考 [PyTorch 安裝指引](https://pytorch.org/get-started/locally/)及[歷史版本命令](https://pytorch.org/get-started/previous-versions/)。NVIDIA 驅動顯示的 CUDA 支援版本與 PyTorch CUDA build、CUDA toolkit 版本並不相同。環境相容性與完整重現範圍見下方說明。

## 資料準備與前處理

候選版本不含原始／處理後資料、checkpoint 或逐事件 scores。下載前請查核來源使用及再散布條件；可以下載不代表允許重新散布。本節指令在**專案根目錄**執行，`work/` 為本地工作資料，不納入版本控制。

| 資料集 | 公開原始資料 | 本研究標籤與狀態 |
|---|---|---|
| Wikipedia | [SNAP JODIE CSV](https://snap.stanford.edu/jodie/wikipedia.csv) | state label 封禁相關最後互動代理；公開來源至歷史三輸入 PASS-BITWISE，歷史原轉換程式仍未知 |
| Bitcoin-Alpha | [SNAP CSV gzip](https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz) | 負信任評分代理，不等於已證實欺詐；公開來源至歷史三輸入 PASS-BITWISE，重建候選不證明歷史原程式 |
| MOOC | [SNAP JODIE CSV](https://snap.stanford.edu/jodie/mooc.csv) | 退課相關最後互動代理；公開來源至歷史三輸入 PASS-BITWISE，並非 AHEAD 論文的六個資料集之一 |
| Digg | [KONECT reply network](https://konect.cc/networks/munmun_digg_reply/) | 歷史資料的合成異常注入完整規則未知；前處理 **BLOCKED**，兩模式均拒絕生成 |

公開檔案可能更新；Historical 模式會嚴格核對來源及輸出 SHA-256。上列三個 PASS-BITWISE 是原研究環境的實際驗證，不保證其他平台重新生成 NPY 時逐位一致，也不表示找回全部歷史原始程式。

下載範例（Wikipedia 約 560 MB、MOOC 約 41 MB）：

```bash
mkdir -p work/raw work/prepared work/runs
curl -fL --output work/raw/wikipedia.csv https://snap.stanford.edu/jodie/wikipedia.csv
curl -fL --output work/raw/mooc.csv https://snap.stanford.edu/jodie/mooc.csv
curl -fL --output work/raw/soc-sign-bitcoinalpha.csv.gz https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz
```

選擇一組資料及不存在的輸出目錄，再執行 Historical 前處理：

```bash
DATASET=mooc
SOURCE="$PWD/work/raw/mooc.csv"
PREP="$PWD/work/prepared/mooc_historical"
python -B src/prepare_reproducible.py --dataset "$DATASET" --mode historical \
  --source "$SOURCE" --destination "$PREP"
python -B src/prepare_reproducible.py --dataset "$DATASET" --mode historical \
  --source "$SOURCE" --destination "$PREP" --verify-only
```

Wikipedia 使用 `DATASET=wikipedia` 和 `SOURCE="$PWD/work/raw/wikipedia.csv"`；Bitcoin-Alpha 使用 `DATASET=bitcoinalpha` 和 `SOURCE="$PWD/work/raw/soc-sign-bitcoinalpha.csv.gz"`，並各自更換尚不存在的 `PREP`。輸出為 `ml_<dataset>.csv`、`ml_<dataset>.npy`、`ml_<dataset>_node.npy` 與 `reproducibility_manifest.json`。Historical 模式不接受 C/F override；若 SHA 或結構不符則停止，不會放寬要求。

自己的合法輸入可使用 `--mode custom`；Wiki/MOOC 應是 JODIE 格式 CSV，Bitcoin-Alpha 為 SNAP 四欄格式。Custom 可選 `--edge-order C|F --node-order C|F`，輸出 `CUSTOM-VALID` 表示符合自身 manifest，**不是**歷史資料重現。不同 mode 應使用獨立目錄，不能覆蓋既有產物。

## 新實驗：訓練、出分與六方法

以下沿用前節 `DATASET`、`SOURCE`、`PREP`；每個 run 需要不同的 `RUN_ID`，工作區不可在原始 `src/` 下。GPU 需要自行安裝相容 PyTorch／驅動；原研究環境使用 GPU，但新整合入口的 GPU 執行尚未實測。以下以 CPU 為例：

```bash
RUN_ID=my_mooc_run_001
MANIFEST=$(python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PREP/reproducibility_manifest.json")
AHEAD_INTEGRATION_TRAINING_CONFIRM=1 python -B src/train_preprocessed_research.py \
  --dataset "$DATASET" --data-mode historical --source "$SOURCE" --data-dir "$PREP" \
  --manifest-sha256 "$MANIFEST" --experiment-mode historical --device cpu \
  --run-id "$RUN_ID" --workspace-parent "$PWD/work/runs" --execute
```

先透過可信紀錄核對 manifest hash；範例只是方便設定命令變數，單獨計算現有檔案 hash 不構成來源驗證。訓練入口會再次檢查來源、manifest、profile、三個輸入及 schema，將檔案複製至 `work/runs/$RUN_ID/project/`，保留原研究目錄。可將 `--execute` 改為 `--verify-only`（唯讀）或 `--dry-run`（複製並檢查，不訓練）。正式訓練耗時及記憶體需求依資料量與硬體而異。

Historical experiment 從公開的 `src/research_runs/` provenance 讀取原研究參數：5 個 seeds（0–4）、100 epochs、batch 200、recent 20、time gap 2000、Adam 0.0001；`--device cpu` 與歷史 GPU 不同時會記錄差異。Wikipedia／Bitcoin-Alpha 自動選 shared；MOOC 選 segmented。兩種 checkpoint selection 共用同一條訓練軌跡，Validation AUROC 使用 B labels，Train Loss 不以 B labels 決定但共用流程仍計算 B AUROC。

若使用自訂資料及測試參數，改用 `--data-mode custom --experiment-mode custom`，例如加 `--num-runs 1 --num-epochs 1 --batch-size 10`；可用 `--device cpu` 或 `--device cuda:0`、`--learning-rate` 等其他選項詳見 `--help`。資料 mode 與訓練 mode 分開記錄。若改變 val/test ratio、neighbor 數或 sampler，現有出分協定可能拒絕；目前出分要求 0.05／0.5、recent、20 neighbors。

訓練完成後，同一 checkpoint 的六方法共用一次 C/D raw scores 與 sampled degrees，不需訓練六次。下例對 seed 0、兩種選模、全部方法出分與評估；CPU 推論會寫入這個新 run，請先確認工作負載：

```bash
PROJECT="$PWD/work/runs/$RUN_ID/project"
RESULT="$PWD/work/runs/$RUN_ID/calibration_result.json"
python -B src/score_preprocessed_research.py --project-root "$PROJECT" \
  --source "$SOURCE" --run-id "$RUN_ID" --dataset "$DATASET" \
  --seeds 0 --policies val_auroc train_loss --methods all --mode historical \
  --output "$RESULT" --confirm-scores
```

可改 `--seeds 0 1 2 3 4`；單一或多個方法使用如 `--methods none original fixed_1_1`。若要切換已支援校正參數，使用 `--mode custom` 搭配 `--fixed-multiplier`、`--block-size`、`--replicates`、`--bootstrap-seed`；其他固定公式不因參數切換而更動。選擇的 seed/policy 請在第一次出分時一次列齊：已有 `degree_development/` 時會拒絕覆寫或追加。Scores 與事件 ID 位於工作區 `project/research_runs/$RUN_ID/degree_development/`，輸出包含各 seed／policy 的三項指標。

重讀封存並以另一組方法評估（使用者可在確認 seal 來源後執行）：

```bash
SEAL="$PROJECT/research_runs/$RUN_ID/degree_development/development_complete.json"
SEAL_SHA=$(python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$SEAL")
python -B src/evaluate_preprocessed_calibration.py --project-root "$PROJECT" \
  --source "$SOURCE" --run-id "$RUN_ID" --dataset "$DATASET" --seeds 0 \
  --policies val_auroc train_loss --methods none original --mode historical \
  --development-seal-sha256 "$SEAL_SHA" --require-integration \
  --output "$PWD/work/runs/$RUN_ID/rechecked.json"
```

新 scores 在實際推論 batch 保存 event ID；舊 scores 缺 event ID，只有顯式 `--allow-legacy-event-alignment` 才會以 timestamp 限制核對。重算 C 方法會用 C 標籤；實際標籤發布時間未知，D 已探索，不是獨立 Test。

## 結果統計

對已保存的 per-run JSON，執行目錄為專案根目錄：

```bash
python -B src/summarize_calibration_results.py --input "$RESULT" \
  --output "$PWD/work/runs/$RUN_ID/statistics.json"
```

可加 `--datasets mooc --seeds 0 --policies val_auroc --methods none original fixed_1_1` 選擇子集，或在 `--input` 後提供多個不同資料集的結果。輸出含 mean、sample SD（單 seed 為 JSON null／終端 N/A）、逐 seed 配對差與所選基準；原研究方向為 Fixed 1.1 減對照，以及 C-Pairwise／C-Guarded 減 None／Original／Continuous。配對要求同 dataset、seed、policy、checkpoint、scores 及可比較的封存設定。缺基準或 seed 不符會標明，不能補值；同 dataset 的多 seed 必須來自同一封存 run，無法安全合併不同 run。工具輸出 JSON，沒有 CSV 模式，也不重新讀取 scores 或驗證 checkpoint。

`python3 -B scripts/show_results.py` 可直接重算隨附的歷史逐 seed 結果，不需模型或資料；完整歷史結果也能作為 `--input src/research_runs/fixed_degree_up_exploration_20261123/report.json`。

## 200 事件 CPU smoke test

以下腳本建立自己的臨時資料與新 run；執行會**真實訓練 1 epoch 並做 CPU 推論**，不是唯讀命令。原研究環境曾用相同 200 事件、seed 0、batch 10 驗證 shared 與 segmented，候選副本本階段未重跑訓練；乾淨外部環境尚未驗證。請確認資源與使用政策後自行執行：

```bash
mkdir -p work/runs work/prepared work/raw
SMOKE=$(mktemp -d "$PWD/work/raw/smoke-XXXXXXXX")
python -B - "$SMOKE/synthetic.csv" <<'PY'
import csv,sys
with open(sys.argv[1],'x',newline='') as f:
    w=csv.writer(f); w.writerow(['user','item','timestamp','state_label','features'])
    for i in range(200):
        w.writerow([f'user{i%10}',f'item{(i+i//10)%5}',i+1,i%2,0.1+(i%7)*0.01,0.2,0.3,0.4])
PY
DATASET=mooc
SOURCE="$SMOKE/synthetic.csv"
PREP="$SMOKE/prepared"
RUN_ID="smoke_$(basename "$SMOKE")"
python -B src/prepare_reproducible.py --dataset "$DATASET" --mode custom --source "$SOURCE" --destination "$PREP"
python -B src/prepare_reproducible.py --dataset "$DATASET" --mode custom --source "$SOURCE" --destination "$PREP" --verify-only
MANIFEST=$(python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PREP/reproducibility_manifest.json")
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
AHEAD_INTEGRATION_TRAINING_CONFIRM=1 python -B src/train_preprocessed_research.py \
  --dataset "$DATASET" --data-mode custom --source "$SOURCE" --data-dir "$PREP" \
  --manifest-sha256 "$MANIFEST" --experiment-mode custom --device cpu \
  --run-id "$RUN_ID" --workspace-parent "$PWD/work/runs" --num-runs 1 --num-epochs 1 --batch-size 10 --execute
PROJECT="$PWD/work/runs/$RUN_ID/project"
RESULT="$PWD/work/runs/$RUN_ID/calibration_result.json"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B src/score_preprocessed_research.py --project-root "$PROJECT" --source "$SOURCE" \
  --run-id "$RUN_ID" --dataset "$DATASET" --seeds 0 --policies val_auroc train_loss \
  --methods all --mode custom --block-size 5 --replicates 20 --bootstrap-seed 20261121 \
  --output "$RESULT" --confirm-scores
python -B src/summarize_calibration_results.py --input "$RESULT" --output "$PWD/work/runs/$RUN_ID/statistics.json"
```

改成 `DATASET=wikipedia`、使用新的 `SMOKE` 和 `RUN_ID` 可測 shared；這是合成資料，不是實際 Wikipedia 結果。C-Guarded 不一定會進入 bootstrap：只有觀察到候選在 C 的改善時才執行抽樣。結果僅驗證串接，不代表方法改善。

## 驗證與研究限制

| 層級 | 狀態 |
|---|---|
| 保存指標統計 | 隨附 JSON；`scripts/show_results.py` 可執行 |
| 歷史逐筆 scores 指標重算 | `scripts/verify_scores.py --assets-root <外部原研究 src>`；需要未隨附的歷史模型輸入、checkpoint、scores，並核對程式及封存 hash；沿用 C 選擇、不重跑 bootstrap |
| 公開資料前處理 | Wikipedia、Bitcoin-Alpha、MOOC 在原研究環境達 PASS-BITWISE；Digg BLOCKED |
| 合成資料 CPU 端對端 | 原研究環境中 shared 與 segmented 各 200 事件、1 seed、1 epoch 通過；候選副本與乾淨外部環境未重訓 |
| 真實公開資料的完整新訓練／推論 | 尚未端對端驗證 |
| 新整合流程的 GPU 與乾淨外部環境 | 尚未驗證 |

歷史 scores 缺 event ID，早期 checkpoint 有 Test 選模風險；C 使用歷史標籤而其發布時間不明；D 已探索。MOOC 的退課、Wikipedia 的封禁與 Alpha 負評均是代理標籤，不能直接當成已確認欺詐；歷史輸入的可重建不消除資料或標籤洩漏風險。固定 1.1 在資料集間有改善及退步，不可稱普遍改良。

CPU 測試（專案根目錄；以下不訓練真實模型）：

```bash
python -B scripts/show_results.py
python -B -m unittest discover -s tests -p 'test_statistics_entry.py' -v
(cd src && python -B -m unittest discover -s tests -p 'test_prepare_reproducible.py' -v)
(cd src && python -B -m unittest discover -s tests -p 'test_train_preprocessed_research.py' -v)
(cd src && python -B -m unittest discover -s tests -p 'test_score_preprocessed_research.py' -v)
(cd src && python -B -m unittest discover -s tests -p 'test_evaluate_preprocessed_calibration.py' -v)
```


## 參考資料與授權

本專案基於 [AHEAD 論文](https://doi.org/10.1145/3770855.3817610) 與 [原作者程式](https://github.com/YayaLee666/AHEAD)，參考版本為 `b9d5a805143aa280a665a11d6bdd53d211943888`。本研究聚焦 GraphMixer 的度數校正，不涵蓋原論文所有模型與實驗。

JODIE：S. Kumar, X. Zhang, J. Leskovec, [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://doi.org/10.1145/3292500.3330895), KDD 2019, §4.2。

原作者 MIT 授權與著作權聲明保留於 `LICENSE` 與 `src/LICENSE`。本研究對其有權授權的原創程式新增及修改，採用 `LICENSE` 中的 MIT 條款；著作權仍屬各權利人。第三方程式維持各自適用的授權及聲明，並不因本研究的授權承諾而取得其他權利。DyGLib、GraphMixer、JODIE 的來源及待確認事項見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。資料、模型權重、逐事件分數及歷史摘要等衍生成果的再散布權利仍須依各來源條款確認。
