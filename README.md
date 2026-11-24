# AHEAD：節點度數校正與動態圖異常偵測

本專案以 [AHEAD](https://doi.org/10.1145/3770855.3817610) 的 GraphMixer 模型為基礎，探討節點歷史鄰居數量對異常分數校正的影響。研究比較六種校正方法，使用 Digg、Bitcoin-Alpha、Wikipedia 與 MOOC 四個資料集，以及兩種模型選擇策略和五個隨機種子。

主要發現是：提高低度數事件的異常分數，在 Digg 與 Wikipedia 上改善 AUROC，但在 Bitcoin-Alpha 與 MOOC 上降低 AUROC。校正方向需要考量資料集特性。

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

以下為 Fixed 1.1 減去 Original 的五個隨機種子平均 AUROC 差值，正值表示改善，負值表示下降。兩欄代表不同模型選擇策略。

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

MOOC 在此 D 區間的低度數組沒有正類事件。提高該組分數可能使低度數負類排在部分高度數正類之前。Fixed 1.1 相對 None 的平均 AUROC 差值分別為 −0.004885（Validation AUROC）及 −0.000789（Train Loss）。此結果反映既有分數的排序變化，而非因果關係或其他資料集的普遍規律。

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
| `src/prepare_jodie_research.py` | JODIE 格式資料轉換 |
| `src/research_runs/` | 實驗結果與版本、模型選擇紀錄 |
| `tests/` | 結果展示與驗證工具測試 |

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

如需執行模型或評估指標驗證，請在 Linux 專案根目錄建立虛擬環境。以下為對應原實驗版本的安裝指令：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 依 https://pytorch.org/get-started/locally/ 選擇適合平台的 torch；例如 Linux 的 CUDA 12.6 build：
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

其他平台或 CPU-only 環境請參考 [PyTorch 安裝指引](https://pytorch.org/get-started/locally/)。NVIDIA 驅動顯示的 CUDA 支援版本與 PyTorch CUDA build、CUDA toolkit 版本並不相同。環境相容性與完整重現範圍見下方說明。

## 資料準備

本專案提供程式及實驗統計，不包含原始資料、處理後資料、模型權重或逐事件分數。請依資料來源取得資料並確認使用條款。

模型從 `src/` 工作目錄讀取以下檔案（路徑以專案根目錄表示）：

```text
src/processed_data/<dataset>/ml_<dataset>.csv
src/processed_data/<dataset>/ml_<dataset>.npy
src/processed_data/<dataset>/ml_<dataset>_node.npy
```

`<dataset>` 為 `digg`、`bitcoinalpha`、`wikipedia` 或 `mooc`。CSV 保存互動及標籤，兩個 NPY 檔分別保存邊與節點特徵。

Wikipedia 與 MOOC 可使用 JODIE 格式轉換工具。以下指令從專案根目錄執行，請將 `/path/to/` 替換為實際資料位置：

```bash
python -B src/prepare_jodie_research.py --source /path/to/wikipedia.csv \
  --destination src/processed_data/wikipedia --dataset wikipedia
python -B src/prepare_jodie_research.py --source /path/to/mooc.csv \
  --destination src/processed_data/mooc --dataset mooc
```

輸出包含上述三個檔案及 `conversion_manifest.json`。輸出目錄必須不存在；程式會檢查輸入欄位、事件排序、缺值及特徵格式。轉換使用全資料的 ID mapping，並非嚴格的線上因果流程。

Digg 與 Bitcoin-Alpha 的完整前處理鏈仍有缺項，現有公開資料不足以保證重建與原實驗完全相同的輸入。原始資料檔名及版本資訊可參考 `src/research_runs/` 中的 provenance 紀錄。

## 模型訓練與評估

以下為 Digg 的原實驗參數範例。執行前須準備相符的處理後資料、相依套件、標籤及運算設備。從專案根目錄執行：

```bash
cd src
AHEAD_SHARED_TRAINING_CONFIRM=1 AHEAD_RESEARCH_RUN_ID=my_new_digg_run \
AHEAD_DEGREE_CALIB_MODE=original python -B train_shared_research.py \
  --model_name AHEADGraphMixer --dataset_name digg --num_runs 5 --num_epochs 100 \
  --batch_size 200 --num_neighbors 20 --sample_neighbor_strategy recent \
  --time_gap 2000 --optimizer Adam --learning_rate 0.0001 --val_ratio 0.05 --test_ratio 0.5 --gpu 0
```

`AHEAD_RESEARCH_RUN_ID` 指定實驗名稱，必須使用新名稱；`AHEAD_SHARED_TRAINING_CONFIRM=1` 為訓練確認開關。輸出位於 `src/research_runs/my_new_digg_run/`。`--gpu 0` 使用可見 GPU 0，`--gpu -1` 選擇 CPU。

Wikipedia 與 Bitcoin-Alpha 可調整資料集名稱及實驗名稱；MOOC 使用 `train_segmented_research.py`，並將確認開關改為 `AHEAD_SEGMENTED_TRAINING_CONFIRM=1`。模型選擇資訊保存於各 seed 的 `selection.json` 與 `selection_complete.json`。

開發集推論程式為 `src/run_degree_development.py`，MOOC 使用 `src/run_segmented_development.py`；`src/evaluate_shared_research.py` 為測試集評估程式。這些程式需要相符的實驗檔案與 checkpoint，並會寫入結果，請使用獨立的實驗目錄。

> 注意：原作者的 `src/train.py` 可能刪除 checkpoint，且可能依 Test 表現保存模型。`src/eval_degree_ablation.py` 使用固定輸出路徑，可能覆寫結果。請勿直接在既有實驗目錄執行這些程式。

## 實驗重現與測試

| 等級 | 功能 | 驗證範圍 |
|---|---|---|
| Level 1 | 結果展示 | 已保存指標的統計計算及配對摘要核對 |
| Level 2 | 指標驗證 | 原實驗環境 CPU 驗證，四資料集共 720 項指標 |
| Level 3 | 完整實驗重現 | 從資料前處理、訓練至推論的完整流程仍待驗證 |

### Level 2：評估指標驗證

此功能需要原實驗的 `research_runs/`、`processed_data/`、checkpoint、逐事件分數及相符的程式與封存紀錄。在原實驗相容環境的專案根目錄執行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B scripts/verify_scores.py --assets-root /path/to/original/AHEAD/src --dataset digg
```

`--assets-root` 指向包含 `research_runs/` 與 `processed_data/` 的原實驗 `src/` 或其上一層目錄。`--dataset` 支援 `digg`、`bitcoinalpha`、`wikipedia`、`mooc`、`all`。單一資料集驗證 180 項，全部資料集驗證 720 項。

工具檢查 Python、PyTorch、NumPy、pandas、scikit-learn 的精確版本及相關檔案 SHA-256，重新計算六種方法的評估指標，並與保存結果比對。沿用已保存的 C 倍率選擇結果，不重新執行 Bootstrap、模型推論或訓練。驗證不使用 GPU，也不修改實驗檔案。若驗證失敗，請確認路徑、版本及檔案一致性。

### 單元測試

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -p 'test_repro_tools.py' -v
```

測試涵蓋統計摘要、缺少檔案、雜湊值不一致及錯誤處理，不需要原始資料或 GPU。Windows 可執行 `py -3 -B -m unittest discover -s tests -p test_repro_tools.py -v`。

## 研究限制

- D 區間已用於探索，Fixed 1.1 的方向受探索結果影響，因此結果不是獨立測試集的確認性證據。
- C 方法使用歷史標籤，標籤可用時間尚未確定；保存的逐事件分數缺少 event ID。MOOC 的觀察窗截尾亦未確定，因此不能將結果解讀為即時退課預測或已驗證的線上因果流程。
- 原始 Test 選模可能存在資料洩漏。Level 2 不使用舊 Test 標籤評估或選參數，但完整檔案雜湊與 CSV／OS 緩衝可能讀取其他區段的位元組，並非作業系統層級隔離。
- Digg 合成異常注入及 Bitcoin-Alpha 標籤處理的完整重建流程仍不完整。MOOC 的來源、轉換程式與產物雜湊已與原 manifest 核對，但 Wikipedia 與 MOOC 的重新轉換尚未驗證與原產物完全一致。
- Level 2 的驗證以原實驗環境為準；全新環境安裝、跨平台相容性、完整 CPU 訓練及從新訓練到完整分數的流程仍未驗證。分別執行模型推論也不保證原始分數完全一致。

## 參考資料與授權

本專案基於 [AHEAD 論文](https://doi.org/10.1145/3770855.3817610) 與 [原作者程式](https://github.com/YayaLee666/AHEAD)，參考版本為 `b9d5a805143aa280a665a11d6bdd53d211943888`。`src/README.md` 保留原作者說明。本研究聚焦 GraphMixer 的度數校正，不涵蓋原論文所有模型與實驗。

JODIE：S. Kumar, X. Zhang, J. Leskovec, [Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://doi.org/10.1145/3292500.3330895), KDD 2019, §4.2。

原作者 MIT 授權保留於 `LICENSE` 與 `src/LICENSE`。新增程式的權利與授權由相關權利人決定。DyGLib、GraphMixer、JODIE 的來源及授權範圍見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。資料、模型權重、逐事件分數及衍生結果的再散布權利須依各來源條款確認。
