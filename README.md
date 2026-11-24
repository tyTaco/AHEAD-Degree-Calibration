# AHEAD：degree 校正與實驗重現

本研究基於 [AHEAD 論文](https://doi.org/10.1145/3770855.3817610) 與[作者程式](https://github.com/YayaLee666/AHEAD)，作者參考 commit `b9d5a805143aa280a665a11d6bdd53d211943888`；作者 MIT 授權保留於 `src/LICENSE` 及根目錄 `LICENSE`；DyGLib 完整授權與其他來源的核對界線見 `THIRD_PARTY_NOTICES.md`。研究範圍為 AHEAD 的 degree 校正與比較，而非從零實作整篇論文。根目錄 LICENSE 適用於原作者授權的內容；新增程式的權利與授權由相關權利人決定。

作者 Original 在 source node 的 sampled degree `<3` 時將分數乘 0.9；本研究模型新增 None 與 Continuous。Fixed 1.1 是獨立後處理，不是模型的第四模式；C-Pairwise、C-Guarded 使用歷史 C 標籤。主要位置：`src/models/AHEADGraphMixer.py`、`src/run_fixed_degree_up_development.py`、`src/retrospective_pairwise_development.py`、`src/conservative_pairwise_development.py`。

## 實驗重現範圍

| 等級 | 內容 | 可取得條件與已驗證範圍 |
|---|---|---|
| Level 1 | 從已保存 JSON 重新計算四資料集、兩種選模、五個隨機種子、六方法的 AUROC／AP／PR-AUC 均值與樣本 SD，及 Fixed 1.1 對其他方法的配對差 | **已實測**；取得本專案即可以 Python 標準函式庫執行。它不讀逐筆分數、不是模型重跑。 |
| Level 2 | 使用外部原研究資產與保存 raw scores、degree、D labels，重新計算六方法與三指標並比對 JSON | **已在原研究資產與原 Python 環境 CPU 實測**；需外部檔案及完全匹配的版本／hash。**不執行模型或訓練**。 |
| Level 3 | 資料前處理、訓練、checkpoint 選擇、推論與最終評估 | **尚未完成端對端驗證**；本專案不含資料、checkpoint、逐筆 scores；Digg 注入及 Alpha 資料重建尚待確認。 |

### Level 1：離線摘要

在**專案根目錄**執行：

```bash
python3 -B scripts/show_results.py
```

顯示 144 組方法均值與樣本 SD、120 組 Fixed 1.1 相對其他方法的配對差均值與樣本 SD，並核對保存的配對摘要。不安裝套件、不需要外部資產、不寫檔。輸出是已保存資料的重新計算，不是重新訓練或新測試。

### Level 2：外部資產逐筆分數核對

需要原研究目錄的 `src/research_runs/`、`src/processed_data/`、選定 checkpoint 和原程式；本專案不提供這些大型資料。Level 2 的 `--assets-root` 接受包含 `research_runs/`、`processed_data/` 的原研究 `src/`，也接受其上一層專案根目錄。請依實際資產位置調整路徑；不須複製大型資料到本專案。

在**專案根目錄**，且已啟用與原研究封存相符的 Python 環境後執行：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B scripts/verify_scores.py --assets-root /path/to/original/AHEAD/src --dataset all
```

單一資料集可改為 `--dataset digg`，其他值為 `bitcoinalpha`、`wikipedia`、`mooc`。此入口驗證外部報告、程式、資料、score、checkpoint 及封存，重新算六方法三指標並對照既有 JSON：全資料集 **720 項**，單資料集 **180 項**。不呼叫原本會寫報告的 `main`，不寫資產、不使用 GPU；缺檔、hash 不一致與環境不相符會失敗並顯示錯誤。

已在原研究 `.venv`、CPU、四資料集實測；封存驗證會核對 Python、PyTorch、NumPy、pandas、scikit-learn 的精確版本及原程式與資料雜湊值，其他環境可能無法通過，即使套件一般情況可執行。執行時間取決於磁碟與資源。程式使用儲存的 C 選擇結果，**不重新執行 C bootstrap**。

### Level 3：尚未驗證

本專案不包含完整資料處理到模型訓練、推論的可驗證一鍵流程。`src/train_shared_research.py`、`src/train_segmented_research.py` 等為研究入口，但不能宣稱已完成端對端訓練重現。**請勿直接執行 `src/train.py`**：它可能刪除 checkpoint，且 Test 表現可能影響保存。`src/eval_degree_ablation.py` 使用固定輸出路徑，也可能覆寫既有結果。

## 環境與測試

Level 1 僅需 Python 3。Level 2 的原研究環境為 Python 3.12.3、PyTorch 2.6.0+cu126，其他版本見 `requirements.txt`。請依 [PyTorch 官方指引](https://pytorch.org/get-started/locally/)選擇相容版本，並依執行需求安裝其餘套件；全新環境安裝及跨平台相容性**尚未實測**。

在**專案根目錄**執行以下小型合成測試，不需研究資產或 GPU：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -p 'test_repro_tools.py' -v
```

測試摘要數量、錯誤摘要、缺檔、報告／程式 hash 不一致與失敗時不新增資產檔；它不代表真實模型已訓練或證明方法有效。

## 資料使用與研究限制

JODIE 的 [Wikipedia、MOOC 資料來源](https://snap.stanford.edu/jodie/)及 [SNAP Bitcoin-Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) 可供查閱；Digg 需依原資料與論文來源確認授權。`src/prepare_jodie_research.py` 為現有 JODIE 轉換程式，但轉換使用全檔 metadata。下載原始資料不保證與研究時資料相同，尤其 Digg 合成異常與 Alpha 標籤的完整重建仍待確認；公開或再散布資料、權重與 scores 前應核實各自授權。

D 已被探索；Fixed 1.1 方向受探索結果影響；C-Guarded 使用歷史標籤且發布時間未知；舊 scores 缺 event ID；舊 Test 的 checkpoint 選擇可能存在 leakage。完整檔案雜湊和 CSV／OS 緩衝可能讀取其他區段 bytes，並非 OS 層級隔離。Level 2 不以舊 Test labels 評估或選參數，也不表示逐事件線上因果流程已驗證。

Digg 的部分條件改善，但 Alpha／MOOC 等條件退步，不能宣稱跨資料集普遍改善或論文已完整重現。`src/README.md` 為作者原文；實際封裝操作以本文件為準。

## Experimental Environment：實驗環境與安裝

下表列出原研究使用的硬體與軟體環境（2026-11-24 紀錄）。全新環境的安裝流程尚未驗證。

| 項目 | 查詢結果 |
|---|---|
| OS | Ubuntu 24.04.5 LTS，Linux |
| CPU／RAM | Intel Core i7-13700K，16 核、24 執行緒；記憶體顯示約 31 GiB |
| GPU／顯存 | NVIDIA GeForce RTX 3090，24,576 MiB |
| NVIDIA driver | 595.99.02；`nvidia-smi` 顯示 driver 支援 CUDA 13.2，這不是 PyTorch build 版本 |
| Python／PyTorch | Python 3.12.3；PyTorch 2.6.0+cu126，PyTorch CUDA build **12.6** |
| 套件 | NumPy 1.26.4、SciPy 1.13.1、pandas 2.2.3、scikit-learn 1.5.2；其餘參考 `requirements.txt` |

Level 1 只用 Python 標準函式庫；Level 2 和訓練相關程式需要相依套件。Level 1、Level 2 及以下工具單元測試不使用 GPU。模型訓練負荷高；程式可用 `--gpu -1` 選 CPU，但完整 CPU 訓練的耗時與可行性未驗證。GPU 訓練需要可用的相容 PyTorch CUDA build、驅動及足夠資源；driver 顯示的 CUDA 版本不能視為本研究安裝的 CUDA toolkit 版本。

以下在**專案根目錄**執行；此為安裝示例，**尚未在全新環境實測**：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 依 https://pytorch.org/get-started/locally/ 選擇適合平台的 torch；例如 Linux 的 CUDA 12.6 build：
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

上例下載套件且需要網路、磁碟及平台相容性；CPU-only 安裝請以 PyTorch 官方頁當前指令替換 torch 那一行。Level 2 的封存檢查要求原環境精確版本與 hash，CPU-only torch 或其他版本即使可匯入也可能被拒絕。缺 `python3.12`、venv 支援或網路時，需先完成相關環境設定。

## Dataset Sources：來源、標籤與檔案

本專案不含資料。四者均用於本研究異常分數比較，但標籤意義不同；以下是本研究標籤定義，不應直接等同論文的完整標註流程。下載前需確認來源條款及引用要求；可下載不等於可自由再散布。

| 資料集 | 公開來源、原始格式與本研究用途／標籤 | 原論文資料集 |
|---|---|---|
| Digg | [KONECT Digg reply 頁及 TSV 下載](https://www.konect.cc/networks/munmun_digg_reply/)；時間性回覆網路、TSV 壓縮檔。研究版合成替換邊為正類。 | 是 |
| Bitcoin-Alpha | [SNAP 說明](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)、[CSV gzip](https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz)，欄位 source、target、rating、time；本研究負信任評分為正類代理，非已證實詐欺。 | 是；本研究正類數與論文不同 |
| Wikipedia | [JODIE 資料頁](https://snap.stanford.edu/jodie/)、[CSV](https://snap.stanford.edu/jodie/wikipedia.csv)，使用者、項目、時間、state label 與邊特徵；封禁相關的最後互動是正類代理。 | 是 |
| MOOC | [JODIE 資料頁](https://snap.stanford.edu/jodie/)、[CSV](https://snap.stanford.edu/jodie/mooc.csv)，相同 JODIE 格式；退課相關的最後互動是正類代理。 | **否，不在原論文六個實驗資料集內** |

### MOOC 的選擇與研究價值

既有研究紀錄顯示，最初尋找額外動態圖資料時曾同時查閱 MOOC 和 Reddit，並核對 MOOC 標籤；紀錄未完整保存最終選定 MOOC 的個別決策理由。就研究適用性而言，MOOC 提供依時間排序的學生–課程活動互動，可在 AHEAD 論文六個資料集之外，觀察另一領域與標籤定義下的 degree 校正行為；因此，以下說明屬於事後分析，而非原始選擇動機。

[SNAP 官方說明](https://snap.stanford.edu/data/act-mooc.html)的全資料統計是 7,047 位使用者、97 種目標活動、411,749 筆互動及 4,066 個正標籤；每事件有時間與 4 維特徵。本研究 MOOC 轉換 manifest 也記錄 411,749 筆、7,047／97 節點與 4 維特徵。這是有向時間二分圖；來源將退課狀態變化標在使用者最後一次互動，本研究將其作為**異常評分代理標籤**，不等於欺詐或一般異常。標籤發布時間與觀察窗截尾仍未知。引用：S. Kumar, X. Zhang, J. Leskovec, [*Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*](https://doi.org/10.1145/3292500.3330895), KDD 2019，§4.2。

下表為已探索 **D 區間**的互動事件，`d` 是 source 的 sampled historical degree，而非完整圖 unique degree；比例是組內正類／組內事件。兩種選模策略及五個隨機種子使用相同事件分組，不是十份獨立資料。

| 資料集、D 列位置（左閉右開） | 低 d<3 事件／正類／比例 | 高 d≥3 事件／正類／比例 |
|---|---:|---:|
| Digg `[52576,61339)` | 2,885／245／8.4922% | 5,878／148／2.5179% |
| Bitcoin-Alpha `[14532,16940)` | 471／12／2.5478% | 1,937／213／10.9964% |
| Wikipedia `[94484,110232)` | 1,288／13／1.0093% | 14,460／8／0.0553% |
| MOOC `[247049,288224)` | 865／0／0% | 40,310／389／0.9650% |

數值來自 `src/research_runs/unlabeled_median_exploration_20261123/report.json` 的 groups，並以原研究 `processed_data` D labels 及 `degree_development/*_scores.npz` 的 `development_degrees` 核對；逐筆資產不包含於本專案。MOOC 低組沒有正類只適用此 D。Fixed 1.1 相對 None 的 MOOC 五個隨機種子的平均 AUROC 差（Validation／Train Loss）為 `−0.004885/−0.000789`；相對 Original 為 `−0.015060/−0.002207`。既有分數的正負配對分解顯示：低組沒有正類可獲得跨組收益，而部分高組正類相對低組負類的排序受損。這是所測分數的**事後排序解釋**，不是資料生成因果或普遍保證；指標來源為 `src/research_runs/fixed_degree_up_exploration_20261123/report.json`。

MOOC 能比較此回顧式退課代理任務中，相同 checkpoint／scores／degree 下不同校正如何改變排序；不能證明即時退課預測或跨資料集普遍改善。D 已探索、C 標籤發布時間未知，舊 scores 缺 event ID。

研究時原始檔分別位於原研究目錄 `src/raw_data/download.tsv.munmun_digg_reply.tar.bz2`、`src/raw_data/soc-sign-bitcoinalpha.csv.gz`、`src/raw_data/wikipedia.csv`、`src/raw_data/mooc_jodie_20190714.csv`；這些路徑是歷史紀錄，專案不含它們。下載後先核對來源、授權及 provenance 中的 hash，檔名相同不保證內容相同。

模型 loader 預期由**本專案 `src/` 工作目錄**讀取：

```text
src/processed_data/<dataset>/ml_<dataset>.csv
src/processed_data/<dataset>/ml_<dataset>.npy
src/processed_data/<dataset>/ml_<dataset>_node.npy
```

`<dataset>` 為 `digg`、`bitcoinalpha`、`wikipedia`、`mooc`；CSV 為互動與 label，兩個 NPY 分別為 edge、node 特徵。MOOC 原研究還有 `conversion_manifest.json`。本 Repository 不包含上述檔案；Level 2 不要求放入本專案，而是讀取外部研究資產。

### 前處理：已核對參數，完整資料重建尚未驗證

JODIE 轉換器 `src/prepare_jodie_research.py` 接受本地 CSV，保留 label 與事件順序，以全檔 metadata 產生 ID mapping。下例**會寫入新目錄**，需先自行下載對應 CSV 並確定輸出目錄不存在；沒有執行過此新目錄重建。工作目錄是**專案根目錄**，不需 GPU：

```bash
python -B src/prepare_jodie_research.py --source /path/to/wikipedia.csv \
  --destination src/processed_data/wikipedia --dataset wikipedia
python -B src/prepare_jodie_research.py --source /path/to/mooc.csv \
  --destination src/processed_data/mooc --dataset mooc
```

`--source` 為已有原始 CSV，`--destination` 是不存在的輸出目錄，`--dataset` 決定檔名前綴；輸出上列三檔及 `conversion_manifest.json`。已存在目錄會拒絕覆寫，輸入 header、排序、缺值或特徵格式不符亦會失敗。MOOC 原研究的 source／converter／產物 hash 與 manifest 符合，但新目錄轉換與逐位重建未測；Wikipedia 也未證明與原研究產物逐位相同。Digg 合成注入的完整重建程式及 Alpha 完整處理鏈尚未確認，**沒有可宣稱完整重現的前處理指令**。全資料 ID mapping 亦不是嚴格線上因果流程。

## How to Run：執行流程

### 1. 取得程式與環境

在欲存放專案的父目錄執行：

```bash
git clone https://github.com/tyTaco/AHEAD-Degree-Calibration.git ahead-course
cd ahead-course
```

亦可下載 ZIP 並解壓縮。獨立 clone 後的執行驗證尚待完成。執行上節 venv 與安裝指令，依資料來源自行取得資料並依上節放置；目前沒有從公開原始資料到四組研究輸入的完整驗證流程。clone／下載／安裝不需要 GPU，但需要網路及相應授權。

### 2. 訓練及 checkpoint selection：歷史設定與未重跑範例

依原實驗 provenance 紀錄：每資料集 seed 0–4、固定 100 epochs、batch 200、recent 20 neighbors、time gap 2000、Adam、learning rate 0.0001、val ratio 0.05、test ratio 0.5。Digg／Alpha／Wikipedia 使用 `src/train_shared_research.py`，MOOC 使用 `src/train_segmented_research.py`。兩種策略選同一條訓練軌跡的最低 Train Loss 與最高 Validation AUROC；後者使用 validation labels，前者雖不使用其 labels 選模，共享流程仍計算 validation AUROC。每 seed 的 `selection.json` 保存 epoch、指標和 checkpoint hash，`selection_complete.json` 封存選擇；不要以舊 `.pkl` 檔名推測選模依據。

以下是**依程式參數及歷史紀錄整理，尚未從公開專案端對端重跑的示例**。必須先在本專案準備相符資料、檢查標籤時序、計算資源與授權，並使用不存在的新 run ID；會訓練及產生新 checkpoint，不是唯讀展示。從 **專案根目錄（再切換至 `src/`）** 執行：

```bash
cd src
AHEAD_SHARED_TRAINING_CONFIRM=1 AHEAD_RESEARCH_RUN_ID=my_new_digg_run \
AHEAD_DEGREE_CALIB_MODE=original python -B train_shared_research.py \
  --model_name AHEADGraphMixer --dataset_name digg --num_runs 5 --num_epochs 100 \
  --batch_size 200 --num_neighbors 20 --sample_neighbor_strategy recent \
  --time_gap 2000 --optimizer Adam --learning_rate 0.0001 --val_ratio 0.05 --test_ratio 0.5 --gpu 0
```

`--gpu 0` 使用可見 GPU 0；改 `--gpu -1` 為 CPU。`AHEAD_SHARED_TRAINING_CONFIRM=1` 是程式的訓練確認開關，用於避免意外啟動訓練。輸出為 `src/research_runs/my_new_digg_run/`；同名 run 已存在會失敗，應選全新名稱。Wikipedia、Alpha 替換 `--dataset_name` 及 run ID；MOOC 使用同樣參數，但將入口換成 `train_segmented_research.py`、確認環境變數換成 `AHEAD_SEGMENTED_TRAINING_CONFIRM=1`、資料集換為 `mooc`。執行前須備妥資料、相依套件、相容設備及所需標籤。`src/train.py` 可能刪除 checkpoint 且可能由 Test 選模，**不要直接執行**。

### 3. 分數、校正與指標

模型 `src/models/AHEADGraphMixer.py::compute_anomaly_score` 產生異常分數並計算 source 的 sampled historical degree `d`。它是取樣到的有效鄰居數，不是完整圖的唯一鄰居數。模型模式由 `AHEAD_DEGREE_CALIB_MODE=original`、`none`、`continuous` 設定；三者分別為 `s×(0.9 if d<3 else 1)`、`s`、`s×(1−0.1 exp(−d/3))`。Original 是論文／作者既有方法，後兩者是本研究修改；不同推論分別執行並不保證得到完全相同 raw score。

Fixed 1.1 是 `src/run_fixed_degree_up_development.py::analyze` 的**獨立後處理**：`s×(1.1 if d<3 else 1)`，不是模型第四模式。C-Pairwise 在 `src/retrospective_pairwise_development.py::choose_factor` 用較早 C labels 從 1、0.9、1.1 選倍率；C-Guarded 在 `src/conservative_pairwise_development.py::choose_guarded_factor` 增加時間區塊 bootstrap 的保守條件。兩者都是使用歷史標籤的候選；標籤何時可用尚未確定。

歷史開發推論入口為 `src/run_degree_development.py`；MOOC segmented 為 `src/run_segmented_development.py`。它們需要對應原始研究資產及已封存 checkpoint，會做模型推論並寫入開發結果；歷史 Test 評估入口 `src/evaluate_shared_research.py` 則會讀取 Test labels 並寫檔。**本專案尚未驗證從新 run 到完整 score 的端對端命令**，不要對既有 run 執行這些寫入型入口。`src/eval_degree_ablation.py` 亦有固定輸出路徑，可能覆寫。核對既有分數的指令請用下方 Level 2；它重新計算 Fixed／C 方法及 AUROC、AP、梯形 PR-AUC，沿用既有 C 選擇，並不重跑模型或 C bootstrap。AP 與梯形 PR-AUC 是不同指標。

### 4. 結果展示與測試

在**專案根目錄**，Level 1 讀 `src/research_runs/fixed_degree_up_exploration_20261123/report.json`，輸出四資料集、兩種選模策略、六方法三指標的均值／樣本 SD 及配對差，不寫檔、不需 GPU：

```bash
python3 -B scripts/show_results.py
```

Level 2 在同一工作目錄、原研究相容環境執行；`--assets-root` 是外部研究 `src/` 或專案根目錄，需原始 processed data、checkpoint、scores、程式及封存，`--dataset digg` 可改 `all`、`bitcoinalpha`、`wikipedia`、`mooc`：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B scripts/verify_scores.py --assets-root /path/to/original/AHEAD/src --dataset digg
```

單資料集預期比對 180 項，`all` 為 720 項；只在終端顯示，不重寫成果。缺檔、版本或 hash 不符會報錯；先檢查資產、版本和路徑，勿修改封存來強制通過。以下 CPU 合成測試不需資料、GPU 或寫入研究成果：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -p 'test_repro_tools.py' -v
```

### 5. 結果與驗證狀態

四資料集 Fixed 1.1 相對 Original 的五個隨機種子的平均 AUROC 差（Validation／Train Loss 選模）分別為 Digg `+0.063857/+0.126480`、Bitcoin-Alpha `−0.043053/−0.020316`、Wikipedia `+0.034676/+0.040315`、MOOC `−0.015060/−0.002207`。差值為 AUROC 的直接相減結果，兩種選模與資料集分開解讀；完整六方法及 AP／PR-AUC 由 Level 1 重新計算。這包含模型模式修改、獨立後處理、選模與跨資料集比較，並非僅在原資料集執行原程式；D 已探索，不能當獨立確認或宣稱普遍改善。

| 狀態 | 項目 |
|---|---|
| 已實際執行並驗證 | 既有環境 Level 1；外部原研究資產 CPU Level 2 720 項及單資料 180 項；小型合成工具測試。 |
| 依歷史實驗紀錄整理 | 模型訓練設定、checkpoint 與四資料集結果；不等於重新訓練。 |
| 已核對參數但尚未重新執行 | 上述全新環境安裝、JODIE 轉換與訓練範例；其輸入、環境與結果一致性仍須驗證。 |
| 尚缺資料或無法確認 | Digg／Alpha 原始資料完整重建、乾淨環境端對端 Level 3、標籤發布時間、舊 scores 的 event ID 對齊及所有資料再散布權利。 |
