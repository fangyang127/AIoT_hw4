八哥辨識器

說明
- 這個專案使用 ResNet50V2 擷取特徵，並在本地訓練一個簡單的分類器來辨識三類八哥（`crested_myna`, `javan_myna`, `common_myna`）。
- 已加入基本的 Data Augmentation（水平翻轉、旋轉、亮度/對比調整、雜訊）。

執行方式（本機 / 在雲端）

1. 建議建立虛擬環境並安裝相依套件（已在本機使用 `tensorflow==2.13.0` 開發）：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

注意：若你在不同環境（例如 Streamlit Cloud）部署，請確保安裝與訓練時相同的 `tensorflow` 版本以避免模型載入錯誤。大型套件（如 TensorFlow）在某些雲端環境可能無法順利安裝或會導致啟動緩慢；必要時請改用輕量模型或在部署前先將模型儲存並上傳到可下載位置（例如 GitHub Releases、S3 或 Google Drive）。

2. 直接執行檔案會先訓練模型（若有大量資料，訓練會花較久）：

```powershell
python .\7114056047_hw4.py
```

3. 使用 Streamlit 啟動 Web 介面：

```powershell
streamlit run 7114056047_hw4.py
```

將專案上傳到 GitHub（範例指令）

```powershell
git init
git add .
git commit -m "Add myna bird classifier with augmentation and streamlit UI"
# 建立遠端 repo 並推送（需替換為你的 repo URL）
# git remote add origin https://github.com/<yourname>/<repo>.git
# git push -u origin main
```

注意
- 若你的環境沒有 GPU 或記憶體有限，請在程式中將 `USE_AUGMENT=False` 或減少 `AUGMENT_PER_IMAGE`，以及減少 `epochs` 或 `batch_size`。
 - 本程式會在第一次執行時訓練模型並將其儲存在 `myna_model`（SavedModel 格式）。後續執行會自動載入 `myna_model`，因此不需要每次啟動時重新訓練。
 - 本程式會在第一次執行時訓練模型並將其儲存在 `myna_model.h5`（`.h5` 格式）。後續執行會優先載入 `myna_model.h5`（若存在），因此不需要每次啟動時重新訓練。
 - 建議在部署前於本機訓練並將 `myna_model.h5` 上傳至可下載位置，然後在部署的環境中僅下載並載入模型，以避免在雲端直接安裝並編譯 TensorFlow。