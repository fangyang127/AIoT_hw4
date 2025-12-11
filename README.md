八哥辨識器

說明
- 這個專案使用 ResNet50V2 擷取特徵，並在本地訓練一個簡單的分類器來辨識三類八哥（`crested_myna`, `javan_myna`, `common_myna`）。
- 已加入基本的 Data Augmentation（水平翻轉、旋轉、亮度/對比調整、雜訊）。

執行方式（本機）
1. 建議建立虛擬環境並安裝相依套件：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
 - 本程式會在第一次執行時訓練模型並將其儲存在 `myna_model`（SavedModel 格式）與 `myna_model.h5`，後續執行會優先載入 `myna_model.h5`（若存在），因此不需要每次啟動時重新訓練。