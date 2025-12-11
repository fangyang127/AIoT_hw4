import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
import PIL.Image as Image
from PIL import ImageOps, ImageEnhance
import streamlit as st
import io

# 辨識類別
category_en = "crested_myna,javan_myna,common_myna"

# 辨識類別的中文, 顯示時用的名稱
category_zh = "土八哥,白尾八哥,家八哥"

# APP 的名稱
title = "八哥辨識器"

# APP 的說明
description="請輸入一張八哥照片, 我會告訴你是什麼八哥!"

categories = category_en.split(',')
labels = category_zh.split(',')

# 辨識有幾類
N = len(categories)

# 讀取圖片資料
base_dir = 'myna/'
thedir = base_dir + categories[0]
data = []
target = []

# Data augmentation settings
USE_AUGMENT = True
AUGMENT_PER_IMAGE = 2

def augment_array(x_arr):
    # x_arr: NumPy array HxWxC, dtype float或uint8
    img_pil = Image.fromarray(np.uint8(x_arr))

    # 隨機水平翻轉
    if random.random() < 0.5:
        img_pil = ImageOps.mirror(img_pil)

    # 隨機旋轉（-25 到 25 度）
    angle = random.uniform(-25, 25)
    img_pil = img_pil.rotate(angle)

    # 隨機亮度與對比度調整
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(random.uniform(0.8, 1.2))

    # 隨機加入高斯雜訊
    arr = np.array(img_pil).astype(np.float32)
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255)

    # 確保尺寸仍為 224x224
    img_resized = Image.fromarray(np.uint8(arr)).resize((224,224), Image.Resampling.LANCZOS)
    return np.array(img_resized)

# 讀取所有圖片並轉成 NumPy 陣列
for i in range(N):
    thedir = base_dir + categories[i]
    file_names = os.listdir(thedir)
    for fname in file_names:
        img_path = thedir + '/' + fname
        img = load_img(img_path , target_size = (224,224))
        x = img_to_array(img)
        data.append(x)
        target.append(i)

        if USE_AUGMENT:
            for _ in range(AUGMENT_PER_IMAGE):
                aug = augment_array(x)
                data.append(aug)
                target.append(i)

data = np.array(data)

# 將圖片資料做前處理
x_train = preprocess_input(data)

y_train = to_categorical(target, N)
# 模型保存路徑（可修改）
MODEL_H5 = 'myna_model.h5'

# 若存在 .h5 檔則載入（方便移植與下載），否則訓練並儲存為 .h5
if os.path.exists(MODEL_H5):
    print(f"Loading model from {MODEL_H5} ...")
    model = load_model(MODEL_H5)
else:
    # 建立模型
    resnet = ResNet50V2(include_top=False, pooling="avg")
    # 建立序列模型
    model = Sequential()
    # 加入 ResNet50V2 作為特徵擷取器
    model.add(resnet)
    # 加入輸出層
    model.add(Dense(N, activation='softmax'))
    # 凍結 ResNet50V2 的權重
    resnet.trainable = False

    # 顯示模型摘要
    model.summary()

    # 編譯模型
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 訓練模型（可依需求調整 batch_size、epochs）
    model.fit(x_train, y_train, batch_size=10, epochs=10)

    # 評估模型
    loss, acc = model.evaluate(x_train, y_train)
    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")

    # 儲存模型為 .h5（方便部署與下載）
    model.save(MODEL_H5)
    print(f"Model saved to {MODEL_H5}")

    y_predict = np.argmax(model.predict(x_train), -1)

def resize_image(inp):
    # 將 NumPy array 轉換成 PIL Image 對象
    img = Image.fromarray(inp)

    # 將圖片調整為 224x224 像素
    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)

    # 將調整大小後的圖片轉換回 NumPy array
    img_array = np.array(img_resized)

    return img_array

def classify_image(inp):
    img_array = resize_image(inp)
    inp = img_array.reshape((1, 224, 224, 3))
    inp = preprocess_input(inp)
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(N)}
# ------------------ Streamlit app ------------------
st.set_page_config(page_title=title)
st.title(title)
st.write(description)

st.sidebar.header("操作")
st.sidebar.write("上傳一張八哥照片後點選『分類』來取得結果。若尚未訓練模型，第一次會花時間訓練並儲存 `myna_model.h5`。")
# 準備範例圖片清單（從本地 myna/ 資料夾）
sample_items = []
for cat in categories:
    thedir = os.path.join(base_dir, cat)
    if os.path.isdir(thedir):
        for fname in os.listdir(thedir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                display_name = f"{cat}/{fname}"
                sample_items.append((display_name, os.path.join(thedir, fname)))

sample_display = ['(無)'] + [s[0] for s in sample_items]
sample_choice = st.sidebar.selectbox('範例圖片', options=sample_display)
if 'selected_image' not in st.session_state:
    st.session_state['selected_image'] = None

if st.sidebar.button('載入範例') and sample_choice != '(無)':
    # 找到對應路徑並載入到 session state
    idx = sample_display.index(sample_choice) - 1
    sample_path = sample_items[idx][1]
    try:
        st.session_state['selected_image'] = Image.open(sample_path).convert('RGB')
    except Exception as e:
        st.sidebar.error(f"載入範例失敗: {e}")

uploaded_file = st.file_uploader("上傳八哥照片", type=['png','jpg','jpeg'])
if uploaded_file is not None:
    # 使用者上傳圖片，覆寫 session 中的 selected_image
    st.session_state['selected_image'] = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')

if st.session_state['selected_image'] is not None:
    img = st.session_state['selected_image']
    st.image(img, caption='選擇的圖像', use_column_width=True)

    # 等使用者按下按鈕才執行分類，避免每次上傳就執行
    if st.button('分類'):
        with st.spinner('正在分析圖片，請稍候...'):
            arr = np.array(img)
            preds = classify_image(arr)

        # 整理成 DataFrame，並由高到低排序
        df = pd.DataFrame(list(preds.items()), columns=['species','probability'])
        df['probability'] = df['probability'].astype(float)
        df = df.sort_values('probability', ascending=False).reset_index(drop=True)

        # 顯示 Top-1 結果
        top_species = df.loc[0, 'species']
        top_prob = df.loc[0, 'probability']
        st.markdown(f"### 預測結果： **{top_species}**  ({top_prob*100:.1f}%)")

        # 顯示所有類別的機率長條圖與表格
        st.subheader('各類別機率')
        chart_df = df.set_index('species')
        st.bar_chart(chart_df)
        st.table(df)

else:
    st.info('請在左側或此處上傳一張八哥照片，然後按「分類」。')

st.markdown("---")
st.write("本程式支援本機訓練與辨識；如需部署，可將此資料夾 push 到 GitHub，並以 Streamlit 部署或在本機執行 `streamlit run 7114056047_hw4.py`。")