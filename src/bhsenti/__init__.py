# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/11/06
# Website: http://www.bhshare.cn
from typing import Dict
import numpy as np
import requests
from tqdm import tqdm
from transformers import BertTokenizer
from onnxruntime import InferenceSession
import os

model_path = os.path.join(os.path.dirname(__file__), "onnx")

if not os.path.exists(model_path + "/model.onnx"):
    url = 'https://media.githubusercontent.com/media/yaokui2018/bhsenti/master/src/bhsenti/onnx/model.onnx'
    print(f"[bhsenti] 首次使用，需要下载模型，请稍候片刻...\n{url}")
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))
    with open(model_path + "/model.onnx.download", 'wb') as file, tqdm(
            desc="正在下载模型文件",
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    os.rename(model_path + "/model.onnx.download", model_path + "/model.onnx")

tokenizer = BertTokenizer.from_pretrained(model_path)
session = InferenceSession(model_path + "/model.onnx")


def predict(text: str) -> str:
    """
    判断文本的情感偏向
    :param text: 待预测文本
    :return: 情感偏向
    """
    return predict_info(text)['result']


def predict_info(text: str) -> Dict:
    """
    判断文本的情感极性
    :param text: 待预测文本
    :return: {
        "text": str 接收到的文本,
        "result": str 预测结果,
        "classes": int 情感类别序号： 0.消极 1.中性 2.积极,
        "score": list[double] 每个类别的预测得分
    }
    """
    inputs = tokenizer(text, return_tensors="np")
    for key in inputs:
        inputs[key] = inputs[key].astype(np.int64)
    outputs = session.run(output_names=None, input_feed=dict(inputs))[0][0]
    result = np.exp(outputs) / np.sum(np.exp(outputs))
    predicted_classes = result.argmax(axis=-1)
    label = ""
    if predicted_classes == 2:
        label = '积极'
    elif predicted_classes == 1:
        label = '中性'
    elif predicted_classes == 0:
        label = '消极'
    return {
        "text": text,
        "result": label,
        "classes": predicted_classes,
        "score": result.tolist()
    }
