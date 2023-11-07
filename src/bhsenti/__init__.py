# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/11/06
# Website: http://www.bhshare.cn
from typing import Dict
import numpy as np
from transformers import BertTokenizer
from onnxruntime import InferenceSession
import os

model_path = os.path.join(os.path.dirname(__file__), "onnx")
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
    :return: map
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
