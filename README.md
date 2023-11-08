## 基于三分类的文本情感分析

基于bert模型，能够识别文本postive, neutral, negative三种情感。

### 安装
`pip install bhsenti`

### 使用

```python 
import bhsenti

pre = bhsenti.predict("待预测文本")
pre_info = bhsenti.predict_info("待预测文本")

print(pre)
# 积极
print(pre_info)
# {'text': '待预测文本', 'result': '积极', 'classes': 2, 'score': [0.3605044484138489, 0.009216712787747383, 0.6302788257598877]}
```
- text: str 接收到的文本,
- result: str 预测结果,
- classes: int 情感类别序号： 0.消极 1.中性 2.积极,
- score: list[double] 每个类别的预测得分