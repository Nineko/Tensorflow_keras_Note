# 訓練流程

## 基本架構
```python
model   = model()
seq = DataSequence()
model.compile()
history = model.fit_generator(seq,
                              steps_per_epoch=100,
                              epochs=5
                              )
model.save_weights(Save_Name)
```
## 使用既有模型的輸出
其中若 DataSequence() 中需要既有模型的輸出
則在產生 seq 前需要先進行該模型的運行
例如其中需要模型A的輸出
```
A.load_weights(A_weight,by_name = True)
A.predict(np.zeros(( A_input_size )))
```
若不進行上述步驟，則在訓練時會出現錯誤
