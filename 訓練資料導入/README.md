# 訓練資料導入
紀錄為了能將訓練資料序列化所進行的轉換動作
利用 Pandas package 將資料轉換成 Data frame , 再以PKL檔案格式儲存
## 欲處理資料格式
在本例中以CAD模型的圖像資料為例 , 每個類別中有不同虛擬相機視點的組合 , 每種組合中都儲存了模型的圖像及模型線框的圖像
```
Class 1 --
 model --
  Angle 1--
      img
  Angle 2--
      img
 wireframe --
  Angle 1--
      img
  Angle 2--
      img
Class 2 --
  ... ...
```
當然只要有辦法處裡，用任何結構儲存都行
```python
c = list(zip(ModelImageList, WireImageList))
shuffle(c)
ModelImageList, WireImageList = zip(*c)
```
此步驟將資料綁定後進行洗亂的動作
```python
final_dict = {
  "Model"    : ModelImageList,
  "Wireframe": WireImageList
}
df = pd.DataFrame(final_dict)
df.to_pickle(saveName)
```
此為轉換成PKL檔案的主要指令
另外也能夠存成CSV檔案
不過若是儲存的內容屬於多維度的資訊
使用PKL會比較好一些
