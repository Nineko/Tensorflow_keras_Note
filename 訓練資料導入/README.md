# 訓練資料導入
紀錄為了能將訓練資料序列化所進行的轉換動作
利用 Pandas package 將資料轉換成 Data frame , 再以PKL檔案格式儲存
## 欲處理資料格式
在本例中以CAD模型的圖像資料為例 , 每個類別中有不同虛擬相機視點的組合 , 每種組合中都儲存了模型的圖像及模型線框的圖像
'''
Class 1 --
 model --
  Angle 1--
      img
      img
  Angle 2--
      img
      img
 wireframe --
  Angle 1--
      img
      img
  Angle 2--
      img
      img
'''
