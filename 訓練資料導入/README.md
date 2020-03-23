# 訓練資料導入
紀錄為了能將訓練資料序列化所進行的轉換動作
利用 Pandas package 將資料轉換成 Data frame , 再以PKL檔案格式儲存
## 資料預處理
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
## 訓練時使用PKL檔案
使用 keras 的 Sequence 類 , 能夠建立一個能夠將資料在訓練中依照 batch 數量即時讀入並進入訓練的類別
基本架構為
```python
class DataSequence(Sequence):

    def __init__(self, df, batch_size, mode='train'):
        self.df = df
        self.bsz = batch_size
        self.mode = mode 
        
    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))
        
    def __getitem__(self, idx):
    
```
df 為前述的PKL檔案 , 能利用 data = pd.read_pickle(PKL_Dir) 進行讀檔後輸入
讀入的 df 中包含許多資料 , 在 __init__ 中進行進一步的讀取為
```python
    def __init__(self, df, batch_size, mode='train'):
        self.df = df
        self.bsz = batch_size
        self.mode = mode 
        self.Wireframe_list = self.df['Wireframe'].tolist()
        self.Model_list = self.df['Model'].tolist()
        self.indexes = np.arange(len(self.df['Model'].tolist()))
```
其中因為本例中 PKL 檔案儲存的為影像的路徑，為 list 資料 , 所以使用 .tolist() 進行讀取 , 若儲存的為數值型資料 , 能夠利用 .values 進行讀取
此外再新增 on_epoch_end() 使整個epoch的資料使用完時 , 能夠在自動洗亂一次
```python
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.Model_list))
        if self.mode == 'train':
        np.random.shuffle(self.indexes)
```
並新增獲取 batch 內資料的 get_batch_Models() 及 get_batch_Wireframes()
```python
    def get_batch_Models(self, idx):
        Batch_indexes = self.indexes[idx*self.bsz:(idx+1)*self.bsz]
        Train_list_temp = [self.Model_list[k] for k in Batch_indexes]
        batch_model = np.array([load_image(im,color_mode='rgb') for im in Train_list_temp])
        return batch_model
        
    def get_batch_Wireframes(self, idx):
        Batch_indexes = self.indexes[idx*self.bsz:(idx+1)*self.bsz]
        Answer_list_temp = [self.Wireframe_list[k] for k in Batch_indexes]
        return np.array([load_image(im,color_mode='grayscale') for im in Answer_list_temp])
```
最後完善 __getitem__
```python
    def __getitem__(self, idx):
        batch_model = self.get_batch_Models(idx)
        batch_wireframe = self.get_batch_Wireframes(idx)
        return batch_model,batch_wireframe
```
