import csv
import os
import sys
import pandas as pd
import numpy as np 
from random import shuffle



np.set_printoptions(threshold=np.inf)
saveName = 'Saved_PKL_File_Name'

classDir = ['Need2Process_Folder1','Need2Process_Folder2']

ModelImageList = []
WireImageList = []

"""
Into the folder
We have two subfolder 'model' and 'wireframe' in this case
This example will save the dir of the file
"""
for Folder in classDir:
  print("Process ",Folder)
  #Read model folder
  InModel = os.listdir(Folder+"/model")
  for AngleFile in InModel:
    FileName = os.listdir(Folder+"/model/"+AngleFile)
    for name in FileName:
     ModelImageList.append(Folder+"/model/"+AngleFile+"/"+name)
  
  #Read wireframe folder
  InWireframe = os.listdir(Folder+"/wireframe")
  for AngleFile in InWireframe:
    FileName = os.listdir(Folder+"/wireframe/"+AngleFile)
    for name in FileName:
     WireImageList.append(Folder+"/wireframe/"+AngleFile+"/"+name)

ModelImageList = np.array(ModelImageList)
WireImageList = np.array(WireImageList)

#shuffle
c = list(zip(ModelImageList, WireImageList))
shuffle(c)
ModelImageList, WireImageList = zip(*c)

ModelImageList = np.array(ModelImageList)
WireImageList = np.array(WireImageList)

TotalCount = (ModelImageList.shape)[0]
saveName = saveName+"_"+str(TotalCount)+".pkl"

print("ModelImageList : ",ModelImageList.shape)
print("WireImageList : " ,WireImageList.shape)
print("Transfer to Dataframe")

final_dict = {
  "Model"    : ModelImageList,
  "Wireframe": WireImageList
}

df = pd.DataFrame(final_dict)


print("Save as PKL : ",saveName)

df.to_pickle(saveName)