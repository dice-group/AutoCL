import pandas as pd
import xlwt
data = pd.read_excel('/Users/ljymacbook/Downloads/num.xls',usecols=[0,1])

datafinal=data.values     
column=0
total=dict()
while column < len(datafinal):
 if datafinal[column][0] not in total.keys():
    total[datafinal[column][0]]=datafinal[column][1] 
    
 else:
    unitbegin=total.get(datafinal[column][0])
    unitbegin=unitbegin+datafinal[column][1] 
    total[datafinal[column][0]]=unitbegin
 column=column+1
print(total)
dict = {'Name': total.keys(), 'Unit': total.values()}
df1 = pd.DataFrame(dict)
df1.to_excel('/Users/ljymacbook/Desktop/final.xls', encoding='utf-8')




"""
stu_df = pd.DataFrame(datafinal, columns =['Name', 'Unit']) 
column=0
total=dict()
for column in stu_df:
   columnSeriesObj= stu_df[column]
   
   name=columnSeriesObj.values[column]
   print(name)
   if name not in total.keys():
    total[name]=columnSeriesObj
   else:
    unitbegin=total.get(name)
    unitbegin=columnSeriesObj+unitbegin
    total[name]=unitbegin
print(total.items)

"""
