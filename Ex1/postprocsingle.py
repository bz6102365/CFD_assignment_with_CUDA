import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('D:\\cfd\\CudaSolution1.2\\x64\\Release\\output.csv')
data241=df['data']
data241=data241.tolist()

index241=[i*1/240 for i in range(241)]

l3=plt.plot(index241,data241,label='241')

plt.legend()

plt.title("BTCS") # 图形标题
plt.xlabel("x") # x轴名称
plt.ylabel("C") # y 轴名称

plt.show()