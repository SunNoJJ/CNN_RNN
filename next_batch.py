#!/home/user/anaconda3/bin/python3 
#-*-coding:utf-8-*-
# 取 next batch N
import numpy as np
import matplotlib.pyplot as plt

data = np.arange(0,120,2).reshape(20,3) 
data = iter(data)

def next_batch_Array(value,n):
    #新取一行数据，设置取完返回值为 0 整型
    result = next(value,0)
    print(isinstance(result,int))
    #数据类型判断  用于判断是否next到末尾
    if isinstance(result,int):
        return 0
    # 当只取以行时到此结束
    if n == 1:
        return result
    # 取多行时，需要再取 n-1 行数据
    for i in range(n-1):
        data4 = next(value,0)
        if isinstance(data4,int):
            break
        #取多行时需要安行合并
        result = np.row_stack((result, data4 ))
    return result
   


file_open = open("./data/vec2test10.txt",'r',encoding = "utf-8")

def next_batch_txtfile(file_name,n):
    #新取一行数据，设置取完返回值为 0 整型，len = 1
    result = next(file_name,"0").strip().split()
    result = np.array(list(map(eval,result)))
    #数据长度判断  用于判断是否next到末尾
    if len(result) == 1:
        return 0
    # 当只取以行时到此结束
    if n == 1:
        return result
    # 取多行时，需要再取 n-1 行数据
    for i in range(n-1):
        data4 = next(file_name,"0").strip().split()
        data4 = np.array(list(map(eval,data4)))
        
        if len(data4) == 1:
            break
        #取多行时需要安行合并
        result = np.row_stack((result, data4 ))
    return result  
    
data_random = np.random.rand(128,1)
print(data_random)
for i in range(100):
    nex_data = next_batch_txtfile(file_open,train_batch)
    if isinstance(nex_data,int):
        print("fanish>>>>>>>.")
        break  
    print(nex_data.shape[0]) 
    if nex_data.shape[0] == 130: 
        q = np.dot(nex_data[2:130],data_random)
        w = nex_data[0:2]
    else:            
        q = np.dot(nex_data[:,2:130],data_random)
        w = nex_data[:,0:2]
        
'''
if nex_data.shape[0] == (output_len+input_len): 
    nex_data = nex_data.reshape(1,(output_len+input_len)) 
    x_batch = nex_data[0,output_len:(output_len+input_len)].reshape(1,input_len)
    y_batch = nex_data[0,0:output_len].reshape(1,output_len)            
else:
    x_batch = nex_data[:,output_len:(output_len+input_len)]
    y_batch = nex_data[:,0:output_len]#.reshape(len(nex_data[:,0]),2)
'''  
file_open.close()





















