#!/home/user/anaconda3/bin/python3 
#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import config

input_len = config.INPUT_SIZE
output_len = config.OUTPUT_SIZE 
layer_1  = config.L_1
layer_2 = config.L_2
layer_3 = config.L_3
model_path = config.RNN_MODEL_PATH
train_epoch_step = config.TRAIN_TIME
train_batch = config.TRAIN_BATCH
learning_rate = config.LEARNING_TATE
display_step = config.DISPLAY_STEP

file_open = open(config.TEST_FILE,'r',encoding = "utf-8") 

    
def next_batch_txtfile(file_name,n):
    #新取一行数据，设置取完返回值为 0 整型，len = 1
    result = next(file_name,"0").strip().split()
    result = list(map(eval,result))
    #数据长度判断  用于判断是否next到末尾
    if len(result) == 1:
        return 0
    # 当只取以行时到此结束
    if n == 1:
        return result
    # 取多行时，需要再取 n-1 行数据
    for i in range(n-1):
        data4 = next(file_name,"0").strip().split()
        data4 = list(map(eval,data4))
        if len(data4) == 1:
            break
        #取多行时需要安行合并
        result = np.row_stack((result, data4 ))
    return result  
    
  
##########################################################################
##########################################################################
##########################################################################

#交互式输入
test_batch  = int(input('please input a number between 0~10000:'))

nex_data = next_batch_txtfile(file_open,test_batch)
x_batch = nex_data[:,2:130]
y_batch = nex_data[:,0:2]
file_open.close()  
 
xxx_plt = range(test_batch)    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
real_lable = np.dot(y_batch ,[[0.],[1.]])
ax.scatter(xxx_plt,real_lable,c = 'r',marker = '*')
plt.ion()### continiu
plt.show()

#获取最新路径  
checkpoint = tf.train.get_checkpoint_state("./rnn_model") 
model_name = checkpoint.model_checkpoint_path+".meta"
#打开会话
with tf.Session() as sess:		
    print(model_name)
    #载入模型  
    saver = tf.train.import_meta_graph(model_name)
	#恢复模型
    saver.restore(sess,tf.train.latest_checkpoint('./rnn_model/'))          
	#使用模型进行预测
    pred = sess.run('output:0',feed_dict={'input:0':x_batch})
    
    
    rightNum = 0  
    for i, preid in enumerate(pred):
        real_pre = np.dot(y_batch[i] ,[[0.],[1.]])
        if np.dot(preid,[[0.],[1.]]) > 0.5:
            print(1,"::",real_pre)
            ax.plot(i, 1, 'co', lw = 0.5)                   
            plt.pause(0.1)            
            if real_pre == 1.:
                rightNum += 1 
        else:
            print(0,"::",real_pre)            
            ax.plot(i, 0, 'co', lw = 0.5)        
            plt.pause(0.1)
            if real_pre == 0.:
                rightNum += 1 

    
    title = 'right_rate:'+'{:.2f}'.format(rightNum/test_batch*100.0)+'%'
    plt.title(title)    
    plt.savefig('predict.png')
    while True:
        pass




