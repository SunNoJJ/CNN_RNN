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
model_path = config.MODEL_PATH
train_epoch_step = config.TRAIN_TIME
train_batch = config.TRAIN_BATCH
learning_rate = config.LEARNING_TATE
display_step = config.DISPLAY_STEP

file_open = open(config.TRAIN_FILE,'r',encoding = "utf-8") 


    
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
    
  
#####################################
#####################################
#####################################
X = tf.placeholder(tf.float32,[None,input_len],name = 'input')
Y = tf.placeholder(tf.float32,[None,output_len])

wight = {
    'h1':tf.Variable(tf.random_normal([input_len,layer_1]),name = 'w_h1',dtype = tf.float32),
    'h2':tf.Variable(tf.random_normal([layer_1,layer_2]),name = 'w_h2',dtype = tf.float32),
    'out':tf.Variable(tf.random_normal([layer_2,output_len]),name = 'w_out',dtype = tf.float32)   
}

biase = {
    'h1':tf.Variable(tf.random_normal([layer_1]),name = 'b_h1',dtype = tf.float32),
    'h2':tf.Variable(tf.random_normal([layer_2]),name = 'b_h2',dtype = tf.float32),
    'out':tf.Variable(tf.random_normal([output_len]),name = 'b_out',dtype = tf.float32)
}

recursive = {
    'h1':tf.Variable(tf.random_normal([layer_1,layer_1]),name = 'r_h1',dtype = tf.float32),
    'h2':tf.Variable(tf.random_normal([layer_2,layer_2]),name = 'r_h2',dtype = tf.float32),
    'out':tf.Variable(tf.random_normal([output_len,output_len]),name = 'r_out',dtype = tf.float32)
}

old_output = {
    'h1':tf.Variable(tf.random_normal([layer_1]),name = 'o_h1',dtype = tf.float32),
    'h2':tf.Variable(tf.random_normal([layer_2]),name = 'o_h2',dtype = tf.float32),
    'out':tf.Variable(tf.random_normal([output_len]),name = 'o_out',dtype = tf.float32)
}

def network_structure(x,w,b,r,old):
    tmp1 = tf.matmul(x,w['h1'])
    tmp2 = tf.add(tmp1,b['h1'])
    tmp3 = tf.matmul([old['h1']],r['h1'])
    tmp4 = tf.add(tmp2,tmp3)
    old['h1'] = net1 = tf.nn.relu(tmp4)       
    old['h2'] = net2 = tf.nn.relu(tf.add(tf.matmul([old['h2']],r['h2']),tf.add(tf.matmul(old['h1'],w['h2']),b['h2'])))   
    old['out']= net4 = tf.add(tf.matmul([old['out']],r['out']),tf.add(tf.matmul(old['h2'],w['out']),b['out']))
    return old['out']

### train  训练    
pre = tf.nn.softmax(network_structure(X,wight,biase,recursive,old_output),name = 'output')
##  选择损失函数计算  残差
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = pre , labels = Y))                        
## 选择优化器  设置优化学习率
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)### optimizer
## 设置优化目标
train = optimizer.minimize(loss)## train
## 初始化全部Tensor
init = tf.global_variables_initializer()



### 启动TF
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep = 2)
    sess.run(init) 
    ckpt = tf.train.get_checkpoint_state(model_path)#resource/model/
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        print(  'loading pre-trained model from %s.....'% path)
        saver.restore(sess, path) 
        
    mini_loss = 999. ## 最小残差 
    for epoch in range(train_epoch_step):
        avg_cost = 0  
        nex_data = next_batch_txtfile(file_open,train_batch)      
        if isinstance(nex_data,int):
            file_open.close()
            file_open = open(config.TRAIN_FILE,'r',encoding = "utf-8") 
            continue      
        x_batch = nex_data[:,2:130]
        y_batch = nex_data[:,0:2]#.reshape(len(nex_data[:,0]),2)
        
        _, cost_output = sess.run([train, loss], feed_dict = {X:x_batch, Y:y_batch}) ## train some  
            
        avg_cost = cost_output / 9505   
        ### 保存最好模型
        if cost_output < mini_loss:
            mini_loss = cost_output
            saver.save(sess, model_path,global_step = epoch)        
        if epoch % display_step == 0:                      
            ## 打印训练信息      
            print("iter:",'{:.1f}'.format(epoch),"  l_rate:",'{:.4f}'.format(learning_rate),"  cost:",'{:.6f}'.format(cost_output),\
            "  avg_cost:",'{:.10f}'.format(avg_cost),"  mini_loss:",'{:.6f}'.format(mini_loss) )           
                       
    print("mini_loss:",'{:.6f}'.format(mini_loss),"   finish!!!!!!!")
    file_open.close()
    while True:
        pass 
      
   
       







