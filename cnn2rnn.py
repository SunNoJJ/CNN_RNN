#!/home/user/anaconda3/bin/python3 
#-*-coding:utf-8-*-
# cnn2rnn.py
import numpy as np
import tensorflow as tf
import config
import time
import pdb
## config
input_len = 256
output_len = 2
layer_1  = 128
layer_2 = 20
layer_3 = 8
model_path = "cnn2rnn/"
model_name = "cnn2rnn"
train_epoch_step = 999999
train_batch = 3000
learning_rate = 0.075
display_step = 20
train_file = "./data/ww122_train.txt"

file_open = open(train_file,'r',encoding = "utf-8") 
   
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


#####################################
#####################################
#####################################
X = tf.placeholder(tf.float32,[None,input_len],name = 'input')
cnn_in_batch = tf.placeholder(tf.int32,[1],name = 'cnn_in_batch')
Y = tf.placeholder(tf.float32,[None,output_len])

wight = {###                          下面的值为CNN输出列长
    'h1':tf.Variable(tf.random_normal([64,layer_1]),name = 'w_h1',dtype = tf.float32),
    'out':tf.Variable(tf.random_normal([layer_1,output_len]),name = 'w_out',dtype = tf.float32)   
}

biase = {
    'h1':tf.Variable(tf.random_normal([layer_1]),name = 'b_h1',dtype = tf.float32),
    'out':tf.Variable(tf.random_normal([output_len]),name = 'b_out',dtype = tf.float32)
}

recursive = {
    'h1':tf.Variable(tf.random_normal([layer_1,layer_1]),name = 'r_h1',dtype = tf.float32)
}

old_output = {
    'h1':tf.Variable(tf.random_normal([layer_1]),name = 'o_h1',dtype = tf.float32)
}

def network_structure_CNN(value,cnn_in_batch):
    ## CNN  conv2d
    ## 生成卷积的输入: 输入batch量、图高、图宽、图像通道数
    CNNtemp1 = tf.reshape(value,[cnn_in_batch[0],2,128,1])
    ## 设置卷卷积核 核高、核宽、图像通道数、卷积核个数
    conv2d_filter_0 = tf.Variable(tf.random_normal([2,33,1,2]),dtype = tf.float32,name = 'conv2d_filter')
    conv2d_filter_1 = tf.Variable(tf.random_normal([2,17,1,2]),dtype = tf.float32,name = 'conv2d_filter')
    conv2d_filter_2 = tf.Variable(tf.random_normal([2,17,1,1]),dtype = tf.float32,name = 'conv2d_filter')
    ## 配置卷积 输入、卷积核、步长、卷积方式
    CNNtemp2_0 = tf.nn.conv2d(CNNtemp1,filter=conv2d_filter_0, strides = [1,1,1,1],padding = 'VALID')#SAME
    ###
    CNNtemp2_1 = tf.tanh(tf.reshape(CNNtemp2_0,[cnn_in_batch[0],2,96,1]))
    CNNtemp2_2 = tf.nn.conv2d(CNNtemp2_1,filter=conv2d_filter_1, strides = [1,1,1,1],padding = 'VALID')#SAME
    ###
    CNNtemp2_3 = tf.tanh(tf.reshape(CNNtemp2_2,[cnn_in_batch[0],2,80,1]))
    CNNtemp2_4 = tf.nn.conv2d(CNNtemp2_3,filter=conv2d_filter_2, strides = [1,1,1,1],padding = 'VALID')#SAME    
    ###
    CNNtemp3 = tf.reshape(CNNtemp2_4,[cnn_in_batch[0],64])## 元素个数 = 单个卷积结果*卷积核个数  
    return tf.tanh(CNNtemp3)
        
def network_structure_RNN(x,w,b,r,old):
    ## RNN
    RNNtmp1 = tf.matmul(x,w['h1'])
    RNNtmp2 = tf.add(RNNtmp1,b['h1'])
    RNNtmp3 = tf.matmul([old['h1']],r['h1'])
    RNNtmp4 = tf.add(RNNtmp2,RNNtmp3)
    old['h1'] = net1 = tf.tanh(RNNtmp4)  
       
    RNNtmp5 = tf.matmul(old['h1'],w['out'])
    old['out']= net4 = tf.add(RNNtmp5,b['out'])   
    return tf.nn.softmax(old['out'],name = 'output')

CNNresult = network_structure_CNN(X,cnn_in_batch)

### train  训练    
pre = network_structure_RNN(CNNresult,wight,biase,recursive,old_output)
##  选择损失函数计算  残差
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = pre , labels = Y))                        
## 选择优化器  设置优化学习率
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
## 设置优化目标
train = optimizer.minimize(loss)## train
my_global_step = tf.Variable(0, trainable=True,name = 'my_global_step') 
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
    epoch = sess.run(my_global_step)       
    print("epoch:{:5d}".format(epoch))
        
    mini_loss = 999. ## 最小残差 
    all_trian = 1
    while True:
        if epoch > train_epoch_step:break
        avg_cost = 0  
        next_data = next_batch_txtfile(file_open,train_batch)      
        if isinstance(next_data,int):
            all_trian += 1
            file_open.close()
            file_open = open(train_file,'r',encoding = "utf-8") 
            continue 
        if next_data.shape[0] == 322: 
            label = next_data[0:2]
            x_input_1 = next_data[2:258]  
            x_input_2 = next_data[258:322]  
        else: 
            label = next_data[:,0:2]
            x_input_1 = next_data[:,2:258]   
            x_input_2 = next_data[:,258:322]     
             
        x_batch = x_input_1
        y_batch = label
        
        _, _, cost_output = sess.run([CNNresult,train, loss],\
        feed_dict = {X:x_batch,cnn_in_batch:[len(x_batch)],Y:y_batch})## train some  
            
        avg_cost = cost_output / 9505   
        ### 保存最好模型
        if cost_output < mini_loss:
            sess.run(tf.assign(my_global_step, epoch))
            mini_loss = cost_output
            saver.save(sess, model_path+model_name,global_step = epoch)      
        if epoch % display_step == 0:                      
            ## 打印训练信息 
            t0 = time.localtime()
            t1 = time.asctime(t0)     
            print("time:",str(t1),\
                " all_trian:{:4d}".format(all_trian),\
                " iter:{:7d}".format(epoch),\
                " cost:{:.6f}".format(cost_output),\
                " mini_loss:{:.6f}".format(mini_loss) ) 
            

            with open(model_path+"log.utf8",'a+',encoding = 'utf-8') as log_out:
                log_out.write("time:"+str(t1)+" ||all_trian:{:5d}".format(all_trian)+\
                    " ||iter:{:7d}".format(epoch)+\
                    " ||l_rate:{:.3f}".format(learning_rate)+\
                    " ||cost:{:.3f}".format(cost_output)+\
                    " ||mini_loss:{:.6f}".format(mini_loss)+'\n')
        epoch += 1
                       
    print("mini_loss:",'{:.6f}'.format(mini_loss),"   finish!!!!!!!")
    file_open.close()
    while True:
        pass 
 













