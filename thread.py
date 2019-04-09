#!/home/user/anaconda3/bin/python3 
#-*-coding:utf-8-*-
##  thread.py
import threading
import time

class myThread (threading.Thread):
    def __init__(self,name,delay):
        threading.Thread.__init__(self)       
        self.name = name
        self.delay = delay
        
    def run(self):
        print("Starting " ,self.name)
        print_time(self.name,self.delay,1)
        print("Exiting",self.name)

def print_time(threadName, delay,counter):
    while counter:
        time.sleep(delay)
        print("%s: %s"%(threadName,time.ctime(time.time())))
        counter -= 1

def main():
    '''
    thread_1 = myThread("Thread-1",3)
    thread_2 = myThread("Thread-2",2)
    thread_3 = myThread("Thread-2",5)
    
    thread_1.start()
    thread_2.start()
    '''    
    for delay in range(6):
        ## 后调用先打印
        myThread("Thread-"+str(delay),7-delay).start()
    print("end==========")
    
    

if __name__ == "__main__":
    main()



