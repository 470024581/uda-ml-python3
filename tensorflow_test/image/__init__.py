
import tensorflow as tf 

print(tf.VERSION)
# t1=tf.constant([1,2,3])  
# t2=tf.constant([4,5,6])  
#concated = tf.concat(1, [t1,t2])这样会报错  
t1=tf.expand_dims(tf.constant([1,2,3]),1)  
t2=tf.expand_dims(tf.constant([4,5,6]),1)  
concated = tf.concat([t1,t2], 1)#这样就是正确的  



# print(tf.concat(0, [t1, t2]))
# print(tf.concat(1, [t1, t2]))
print(concated)

