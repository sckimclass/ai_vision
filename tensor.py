import tensorflow as tf

ts = tf.constant(3)
print(ts)

ts1 = tf.constant([3, 4, 5])
print(ts1)

ts2 = tf.constant([6, 8, 10])
print(ts1 + ts2)
print(ts1 - ts2)
print(ts1 * ts2)
print(ts1 / ts2)
