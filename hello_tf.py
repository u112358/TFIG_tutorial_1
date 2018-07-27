import tensorflow as tf

# x = tf.constant(3.0, dtype=tf.float32, name='x')
# w = tf.constant(4.2, dtype=tf.float32, name='y')
#
# b = tf.constant(5.0, dtype=tf.float32, name='b')
#
# result = tf.add(tf.multiply(x,w),b)
#
#
#
# with tf.Session() as sess:
#     # print(result)
#     # summary_op = tf.summary.merge_all()
#     summary_writer = tf.summary.FileWriter('./log/example1',graph=sess.graph)
#     print(sess.run(result))


x = tf.placeholder(dtype=tf.float32, name='x', shape=(None))
w = tf.constant(4.2, dtype=tf.float32, name='y')

b = tf.constant(5.0, dtype=tf.float32, name='b')

result = tf.add(tf.multiply(x,w),b)



with tf.Session() as sess:
    # print(result)
    # summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./log/example2',graph=sess.graph)
    print(sess.run(result,feed_dict={x:3.0}))

