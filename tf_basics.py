import tensorflow as tf


# dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#
# print(dataset.output_shapes)
# print(dataset.output_types)
#
# dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
#
# sess = tf.InteractiveSession()
#
# # iterator = dataset.make_one_shot_iterator()
# # next_element = iterator.get_next()
# #
# # for i in range(3):
# #     value = sess.run(next_element)
# #     print(value)
#
# iterator2 = dataset2.make_initializable_iterator()
# next_element = iterator2.get_next()
# sess.run(iterator2.initializer)
#
# for i in range(4):
#     value = sess.run(next_element)
#     print(value)
#
# sess.close()


# mat1 = tf.constant([[1, 2, 3]])
# mat2 = tf.constant([[3], [2], [1]])
#
# print(mat1.dtype)
#
# product = tf.matmul(mat1, mat2)
#
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()


state = tf.Variable(0, name='counter')

one = tf.constant(1)
new_val = tf.add(state, one)
update = tf.assign(state, new_val)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    print(sess.run(update))
    print(sess.run(state))
