import tensorflow as tf


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

print(dataset.output_shapes)
print(dataset.output_types)

dataset2 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

sess = tf.InteractiveSession()

# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# for i in range(3):
#     value = sess.run(next_element)
#     print(value)

iterator2 = dataset2.make_initializable_iterator()
next_element = iterator2.get_next()
sess.run(iterator2.initializer)

for i in range(4):
    value = sess.run(next_element)
    print(value)

sess.close()

