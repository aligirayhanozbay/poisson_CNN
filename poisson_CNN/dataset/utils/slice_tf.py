import tensorflow as tf
from poisson_lhs_matrix import place_diagonal

s = 0.05*tf.ones((30,40,50))
ct = tf.Variable(tf.zeros(s.shape))

data = s[:29]

print(place_diagonal(data,(-1,0,0),ct))

