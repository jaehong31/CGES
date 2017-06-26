import tensorflow as tf
import numpy as np

glayerwise = [1.,1.0, 1./15, 1./144]
elayerwise = [1.,0.5, 15., 144.]

def _comp(all_vars):
    weight_size = []
    sparsity = []
    total_sp = 0.
    for i in range(len(all_vars)):
        per_weight = tf.reduce_prod(tf.shape(all_vars[i]))
        per_sparsity = tf.reduce_mean(tf.cast(tf.equal(all_vars[i], 0.), tf.float32))
        weight_size.append(per_weight)
        sparsity.append(per_sparsity)
        total_sp = total_sp + tf.cast(per_weight, tf.float32) * per_sparsity
    total_sp = total_sp / tf.cast(tf.reduce_sum(weight_size), tf.float32)
    return (1. - total_sp), sparsity

def _cost(sparsity):
    full_cost = [(28-5+1)*(28-5+1)*(1-1+1)*5*5*1*32., \
                        (14-5+1)*(14-5+1)*(32-32+1)*5*5*32*64., 7*7*64*1024, 1024*10]
    flop = sparsity[0]*full_cost[0] + sparsity[1]*full_cost[1] + sparsity[2]*full_cost[2] + sparsity[3]*full_cost[3]
    flop = np.float32(np.sum(full_cost)-flop) / np.sum(full_cost)
    return flop
