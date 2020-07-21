from model.GAT.gat import *
import tensorflow as tf
def get_weighted_adj_matrix(traffic_data, n_rount):
    n_heads = [8,1]
    hid_units = [8]
    nb_nodes = n_rount
    traffic_data_tf = traffic_data
    logits_mean= inference(traffic_data_tf,nb_nodes,hid_units,n_heads)
    return logits_mean

def first_approx(W_tensor, n):
    A = tf.add(W_tensor, tf.eye(n))
    d = tf.reduce_sum(W_tensor, 1)
    sinvD = tf.matrix_inverse(tf.sqrt(tf.diag(d)))
    mul_f = sinvD * A * sinvD
    return tf.add(tf.eye(n), mul_f)



