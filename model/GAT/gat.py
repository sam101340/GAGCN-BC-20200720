from model.GAT.layer import *

#
def inference(inputs,nb_nodes, hid_units, n_heads,ffd_drop=0.0,attn_drop=0.0,residual=False,activation=tf.nn.elu):
    attns = []
    for _ in range(n_heads[0]):
        attns.append(attn_head(inputs, out_sz=hid_units[0]))
        h_1 = tf.concat(attns, axis=-1)
    for i in range(1, len(hid_units)):
        h_old = h_1
        attns = []
        for _ in range(n_heads[i]):
            attns.append(attn_head(h_1, out_sz=hid_units[i], activation=activation,in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        h_1 = tf.concat(attns, axis=-1)
    out = []
    for i in range(n_heads[-1]):
        out.append(attn_head(h_1,out_sz=nb_nodes, activation=lambda x: x,in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
    logits = tf.add_n(out) / n_heads[-1]
    logits_mean = tf.reduce_mean(logits, axis=0)
    logits_mean = tf.reshape(logits_mean, (nb_nodes, nb_nodes))
    return logits_mean
