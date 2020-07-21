import tensorflow as tf

conv1d = tf.layers.conv1d
def attn_head(seq, out_sz, in_drop=0.0,coef_drop=0.0,residual=False,activation=tf.nn.elu):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 5,padding="same", use_bias=False)

        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 4,padding='same')

        logits = tf.matmul(f_1,tf.transpose(f_2, [0, 2, 1]))
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        ret = tf.contrib.layers.bias_add(coefs)

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)
            else:
                ret = ret + seq

        return activation(ret)


