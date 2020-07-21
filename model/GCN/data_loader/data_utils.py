import numpy as np
from model.GCN.utils.math_utils import z_score


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen_gat(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=3):
    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq
def data_gen_gat_new(data_seq, data_config, n_route, n_frame, day_slot=288):
    n_train, n_val, n_test = data_config
    seq_train = seq_gen_gat(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen_gat(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen_gat(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)
    x_stats_speed_train = {'mean': np.mean(seq_train[:,:,:,2]), 'std': np.std(seq_train[:,:,:,2])}
    x_stats_flow_train = {'mean': np.mean(seq_train[:, :, :, 0]), 'std': np.std(seq_train[:, :, :, 0])}
    x_stats_ocp_train = {'mean': np.mean(seq_train[:, :, :, 1]), 'std': np.std(seq_train[:, :, :, 0])}

    x_train_s = z_score(seq_train[:,:,:,2], x_stats_speed_train['mean'], x_stats_speed_train['std'])
    x_train_f = z_score(seq_train[:,:,:,0], x_stats_flow_train['mean'], x_stats_flow_train['std'])
    x_train_o = z_score(seq_train[:,:,:,1], x_stats_ocp_train['mean'], x_stats_ocp_train['std'])

    x_val_s = z_score(seq_val[:,:,:,2], x_stats_speed_train['mean'], x_stats_speed_train['std'])
    x_val_f = z_score(seq_val[:,:,:,0], x_stats_flow_train['mean'], x_stats_flow_train['std'])
    x_val_o = z_score(seq_val[:,:,:,1], x_stats_ocp_train['mean'], x_stats_ocp_train['std'])

    x_test_s = z_score(seq_test[:,:,:,2], x_stats_speed_train['mean'], x_stats_speed_train['std'])
    x_test_f = z_score(seq_test[:,:,:,0], x_stats_flow_train['mean'], x_stats_flow_train['std'])
    x_test_0 = z_score(seq_test[:,:,:,1], x_stats_ocp_train['mean'], x_stats_ocp_train['std'])

    x_train=np.zeros((seq_train.shape))
    x_train[:,:,:,2] = x_train_s
    x_train[:, :, :, 0] = x_train_f
    x_train[:, :, :, 1] = x_train_o

    x_val  = np.zeros((seq_val.shape))
    x_val[:, :, :, 2] = x_val_s
    x_val[:, :, :, 0] = x_val_f
    x_val[:, :, :, 1] = x_val_o

    x_test = np.zeros((seq_test.shape))
    x_test[:,:,:,2] = x_test_s
    x_test[:, :, :, 0] = x_test_f
    x_test[:, :, :, 1] = x_test_0


    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats_speed_train)
    return dataset



def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        yield inputs[slide]
