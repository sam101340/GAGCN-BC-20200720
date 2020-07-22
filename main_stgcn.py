import os
import tensorflow as tf

print(tf.__version__)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from model.GCN.data_loader.data_utils import *
from model.GCN.models.trainer import model_train
from model.GCN.models.tester import model_test
import argparse


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--traffic_data_path', default=r'data\dataset.npz')
parser.add_argument('--n_route', type=int, default=50)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=1)
parser.add_argument('--kt', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')



args = parser.parse_args()

print(f'Training configs: {args}')
n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
blocks = [[1, 32, 64], [64, 32, 128]]

n_train, n_val, n_test = 34, 5, 5
data = np.load(args.traffic_data_path)



PeMS = data_gen_gat_new(data['dataset'], (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS,args.batch_size, n_his, n_pred, args.inf_mode)
