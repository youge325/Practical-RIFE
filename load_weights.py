from train_log.RIFE_HDv3_paddle import Model
import pickle

model = Model()
with open('train_log/flownet_paddle.pdparams', 'rb') as f:
    state_dict = pickle.load(f)
model.flownet.set_dict(state_dict)
print('Weights loaded successfully')