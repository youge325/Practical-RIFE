import pickle
with open('train_log/flownet_paddle.pdparams', 'rb') as f:
    data = pickle.load(f)
print(list(data.keys())[:10])