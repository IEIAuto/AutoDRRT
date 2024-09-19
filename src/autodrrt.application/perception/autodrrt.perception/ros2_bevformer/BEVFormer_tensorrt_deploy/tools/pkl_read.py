import pickle

with open('/home/orin/disk/BEVFormer_tensorrt/data/nuscenes/nuscenes_infos_temporal_val.pkl','rb')as file:
    model=pickle.load(file)

print(model)
