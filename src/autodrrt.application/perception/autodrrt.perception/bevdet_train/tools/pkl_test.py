import pickle
root_path = "/home/liry/swpld/AD1129/label/kunyi_infos_val.pkl"
with open(root_path, 'rb') as file:
    data = pickle.load(file)

print(data["infos"])