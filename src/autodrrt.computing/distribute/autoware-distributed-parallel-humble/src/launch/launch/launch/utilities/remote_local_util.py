
from ament_index_python import get_package_share_directory
import os
import yaml

from ..ssh_machine import SshMachine
multi_machine_pkg_prefix = get_package_share_directory('launch')

node_config_param_file = os.path.join(
        multi_machine_pkg_prefix, 'param/node_config.param.yaml')

print(node_config_param_file)

# 判断是否有路径

# 读取yaml文件
def read_yaml_all():
    try:
        # 打开文件
        with open(node_config_param_file,"r",encoding="utf-8") as f:
            data = f.read()
            result = yaml.load(data)
            return result
    except:
        print("read node config info fail.")
 
def remote_local_select(name):
    
    if not os.path.exists(node_config_param_file):
         print("Started local machine because config file not exist.")
         return (0, 1)
    else:
        data = read_yaml_all()
        
        print(name)

        if data["flag"]=="False":
            print("Started local machine because flag is set to false.")
            return (0, 1)
        

        tmp = []
        exec_name_list = list(data["remote_info"].values())
        for x in exec_name_list:
            tmp = tmp + x
        
        if name.startswith('/perception/object_recognition/detection/object_association_merger'):
            name = '/perception/object_recognition/detection/object_association_merger'

        if name not in tmp:
            return (0,1)
        if data["flag"]=="True":
            
            # info = {'ip': '192.168.1.3', 'username':'root'}
            # if name!='/rviz2':
            #     remote = SshMachine(hostname="inspur", env='source ' + data['env'], info=info)
            #     return (1, remote)

            for k, v in data["remote_info"].items():
                if name in v:
                    info = {'ip': k.split("@")[1], 'username':k.split("@")[0]}
                    if 'env' in data.keys():
                        remote = SshMachine(hostname="inspur", env='source ' + data['env'], info=info)
                    else:
                        remote = SshMachine(hostname="inspur", env=None, info=info)
                    return (1, remote) 
