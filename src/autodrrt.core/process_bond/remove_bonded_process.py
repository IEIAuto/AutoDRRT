import os
import subprocess
cpu_cores = [0,1,2,3,4,5,6,7,8,9,10,11]
base_directory = "/sys/fs/cgroup/cpuset/autoware_test"
processed_pids = {}
for cpu in  cpu_cores:
    directory_name = os.path.join(base_directory, "cpu_{}".format(cpu))
    subprocess.getoutput("rmdir {}/*".format(directory_name))
    subprocess.getoutput("rmdir {}".format(directory_name))