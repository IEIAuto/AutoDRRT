import os
from glob import glob
from setuptools import setup

package_name = 'ssh_machine'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('launch/*.xml'))
    ],
    install_requires=['setuptools', 'launch', 'asyncssh'],
    zip_safe=True,
    maintainer='P. J. Reed',
    maintainer_email='preed@swri.org',
    description='Machine for launching nodes over SSH',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
