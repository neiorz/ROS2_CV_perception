from setuptools import find_packages, setup

package_name = 'yaqz_vision_integration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nourz',
    maintainer_email='nourz@todo.todo',
    description='Yaqz Security Robot - Vision Integration Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = yaqz_vision_integration.vision_node:main'
        ],
    },
)