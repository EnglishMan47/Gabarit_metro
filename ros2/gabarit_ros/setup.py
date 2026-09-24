from glob import glob

from setuptools import setup

package_name = "gabarit_ros"

setup(
    name=package_name,
    version="2.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Gabarit obstacle detector ROS 2 node",
    license="MIT",
    entry_points={"console_scripts": ["gabarit_node = gabarit_ros.node:main"]},
)
