## Single arm motion specification

### Installation instructions

Install SOEM as a system library
```
cd ~/
git clone https://github.com/OpenEtherCATsociety/SOEM.git
cd SOEM

mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=On -DCMAKE_INSTALL_PREFIX=/usr/local/ ..
sudo make install
```

Install following packages
```
sudo apt install ros-${ROS_DISTRO}-tf2-ros
sudo apt install ros-${ROS_DISTRO}-tf2-kdl
```

Clone kinova mediator and motion specification action server repositories in your workspace
```
cd <workspace path>/src/
git clone https://github.com/RoboticsCosmos/kinova_mediator.git
git clone https://github.com/secorolab/freddy_single_arm_control.git

cd <workspace path>/
colcon build
```

Note to developer: changes to the kinova mediator is yet to be commited
