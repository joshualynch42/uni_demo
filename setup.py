from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="dobot",
    version="0.1.0",
    description="Dobot Interface",
    license="GPLv3",
    long_description=long_description,
    author="Joshua Lynch",
    author_email="josh.g.lynch@hotmail.com",
    url="https://github.com/joshualynch42/individual_project",
    packages=["cri", "cri.abb", "cri.ur", "cri.ur.rtde", "cri.dobot", "cri.dobot.magician", "cri.dobot.mg400", "vsp", "vsp.pygrabber"],
	package_data={"cri.ur": ['rtde_config.xml'],
            "cri.dobot.magician": ['DobotDll.dll',"msvcp120.dll","msvcr120.dll","Qt5Core.dll","Qt5Network.dll","Qt5SerialPort.dll"],
            "cri.dobot.mg400": ['Dobot.dll',"libgcc_s_seh-1.dll","libstdc++-6.dll","libwinpthread-1.dll","Qt5Core.dll","Qt5Network.dll","Qt5SerialPort.dll"]},
    install_requires=["numpy", "transforms3d", "scipy", "opencv-python>=3.4.5.20", "comtypes", "matplotlib",
 "pandas", "h5py", "gym", "keyboard", "pygame", "tensorflow==2.8.0", "keras==2.8.0", "keyboard"]
)
