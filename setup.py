from setuptools import setup

setup(
    name='gymnasium_snake',
    version='0.0.1',  # original gym_hybrid version='0.0.1'
    packages=['gymnasium_snake'],
    install_requires=['gymnasium>=1.0.0', 
                      'numpy',
                      'pygame>=2.3.0'
                      ],
)
