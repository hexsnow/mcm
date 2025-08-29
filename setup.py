from setuptools import setup, find_packages

setup(
    name='mcm',
    version='0.1.0',
    description='A Python project',
    author='',
    author_email='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'mcm = mcm.main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
