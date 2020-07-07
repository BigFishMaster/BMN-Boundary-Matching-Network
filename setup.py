from setuptools import setup, find_packages

setup(
    name='bmn',
    description="BMN: Video action localization",
    version='0.1.0',
    zip_safe=False,
    python_requires='>=3.6',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'bmn_runner=bmn.runner:main',
        ],
    },
)
