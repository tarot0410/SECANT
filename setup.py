from setuptools import setup

setup(name='SECANT',
      version='0.1',
      description='biology-guided SEmi-supervised method for Clustering, classification, and ANnoTation of single-cell multi-omics',
      url='https://github.com/tarot0410/SECANT',
      author='Xinjun Wang',
      author_email='xiw119@pitt.edu',
      license='MIT',
      packages=['SECANT'],
      install_requires=[
          'pandas','numpy','torch','sklearn'
      ],
      zip_safe=False)
