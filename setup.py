from setuptools import setup, find_packages

setup(name='proj',
      packages=[package for package in find_packages()
                if package.startswith('proj')],
      install_requires=[
          'gym[atari,box2d,classic_control]',
          'numpy',
          'scipy',
          'python-dateutil',
          'tqdm',
          'flask',
          'plotly',
          'tblib',
          'click',
          'cloudpickle',
          'torch',
          'torchvision',
          'sacred', # temporary
          'pymongo', # temporary
          'opencv-python'
      ],
)
