from setuptools import setup, find_packages

setup(name='proj',
      packages=[package for package in find_packages()
                if package.startswith('proj')],
      install_requires=[
          'gym[atari,box2d,classic_control]',
          'numpy',
          'python-dateutil',
          'tqdm',
          'flask',
          'plotly',
          'click',
          'cloudpickle',
          'torch',
          'opencv-python'
      ],
)
