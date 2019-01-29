from setuptools import setup, find_packages

setup(name='proj',
      packages=[package for package in find_packages()
                if package.startswith('proj')],
      install_requires=[
          'gym',
          'numpy',
          'tqdm',
          'flask',
          'plotly',
          'click',
          'cloudpickle',
          'torch'
      ],
)
