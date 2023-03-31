from setuptools import setup, find_packages


setup(
  name = 'cliff_pose',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.1',
  license='MIT',
  description = 'CLIFF',
  author = 'Pranoy R',
  author_email = 'pranoyalkr@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/pranoyr/cliff-pose-estimation',
  keywords = [
    'pose estimatiom',
  ],
  install_requires=[],
  classifiers=[
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
