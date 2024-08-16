from setuptools import setup, find_packages

setup(
  name='VJModels',  # How you named your package folder (MyLib)
  packages=find_packages(),  # Automatically find all sub-packages
  version='1.0.0',  # Start with a small number and increase it with every change you make
  license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='My personal machine learning models',  # Give a short description about your library
  author='Vanderval Borges de Souza Junior',  # Type in your name
  author_email='vander31bs@gmail.com',  # Type in your E-Mail
  url='https://github.com/Vanderval31bs/VJModels',  # Provide either the link to your github or to your website
  download_url='https://github.com/Vanderval31bs/VJModels/archive/refs/tags/v1.0.0-alpha.tar.gz',  # I explain this later on
  keywords=['MachineLearning', 'Models', 'Forests'],  # Keywords that define your package best
  install_requires=[  # I get to this in a second
      'pandas',
      'scikit-learn',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',  # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  # Again, pick a license
    'Programming Language :: Python :: 3.11',  # Specify which python versions that you want to support
  ],
)