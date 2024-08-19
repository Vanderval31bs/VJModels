from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  name='VJModels',
  packages=find_packages(),  
  version='2.0.0',
  license='MIT',  
  description='My personal machine learning models',
  long_description=long_description,
  long_description_content_type="text/markdown", 
  author='Vanderval Borges de Souza Junior',  
  author_email='vander31bs@gmail.com',  
  url='https://github.com/Vanderval31bs/VJModels',  
  download_url='https://github.com/Vanderval31bs/VJModels/archive/refs/tags/v2.0.0-alpha.tar.gz',  
  keywords=['MachineLearning', 'Models', 'Forests'],
  install_requires=[
      'pandas',
      'scikit-learn',
      "statsmodels",
      "statstests",
      "scipy"
  ],
  classifiers=[
    'Development Status :: 3 - Alpha', 
    'Intended Audience :: Developers',  
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.11', 
  ],
)