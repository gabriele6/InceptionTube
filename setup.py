from setuptools import setup

setup(name='inceptiontube',
      version='0.1',
      description='Using InceptionV3 to scrape and classify youtube videos',
      url='http://github.com/gabriele6/InceptionTube',
      author='Gabriele Tenucci',
      author_email='tenucci.gabriele@gmail.com',
      license='MIT',
      packages=['inceptiontube'],
      zip_safe=False, install_requires=['youtube_dl', 'keras', 'tensorflow', 'numpy', 'opencv-python', 'Pillow'])