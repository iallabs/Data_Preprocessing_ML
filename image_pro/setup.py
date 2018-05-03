from setuptools import setup, find_packages


setup(
name='image_pro',
version='0.0.1',
description='Image processing and save into Tfrecords',
author='',
author_email='hilalyamine@gmail.com/alaa.el.bouchti@gmail.com',
license='Open',
packages=find_packages(include=['image_pro','image_pro.*','tfRecord', 'utils_data']),
packages_dir={'tfrecord':'tfRecord',
              'tfrecord.tfrecord':'tfRecord/tfrecord.py',
              'tfrecord.decode':'tfRecord/decode.py',
              'utils': 'utils_data',
              'utils.colors': 'utils_data/colors',
              'utils.colors.colors': 'utils_data/colors/colors.py',
              'utils.contours':'utils_data/contours',
              'utils.contours.contours':'utils_data/contours/contours.py',
              'utils.geotransf':'utils_data/geotransf',
              'utils.geotransf.geotransf':'utils_data/geotransf/geotransf.py',
              'utils.norma':'utils_data/norma',
              'utils.norma.norma':'utils_data/contours/norma.py',
              'utils.visualize':'utils_data/contours',
              'utils.visualize.visu':'utils_data/visualize/visu.py'            
},
scripts = ['parameters.py']
)