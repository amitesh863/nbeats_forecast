
from distutils.core import setup
setup(
  name = 'nbeats_forecast',        
  packages = ['nbeats_forecast'],   
  version = '1.0',     
  license='MIT',        
  description = 'TYPE YOUR DESCRIPTION HERE',   
  author = 'Amitesh Sharma',                   
  author_email = 'amitesh863@gmail.com',     
  url = 'https://github.com/amitesh863/nbeats',   
  download_url = 'https://github.com/amitesh863/nbeats_forecast/archive/v_10.tar.gz',    
  keywords = ['nbeats', 'timeseries', 'forecast', 'nueral beats' , 'univariate timeseries forecast' , 'timeseries forecast', 'univariate timeseries forecast'],   
  install_requires=[
          'nbeats-pytorch',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Data Scientists',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
  ],
)
