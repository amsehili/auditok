import re
import ast
from setuptools import setup


_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('auditok/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))


setup(
    name='auditok',
    version=version,
    url='http://github.com/amsehili/auditok/',
    license='GNU General Public License v3 (GPLv3)',
    author='Amine Sehili',
    author_email='amine.sehili@gmail.com',
    description='A module for Audio/Acoustic Activity Detection',
    long_description= open('quickstart.rst').read().decode('utf-8'),
    packages=['auditok'],
    include_package_data=True,
    package_data={'auditok': ['data/*']},

    #data_files=[(['README.md', 'quickstart.rst', 'LICENSE', 'INSTALL', 'CHANGELOG']),
    #            ('share/doc/pdoc', ['doc/pdoc/index.html']),
    #           ],

    zip_safe=False,
    platforms='ANY',
    provides=['auditok'],
    requires=['PyAudio'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],

)
