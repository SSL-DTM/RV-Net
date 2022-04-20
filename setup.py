from setuptools import setup

setup(
    name='encdec',
    version='0.1.0',
    author='Bashir Kazimi',
    author_email='kazimibashir907@gmail.com',
    packages=['encdec', 'encdec.test'],
    url='http://pypi.python.org/pypi/encdec/',
    license='LICENSE.txt',
    description='A Pytorch encoder-decoder Framework',
    long_description=open('README.md').read(),
    install_requires=[
    ],
)
