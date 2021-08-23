from setuptools import setup

#def readme():
#    with open('README.md') as f:
#        return f.read()

setup(name='HMOBSTER',
      version='0.0.38',
      description='VAF clustering for multiple karyotypes',
      url='https://github.com/Militeee/hmobster',
      author='Salvatore Milite',
      author_email='militesalvatore@gmail.com',
      license='GPL-3.0',
      packages=['mobster'],
      install_requires=[
            'matplotlib>=3.1',
            'pandas>=1.0',
            'pyro-ppl>=0.4',
            'numpy>=1.18',
            'scikit-learn',
            'seaborn'

      ],
      include_package_data=True,
      #long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      keywords='DNA sequencing clonal_evolution VAF',
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)