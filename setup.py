from setuptools import setup, find_packages

setup(
    name='gpusync',
    version='1.0.0',
    description='A High-Performance Out-of-Core Async Matrix Math Library for OpenCL & CUDA via AST Transpilation.',
    author='GalTechDev',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyopencl',
        # 'cupy-cuda12x', # Requis pour le backend CUDA. Dé-commentez ou installez manuellement selon votre version CUDA.
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: System :: Hardware :: Symmetric Multi-processing',
    ],
    python_requires='>=3.8',
)
