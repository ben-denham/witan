from setuptools import setup, find_packages
setup(
    name='witan',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List packages and versions you depend on.
        'ipytree==0.2.1',
        'networkx==2.3',
        'numpy==1.19.5',
        'orange3==3.29.3',
        'pandas==1.3.5',
        'plotly==5.3.1',
        'requests==2.24.0',
        'scikit-learn==0.24.2',
        'scipy==1.5.2',
        'sly==0.4',
        'snorkel==0.9.7',
        'tensorflow==2.6.0',
        'torch==1.9.0',
        'typing-extensions==3.7.4.3',
    ],
    extras_require={
        # Best practice to list non-essential dev dependencies here.
        'dev': [
            'flake8==3.7.9',
            'mypy==0.812',
            'pytest==5.2.2',
            'pytest-cov==2.8.1',
            # Typing stubs.
            'pandas-stubs==1.1.0.7',
        ]
    }
)
