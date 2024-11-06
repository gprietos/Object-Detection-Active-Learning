from setuptools import setup, find_packages


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='active_learning_object_detection',
    version='1.0.0',
    author="Guillermo Prieto Sánchz",
    author_email="gprietos@indra.es",
    packages=find_packages(),
    python_requires='>=3.8,<3.11',
    install_requires=get_requirements(),
)