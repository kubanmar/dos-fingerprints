import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nomadDOSfingerprints",
    version="0.1",
    author="Martin Kuban",
    author_email="kuban@physik.hu-berlin.de",
    description="An implementation of DOS fingerprints for the NOMAD Laboratory.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.physik.hu-berlin.de/kuban/nomad-dos-fingerprints",
    install_requires = ['numpy'],
    packages=['nomad_dos_fingerprints']
)
