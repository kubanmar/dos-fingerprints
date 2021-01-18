import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nomad_dos_fingerprints",
    version="1.0",
    author="Martin Kuban",
    author_email="kuban@physik.hu-berlin.de",
    description="An implementation of DOS fingerprints for the NOMAD Laboratory.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.mpcdf.mpg.de/nomad-lab/nomad-dos-fingerprints",
    install_requires=['numpy', 'bitarray'],
    packages=['nomad_dos_fingerprints']
)
