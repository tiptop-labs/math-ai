from setuptools import find_packages, setup

setup(
    name = "mult999",
    version = "0.1",
    author = "TipTop Labs",
    author_email = "office@tiptop-labs.com",
    url = "https://github.com/tiptop-labs/math-ai",
    install_requires = ["google-cloud-storage"],
    packages = find_packages())
