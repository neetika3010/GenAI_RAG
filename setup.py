from setuptools import find_packages,setup

setup(
    name="Ecommerce", 
    version="0.1.0", 
    author="Neetika", 
    author_email="neetikashree@gmail.com", 
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain','langchain-openai','datasets','pypdf','python-dotenv','flask']
)