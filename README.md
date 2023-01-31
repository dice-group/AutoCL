# AutoCL

AutoCL is one AutoML pipeline for concept learning including feature selection and hyperaparameter optimization

- [Installation](#installation)

# Installation

### Installation AutoCL from source

```shell
git clone https://github.com/dice-group/Ontolearn.git](https://github.com/AutoCL2023/AutoCL.git
cd intolearn
conda create --name temp python=3.8
conda activate temp
conda env update --name temp
python -c 'from setuptools import setup; setup()' develop
python -c "import ontolearn"
python -m pytest tests # Partial test with pytest
tox  # full test with tox

```
### Download external files (.link files)

Some resources like pre-calculated embeddings or `pre_trained_agents`
are not included in the Git repository directly. Use the following
command to download them from our data server.

```shell
./big_gitext/download_big.sh examples/pre_trained_agents.zip.link
./big_gitext/download_big.sh -A  # to download them all into examples folder
```

To update or upload resource files, follow the instructions
[here](https://github.com/dice-group/Ontolearn-internal/wiki/Upload-big-data-to-hobbitdata)
and use the following command.

```shell
./big_gitext/upload_big.sh pre_trained_agents.zip
```

### Building (sdist and bdist_wheel)

```shell
tox -e build
```

#### Building the docs

```shell
tox -e docs
```




## Contribution
Feel free to create a pull request

### Simple Linting

Run
```shell
tox -e lint --
```

This will run [flake8](https://flake8.pycqa.org/) on the source code.

For any further questions, please contact:  ```onto-learn@lists.uni-paderborn.de```
