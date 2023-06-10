# Device price prediction tool

This application has been developed in the context of a bachelor’s thesis for the Computer Engineering degree in FIB (Facultad d'informàtica de Barcelona) from UPC (Universitat Politècnica de Catalunya).

It provides price estimation for desktop, laptops, smartphones and tablets. It is optimized for devices older than 2020, though it can also work with newer devices.

Data was collected from [Backmarket](https://www.backmarket.es/es-es). Model has already been trained and is provided as a dump with [Joblib](https://joblib.readthedocs.io/en/latest/). We do also provide datasets as .xlsx files for other approaches or research.

The application consists of a Python script that uses models on [Sklearn](https://scikit-learn.org/stable/) to obtain predictions for a given device.

## Installation
### Data prediction tool

The following command will install all dependencies needed for running the application
```bash
pip install -r requirements.txt
```

## Usage
Just by executing the Python script, the API run running on localhost.

```bash
python3 main.py
```

By default, the application will be running on 0.0.0.0:8000

It is recommended to go to 0.0.0.0:8000/docs once the application is running to use it easily.

### Web frontend for using the API based on React
We have also developed a [frontend](https://github.com/Kerbolerr/TFGDavid23Web) for using the application.
Specifically, since the project has been done on collaboration with [e-reuse](https://www.ereuse.org/), this web application automatically reads device reports generated with e-reuse tools (i.e. Workbench). [Example files](https://github.com/eReuse/devicehub-teal/blob/master/ereuse_devicehub/dummy/files/asus-eee-1000h.snapshot.11.yaml)

##
David Gálvez Alcántara

2023
