# Device price prediction tool

This application has been developed in the context of a bachelor’s thesis for the Computer Engineering degree in FIB (Facultad d'informàtica de Barcelona) from UPC (Universitat Politècnica de Catalunya).

It provides price estimation for desktop, laptops, smartphones and tablets.

Data was collected from [Backmarket](https://www.backmarket.es/es-es). Model has already been trained and is provided as a dump with [Joblib](https://joblib.readthedocs.io/en/latest/). However, cleaned datasets for devices are provided as .xlsx files.

The application consists of a Python script that uses models on [Sklearn](https://scikit-learn.org/stable/) to obtain prediction for a given device.
The repository do also include a React frontend to use the API easily and that implements other features.

Specifically, since the project has been done on collaboration with [e-reuse](https://www.ereuse.org/), the web application automatically reads device reports generated with e-reuse tools (i.e. Workbench).

## Installation
### Data prediction tool

```bash
pip install -r requirements.txt
```

### Web frontend for using the API


## Usage
### Data prediction tool

Just by executing the Python script, the API will start running on localhost.

```bash
python3 main.py
```

By default, the application will be running on 0.0.0.0:8000

### Web frontend for using the API
npm start on the project's folder will make the web application start executing on the browser.

##
David Gálvez Alcántara 2023
