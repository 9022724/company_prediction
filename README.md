# Overview

## Installation

## Setup

* If you wish to run your own build, first ensure you have python globally installed in your computer. If not, you can
  get python [here](https://www.python.org").
* After doing this, confirm that you have installed virtualenv globally as well. If not, run this:
    ```bash
        $ pip install virtualenv
    ```


* #### Dependencies
    1. Cd into your the cloned repo as such:
        ```bash
            $ cd project
        ```
    2. Create and fire up your virtual environment:
        ```bash
            $ virtualenv  venv -p python3
            $ source venv/bin/activate
        ```
    3. Install the dependencies needed to run the app:
        ```bash
            $ pip install -r requirements.txt
        ```

    
* #### Run It
  Fire up the server using this one simple command:
    ```bash
        $ uvicorn main:app --reload
    ```
  You can now access the file api service on your browser by using
    ```
        http://localhost:8000/
    ```