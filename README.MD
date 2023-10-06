# ABCE
run code from console for imports to be resolved properly
````
python -m main
````

GIT
===
To initialize a git repo, create it on github, swith to the direcory the project ought to be saved locally. Then copy the https link and run 
````
>>> git clone <url>
````

To tell git that we want to track changes made to a file called hello.py add it:

````
>>> git add hello.py
```` 

To save the changes made to the file hello.py commit it:
````
>>> git commit -m "message"
````

To add and commit everything simultaniously run
````
>>> git commit -am "message"
````

To create a new branch
````
>>> git checkout -b branch1
````

To merge the branch, "branch1" into main, first switch to main and then run:
````
>>> git merge "branch1"
````

To switch back to main run
````
>>> git checkout main
````

Push main to github
````
git push -u origin <branch>
````

An additional handy command is git status which reports which files are tracked etc. 
````
>>> git status
````

## Activate environment
To activate virtual environment run:
````
python -m venv venv
. venv\scripts\activate
````
Then pip install all libraries. For instance to virtuall install 'Flask' run:
````
pip install Fask
````
To install all requirements run:
````
pip install -r requirements.txt
````
## Run flask application
In order to run the application move to the 'flaskr' directory and type:
````
flask run
````
This automatically regocnizes the application since it is called 'app.py'. Follow the provided link and the html opens.
To run the application in the debug mode and enable f.i. auto-updating open powershell terminal and type:
````
$env:FLASK_ENV = "development"
flask run
````
To use debug mode:
````
flask --app app.py --debug run
````

## Miscilanious
To execute the unittests without printing anything to the console use the complier flag- b
````
python -m Test -b
````