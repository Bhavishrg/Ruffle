## "Ruffle: Rapid 3-Party Shuffle Protocols"
Implementation of paper title "Ruffle: Rapid 3-Party Shuffle Protocols"

## WARNING: This is not production-ready code.

This is software for a research prototype. Please
do *NOT* use this code in production.


## Getting started

First, make sure that you have a working Python installation:
This code was tested in Python 3.7.12

```
$ python3 --version   
Python 3.7.12
$ pip --version
pip 22.2.1
```

Now run the following steps to install the necessary libraries:

```
$cd Ruffle/cryptenlocal
$ pip install -r requirements.txt" 
```
You should now be set to run the code. run the following command:


## Anonymous broadcast
To test Anonymous Broadcast system of Ruffle. run the following code:
```
$ cd SwiftMPC
$ python3 run.py -n 10000 -m 2
```
To print help: 
```
$ python3 run.py -h
```
To test Anonymous Broadcast system of Clarion. run the following code:
```
$ cd clarion
$ python3 run.py -n 10000 -m 2
```
To print help: 
```
$ python3 run.py -h
```

Parameters:
-n, --numclients: Number of clients
-m, --msgblocks: number of blocks of messages of size 8 Bytes