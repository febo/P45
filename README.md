# P45: A Python C4.5 implementation

This is an accurate (or as accurate as possible) re-implementation of the classic C4.5 decision tree algorithm. The implementation includes:
- separate-and-conquer recursive partitioning method
- error-based pruning
- copes with both numeric and categorial attributes directly
- support for missing values

Current assumptions (when using the `build_decision_tree` function directly):
- class attribute is the last attribute of the dataframe
- data types on the dataframe are correctly set for each column: `str` for categoritcal attributes and `float` for numeric attributes

## Running P45

There are two options to run the algorihtm, either directly from the command-line or imported as a library.

> You may need to install additional dependencies to be able to run the algorithm &mdash; easiest way
> to do that is running the following in your terminal:
>
> ```bash
> pip install -r requirements.txt
> ```

### Command-line

The command-line allows you to run the algorithm and provide its parameters. The complete set of options can be viewed by running:

```bash
python p45.py
```

This will display a list of commands and their short description:

```
usage: p45.py [-h] [-m cases] [--seed seed] [--unpruned] [--csv] [-t <test file>] <training file>

P45: A Python C4.5 implementation.

positional arguments:
  <training file>  training file

optional arguments:
  -h, --help       show this help message and exit
  -m cases         minimum number of cases for at least two branches of a split
  --seed seed      random seed value
  --unpruned       disables pruning
  --csv            reads input as a CSV file
  -t <test file>   test file
```

The minimum set of parameters is to specify the training file:

```bash
python p45.py iris.arff
```

After a successful run, the algorithm produces a decision tree:

```
P45 [Release 1.0]                                      Thu April 7 06:00:00 2022
-----------------

    Options:
        Pruning=True
        Cases=2
        Seed=0

Class specified by attribute 'Class'

Read 150 cases (4 predictor attributes) from:
    -> iris.arff

Decision tree:

petal-width <= 0.6: Iris-setosa (50.0)
petal-width > 0.6: 
|    petal-width > 1.7: Iris-virginica (46.0/1.0)
|    petal-width <= 1.7: 
|    |    petal-length <= 4.9: Iris-versicolor (48.0/1.0)
|    |    petal-length > 4.9: 
|    |    |    petal-width <= 1.5: Iris-virginica (3.0)
|    |    |    petal-width > 1.5: Iris-versicolor (3.0/1.0)

Time: 2.7 secs
```

### Import

You can also add `P45` as a dependency to your project:

```python
import p45
```

Then you can you can use the `build_decision_tree` function to create a decision tree. The return of this function is a `Node` object that represents the root node of the tree. From the root node, you can classify new instances by calling the `predict` or `probabilities` functions.

### Limitations

Currently the code can take a while to run when using large datasets. The most likely reason for this is the (over-)use of dataframe operations to split the data during the recursive tree creation procedure.
