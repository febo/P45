# P45: A Python C4.5 implementation

This is an accurate (or as accurate as possible) re-implementation of the classic C4.5 decision tree algorithm. The implementation includes:
- separate-and-conquer recursive partitioning method
- error-based pruning
- copes with both numeric and categorial attributes directly
- support for missing values

Current assumptions:
- class attribute is the last attribute of the dataframe
- data types on the dataframe are correctly set for each column: `str` for categoritcal attributes and `float` for numeric attributes
