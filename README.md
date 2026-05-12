# gradient descent ML optimizer (C/python with ctypes)

## overview
gradient descent is used in ML to iteratively estimate the line of best fit for a given dataset

the linear reg traditional method is to use ordinary least squares (OLS) to minimize the squared distance of the datapoints to the line

alternatively run reg to have c calculate the gradients

## start

python files work normally by running main

c gradient calculation

```
# compile with
gcc -shared -fPIC -o gradients.so gradients.c

# then run reg.py, make sure the filename is right
```

## motivation

+ learning project for understanding pytorch operations
+ mini version of geogebra or loggerpro LBF features