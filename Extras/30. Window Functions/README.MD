# Basic Concepts
With basic SQL commands there are basically two types of results: all rows (with or without filters) or rows grouped.

If more information must be shown (the sum in all rows, the mean without groupping), Window Functions are needed!

The OVER function creates a "window" where calculations can be made.

The PARTITION BY is similar to GROUP BY, but it only returns a value. Using OVER + PARTITION, is possible to generate calculations and separate them in the table (or partition them ;))

RANK() creates a ranking that can be general or with group/partition

LAG returns the value from the previous row

LEAD returns the value from the following row