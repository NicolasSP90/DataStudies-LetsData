# PySpark

# Main Goals
Basic understanding of PySpark;
Understand it's advantages;
Understand it's API and characteristics;

# Concept
Spark is a Apache project, to help and facilitate the usage of big data in clusters, different from Polars, DuckDB and Pandas that were created to be used in one machine. Spark sees clusters of computers as if they were one central device (But can also be used to work in one single machine).

It is created in Scala. It's API supports several programming languages.

It is not meant to be used locally. Usually, the user connects to one ot its clusters to work.

Uses SQL language for querys.

Supports: batch of data, real time streaming, machine learning and graphs.

It is a viable solution when the amount of data is too big.

The working process is similar to Pandas, Polars and DuckDB - using dataframes. That been said, it is based on RDDs (Resilient Distributed Datasets), that are basically databases distributed into several clusters.