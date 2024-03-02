# MultiLabel Binarizer
The current objectives of this study is:
<li>Understand better multilabel problems and it's importance for model creation;
<li>Understand how multilabel enchances feature engineering;
<li>Understand how scikit-learn's MultiLabelBinarizer works;
<li>Implement it on a fictional database;

As a example, a store can classify a certain product with several labels:
<li>smartphone: electronic, portable, communication
<li>notebook: electronic; portable, computer
<li>bicycle: sports, transportation, outdoor

Storing several labels on one single feature can be problematic. There are some encoding methods that can be used when dealing with one single categoric variable (One Hot Encoding, Ordinal Encoding). But What about multiple labels? For that: <b>MultiLabelBinarizer</b> will do the job (in other works, it's a One Hot Encoder with steroids.)

