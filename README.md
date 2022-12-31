# AIproj_humordetector
Install argilla -> https://docs.argilla.io/en/latest/getting_started/installation/installation.html
Imports pandas, sklearn, matplot, modAL, argilla, csv

code.py is the script to manually building and label a dataset from scratch, using active learning query techniques.
Run python -m argilla in one shell.
After that you should open a python shell. Run the code until line 51 (the start of the loop).

Then,
(1) run line 51 to 76 . The learner will query the samples using max uncertainty. Then you should go to argilla dataset 
localhost:6900/datasets. If it asks for a password, the user is argilla and password is 1234
Then you should enter the dataset and see the samples. Click on hand labeling (see the section option on Mode on the right side of the computer) 
and choose the label for each sample. To just see the unlabeled samples click on status and filter default.

Then (2)continue the loop from running line 80 to 97 to teach the classifier with the annotated samples. 
Repeat (1) and (2) for the number of iterations you wish.
Finally (3) run line 101 to 109 to plot the accuracy evolution through iteration.

code_plot.py is the code to plot the differences in the accuracy evolution by using different query strategys. It is just need to run it on the shell.
