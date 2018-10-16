# spmf-mnar
This is our source code for the paper: <br>
Jiawei Chen, Can Wang, Martin Ester, Qihao Shi, Yan Feng, and Chun Chen. Social Recommendation with Missing Not at Random Data. 2018 IEEE 18th International Conference on data mining.

# Example to run the code.
We implement the SPMF-MNAR in MATLAB. Also, we implement some important function in C++ to speed up inference. So, beforing running the code, please compile c++ source codes to generate mex file in matlab enviornment:
```matlab
mex myv2c.cpp
mex myv2s.cpp
```

Then, we can run the code for the exampledata:
```matlab
spmfmnar('exampledata/train.txt','exampledata/test.txt','exampledata/trustnetwork.txt')
```
Where the inputs of the spmfmnar function are the paths of the trainning data, the test data and the trustnetwork respectively.<br>
Each line of train.txt is: UserID \t ItemID \t ratingvalue <br>
Each line of test.txt is :UserID \t ItemID \t ratingvalue <br>
Each line of trustnetwork.txt is: User1ID \t User2ID \t strength(1) <br>



