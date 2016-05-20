# Customer potential value prediction using Generalized Binomial Model
**The idea for this analysis is from Nicolas Glady's paper and we will show appreciation to his work.** <br>
Reference:[*Predict customer wallet without survey data*](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwjB7__3nOjMAhWKmZQKHWqRAxQQFggiMAE&url=https%3a%2f%2flirias%2ekuleuven%2ebe%2fbitstream%2f123456789%2f200989%2f1%2fSOW%2epdf&usg=AFQjCNHicZ1ITI6oBLIk7RvyfJHzB-o-DA&sig2=vm0xI_YEyKhw-W6btlO0SA&bvm=bv.122129774,bs.1,d.c2I&cad=rjt)

## Introduction
Normally, a company only provides part of total volume of products required by a customer. Size of wallet (total volume of transaction made by a customer) and share of wallet (percentage of transactions of a customer made with a focal company) are 2 important clues to predict a customer's potential value in the future business with the company. Then we derive a concept called potential of wallet to indicate a customer's consumption potential using the discrepancy between the size of wallet and actual business volume with the focal company. For instance, those who have few transaction with the company but with high potential value could be the main growth point in later sales.

For a customer's size of wallet and share of wallet are generally hard to obtain or observe. These 2 factors are normally explored by expensive time-consuming directory survey. Here we describe a protocol to predict them using only easily obtained transaction data - the RFM (recency, frequency and monetary) variables. Of course, the accuracy is relatively low compared with conventional method.

## Methodology
We used Generalized Binomial Model (GBM) models the total number of transactions a customer makes.

**algorithm**<br>
1. Total number of transactions customer i makes: ni
We assume the number of transactions of all customers should follow a Poisson distribution with individual parameter λi. <br>
![1](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq1.png)<br>
λi could be modeled using observable predictor RFM variables vi, so we could estimate λi by this:<br>
![2](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq2.png)<br>
We think the size of wallet should be positively related to the monetary and negatively related to total time duration of the transactions. This idea will be used in data preprocessing: reciprocal conversion for total time duration variables. <br>

2. Number of transactions made by a customer at focal company xi
We assume customer i chooses the focal company wtih possibility πi when making a transaction. So the number of transactions made in the focal company (xi: observable) should follow a binomial distribution. <br>
![3](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq3.png)<br>
πi could be modeled by logistic regression model with RFM variables wi. We think the share of wallet is negatively related to average transaction intervals and recency at the focal company. <br>
![4](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq4.png)<br>
After α is estimated, πi follows as: <br>
![5](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq5.png)

3. Maximum likelihood(ML) estimate parameter α and β
THe likelihood of an individual observation equals <br>
![6](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq6.png)<br>
given equation 1 and 3: <br>
![7](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq7.png)<br>
Using equation 2 and 4 and substitute mi=ni-xi, after several transformations: <br>
![8](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq8.png)<br>
or more in detail where N is the total number of observations (customers).<br>
![9](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq11.png)

4. gradient function of log-likelihood function L to parameters α and β
For β: <br>
![12](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq12.png)<br>
From equation 2, it follows:<br>
![13](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq13.png)<br>
For α: <br>
![14](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq14.png)<br>
From equation 5, it follows: <br>
![15](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq15.png)

5. Total transaction count, ni
For each individual customer i , ni >= xi (easy to understand). So the ni follows: <br>
![10](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq9.png)<br>
It could be rewrite as: <br>
![11](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq10.png)<br>
where F(x,λ) is the cumulative distribution function of a Poisson distribution with
parameter λ at the value x. The Potential-of-Wallet is then predicted as ni-xi

**Input data**<br>
We use 1/time-duration and total monetary and then scale them to range 0 to 1 for estimating parameter beta or size of wallet; also use 1/average-transaction-interval and 1/recency and then scale them to range 0 to 1 for estimating parameter alpha or share of wallet. (Here we only use customers with > 4 transactions with focal company.) <br>
The example.csv file contains our input information.

**Script**<br>
The program is written in python with scipy, numpy, pandas and scikit-learn package. One key function is scipy.optimize.minimize() and detailed parameters could be found [here](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.minimize.html). We use the 'TNC' method.
