# 客户潜在价值预测模型
参考文献：[*Predict customer wallet without survey data*](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwjB7__3nOjMAhWKmZQKHWqRAxQQFggiMAE&url=https%3a%2f%2flirias%2ekuleuven%2ebe%2fbitstream%2f123456789%2f200989%2f1%2fSOW%2epdf&usg=AFQjCNHicZ1ITI6oBLIk7RvyfJHzB-o-DA&sig2=vm0xI_YEyKhw-W6btlO0SA&bvm=bv.122129774,bs.1,d.c2I&cad=rjt)

## 背景
一般情况下，一家商户只能提供一个顾客所需的部分产品，也就是说顾客会在多家同类型的店进行消费。对商家而言，判断顾客的消费能力（size of wallet，SoW）--顾客对此类店能投入的金钱总额与客户的忠诚度（share of wallet， ShoW）--顾客在这家店消费的金钱的占比这两项关键的指标对顾客进行划分从而挖掘高潜在价值客户对商铺未来一段时间内的收益增长有着十分重要的意义。在一般情况下这两项指标在没有具体问卷调查数据的情况下是难以获取或评估的。这里我们通过广义二项分布模型通过对直观的交易数据（R,F,M）进行建模分析，实现了对这两项指标的评估。

## 方法学步骤
**算法**<br>
1. 估计顾客i的交易总次数：ni
我们假设所有用户的交易次数符合带有客户个性化参数λi的泊松分布模型。<br>
![1](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq1.png)<br>
λi可以利用已知的RFM交易数据通vi过回归分析获得:<br>
![2](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq2.png)<br>
我们认为SoW与用户消费的总金额正相关而与消费的总时间长度负相关，因此在数据预处理的时候我们将这消费时长参数的数据先做倒数转换然后再缩放到0-1区间，然后再投入模型计算。<br>

2. 估计客户在固定商家的交易次数：xi
我们假设所有用户在所有店中在某家店消费的概率是πi,因此在这家店消费次数（xi）符合二项分布。 <br>
![3](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq3.png)<br>
πi 可以利用RFM数据wi通过逻辑斯蒂回归模型分析获得。我们认为ShoW与在这家店的平均交易间隔和距离上次交易的时间负相关，因此在数据预处理的时候我们将这两个参数的数据先做倒数转换然后再缩放到0-1区间，然后再投入模型计算。<br>
![4](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq4.png)<br>
在估算出α之后，πi可以通过以下等式计算: <br>
![5](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq5.png)

3. 最大似然法估计参数（Maximum likelihood(ML)）α和β
基于α和β的最大似然公式如下<br>
![6](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq6.png)<br>
结合之前的等式1和3: <br>
![7](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq7.png)<br>
结合之前的等式2和4并且用mi=ni-xi替换等式,几次转换以后可以得到如下等式: <br>
![8](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq8.png)<br>
具体而言在这次分析中我们用gamma分布替换公式的残差部分得到如下公式，其中N是所有客户的总数.<br>
![9](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq11.png)

4. 最大似然方程的基于α和β梯度函数
基于β求导: <br>
![12](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq12.png)<br>
结合等式2可化为:<br>
![13](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq13.png)<br>
基于α求导: <br>
![14](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq14.png)<br>
结合等式5化为: <br>
![15](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq15.png)

5. 估计总交易次数ni
对于每个用户i , ni >= xi。 <br>
![10](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq9.png)<br>
等价于: <br>
![11](https://github.com/CasiaFan/customer-potential-value-prediction-/blob/master/equation-pic/eq10.png)<br>
F(x,λ)是参数为λ的泊松分布的累积分布函数在x时候的值，客户的潜在消费能力为ni-xi

**输入数据**<br>
我们将1/消费交易时间长度和总消费金额缩放到（0，1）后估计β，从而获得SoW；将1/平均交易间隔和1/距离上次消费时间缩放到（0，1）后估计α，从而获得ShoW。结果我们发现前者主要和消费金额相关而后者主要和平均交易间隔相关。这里我们使用的测试数据为消费次数大于4的所有顾客的数据（见example.csv）<br>

**python脚本**<br>
我们的预测程序是基于python写的，里面主要用到了scipy, numpy, pandas和scikit-learn包。其中一个关键函数为scipy.optimize.minimize() ，具体参数设置参考[这里](http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.minimize.html)。测试中，我们用的method参数为‘TNC’
