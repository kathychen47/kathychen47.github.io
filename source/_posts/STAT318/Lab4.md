Good morning everyone, and welcome back to Lab class! I hope you all had a great break and are feeling refreshed and ready to dive into today's lesson. Today, we'll be covering the validation set approach, k-fold cross validation and Bootstrap method. These three method are both used to evaluate performance of the model. 

I will introduce you the validation set approach first. For the validation set approach, the data is split into two parts: a training set and a validation set. The model is trained on the training set and the validation set is used to estimate its performance. 

 First of all, Let’s see what the dataset looks like, just like we do every time we get a new dataset. 

```R
library(ISLR)
set.seed(6678)
data(Auto)
attach(Auto)
dim(Auto)
names(Auto)
Auto[1:4,]
```

We will continue using Auto dataset. We load the library ISLR for the use of Auto dataset and ggplots library for ploting. We set the seed to ensure we can generate the same result each time we run these codes. we attach Auto dataset so that we can refer to the variables in this dataset without specifing them. Then we output the its dimention,variable names and the first 4 row of the dataset.  Hope you still remember these very basic codes. 



```R
train=sample(392,196)
```

The we split the dataset into training dataset and testing dataset using the sample function, where we create a random sample of size 196 as training dataset from a population of size 392. Which accounts for 50% of the total data. And the sample function will output the index instead of the value.



```r
plot(horsepower,mpg)
#plot training set data
points(horsepower[train],mpg[train],col="red")
```

We are still analyzing the statistical relationship between horsepower and mile per gallon, so we can draw a scatter plot to show our original data and the splited  training data.  As you can see, we create a plot with horsepower as the x-axis and mpg as y-axis, the total dataset point is shown as circle black and the training dataset is circle red ones. 



```r
# Define mean square error function
mse <- function(y, y_pred) {
  mean((y - y_pred)^2)
}
```

Before we fit the model, we can define the a function for caclulation of MSE so that we can use it easier. 



We will use linear regression model, quadratic regression and cubic regression model and compare their donation in testing Mean square error. 

```r
# Fit linear regression model and calculate mean squared error
lm.fit=lm(mpg~horsepower,data=Auto,subset=train)
MSE = mse(mpg[-train], predict(lm.fit,Auto)[-train])
MSE
abline(lm.fit, col='black')
```

We first conduct the linear regression model, we specify the subset equal to train for training. And use the mse function to calculate mean square error, as we want to calculate the test MSE to evaluate the effect of this model. So we need to use the test dataset of calculation. Thus, here, the negative sign is used to exclude the training index. so the rest index is the test dataset index. The output MSE is 25.2914, and we add the fitted curved to the plot with the color black. So this line is our linear regression fitted curve. 



```R
# Fit quadratic regression model and calculate mean squared error

lm.fit2=lm(mpg~poly(horsepower,2),data=Auto,subset=train)
MSE = mse(mpg[-train], predict(lm.fit2,Auto)[-train])
MSE
horselims = range(horsepower) #返回horsepower里的最大值和最小值
horsepower.grid = seq(from=horselims[1],to=horselims[2])# 创建等间隔的数列, horselims[1]是最小值
preds=predict(lm.fit2,newdata=list(horsepower=horsepower.grid)) 
lines(horsepower.grid,preds,col="green")
```

 Then, repeat the process, we use the ploy function to conduct the cubic regression model, we calculate the MSE which is 20.  We also want to plot the fitted curve for this model, but we cannot use the abline to draw curve, cus abine can only be used to draw straight line not the curves. So Alternatively, we create some data to plot this curve. We can produce a equally-spaced series, from the minimum values of the predator variable to the maximun values of the predator variable, so that the line we plot can be drawn from the leftmost end of the x coordinate in the image through all the data up to the rightmost end.  We use the pred function to predict the estimated value of mpg based one the preditor value we create. So that we can plot the fitted curve in this plot. 



```r
# Fit cubic regression model and calculate mean squared error
lm.fit3=lm(mpg~poly(horsepower,3),data=Auto,subset=train)

MSE = mse(mpg[-train], predict(lm.fit3,Auto)[-train])
MSE

preds=predict(lm.fit3,newdata=list(horsepower=horsepower.grid))
lines(horsepower.grid,preds,col="blue")
```

We conduct the cubic regression model with the same process, just change two to three. The MSE is 20.02832 and the blue line is our cubic regression line. 

So according to the validation set approach, we consider the cubic regression model is the best model can describe the realtionship of these two varivales becasue it has the lowest MSE. 

However, the validation set approach has some limitations. First, it can result in high variance because the performance estimate depends heavily on which samples end up in the validation set. Second, it can be inefficient because a large portion of the data is not used for training. In this case, the training dataset only accounts for 50 percent. And if we split the dataset in different way, for example, if we change the seed to other values, our best estimated model changed from quardictic model to cubic regression model. 





K-fold cross-validation is often preferred over the validation set approach because it provides a more reliable estimate of the model's performance on new, unseen data.

cross-validation addresses these limitations by dividing the data into K non-overlapping folds, where K is usually between 5 and 10. The model is trained on K-1 folds and tested on the remaining fold, and this process is repeated K times. Each fold is used exactly once for testing, and the performance estimates from each fold are averaged to obtain the final estimate. This approach reduces the variance in the performance estimate because each sample is used for both training and testing. The drawback of this method is very obvious, it can be more computationally expensive, especially for large datasets and complex models because it will be repeated K times. 

We can have a try of this approach. 

```{r}
library(boot)
set.seed(1)
cv.error.10=rep(0,10)
cv.error.10
for (i in 1:10){
  glm.fit=glm(mpg~poly(horsepower,i),data=Auto)     
  cv.error.10[i]=cv.glm(Auto,glm.fit,K=10)$delta[1] 
}
cv.error.10
plot(1:10,cv.error.10,type="b",col="red")
```

we want to conduct the 10-fold cross-validation, This line loads the `boot` package, which contains functions for performing bootstrap and cross-validation procedures. We set the seed for a random number and create a sequence with the length of 10, and their initial values for these sequence is zero, and we will update it with mean square error for each fold later.

This loop iterates from 1 to 10, and for each iteration it uisng the glm function to fit a polynomial regression model on the horsepower variables with of order ranging from 1 to 10. 

Then, the cv.glm function is used to perform 10-fold cross-validation and obtain the cross-validation error for each order, which is saved in cv.error.10 array. And there are two elements in delta paramater, the first one is the Mean square error, the second one is standard deviation of error.  When evaluating models, we typically use the mean error (such as mean squared error or mean absolute error) rather than the standard deviation of the error. 

This is because the mean error measures the average deviation between the predicted values and the actual values, while the standard deviation of the error measures the variability of the errors around their mean. The mean error provides a more straightforward and interpretable measure of model performance, as it tells us, on average, how far off the model's predictions are from the actual values. The standard deviation of the error, on the other hand, is useful for understanding the spread of the errors and the overall uncertainty of the model's predictions. So In this case, we want to store the Mean square error of each fold, so we specify the index of delta as 1. 

Finally, this line plots the cross-validation error estimates for each polynomial order. The `type = "b"` argument specifies that both points and lines should be plotted, while the `col = "red"` argument sets the color of the points and lines to red. The x-axis displays the polynomial order, while the y-axis shows the mean squared error (MSE) of the cross-validation error estimates. So from the plot, by using the k fold cross validation, we can say the model with polynomial order 7 fit the model the best and can explain the data the best. 





















