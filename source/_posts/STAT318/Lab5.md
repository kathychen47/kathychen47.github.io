



Well, today, we are going to learn Decision Tress including classification tree and regression tree. I think you already know the principle of decision tree in class, so today we will learn how to implement them in code.



## Classification treee



```r
library(tree)
library(ISLR)
attach(Carseats)
```

We’re starting off by loading some libraries. Apart from the ISLR library we used before, we will introduce a new library called tree, which we can use to build and visualize the decision tree. And we are going to use a new dataset: Carseats. Amd we attach it into R. 



```R
str(Carseats)
```

First of all, we still need to get some information of this dataset, we print out the structure of this dataset, As it shown, Apart from Shelve Location, Urban and US, these three variables type are factor, others are all numeric variables.  



```R
High=ifelse(Sales<=8,"No","Yes") #no 0 yes 1
Carseats1=data.frame(Carseats,High)
Carseats1$High=as.factor(Carseats1$High)
str(Carseats1)
```

Sales is the most important variable because our target is to predict the Sales based on all the other variables. In the regression tree, we want to estimate the number of sales, but in classification tree, we should  turn this dataset into a classification problem first.  So we’re creating new variable called High, if the number of car seat sold is lower than or equal to 8, we label it as NO, Otherwise, we label it as Yes. Then, we wanna add this new variable in the dataset, so we create a new data frame called Carseats1 which include all the variables in the original dataset and this new variable High. As you can see, the type of the High variable is character, so we should use as. factor function to transform it to factor type first, so that it can be used for classification. 



```
tree.carseats=tree(High~.-Sales,Carseats1)
summary(tree.carseats)
```

After all these preparation, now, we can build a classification tree. The Tree function is used to build the classification tree, High is response variable, while other variables in the Carseats1 dataset except Sales are the predictor variables. We can print out a brief introduction of this model use the summary function. 

First of all, it’s a classification tree, these are the variables actually used in tree construction.  There are 27 terminal nodes in this tree, that is, the model divides the dataset into 27 different categories or subsets. 

Here is formula for calculate the residual mean deviance, in this formula, yi is .... So in this case, n is 400 cus there are 400 observations and T is 27. This value indicates that the model has a small average deviation between predicted and true values, indicating good model fit.  

The meaning of misclassification error is very obvious, 36 is the sample that we are false classified, this rate is only 0.09, which indicate a good model fit as well. 



```r
plot(tree.carseats)
text(tree.carseats,pretty=0,cex = 0.6)
```

Finally, we can visualize this tree by plotting the tree, and text() function is used to add the label in the plot, and, the `cex` parameter controls the size of the text labels. When your tree are very big, you can reduce the cex, so that the label will not overlap with each other. 

The last line `tree.carseats` just prints the fitted decision tree model. This will show all the detail in the tree plot. This is this format. 

The first element is the `node`, which is the node number, and if there are star signal at the end of it, it shows that this node is the terminal node. 

The second element is `split`, which shows the predictor variable and the splitting threshold to partition the date at the node. 

`n` is the number of observations that reach the node

and `deviance` is the evaluation metric, for classification tree, it usually the Gini index or cross-entropy. For regression tree, it’s usually the residual sum of squares. 

`yval` is the predicted class. 

`yprob` is the class probabilities at the node. 



```R
set.seed(1)
train=sample(1:nrow(Carseats1), 200)
Carseats.test=Carseats1[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats1,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(98+56)/200
```



Even though we seem to have a good result, but it might be overfitting, so we need to evaluate this model to see how well our decision tree actually predict variable High. So we can use the evaluation method that we have learned in the previous class, The validation set approach. We split the dataset into training dataset and validation set. We're setting a random seed for reproducibility, and then randomly selecting 200 rows from our dataset to use as our training data. We're then using the remaining rows as our testing data. We're fitting our decision tree using only the training data, and then using that tree to predict `High` for the testing data. The `table()` function helps us see how many of our predictions were correct. In this case, 98 + 56 of our predictions were correct, out of a total of 200 testing data points. 



```R
set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
cv.carseats

# oupput:
## $size
## [1] 20 18 10  8  6  4  2  1
## 
## $dev
## [1] 53 53 52 52 54 49 72 83
## 
## $k
## [1] -Inf  0.0  0.5  1.5  2.0  4.0 12.0 19.0
## 
## $method
## [1] "misclass"
## 
## attr(,"class")
## [1] "prune"         "tree.sequence"
```

To better fit the model, we can use corss-validation to choose the tree complexity.  We can use cv. tree function to perform the cross validation. The first element inside is the mode we fit. And the FUN argument is to specify the function we want to used for pruning. In this case, we specify the pruning function as misclassification. We can also set it as gini index, cost-complexity or mean square error.

From the output of `cv.carseats`:

`size` shows different size of decision trees, from the largest with 20 nodes to the smallest with one node

`dev` shows the deviations of different size of the decision tree in cross-validation, which are the error rate. The smaller the deviation, the better the model. 

`k` is the pruning points of different size of decision trees. The pruning point is a numeric value that reflects the cost complexity between pruned and unpruned decision trees. The larger the pruning point, the simpler the pruned decision tree. 



```R
plot(cv.carseats$size,cv.carseats$dev,type="b")
```

We can plot the size and deviation into a line graph, we can see that, the x-axis is the number of terminal nodes, and the y

axis is the error rate. When the size of decision tree is 4, the deviation reaches the minimum value. This means that this size of decision tree is optimal. 



```R
prune.carseats=prune.misclass(tree.carseats,best=4)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
accuracy=sum((tree.pred == High.test))/length(tree.pred)
accuracy

prune.carseats=prune.misclass(tree.carseats,best=6)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
accuracy=sum((tree.pred == High.test))/length(tree.pred)
accuracy
summary(prune.carseats)
```

Then, we can prune the tree, we first use the optimal size of decision decided by cross validation, which is 4, again we plot the tree, predict, and calculate the accuracy. We also set the best to 5 and compare their accuracy. As it shown, the accuracy for size 4 is actually lower than size of 5, it’s might be a little bit confusing, because the optimal model we get is clearly of size equal to 4. This is not uncommon and can happen due to randomness in the test set, the specific subset of data used for training and validation during cross-validation, or other factors. It is important to keep in mind that cross-validation provides an estimate of the model's performance on unseen data, but the actual performance may differ on a particular test set. 

You can see from the codes. When we perform the cross validation, the dataset we use is the total Carseats dataset, where the dataset will be split into k fold training set and validation set. But the dataset we used to calculate the accuracy is the test set. So, there is a certain amount of chance that the optimal model do not show the best effect in the test set. 



```R
### Using CART to solve this problem
library(rpart)
fit = rpart(High~.-Sales,method="class",data=Carseats1[train,])
# control=rpart.control(minsplit=2,cp=0)) 
summary(fit)
plot(fit)
text(fit)
```

The rpart() is also a function for building decision tree. Tree() and rpart() have some differences. As we have done for building a classification tree, we use tree()function to construct the classification tree, to avoid overfitting, we use cross validation to find the best size of the tree, and we prune the tree and we finally obtain the optimal classification tree model. 

However, rpart() is an extension of the basic decision tree algorithm that. It include a build-in pruning method that can aviod overfitting and simplifying the tree structure. It also has build-in corss validation that can be used to decide the pruning parameter. 

rpart() allows you to manually set up many parameters such as minsplit, cp, method. You can see all the parameters that can be set here. just use a question signal in front of the name of the package. 

You can use plot function to draw the tree or use a specific plotting tool in rpart.tool package. As you can see, the graph plotted by rpart are more colorful. 

you can also plot the complexity parpamter against the error of cross validation. The larger the cp parameter, the complex the tree, so we want to choose a simplest tree but not underfitting. 

The red dashed line represents the minimum cross-validated error rate, and the vertical bars represent the 1-standard error rule, where the smallest value of CP within one standard error of the minimum is chosen as the optimal CP.

## Regression tree

```R
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
```

Apart from classification tree, we can also use tree function to fit the regression tree. This is am example, we use the Boston dataset, We split  the dataset, 50% will be used for training and testing respectively. 



```R
# Grow the regression tree
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty=0)
```

Then, we can grow the regression tree, we use medv as response variable, and all the other variables as the predictor variables, we use subset train for training. 

We can print out the summary of this model, just as the classification tree, it will output the variables actually used in the tree construction, the number of terminal nodes and the residual mean deviance and the distribution of residuals. 

and we plot the tree, It is obvious that different from the classification tree, the estimated value of terminal nodes is not yes or no, its a numeric value. 

```R
# Use cross validation to choose tree complexity
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
cv.boston
```

Then we can also use cross validation to choose tree complexity and prune the tree based on this. Again, we use cv.tree to perform the corss validation, and plot size of regression tree against deviation. As it shown, the regreesion tree with size 7 have the lowest deviation, which is our optimal model. 

```R
prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston,pretty=0)
```

After that, we can prune the tree, which the optimal size of 7. 

```R
# Estimate the error of the tree
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
mean((yhat-boston.test)^2)

yhat=predict(prune.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
mean((yhat-boston.test)^2)
```

And it is a regression model, so we can use the mean square error to evaluate this model. We calculate the MSE before pruning and after pruning. 

























































 
