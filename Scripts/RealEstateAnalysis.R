#### Importing of libraries and dataset ####
install.packages("corrplot")
install.packages("randomForest")
install.packages("Metrics")
install.packages("clue")
install.packages("bit64")

library(car)
library(data.table)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(corrplot)
library(bit64)
## These two libraries are obtained from the website provided by the course coordinator in learning RF.
library(randomForest)
library(Metrics)
library(caTools)

library(quanteda)
library(readtext)

library(dplyr)
library(clue)


##### START HOUSE_SALE DATASET ######

loc <- "C:\\Users\\Brandon\\OneDrive - Nanyang Technological University\\NTU Notes\\Biz\\BC2406 Analytics I Visual & Predictive Techniques\\Team Assignment and Project\\Data"
setwd(loc)
house.sale.data <- fread("kc_house_data.csv")

house.sale.data
cleaned.house.sale.data <- house.sale.data[, c(-2, -17, -18, -19)]
cleaned.house.sale.data
sapply(cleaned.house.sale.data, class)

#### Data Cleaning ####
cleaned.house.sale.data$price <- as.numeric(cleaned.house.sale.data$price)
cleaned.house.sale.data$bedrooms <- as.numeric(cleaned.house.sale.data$bedrooms)
cleaned.house.sale.data$bathrooms <- as.numeric(cleaned.house.sale.data$bathrooms)
cleaned.house.sale.data$floors <- as.numeric(cleaned.house.sale.data$floors)

cleaned.house.sale.data$yr_built <- as.numeric(cleaned.house.sale.data$yr_built)
cleaned.house.sale.data$yr_renovated <- as.numeric(cleaned.house.sale.data$yr_renovated)
cleaned.house.sale.data$sqft_living <- as.numeric(cleaned.house.sale.data$sqft_living)
cleaned.house.sale.data$sqft_lot <- as.numeric(cleaned.house.sale.data$sqft_lot)
cleaned.house.sale.data$sqft_above <- as.numeric(cleaned.house.sale.data$sqft_above)
cleaned.house.sale.data$sqft_basement <- as.numeric(cleaned.house.sale.data$sqft_basement)
cleaned.house.sale.data$sqft_living15 <- as.numeric(cleaned.house.sale.data$sqft_living15)
cleaned.house.sale.data$sqft_lot15 <- as.numeric(cleaned.house.sale.data$sqft_lot15)

# Feature Engineering
cleaned.house.sale.data$age <- 2020 - pmax(cleaned.house.sale.data$yr_built, cleaned.house.sale.data$yr_renovated)


cleaned.house.sale.data$waterfront <- factor(cleaned.house.sale.data$waterfront)
cleaned.house.sale.data$view <- factor(cleaned.house.sale.data$view, ordered = T, levels = c(0, 1,2,3,4))
cleaned.house.sale.data$condition <- factor(cleaned.house.sale.data$condition, ordered = T, levels = c(1,2,3,4,5))
cleaned.house.sale.data$grade <- factor(cleaned.house.sale.data$grade, ordered = T, levels = c(1,2,3,4,5,6,7,8,9,10,11,12,13))


summary(cleaned.house.sale.data)
colSums(is.na(cleaned.house.sale.data))
cleaned.house.sale.data

# Is it possible for a house to have 0 bedrooms but 3.5 floors?
cleaned.house.sale.data[bedrooms == 0]
nrow(cleaned.house.sale.data[bedrooms == 0])

# Is it possible for a house to have 0 bathrooms but 1 bedroom?
cleaned.house.sale.data[bathrooms == 0]
nrow(cleaned.house.sale.data[bathrooms == 0])

# Only 13 rows for bedrooms == 0 and 10 rows for bathrooms == 0
# Some of these buildings have high sqft count transaction price and number of floors.
# Highly unlikely for it to be correct data.
# Those with no bedroom or bathrooms will be dropped
# For those with 1 bedroom but no bathroom, data likely to be correct as they have very low
# sqft_living.

## We believe that those listing with no bathrooms and 1 bedroom are small units 
## and those with no bedroom and bathrooms, and no bedroom and has bathroom are farmland
## However, because of the size of our dataset and the purpose of our model, we will
## be dropping all 16 rows.
cleaned.house.sale.data[bathrooms == 0 & bedrooms != 0]
cleaned.house.sale.data[bathrooms != 0 & bedrooms == 0]
cleaned.house.sale.data[bathrooms == 0 & bedrooms == 0]
nrow(cleaned.house.sale.data[bathrooms == 0 | bedrooms == 0])
cleaned.house.sale.data[bedrooms == 0 | bathrooms == 0]

cleaned.house.sale.data <- cleaned.house.sale.data[bathrooms != 0 & bedrooms != 0]
cleaned.house.sale.data


# Dealing with Outlier
cleaned.house.sale.data[bedrooms > 30]
# With 6000 sqft, 1 floor and going for only 640,000. This is unlikely to have 33 bedrooms.
# Changing value to 3.
cleaned.house.sale.data[bedrooms > 30, bedrooms:= 3]

## Generation of cluster to aid models ##
# Evaluation of WCSS to determine starting point of cluster visualization
# Looking at the documentation of kmeans, we should only do kmeans clustering on numeric values

numeric.cleaned.house.sale.data <- cleaned.house.sale.data[,.(price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, yr_built, yr_renovated, sqft_living15, sqft_lot15, age)]

numeric.cleaned.house.sale.data

wss.val <- c()
for (i in 1:30) {
  temp <- kmeans(numeric.cleaned.house.sale.data,i)
  wss.val <- c(wss.val, sum(temp$withinss))
}

wss.df <- data.frame(wss = wss.val, num.clusters = 1:30)
options(scipen=0)
ggplot(wss.df, aes(x = num.clusters, y = wss)) + geom_line()

# Initially tried 5 clusters, but characteristics of 3 clusters were very similar.
# So we tried 4 and then 3 before those characteristics were distinct.

set.seed(2014)
kmeans.result <- kmeans(numeric.cleaned.house.sale.data, 3)
cleaned.house.sale.data.clusters = copy(cleaned.house.sale.data)
cleaned.house.sale.data.clusters$cluster = factor(kmeans.result$cluster, ordered = T, levels = c(1,2,3))

cleaned.house.sale.data.clusters

### Visualization of clusters ###
# Looking at the count of clusters
table(cleaned.house.sale.data.clusters$cluster)

ggplot(cleaned.house.sale.data.clusters, aes(cluster, price)) + geom_boxplot()

ggplot(cleaned.house.sale.data.clusters, aes(bedrooms)) + geom_bar() + facet_grid(cols = vars(cluster))
ggplot(cleaned.house.sale.data.clusters, aes(cluster, sqft_living15)) + geom_boxplot()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, sqft_above)) + geom_boxplot()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, age)) + geom_boxplot()

ggplot(cleaned.house.sale.data, aes(age, price)) + geom_point()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, fill = view)) + geom_bar(position = "fill")
ggplot(cleaned.house.sale.data.clusters, aes(cluster, fill = waterfront)) + geom_bar(position = "fill")
ggplot(cleaned.house.sale.data.clusters, aes(cluster, fill = grade)) + geom_bar(position = "fill")
ggplot(cleaned.house.sale.data.clusters, aes(cluster, fill = condition)) + geom_bar(position = "fill")
ggplot(cleaned.house.sale.data.clusters, aes(cluster, fill = factor(floors))) + geom_bar(position = "fill")
ggplot(cleaned.house.sale.data.clusters, aes(cluster, sqft_living15)) + geom_boxplot()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, sqft_lot15)) + geom_boxplot()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, age)) + geom_boxplot()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, bedrooms)) + geom_boxplot()
ggplot(cleaned.house.sale.data.clusters, aes(cluster, bathrooms)) + geom_boxplot()


#### Data Visualization ####
options(scipen=999)

numeric.cleaned.house.sale.data <- cleaned.house.sale.data[,.(price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement, yr_built, yr_renovated, sqft_living15, sqft_lot15, age)]
numeric.cleaned.house.sale.data

m <- round(cor(numeric.cleaned.house.sale.data), 4)
corrplot(cor(numeric.cleaned.house.sale.data), method="color")
print(m[,1])

## Visualization of variables w.r.t price
cleaned.house.sale.data
# Categorical
ggplot(cleaned.house.sale.data, aes(view, price, fill = view)) + geom_boxplot()
ggplot(cleaned.house.sale.data, aes(waterfront, price, fill = waterfront)) + geom_boxplot()
ggplot(cleaned.house.sale.data, aes(grade, price, fill = grade)) + geom_boxplot()
ggplot(cleaned.house.sale.data, aes(condition, price, fill = condition)) + geom_boxplot()

# Numerical
plots.dt <- cleaned.house.sale.data %>% group_by(bedrooms) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=bedrooms, y=median_price, fill=bedrooms), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=bedrooms, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

plots.dt <- cleaned.house.sale.data %>% group_by(bathrooms) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=bathrooms, y=median_price, fill=bathrooms), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=bathrooms, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

plots.dt <- cleaned.house.sale.data %>% group_by(floors) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=floors, y=median_price, fill=floors), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=floors, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(sqft_living, price, colour = sqft_living)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(sqft_lot, price, colour = sqft_lot)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(sqft_above, price, colour = sqft_above)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(sqft_basement, price, colour = sqft_basement)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(yr_built, price, colour = yr_built)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(yr_renovated, price, colour = yr_renovated)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(sqft_living15, price, colour = sqft_living15)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(sqft_lot15, price, colour = sqft_lot15)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

ggplot(cleaned.house.sale.data, aes(age, price, colour = age)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")


# Houses are mostly under $1M.
cleaned.house.sale.data
ggplot(cleaned.house.sale.data, aes(price)) + geom_density(kernel = "gaussian")
ggplot(cleaned.house.sale.data, aes(factor(bedrooms), price)) + geom_boxplot()
ggplot(cleaned.house.sale.data, aes(factor(bathrooms), price)) + geom_boxplot()

# Bedrooms
plots.dt <- cleaned.house.sale.data %>% group_by(bedrooms) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=bedrooms, y=median_price, fill=bedrooms), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=bedrooms, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

# Bathrooms
plots.dt <- cleaned.house.sale.data %>% group_by(bathrooms) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=bathrooms, y=median_price, fill=bathrooms), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=bathrooms, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")


# Sqft plots
ggplot(cleaned.house.sale.data, aes(sqft_living, sqft_living15, color = sqft_living)) + geom_point(size=1, shape=1)+ geom_smooth(formula = y~x, method = "lm", se=FALSE, color="red") + theme(legend.position="none")
ggplot(cleaned.house.sale.data, aes(sqft_lot, sqft_lot15, color = sqft_living)) + geom_point(size=1, shape=1)+ geom_smooth(formula = y~x, method = "lm", se=FALSE, color="red") + theme(legend.position="none")



ggplot(cleaned.house.sale.data, aes(age, price)) + geom_point()

ggplot(cleaned.house.sale.data, aes(bedrooms, price)) + geom_point()
ggplot(cleaned.house.sale.data, aes(bathrooms, price)) + geom_point()

#### Dump the results #####
write.csv(cleaned.house.sale.data.clusters, "kc_house_data_cleaned.csv")


colSums(is.na(cleaned.house.sale.data.clusters))

#### Train-test split ####
## Test set will be used as POC demo ##
# Full dataset will be used for CART

set.seed(2020)
train <- sample.split(Y = cleaned.house.sale.data.clusters$price, SplitRatio = 0.7)
cleaned.house.sale.data.clusters.train <- subset(cleaned.house.sale.data.clusters, train == T)
cleaned.house.sale.data.clusters.test <- subset(cleaned.house.sale.data.clusters, train == F)

#### Linear Regression Models ####

### Standard linear regression on dataset ###
house.sale.lin.reg.train <- lm(price~ . -id -yr_built -yr_renovated - price, data = cleaned.house.sale.data.clusters.train)
summary(house.sale.lin.reg.train)

n = length(resid(house.sale.lin.reg.train))
step(house.sale.lin.reg.train, direction = "backward", k = log(n))

house.sale.lin.reg.train.reduced <- lm(formula = price ~ bedrooms + bathrooms + sqft_living + floors + 
                              waterfront + view + grade + sqft_above + sqft_lot15 + age + 
                              cluster, data = cleaned.house.sale.data.clusters.train)
summary(house.sale.lin.reg.train.reduced)
car::vif(house.sale.lin.reg.train.reduced)

# Noticed sqft_living and sqft_living15 must be correlated. One should be a function
# of the other.
# High Adjusted multicolinearity score for sqft_living and sqft_above

# Try remove sqft_living
house.sale.lin.reg.train.reduced.cleaning <- lm(formula = price ~ bedrooms + bathrooms + floors + 
                                 waterfront + view + grade + sqft_above + sqft_lot15 + age + 
                                 cluster, data = cleaned.house.sale.data.clusters.train)

summary(house.sale.lin.reg.train.reduced.cleaning)

car::vif(house.sale.lin.reg.train.reduced.cleaning)
# 0.1% fall in r^2 value. Large change in coefficient of bedrooms. Furthermore, sqft_living seems
# to be a function of bedrooms, bathrooms and floors. Thus indicating possible multi-collinearity
# Will remove.

# Try remove sqft_above
house.sale.lin.reg.train.reduced.cleaning <- lm(formula = price ~ bedrooms + bathrooms + floors + 
                                       waterfront + view + grade + sqft_lot15 + age + 
                                       cluster, data = cleaned.house.sale.data.clusters.train)

summary(house.sale.lin.reg.train.reduced.cleaning)

car::vif(house.sale.lin.reg.train.reduced.cleaning)
# No significant fall in r^2 value. Large change in coefficient of bedrooms with a change in sign.
# Now coefficient of bedrooms is more suited to what we expect. An increase in bedroom should typically
# mean increase in house price

# As there are statistically insigificant variables now, we will run backwards AIC to determine if
# this is the optimal model complexity

n = length(resid(house.sale.lin.reg.train.reduced.cleaning))
step(house.sale.lin.reg.train.reduced.cleaning, direction = "backward", k = log(n))

house.sale.lin.reg.train.reduced.cleaning <- lm(formula = price ~ bathrooms + waterfront + view + grade + 
                                 sqft_lot15 + age + cluster, data = cleaned.house.sale.data.clusters.train)

summary(house.sale.lin.reg.train.reduced.cleaning)
car::vif(house.sale.lin.reg.train.reduced.cleaning)

# Low multicollinearity scores and high statistical significance for most variables
# Final model for linear regression
house.sale.lin.reg.train <- lm(formula = price ~ bathrooms + waterfront + view + grade + 
                                sqft_lot15 + age + cluster, data = cleaned.house.sale.data.clusters.train)
summary(house.sale.lin.reg.train)

house.sale.train.set.error <- residuals(house.sale.lin.reg.train)
SSE.house.sale.train.set <- sum(house.sale.train.set.error^2)
mean.house.sale.train <- mean(cleaned.house.sale.data.clusters.train$price)
TSS.house.sale.train.set <- sum((cleaned.house.sale.data.clusters.train$price - mean.house.sale.train)^2)
rsql.house.sale.train.set <- 1 - (SSE.house.sale.train.set/TSS.house.sale.train.set)

predict.house.sale.lin.reg.test <- predict(house.sale.lin.reg.train, newdata = cleaned.house.sale.data.clusters.test)
house.sale.test.set.error <- cleaned.house.sale.data.clusters.test$price - predict.house.sale.lin.reg.test
SSE.house.sale.test.set <- sum(house.sale.test.set.error^2)
mean.house.sale.test <- mean(cleaned.house.sale.data.clusters.test$price)
TSS.house.sale.test.set <- sum((cleaned.house.sale.data.clusters.test$price - mean.house.sale.test)^2)
rsql.house.sale.test.set <- 1 - (SSE.house.sale.test.set/TSS.house.sale.test.set)

c(rsql.house.sale.train.set, rsql.house.sale.test.set)


cleaned.house.sale.data.clusters

set.seed(2020)
#### CART model for continuous price ####
# Using cleaned.data -> 0.71 R^2. 0.86 R^2 on clustered data
cart.house.sale.all <- rpart(price~. -id, data = cleaned.house.sale.data.clusters, method = "anova", control = rpart.control(cp = 0))
#rpart.plot(m2, nn = T, main = "Maximal Tree of housing_price.csv")
plotcp(cart.house.sale.all)
printcp(cart.house.sale.all)

# Automatic tree pruning script from Prof. Neumann Chew
CVerror.cap <- cart.house.sale.all$cptable[which.min(cart.house.sale.all$cptable[,"xerror"]), "xerror"] + cart.house.sale.all$cptable[which.min(cart.house.sale.all$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4
while (cart.house.sale.all$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cp.opt = ifelse(i > 1, sqrt(cart.house.sale.all$cptable[i,1] * cart.house.sale.all$cptable[i-1,1]), 1)

cart.house.sale.all.pruned <- prune(cart.house.sale.all, cp = cp.opt)
rpart.plot(cart.house.sale.all.pruned, nn = T, main = paste("Pruned tree with cp ", toString(cp.opt)))
plotcp(cart.house.sale.all.pruned)
printcp(cart.house.sale.all.pruned)

# Accuracy metrics of CART model #
temp <- printcp(cart.house.sale.all.pruned)
temp[,c(3,4)]
rsql.val <- 1-temp[,c(3,4)]
rsql.val[nrow(rsql.val)]

# Best model. Recreate tree to reduce input variable dependency

set.seed(2020)
cart.house.sale.trim <- rpart(price~bathrooms+cluster+grade+sqft_living+sqft_living15+sqft_lot+view+waterfront+yr_built, data = cleaned.house.sale.data.clusters, method = "anova", control = rpart.control(cp = 0))

CVerror.cap <- cart.house.sale.trim$cptable[which.min(cart.house.sale.trim$cptable[,"xerror"]), "xerror"] + cart.house.sale.trim$cptable[which.min(cart.house.sale.trim$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4
while (cart.house.sale.trim$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cp.opt = ifelse(i > 1, sqrt(cart.house.sale.trim$cptable[i,1] * cart.house.sale.trim$cptable[i-1,1]), 1)

cart.house.sale.trim.pruned <- prune(cart.house.sale.trim, cp = cp.opt)
rpart.plot(cart.house.sale.trim.pruned, nn = T, main = paste("Pruned tree with cp ", toString(cp.opt)))

# Calculation of R^2 on CART model
temp <- printcp(cart.house.sale.trim.pruned)
temp[,c(3,4)]
rsql.val <- 1-temp[,c(3,4)]
rsql.val[nrow(rsql.val)]
# Performs better than the one on the other two models
# 0.854 r^2

summary(cart.house.sale.trim.pruned)

cart.house.sale.trim.pruned$variable.importance

# relerror should represent R^2 of the resultant model
# https://stats.stackexchange.com/questions/103018/difference-between-rel-error-and-xerror-in-rpart-regression-trees
# https://stackoverflow.com/questions/29197213/what-is-the-difference-between-rel-error-and-x-error-in-a-rpart-decision-tree/47803431
# https://community.alteryx.com/t5/Alteryx-Designer-Knowledge-Base/Understanding-the-Outputs-of-the-Decision-Tree-Tool/ta-p/144773#:~:text=Rel%20error%20(relative%20error)%20is,built%2Din%20cross%20validation).

##### END HOUSE_SALE DATASET #####

##### START AIRBNB DATASET #####

listings.dt <- fread("listings.csv")
summary(listings.dt)

#Summary of Identified Variables
listings2.dt <- listings.dt[,c("id", "host_is_superhost", "neighbourhood_group_cleansed",
                               "latitude", "longitude", "property_type", "room_type", "accommodates",
                               "bathrooms", "bedrooms", "beds", "square_feet", "price", "security_deposit",
                               "cleaning_fee", "guests_included", "availability_365", "number_of_reviews",
                               "instant_bookable", "review_scores_rating")]
summary(listings2.dt$property_type)
sapply(listings2.dt,class)
listings2.dt[duplicated(listings2.dt)]

##################################### Data Cleaning (listings.csv) ##############################
cleanlistings.dt <- copy(listings2.dt)

#Character to Numeric
cleanlistings.dt$price <- as.numeric(gsub("\\$", "", listings2.dt$price))
cleanlistings.dt$security_deposit <- as.numeric(gsub("\\$", "", listings2.dt$security_deposit))
cleanlistings.dt$cleaning_fee <- as.numeric(gsub("\\$", "", listings2.dt$cleaning_fee))

#Character to Factor
cleanlistings.dt <- cleanlistings.dt[, property_type:=as.factor(property_type)]
cleanlistings.dt <- cleanlistings.dt[, room_type:=as.factor(room_type)]
cleanlistings.dt <- cleanlistings.dt[, instant_bookable:=as.factor(instant_bookable)]
cleanlistings.dt <- cleanlistings.dt[, host_is_superhost:=as.factor(host_is_superhost)]
cleanlistings.dt <- cleanlistings.dt[, neighbourhood_group_cleansed:=as.factor(neighbourhood_group_cleansed)]

#Convert Integer to Numeric
sapply(cleanlistings.dt,class)
columns <-c("accommodates", "bedrooms", "beds", "guests_included", "availability_365", "number_of_reviews", "review_scores_rating")
cleanlistings.dt[, columns] <- lapply(columns, function(x) as.numeric(cleanlistings.dt[[x]]))
sapply(cleanlistings.dt,class)

#Group Categorical Variables to Smaller Groups
#Property Type
summary(cleanlistings.dt$property_type)
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Boat'] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Tent'] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Treehouse'] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Yurt'] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Camper/RV'] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == ''] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Dorm'] <- 'Other'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Bed & Breakfast'] <- 'House'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Townhouse'] <- 'House'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Cabin'] <- 'House'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Chalet'] <- 'House'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Bungalow'] <- 'House'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Loft'] <- 'Apartment'
cleanlistings.dt$property_type[cleanlistings.dt$property_type == 'Condominium'] <- 'Apartment'

cleanlistings.dt <- droplevels(cleanlistings.dt)
summary(cleanlistings.dt$property_type)

#Summary of Factor/Numeric
summary(cleanlistings.dt)
summary(cleanlistings.dt$property_type)
summary(cleanlistings.dt$room_type)
summary(cleanlistings.dt$instant_bookable)
summary(cleanlistings.dt$host_is_superhost)
summary(cleanlistings.dt$neighbourhood_group_cleansed)
summary(cleanlistings.dt$price)
summary(cleanlistings.dt$security_deposit)
summary(cleanlistings.dt$cleaning_fee)

#Missing Values (NA's)
sapply(cleanlistings.dt, function(x) sum(is.na(x)))

#NA's to Median
#Price
#Check row without price - cleanlistings.dt[is.na(cleanlistings.dt$price),]
cleanlistings.dt[, price.mean := mean(price, na.rm = TRUE) # calculate mean
][is.na(price), price := price.mean # replace NA with mean
][, price.mean := NULL # remove mean col
]
summary(cleanlistings.dt$price)

#Security Deposit
cleanlistings.dt[is.na(security_deposit), security_deposit := 0] # replace NA with 0
summary(cleanlistings.dt$security_deposit)

#Cleaning Fee
cleanlistings.dt[is.na(cleaning_fee), cleaning_fee := 0] # replace NA with 0
summary(cleanlistings.dt$cleaning_fee)


#Bathrooms
cleanlistings.dt[, bathrooms.median := median(bathrooms, na.rm = TRUE) # calculate median
][is.na(bathrooms), bathrooms := bathrooms.median # replace NA with median
][, bathrooms.median := NULL # remove median col
]
summary(cleanlistings.dt$bathrooms)

#Bedrooms
cleanlistings.dt[, bedrooms.median := median(bedrooms, na.rm = TRUE) # calculate median
][is.na(bedrooms), bedrooms := bedrooms.median # replace NA with median
][, bedrooms.median := NULL # remove median col
]
summary(cleanlistings.dt$bedrooms)

#Beds
cleanlistings.dt[, beds.median := median(beds, na.rm = TRUE) # calculate median
][is.na(beds), beds := beds.median # replace NA with median
][, beds.median := NULL # remove median col
]
summary(cleanlistings.dt$beds)

#Review Scores Rating
cleanlistings.dt[, review_scores_rating.median := median(review_scores_rating, na.rm = TRUE) # calculate median
][is.na(review_scores_rating), review_scores_rating := review_scores_rating.median # replace NA with median
][, review_scores_rating.median := NULL # remove median col
]
summary(cleanlistings.dt$review_scores_rating)

#Host is Superhost
cleanlistings.dt$host_is_superhost[cleanlistings.dt$host_is_superhost == '']<-'f'
cleanlistings.dt <- droplevels(cleanlistings.dt)
summary(cleanlistings.dt$host_is_superhost)

#Check for any leftover missing values
sapply(cleanlistings.dt, function(x) sum(is.na(x)))

#Drop square_feet
cleanlistings.dt[, square_feet:=NULL]

#Summary of cleaned dataset
summary(cleanlistings.dt)

write.csv(cleanlistings.dt, "listings_cleaned.csv")

################################## Data Cleaning (reviews.csv) ################################
quanteda_options("threads" = 12)

review.file <- "/reviews.csv"

review.data <- readtext(paste(loc, review.file, sep = ""), text_field = "comments")
summary(review.data)


## Removing german and chinese from the dataset or other non-english characters by converting to ASCII
data.corpus <- corpus(review.data)
texts(data.corpus) <- iconv(texts(data.corpus), from="UTF-8", to="ASCII", sub = "")

summary(data.corpus, 5)

# Tokenization
data.tokens <- tokens(data.corpus, remove_punct = TRUE, remove_symbols = TRUE)
sum(ntoken(data.tokens))

# Stopword removal
data.tokens <- tokens_remove(data.tokens, pattern = stopwords("en"))
data.tokens <- tokens_remove(data.tokens, pattern = stopwords("german"))
sum(ntoken(data.tokens))

# Stemming of reviews
data.tokens <- tokens_wordstem(data.tokens)
data.tokens <- tokens_tolower(data.tokens)
sum(ntoken(data.tokens))

# Exploration of tokens
freq.tokens <- ntoken(data.tokens)
summary(factor(freq.tokens))
max(freq.tokens)

# Explore the longest review
texts(data.corpus)[which(freq.tokens == max(freq.tokens, na.rm = T))]

# Building Document-Feature-Matrix
dfm.lsd <- dfm(data.tokens, dictionary=data_dictionary_LSD2015)
dfm.lsd.df <- convert(dfm.lsd, to = "data.frame")

dfm.lsd.df

# Calculation of sentiment scores
dfm.lsd.df$adjusted_neg <- dfm.lsd.df$negative + dfm.lsd.df$neg_positive - dfm.lsd.df$neg_negative
dfm.lsd.df$adjusted_pos <- dfm.lsd.df$positive + dfm.lsd.df$neg_negative - dfm.lsd.df$neg_positive

dfm.lsd.df$sentiment <- dfm.lsd.df$adjusted_pos - dfm.lsd.df$adjusted_neg

dfm.lsd.df


# Exploration of data
texts(data.corpus)[which(dfm.lsd.df$sentiment == max(dfm.lsd.df$sentiment))]
texts(data.corpus)[which(dfm.lsd.df$sentiment == min(dfm.lsd.df$sentiment))]
texts(data.corpus)[which(dfm.lsd.df$neg_positive == max(dfm.lsd.df$neg_positive))]


# Creation of labelled data for Linear Regression
final.review.data <- copy(review.data)
final.review.data$sentiment = dfm.lsd.df$sentiment

final.review.data


x <- tapply(final.review.data$sentiment, final.review.data$listing_id, mean)
mean.sentiment.df <- data.frame(id = names(x), x)
mean.sentiment.df

mean.sentiment.df <- setNames(aggregate(final.review.data$sentiment~final.review.data$listing_id, FUN = mean), c("id", "sentiment"))

# Exploring labelled data
max(mean.sentiment.df$sentiment)
min(mean.sentiment.df$sentiment)

#Adding Sentiment Analysis into cleanlistings.dt
cleanlistings.dt <- dplyr::inner_join(mean.sentiment.df, cleanlistings.dt, by="id", na_matches = "na")
summary(cleanlistings.dt$sentiment)
setDT(cleanlistings.dt)
################################### Data Exploration (listings.csv) #######################################

### Categorical variables ###
ggplot(cleanlistings.dt, aes(x=neighbourhood_group_cleansed, y=price, fill=neighbourhood_group_cleansed)) +
  geom_boxplot() +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

ggplot(cleanlistings.dt, aes(x=room_type, y=price, fill=room_type)) +
  geom_boxplot() +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

ggplot(cleanlistings.dt, aes(x=property_type, y=price, fill=property_type)) +
  geom_boxplot() +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

ggplot(cleanlistings.dt, aes(x=instant_bookable, y=price, fill=instant_bookable)) +
  geom_boxplot() +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

ggplot(cleanlistings.dt, aes(x=host_is_superhost, y=price, fill=host_is_superhost)) +
  geom_boxplot() +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

### Continuous variables ###

##Visualize correlation matrix of  using correlogram for Numeric Values
num_cleanlistings.dt <- cleanlistings.dt[, c("price", "sentiment", "accommodates", "bathrooms", "bedrooms", "beds", "security_deposit",
                                             "cleaning_fee", "guests_included", "availability_365", "number_of_reviews", "review_scores_rating")]

m <- round(cor(num_cleanlistings.dt),4)
corrplot(cor(num_cleanlistings.dt), method="color")
print(m[,1])

#Bar/Scatter Plot (Continuous Variables)
#Price
ggplot(cleanlistings.dt, aes(x=price)) + 
  geom_histogram(aes(y=..density..), bins=80, colour="black", fill="white") +
  geom_density(alpha=.2, fill="blue")

#Continuous Variables with Price (Barplot using dplyr)
library(dplyr)
#Accommodates
plots.dt <- cleanlistings.dt %>% group_by(accommodates) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=accommodates, y=median_price, fill=accommodates), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=accommodates, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Bedrooms
plots.dt <- cleanlistings.dt %>% group_by(bedrooms) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=bedrooms, y=median_price, fill=bedrooms), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=bedrooms, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Bathrooms
plots.dt <- cleanlistings.dt %>% group_by(bathrooms) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=bathrooms, y=median_price, fill=bathrooms), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=bathrooms, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Beds
plots.dt <- cleanlistings.dt %>% group_by(beds) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=beds, y=median_price, fill=beds), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=beds, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Cleaning Fee
ggplot(cleanlistings.dt, aes(x=cleaning_fee, y=price, color=cleaning_fee)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Guests Included
plots.dt <- cleanlistings.dt %>% group_by(guests_included) %>% summarise(median_price = median(price), sd = sd(price))
ggplot(aes(x=guests_included, y=median_price, fill=guests_included), data = plots.dt) + geom_bar(color="black", stat = "identity") +
  geom_errorbar(aes(x=guests_included, ymin=median_price, ymax=median_price+sd), width=0.3, colour="black") +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Security Deposit
ggplot(cleanlistings.dt, aes(x=security_deposit, y=price, color=security_deposit)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Availability 365
ggplot(cleanlistings.dt, aes(x=availability_365, y=price, color=availability_365)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Number of Reviews
ggplot(cleanlistings.dt, aes(x=number_of_reviews, y=price, color=number_of_reviews)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Review Scores Rating
ggplot(cleanlistings.dt, aes(x=review_scores_rating, y=price, color=review_scores_rating)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")

#Sentiment
ggplot(cleanlistings.dt, aes(x=sentiment, y=price, color=sentiment)) + geom_point(size=1, shape=1) +
  geom_smooth(formula = y~x, method = "lm", se=FALSE, color="black") + theme(legend.position="none")


### Clustering with Sentiment###
cleanlistings.cluster.sentiment.dt <- cleanlistings.dt[, c("sentiment", "accommodates", "bathrooms", "bedrooms", "beds", 
                                                           "security_deposit", "cleaning_fee", "guests_included")]
set.seed(2020)
wss.val <- c()
for (i in 1:30) {
  temp <- kmeans(cleanlistings.cluster.sentiment.dt,i)
  wss.val <- c(wss.val, sum(temp$withinss))
}

wss.df <- data.frame(wss = wss.val, num.clusters = 1:30)
ggplot(wss.df, aes(x = num.clusters, y = wss)) + geom_line()

set.seed(2020)
kmeans.result <- kmeans(cleanlistings.cluster.sentiment.dt, 3)
cleanlistings.dt$cluster = factor(kmeans.result$cluster, ordered = T, levels = c(1,2,3))
ggplot(cleanlistings.dt, aes(cluster, price, fill = cluster)) + geom_boxplot()
ggplot(cleanlistings.dt, aes(cluster, sentiment, fill = cluster)) + geom_boxplot()


### Clustering without Sentiment###

cleanlistings.cluster.dt <- cleanlistings.dt[, c("accommodates", "bathrooms", "bedrooms", "beds", 
                                                 "security_deposit", "cleaning_fee", "guests_included")]
set.seed(2020)
wss.val <- c()
for (i in 1:30) {
  temp <- kmeans(cleanlistings.cluster.dt,i)
  wss.val <- c(wss.val, sum(temp$withinss))
}

wss.df <- data.frame(wss = wss.val, num.clusters = 1:30)
ggplot(wss.df, aes(x = num.clusters, y = wss)) + geom_line()

# Initially tried 5 clusters, but characteristics of 3 clusters were very similar.
# So we tried 4 and then 3 before those characteristics were distinct.
set.seed(2020)
kmeans.result <- kmeans(cleanlistings.cluster.dt, 5)
cleanlistings.dt$cluster = factor(kmeans.result$cluster, ordered = T, levels = c(1,2,3,4,5))
ggplot(cleanlistings.dt, aes(cluster, price, fill = cluster)) + geom_boxplot()

set.seed(2020)
kmeans.result <- kmeans(cleanlistings.cluster.dt, 4)
cleanlistings.dt$cluster = factor(kmeans.result$cluster, ordered = T, levels = c(1,2,3,4))
ggplot(cleanlistings.dt, aes(cluster, price, fill = cluster)) + geom_boxplot()

set.seed(2020)
kmeans.result <- kmeans(cleanlistings.cluster.dt, 3)
cleanlistings.dt$cluster = factor(kmeans.result$cluster, ordered = T, levels = c(1,2,3))
ggplot(cleanlistings.dt, aes(cluster, price, fill = cluster)) + geom_boxplot()

set.seed(2020)
kmeans.result <- kmeans(cleanlistings.cluster.dt, 2)
cleanlistings.dt$cluster = factor(kmeans.result$cluster, ordered = T, levels = c(1,2))
ggplot(cleanlistings.dt, aes(cluster, price, fill = cluster)) + geom_boxplot()

ggplot(cleanlistings.dt, aes(cluster, fill = cluster)) + geom_bar()
ggplot(cleanlistings.dt, aes(cluster, fill = room_type)) + geom_bar(position = "fill")

ggplot(cleanlistings.dt, aes(cluster, bedrooms, fill = cluster)) + geom_boxplot()
ggplot(cleanlistings.dt, aes(cluster, bathrooms, fill = cluster)) + geom_boxplot()
ggplot(cleanlistings.dt, aes(cluster, accommodates, fill = cluster)) + geom_boxplot()
ggplot(cleanlistings.dt, aes(cluster, beds, fill = cluster)) + geom_boxplot()
ggplot(cleanlistings.dt, aes(cluster, fill = neighbourhood_group_cleansed)) + geom_bar(position = "fill")
################################ Data Visualisation (reviews.csv) ##################################

all.review.dfm <- dfm(data.corpus, remove = stopwords("en"), remove_punct = TRUE) %>% 
  dfm_trim(min_termfreq = 100, verbose = FALSE)
set.seed(2014)
textplot_wordcloud(all.review.dfm, max_words = 100)

neg.review.dfm <- data.corpus[which(final.review.data$sentiment < 0)] %>% 
  dfm(remove = stopwords("en"), remove_punct = TRUE) %>% 
  dfm_trim(min_termfreq = 50, verbose = FALSE)
set.seed(2014)
textplot_wordcloud(neg.review.dfm, max_words = 100)

pos.review.dfm <- data.corpus[which(final.review.data$sentiment > 0)] %>% 
  dfm(remove = stopwords("en"), remove_punct = TRUE) %>% 
  dfm_trim(min_termfreq = 100, verbose = FALSE)
set.seed(2014)
textplot_wordcloud(pos.review.dfm, max_words = 100)

neutral.review.dfm <- data.corpus[which(final.review.data$sentiment == 0)] %>% 
  dfm(remove = stopwords("en"), remove_punct = TRUE) %>% 
  dfm_trim(min_termfreq = 50, verbose = FALSE)
set.seed(2014)
textplot_wordcloud(neutral.review.dfm, max_words = 100)


############################## Data Modelling (Airbnb Dataset) ####################################

#Train Test Split for Linear Regression
set.seed(2014)
dt = sort(sample(nrow(cleanlistings.dt), nrow(cleanlistings.dt)*.7))
train.listings<-cleanlistings.dt[dt,]
test.listings<-cleanlistings.dt[-dt,]

#Linear Regression with Identified Variables
#Analyse AIC (dropped security deposit)
m1 <- lm(price~ accommodates + bathrooms + bedrooms + beds + security_deposit + cleaning_fee 
         + guests_included + room_type + neighbourhood_group_cleansed,
         data = train.listings)
step(m1, direction = "backward")

#Analyse p-value (dropped neighbourhood_group_cleansed)
m2 <- lm(price~ accommodates + bathrooms + bedrooms + beds + cleaning_fee 
         + guests_included + room_type + neighbourhood_group_cleansed,
         data = train.listings)
summary(m2)

#Analyse adjusted VIF (dropped adjusted VIF > 2 - beds and accommodates)
m3 <- lm(price~ accommodates + bathrooms + bedrooms + beds + cleaning_fee 
         + guests_included + room_type,
         data = train.listings)
summary(m3)
vif(m3)

#Check final linear regression model
m4 <- lm(formula = price ~ bathrooms + bedrooms + 
           cleaning_fee + guests_included + room_type, data = train.listings)
summary(m4)
vif(m4)


#R^2 Train Set
train.set.error <- residuals(m4)
SSE.train.set <- sum(train.set.error^2)
mean.train <- mean(train.listings$price)
TSS.train.set <- sum((train.listings$price - mean.train)^2)
rsql.train.set <- 1 - (SSE.train.set/TSS.train.set)

predict.lin.reg.test <- predict(m4, newdata = test.listings)
test.set.error <- test.listings$price - predict.lin.reg.test
SSE.test.set <- sum(test.set.error^2)
mean.test <- mean(test.listings$price)
TSS.test.set <- sum((test.listings$price - mean.test)^2)
rsql.test.set <- 1 - (SSE.test.set/TSS.test.set)

c(rsql.train.set, rsql.test.set)


#CART
set.seed(2014)
cart1 <- rpart(price~ accommodates + bathrooms + bedrooms + beds + security_deposit + cleaning_fee 
               + guests_included + room_type + property_type,
               data = cleanlistings.dt, method = "anova", control = rpart.control(cp = 0))

plotcp(cart1)
printcp(cart1)

#Automatic Pruning
CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] + 
  cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]

i<- 1; j<-4
while (cart1$cptable[i,j] > CVerror.cap) {
  i <- i+1
}

cp.opt = ifelse(i>1, sqrt(cart1$cptable[i,1]*cart1$cptable[i-1,1]),1)
cart2 <- prune(cart1, cp=cp.opt)

printcp(cart2)
plotcp(cart2)
summary(cart2)
rpart.plot(cart2, nn=T)

# Accuracy metrics of CART model #
temp <- printcp(cart2)
rsql.val <- 1-temp[,c(3,4)]
rsql.val[nrow(rsql.val)]



#### PROOF OF CONCEPT ####
poc.data <- cleaned.house.sale.data.clusters.test
pred.val <- predict(cart.house.sale.trim.pruned, newdata = poc.data)
poc.data$profit <- pred.val - poc.data$price
poc.data$profit.margin <- poc.data$profit/poc.data$price

head(poc.data[order(-profit)], 20)

poc.rental.data <- copy(poc.data)

poc.rental.data$beds <- round(1.5 * poc.rental.data$bedrooms)
poc.rental.data$accommodates <-  round(1.5 * poc.rental.data$beds)
poc.rental.data$guests_included <- round(0.5 * poc.rental.data$accommodates)
poc.rental.data$security_deposit <- median(cleanlistings.dt$security_deposit)
poc.rental.data$cleaning_fee <- median(cleanlistings.dt$cleaning_fee)
poc.rental.data$room_type <- "Entire home/apt"

summary(poc.rental.data)

poc.rental.data$listing_price <- predict(m3, newdata=poc.rental.data)
days_per_year_occupied <- round(0.6 * 365)
poc.rental.data$rental_revenue <- poc.rental.data$listing_price * days_per_year_occupied
poc.rental.data

poc.data$listing_price <- poc.rental.data$listing_price
poc.data$rental_revenue <- poc.rental.data$rental_revenue

poc.data[, rental_yield:=.(rental_revenue / price)]
poc.data[, adjusted_profit:=.(profit + rental_revenue)]
poc.data[, adjusted_profit_margin:=.(adjusted_profit / price)]

display.data <- poc.data[, .(id, price, profit, profit.margin, rental_revenue, rental_yield, adjusted_profit, adjusted_profit_margin)]

display.data

head(display.data[order(-adjusted_profit)], 20)
head(display.data[which(display.data$price > 1000000)][order(-adjusted_profit_margin)], 20)
head(display.data[which(display.data$price > 500000)][order(-adjusted_profit_margin)], 20)
