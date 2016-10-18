# set up working directory
setwd("~/R files/Federalist")
# load text mining packages
library(tm)
library(NLP)
library(SnowballC)
#============================================================================
# Step 1. Data preprocessing
preprocess.directory = function(dirname){
  # the directory must have all the relevant text files
  ds = DirSource(dirname)
  # Corpus will make a tm document corpus from this directory
  fp = Corpus(ds)
  # data cleaning
  # make all words lower case
  fp = tm_map( fp , content_transformer(tolower))
  # remove all punctuation
  fp = tm_map( fp, removePunctuation)
  # remove stopwords like the, a, and so on
  fp = tm_map( fp, removeWords, stopwords("english"))
  # remove stems like suffixes
  fp = tm_map( fp, stemDocument)
  # remove extra whitespace
  fp = tm_map( fp, stripWhitespace)
  # write the corpus out to the files for our future use
  writeCorpus( fp , sprintf('%s_clean',dirname))
}

preprocess.directory("FederalistPapers/fp_hamilton_test")
preprocess.directory("FederalistPapers/fp_hamilton_train")
preprocess.directory("FederalistPapers/fp_madison_test")
preprocess.directory("FederalistPapers/fp_madison_train")
#============================================================================
# Step 2. load data from their corresponding directory into workspace
read.directory <- function(dirname) {
    # Store the infiles in a list
    infiles = list();
    # Get a list of filenames in the directory
    filenames = dir(dirname,full.names=TRUE);
    for (i in 1:length(filenames)){
        infiles[[i]] = scan(filenames[i],what="",quiet=TRUE);
         }
    return(infiles)
}
hamilton.train <- read.directory('fp_hamilton_train_clean')
hamilton.test <- read.directory('fp_hamilton_test_clean')
madison.train <- read.directory('fp_madison_train_clean')
madison.test <- read.directory('fp_madison_test_clean')
#============================================================================
# Step 3. Make dictionary sorted by number of times a word appears in corpus
make.sorted.dictionary.df <- function(infiles){
    # This returns a dataframe that is sorted by the number of times a word appears
  
    # List of vectors to one big vetor
    dictionary.full <- unlist(infiles) 
    # Tabulates the full dictionary
    tabulate.dic <- tabulate(factor(dictionary.full)) 
    # Find unique values
    dictionary <- unique(dictionary.full) 
    # Sort them alphabetically
    dictionary <- sort(dictionary)
    dictionary.df <- data.frame(word = dictionary, count = tabulate.dic)
    sort.dictionary.df <- dictionary.df[order(dictionary.df$count,decreasing=TRUE),];
    return(sort.dictionary.df)
}
dictionary <- make.sorted.dictionary.df(c(hamilton.train,hamilton.test,madison.train,madison.test))
#============================================================================
# Step 4. Document Term Matrix(DTM) table
make.document.term.matrix <- function(infiles,dictionary){
    # This takes the text and dictionary objects from above and outputs a DTM
    num.infiles <- length(infiles);
    num.words <- nrow(dictionary);
    # Instantiate a matrix where rows are documents and columns are words
    dtm <- mat.or.vec(num.infiles,num.words); # A matrix filled with zeros
    for (i in 1:num.infiles){
        num.words.infile <- length(infiles[[i]]);
        infile.temp <- infiles[[i]];
        for (j in 1:num.words.infile){
            ind <- which(dictionary == infile.temp[j])[[1]];
            # print(sprintf('%s,%s', i , ind))
            dtm[i,ind] <- dtm[i,ind] + 1;
            #print(c(i,j))
        }
    }
return(dtm);
}
# OR use DocumentTermMatrix() function transform corpus into dtm
# get dtm frequency table with data manipulate package 
dtm.hamilton.train <- make.document.term.matrix(hamilton.train,dictionary)
dtm.hamilton.test <- make.document.term.matrix(hamilton.test,dictionary)
dtm.madison.train <- make.document.term.matrix(madison.train,dictionary)
dtm.madison.test <- make.document.term.matrix(madison.test,dictionary)
#============================================================================
# Step 5. Build a Naive Bayes classifier
make.log.pvec <- function(dtm,mu){
    # Sum up the number of instances per word
    pvec.no.mu <- colSums(dtm)
    # Sum up number of words
    n.words <- sum(pvec.no.mu)
    # Get dictionary size
    dic.len <- length(pvec.no.mu)
    # Incorporate mu and normalize
    log.pvec <- log(pvec.no.mu + mu) - log(mu*dic.len + n.words)
    return(log.pvec)
}
D <- nrow(dictionary)
mu <- 1/D
# calculate the log probability vectors for all document term matrices
logp.hamilton.train <- make.log.pvec(dtm.hamilton.train,mu)
logp.hamilton.test <- make.log.pvec(dtm.hamilton.test,mu)
logp.madison.train <- make.log.pvec(dtm.madison.train,mu)
logp.madison.test <- make.log.pvec(dtm.madison.test,mu)
#============================================================================
# Step 6. Tree classification
# 6.1 use rpart classification with Gini impurity coefficient splits
# make training and test set y=0 if Madision, y=1 if Hamilton
dat.train <- as.data.frame(rbind(dtm.hamilton.train, dtm.madison.train))
dat.test <- as.data.frame(rbind(dtm.hamilton.test, dtm.madison.test))
names(dat.train) <- names(dat.test) <- as.vector(dictionary$word)
dat.train$y <- as.factor(c(rep(1, nrow(dtm.hamilton.train)), rep(0, nrow(dtm.madison.train))))
dat.test$y <- as.factor(c(rep(1, nrow(dtm.hamilton.test)), rep(0, nrow(dtm.madison.test))))
# fit by rpart
tree.gini <- rpart(y~., data=dat.train, parms=list(split='gini'))
pred.tree.gini <- predict(tree.gini, newdata = dat.test, type="class")
# cross table of predicted and true
xtabs(~pred.tree.gini+dat.test$y)
mean(pred.tree.gini==dat.test$y)
mean(pred.tree.gini[dat.test$y==1]==0)
mean(pred.tree.gini[dat.test$y==0]==1)
plot(tree.gini, margin=0.1, main=" Tree (Gini)")
text(tree.gini, pretty=TRUE, fancy=TRUE)
# 6.2 with information gain splits, to predict the author
tree.info <- rpart(y~., data=dat.train, parms=list(split='infomation'))
pred.tree.info <- predict(tree.gini, newdata = dat.test, type="class")
xtabs(~pred.tree.info+dat.test$y)
# proportion classified correctly, the proportion of false negatives, and the proportion of false positives
mean(pred.tree.info==dat.test$y)
mean(pred.tree.info[dat.test$y==1]==0)
mean(pred.tree.info[dat.test$y==0]==1)
plot(tree.info, margin=0.1, main="Tree (Info)")
text(tree.info, pretty=TRUE, fancy=T)
#============================================================================
Step 7. Regularized logistic regression 
# center and scale data first
mean.train <- apply(dat.train[,-4876], 2, mean)
sd.train <- apply(dat.train[,-4876], 2, sd)
#standardize training x
x.train <- scale(dat.train[,-4876])
x.train[,sd.train==0] <- 0
#standardize test x
x.test <- scale(dat.test[,-4876], center = mean.train, scale=sd.train)
x.test[,sd.train==0] <- 0
y.train <-dat.train$y
y.test <- dat.test$y
# 7.1 fit ridge logistic model
library(glmnet)
fit.ridge <- cv.glmnet(x.train, y.train, family="binomial", alpha = 0, standardize=F)
plot(fit.ridge)
coef.opt.ridge <- coef(fit.ridge, s="lambda.min")
# 10 most important words according to the model along with their coefficients
coef.opt.ridge[order(abs(coef.opt.ridge[,1]), decreasing=T),][1:11]
pred.ridge <- predict(fit.ridge,newx=x.test, s="lambda.min", type="class")
xtabs(~pred.ridge+y.test)
# proportion classified correctly, the proportion of false negatives, and the proportion of false positives
mean(pred.ridge==dat.test$y)
mean(pred.ridge[dat.test$y==1]==0)
mean(pred.ridge[dat.test$y==0]==1)
# 7.2 fit lasso regression model
fit.lasso <- cv.glmnet(x.train, y.train, family="binomial", alpha = 1, standardize=F)
plot(fit.lasso)
coef.opt.lasso <- coef(fit.lasso, s="lambda.min")
# 10 most important words according to the model along with their coefficients
coef.opt.lasso[order(abs(coef.opt.lasso[,1]), decreasing=T),][1:10]
pred.lasso <- predict(fit.lasso,newx=x.test, s="lambda.min", type="class")
xtabs(~pred.lasso + y.test)
mean(pred.lasso==dat.test$y)
mean(pred.lasso[dat.test$y==1]==0)
mean(pred.lasso[dat.test$y==0]==1)








