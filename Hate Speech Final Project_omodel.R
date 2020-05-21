###############################################################################
###############################################################################
##                                                                           ##
##   Duke University - Masters of Interdisciplinary Data Science             ##
##   IDS 703 - Data Scraping and Text Analysis                               ##
##   Instructor: Dr. Chris Bail                                              ##
##   Team Members: Joe Littell, Emma Sun, Chang Shu, Julia Oblasova          ##
##                                                                           ##
##   This project is meant to test if a given Tweet is racist.               ##
##   In order to do so we will utilize screen scraping to pull in a          ##
##   steam of random tweets (approximately XXX in total), a list of          ##
##   known slurs, and utilize machine learning to determine if a tweet       ##
##   is using hatespeech. In order to do so we will utilize sentiment        ##
##   analysis, demographic data through twitter profiles, and other methods  ##
##   to build an accurate and efficient model.                               ##
##                                                                           ##
###############################################################################
###############################################################################

# Libraries needed
library(rtweet)                # General-purpose Twitter API package
library(tidytext)              # General-purpose data wrangling
library(rvest)                 # Parsing of HTML/XML files  
library(stringr)               # String manipulation
library(rebus)                 # Verbose regular expressions
library(lubridate)             # Eases DateTime manipulation
library(dplyr)                 # grep/text cleaning
library(topicmodels)           # General Purpose Topic Modeling
library(stm)                   # Special Topic Modeling
library(ggplot2)               # Visualization Package
library(caret)                 # Multi-purpose Machine Learning
library(kernlab)               # Kernel-based machine learning
library(tm)                    # Corpus Manipulation
library(splitstackshape)       # Another data manipulation/cleaning package
library(e1071)                 # Support Machine Vectors 
library(SnowballC)             # Stemming and corpus manipulation
library(wordcloud)             # Simple creation of word clouds


###############################################################################
#                                                                             #
#               Required Twitter credentialing for API use                    #
#                                                                             #
###############################################################################

# This is Joe Littell's API Information
app_name          <- "Duke Research"
consumer_key      <- "p68khdmCHhOnBXKWm71MCKbLh"
consumer_secret   <- "9pH2UidhR1LFyvEDRDE7Cveeshfy2AGPcJcD57uye8URXL9W9I"
access_token      <- "1036689188798771201-6O3if3ZOzStP6PVnoYV2nlXt9PuAtT"
access_secret     <- "rHW1HpsIHYj5JYOEXfJkwsJpZ53o92r8Z4WHSEibS8cTP"

#creating the token from our specific keys and secrets
create_token(
  app             <- app_name,
  consumer_key    <- consumer_key,
  consumer_secret <- consumer_secret,
  access_token    <- access_token,
  access_secret   <- access_secret,
  set_renv        <- TRUE
) -> twitter_token

###############################################################################
#                                                                             #
#               The Following in creating the needed information              #
#               used in the end analysis. This will utilize API               #
#               and screen scraping using the rvest and tidytext              #
#               libraries to build our data sets.                             #
#                                                                             #
###############################################################################

# In order to intially flag tweets for hate speech we will need to build criteria to 
# look at the words and context in order to compare against.

# The first, and arguably most straight forward method is to build a database of racial slurs
# Luckily a number of webages have this information for free.

# The Racial slur Database (RSDB) at http://www.rsdb.org/full
# is the most robust with 2655 slurs. Not all of these are explicit, but it is the most complete.
# We will use the "rvest" package to pull the table into a dataframe

SlurDB <- read_html('http://www.rsdb.org/full')

SlurDB %>%
  html_node("slur_2 td") %>%        # Pull information from the HTML node titled "slur_2 td"
  html_text()                       # convert the node into text

# After we pull the data we convert it to and HTML table for manipulation using rvest

SlurDB %>%
  html_nodes("table") %>%           # convert the node/text to a table
  .[[1]] %>%
  html_table() %>%                  # convert the table to a data frame
  select(Slur) -> Slurs             # only select the column Slur and save it as "Slurs" 

# Then we select the slur column and unnest the tokens 
# so that we can more to text preprocessing

# Finally we will write this to a CSV file so that we have it for later in case the site 
# changes drastically enough that our screen scraping is rendered useless

write.csv(Slurs, file = "RSDB.csv")

# Initually we were planning on using a dataset of random tweets read from the stream.
# This became an issue with identifying racist tweets due to the limited amount of tweets 
# which were being pulled every 15 seconds over the course of an hour

# In our attempt to utilize a dictionary based search, we realized that a number of words would
# not be useful as they are generally words used in normal conversation (I.E Apple and Banana)
# to adjust for this we then culled the dictionary to only pick out the most explicit slurs
# first (Ni****, Chi**, Sp**) in order to pull definately racist tweets for our machine learning
# algorithm to use.

Slurs <- read.csv("file:///C:/Users/Joseph/Documents/R/RSDB v2.csv")

Slurs <- Slurs %>%                                          # look at only the slur comulmn in file
  select(Slur) 

slur_tweets<-as.data.frame(NULL)                            # empty data-frame to store tweets


for(i in 1:nrow(Slurs)){                                    # Loop through all Slurs (length 405(Originally2655))
  tweets <- search_tweets(as.character(Slurs$Slur[i]),      # Search each slur word 
                        n = 10,                             # Pull the first 10 matches
                        include_rts = FALSE,                # Do not pull retweets, only originals
                        "lang:en",                          # Only english language tweets
                        geocode = lookup_coords("usa"),     # from the United States
                        type ="recent")                     # only the most recent tweets are to be selected
  
  # add the tweet that meats the above criteria into a data frame for storage.
  slur_tweets <- rbind(slur_tweets, tweets)                   
}

# Perminantly save the tweets to a CSV file
write.csv(slur_tweets$text, file = "slur_tweets.csv")


# Pulling in an hour worth of random tweets from the stream to analyize
RandomTweet <- stream_tweets(lookup_coords("USA"),              # only pulling data from the USA
                             "lang:en",                         # only English Language Tweets
                             timeout = 60 * 60,             		# 60 seconds times 60 minutes
                             file_name = "RandomTweets.csv",  	# data also saved to csv 
                             parse = FALSE)

write.csv(RandomTweet$text, file = "slur_tweets.csv")


###############################################################################
#                                                                             #
#               The following is the supervised machine learning              #
#               for dectecting racism and hate speech in Tweets               #
#               utilizing support vector machines (caret::ksvm)               #
#                                                                             #
###############################################################################

# read in the traning data from file
TrainingDoc <- readLines("file:///C:/Users/Joseph/Documents/R/RTrain.txt")

# read the training document into a corupus
train <- Corpus(VectorSource(TrainingDoc))

# cleaning the data
train <- tm_map(train, content_transformer(stripWhitespace))        # removes whitespace from corpus
train <- tm_map(train, content_transformer(tolower))                # convert the corpus to lowercase
train <- tm_map(train, content_transformer(removeNumbers))          # removes numbers from corpus
train <- tm_map(train, content_transformer(removePunctuation))      # removes punctuation from corpus

# convert the training courpus to a document term matrix
train.dtm <- as.matrix(DocumentTermMatrix(train, control=list(wordLengths=c(1,Inf))))

# read in the testing data from file
TestingDoc <- readLines("file:///C:/Users/Joseph/Documents/R/RTest.txt")

# read the testing document into a corupus
test <- Corpus(VectorSource(TestingDoc))

# cleaning the data 
test <- tm_map(test, content_transformer(stripWhitespace))          # removes whitespace from corpus
test <- tm_map(test, content_transformer(tolower))                  # convert the corpus to lowercase
test <- tm_map(test, content_transformer(removeNumbers))            # removes numbers from courpus
test <- tm_map(test, content_transformer(removePunctuation))        # removes punctuation from corpuse

# convert the testing corpus to a document term matrix
test.dtm <- as.matrix(DocumentTermMatrix(test, control=list(wordLengths=c(1,Inf))))

# determining the intersectiong between the two corpuses (corpii?)
train.df <- data.frame(train.dtm[,intersect(colnames(train.dtm), colnames(test.dtm))])
test.df  <- data.frame(test.dtm[,intersect(colnames(test.dtm), colnames(train.dtm))])

# label the corpus portions correctly for racist vs non-racist
label.df           <- data.frame(row.names(train.df))
colnames(label.df) <- c("filenames")
label.df           <- cSplit(label.df, 'filenames', sep="_", type.convert=FALSE)
train.df$corpus    <- label.df$filenames_1
test.df$corpus     <- c("Neg")

# run the Support Vector Machine(ksvm) on the training set.
df.train           <- train.df
df.test            <- train.df
df.model           <- ksvm(corpus~., data= df.train, kernel="rbfdot")

# run the SVM model created above on the test set
df.pred            <- predict(df.model, df.test)

# create a confusion matrix of the results
con.matrix         <- caret::confusionMatrix(as.factor(df.pred), as.factor(df.test$corpus))

# print the confusion matrix to see the accuracy of the model 
print(con.matrix)  # of 60 examples in the test set, it correctly identified 

df.test            <- test.df
df.pred            <- predict(df.model, df.test)
results            <- as.data.frame(df.pred)
rownames(results)  <- rownames(test.df)
print(results)


##################################################


happy          <- readLines("file:///C:/Users/Joseph/Documents/R/nonracist_train.txt")
sad            <- readLines("file:///C:/Users/Joseph/Documents/R/racist_train.txt")
happy_test     <- readLines("file:///C:/Users/Joseph/Documents/R/nonracist_test.txt")
sad_test       <- readLines("file:///C:/Users/Joseph/Documents/R/racist_test.txt")

tweet          <- c(happy, sad)
tweet_test     <- c(happy_test, sad_test)
tweet_all      <- c(tweet, tweet_test)
sentiment      <- c(rep("happy", length(happy) ), 
                    rep("sad", length(sad)))
sentiment_test <- c(rep("happy", length(happy_test) ), 
                   rep("sad", length(sad_test)))
sentiment_all  <- as.factor(c(sentiment, sentiment_test))

library(RTextTools)

# naive bayes
mat            <- create_matrix(tweet_all, 
                                language="english", 
                                removeStopwords=FALSE, 
                                removeNumbers=TRUE, 
                                stemWords=FALSE, 
                                tm::weightTfIdf)

mat            <- as.matrix(mat)

classifier     <- naiveBayes(mat[1:120,], 
                             as.factor(sentiment_all[1:120]))
predicted      <- predict(classifier, 
                          mat[60:120,]); predicted

table(sentiment_test, predicted)
recall_accuracy(sentiment_test, predicted)Copy


# the other methods
mat            <- create_matrix(tweet_all, 
                                language="english", 
                                removeStopwords=FALSE, 
                                removeNumbers=TRUE, 
                                stemWords=FALSE, 
                                tm::weightTfIdf)

container      <- create_container(mat, 
                                   as.numeric(sentiment_all),
                                   trainSize=1:120, 
                                   testSize=60:120,
                                   virgin=FALSE) #removeSparseTerms

models         <- train_models(container, 
                               algorithms=c("MAXENT",
                                            "SVM",
                                            "SLDA",
                                            "BAGGING", 
                                            "RF", 
                                            "TREE"))


# test the model
results        <- classify_models(container, models)

table(as.numeric(as.numeric(sentiment_all[60:120])), results[,"FORESTS_LABEL"])

recall_accuracy(as.numeric(as.numeric(sentiment_all[60:120])), results[,"FORESTS_LABEL"])

###############################################################################
#                                                                             #
#               Topic Modeling stuff below to ensure I knew what I            #
#               was doing.                                                    #
#                                                                             #
###############################################################################


slur_tweets_for_eval <- slur_tweets %>%           # create a new dataframe for analysis
  select(created_at,text) %>%                     # selects Created_at and Text columns from tweets
  unnest_tokens("word", text)                     # unnests the words to allow

# load the stop words data frame
data("stop_words")

# remove stop words from the text of tweets
slur_tweets_for_eval <- slur_tweets_for_eval%>%            
  anti_join(stop_words) %>%
  count(word) %>%
  arrange(desc(n))

# Remove additional common information
slur_tweets_for_eval <-                                  
  slur_tweets_for_eval[-grep("https|t.co|amp|rt|1|2|day|de|don't|el|en|I'm|it's|la|lol|love|people|se|time|video|youtube",
                             slur_tweets_for_eval$word),]

# Choosing only the top 20 words present once common words are removed
top_20 <- slur_tweets_for_eval[1:20,]

# Factor to sort by frequency
slur_tweets_for_eval$word <- factor(slur_tweets_for_eval$word,
                                    levels = slur_tweets_for_eval$word[order(slur_tweets_for_eval$n,decreasing=TRUE)])


ggplot(top_20, aes(x=word, y=n, fill=word))+                 # Plot the top_20 words where each word is a new color
  geom_bar(stat="identity")+                                 #
  theme_minimal()+                                           # Keeping the theme minimal
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+  # Rotate the words so they are perpendicular to X axis
  ylab("Number of Times Word Appears in racist tweets")+     # y label
  xlab("")+                                                  # X label
  guides(fill=FALSE)                                         # no legend


###############################################################################
#                                                                             #
#               Determining the most common slurs against blacks              #
#                                                                             #
###############################################################################

#read in selected slur words. I further filtered Joe's selection. Filter out some ambiguous words like banana. 
slurwords <- read.csv("file:///C:/Users/Joseph/Documents/R/RSDB top3s.csv")

#loop to concatenate and search posts
#concatenate first 20 keywords into a giant string
str1              <- '"'
space             <- " "
ORword            <- "OR"
totalstringloop   <- ''

for(word in slurwords$Slur[]){
  slurwordstring  <- paste(str1,word,str1,space, ORword,space,sep="")
  totalstringloop <- paste(totalstringloop,slurwordstring)
}
#remove the last OR to get final totalstring
totalstringloop   <- substring(totalstringloop,1,nchar(totalstringloop)-3)
#check number of characters in the string, maximum character length is 500 for search_tweet function
nchar(totalstringloop)

#search_tweet with stringed keyword
racisttweet       <- search_tweets(q=totalstringloop,
                                   "lang:en",
                                   geocode=lookup_coords("usa"),
                                   n=10000, include_rts=FALSE, 
                                   type="recent", 
                                   retryonratelimit=TRUE)

#before this step, i already pulled down tweets containing racial slurs, I call the tweets dataset trulyracisttweet
trulyracisttweet   <- blackracisttweet

#tokenize and see top frequency words, sentiment and topic
tidy_trulyracisttweet <- trulyracisttweet%>%
  dplyr::select(created_at,text) %>%
  unnest_tokens("word",text)

#remove stop words
data("stop_words")
tidy_trulyracisttweet <- tidy_trulyracisttweet %>%
  anti_join(stop_words)

#remove numbers
tidy_trulyracisttweet <- tidy_trulyracisttweet[-grep("\\b\\d+\\b", 
                                                     tidy_trulyracisttweet$word),]

#remove whitespaces
tidy_trulyracisttweet$word <- gsub("\\s+",
                                   "",
                                   tidy_trulyracisttweet$word)

# stemming to the root words
tidy_trulyracisttweet <- tidy_trulyracisttweet %>%
  mutate_at("word", funs(wordStem((.), language="en")))

#change slurword list to lower letters for easier comparison
#this is important, because our slur database use capital letters, and grep won't recognize unless you change it to lower
slurwords$Slur <- tolower(slurwords$Slur)

#count frequency of each slur word
#create empty dataframe which contains slur word and its count in the 18000 tweets of 10 keywords
#make stringasfactors=false or there will be some problem
slurvocab <- data.frame(vocab = character(nrow(slurwords)), 
                        count=numeric(nrow(slurwords)), 
                        stringsAsFactors = FALSE)
i <- 1

for(each in slurwords$Slur){
  slurvocab$vocab[i]=each
  #grep returns the indices of rows containing the slur
  rownumber=grep(each,tidy_trulyracisttweet$word)
  #number of rows will be the frequencies of the slur
  slurvocab$count[i]=length(rownumber)
  i <- i+1
}

#Barplot
selectedslurvocab=slurvocab%>%
  arrange(desc(count)) %>%
  top_n(10)

ggplot(selectedslurvocab, aes(reorder(vocab,-count), count, fill=factor(vocab))) + 
  geom_bar(stat="identity")+
  labs(x="slurvocab", y="count",
    title="Top 10 slurwords for Blacks in 18000 random tweets",
    caption="\nSource: Data collected from Twitter's REST API via rtweet")


#time series plot of tweets
ts_plot(racisttweet, "hours") + 
  labs(x = "Date and time",
       y = "Frequency of tweets",
       title = "Time series of HateSpeech tweets",
       subtitle = "Frequency of Twitter statuses calculated in one-hour intervals.") 

# locational maps
geocoded <- lat_lng(racisttweet)
library(maps)

par(mar = c(0, 0, 0, 0))
maps::map("state", lwd = .25)
with(geocoded, points(lng, lat, pch = 20, cex = .75, col = rgb(0, .3, .7, .75)))

summary(geocoded)

