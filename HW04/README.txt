Judy Wang 

Analysis
What is the role of the learning rate?
    The role of the learning rate is so that the machine learning algorithm does not make too big or small of a step or jump when changing it's betas. If the learning rate is too big
    then there is the probability that your beta is not the best since the algorithm is always taking such big jumps around the potential, most optimal beta. If the learning rate is too small
    than it might take forever for the converge to get the best beta. 

How many passes over the data do you need to complete for the model to stabilize? The various metrics can give you clues about what the model is doing, but no one metric is perfect.
    The metric I used was accuracy. After 1063 passes, the model stabilized and the lp started to converge. I stopped iterating after the accuracy started to decrease. The other metrics that
    I also tried out was to see when the lg started to converge and have simular numbers to those around it. I found out that this was not a good method because the lg values are pretty close togother
    from the start. I also tried this method for accuracy, but the accuracies are pretty similar to it around it from the start as well. 

Pretend you need to explain to your boss what features you use to discover a post is about baseball or hockey. How would you answer this question? What words are the best predictors of each class? How (mathematically) did you find them?
    I would look at the top features that have the highest confidence that it is either baseball or hockey. In this case, the lower the p value was the most confidence we were that the word is correlated to the topic. 
    I approached this question in two methods and recieved different results. 

    In my first attempt, I went through every example and extracted the word that had the highest count and associated that p value with that word. From doing this, the top ten words were:
    Hockey: ['team', 'goal', 'pts', 'people', 'arena', 'six', 'went', 'lot', 'way', 'period']
    Baseball: ['runs', 'lock', 'show', 'well', 'learn', 'stance', 'see', 'fielding', 'swing', 'good']

    In my second attempt, I added all of the probabilities of a word from all of the examples and then pick the lowest ones. This is different from the first implementation because each word 
    is associated with the p value, not just those that have the highest word count in that example. I thought this was a better approach because we lose less information in the process. On the other hand, 
    it also gives words that appear less often a advantage since their total value would be higher. 
    The values that were most associated with the topics are:
    Hockey: ['percent', 'weakest', 'bounces', 'derogatory', 'explosive', 'heap', 'praise', 'rushing', 'singled', 'taught']
    Baseball: ['estimate', 'trigger', 'users', 'bear', 'court', 'miles', 'paragraph', 'strategy', 'supporting', 'admittedly']

What words aren’t good predictors of either class? How (mathematically) did you find them?
    I used the same method as above, but chose the highest values associated with each words. These are my results.

    First implementation:
    Hockey: ['list', 'ching', 'exactly', 'jake', 'huot', 'scored', 'wangr', 'o', 'anything', 'reading']
    Baseball: ['ching', 1, 'names', 'newsletter', 'jerseys', 'kbos', 'question', 'roster', 'trade', 'tonight']

    Second Implementation:
    Hockey: ['year', 'good', 'like', 'know', 'team', 'would', 'one', 'game', 'article', 'writes']
    Baseball: ['anyone', 'games', 'one', 'good', 'like', 'think', 'would', 'year', 'article', 'writes']

Feedback 
How long did you spend on the assignment?
    Maybe 7 hours. I was very confused the 1-2 hours because I was trying to discover how use_tfidf was involved in the implementation. 
    I also tried multiple implementations of different things for the questions. 

What did you learn by doing this assignment?
    I got better at other people's code. I learned how weights are involved in calculations of stochastic gradient ascent and how it is used to 
    increase the log likeihood. 

Briefly describe a Computational Text Analysis/Text as Data research question where using LDA would be useful (keep this answer to a maximum of 3 sentences).
    You can use this information to discover if a sentence was made by ChatGPT and see if the words are more associated with human written styles or a chat bot like 
    ChatGPT. 
