Judy Wang

Write Up
1. What are words that, specifically for the collection in data/obits.json, appear in a lot of documents and are thus not helpful query terms?
    When passing in the obits.json data the top ten words in the entire document are [(',', 62600), ('the', 52792), ('of', 33066), ('and', 27118), ('in', 23584), ('a', 22373), ('to', 21938), ('was', 17102), ('his', 13109), ('he', 12644)]. 
    With the top token being a comma followed by pronouns and stop words, it is apparent that these words do not depict any meaningful information besides that the text is certained around a person that uses he/him pronouns. These are not very helpful query information 
    because they are not unique and do not describe the underlying meaning of the text. 

2. Modify the main method in tfidf.py so that you convert Karen Spark Jones’ obituary into a document vector based on the vocabularies developed in training. Based on tf-idf, what words are indicative of her obituary? Why might you think these words are indicative of Karen Spark Jones’s obituary.
    After modifying the main method in tfidf.py, the top ten words are computer, computers, programming, document, seminal, researchers, search, engines, language, and her. These words are indicative of her obituary because she made a significant 
    impact on discovering seminality of language through programming on computers, which makes sense why these are prominate words of the text. 

TF-iDF
1. Why do you think it might be better to use sklearn (or another libraries) implementation compared to your own?
    It might be better to use sklearn or another libraries because their implementation has the ability to tailor the results to your liking, which would take a long time if you were to implement it yourself.
    For example, the TF-iDF function has different parameters such as max_df, min_df, stop_words, use_idf and norm. If you were to do it otherwise, then it would take a lot of time and effort. In addition you 
    avoid the risk of misinterpreting the algorithm, but if it is already in a library you know that it was optimized and looked over by multiple people if it was made by a reliable and creditable library such as sklearn.

2. Make sure you understand what is happening in each cell. Starting at the 4th cell (the one with the from sklearn.feature_extraction.text import TfidfVectorizer line), write a brief comment explaning what is happening in each cell.
    [Also added all of these comments in the ipynb as well]
    Cell 4 (from sklearn.feature_extraction.text import TfidfVectorizer ...): 
        Creates a TfidfVectorizer function that ignore terms that have a document frequency strictly higher than .65 and lower than 1, ignore 
        stop words, enables inverse-document-frequency reweighting, and normalize by having the sum of squares of vectors to 1. Using this function, it then 
        vectorizes all of the documents. 

    Cell 5 
        Transforms all of the vectors into arrays and then get the length of that array to verify if we have the same number of arrays as documents that were read in.
    
    Cell 6 
        Import pandas so we can use its data structures and create the data where all of the tf-idf calculations will lie. 
    
    Cell 7
        Get the list of the vocabularies of all the documents and vectorize them and put the term and the tf-idf score into the correlating document csv. 

3. Choose any obituary and compare the document representation in that notebook with the document representation for the same obituary from your tf-idf.py implementation. What differences do you see?
    I choose 0101.txt and the document representation looks very different from the sklearn document represetation because the testing data is different. In this way, there are way less
    documents to look over. When looking at the highest score, mine returned director, while the sklearn one returned hoover. This makes sense because what are "unqiue" terms for my model 
    might not be the same in the sklearn one. In addition, the sklearn one cut off words that were high and on the low end, so their top terms would show the essense of what the document meant, but mine does not have a cut off for high terms. 

4. What changes do you think you’d have to make to the settings of the TfidVectorizer object in the notebook to make the results be more similar to the results in your implementation.
    I would have a min threshold of 2, eliminate the low threshold, allow stop words, and not normalize the vector so that the sklearn would have similar scores to mine.

Feedback
How long did you spend on the assignment?
Maybe 10 hours. 

What did you learn by doing this assignment?
I learned a lot. I was able to understand what tf-idf what SVD is on a deeper level. I learned how to trace backwards and thread difference aspects of my understanding together to complete the assignment. I think it 
was the right amount of challenging. 

Briefly describe a Computational Text Analysis/Text as Data research question where using tf-idf would be useful (keep this answer to a maximum of 3 sentences).
How does our speech evolve through time? By analyzing Facebook post throughout time, we can analyze how language has evolved with the evolution of the internet and development of social media. It would be interesting
to crossover other social media sites like myspace, tumblr, and twitter to accumulate more information. 

Feel free to add additional/optional feedback, e.g. what did you like or dislike about the assignment?
I liked this assignment because it made me think deeper about my understanding of concepts and clarify misunderstandings I had about concepts. 
That being said, I will say the combination of the readings and this assignment is a lot of work. I like that it wasn't due on the same day like it was before, 
but if HW2 and Reading 2 were both due on Monday, I don't think I would have worked so throughout on this assignment. I probably would have just done the assignment for 
the sake of doing it rather than taking my time to fully understand the code I was seeing and solidify my understanding.




