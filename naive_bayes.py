import pandas as pd 
import numpy as np 
import re

# Dataset import for data folder
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
X_train = train['Abstract']
y_train = train['Category']
X_test = test['Abstract']

# Processing the data- lowercase, only keeping alphanumerics and removing the special characters and tokenization
def preprocess(sentence):
    output=[]
    sentence=sentence.lower() 
    pattern = re.compile(r'\b\w+\b')
    for word in re.findall(pattern, sentence):
        output.append(word)   
    return output

class Naive_Bayes():
        
    def fit(self, data, label):
        
        # find the class labels
        self.label_count = np.unique(label)
        self.count = 0
        self.bow = {}
        self.word_list = {}
        
        # Creating the bag of words
        # find the number of samples in each class
        for i in label:
            if i not in self.bow:
                self.bow[i]= {'count_samples':0 ,'count_words': 0}
        
        for i in label:
            self.bow[i]['count_samples']+=1
    
        for i,j in enumerate(data):
            #proprocess the data
            cleaned_data = preprocess(j)
            
            for word in cleaned_data:
                # get the count of unique words for laplace smoothing
                if word not in self.word_list:
                    self.word_list[word]=1
                    self.count+=1
                    
                # get count of total number of words in each class and the count of each word in each class
                self.bow[label[i]]['count_words']+=1
                if word not in self.bow[label[i]]:
                    self.bow[label[i]][word]=0
                self.bow[label[i]][word]+=1
                
        # Prior probability of each class P(c)
        # Denominator value of each class = Total number of words in each class + Total number of unique words + 1
        # These two values are precomputed to save time during test time
        self.P_c = np.empty(len(self.label_count))
        self.deno = np.empty(len(self.label_count))
        for i,j in enumerate(self.label_count):
            # Convert prior probability to log value to prevent underflow
            self.P_c[i] = np.log(self.bow[j]['count_samples']/len(label))            
            self.deno[i] = self.bow[j]['count_words']+ self.count+1
            
        self.pre_prob = {}
        
        # Store the prior probability and denominator in a dictionary for easy usage
        for i,j in enumerate(self.label_count):
            self.pre_prob[j]= {'P_c':self.P_c[i] ,'deno':self.deno[i]}      
        
                
    def transform(self, test_data):
        outputs= []
        label_count = self.label_count
        pre_prob = self.pre_prob
        bow = self.bow
        for i in test_data:
            probability = np.empty(len(label_count))
            cleaned_text = preprocess(i)
            for j,k in enumerate(label_count):
                
                # Prior Probability
                prob = pre_prob[k]['P_c']
                # Denominator
                denominator = pre_prob[k]['deno']
                
                # Count of each test word from the stored bag of words
                for word in cleaned_text:
                    if word not in bow[k]:
                        count_word=0
                    else:
                        count_word=bow[k][word]
                    
                    # 1 is added to numerator. Here 1 is the value of laplace transform
                    # Convert to log to prevent underfow 
                    # This is the log of likelihood and it is added to 
                    # the log of prior probability to get posterior probability
                    post = np.log((count_word+1)/denominator)
                    prob += post
                
                # Probability of each class is stored
                probability[j]=prob
            
            # Find the label with the maximum probability
            classification = label_count[np.argmax(probability)]
            outputs.append(classification)
            
        return np.array(outputs)
    
NB_test = Naive_Bayes()

# Fit the model to training data
NB_test.fit(X_train, y_train)

# Predict the output
y_pred = NB_test.transform(X_test)

# Convert the predictions to proper format to upload to kaggle
sub =  open('nb_submission.csv','w+')
sub.write('Id,Category\n')
for index, prediction in enumerate(y_pred):
    sub.write(str(index) + ',' + prediction + '\n')
sub.close()