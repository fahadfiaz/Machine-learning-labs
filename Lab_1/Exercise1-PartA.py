#!/usr/bin/env python
# coding: utf-8

# # Utility Functions

# ## Importing libraries

# In[52]:


from collections import Counter
import matplotlib.pyplot as plt


# ## Function to count unique words in file

# In[60]:


def word_count(words):
    
    # This loop check if word does not exist in counter dictionary, then put word with count 1 in counter dictionary 
    # else increase the count of word by 1 in counter dictionary
    
    counter = {}
    for word in words: 
        if not word in counter:
            counter[word] = 1
        else:
            counter[word] += 1
    
    return counter


# ## Function to count top 5 most frequent words 

# In[61]:


def count_top_5_frequent_words(word_count_dict):
    
#     Dictionary subclass Counter used for counting occurence of words. 
#     Then applying most_common function which return list of the 10 most common elements and their counts
    
    top_5_frequent_words_dict = dict(Counter(word_count_dict).most_common(5))
    return top_5_frequent_words_dict


# ## Function to display results

# In[62]:


def show_results(title, x_label, y_label, x_axis_val, y_axis_val, style = "fivethirtyeight"):
    
    plt.style.use(style)  # using pre-defined style provided by Matplotlib.
    plt.title(title)  # Set title for the axes.
    plt.xlabel(x_label)  # Set the label for the x-axis.
    plt.ylabel(y_label)  # Set the label for the y-axis.
    plt.tight_layout(0)  # used to give padding
    plt.bar(x_axis_val, y_axis_val, color='#30475e', edgecolor="black")  # bar graph to plot chart
    plt.show()  # show chart


# ## Reading given text file and splitting lines to list of words

# In[63]:


# Open file in read mode

file = open("random_text.txt", "r") 

# Read data from file and split it into words. Here we have not passed any argument to split function. 
# In this case split function will consider consecutive whitespace as a single separator, and the result will 
# contain no empty strings at the start or end if the string has leading or trailing whitespace. Consequently, 
# splitting an empty string or a string consisting of just whitespace with a None separator returns [].

total_words = file.read().split()


# <hr style="border:2px solid gray"> </hr>

# # Part A) The number of unique words

# In[65]:


# I assume here that we have to find count of words in case sensitive manner. However if that is not the case,
# we can simply use words.lower() buildin function in word_count() function to count frequency of words in case insenstive manner.

count_of_words = word_count (total_words)

#sorting frequency of words in ascending order based on count of words. 
sorted_count_of_words = {w:count_of_words[w] for w in sorted(count_of_words, key=count_of_words.get)}

print("Unique words list:\n\n", sorted_count_of_words)


# <hr style="border:2px solid gray"> </hr>

# # Part B) The top 5 most frequent words

# In[59]:


top_5_frequent_words = count_top_5_frequent_words (count_of_words) #get dictionary of top 5 most frequent words

#sorting top 5 frequent words in ascending order based on count of words. 
sorted_top_5_frequent_words = {w:top_5_frequent_words[w] for w in sorted(top_5_frequent_words, key=top_5_frequent_words.get)}

print("The top 5 most frequent words: ", top_5_frequent_words)

show_results("The top 5 most frequent words", "Words", "Count", top_5_frequent_words.keys(), top_5_frequent_words.values())


# <hr style="border:2px solid gray"> </hr>
