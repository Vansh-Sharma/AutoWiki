# AutoWiki
## Automatic Wikipedia Article Generation

### Purpose

Project simulates the automatic generation of encyclopedia-like articles for given topics, specifically directed for niche computer science ones. 

### Deployment

Simply clone the repository into a workspace. With this done modifications to original code can be made. Furthermore, if you wish to use the program, import Automatic Wikipedia Article Generation.py into a file and from this import the createArticle function. Pass any query as a parameter and the output will be a string representing the article. 

### Usage

Follow the above steps. The createArticle function serves as a client facing function allowing access to the produced output all other functions help build. By simply calling this inside another file, one has access to the results of this program. 

### Requirements and Dependencies

Beyond the two other files within the repository, to run the main file requires the following dependencies: 
- nltk
- BeautifulSoup
- requests
- googlesearch
- torch 
- transformers
- Sentence_transformers
- spacy
Most other import statements are related to the above main ones. Pip installing the latest versions of these packages will suffice. 

### System Architecture

The program is split into three different files, the main one being Automatic Wikipedia Article Generation.py. This houses most of the functions, including the main createArticle function which calls all other necessary ones. createArticle relies on entity extraction from getEntities and getMostCommonEntity, extractive summarization from wiki_Sum.py, and web scraping from web_scraper.py. These two files do as they are named: wiki_Sum.py providing an extractive summary using word frequencies and sentence scoring and web_scraper.py providing visible text extracted from given websites. These two functionalities are split into different files from the main one mainly because of their importance and my desire to edit them independently to keep things organized. Many of the functions in Automatic Wikipedia Article Generation.py, however, can be used as replacements for the other two files, but the other two offer more efficient ways of solving their respective problems. For example, Automatic Wikipedia Article Generation.py has an extractive summarization function named extractiveSummary which does a similar thing as wiki_Sum, but wiki_Sum incorporates weights which makes the process more accurate. Thus, Automatic Wikipedia Article Generation.py with a few adjustments can be used as a standalone program if needed. 
 
### Codebase Organization

The code in the repository is organized into three files. Automatic Wikipedia Article Generation.py provides the main file with a function that can be called in another program to produce Wikipedia-like articles. wiki_Sum.py and web_scraper.py hold important functions which are imported by Automatic Wikipedia Article Generation.py. 

### Description

Given a query, the code searches for K number of websites, scraps their visible information, and then passes it to an extractive summarizer. With this output, the program compares it to a valid standard to check its relevance using cosine similarity scores. Named entity extraction is then applied to the string and the most frequent entity is pulled out to act as a section header. The top X most relevant passages are outputted with their most frequent entity being their header, producing the appearance of Wikipedia articles. All of this takes place in the createArticle function.

### Limitations and Improvements

A major way to improve the output includes applying entity extraction to all scraped data before applying extractive summarization. Then, taking the top K entities and web searching them as queries to produce extensive output on a certain topic. This also handles the assumption that information on our original query may not exist and thus we can use related topics to build an article for the original one. This also handles the problem of repetition as we can handle what information we get in the entity extraction phase. Furthermore, we can consider applying abstractive models to reduce sentence redundancy. 
