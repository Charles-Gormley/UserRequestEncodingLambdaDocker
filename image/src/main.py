################# Libraries #################
import os
import json
import logging
from datetime import datetime, timedelta
from time import time
import math

from encoder import encode_single_article
from transformers import BertTokenizer, BertModel
from recommender import Recommendations

import boto3
import torch

################# Config #################
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Connecting to s3")
s3_client = boto3.client('s3')
embedding_bucket = "toast-encodings"
embedding_file = "embeddings.pth"
embedding_lambda_file = "/tmp/"+embedding_file

logging.info("Connecting to lambda")
lambda_client = boto3.client('lambda')



################# Loading Classes #################
# Encoder
logging.info("loading in bert models")
tokenizer = BertTokenizer.from_pretrained("tokenizer/")
model = BertModel.from_pretrained("model/")
model.eval()

# Recommendation System
recommend = Recommendations()



def calculate_requests(length:float, format:str, num_topics:int, time_per_article:int):
    '''
        length: minutes
        time_per_article: seconds
    '''

    reading_wpm = 300 # Newsletter. 
    talking_wpm = 200 # Podcast - Based on popular ted talks 

    requests_per_topic = int(((length/num_topics)*60)/time_per_article)
    requests_per_topic = 1 if requests_per_topic == 0 else requests_per_topic # Setting to 1 if unreasonable. 
    if format == "Newsletter":
        target_word_count = reading_wpm * (time_per_article/60)

    else:
        target_word_count = talking_wpm * (time_per_article/60)

    return requests_per_topic, target_word_count
    
def load_article_encoding_store_from_s3(hours, days):
    '''This is going to hold the data for past X days of article encodings. In the future this is going to run on zillis. '''
    s3_client.download_file(embedding_bucket, embedding_file, embedding_lambda_file)
    embeddings_dict = torch.load(embedding_lambda_file)
    time_date = datetime.now() - timedelta(days=days, hours=hours)
    filtered_embeddings_dict = {k: v[embeddings_dict['unixTime'] >= time_date.timestamp()] for k, v in embeddings_dict.items()}


    content_IDs = filtered_embeddings_dict["articleID"].cpu() # Is transfering tensor to cpu necessary?
    content_embeddings = filtered_embeddings_dict["tensor"].cpu()

    content_embedding_array = content_embeddings.numpy()
    content_ID_array = content_IDs.numpy()

    return content_embedding_array, content_ID_array # Numpy arrays being returned. 
    

def userRecommendedIds(event):
    logging.debug("User Recommendation Started")
    logging.debug(f"Event For Recommendations: {event}")

    basicTopics = event['basicTopics']
    advancedTopics = event['advancedTopics']
    userId = event['userId']
    length = event['length']
    format = event['format']
    dateSinceDay = event['dateSinceDay']
    dateSinceHour = event['dateSinceHour']
    timePerArticle = event['timePerArticle']
    tone = event["tone"]

    topics = basicTopics + advancedTopics
    requests_per_topic, target_word_count = calculate_requests(length, format, num_topics=len(topics), time_per_article=timePerArticle) 

    logging.info(f"Requests Per Topic: {requests_per_topic}")
    logging.info(f"Target Word Count: {target_word_count}")
    logging.info(f"Number of Articles being recommended: {requests_per_topic*len(topics)}")
    print("Days", dateSinceDay)
    print("Hours", dateSinceHour)
    content_embedding_array, content_ID_array = load_article_encoding_store_from_s3(hours=dateSinceHour, days=dateSinceDay)
    assert len(content_embedding_array) == len(content_ID_array)
    print("Number of articles to choose from: ", len(content_embedding_array))
    recommend.add_vectors_to_index(content_embedding_array)


    encodings = []
    articles = []
    unique_ids = set()
    # Iterate over basicTopics & advancedTopics
    logging.info(f"Topics: {topics}")
    for topic in topics:
        print(f"Chosen Topic: {topic}")
        # Get Recommendations 
        encoded_article = encode_single_article(topic, model, tokenizer) # Encode the topic the user requested . 
        print(f"Encoded Article: {encoded_article}")        
        distances, indices = recommend.search(encoded_article, requests_per_topic)
        

        # Basic assertions to check if everything is functioning
        assert distances.shape[1] == recommend.k
        assert indices.shape[1] == recommend.k
        
        
         
        sub_content_ID_array = content_ID_array[indices.ravel()]
        print(f"Chosen Article indices for {topic}: {indices.ravel()}")
        print(f"Chosen Article IDs for {topic}: {sub_content_ID_array}")
        
        content_ID_list = sub_content_ID_array.tolist()
        logging.info(f"Content Identification: {content_ID_list}")
        for article_id in content_ID_list:
            if article_id not in unique_ids:
                unique_ids.add(article_id)
                article_dict = {"topic": topic, "articleID": article_id}
                logging.info(f"Article Dict: {article_dict}")
                articles.append(article_dict)
            
    logging.info(f"Article Recommendations: {articles}")
    return articles, target_word_count # Only receiving unique article ids.

def handler(event=None, context=None):
    print("Event:", event)
    logging.debug(f"Event Input in handler: {event}")
    if event == {}:
        # Boto3 check
        session = boto3.Session()
        # Attempt to retrieve credentials
        credentials = session.get_credentials()
        

        event["basicTopics"] = ["AI Engineering", "Science"]
        event["advancedTopics"] = []
        event["userId"] = 12345
        event['timePerArticle'] = 10 # seconds
        event["length"] = 5 # Minutes
        event["format"] = "Podcast"
        event["dateSinceDay"] = 2
        event["dateSinceHour"] = 3
        event["tone"] = "Business Casual"
        
    # TODO: Create s3 Path instead and pass it through
    s3_filename = f'{event["userId"]}/{int(time())}-podcast.wav'
    
    articles, target_word_count = userRecommendedIds(event)

    event["s3Path"] = s3_filename
    event['articleIdRecommendations'] = articles
    event['targetWordCount'] = target_word_count

    try:
        healthcheck = event["healthcheck"]
    except:
        healthcheck = False

    # Trigger generative podcast lambda
    lambda_function = 'ContentDeliveryLambdaDockerStac-DockerFuncF47DA747-e3iEuK6T48OO'
    print("Event Dictionary being sent: ", event)

    if not healthcheck:
        lambda_client.invoke(FunctionName=lambda_function,
                            InvocationType="Event",
                            Payload=json.dumps(event)
                            )
    
    
    
    # Convert to JSON string
    return {
        'statusCode': 200,
        'articleIdRecommendations': articles,
        'targetWordCount': target_word_count,
        'format': event["format"],
        'tone': event["tone"],
        'userId': event["userId"],
        's3Path': s3_filename
    }

if __name__ == "__main__":
    event = dict()
    healthcheck = True

    event["basicTopics"] = ["AI Engineering", "The ultimate Video Game is here."]
    event["advancedTopics"] = []
    event["userId"] = 12345
    event['timePerArticle'] = 120 # seconds 
    event["length"] = 5 # Minutes
    event["format"] = "Podcast"
    event["dateSinceDay"] = 2
    event["dateSinceHour"] = 3
    event["tone"] = "Business Casual"

    handler(event)