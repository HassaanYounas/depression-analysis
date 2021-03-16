# from tweepy import Stream
# from tweepy import OAuthHandler
# from tweepy.streaming import StreamListener
# import json

# #consumer key, consumer secret, access token, access secret.

ckey="6on0QmhK6pVp9L5X9SKdbFbmk"
csecret="O9nief3IswxFKc0JzfHj6r4teaUDKF8K8a3TvEDE2gJNOzJi2K"
atoken="2680307251-FFvOf2HWG1UrGjkpeNmEsjqXXZ8Z4CnsRQxbQIN"
asecret="gAcoYU2kEqxQw3ZktMv9VUNWzPo6kTXoEd1lWBGxpyuo8"

# class listener(StreamListener):

#     def on_data(self, data):
#          print(data)
#             tweet = data.split(',"text":"') [1] .split(',"source":"') [0]
#             print (tweet)
#             saveThis = tweet
#             saveFile = open('twitDB.csv','a')
#             saveFile.write(data)
#             saveFile.write('\n')
#             saveFile.close()
#             return(True)

#     def on_error(self, status):
#         print (status)

# auth = OAuthHandler(ckey, csecret)
# auth.set_access_token(atoken, asecret)

# twitterStream = Stream(auth, listener())
# twitterStream.filter(track=["car"])

import requests
import os
import json

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'


def auth():
    return "AAAAAAAAAAAAAAAAAAAAALgqKQEAAAAA0aV92cJTEwr3blxfW5LUOoq4m5w%3DPYmrUTCWYp85NIMJbXnih9Vf7aWaX8nyAbfZA1Z2EJlUFfzOdc"


def create_url():
    query = "from:twitterdev -is:retweet"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    tweet_fields = "tweet.fields=author_id"
    url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}".format(
        query, tweet_fields
    )
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    bearer_token = auth()
    url = create_url()
    headers = create_headers(bearer_token)
    json_response = connect_to_endpoint(url, headers)
    print(json.dumps(json_response, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()