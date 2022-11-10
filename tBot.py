# Should instal tweepy
# pip install tweepy==4.10.1


import tweepy

consumer_key = 'BI9PK5YrIGqarth9MKs499B8J'
consumer_secret = 'aovhMmmp9LmjcLJqUn2JaekTL8IByiOorO3jGI4XOf75xcyOg4'
access_token = '1589819209575698432-otNSUl4YPgXzFCF7sll3ZzjCcdV8Fc'
access_token_secret = 'MPWWPZa0u6y8r6cpcdLC6NOXuZ5BZoaN9jX4utXnhUPyg'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAC9AjAEAAAAAYsxpicwvSqqnYCLj2KiaLePJ8VM%3DZWyWRfeSk2vAcgUJaWVbAHsjwN9fy8yV1kZ5rxZgv07ohQ5C0S'

client = tweepy.Client(bearer_token=bearer_token)
client = tweepy.Client(
    consumer_key=consumer_key, consumer_secret=consumer_secret,
    access_token=access_token, access_token_secret=access_token_secret
)

keyword = "gun ownership"

api_result = []

tweet = client.search_recent_tweets(query = keyword, max_results=100, user_auth=True)

for tw in tweet:
    for t in tw:
        print(t)
        print("\n")
