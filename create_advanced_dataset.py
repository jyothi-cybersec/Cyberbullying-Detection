import pandas as pd
import random

# Strong cyberbullying
strong = [
    "i will kill you",
    "go die",
    "you are trash",
    "you are useless",
    "nobody likes you",
    "you are pathetic"
]

# Slight harmful / sarcasm / rude humor
slight = [
    "what nonsense is this",
    "this is so bad lol",
    "are you serious bro",
    "this is weird af",
    "not your best work",
    "this makes no sense",
    "bro what is this 😂",
    "this is kinda stupid",
    "i expected better",
    "this is embarrassing"
]

# Positive
positive = [
    "you are amazing",
    "nice post",
    "great job",
    "keep going",
    "love your content",
    "well done",
    "so inspiring",
    "proud of you"
]

data = []

# Cyberbullying (strong + slight)
for _ in range(1000):
    text = random.choice(strong + slight)
    data.append([text, "cyberbullying"])

# Non-cyberbullying (only positive)
for _ in range(1000):
    text = random.choice(positive)
    data.append([text, "not_cyberbullying"])

df = pd.DataFrame(data, columns=["tweet_text", "cyberbullying_type"])
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("cyberbullying_tweets.csv", index=False)

print("✅ Dataset created with slight harmful included")
