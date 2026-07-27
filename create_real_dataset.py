import pandas as pd
import random

cyberbully = [
    "you are such an idiot 😂",
    "i will kill you bro",
    "nobody likes you",
    "go die please",
    "you are trash 💀",
    "worst post ever",
    "you are so dumb",
    "stop posting nonsense",
    "you look ugly af",
    "pathetic content 🤮"
]

normal = [
    "you are amazing ❤️",
    "nice post 🔥",
    "love your content 😍",
    "great job 👏",
    "keep going bro",
    "this is awesome 💯",
    "so inspiring 😊",
    "well done 🙌",
    "respect bro",
    "you did great"
]

data = []

# 250 cyberbullying (real style)
for _ in range(250):
    text = random.choice(cyberbully)
    data.append([text, "cyberbullying"])

# 250 normal
for _ in range(250):
    text = random.choice(normal)
    data.append([text, "not_cyberbullying"])

df = pd.DataFrame(data, columns=["tweet_text", "cyberbullying_type"])
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv("cyberbullying_tweets.csv", index=False)

print("Real dataset created ✅")
