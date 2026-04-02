from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

categories = ['sports', 'news', 'finance', 'entertainment', 'health', 'technology']

print("Producer started...")

while True:
    event = {
        "UserID": f"U{random.randint(1,100)}",
        "NewsID": f"N{random.randint(1,500)}",
        "Category": random.choice(categories),
        "Click": random.choice([0,1])
    }

    producer.send('news-topic', value=event)
    print(json.dumps(event))

    time.sleep(1)