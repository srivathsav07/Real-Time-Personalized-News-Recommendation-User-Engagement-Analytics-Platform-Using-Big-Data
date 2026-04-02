from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'news-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest'
)

print("Consumer started...")

for message in consumer:
    print(message.value.decode('utf-8'))