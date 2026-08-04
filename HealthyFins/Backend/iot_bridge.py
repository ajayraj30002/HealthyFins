import os
import json
import ssl
import paho.mqtt.client as mqtt
from kafka import KafkaProducer

# Pull secrets from Render Environment Variables
HIVEMQ_URL = os.getenv("HIVEMQ_URL")
HIVEMQ_USER = os.getenv("HIVEMQ_USER")
HIVEMQ_PASS = os.getenv("HIVEMQ_PASS")

AIVEN_KAFKA_SERVER = os.getenv("AIVEN_KAFKA_SERVER")
AIVEN_KAFKA_USER = os.getenv("AIVEN_KAFKA_USER")
AIVEN_KAFKA_PASS = os.getenv("AIVEN_KAFKA_PASS")

# Lazy initialize Kafka Producer so it doesn't crash if vars are missing on boot
producer = None

def get_kafka_producer():
    global producer
    if producer is None and AIVEN_KAFKA_SERVER:
        try:
            producer = KafkaProducer(
                bootstrap_servers=AIVEN_KAFKA_SERVER,
                security_protocol="SASL_SSL",
                sasl_mechanism="PLAIN",
                sasl_plain_username=AIVEN_KAFKA_USER,
                sasl_plain_password=AIVEN_KAFKA_PASS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except Exception as e:
            print(f"❌ Failed to init Kafka Producer: {e}")
    return producer

def on_mqtt_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        print(f"📥 Bridge received from HiveMQ: {payload}")
        
        prod = get_kafka_producer()
        if prod:
            prod.send("healthyfins-telemetry", value=payload)
            prod.flush()
            print("🚀 Forwarded to Aiven Kafka!")
    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

def start_mqtt_bridge():
    if not all([HIVEMQ_URL, HIVEMQ_USER, HIVEMQ_PASS, AIVEN_KAFKA_SERVER, AIVEN_KAFKA_USER, AIVEN_KAFKA_PASS]):
        print("⚠️ Missing MQTT/Kafka env vars. Bridge will not start.")
        return

    print("🌉 Starting MQTT-to-Kafka Bridge Thread...")
    mqtt_client = mqtt.Client(client_id="HealthyFins_Bridge")
    mqtt_client.username_pw_set(HIVEMQ_USER, HIVEMQ_PASS)
    mqtt_client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect(HIVEMQ_URL, 8883, 60)
        mqtt_client.subscribe("healthyfins/tank_1/telemetry")
        mqtt_client.loop_start()  # Runs securely in background
        print("✅ MQTT Bridge connected and listening!")
    except Exception as e:
        print(f"❌ MQTT Connection failed: {e}")
