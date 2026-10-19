import argparse
import json
import os
import subprocess
import sys
import time

import pika

QUEUE = "train-requests"

def connect():
    params = pika.ConnectionParameters(
        host=os.environ.get("RABBIT_HOST", "rabbitmq"),
        credentials=pika.PlainCredentials(os.environ["RABBIT_USER"], os.environ["RABBIT_PASS"])
    )
    for _ in range(30):
        try:
            return pika.BlockingConnection(params)
        except pika.exceptions.AMQPConnectionError:
            print("Queue not reachable yet; retry in 2s", flush=True)
            time.sleep(2)
    sys.exit("Could not connect to the queue")

def send(args):
    body = {"run": args.run, "data": args.data, "c": args.c}
    if args.crash:
        body["crash"] = args.crash

    with connect() as conn:
        ch = conn.channel()
        ch.queue_declare(queue=QUEUE, durable=True)
        ch.confirm_delivery() # wait until the queue confirms it stored the message
        ch.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=json.dumps(body),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json")
        )
    print(f"Sent: {body}")

def handle(ch, method, properties, body):
    first = not method.redelivered
    try:
        req = json.loads(body)
        run, data, c = req["run"], req["data"], req["c"]
    except (ValueError, KeyError, TypeError) as e:
        print(f"Rejected malformed message: {e}", flush=True)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    crash = req.get("crash")
    print(f"Received {run} (redelivery={method.redelivered})", flush=True)

    # Failure injection for the next stage. Each crash happens only on
    # the first delivery, so a redelivery can show what heppens next.
    if crash == "ack-then-crash" and first:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Acknowledeged BEFORE training; crashing now", flush=True)
        time.sleep(1) # Let the ack reach the queue
        os._exit(1)

    result = subprocess.run([
        sys.executable, "train.py",
        "--c", str(c), "--run-name", run, "--data", data,
    ])

    if crash == "train-then-crash" and first:
        print("Trained; crashing BEFORE acknowledging", flush=True)
        os._exit(1)

    if result.returncode == 0:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Acknowledged {run}", flush=True)
    else:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        print(f"Training failed for {run}; message discarded", flush=True)

def consume():
    conn = connect()
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1) # one request at a time
    ch.basic_consume(queue=QUEUE, on_message_callback=handle)
    print(f"Waiting for training requests on '{QUEUE}'", flush=True)
    ch.start_consuming()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    s = sub.add_parser("send")
    s.add_argument("run")
    s.add_argument("data")
    s.add_argument("--c", type=float, default=1.0)
    s.add_argument("--crash", choices=["ack-then-crash", "train-then-crash"])

    sub.add_parser("consume")

    args = parser.parse_args()
    if args.mode == "send":
        send(args)
    else:
        consume()

if __name__ == "__main__":
    main()
