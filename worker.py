import argparse
import json
import os
import subprocess
import sys
import time

import pika

QUEUE = "train-requests"
DLQ = "train-requests.dlq"
DELIVERY_LIMIT = 3

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

def declare(ch):
    """Sender and worker must declare the queues identically"""
    ch.queue_declare(queue=DLQ, durable=True)
    ch.queue_declare(queue=QUEUE, durable=True, arguments={
        "x-queue-type": "quorum",
        "x-delivery-limit": DELIVERY_LIMIT,
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": DLQ,
    })

def send(args):
    body = {"run": args.run, "data": args.data, "c": args.c}
    if args.crash:
        body["crash"] = args.crash

    with connect() as conn:
        ch = conn.channel()
        declare(ch)
        ch.confirm_delivery() # wait until the queue confirms it stored the message
        ch.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=json.dumps(body),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json")
        )
    print(f"Sent: {body}")

def reject(ch, method, reason):
    # requeue=False sends the message to the dead-letter queue.
    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    print(f"{reason}; sent to dead-letter queue", flush=True)

def handle(ch, method, properties, body):
    first = not method.redelivered
    try:
        req = json.loads(body)
        run, data, c = req["run"], req["data"], req["c"]
    except (ValueError, KeyError, TypeError) as e:
        reject(ch, method, f"Rejected malformed message: {e}")
        return

    crash = req.get("crash")
    previous = (properties.header or {}).get("x-delivery-count", 0)
    print(f"Received {run} (delivery {previous + 1}, "
          f"redelivery={method.redelivered})", flush=True)

    # Failure injection
    if crash == "always-crash":
        print("Simulated poison message: crashing before acknowledging", flush=True)
        os._exit(1)

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
        reject(ch, method, f"Training failed for {run}")

def consume():
    conn = connect()
    ch = conn.channel()
    declare(ch)
    ch.basic_qos(prefetch_count=1) # one request at a time
    ch.basic_consume(queue=QUEUE, on_message_callback=handle)
    print(f"Waiting for training requests on '{QUEUE}'", flush=True)
    ch.start_consuming()

def show_dlq():
    count = 0
    with connect() as conn:
        ch = conn.channel()
        declare(ch)
        while True:
            method, props, body = ch.basic_get(queue=DLQ, auto_ack=False)
            if method is None:
                break
            count += 1
            headers = props.headers or {}
            reason = headers.get("x-first-death-reason")
            if reason is None:
                reason = (headers.get("x-death") or [{}])[0].get("reason")
            print(f"{count} message(s) in {DLQ}; left in place")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    s = sub.add_parser("send")
    s.add_argument("run")
    s.add_argument("data")
    s.add_argument("--c", type=float, default=1.0)
    s.add_argument("--crash", choices=["ack-then-crash", "train-then-crash", "always-crash"])

    sub.add_parser("consume")
    sub.add_parser("dlq")

    args = parser.parse_args()
    if args.mode == "send":
        send(args)
    elif args.mode == "dlq":
        show_dlq()
    else:
        consume()

if __name__ == "__main__":
    main()
