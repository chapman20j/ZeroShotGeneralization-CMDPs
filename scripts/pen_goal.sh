#!/bin/bash

ENV="pen_goal"
SAMPLES=10
CONCURRENT=5
EPOCHS=1000
DELAY=10
SEQUENTIAL=1
EVALINTERVAL=20
VECTORIZATION=8

if [ $SEQUENTIAL -eq 1 ]; then
    python main.py --env $ENV --samples $SAMPLES --cse 0 --dr 0 --max_concurrent $CONCURRENT --epochs $EPOCHS --vectorization $VECTORIZATION --custom_eval --eval_interval $EVALINTERVAL
    python main.py --env $ENV --samples $SAMPLES --cse 1 --dr 0 --max_concurrent $CONCURRENT --epochs $EPOCHS --vectorization $VECTORIZATION --custom_eval --eval_interval $EVALINTERVAL
    python main.py --env $ENV --samples $SAMPLES --cse 0 --dr 1 --max_concurrent $CONCURRENT --epochs $EPOCHS --vectorization $VECTORIZATION --custom_eval --eval_interval $EVALINTERVAL
else
    python main.py --env $ENV --samples $SAMPLES --cse 0 --dr 0 --max_concurrent $CONCURRENT --epochs $EPOCHS --vectorization $VECTORIZATION  --custom_eval --eval_interval $EVALINTERVAL &
    sleep $DELAY
    python main.py --env $ENV --samples $SAMPLES --cse 1 --dr 0 --max_concurrent $CONCURRENT --epochs $EPOCHS --vectorization $VECTORIZATION  --custom_eval --eval_interval $EVALINTERVAL &
    sleep $DELAY
    python main.py --env $ENV --samples $SAMPLES --cse 0 --dr 1 --max_concurrent $CONCURRENT --epochs $EPOCHS --vectorization $VECTORIZATION  --custom_eval --eval_interval $EVALINTERVAL &
fi

