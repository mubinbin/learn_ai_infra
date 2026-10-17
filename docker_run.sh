docker run --rm \
    --name iris-api \
    -p 127.0.0.1:8000:8000 \
    -e MODEL_RUN_ID="$MODEL_RUN_ID" \
    --mount "type=bind,source=$PWD/artifacts/$MODEL_RUN_ID,target=/app/artifacts/$MODEL_RUN_ID,readonly" \
    iris-api:local