FROM yiluxiangbei/all-paddle-hub:v1.1

WORKDIR /app
RUN rm -rf best_model
COPY best_model best_model

RUN hub convert --model_dir best_model \
            --module_name oral-detection \
            --module_version 1.0.0 \
            --output_dir oral-detection-hub

RUN hub install oral-detection-hub/oral-detection.tar.gz
RUN mkdir config
COPY docker/config.json config
# hub serving start -c config.json
# hub serving start --modules fire-smoke-detect-paddle
# hub serving start -c config/config.json
# CMD ["tail", "-f", "/dev/null"]
ENTRYPOINT ["hub", "serving", "start", "-c", "config/config.json"]