FROM yiluxiangbei/fire-smoke-detect-paddle:v1.0

WORKDIR /app
RUN rm -rf inference_model
COPY inference_model .

RUN hub convert --model_dir inference_model \
            --module_name helmet-detection \
            --module_version 1.0.0 \
            --output_dir helmet-detection-hub

RUN hub install helmet-detection-hub/helmet-detection.tar.gz
RUN mkdir config
COPY docker/config.json config
# hub serving start -c config.json
# hub serving start --modules fire-smoke-detect-paddle
# hub serving start -c config/config.json --cors True
ENTRYPOINT ["tail", "-f", "/dev/null"]
# ENTRYPOINT ["hub", "serving", "start", "-c", "config/config.json", "--cors", "True"]