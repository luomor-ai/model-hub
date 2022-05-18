FROM yiluxiangbei/fire-smoke-detect-paddle:v1.0

WORKDIR /app
RUN rm -rf inference_model
COPY inference_model .

RUN hub convert --model_dir inference_model \
            --module_name helmet-detection \
            --module_version 1.0.0 \
            --output_dir helmet-detection-hub

RUN hub install helmet-detection-hub/helmet-detection.tar.gz
# hub serving start --modules fire-smoke-detect-paddle
# CMD ["tail", "-f", "/dev/null"]
ENTRYPOINT ["hub", "serving", "start", "--modules", "fire-smoke-detect-paddle,helmet-detection-hub"]