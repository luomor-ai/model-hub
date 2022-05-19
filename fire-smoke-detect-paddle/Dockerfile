FROM yiluxiangbei/paddlehub:v1.0

WORKDIR /app
RUN rm -rf inference_model
COPY inference_model .

RUN hub convert --model_dir inference_model \
            --module_name fire-smoke-detect-paddle \
            --module_version 1.0.0 \
            --output_dir fire-smoke-detect-paddle-hub

RUN hub install fire-smoke-detect-paddle-hub/fire-smoke-detect-paddle.tar.gz
# hub serving start --modules fire-smoke-detect-paddle
CMD ["tail", "-f", "/dev/null"]
# ENTRYPOINT ["hub", "serving", "start", "--modules", "fire-smoke-detect-paddle"]