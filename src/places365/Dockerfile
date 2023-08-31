FROM floydhub/dl-docker:cpu

COPY . /root/caffe

WORKDIR /root/caffe
RUN mv vgg16_places365.caffemodel deploy_vgg16_places365.prototxt models_places
CMD [ "python", "PLACES365_pretrained_model_preprocessed_data.py"]
