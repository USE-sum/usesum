# TensorHubServing

## Installation
Please run pip install -r requirements.txt

## Usage:
  * Execute ``` python export_hub_model.py``` To save locally the model
  * To test, execute ``` saved_model_cli run --dir ../models --tag_set serve --signature_def serving_default --input_exprs 'text=["what this is"]'```

## To serve using Docker Docker:
  * ```docker-compose up```
  * To test, run: ```curl -d '{"instances": ["hello world!"]}'   -X POST http://localhost:8501/v1/models/universal_encoder:predict ```



