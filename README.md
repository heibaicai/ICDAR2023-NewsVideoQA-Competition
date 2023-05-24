## Training / Testing BERT QA models on NewsVideoQA

For finetuning pretrained BERT models, we use simpletransformers which in turn is based on transformers package. 

### Installing simpletransformers
For installing simpletransformers please follow original instructions We dont use fp16 while finetuning BERT models in our experiments. You need not install Apex if you dont want to use fp16 training.

### Fine tune a pretrained BERT model on DocVQA
We have experimented with three different pre-trained models and performed model fusion. Here are the names of the models:
- bert-large-cased-whole-word-masking-finetuned-squad
- deepset-bert-large-uncased-whole-word-masking-squad2
- bert-large-uncased-whole-word-masking-finetuned-squad
### Data
For each video, we organize the content into SQUAD format. We utilize OCR, ASR, and text tracking technologies to create a comprehensive textual representation of each video. We use Whisper for ASR, and our own developed models for OCR and text tracking.

### Model
This is one of our trained models, ready for inference purposes.
[model](https:)
### Train
``` bash train.sh ```
### Test / Inference
``` bash test.sh ```
### Ensemble
``` bash ensemble.sh ```
- We have trained numerous models, and model ensemble is achieved by utilizing a voting mechanism among multiple models to obtain the final output. Due to the large size and the significant number of models, we are only uploading one of them as an example.On the validation set, the metrics achieved are an accuracy of **50.42%** and an ANLS of **65.42%**.
- Applying model ensemble techniques, which involved simple voting among the top 6 best-performing models, we further improved the performance to an accuracy of **51.68%** and an ANLS score of **66.83%** on the same validation set.