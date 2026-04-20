from Transformers import BertTokenizer, BertModel

# load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
#Tokenize a sample imput 

text="Transformers are powerful models for NLP tasks."
inputs = tokenizer(text, return_tensors='pt')

#Pass the input through the BERT model
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # Output shape: (batch_size, sequence_length, hidden_size)