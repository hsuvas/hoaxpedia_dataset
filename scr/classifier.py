import torch
import transformers
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification, AutoModelForMaskedLM,DefaultDataCollator,Trainer,TrainingArguments,DataCollatorWithPadding,DataCollatorForTokenClassification
from datasets import load_dataset, load_from_disk,load_metric,Dataset,concatenate_datasets 
import numpy as np
import pandas as pd   
import torch.nn.functional as F
import json
import argparse
from os.path import join as pjoin
import os
os.environ['WANDB_DISABLED'] = 'true'
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import argparse
import logging




def compute_metric_all(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    num_classes = logits.shape[-1]
    metrics = {}

    for class_idx in range(num_classes):
        binary_labels = (labels == class_idx).astype(int)
        binary_predictions = (predictions == class_idx).astype(int)
        accuracy = accuracy_score(binary_labels, binary_predictions)
        precision = precision_score(binary_labels, binary_predictions)
        recall = recall_score(binary_labels, binary_predictions)
        f1 = f1_score(binary_labels, binary_predictions)

        metrics.update({
            f'accuracy_class_{class_idx}': accuracy,
            f'precision_class_{class_idx}': precision,
            f'recall_class_{class_idx}': recall,
            f'f1_class_{class_idx}': f1
        })
    return metrics

  

def preprocess_data(data_to_process):
    inputs = [dialogue for dialogue in data_to_process['text']]
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='longest', truncation=True)

    label_index = [0,1]
    labels = [label_index.index(label_val) for label_val in data_to_process['label']]
    
    model_inputs['labels'] = torch.tensor(labels)  

    return model_inputs

def train_model(model, output_path_model ,output_log_training, device,tokenized_data,data_collator,lr,init_weights):
 
    class WeightedCrossEntropyLoss(torch.nn.Module):
        def __init__(self, weight):
            super(WeightedCrossEntropyLoss, self).__init__()
            self.weight = weight

        def forward(self, logits, labels):
            ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)
            return ce_loss(logits, labels)
    class_weights = torch.tensor(init_weights).to(device)

    #TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_path_model,
        evaluation_strategy="no",
        learning_rate=lr, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs= 30,
        push_to_hub=False,
        logging_dir=output_log_training,
        save_total_limit=2,
        metric_for_best_model="f1",
        gradient_accumulation_steps=4,
        do_eval=False,
        warmup_steps=100,
    )
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        data_collator=data_collator,
        compute_metrics=compute_metric_all,
    )
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(trainer.get_train_dataloader()):
            model.train()
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, labels=labels)
            loss = WeightedCrossEntropyLoss(class_weights)(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            if step % training_args.logging_steps == 0:
                logging.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}") 
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                

    #Test phase
    trained_model = trainer.model
    trainer = Trainer(
        model=trained_model,
        args=TrainingArguments(output_dir=output_path_model, evaluation_strategy="no", seed=42),
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        compute_metrics=compute_metric_all
    )

    metric = trainer.evaluate()
    return trainer,metric

def save_model(trainer,tokenizer, model_name_upd,task,repo_id,token):
    token = token 
    repo_id = repo_id+model_name_upd+'_weighted_hoax_classifier_'+task
    trainer.model.push_to_hub(repo_id, use_auth_token=token)
    tokenizer.push_to_hub(repo_id,use_auth_token=token)

def test_model(model, tokenizer, batch_size, model_name,data_path,output_path):
    test_data = load_dataset("csv", data_files={"test": data_path})
    test_dataloader = torch.utils.data.DataLoader(test_data['test'], batch_size=batch_size)
    model.eval()


    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            predicted_batch_labels = torch.argmax(logits, dim=-1)
            predicted_labels.extend(predicted_batch_labels.tolist())
            true_labels.extend(batch['label'].tolist()) 

    predicted_labels_tensor = torch.tensor(predicted_labels)
    true_labels_tensor = torch.tensor(true_labels)
    true_zeros = ((predicted_labels_tensor == 0) & (true_labels_tensor == 0)).sum().item()
    true_ones = ((predicted_labels_tensor == 1) & (true_labels_tensor == 1)).sum().item()

    logging.info("Number of true 0 predictions: "+ str(true_zeros))
    logging.info("Number of true 1 predictions: "+ str(true_ones))

    predicted_labels_list = predicted_labels

    #calculate overall p/r/f1
    from sklearn.metrics import precision_recall_fscore_support
    clas_report = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    logging.info('Precision,           Recall,                F1 Score')
    logging.info(str(clas_report))

    #show the confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(test_data['test']['label'], predicted_labels)
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix of definitions For '+model_name)
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plot_path = os.path.join(output_path,'confusion_matrices')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(plot_path+'/'+model_name+'.png')
    return predicted_labels_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-train','--input_train_path',help='Path for training data', required=True)
    parser.add_argument('-test','--input_test_path',help='Path for test data', required=True)
    parser.add_argument('-output','--output_path',help='Path to save results', required=True)
    parser.add_argument('-task','--task',help='Definition/Fulltext task', required=False, default='definition')
    parser.add_argument('-h_key','--huggingface_key',help='Huggingface Token', required = True, default='')
    parser.add_argument('-repo_id','--repo_id',help='Huggingface repo id', required=True, default='')

    args = parser.parse_args()
    output_path = args.output_path
    input_train_path = args.input_train_path
    input_test_path = args.input_test_path
    task = args.task
    huggingface_key = args.huggingface_key
    repo_id = args.repo_id

    log_file = os.path.join(output_path, task+'_logfile_all_noweight.log')
    logging.basicConfig(filename=log_file,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)

    max_input = 128
    max_target = 1
    batch_size = 16
    
    models ={
        'roberta-base': 2e-05,
        'roberta-large': 2e-05,
        'bert-base-uncased': 2e-06,
        'bert-large-uncased': 2e-06,
        'albert-base-v2': 2e-06,
        'albert-large-v2': 2e-06,
        'allenai/longformer-base-4096': 2e-06,
        'allenai/longformer-large-4096': 2e-06,
    }
    


    for model_name, lr in models.items():
        logging.info('\n\n\n')
        logging.info('Started for: '+ model_name)
    
        output_path_model = os.path.join(output_path, 'models', model_name)
        os.makedirs(output_path_model, exist_ok=True)

        output_log_training = os.path.join(output_path, 'training_logs', model_name)
        os.makedirs(output_log_training, exist_ok=True)

        output_log_eval = os.path.join(output_path, 'eval_logs', model_name)
        os.makedirs(output_log_eval, exist_ok=True)

        logging.info('Initiating Process for: '+ model_name)

        model_checkpoints = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)


        dataset = load_dataset("csv", data_files={"train": input_train_path,
                                        "test" : input_test_path}) 

        tokenized_data = dataset.map(preprocess_data, batched = True,remove_columns= ['label','text'])
        data_collator = DataCollatorWithPadding(tokenizer)
        logging.info('Data and model Loaded for: '+ model_name)

        #initial weights
        df = pd.read_csv(input_train_path)
        w0 = len(df)/(2*len(df[df['label']==0]))
        w1 = len(df)/(2*len(df[df['label']==1]))
        init_weights = [w0,w1]
        logging.info('Initial weights for each class[0,1]: '+ str(init_weights))
        print(init_weights)


        trainer,metric = train_model(model,output_path_model,output_log_training, device,tokenized_data,data_collator, lr, init_weights)
        if '/' in model_name:
            model_name_upd = model_name.split('/')[1]
        else:
            model_name_upd = model_name
        
        with open(output_log_eval+'/'+model_name_upd+'_eval_log.json','w') as fp:
            json.dump(metric, fp)
        logging.info(str(metric))

        save_model(trainer,tokenizer, model_name,task,repo_id,huggingface_key)
        logging.info('Model saved')
        
        predicted_labels_list = test_model(trainer.model, tokenizer, 16, model_name, input_test_path, output_log_eval)
        # Save the list to a file (e.g., a text file)
        output_file = os.path.join(output_path,model_name_upd+'_predicted_labels.txt')
      
        with open(output_path+model_name_upd+'_predicted_labels.txt', 'w') as f:
            for label in predicted_labels_list:
                f.write(str(label) + '\n')
        logging.info("Predicted labels saved to: "+ output_file)
        logging.info('Finished for: '+ model_name)
    
    
