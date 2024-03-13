import os
import torch
import logging
# import tensorflow
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import pandas as pd
import io
#import pyodbc
import numpy as np

from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer
from pathlib import Path
import random

# Transformer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation, LoggingHandler, util, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from scipy.special import softmax
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Path where store the auxiliar files generated during training process (this is only for training)

def create_folder(root):
    now = datetime.now()
    dt_suffix = now.strftime("%m.%d.%Y..%H.%M")
    training_model_process_folderpath = os.path.join(root, f"cls_checkpoints_{dt_suffix}")
    Path(training_model_process_folderpath).mkdir(parents=True, exist_ok=True)
    return training_model_process_folderpath

def preprocess_function(examples, tokenizer):
    return tokenizer(list(examples["Text"].values), truncation=True, padding=True)

class AcinDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, extra_encodings=None):
        self.encodings = encodings
        self.labels = labels
        self.extra_encodings = extra_encodings
        
    def __getitem__(self, idx):
        if self.labels[idx] == -1:
            # Random extra negative examples 
            nextra = len(extra_encodings['input_ids'])
            rdn = random.randint(0,nextra-1)            
            item = {key: torch.tensor(self.extra_encodings[key][rdn]) for key in self.extra_encodings.keys()}
            item['labels'] = torch.tensor(0)
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def sentence_pairs_generation(sentences, labels, pairs):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sentences)):      
        currentSentence = sentences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList==label)[0][0]])
        posSentence = sentences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0)) # Pair them up

        negIdx = np.where(labels != label)[0]
        negSentence = sentences[np.random.choice(negIdx)] # Filter only examples with a negative label
        # prepare a negative pair of images and update our lists
        pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0)) # Pair up a positive sentence with a negative one
  
        # return a 2-tuple of our image pairs and labels
    return (pairs)

def train_random_init(model_checkpoint,
                      train,
                      test,
                      extra = None,
                      train_extra_samples = 1000,
                      batch_size = 16, 
                      epochs = 10, 
                      warmup_steps=100, # number of warmup steps for learning rate scheduler
                      weight_decay=0.01, # strength of weight decay
                      learning_rate=1.5e-6,
                      adam_epsilon=1e-8,
                      seed= 15,
                      out_dir=None):

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        
    extra_encodings = None
    if extra is not None: 
        extra_encodings = preprocess_function(extra, tokenizer)        # We encode all the extra examples
        df_sample = extra.sample(n=train_extra_samples, replace=False) # But we only use a portion in the training
        train = pd.concat([train, df_sample])       # We added this portion only to add label None 
                                                    # (for each iteration the AcinDataset is gonna take one example different)
        train = train.sample(frac=1) # Randomize
        

    train_encodings = preprocess_function(train, tokenizer)
    test_encodings = preprocess_function(test, tokenizer)

    train_labels = [int(label) if label is not None else -1 for label in list(train["Label"].values)]
    test_labels = [int(label) for label in list(test["Label"].values)]    

    train_dataset = AcinDataset(train_encodings, train_labels, extra_encodings)
    test_dataset = AcinDataset(test_encodings, test_labels)

    num_labels = len(train.Label.unique())

    training_model_process_folderpath = create_folder(out_dir)
    print(f"Checkpoints generated in: {training_model_process_folderpath}")

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    eval_batch_size = batch_size
    
    training_args = TrainingArguments(
        output_dir=training_model_process_folderpath,   # output directory
        num_train_epochs=epochs,                            # total number of training epochs
        learning_rate=learning_rate, 
        warmup_steps=warmup_steps,                    
        weight_decay=weight_decay, 
        adam_epsilon=adam_epsilon,
        seed= seed,
        per_device_train_batch_size=batch_size,         # batch size per device during training
        per_device_eval_batch_size=eval_batch_size,        # batch size for evaluation
        logging_dir=os.path.join(training_model_process_folderpath,'./logs'),            # directory for storing logs
        load_best_model_at_end=False,
        metric_for_best_model='f1',
        greater_is_better = True,
        save_strategy="epoch",
        evaluation_strategy = "epoch"
    )


    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,
    )


    trainer.train()

    return trainer

def train_random_init_setfit(model_checkpoint,
                      train,
                      final_path, # location for storing the model weights
                      interim_path, # location for checkpoint saves
                      model_max_len= 256, # token input length limit
                      batch_size = 16, # training batch size
                      epochs = 10, # number of epochs train the model for
                      warmup_steps=100, # number of warmup steps for learning rate scheduler
                      weight_decay=0.01, # strength of weight decay
                      seed= 15, # seed for reproducibility of results
                      num_training= 32, # number of training points to create the sentence-pair training set
                      retraining = False, # to indicate if a new sentence transformer should be loaded from the model hub
                      num_itr = 3, # number of iterations for sampling for the sentence-pair training set
                      **kwargs): # fine-tuned model to load       
    
    model_foldername = kwargs.get('model_foldername', '')
    
    if retraining == False:
        model = SentenceTransformer(model_checkpoint) # Instantiate the underlying language model
    else:
        model = SentenceTransformer(os.path.join(final_path , model_foldername))
    model.max_seq_length = model_max_len
    
    # Create the sentence pairs for the sentence transformer
    # Equal samples per class training
    training_set = pd.concat([train[train['Label']==0].sample(num_training), train[train['Label']==1].sample(num_training)])

    x_train = training_set['Text'].values.tolist() # Extract the sentence text
    y_train = training_set['Label'].values.tolist() # Extract the labels

    train_examples = [] 
    for x in tqdm(range(num_itr)):
        train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)
        
    # Create an evaluation dataset
    X_train, X_eval = train_test_split(train_examples, test_size=0.15, random_state=seed)
    train_dataset = SentencesDataset(X_train, model)
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])

    # Read the dataset - Load the sentence pairs
    train_dataloader = DataLoader(train_examples, batch_size, shuffle=True) 
    
    # Training loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    start_epoch = 0
    num_epochs = epochs
    
    # Final model save path
    model_save_path = 'training_acin-'+ model_checkpoint +'-label_training_size_' + str(num_training) + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(final_path, model_save_path)

    # Checkpoint save path
    interim_model_path = 'training_acin-'+ model_checkpoint +'-label_training_size_'+ str(num_training) + '_'  + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    interim_model_path= os.path.join(interim_path, interim_model_path)

    # Development set: Measure correlation between cosine score and gold labels
    logging.info("ACIN dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(X_eval, name='acin-dev')

    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Model save path: {}".format(model_save_path))
    logging.info("Number of epochs: {}".format(num_epochs))
    logging.info("Model sequence-len: {}".format(model.max_seq_length))
        
    training_start = timer()
    
    # Fit the ST model
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=epochs,
              evaluation_steps=15000,
              warmup_steps=warmup_steps,
              weight_decay=weight_decay,
              output_path=model_save_path,
              checkpoint_path= interim_model_path,
              checkpoint_save_total_limit= 3,
              save_best_model=True,
              show_progress_bar=True
             )

    training_end = timer()
    time_elapsed = timedelta(seconds=training_end - training_start)
    print(f"The amount of training time required on A100 GPU is {time_elapsed.total_seconds()} seconds.")
    
    ## To continue with the Logistic Regression training and predictions
    
    return model

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_predictions(texts, model, tokenizer, gpu_id='cuda', batch_size=8):
    predictions = []
    for bch in tqdm(batch(texts, batch_size)):

        tbch = tokenizer(bch, truncation=True, padding=True, max_length=250, return_tensors="pt")

        tbch.to(gpu_id)

        outputs = model(**tbch)
    #     predictions.extend(outputs.logits.argmax(-1))
        predictions.extend([softmax(logit.to("cpu").detach().numpy()) for logit in outputs.logits])
    return predictions

def get_predictions_ensemble(texts, models, tokenizers, gpu_id='cuda', batch=8):
    out = None
    n = 0
    for m,t in zip(models, tokenizers):
        new_pred = get_predictions(texts, m, t, gpu_id=gpu_id, batch_size=batch)
        if out is None:
            out = np.array(new_pred)
            n = 1
        else:
            out = out + new_pred
            n = n+1
    
    out = out/n
    
    return out

def get_predictions_setfit(train, test, st_model, seed, max_iterations = 100):
    """ This function trains a Logistic Regression classifier using the encodings from the fine-tuned sentence
        transformer and subsequently makes label predictions. """
    
    LR = LogisticRegression(random_state= seed, max_iter = max_iterations)
    model = st_model
    
    x_train = train['Text'].values.tolist() 
    y_train = train['Label'].values.tolist() 
    x_test= test['Text'].values.tolist() 
    y_test = test['Label'].values.tolist() 
    
    # Measure the inference time
    inference_start = timer()
    
    X_train = model.encode(x_train)
    X_test = model.encode(x_test)
    
    inference_end = timer()
    time_elapsed = timedelta(seconds=inference_end - inference_start)
    print(f"The amount of inference time required on A100 GPU is {time_elapsed.total_seconds()*1000} milliseconds.")
    
    # Train the logistic regression
    LR.fit(X_train, y_train)
    
    # Make predictions for training and test set
    y_pred_train = LR.predict(X_train)
    y_pred_test = LR.predict(X_test)
    prob_pred_train = LR.predict_proba(X_train)
    prob_pred_test = LR.predict_proba(X_test)
    
    # Extract prediction logits
    probs_train = np.array(prob_pred_train)
    probs_test = np.array(prob_pred_test)
    
    # Extract prediction labels
    predictions_train = list(y_pred_train)
    predictions_test = list(y_pred_test)
    
    return predictions_train, probs_train, predictions_test, probs_test

def describe_evaluation(y_true, predictions):
    # Get the labels to evaluate (Label = None is False)
    y_pred = [bool(pred.argmax()) for pred in predictions]
    
    eval_acc = accuracy_score(y_true, y_pred)
    
    #Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    #Now the normalize the diagonal entries
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #eval_acc_per_class = cm.diagonal()
    #precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,pos_label=True)
    #print(f"F1 : {f1}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    #print(f"Evaluation accuracy per False : {eval_acc_per_class[0]}")
    #print(f"Evaluation accuracy per True : {eval_acc_per_class[1]}")
    
    print(f"Evaluation accuracy : {eval_acc}")
    
    
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=False)

    precision, recall, thresholds = precision_recall_curve(y_true, [pred[1] for pred in predictions], pos_label=True)
    lr_f1, lr_auc = f1_score(y_true, y_pred, pos_label=True), auc(recall, precision)

    pyplot.figure()
    # summarize scores
    print('F1=%.3f AUC=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = y_true.count(True) / len(y_true)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    return eval_acc
    
def plot_calibration(y_true, predictions, label=0):
    conf = []
    acc = []
    val = []
    nn = []
    for i in np.arange(0, 1, 0.1):
        pos = np.where(np.logical_and(predictions[:,label] >= i, predictions[:,label] < i+0.1))[0]
        if len(pos) > 0:
            mean_conf = predictions[pos, label].mean()
            mean_acc = sum(y_true[pos] == label)/len(pos)
        else:
            mean_conf = i + 0.05
            mean_acc = 0

        nn.append(len(pos))
        conf.append(mean_conf)
        acc.append(mean_acc)
        val.append(i)

    # the histogram of the data
    pyplot.bar(val, acc, width=0.1, facecolor='g', align='edge')#, density=True, , alpha=0.75)


    pyplot.xlabel('Confidence')
    pyplot.ylabel('Accuracy')
    pyplot.grid(True)
    pyplot.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='r')
    pyplot.show()
    
    # Calculate expected calibration error
    cal_error = (nn*np.abs(np.array(conf)-np.array(acc))).sum()/np.sum(nn)
    
    return cal_error


def plot_confidence_inc(y_true, predictions):
    confidence = np.max(predictions, axis=1)
    cacc = []
    x = []
    for i in np.arange(0.5, 1, 0.05):
        cc = np.where(confidence > i)[0]
        acc = accuracy_score(np.argmax(predictions[cc, :], axis=1), np.array(y_true)[cc]) # The argmax returns the 
        x.append(i)                                                         # predicted label from the predictions
        cacc.append(acc)
        pyplot.plot(x, cacc)
        
def train_random_init_setfit_auto(model,
                      model_checkpoint,
                      train,
                      final_path, # location for storing the model weights
                      interim_path, # location for checkpoint saves
                      model_max_len= 256, # token input length limit
                      batch_size = 16, # training batch size
                      epochs = 10, # number of epochs train the model for
                      warmup_steps=100, # number of warmup steps for learning rate scheduler
                      weight_decay=0.01, # strength of weight decay
                      seed= 15, # seed for reproducibility of results
                      num_training= 32, # number of training points to create the sentence-pair training set
                      num_itr = 3): # number of iterations for sampling for the sentence-pair training set   
    
    model.max_seq_length = model_max_len
    
    # Create the sentence pairs for the sentence transformer
    # Equal samples per class training
    training_set = pd.concat([train[train['Label']==0].sample(num_training), train[train['Label']==1].sample(num_training)])

    x_train = training_set['Text'].values.tolist() # Extract the sentence text
    y_train = training_set['Label'].values.tolist() # Extract the labels

    train_examples = [] 
    for x in tqdm(range(num_itr)):
        train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)
        
    # Create an evaluation dataset
    X_train, X_eval = train_test_split(train_examples, test_size=0.15, random_state=seed)
    train_dataset = SentencesDataset(X_train, model)
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])

    # Read the dataset - Load the sentence pairs
    train_dataloader = DataLoader(train_examples, batch_size, shuffle=True) 
    
    # Training loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    start_epoch = 0
    num_epochs = epochs
    
    # Final model save path
    model_save_path = 'training_acin-'+ model_checkpoint +'-label_training_size_' + str(num_training) + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(final_path, model_save_path)

    # Checkpoint save path
    interim_model_path = 'training_acin-'+ model_checkpoint +'-label_training_size_'+ str(num_training) + '_'  + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    interim_model_path= os.path.join(interim_path, interim_model_path)

    # Development set: Measure correlation between cosine score and gold labels
    logging.info("ACIN dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(X_eval, name='acin-dev')

    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Model save path: {}".format(model_save_path))
    logging.info("Number of epochs: {}".format(num_epochs))
    logging.info("Model sequence-len: {}".format(model.max_seq_length))
        
    training_start = timer()
    
    # Fit the ST model
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=epochs,
              evaluation_steps=15000,
              warmup_steps=warmup_steps,
              weight_decay=weight_decay,
              output_path=model_save_path,
              checkpoint_path= interim_model_path,
              checkpoint_save_total_limit= 3,
              save_best_model=True,
              show_progress_bar=True
             )

    training_end = timer()
    time_elapsed = timedelta(seconds=training_end - training_start)
    print(f"The amount of training time required on A100 GPU is {time_elapsed.total_seconds()} seconds.")
    
    ## To continue with the Logistic Regression training and predictions
    
    return model

def setfit_data_augmentation(external_dataset, # dataset to augment with
                         no_external_training, # number of examples to add to the original training sample
                         model,
                         train,
                         train_aug,
                         test,
                         final_path,
                         interim_path,
                         epochs,
                         num_training,
                         num_itr,
                         random_state, # to reproduce the randomisation
                         model_checkpoint,
                         max_iterations = 100):
   
    # Limiting the inference time for predictions
    if len(external_dataset)>10000:
        external_dataset = external_dataset.iloc[:10000,:]
    
    augmented_training_set = train_aug
    
    # Evaluate to calculate the accuracy score on the external dataset and the test
    pred_train, probs_train, pred_external, probs_external = get_predictions_setfit(train = augmented_training_set, 
                                                                                    test = external_dataset, 
                                                                                    st_model = model, 
                                                                                    seed = random_state,
                                                                                    max_iterations = max_iterations)
    
    print("Evaluation results on external dataset \n Phase 1")
    plt.rcParams["figure.figsize"] = (5,5)
    y_true_external = [False if l is None else l for l in list(external_dataset.Label.values)]
    describe_evaluation(y_true_external, probs_external)
    
    pred_train, probs_train, pred_test, probs_test = get_predictions_setfit(train = augmented_training_set, 
                                                                                    test = test, 
                                                                                    st_model = model, 
                                                                                    seed = random_state,
                                                                                    max_iterations = max_iterations)
    
    print("Evaluation results on test set \n Phase 1")
    plt.rcParams["figure.figsize"] = (5,5)
    y_true_test = [False if l is None else l for l in list(test.Label.values)]
    describe_evaluation(y_true_test, probs_test)
    
    # Extract the accuracy
    eval_acc_test = accuracy_score(y_true_test, pred_test)
    
    # Sort the external dataset
    external_dataset['Predicted_labels'] = pred_external
    external_dataset['Probability_neg'] = probs_external[:,0]
    external_dataset['Probability_pos'] = probs_external[:,1]
    augmented_df = external_dataset.sort_values('Probability_pos', ascending= False)
    
    # Augment using sorted logic
    additional_examples = augmented_df.iloc[0:no_external_training, :]
    train_aug = pd.concat([train, additional_examples])
    train_aug = train_aug.sample(frac=1) 

    print(f'Number of training examples : {len(train_aug)}')
    print(f'\t - True examples : {len(train_aug[train_aug["Label"]==True])}')
    print(f'\t - False examples : {len(train_aug[train_aug["Label"]==False])}')
    print(f'\t\t > Augmented examples : {no_external_training}') 
        
    # Fine tune the sentence transformer on randomly-created augmented dataset
    model = train_random_init_setfit_auto(model = model,
                         model_checkpoint = model_checkpoint,
                         train= train_aug,
                         final_path = final_path,
                         interim_path = interim_path,
                         epochs = epochs,
                         num_training = num_training,
                         num_itr = num_itr)

    # Repeat the evaluation process
    pred_train, probs_train, pred_external, probs_external = get_predictions_setfit(train = train_aug, 
                                                                                    test = external_dataset, 
                                                                                    st_model = model, 
                                                                                    seed = random_state,
                                                                                    max_iterations = max_iterations)
    
    print("Evaluation results on external dataset \n Phase 2")
    plt.rcParams["figure.figsize"] = (5,5)
    y_true_external = [False if l is None else l for l in list(external_dataset.Label.values)]
    describe_evaluation(y_true_external, probs_external)
    
    pred_train, probs_train, pred_test, probs_test = get_predictions_setfit(train = train_aug, 
                                                                                    test = test, 
                                                                                    st_model = model, 
                                                                                    seed = random_state,
                                                                                    max_iterations = max_iterations)
    
    print("Evaluation results on test set \n Phase 2")
    plt.rcParams["figure.figsize"] = (5,5)
    y_true_test = [False if l is None else l for l in list(test.Label.values)]
    describe_evaluation(y_true_test, probs_test)
    
    # Extract the accuracy
    eval_acc_test_new = accuracy_score(y_true_test, pred_test)
    accuracy_delta = eval_acc_test_new - eval_acc_test
    
    return accuracy_delta

def train_transformer_setfit(model_checkpoint,
                      train,
                      final_path, # location for storing the model weights
                      interim_path, # location for checkpoint saves
                      model_max_len= 256, # token input length limit
                      batch_size = 16, # training batch size
                      epochs = 10, # number of epochs train the model for
                      warmup_steps=100, # number of warmup steps for learning rate scheduler
                      weight_decay=0.01, # strength of weight decay
                      seed= 15, # seed for reproducibility of results
                      num_training= 32, # number of training points to create the sentence-pair training set
                      retraining = False, # to indicate if a new sentence transformer should be loaded from the model hub
                      num_itr = 3, # number of iterations for sampling for the sentence-pair training set
                      **kwargs): # fine-tuned model to load       
    
    model_foldername = kwargs.get('model_foldername', '')
    
    if retraining == False:
        word_embedding_model = models.Transformer(model_checkpoint) # Use an existing regular transformer
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()) # Apply a pool function over the token embeddings
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) # Join the two using the modules argument
    else:
        model = SentenceTransformer(os.path.join(final_path , model_foldername))
    model.max_seq_length = model_max_len
    
    # Create the sentence pairs for the sentence transformer
    # Equal samples per class training
    training_set = pd.concat([train[train['Label']==0].sample(num_training), train[train['Label']==1].sample(num_training)])

    x_train = training_set['Text'].values.tolist() # Extract the sentence text
    y_train = training_set['Label'].values.tolist() # Extract the labels

    train_examples = [] 
    for x in tqdm(range(num_itr)):
        train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)
        
    # Create an evaluation dataset
    X_train, X_eval = train_test_split(train_examples, test_size=0.15, random_state=seed)
    train_dataset = SentencesDataset(X_train, model)
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])

    # Read the dataset - Load the sentence pairs
    train_dataloader = DataLoader(train_examples, batch_size, shuffle=True) 
    
    # Training loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    start_epoch = 0
    num_epochs = epochs
    
    # Final model save path
    model_save_path = 'training_acin-'+ model_checkpoint +'-label_training_size_' + str(num_training) + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(final_path, model_save_path)

    # Checkpoint save path
    interim_model_path = 'training_acin-'+ model_checkpoint +'-label_training_size_'+ str(num_training) + '_'  + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    interim_model_path= os.path.join(interim_path, interim_model_path)

    # Development set: Measure correlation between cosine score and gold labels
    logging.info("ACIN dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(X_eval, name='acin-dev')

    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Model save path: {}".format(model_save_path))
    logging.info("Number of epochs: {}".format(num_epochs))
    logging.info("Model sequence-len: {}".format(model.max_seq_length))
        
    training_start = timer()
    
    # Fit the ST model
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=epochs,
              evaluation_steps=15000,
              warmup_steps=warmup_steps,
              weight_decay=weight_decay,
              output_path=model_save_path,
              checkpoint_path= interim_model_path,
              checkpoint_save_total_limit= 3,
              save_best_model=True,
              show_progress_bar=True
             )

    training_end = timer()
    time_elapsed = timedelta(seconds=training_end - training_start)
    print(f"The amount of training time required on A100 GPU is {time_elapsed.total_seconds()} seconds.")
    
    ## To continue with the Logistic Regression training and predictions
    
    return model
      