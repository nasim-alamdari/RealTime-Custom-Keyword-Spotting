# Nasim Alamdari
# last Update March 2023
import sys

from func import transfer_learning, input_data

import tensorflow as tf
import numpy as np
import IPython
from pathlib import Path
import os
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

# for emdedding:
import pandas as pd
from pathlib import Path
import glob


def train(
    keyword: str,
    samples_dir: os.PathLike,
    embedding: os.PathLike,
    unknown_words: os.PathLike,
    background_noise: os.PathLike,
    #output: os.PathLike,
    num_epochs: int = 4,
    num_batches: int = 1,
    primary_learning_rate: float = 0.001,
    batch_size: int = 64,
    unknown_percentage: float = 50.0,
    base_model_output: str = "dense_2",
):
    """Fine-tune few-shot model from embedding representation. The embedding
    representation and unknown words can be downloaded from
    https://github.com/harvard-edge/multilingual_kws/releases
    The background noise directory can be downloaded from:
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

    Args:
      keyword: target keyword
      samples_dir: directory of 1-second 16KHz target .wav samples
      embedding: path to embedding representation
      unknown_words: path to unknown words directory
      background_noise: path to Google Speech Commands background noise directory
      output: modelname for saving the model (specified as a path)
      num_epochs: number of finetuning epochs
      num_batches: number of finetuning batches
      primary_learning_rate: finetuning LR
      batch_size: batch size
      unknown_percentage: percentage of samples to draw from unknown_words
      base_model_output: layer to use for embedding representation
    """

    assert (
        Path(background_noise).name == "_background_noise_"
    ), f"only tested with GSC _background_noise_ directory, please provide a path {background_noise}"

    for d in [samples_dir, embedding, unknown_words, background_noise]:
        assert os.path.isdir(d), f"directory {d} not found"

    #if os.path.exists(output):
    #    print(f"Warning: overwriting {output}")

    #samples = glob.glob(samples_dir + os.path.sep + "*.wav")
    samples = glob.glob(os.path.join(samples_dir , "*.wav"))
    assert len(samples) > 0, "no sample .wavs found"
    """for s in samples:
        cmd = f"soxi {s}"
        res = subprocess.check_output(shlex.split(cmd))
        out = res.decode("utf8")
        checks = ["75 CDDA sectors", "16000 samples", "00:00:01.00"]

        if not all([c in out for c in checks]):
            raise ValueError(
                f"{s} appears to not be a 16KHz 1-second wav file according to soxi \n{out}"
            )"""

    #print(f"{len(samples)} training samples found:\n" + "\n".join(samples))
    print(f"{len(samples)} training samples found:\n") 

    uftxt = "unknown_files.txt"
    unknown_words = Path(unknown_words)
    assert os.path.isfile(unknown_words / uftxt), f"{unknown_words/uftxt} not found"
    unknown_files = []
    with open(unknown_words / uftxt, "r") as fh:
        for w in fh.read().splitlines():
            unknown_files.append(str(unknown_words / w))

    
    #==========================================
    # In the first step we will split the data in training and remaining dataset
    # because we only want 5 shot for training:
    n_shot = 5
    train_percent = np.ceil((n_shot*100)/len(samples))
    print("train_percent:",train_percent)
    train_samples, X_rem= train_test_split(samples, train_size=train_percent/100, random_state=42)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    dev_samples, test_samples = train_test_split(X_rem, test_size=0.5, random_state=42)
    
    five_samples = train_samples[:5]
    
    print("size of validation and test sets:", len(dev_samples), len(test_samples))
    #==========================================
    
    print("Training model")
    model_settings = input_data.standard_microspeech_model_settings(3) 
    name, model, details = transfer_learning.transfer_learn(
        target=keyword,
        train_files=five_samples,
        val_files=dev_samples,
        unknown_files=unknown_files,
        num_epochs=num_epochs,
        num_batches=num_batches,
        batch_size=batch_size,
        primary_lr=primary_learning_rate,
        backprop_into_embedding=False,
        embedding_lr=0,
        model_settings=model_settings,
        base_model_path=embedding,
        base_model_output=base_model_output,
        UNKNOWN_PERCENTAGE=unknown_percentage,
        bg_datadir=background_noise,
        csvlog_dest=None,
    )

    #print(f"saving model to {output}")
    #model.save(output)
    return model, five_samples, dev_samples, test_samples



def predict(
    keyword: str,
    kws_model, 
    test_samples: list,
    nontarget_dir: os.PathLike,
    background_noise: os.PathLike):
    
    #===========================================
    # to gather non-target file names for testing
    non_target_examples = []
    for word in os.listdir(nontarget_dir):
        if word == keyword:
            continue
        non_target_examples.extend(Path(nontarget_dir/word).glob("*.wav"))

    #print("n non_target_examples:", len(non_target_examples))
    
    # using using TensorFlow Lite Micro's speech preprocessing frontend for spectrograms
    settings = input_data.standard_microspeech_model_settings(label_count=1)
    
    #===========================================
    # Test the trained FS-KWS model on test sets
    test_spectrograms = np.array([input_data.file2spec(settings, f) for f in test_samples])

    # fetch softmax predictions from the finetuned model:
    predictions = kws_model.predict(test_spectrograms)
    categorical_predictions_target = np.argmax(predictions, axis=1)

    #===========================================
    # verifying the keyword spotting model correctly categorizes non-target words as "unknown" by
    # running predictions on their spectrograms.
    rng = np.random.RandomState(42)
    n_file  =  len(test_samples) # equal to number of target-test samples
    non_target_examples = rng.choice(non_target_examples, n_file, replace=False).tolist()    
    print("Number of non-target examples", len(non_target_examples))
    non_target_spectrograms = np.array([input_data.file2spec(settings, str(f)) for f in non_target_examples])

    # fetch softmax predictions from the finetuned model:
    predictions = kws_model.predict(non_target_spectrograms)
    categorical_predictions_nontarget = np.argmax(predictions, axis=1)

                        
    return  categorical_predictions_target,   categorical_predictions_nontarget                                 
    
                                        
                                        
def report_results (target_pred, nontarget_pred):

    #===========================================

    # which predictions match the target class? 
    accuracy_target = target_pred[target_pred == 2].shape[0] / target_pred.shape[0]
    print(f"Test accuracy on testset: {accuracy_target:0.2f}")
          
    # which predictions match the non-target class? 
    accuracy_nontarget = nontarget_pred[nontarget_pred == 1].shape[0] / nontarget_pred.shape[0]
    print(f"Estimated accuracy on non-target samples: {accuracy_nontarget:0.2f}") 
                                        
    frr_val = target_pred[target_pred != 2].shape[0] / target_pred.shape[0]
    far_val = nontarget_pred[nontarget_pred == 2].shape[0] / nontarget_pred.shape[0]


    print("FRR: {} %".format( np.floor(100000*frr_val)/1000))
    print("FAR: {} %".format( np.floor(10000*far_val)/100))
                                        
    return frr_val,far_val,  accuracy_target, accuracy_nontarget
    
    
    

#if __name__ == "__main__":
#    fire.Fire(dict(inference=inference, train=train))
#    train(keyword: str,samples_dir: os.PathLike,embedding: os.PathLike,
#          unknown_words: os.PathLike,
#          background_noise: os.PathLike,
#          output: os.PathLike)


