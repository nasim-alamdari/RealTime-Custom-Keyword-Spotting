import datetime
from pathlib import Path
import os
import joblib
import pandas as pd
from func import audio_record, kws_train, input_data, spk_segment, inference

import argparse

BASE_DIR = Path(__file__).resolve(strict=True).parent
TODAY = datetime.date.today()

def record(KEYWORD:str):
    keyword_dir = os.path.join(BASE_DIR , './content/target_kw/recording/',KEYWORD )
    if not os.path.exists(keyword_dir):
        os.mkdir(keyword_dir)
    record_name = KEYWORD+'.wav'
    
    duration = 10 # 20 sec
    audio_record.record (duration, record_name, keyword_dir )
    return keyword_dir


def preproc(KEYWORD,keyword_dir):
    spk_segments_path = spk_segment.segment (KEYWORD,keyword_dir)
    return spk_segments_path

def train(keyword="amelia", keyword_dir = './content/target_kw/amelia/'):
    keyword_dir   = Path(BASE_DIR).joinpath(keyword_dir)
    base_model_dir = Path(BASE_DIR).joinpath('./content/multilingual_context_73_0.8011')
    unknown_words_dir = Path(BASE_DIR).joinpath('./content/unknown_files/')
    background_noise_dir =  Path(BASE_DIR).joinpath('./content/speech_commands_v0.02/_background_noise_/')
    

    model, five_samples, dev_samples, test_samples = kws_train.train(keyword= keyword,
                     samples_dir= keyword_dir,
                     embedding= base_model_dir,
                     unknown_words= unknown_words_dir,
                     background_noise= background_noise_dir)


    joblib.dump(model, Path(BASE_DIR).joinpath(f"{keyword}.joblib"))
    # joblib.dump(model, Path(BASE_DIR).joinpath(f"{ticker}_{TODAY.strftime("%Y-%m-%d")}.joblib"))
    return test_samples


def predict(keyword, test_samples):
    
    model_file = Path(BASE_DIR).joinpath(f"{keyword}.joblib")
    nontarget_dir = Path(BASE_DIR).joinpath('./content/nontarget_mswc_microset_wav/en/clips/')
    background_noise_dir =  Path(BASE_DIR).joinpath('./content/speech_commands_v0.02/_background_noise_/')
    
    
    if not model_file.exists():
        return False

    # load model
    model = joblib.load(model_file)
    
    # predictions
    target_pred, nontarget_pred = kws_train.predict(keyword, model, test_samples,
                                   nontarget_dir, background_noise_dir )
    
    # comment out for FastAPI app
    #model.plot(forecast).savefig(f"{ticker}_plot.png")
    #model.plot_components(forecast).savefig(f"{ticker}_plot_components.png")

    return target_pred, nontarget_pred

def report_result (target_pred, nontarget_pred):
    # here compute FRR and FAR:
    frr_val,far_val,  accuracy_target, accuracy_nontarget = kws_train.report_results(target_pred, nontarget_pred)
    
    return frr_val,far_val


def infer(keyword ,duration):
    
    model_file = Path(BASE_DIR).joinpath(f"{keyword}.joblib")
    # load model
    print("Loading the fine-tuned model...")
    model = joblib.load(model_file)
    
    # define threshold for detection of keyword, other words, and background noise
    ths = [0.9,0.9, 0.55]
    
    # processing time for 1-sec audio is on average 60 ms
    print("Inference will start shortly...")
    result = inference.stream_proc_audio(duration,keyword, model, ths)
    return result


# comment out for FastAPI app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--keyword', type=str, default='amelia', help='Keyword Name')
    parser.add_argument('--train_flag', type=int, default=1, help='Flag for record and train or just load pre-trained model for the keyword')

    args = parser.parse_args()
    print("Custom Keyword is: ", args.keyword)
    print("Train flag is set to: ", args.train_flag)
    
    
    if args.train_flag == 1:
        keyword_dir = record(args.keyword)
        spk_segments_path = preproc(args.keyword,keyword_dir)
        test_samples= train(args.keyword, spk_segments_path)
        target_pred, nontarget_pred = predict(args.keyword, test_samples)
        frr_val,far_val = report_result (target_pred, nontarget_pred)
        output = [frr_val,far_val]
        print("Training FRR and FAR result:", output)
    else:
        result = infer(args.keyword,duration=30)
        print("inference result:",result)
