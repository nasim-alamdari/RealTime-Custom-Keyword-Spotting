# Nasim Alamdari
# last Update March 2023
import datetime
from pathlib import Path

import joblib
import pandas as pd
from func import kws_train

import argparse

BASE_DIR = Path(__file__).resolve(strict=True).parent
TODAY = datetime.date.today()


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
    return keyword, test_samples


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


# comment out for FastAPI app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--keyword', type=str, default='amelia', help='Keyword Name')
    parser.add_argument('--keyword_dir', type=str, default='./content/target_kw/amelia/', help='Keyword File Path')
    #parser.add_argument('--days', type=int, default=7, help='Number of days to predict')
    args = parser.parse_args()
    
    keyword, test_samples= train(args.keyword, args.keyword_dir)
    target_pred, nontarget_pred = predict(keyword, test_samples)
    frr_val,far_val = report_result (target_pred, nontarget_pred)
    output = [frr_val,far_val]
    print(output)
