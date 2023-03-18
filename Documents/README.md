![Solution Architecture](MLE11 KWS Solution Architecture.jpg)

# Performance, interpretation, and learnings

From the [Few-Shot Keyword Spotting in Any Language, Interspeech 2021](https://www.isca-speech.org/archive/pdfs/interspeech_2021/mazumder21_interspeech.pdf): Copyright © 2021 ISCA

"5-Shot KWS Classification Accuracy. ROC curves for 5-shot KWS models with 20 randomly selected keywords per language. For each language, the mean is drawn as ab olded curve over the shaded standard deviation(all keywords are shown as a hairline trace).

(a) 5-shot KWS models using an embedding representation trained per language for six languages.

[Average F1@0.8 = 0.58]

(b) 5shot models using a multilingual embedding trained on nine languages — accuracy improves relative to (a).

[Avg. F1@0.8 = 0.75]

(c) 5-shot models using the same multilingual embedding from (b) for random keywords in 13 languages which are out-of-embedding (i.e., which the feature extractor has never encountered), showing that our embedding generalizes to new languages.

[Avg. F1@0.8 = 0.65]

    We have yet to iteratively improve our model performance, as we have only been able to perform model comparisons between existing works. Previously we were using (1) zero-shot learning with wav2vec2-asr model and later (2) fine-tuned model of wav2vec2 for keyword spotting, but we found them having very poor performance for unseen keywords.  
    We will be working on model improvement on the currently selected model , Harvard multilingual few shot keyword spotting , in the following weeks.
    We also investigated possible ways for real-time keyword spotting app development, and identified possible challenges. (e.g. model deployment on iOS needs Swift programming). We may first deploy the model in a wab-base fashion; and if time allows on an app.

    Since our model uses a large volume of data, we spend a lot of time learning how to use S3 and access data from SageMaker via S3.

Report and Observation:
the accuracy of a few-shot model can be quite sensitive to the particular samples chosen. A different set of five training samples will likely vary by a few percentage points from this model. The subset of unknown words and background noise also impact accuracy. Furthermore, some of the extracted test samples may be truncated (due to incorrect word boundary estimates) or occasionally anomalous due to issues with the originating crowdsourced data so a more accurate estimate of the test performance can only be made after manually listening to all test samples and discarding any malformed samples.
