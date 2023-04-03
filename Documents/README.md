![Solution Architecture](Solution_Architecture_updated.png)

# Performance, interpretation, and learnings



Report and Observation:
the performance of a custom few-shot keyword spotting model can be sensitive to the particular samples chosen for fine-tuning the model. A different set of k=five training samples will likely vary by a few percentage points. Another performance factor is subset of unknown words and background noise we use for training and fine-tuning. Furthermore, some of the extracted test samples may be truncated (due to imperfect speaker segmentation).


Model to Fine-tune Keyword spotting Model:
![model](custom_FS_kws_model.png)

Performance Report:
![performance1](OfflineEvaluation_CustomKWS_1.png)
![performance2](OfflineEvaluation_CustomKWS_2.png)

