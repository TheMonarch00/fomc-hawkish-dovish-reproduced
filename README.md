# Reproduced work of Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis

We obtained code from the authors GitHub repo and closely followed instructions to rerun experiments. No modifications were made. Except that we couldn’t generate the Dataset and were forced to use the already existing one.

The paper is available at [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4447632) 


### How to use
```
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa", do_lower_case=True, do_basic_tokenize=True)

model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/FOMC-RoBERTa", num_labels=3)

config = AutoConfig.from_pretrained("gtfintechlab/FOMC-RoBERTa")

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, config=config, device=0, framework="pt")
results = classifier(["Such a directive would imply that any tightening should be implemented promptly if developments were perceived as pointing to rising inflation.", 
                      "The International Monetary Fund projects that global economic growth in 2019 will be the slowest since the financial crisis."], 
                      batch_size=128, truncation="only_first")

print(results)
```

### Label Interpretation
LABEL_2: Neutral  
LABEL_1: Hawkish  
LABEL_0: Dovish 

##Datasets
1. Statistics:
• 1,132 labeled meeting minutes sentences
• 322 labeled press conference sentences
• 1,026 labeled speech sentences
• Class distributions:
• Meeting Minutes:
o Hawkish:38% o Dovish:29% o Neutral:33%
• Press Conferences & Speeches have similar 3-way class balance.
2. Train/Val/Test Splits:
• 80/20 train/test split stratified by labels
• Further 80/20 train/val split
3. Preprocessing:
• Sentence tokenization
• Custom filtrations on raw texts and titles
• Sentence splitting using keywords (co-ordinating conjunctions and semicolons)
Link to the dataset: https://github.com/gtfintechlab/fomc-hawkish- dovish/tree/main/training_data/test-and-training

##Hyperparameters
Hyperparameter Search Method:
• Grid search over learning rate and batch size
• Learning rates: {1e-4, 1e-5, 1e-6, 1e-7}
• Batch sizes: {32, 16, 8, 4}
• Total hyperparameter combinations: 16
The authors performed a grid search over the learning rate and batch size for all models. For each model, every combination was evaluated based on the validation F1 score after the training finished.
Best Hyperparameters:
For the RoBERTa-large model:
  
• Learning rate: 1e-5
• Batch size: 16
This configuration achieved the highest weighted F1 score of 0.71 on the validation set. So the authors used these optimal hyperparameters for the final RoBERTa-large model training.
While the full results table for the grid search is not provided, we can infer that 1e-5 with batch 16 works best for RoBERTa-large on this dataset among the searched options. Running all 16 hyperparameter trials enabled efficient tuning without needing to try many more combinations.


##Experimental setup and code
• Environment: Python 3.7, PyTorch 1.8
• Framework: HuggingFace Transformers
• Evaluation Metric: Weighted F1 Score
The experiments were run using Python 3.7 with PyTorch 1.8 as the deep learning framework.. The HuggingFace Transformers library provided model implementation and helpers.
The authors evaluated model performance using Weighted F1 since the classes are imbalanced. Weighted F1 calculates F1 for each class and computes their average weighted by class support. This was computed on the test set after training finished.
The complete experimental code, including data loading, model training, evaluation, and results analysis is available in the linked GitHub repo. It contains Python scripts and Markdown documentation to replicate the full experimental framework end-to-end. The readme provides precise instructions to set up the environment, dependencies, and execution using the released codebase.
By outlining these specifics and supplying full documented code, the authors enabled others to accurately reconstruct experiments for reproducible results and analysis. Python version, Transformers calls, and evaluation metric details allow precise reconstruction.

##Computational requirements
We think that the authors leveraged a high-memory GPU server with a 64-core CPU and 256GB RAM to run experiments. This allowed efficient parallelized computing. Specifically, fine-tuning the RoBERTa- large model required extensive computing. With a batch size of 16, each validation batch took 0.42s on average. The total fine-tuning time was reported as 4.2 hours.
These runtime measurements provide context on the computational intensity for readers interested in reproducing the approach. Given the scale of models like RoBERTa-large, powerful hardware is essential for feasible experimentation. The specifics around the batch throughput, and total training hours offer helpful guidelines.
By outlining the hardware capabilities and model runtimes, readers can determine if they have suitable resources to run these monetary policy text analysis experiments and models from the paper. The compute requirements provide practical insight into reproducibility.

##Performance Results
• RoBERTa-large achieved the best performance across all datasets compared to other models, with F1 scores ranging from 0.5546 to 0.7345. This supports the paper's claim that fine-tuned PLMs perform significantly better than rule-based and RNN-based models on the hawkish vs dovish classification task.
• RoBERTa-large benefited from sentence splitting, showing improved performance on the split datasets for meeting minutes and press conferences but not speeches. This aligns with the paper's observation that sentence splitting helps improve performance overall but not uniformly across datasets.

