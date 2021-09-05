from transformers import AutoModelForSequenceClassification,TrainingArguments,Trainer
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import numpy as np

def run_model(dataset):
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',num_labels=1)
    modules = [model.roberta.embeddings, *model.roberta.encoder.layer[:6]] #freeze all layers
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    def compute_metrics(p):
        logits, labels = p.predictions,p.label_ids
        logits = np.rint(((logits*40)).flatten())
        labels = np.rint(((labels*40)).flatten())
        rmse = np.sqrt(np.mean((logits-labels)**2))
        pearson = np.corrcoef(logits,labels)[0,1]
        coef, p = spearmanr(logits, labels)
        kappa = cohen_kappa_score(logits, labels)
        return {
                "rmse": rmse,
                "pearson": pearson,
                "spearman" : coef,
                "kappa":kappa
              }

    batch_size = 8
    label_names = ['labels']
    args = TrainingArguments(
            output_dir = '/content',
            save_total_limit = 1,
            evaluation_strategy = "epoch",
            learning_rate = 5e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            num_train_epochs = 30,
            weight_decay = 0.01,
            save_strategy = 'epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            label_names=label_names
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        compute_metrics=compute_metrics
        )
    trainer.train()

    return model