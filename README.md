# malicious-classification

Here I try to use data about responses to websites and classify them as 'malicious' or not.
Data is available at: https://www.kaggle.com/xwolf12/malicious-and-benign-websites 

I was able to achieve a CV score (10 fold) of the mid 90s with a random-forest classifier and no hyperparameter tuning, however this isn't always consistent and certain folds sometimes have really low scores which leads me to believe that this dataset should get more data!!

I'm also not really happy with the recall on the 'malicious' websites as, to me at least, correctly classifying all of the malicious websites is more important in reality and I'd rather be safe than sorry if this were for some sort of application.

I did however remove any features that have 0 feature importance.

```
FOREST LIMITED FEATURES:
             precision    recall  f1-score   support

          0       0.95      0.99      0.97       283
          1       0.84      0.57      0.68        37

avg / total       0.93      0.94      0.93       320

('CV: ', 0.9494380741987516)
```