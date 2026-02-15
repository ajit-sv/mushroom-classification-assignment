# Mushroom classification assignment

## a. Problem Statement
Mushrooms are good a source of nutrition. However, many mushrooms are inedible and poisonous; even deadly. Casual observers can easily mistake inedible mushrooms with edible ones. This mushroom classification application uses Kaggle's mushroom classification dataset to train a set of machine learning models which are then used to classify mushrooms as edible or poisonous by making predictions on user fed test data.

## b. Dataset description
This application was trained on Kaggle's mushroom classification dataset (originally from UCI). The data set has 23 columns in total. Of these 22 are features and one colums has classification information. It has a total of 8124 data points. Of this, 126 data points were removed from original dataset and kept aside as test data. The rest 7998 data points were used for training the models. Some of the features are 
- cap-shape having values bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
- cap-surface having values fibrous=f, grooves=g, scaly=y, smooth=s
- cap-color having values brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
bruises having values bruises=t, no=f
- odor having values almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
- gill-attachment having values attached=a, descending=d, free=f, notched=n
- gill-spacing having values close=c, crowded=w, distant=d
- gill-size having values broad=b, narrow=n
- gill-color having values black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
- stalk-shape having values enlarging=e, tapering=t
- stalk-root having values bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
- stalk-surface-above-ring having values fibrous=f, scaly=y, silky=k, smooth=s
- stalk-surface-below-ring having values fibrous=f, scaly=y, silky=k, smooth=s
- stalk-color-above-ring having values brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
- stalk-color-below-ring having values brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
- veil-type having values partial=p, universal=u
- veil-color having values brown=n, orange=o, white=w, yellow=y

## c. Models used and comparision of their evaluation metrics

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----------|-----------|--------|----------|-----|
| Logistic Regression | 0.9841 | 0.9670 | 1.0000 | 0.9655 | 0.9825 | 0.9685 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9524 | 0.9678 | 0.9333 | 0.9655 | 0.9492 | 0.9049 |
| Random Forest | 1.0000 | N/A | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Observations Table
| ML Model Name | Observation |
|---------------|-------------|
| Logistic Regression | This model has a linear boundary and hence moderate performance. With the test data I used, two poisonous mushroom varieties were classified as edible. This can be problematic and can't be used in real life situaations |
| Decision Tree | This model provides high accuracy but risk of overfitting is present. With the test data I used, it classified all mushrooms correctly. | 
| KNN | This model is sensitive to the value of k. I used a k value of 5. With this k value, this model managed to correctly classify all mushrooms in my test data. |
| Naive Bayes | This model is very fast and needs very little computation needs. However, with the test data I used, two poisonous mushroom varieties were classified as edible. This can be problematic and can't be used in real life situaations |
| Random Forest | This model is highly accurate and stable. It managed to correctly classify all mushrooms in my test data. |
| XGBoost | This model is also highly accurate. It managed to correctly classify all mushrooms in my test data. |


