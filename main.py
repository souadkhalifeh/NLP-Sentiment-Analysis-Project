import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('read.txt', encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
tokenized_words = word_tokenize(cleaned_text, "english")

final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)


emotion_counts = Counter(emotion_list)
print(emotion_counts)


def sentiment_analyze(sentiment_text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    neu = score['neu']
    compound = score['compound']

    if neg > pos and neg>neu:
        print("Negative Sentiment")
    elif pos > neg and pos>neu:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")

    # Print detailed scores
    print(f"Negative: {neg}, Neutral: {neu}, Positive: {pos}, Compound: {compound}")

    return score

sentiment_scores = sentiment_analyze(cleaned_text)
print(f"Sentiment Scores: {sentiment_scores}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))


ax1.bar(emotion_counts.keys(), emotion_counts.values(),
        color=['blue', 'pink', 'purple', 'green', 'red', 'magenta', 'yellow', 'navy', 'cyan', 'orange'])
ax1.set_title('Emotion Analysis')
ax1.set_xlabel('Emotion')
ax1.set_ylabel('Count')


ax2.bar(['Negative', 'Neutral', 'Positive'], [sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos']], color=['red', 'gray', 'green'])
ax2.set_title('Sentiment Analysis')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Score')

fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
