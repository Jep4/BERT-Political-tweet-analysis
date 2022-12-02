import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df1 = pd.read_csv("data.csv")
df2 = pd.read_csv("data2.csv")
df3 = pd.read_csv("data3.csv")

combined = pd.concat([df1, df2, df3])
neut = combined[combined['class'] == 'N']
repub = combined[combined['class'] == 'R']
democ = combined[combined['class'] == 'D']

neut = neut['tweets'].str.split(' ')
neut = neut.explode('tweets')
df = pd.DataFrame()
df['tweets'] = neut
df['tweets'] = df['tweets'].str.replace("\n", "")
df['tweets'] = df['tweets'].str.replace("\"", "")
df['tweets'] = df['tweets'].str.replace("to", "")
df['tweets'] = df['tweets'].str.replace("the", "")
df['tweets'] = df['tweets'].str.replace("a", "")
df['tweets'] = df['tweets'].str.replace("of", "")
df['tweets'] = df['tweets'].str.replace("is", "")
df = df.groupby(['tweets']).size().reset_index(name="size")
df = df.sort_values('size', ascending=False)

lists = df.set_index("tweets").to_dict()['size']

wordcloud = WordCloud().generate_from_frequencies(lists)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

repub = repub['tweets'].str.split(' ')
repub = repub.explode('tweets')
df = pd.DataFrame()
df['tweets'] = repub
df['tweets'] = df['tweets'].str.replace("\n", "")
df['tweets'] = df['tweets'].str.replace("\"", "")
df['tweets'] = df['tweets'].str.replace("to", "")
df['tweets'] = df['tweets'].str.replace("the", "")
df['tweets'] = df['tweets'].str.replace("a", "")
df['tweets'] = df['tweets'].str.replace("of", "")
df['tweets'] = df['tweets'].str.replace("is", "")
df = df.groupby(['tweets']).size().reset_index(name="size")
df = df.sort_values('size', ascending=False)

lists = df.set_index("tweets").to_dict()['size']

wordcloud = WordCloud().generate_from_frequencies(lists)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



democ = democ['tweets'].str.split(' ')
democ = democ.explode('tweets')
df = pd.DataFrame()
df['tweets'] = democ
df['tweets'] = df['tweets'].str.replace("\n", "")
df['tweets'] = df['tweets'].str.replace("\"", "")
df['tweets'] = df['tweets'].str.replace("to", "")
df['tweets'] = df['tweets'].str.replace("the", "")
df['tweets'] = df['tweets'].str.replace("a", "")
df['tweets'] = df['tweets'].str.replace("of", "")
df['tweets'] = df['tweets'].str.replace("is", "")
df = df.groupby(['tweets']).size().reset_index(name="size")
df = df.sort_values('size', ascending=False)

lists = df.set_index("tweets").to_dict()['size']

wordcloud = WordCloud().generate_from_frequencies(lists)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
