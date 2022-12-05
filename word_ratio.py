import pandas as pd
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


df = df.groupby(['tweets']).size().reset_index(name="size")
df = df.sort_values('size', ascending=False)

df = df[df['tweets'] != "to"]
df = df[df['tweets'] != "the"]
df = df[df['tweets'] != "of"]
df = df[df['tweets'] != "is"]
df = df[df['tweets'] != "in"]
df = df[df['tweets'] != "that"]
df = df[df['tweets'] != "I"]
df = df[df['tweets'] != "a"]
df = df[df['tweets'] != "."]
df = df[df['tweets'] != "The"]
df = df[df['tweets'] != "and"]
df = df[df['tweets'] != "for"]
df = df[df['tweets'] != "be"]
df = df[df['tweets'] != "you"]
df = df[df['tweets'] != "are"]
df = df[df['tweets'] != "it"]
df = df[df['tweets'] != "with"]
df = df[df['tweets'] != ""]
df = df[df['tweets'] != "ownership."]
df = df[df['tweets'] != "ownership,"]
df = df[df['tweets'] != "on"]

plt.bar(df['tweets'][1:16], df['size'][1:16])
plt.title("Word count: Neutral tweets")
plt.ylabel("frequency")
plt.show()

repub = repub['tweets'].str.split(' ')
repub = repub.explode('tweets')
df = pd.DataFrame()
df['tweets'] = repub
df['tweets'] = df['tweets'].str.replace("\n", "")
df['tweets'] = df['tweets'].str.replace("\"", "")

df = df.groupby(['tweets']).size().reset_index(name="size")
df = df.sort_values('size', ascending=False)

df = df[df['tweets'] != "to"]
df = df[df['tweets'] != "the"]
df = df[df['tweets'] != "of"]
df = df[df['tweets'] != "is"]
df = df[df['tweets'] != "in"]
df = df[df['tweets'] != "that"]
df = df[df['tweets'] != "I"]
df = df[df['tweets'] != "a"]
df = df[df['tweets'] != "."]
df = df[df['tweets'] != "The"]
df = df[df['tweets'] != "and"]
df = df[df['tweets'] != "for"]
df = df[df['tweets'] != "be"]
df = df[df['tweets'] != "you"]
df = df[df['tweets'] != "are"]
df = df[df['tweets'] != "it"]
df = df[df['tweets'] != "with"]
df = df[df['tweets'] != ""]
df = df[df['tweets'] != "ownership."]
df = df[df['tweets'] != "ownership,"]
df = df[df['tweets'] != "on"]

plt.bar(df['tweets'][1:16], df['size'][1:16])
plt.title("Word count: Republican tweets")
plt.ylabel("frequency")
plt.show()

democ = democ['tweets'].str.split(' ')
democ = democ.explode('tweets')
df = pd.DataFrame()
df['tweets'] = democ
df['tweets'] = df['tweets'].str.replace("\n", "")
df['tweets'] = df['tweets'].str.replace("\"", "")

df = df.groupby(['tweets']).size().reset_index(name="size")
df = df.sort_values('size', ascending=False)

df = df[df['tweets'] != "to"]
df = df[df['tweets'] != "the"]
df = df[df['tweets'] != "of"]
df = df[df['tweets'] != "is"]
df = df[df['tweets'] != "in"]
df = df[df['tweets'] != "that"]
df = df[df['tweets'] != "I"]
df = df[df['tweets'] != "a"]
df = df[df['tweets'] != "."]
df = df[df['tweets'] != "The"]
df = df[df['tweets'] != "and"]
df = df[df['tweets'] != "for"]
df = df[df['tweets'] != "be"]
df = df[df['tweets'] != "you"]
df = df[df['tweets'] != "are"]
df = df[df['tweets'] != "it"]
df = df[df['tweets'] != "with"]
df = df[df['tweets'] != ""]
df = df[df['tweets'] != "ownership."]
df = df[df['tweets'] != "ownership,"]
df = df[df['tweets'] != "on"]

plt.bar(df['tweets'][1:16], df['size'][1:16])
plt.title("Word count: Democratic tweets")
plt.ylabel("frequency")
plt.show()
