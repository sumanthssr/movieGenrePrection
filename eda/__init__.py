import warnings
warnings.filterwarnings("ignore")
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_load import merged_data_df


all_genres = sum(merged_data_df['genre_new'], [])


all_genres = nltk.FreqDist(all_genres)

tag_freq = []
for key, value in all_genres.items():
    tag_freq.append([key, value])
tag_freq_df = pd.DataFrame(tag_freq, columns=['Genre', 'Count'])
tag_freq_df = tag_freq_df.sort_values(by=['Count'], ascending=False).reset_index(drop=True)


# plotting
values = tag_freq_df['Count'].values

plt.plot(values)
plt.grid()
plt.title('Distribution of Genres')
plt.xlabel('Genre number')
plt.ylabel('Number of times genre appeared')
plt.show()

i = np.arange(30)
tag_freq_df.head(30).plot(kind='bar')
plt.title('top 30 tags')
plt.xticks(i,tag_freq_df['Genre'])
plt.xlabel('Tags')
plt.ylabel('Count')
plt.show()




