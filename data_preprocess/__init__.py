import warnings
warnings.filterwarnings("ignore")
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from data_load import merged_data_df


def stemmer_stopwords_cleaner(plot_summary):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    plot_summary = re.sub(r'[^A-Za-z]+', ' ', plot_summary)
    words = word_tokenize(str(plot_summary.lower()))
    # Removing all single letter and and stopwords
    plot_summary = ' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j) != 1))
    return plot_summary


print('Starting preprocessing text content !!')
merged_data_df['plot_summary'] = merged_data_df['plot_summary'].apply(lambda x : stemmer_stopwords_cleaner(x))
preprocessed_df = merged_data_df
print('Finished processing text content !!')

