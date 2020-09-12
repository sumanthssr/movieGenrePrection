import pandas as pd
import json
import csv
from tqdm import tqdm

# Load movie meta tsv data
meta_data = pd.read_csv("D:\\myApp\\MachineLearningCaseStudies\\MovieGenrePrediction\\data_load\\MovieSummaries\\movie.metadata.tsv", sep='\t', header=None)

# Load character tsv data
character_data = pd.read_csv("D:\\myApp\\MachineLearningCaseStudies\\MovieGenrePrediction\\data_load\\MovieSummaries\\character.metadata.tsv", sep='\t', header=None)

# renaming column names
meta_data.columns = ['movie_id', 1, 'movie_name', 3, 4, 5, 6, 7, 'genre']

# load plot summaries to list
plots = []
with open('D:\\myApp\\MachineLearningCaseStudies\\MovieGenrePrediction\\data_load\\MovieSummaries\\plot_summaries.txt', 'r', encoding="utf-8") as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)

# create plots data frame
plots_df = pd.DataFrame(plots, columns=['movie_id', 'plot_summary'])


# changing column data types
print('columns in plot table are :{}'.format(plots_df.columns))
print('columns in meta data table are :{}'.format(meta_data.columns))
meta_data['movie_id'] = pd.to_numeric(meta_data['movie_id'])
plots_df['movie_id'] = pd.to_numeric(plots_df['movie_id'])

# now merge both data frames to single data frame
data = pd.merge(meta_data[['movie_id','movie_name','genre']], plots_df, on='movie_id')

merged_data_df = data[['movie_id','movie_name','plot_summary','genre']]

merged_data_df['genre_new'] = merged_data_df['genre'].apply(lambda x: list(json.loads(x).values()))

merged_data_df = merged_data_df[['movie_id','movie_name','plot_summary','genre_new']]
print('data load finished !')

