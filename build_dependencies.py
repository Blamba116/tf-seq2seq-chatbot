import urllib.request

# cornell data
url = ['https://github.com/Conchylicultor/DeepQA/raw/master/data/cornell/movie_conversations.txt'
       ,'https://github.com/Conchylicultor/DeepQA/raw/master/data/cornell/movie_lines.txt'
      ]

filename = ['cornell/movie_conversations.txt', 'cornell/movie_lines.txt']
for i, j in zip(url, filename):    
    urllib.request.urlretrieve(i, j)