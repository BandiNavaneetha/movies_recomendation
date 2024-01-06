```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline 
import warnings
warnings.filterwarnings('ignore')

```


```python
af = pd.read_csv("movies.csv") 
bf = pd.read_csv("ratings.csv") 
cf = pd.read_csv("links.csv") 
df = pd.read_csv("tags.csv") 
```


```python
print(af.shape)
print(bf.shape)
```

    (9742, 3)
    (100836, 4)
    


```python
import csv
csv_reader = csv.reader(bf, delimiter=',')
uniqueIds = set()

for row in csv_reader:
    uniqueIds.add(row[0])

print(len(uniqueIds))
```

    4
    


```python
bf['userId'].unique()
```




    array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
            14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
            27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
            40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
            53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
            66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
            79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
            92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
           118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
           131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
           144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
           157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
           170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
           183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
           196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
           209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
           222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
           235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
           248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
           261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
           274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
           287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
           300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
           313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
           326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
           339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
           352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364,
           365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
           378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
           391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
           404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
           417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429,
           430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442,
           443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
           456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468,
           469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481,
           482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
           495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
           508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
           521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
           534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
           547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
           560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
           573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585,
           586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
           599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610],
          dtype=int64)




```python
ratings = bf["rating"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
fig = px.pie(bf, values=quantity, names=numbers)
fig.show()
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.20.0
* Copyright 2012-2023, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>




<div>                            <div id="081d73b2-88dc-4063-9734-df6020bbeaa7" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("081d73b2-88dc-4063-9734-df6020bbeaa7")) {                    Plotly.newPlot(                        "081d73b2-88dc-4063-9734-df6020bbeaa7",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"label=%{label}<br>value=%{value}<extra></extra>","labels":[4.0,3.0,5.0,3.5,4.5,2.0,2.5,1.0,1.5,0.5],"legendgroup":"","name":"","showlegend":true,"values":[26818,20047,13211,13136,8551,7551,5550,2811,1791,1370],"type":"pie"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('081d73b2-88dc-4063-9734-df6020bbeaa7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
max_ratings_movie = bf.loc[bf['rating'].idxmax()]['movieId']

print(f"The movie with the maximum number of user ratings is: {max_ratings_movie}")
```

    The movie with the maximum number of user ratings is: 47.0
    


```python
bf.groupby('movieId')['rating'].mean()
```




    movieId
    1         3.920930
    2         3.431818
    3         3.259615
    4         2.357143
    5         3.071429
                ...   
    193581    4.000000
    193583    3.500000
    193585    3.500000
    193587    3.500000
    193609    4.000000
    Name: rating, Length: 9724, dtype: float64




```python
rating_columns = ['user_id', 'movie_id', 'rating','timestamp']
ratings = pd.read_csv("ratings.csv")
links_columns=['movie_id','imdbld','tmdbld']
links = pd.read_csv("links.csv")
movie_columns = ['movie_id', 'title','genres']
movies = pd.read_csv("movies.csv")
tag_columns = ['user_id', 'movie_id', 'tag','timestamp']
tags = pd.read_csv("tags.csv")
```


```python

movie_ratings = pd.merge(movies, ratings)

movie_data = pd.merge(movies,tags)
links_data=pd.merge(movies,links)

```


```python
movie_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>5</td>
      <td>4.0</td>
      <td>847434962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>7</td>
      <td>4.5</td>
      <td>1106635946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>15</td>
      <td>2.5</td>
      <td>1510577970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>17</td>
      <td>4.5</td>
      <td>1305696483</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>1139045764</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>474</td>
      <td>pixar</td>
      <td>1137206825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>567</td>
      <td>fun</td>
      <td>1525286013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>1528843929</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>magic board game</td>
      <td>1528843932</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings = movie_ratings['rating'].value_counts()
ratings
```




    4.0    26818
    3.0    20047
    5.0    13211
    3.5    13136
    4.5     8551
    2.0     7551
    2.5     5550
    1.0     2811
    1.5     1791
    0.5     1370
    Name: rating, dtype: int64




```python

# Specify the film for which you want the average rating
specific_film = 'Terminator 2: Judgment Day (1991)'

# Filter the DataFrame for the specific film and calculate its average rating
avg_rating = movie_ratings[movie_ratings['title'] == specific_film]['rating'].mean()
print(f"The average rating for {specific_film} is: {avg_rating}")
```

    The average rating for Terminator 2: Judgment Day (1991) is: 3.970982142857143
    


```python



# Use value_counts to count the occurrences of each movie and find the one with the maximum count
max_rated_movie = movie_ratings['title'].value_counts().idxmax()
max_ratings_count = movie_ratings['title'].value_counts().max()

print(f"The movie with the maximum number of user ratings is '{max_rated_movie}' with {max_ratings_count} ratings.")
```

    The movie with the maximum number of user ratings is 'Forrest Gump (1994)' with 329 ratings.
    


```python


# Specify the movie you're interested in
movie_of_interest = 'Matrix, The (1999)'

# Filter the DataFrame for the specified movie's tags
tags_for_movie = movie_data[movie_data['title'] == movie_of_interest]['tag'].unique()
print(f"The correct tags for '{movie_of_interest}' submitted by users are: {list(tags_for_movie)}")
```

    The correct tags for 'Matrix, The (1999)' submitted by users are: ['martial arts', 'sci-fi', 'alternate universe', 'philosophy', 'post apocalyptic']
    


```python
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'ratings' with columns 'Movie' and 'Rating'
# Replace this with your actual dataset


# Filter ratings for the "Fight Club (1999)" movie
fight_club_ratings = movie_ratings[movie_ratings['title'] == 'Fight Club (1999)']

# Plotting the data distribution using a histogram
plt.figure(figsize=(8, 6))
plt.hist(fight_club_ratings['rating'], bins=5, edgecolor='black')
plt.xlabel('rating')
plt.ylabel('Frequency')
plt.title('User Ratings Distribution for "Fight Club (1999)"')
plt.grid(True)
plt.show()
```


    
![png](output_16_0.png)
    



```python


# Grouping by 'movie_id' and applying count and mean aggregation functions
#grouped_ratings = movie_ratings.groupby('title').agg({'rating': ['count', 'mean']})
# Grouping by 'title' and applying count and mean aggregation functions
grouped_ratings = movie_ratings.groupby('title').agg({'rating': ['count', 'mean']})

# Assuming you already have the grouped_ratings DataFrame
# (created using the code provided in the previous response)



# Sorting the DataFrame in descending order based on mean rating
sorted_ratings = grouped_ratings.sort_values(by=('rating', 'count'), ascending=False)

# Displaying the sorted DataFrame
print(sorted_ratings)


```

                                              rating          
                                               count      mean
    title                                                     
    Forrest Gump (1994)                          329  4.164134
    Shawshank Redemption, The (1994)             317  4.429022
    Pulp Fiction (1994)                          307  4.197068
    Silence of the Lambs, The (1991)             279  4.161290
    Matrix, The (1999)                           278  4.192446
    ...                                          ...       ...
    King Solomon's Mines (1950)                    1  3.000000
    King Solomon's Mines (1937)                    1  2.500000
    King Ralph (1991)                              1  1.500000
    King Kong Lives (1986)                         1  2.000000
    À nous la liberté (Freedom for Us) (1931)      1  1.000000
    
    [9719 rows x 2 columns]
    


```python
# Assuming you have the original 'movie_ratings' DataFrame
# You can use the following code to find the top 5 popular movies based on the number of user ratings

# Grouping by 'title' and applying count aggregation function
movie_popularity = movie_ratings.groupby('title').agg({'rating': 'count'})

# Sorting the DataFrame in descending order based on the number of user ratings
sorted_popularity = movie_popularity.sort_values(by='rating', ascending=False)

# Selecting the top 5 popular movies
top_5_popular_movies = sorted_popularity.head(5)

# Displaying the top 5 popular movies
print("Top 5 Popular Movies based on Number of User Ratings:")
print(top_5_popular_movies)

```

    Top 5 Popular Movies based on Number of User Ratings:
                                      rating
    title                                   
    Forrest Gump (1994)                  329
    Shawshank Redemption, The (1994)     317
    Pulp Fiction (1994)                  307
    Silence of the Lambs, The (1991)     279
    Matrix, The (1999)                   278
    


```python
# Assuming you have the original 'movie_ratings' DataFrame
# You can use the following code to find the third most popular Sci-Fi movie based on the number of user ratings

# Filter the DataFrame to include only Sci-Fi movies
scifi_movies = movie_ratings[movie_ratings['genres'] == 'Sci-Fi']

# Grouping Sci-Fi movies by 'title' and applying count aggregation function
scifi_popularity = scifi_movies.groupby('title').agg({'rating': 'count'})

# Sorting the DataFrame in descending order based on the number of user ratings
sorted_scifi_popularity = scifi_popularity.sort_values(by='rating')

# Selecting the third most popular Sci-Fi movie
third_most_popular_scifi_movie = sorted_scifi_popularity.index[2]

# Displaying the result
print("Third Most Popular Sci-Fi Movie based on Number of User Ratings:")
print(third_most_popular_scifi_movie)


```

    Third Most Popular Sci-Fi Movie based on Number of User Ratings:
    This Island Earth (1955)
    


```python
# Assuming you have the original 'movie_ratings' DataFrame
# You can use the following code to filter movies with more than 50 user ratings

# Grouping by 'title' and applying count aggregation function
movie_popularity = movie_ratings.groupby('title').agg({'rating': 'count'})

# Filtering movies with more than 50 user ratings
popular_movies = movie_popularity[movie_popularity['rating'] > 50]

# Displaying the result
print("Movies with more than 50 User Ratings:")
print(popular_movies)

```

    Movies with more than 50 User Ratings:
                                       rating
    title                                    
    10 Things I Hate About You (1999)      54
    12 Angry Men (1957)                    57
    2001: A Space Odyssey (1968)          109
    28 Days Later (2002)                   58
    300 (2007)                             80
    ...                                   ...
    X-Men: The Last Stand (2006)           52
    X2: X-Men United (2003)                76
    Young Frankenstein (1974)              69
    Zombieland (2009)                      53
    Zoolander (2001)                       54
    
    [437 rows x 1 columns]
    


```python


# Assuming you have loaded movies data into movies_df and created grouped_df

# Perform inner join on 'movieId' column
merged_df = pd.merge(af,movie_ratings, on='movieId', how='inner')
merged_df.tail(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title_x</th>
      <th>genres_x</th>
      <th>title_y</th>
      <th>genres_y</th>
      <th>userId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100736</th>
      <td>183301</td>
      <td>The Tale of the Bunny Picnic (1986)</td>
      <td>Children</td>
      <td>The Tale of the Bunny Picnic (1986)</td>
      <td>Children</td>
      <td>599</td>
      <td>3.0</td>
      <td>1519148271</td>
    </tr>
    <tr>
      <th>100737</th>
      <td>183317</td>
      <td>Patti Rocks (1988)</td>
      <td>Comedy|Drama</td>
      <td>Patti Rocks (1988)</td>
      <td>Comedy|Drama</td>
      <td>462</td>
      <td>4.5</td>
      <td>1515194979</td>
    </tr>
    <tr>
      <th>100738</th>
      <td>183611</td>
      <td>Game Night (2018)</td>
      <td>Action|Comedy|Crime|Horror</td>
      <td>Game Night (2018)</td>
      <td>Action|Comedy|Crime|Horror</td>
      <td>62</td>
      <td>4.0</td>
      <td>1526244681</td>
    </tr>
    <tr>
      <th>100739</th>
      <td>183635</td>
      <td>Maze Runner: The Death Cure (2018)</td>
      <td>Action|Mystery|Sci-Fi|Thriller</td>
      <td>Maze Runner: The Death Cure (2018)</td>
      <td>Action|Mystery|Sci-Fi|Thriller</td>
      <td>596</td>
      <td>3.5</td>
      <td>1535709593</td>
    </tr>
    <tr>
      <th>100740</th>
      <td>183897</td>
      <td>Isle of Dogs (2018)</td>
      <td>Animation|Comedy</td>
      <td>Isle of Dogs (2018)</td>
      <td>Animation|Comedy</td>
      <td>212</td>
      <td>5.0</td>
      <td>1532361617</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>100831</th>
      <td>193581</td>
      <td>Black Butler: Book of the Atlantic (2017)</td>
      <td>Action|Animation|Comedy|Fantasy</td>
      <td>Black Butler: Book of the Atlantic (2017)</td>
      <td>Action|Animation|Comedy|Fantasy</td>
      <td>184</td>
      <td>4.0</td>
      <td>1537109082</td>
    </tr>
    <tr>
      <th>100832</th>
      <td>193583</td>
      <td>No Game No Life: Zero (2017)</td>
      <td>Animation|Comedy|Fantasy</td>
      <td>No Game No Life: Zero (2017)</td>
      <td>Animation|Comedy|Fantasy</td>
      <td>184</td>
      <td>3.5</td>
      <td>1537109545</td>
    </tr>
    <tr>
      <th>100833</th>
      <td>193585</td>
      <td>Flint (2017)</td>
      <td>Drama</td>
      <td>Flint (2017)</td>
      <td>Drama</td>
      <td>184</td>
      <td>3.5</td>
      <td>1537109805</td>
    </tr>
    <tr>
      <th>100834</th>
      <td>193587</td>
      <td>Bungo Stray Dogs: Dead Apple (2018)</td>
      <td>Action|Animation</td>
      <td>Bungo Stray Dogs: Dead Apple (2018)</td>
      <td>Action|Animation</td>
      <td>184</td>
      <td>3.5</td>
      <td>1537110021</td>
    </tr>
    <tr>
      <th>100835</th>
      <td>193609</td>
      <td>Andrew Dice Clay: Dice Rules (1991)</td>
      <td>Comedy</td>
      <td>Andrew Dice Clay: Dice Rules (1991)</td>
      <td>Comedy</td>
      <td>331</td>
      <td>4.0</td>
      <td>1537157606</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>




```python
# Assuming you have a DataFrame 'movie-ratings' with columns 'title' and 'rating'
# Replace 'movies_data' and column names with your actual data

# Filter movies with more than 50 user ratings
filtered_movies =movie_ratings[movie_ratings['rating']>50]
print(filtered_movies)
```

    Empty DataFrame
    Columns: [movieId, title, genres, userId, rating, timestamp]
    Index: []
    


```python
data = pd.merge(movie_ratings,movie_data)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Movie A</td>
      <td>4</td>
      <td>Sci-Fi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Movie B</td>
      <td>5</td>
      <td>Sci-Fi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Movie C</td>
      <td>3</td>
      <td>Sci-Fi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Movie A</td>
      <td>2</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Movie B</td>
      <td>4</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Movie C</td>
      <td>5</td>
      <td>Action</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd

# Sample Data

movie_ratings = pd.DataFrame(movie_ratings)

# Filter Sci-Fi movies
scifi_movies = movie_ratings[movie_ratings['genre'] == 'Sci-Fi']

# Grouping Sci-Fi movies by 'title' and applying count aggregation function
scifi_popularity = scifi_movies.groupby('title').agg({'rating': 'count'})

# Sorting the Sci-Fi movies DataFrame in descending order based on the number of user ratings
sorted_scifi_popularity = scifi_popularity.sort_values(by='rating', ascending=False)

# Selecting the third most popular Sci-Fi movie
third_most_popular_scifi_movie = sorted_scifi_popularity.index[2]

# Displaying the result
print("Third Most Popular Sci-Fi Movie based on Number of User Ratings:")
print(third_most_popular_scifi_movie)

```

    Third Most Popular Sci-Fi Movie based on Number of User Ratings:
    Movie C
    


```python
import pandas as pd

def find_third_most_popular_scifi_movie(df):
    # Filter Sci-Fi movies
    scifi_movies = df[df['genre'] == 'Sci-Fi']

    # Grouping Sci-Fi movies by 'title' and applying count aggregation function
    scifi_popularity = scifi_movies.groupby('title').agg({'rating': 'count'})

    # Sorting the Sci-Fi movies DataFrame in descending order based on the number of user ratings
    sorted_scifi_popularity = scifi_popularity.sort_values(by='rating', ascending=False)

    # Check if there are at least 3 Sci-Fi movies
    if len(sorted_scifi_popularity) >= 3:
        # Selecting the third most popular Sci-Fi movie
        third_most_popular_scifi_movie = sorted_scifi_popularity.index[2]
        return third_most_popular_scifi_movie
    else:
        return "Not enough Sci-Fi movies available."

# Sample Data
data = pd.DataFrame(movie_data)
movie_ratings = pd.DataFrame(movie_ratings)

# Using the function to find the third most popular Sci-Fi movie
result = find_third_most_popular_scifi_movie(movie_ratings)

# Displaying the result
print("Third Most Popular Sci-Fi Movie based on Number of User Ratings:")
print(result)

```

    Third Most Popular Sci-Fi Movie based on Number of User Ratings:
    Movie C
    


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>imdbId</th>
      <th>tmdbId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>114709</td>
      <td>862.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>113497</td>
      <td>8844.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113228</td>
      <td>15602.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>114885</td>
      <td>31357.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>113041</td>
      <td>11862.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd

# Sample Data



# Finding the movieId of the movie with the highest IMDb rating
highest_rated_movie_id = links.loc[links['imdbId'].idxmax(), 'movieId']

# Displaying the result
print("MovieId of the Movie with the Highest IMDb Rating:")
print(highest_rated_movie_id)

```

    MovieId of the Movie with the Highest IMDb Rating:
    193587
    


```python
import pandas as pd

# Sample Data


# Filtering Sci-Fi movies
scifi_movies = movies[movies['genres'] == 'Sci-Fi']

# Finding the movieId of the Sci-Fi movie with the highest IMDb rating
highest_rated_scifi_movie_id = links.loc[links['imdbld'].idxmax(), 'movieId']

# Displaying the result
print("MovieId of the Highest Rated Sci-Fi Movie:")
print(highest_rated_scifi_movie_id)

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~\anaconda3\lib\site-packages\pandas\core\indexes\base.py:2895, in Index.get_loc(self, key, method, tolerance)
       2894 try:
    -> 2895     return self._engine.get_loc(casted_key)
       2896 except KeyError as err:
    

    File pandas\_libs\index.pyx:70, in pandas._libs.index.IndexEngine.get_loc()
    

    File pandas\_libs\index.pyx:101, in pandas._libs.index.IndexEngine.get_loc()
    

    File pandas\_libs\hashtable_class_helper.pxi:1675, in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    File pandas\_libs\hashtable_class_helper.pxi:1683, in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'genres'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    Cell In[161], line 7
          1 import pandas as pd
          3 # Sample Data
          4 
          5 
          6 # Filtering Sci-Fi movies
    ----> 7 scifi_movies = movies[movies['genres'] == 'Sci-Fi']
          9 # Finding the movieId of the Sci-Fi movie with the highest IMDb rating
         10 highest_rated_scifi_movie_id = links.loc[links['imdbld'].idxmax(), 'movieId']
    

    File ~\anaconda3\lib\site-packages\pandas\core\frame.py:2902, in DataFrame.__getitem__(self, key)
       2900 if self.columns.nlevels > 1:
       2901     return self._getitem_multilevel(key)
    -> 2902 indexer = self.columns.get_loc(key)
       2903 if is_integer(indexer):
       2904     indexer = [indexer]
    

    File ~\anaconda3\lib\site-packages\pandas\core\indexes\base.py:2897, in Index.get_loc(self, key, method, tolerance)
       2895         return self._engine.get_loc(casted_key)
       2896     except KeyError as err:
    -> 2897         raise KeyError(key) from err
       2899 if tolerance is not None:
       2900     tolerance = self._convert_tolerance(tolerance, np.asarray(key))
    

    KeyError: 'genres'



```python

```