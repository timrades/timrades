#!/usr/bin/env python
# coding: utf-8

# # Assignment 4
# ## Description
# In this assignment you must read in a file of metropolitan regions and associated sports teams from [assets/wikipedia_data.html](assets/wikipedia_data.html) and answer some questions about each metropolitan region. Each of these regions may have one or more teams from the "Big 4": NFL (football, in [assets/nfl.csv](assets/nfl.csv)), MLB (baseball, in [assets/mlb.csv](assets/mlb.csv)), NBA (basketball, in [assets/nba.csv](assets/nba.csv) or NHL (hockey, in [assets/nhl.csv](assets/nhl.csv)). Please keep in mind that all questions are from the perspective of the metropolitan region, and that this file is the "source of authority" for the location of a given sports team. Thus teams which are commonly known by a different area (e.g. "Oakland Raiders") need to be mapped into the metropolitan region given (e.g. San Francisco Bay Area). This will require some human data understanding outside of the data you've been given (e.g. you will have to hand-code some names, and might need to google to find out where teams are)!
# 
# For each sport I would like you to answer the question: **what is the win/loss ratio's correlation with the population of the city it is in?** Win/Loss ratio refers to the number of wins over the number of wins plus the number of losses. Remember that to calculate the correlation with [`pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html), so you are going to send in two ordered lists of values, the populations from the wikipedia_data.html file and the win/loss ratio for a given sport in the same order. Average the win/loss ratios for those cities which have multiple teams of a single sport. Each sport is worth an equal amount in this assignment (20%\*4=80%) of the grade for this assignment. You should only use data **from year 2018** for your analysis -- this is important!
# 
# ## Notes
# 
# 1. Do not include data about the MLS or CFL in any of the work you are doing, we're only interested in the Big 4 in this assignment.
# 2. I highly suggest that you first tackle the four correlation questions in order, as they are all similar and worth the majority of grades for this assignment. This is by design!
# 3. It's fair game to talk with peers about high level strategy as well as the relationship between metropolitan areas and sports teams. However, do not post code solving aspects of the assignment (including such as dictionaries mapping areas to teams, or regexes which will clean up names).
# 4. There may be more teams than the assert statements test, remember to collapse multiple teams in one city into a single value!

# ## Question 1
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **NHL** using **2018** data.

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import re
nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]


def nhl_correlation(): 
    # YOUR CODE HERE
    
    df_nhl_2018 = nhl_df.loc[nhl_df['year'] == 2018]
    
    
    
    total_games = pd.to_numeric(df_nhl_2018['W'], errors='coerce')+pd.to_numeric(df_nhl_2018['L'], errors='coerce')
    df_nhl_2018['ratio'] = pd.to_numeric(df_nhl_2018['W'], errors='coerce')/total_games
    
    
    df_nhl_2018['area'] = [None,
                                   'Tampa Bay Area',
                                   'Boston',
                                   'Toronto',
                                   'Miami–Fort Lauderdale',
                                   'Detroit',
                                   'Montreal',
                                   'Ottawa',
                                   'Buffalo',
                                   None,
                                   'Washington, D.C.',
                                   'Pittsburgh',
                                   'Philadelphia',
                                   'Columbus',
                                   'New York City',
                                   'Raleigh',
                                   'New York City',
                                   'New York City',
                                   None,
                                   'Nashville',
                                   'Winnipeg',
                                   'Minneapolis–Saint Paul',
                                   'Denver',
                                   'St. Louis',
                                   'Dallas–Fort Worth',
                                   'Chicago',
                                   None,
                                   'Las Vegas',
                                   'Los Angeles',
                                   'San Francisco Bay Area',
                                   'Los Angeles',
                                   'Calgary',
                                   'Edmonton',
                                   'Vancouver',
                                   'Phoenix']
    df_nhl_2018 = df_nhl_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    
    new_df = df_nhl_2018.merge(cities_pop,on='area', how='left').groupby('area').mean().dropna()
    print(new_df)
    
    new_df = new_df.dropna()
    #print(new_df)
    #print(len(new_df['area']))
    #print(nhl_df.head)
    
    #print(cities.head)
    #raise NotImplementedError()
    
    population_by_region = new_df['pop'] # pass in metropolitan area population from cities
    win_loss_by_region = new_df['ratio'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"
    
    return stats.pearsonr(population_by_region, win_loss_by_region)
nhl_correlation()


# In[ ]:





# ## Question 2
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **NBA** using **2018** data.

# In[2]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nba_df=pd.read_csv("assets/nba.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nba_correlation():
    # YOUR CODE HERE
    df_nba_2018 = nba_df.loc[nba_df['year'] == 2018]
    #print(df_nba_2018)
    
    
    total_games = pd.to_numeric(df_nba_2018['W'], errors='coerce')+pd.to_numeric(df_nba_2018['L'], errors='coerce')
    df_nba_2018['ratio'] = pd.to_numeric(df_nba_2018['W'], errors='coerce')/total_games
    
    print(df_nba_2018['team'])
    
    #df_nba_2018['area'] = df_nba_2018['team'].str.extract(r'^\S+\s+')
    #print(df_nba_2018['team'].str.extract(r'^\S+\s+'))

    
    df_nba_2018['area'] = ['Toronto',
                           'Boston',
                           'Philadelphia',
                           'Cleveland',
                           'Indianapolis',
                           'Miami–Fort Lauderdale',
                           'Milwaukee',
                           'Washington, D.C.', 
                           'Detroit',
                           'Charlotte',
                           'New York City',
                           'New York City',
                           'Chicago',
                           'Orlando',
                           'Atlanta',
                           'Houston',
                           'San Francisco Bay Area',
                           'Portland',
                           'Oklahoma City',
                           'Salt Lake City',
                           'New Orleans',
                           'San Antonio',
                           'Minneapolis–Saint Paul',
                           'Denver',
                           'Los Angeles',
                           'Los Angeles',
                           'Sacramento',
                           'Dallas–Fort Worth',
                           'Memphis',
                           'Phoenix']
        
    print(df_nba_2018['area'])

                            
    df_nba_2018 = df_nba_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    print(cities_pop)
    new_df = df_nba_2018.merge(cities_pop,on='area', how='left').groupby('area').mean()
    #print(new_df)
    
    #new_df = new_df.dropna()
    print(new_df)
    #print(len(new_df['area']))
    
    population_by_region = new_df['pop'] # pass in metropolitan area population from cities
    win_loss_by_region = new_df['ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

    #raise NotImplementedError()
    
    
    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)

nba_correlation()


# In[ ]:





# ## Question 3
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **MLB** using **2018** data.

# In[3]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def mlb_correlation(): 
    # YOUR CODE HERE
    
    df_mlb_2018 = mlb_df.loc[mlb_df['year'] == 2018]
    #print(df_mlb_2018)
    
    
    total_games = pd.to_numeric(df_mlb_2018['W'], errors='coerce')+pd.to_numeric(df_mlb_2018['L'], errors='coerce')
    df_mlb_2018['ratio'] = pd.to_numeric(df_mlb_2018['W'], errors='coerce')/total_games
    
    #print(df_mlb_2018['team'])
    
    #df_mlb_2018['area'] = df_mlb_2018['team'].str.extract(r'^\S+\s+')
    #print(df_mlb_2018['team'].str.extract(r'^\S+\s+'))

    
    df_mlb_2018['area'] = ['Boston',
'New York City',
'Tampa Bay Area',
'Toronto',
'Baltimore',
'Cleveland',
'Minneapolis–Saint Paul',
'Detroit',
'Chicago',
'Kansas City',
'Houston',
'San Francisco Bay Area',
'Seattle',
'Los Angeles',
'Dallas–Fort Worth',
'Atlanta',
'Washington, D.C.',
'Philadelphia',
'New York City',
'Miami–Fort Lauderdale',
'Milwaukee',
'Chicago',
'St. Louis',
'Pittsburgh',
'Cincinnati',
'Los Angeles',
'Denver',
'Phoenix',
'San Francisco Bay Area',
'San Diego'
        ]
        
    #print(df_mlb_2018['area'])

                            
    df_mlb_2018 = df_mlb_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    #print(cities_pop)
    new_df = df_mlb_2018.merge(cities_pop,on='area', how='left').groupby('area').mean()

    print(new_df)
    
    population_by_region = new_df['pop'] # pass in metropolitan area population from cities
    win_loss_by_region = new_df['ratio'] # pass in win/loss ratio from mlb_df in the same order as cities["Metropolitan area"]
    #raise NotImplementedError()
    
    
    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)

mlb_correlation()


# In[ ]:





# ## Question 4
# For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the **NFL** using **2018** data.

# In[145]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nfl_correlation(): 
    # YOUR CODE HERE
    
    df_nfl_2018 = nfl_df.loc[nfl_df['year'] == 2018]
    #print(df_nfl_2018)
    
    
    total_games = pd.to_numeric(df_nfl_2018['W'], errors='coerce')+pd.to_numeric(df_nfl_2018['L'], errors='coerce')
    df_nfl_2018['ratio'] = pd.to_numeric(df_nfl_2018['W'], errors='coerce')/total_games
    
    print(df_nfl_2018['team'])
    
    #df_nfl_2018['area'] = df_nfl_2018['team'].str.extract(r'^\S+\s+')
    #print(df_nfl_2018['team'].str.extract(r'^\S+\s+'))

    
    df_nfl_2018['area'] = [None,
'Boston',
'Miami–Fort Lauderdale',
'Buffalo',
'New York City',
None,
'Baltimore',
'Pittsburgh',
'Cleveland',
'Cincinnati',
None,
'Houston',
'Indianapolis',
'Nashville',
'Jacksonville',
None,
'Kansas City',
'Los Angeles',
'Denver',
'San Francisco Bay Area',
None,
'Dallas–Fort Worth',
'Philadelphia',
'Washington, D.C.',
'New York City',
None,
'Chicago',
'Minneapolis–Saint Paul',
'Green Bay',
'Detroit',
None,
'New Orleans',
'Charlotte',
'Atlanta',
'Tampa Bay Area',
None,
'Los Angeles',
'Seattle',
'San Francisco Bay Area',
'Phoenix']
        
    #print(df_nfl_2018['area'])

                            
    df_nfl_2018 = df_nfl_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    print(cities_pop)
    new_df = df_nfl_2018.merge(cities_pop,on='area', how='left').groupby('area').mean()

    print(new_df)
    
    population_by_region = new_df['pop'] # pass in metropolitan area population from cities
    win_loss_by_region = new_df['ratio'] # pass in win/loss ratio from nfl_df in the same order as cities["Metropolitan area"]
    
    #raise NotImplementedError()
    
    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)
nfl_correlation()


# In[ ]:





# ## Question 5
# In this question I would like you to explore the hypothesis that **given that an area has two sports teams in different sports, those teams will perform the same within their respective sports**. How I would like to see this explored is with a series of paired t-tests (so use [`ttest_rel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)) between all pairs of sports. Are there any sports where we can reject the null hypothesis? Again, average values where a sport has multiple teams in one region. Remember, you will only be including, for each sport, cities which have teams engaged in that sport, drop others as appropriate. This question is worth 20% of the grade for this assignment.

# In[23]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
nhl_df=pd.read_csv("assets/nhl.csv")
nba_df=pd.read_csv("assets/nba.csv")
nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]



def make_nhl_df():
    # YOUR CODE HERE
    
    df_nhl_2018 = nhl_df.loc[nhl_df['year'] == 2018]
    
    
    
    total_games = pd.to_numeric(df_nhl_2018['W'], errors='coerce')+pd.to_numeric(df_nhl_2018['L'], errors='coerce')
    df_nhl_2018['ratio'] = pd.to_numeric(df_nhl_2018['W'], errors='coerce')/total_games
    
    
    df_nhl_2018['area'] = [None,
                                   'Tampa Bay Area',
                                   'Boston',
                                   'Toronto',
                                   'Miami–Fort Lauderdale',
                                   'Detroit',
                                   'Montreal',
                                   'Ottawa',
                                   'Buffalo',
                                   None,
                                   'Washington, D.C.',
                                   'Pittsburgh',
                                   'Philadelphia',
                                   'Columbus',
                                   'New York City',
                                   'Raleigh',
                                   'New York City',
                                   'New York City',
                                   None,
                                   'Nashville',
                                   'Winnipeg',
                                   'Minneapolis–Saint Paul',
                                   'Denver',
                                   'St. Louis',
                                   'Dallas–Fort Worth',
                                   'Chicago',
                                   None,
                                   'Las Vegas',
                                   'Los Angeles',
                                   'San Francisco Bay Area',
                                   'Los Angeles',
                                   'Calgary',
                                   'Edmonton',
                                   'Vancouver',
                                   'Phoenix']
    df_nhl_2018 = df_nhl_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    
    new_df = df_nhl_2018.merge(cities_pop,on='area', how='left').groupby('area').mean().dropna()
    #print(new_df)
    
    new_df = new_df.dropna()
    #print(new_df)
    
    return new_df[['ratio']]

def make_nba_df():
    # YOUR CODE HERE
    df_nba_2018 = nba_df.loc[nba_df['year'] == 2018]
    #print(df_nba_2018)
    
    
    total_games = pd.to_numeric(df_nba_2018['W'], errors='coerce')+pd.to_numeric(df_nba_2018['L'], errors='coerce')
    df_nba_2018['ratio'] = pd.to_numeric(df_nba_2018['W'], errors='coerce')/total_games
    
    #print(df_nba_2018['team'])
    
    #df_nba_2018['area'] = df_nba_2018['team'].str.extract(r'^\S+\s+')
    #print(df_nba_2018['team'].str.extract(r'^\S+\s+'))

    
    df_nba_2018['area'] = ['Toronto',
                           'Boston',
                           'Philadelphia',
                           'Cleveland',
                           'Indianapolis',
                           'Miami–Fort Lauderdale',
                           'Milwaukee',
                           'Washington, D.C.', 
                           'Detroit',
                           'Charlotte',
                           'New York City',
                           'New York City',
                           'Chicago',
                           'Orlando',
                           'Atlanta',
                           'Houston',
                           'San Francisco Bay Area',
                           'Portland',
                           'Oklahoma City',
                           'Salt Lake City',
                           'New Orleans',
                           'San Antonio',
                           'Minneapolis–Saint Paul',
                           'Denver',
                           'Los Angeles',
                           'Los Angeles',
                           'Sacramento',
                           'Dallas–Fort Worth',
                           'Memphis',
                           'Phoenix']
        
    #print(df_nba_2018['area'])

                            
    df_nba_2018 = df_nba_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    #print(cities_pop)
    new_df = df_nba_2018.merge(cities_pop,on='area', how='left').groupby('area').mean()
    #print(new_df)
    
    #new_df = new_df.dropna()
    #print(new_df)
    #print(len(new_df['area']))
    
    return new_df[['ratio']]


def make_mlb_df(): 
    # YOUR CODE HERE
    
    df_mlb_2018 = mlb_df.loc[mlb_df['year'] == 2018]
    #print(df_mlb_2018)
    
    
    total_games = pd.to_numeric(df_mlb_2018['W'], errors='coerce')+pd.to_numeric(df_mlb_2018['L'], errors='coerce')
    df_mlb_2018['ratio'] = pd.to_numeric(df_mlb_2018['W'], errors='coerce')/total_games
    
    #print(df_mlb_2018['team'])
    
    #df_mlb_2018['area'] = df_mlb_2018['team'].str.extract(r'^\S+\s+')
    #print(df_mlb_2018['team'].str.extract(r'^\S+\s+'))

    
    df_mlb_2018['area'] = ['Boston',
'New York City',
'Tampa Bay Area',
'Toronto',
'Baltimore',
'Cleveland',
'Minneapolis–Saint Paul',
'Detroit',
'Chicago',
'Kansas City',
'Houston',
'San Francisco Bay Area',
'Seattle',
'Los Angeles',
'Dallas–Fort Worth',
'Atlanta',
'Washington, D.C.',
'Philadelphia',
'New York City',
'Miami–Fort Lauderdale',
'Milwaukee',
'Chicago',
'St. Louis',
'Pittsburgh',
'Cincinnati',
'Los Angeles',
'Denver',
'Phoenix',
'San Francisco Bay Area',
'San Diego'
        ]
        
    #print(df_mlb_2018['area'])

                            
    df_mlb_2018 = df_mlb_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    #print(cities_pop)
    new_df = df_mlb_2018.merge(cities_pop,on='area', how='left').groupby('area').mean()

    #print(new_df)
    
    return new_df[['ratio']]
    

    
def make_nfl_df(): 
    # YOUR CODE HERE
    
    df_nfl_2018 = nfl_df.loc[nfl_df['year'] == 2018]
    #print(df_nfl_2018)
    
    
    total_games = pd.to_numeric(df_nfl_2018['W'], errors='coerce')+pd.to_numeric(df_nfl_2018['L'], errors='coerce')
    df_nfl_2018['ratio'] = pd.to_numeric(df_nfl_2018['W'], errors='coerce')/total_games
    
    #print(df_nfl_2018['team'])
    
    #df_nfl_2018['area'] = df_nfl_2018['team'].str.extract(r'^\S+\s+')
    #print(df_nfl_2018['team'].str.extract(r'^\S+\s+'))

    
    df_nfl_2018['area'] = [None,
'Boston',
'Miami–Fort Lauderdale',
'Buffalo',
'New York City',
None,
'Baltimore',
'Pittsburgh',
'Cleveland',
'Cincinnati',
None,
'Houston',
'Indianapolis',
'Nashville',
'Jacksonville',
None,
'Kansas City',
'Los Angeles',
'Denver',
'San Francisco Bay Area',
None,
'Dallas–Fort Worth',
'Philadelphia',
'Washington, D.C.',
'New York City',
None,
'Chicago',
'Minneapolis–Saint Paul',
'Green Bay',
'Detroit',
None,
'New Orleans',
'Charlotte',
'Atlanta',
'Tampa Bay Area',
None,
'Los Angeles',
'Seattle',
'San Francisco Bay Area',
'Phoenix']
        
    #print(df_nfl_2018['area'])

                            
    df_nfl_2018 = df_nfl_2018[['team','area','ratio']]
    
    cities_pop = cities[['Metropolitan area','Population (2016 est.)[8]']] 
    cities_pop = cities_pop.rename(columns = {'Metropolitan area':'area',
                                             'Population (2016 est.)[8]':'pop'})
    cities_pop['pop'] = pd.to_numeric(cities_pop['pop'], errors='coerce')
    #print(cities_pop)
    new_df = df_nfl_2018.merge(cities_pop,on='area', how='left').groupby('area').mean()

    #print(new_df)
    
    return new_df[['ratio']]
    

def new_sports_df(sport_name):
    if sport_name == 'NHL':
        return make_nhl_df()
    elif sport_name == 'MLB':
        return make_mlb_df()
    elif sport_name == 'NBA':
        return make_nba_df()
    else:
        return make_nfl_df()


def sports_team_performance():
    # YOUR CODE HERE
    #raise NotImplementedError()
    
    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k:np.nan for k in sports}, index=sports)
    
    for sport1 in sports:
        for sport2 in sports:
            if sport1 is not sport2:
                merge_output = pd.merge(new_sports_df(sport1), new_sports_df(sport2), on=['area'], how='inner')
                #print(sport1, ' ', sport2)
                #print(merge_output)
                merge_output['ratio_x'] = pd.to_numeric(merge_output['ratio_x'])
                merge_output['ratio_y'] = pd.to_numeric(merge_output['ratio_y'])
                #print(merge_output['ratio_x'])
                #print(merge_output['ratio_y'])
                #print(stats.ttest_rel(merge_output['ratio_x'], merge_output['ratio_y']))
                p_values.loc[sport1, sport2] = stats.ttest_rel(merge_output['ratio_x'], merge_output['ratio_y'])[1]
                #print(p_values.loc[sport1, sport2])
    
    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values

sports_team_performance()


# In[ ]:




