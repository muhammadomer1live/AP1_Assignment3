import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn import cluster
import errors as err
import sklearn.metrics as skmet


def get_data_frames(filename, countries, indicator):
    '''
    This function returns two dataframes: one with countries as columns and 
    another with years as columns. It transposes the input dataframe, 
    converting rows into columns and columns into rows for specific rows 
    and columns. The function takes three specified arguments.

    Parameters:

    filename: Specifies the name of the file for data retrieval.
    countries: A list used to filter the data based on countries.
    indicator: An Indicator Code utilized to filter the data.
    
    Returns:

    df_countries: A dataframe where countries are represented in rows and 
    years as columns.
    df_years: A dataframe where years are displayed in rows and countries in 
    columns.
    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code']
                          ,'Country Name').reset_index()
    
    df_countries = df
    df_years = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_years.dropna()
    
    return df_countries, df_years


def get_data_frames1(filename, indicator):
    '''
    This function produces two dataframes: one with countries as columns
    and the other with years as columns. It transposes the dataframe, 
    converting rows into columns and columns into rows based on specified 
    columns and rows. The function is configured with three defined arguments 
    as outlined below.

    Parameters

    filename : Name of the file to read data.
    countries :List of countries to filter the data.
    indicator :Indicator Code to filter the data.

    Returns
    
    df_countries : This dataframe contains countries in rows and years as column.
    df_years : This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by indicator codes.
    df = df.loc[df['Indicator Code'].isin(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name'
                           , 'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Indicator Name']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value', ['Years', 'Country Name', 'Country Code']
                          , 'Indicator Code').reset_index()
    
    df_countries = df
    df_indticators = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_indticators.dropna()
    
    return df_countries, df_indticators


def poly(x, a, b, c, d):
    '''
    For fitting, a cubic polynomial is employed.
    '''
    y = a*x**3 + b*x**2 + c*x + d
    return y


def exp_growth(t, scale, growth):
    ''' 
    Computes the exponential function with free parameters for scale and growth.
    '''
    f = scale * np.exp(growth * (t-1960))
    return f


def logistics(t, scale, growth, t0):
    ''' 
    Computes the logistics function with free parameters for scale, 
    growth rate, and time of the turning point.
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f


def norm(array):
    '''
    Returns a normalized array within the range [0,1]. 
    The array may be either a NumPy array or a column from a dataframe.
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled


def norm_df(df, first=0, last=None):
    '''
    Returns and Normalizes all columns (excluding the first containing names)
    in the dataframe to [0,1]. Utilizes the "norm" function for individual 
    column normalization. Parameters "first" and "last" determine the range
    of columns to normalize, defaulting to all (None).
    '''
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df


def map_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns␣
    ↪→in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='plasma')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), ['Population Growth', 'Total Population', 'Urban Growth', 'Total Urban Pop'], rotation=90)
    plt.yticks(range(len(corr.columns)), ['Population Growth', 'Total Population', 'Urban Growth', 'Total Urban Pop'])


# Data fitting for India Population with prediction

countries = ['Italy', 'United Kingdom', 'United States', 'India', 'Brazil']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv',countries,
                             'SP.POP.TOTL')

df_y['Years'] = df_y['Years'].astype(int)

popt, covar = curve_fit(exp_growth, df_y['Years'], df_y['India'])
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df_y['India_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y["India"], label='data', color="#FFC000")
plt.plot(df_y['Years'], df_y['India_exp'], label='fit', color="#fc1111")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Year")
plt.ylabel("India Population")
plt.savefig("first-fit.png", dpi=300)
plt.show()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the 1950 population and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.07 gives a reasonable start value
popt = [7e8, 0.01]
df_y['India_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data', color="#FFC000")
plt.plot(df_y['Years'], df_y['India_exp'], label='fit', color="#fc1111")
plt.legend()
plt.xlabel("Year")
plt.ylabel("India Population")
plt.title("Improved start value")
plt.savefig("improved-start-value.png", dpi=300)
plt.show()

# fit exponential growth
popt, covar = curve_fit(exp_growth, df_y['Years'],df_y['India'], p0=[7e8, 0.02])
# much better
print("Fit parameter", popt)
df_y['India_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data', color="#FFC000")
plt.plot(df_y['Years'], df_y['India_exp'], label='fit', color="#fc1111")
plt.legend()
plt.xlabel("Year")
plt.ylabel("India Population")
plt.title("Final fit exponential growth")
plt.savefig("final-fit-exp.png", dpi=300)
plt.show()


# estimated turning year: 1990
# population in 1990: about 1135185000
# kept growth value from before
# increase scale factor and growth rate until rough fit
popt = [1135185000, 0.02, 1990]
df_y['India_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data', color="#FFC000")
plt.plot(df_y['Years'], df_y['India_log'], label='fit', color="#fc1111")
plt.legend()
plt.xlabel("Year")
plt.ylabel("India Population")
plt.title("Improved start value")
plt.savefig("improved-start-value2.png", dpi=300)
plt.show()

popt, covar = curve_fit(logistics,  df_y['Years'],df_y['India'],
p0=(6e9, 0.05, 1990.0))
print("Fit parameter", popt)
df_y['India_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['India'], label='data', color="#FFC000")
plt.plot(df_y['Years'], df_y['India_log'], label='fit', color="#fc1111")
plt.legend()
plt.xlabel("Year")
plt.ylabel("India Population")
plt.title("Logistic Function")
plt.savefig("logistics-fn1.png", dpi=300)

# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df_y['Years'], logistics, popt, sigma)
plt.figure()
plt.title("logistics function")
plt.plot(df_y['Years'], df_y['India'], label='data', color="#FFC000")
plt.plot(df_y['Years'], df_y['India_log'], label='fit', color="#fc1111")
plt.fill_between(df_y['Years'], low, up, alpha=0.7)
plt.legend()
plt.xlabel("Year")
plt.ylabel("India Population")
plt.savefig("logistics-fn.png", dpi=300)
plt.show()

print("Forcasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err.err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err.err_ranges(2050, logistics, popt, sigma)
print("2050 between ", low, "and", up)

print("Forcasted population")
low, up = err.err_ranges(2030, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)
low, up = err.err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2050, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)



# Data fitting with outliners for Total Population
# List of countries 
countries = ['Italy', 'United Kingdom', 'United States', 'India', 'Brazil']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv',countries,
                             'SP.POP.TOTL')


df_c.dropna()
df_y.dropna()


df_y['Years'] = df_y['Years'].astype(int)
x = df_y['Years'].values
y = df_y['India'].values 
z = df_y['United States'].values
w = df_y['Brazil'].values 

param, covar = curve_fit(poly, x, y)
# produce columns with fit values
df_y['fit'] = poly(df_y['Years'], *param)
# calculate the z-score
df_y['diff'] = df_y['India'] - df_y['fit']
sigma = df_y['diff'].std()
print("Number of points:", len(df_y['Years']), "std. dev. =", sigma)
# calculate z-score and extract outliers
df_y["zscore"] = np.abs(df_y["diff"] / sigma)
df_y = df_y[df_y["zscore"] < 3.0].copy()
print("Number of points:", len(df_y['Years']))

param1, covar1 = curve_fit(poly, x, z)
param2, covar2 = curve_fit(poly, x, w)

plt.figure()
plt.title("Total Popolation (Data Fitting)")
plt.scatter(x, y, label='India')
plt.scatter(x, z, label='United States')
plt.scatter(x, w, label='Brazil')
plt.xlabel('Years')
plt.ylabel('Total Population')
x = np.arange(1960,2021,10)
plt.plot(x, poly(x, *param), 'k')
plt.plot(x, poly(x, *param1), 'k')
plt.plot(x, poly(x, *param2), 'k')
plt.xlim(1960,2021)
plt.legend()
plt.savefig("total-population.png", dpi=300)
plt.show()


# Bar Chart for Urban population growth (annual %)
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv', countries
                             , 'SP.POP.GROW')
num= np.arange(5)
width= 0.2
# Select specific years data 
df_y = df_y.loc[df_y['Years'].isin(['2016', '2017', '2018', '2019', '2020'])]
years = df_y['Years'].tolist() 

#Ploting data on bar chart  
plt.figure()
plt.title('Population growth (annual %)')
plt.bar(num,df_y['Italy'], width, label='Italy')
plt.bar(num+0.2, df_y['United Kingdom'], width, label='United Kingdom')
plt.bar(num-0.2, df_y['United States'], width, label='United States')
plt.bar(num-0.4, df_y['India'], width, label='India')
plt.xticks(num, years)
plt.xlabel('Years')
plt.ylabel('Annual Growth %')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("pop-growth.png", dpi=300)
plt.show()

# Bar Chart for GDP per capita growth (annual %)
df_c, df_y = get_data_frames('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_4748430.csv'
                             , countries, 'NY.GDP.PCAP.KD.ZG')
num= np.arange(5)
width= 0.2
# Select specific years data 
df_y = df_y.loc[df_y['Years'].isin(['2016', '2017', '2018', '2019', '2020'])]
years = df_y['Years'].tolist() 

#Ploting data on bar chart  
plt.figure(dpi=144)
plt.title('GDP per capita growth (annual %)')
plt.bar(num,df_y['Italy'], width, label='Italy')
plt.bar(num+0.2, df_y['United Kingdom'], width, label='United Kingdom')
plt.bar(num-0.2, df_y['United States'], width, label='United States')
plt.bar(num-0.4, df_y['India'], width, label='India')
plt.xticks(num, years)
plt.xlabel('Years')
plt.ylabel('Annual %')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("gdp-growth.png", dpi=300)
plt.show()

# Clustering (k-means Clustering)
indicators = ['SP.POP.GROW', 'SP.POP.TOTL', 'SP.URB.GROW', 'SP.URB.TOTL']
df_y, df_i = get_data_frames1('API_19_DS2_en_csv_v2_4700503.csv'
                             ,indicators)


df_i = df_i.loc[df_i['Years'].eq('2015')]
df_i = df_i.loc[~df_i['Country Code'].isin(['XKX','MAF'])]

df_i.dropna()

# Heat Map Plot
map_corr(df_i)
plt.savefig("heatmap.png", dpi=300)
plt.show()

# Scatter Matrix Plot
pd.plotting.scatter_matrix(df_i, figsize=(9.0, 9.0))
plt.suptitle("Scatter Matrix Plot For All Countries", fontsize=20)
plt.tight_layout() # helps to avoid overlap of labels
plt.savefig("scatter-all.png", dpi=300)
plt.show()


# extract columns for fitting
df_fit = df_i[["SP.POP.GROW", "SP.URB.GROW"]].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fit. This make the plots with the
# original measurements
df_fit = norm_df(df_fit)
print(df_fit.describe())



for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))


# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))

# Individual colours can be assigned to symbols. The label l is used to the
# select the l-th number from the colour table.
plt.scatter(df_fit["SP.POP.GROW"], df_fit["SP.URB.GROW"], c=labels
            , cmap="tab20b")
# color map Accent selected to increase contrast
# show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Population Growth")
plt.ylabel("Urban Population Growth")
plt.title("3 Clusters For All Countries")
plt.savefig("pop-scatter.png", dpi=300)
plt.show()

