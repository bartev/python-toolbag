#!/usr/bin/env python
# coding: utf-8

# # Some plotting examples
# 
# https://seaborn.pydata.org/examples/index.html

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[23]:


from importlib import reload
reload(mpl)
reload(plt)
reload(sns)


# In[28]:


sns.get_data_home()


# In[29]:


fmri = sns.load_dataset('fmri')


# In[30]:


fmri


# In[31]:


sns.set_theme(style="darkgrid")
sns.lineplot(x="timepoint", y="signal",
                 hue="region", style="event",
                 data=fmri)


# # Faceted logistic regression

# In[35]:


sns.set_theme(style='darkgrid')

df = sns.load_dataset('titanic')
df.head()


# In[36]:


# Make a custom palette with gendered colors
pal = dict(male="#6495ED", female="#F08080")


# In[38]:


# Show the survival probability as a function of age and sex
g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=df,
               palette=pal, y_jitter=.02, logistic=True, truncate=False)

g.set(xlim=(0, 80), ylim=(-.05, 1.05))


# # Tutorial from beginning

# https://seaborn.pydata.org/tutorial/function_overview.html

# In[39]:


penguins = sns.load_dataset('penguins')
penguins


# ## Axes level functions (e.g. `histplot`, `scatterplot`)

# In[48]:


sns.histplot(data=penguins, x='flipper_length_mm', hue='species', multiple='stack')


# In[46]:


sns.kdeplot(data=penguins, x='flipper_length_mm', hue='species', multiple='stack', alpha=0.5)


# In[45]:


sns.kdeplot(data=penguins, x='flipper_length_mm', hue='species', multiple='layer', alpha=0.5)


# In[44]:


sns.kdeplot(data=penguins, x='flipper_length_mm', hue='species', multiple='fill', alpha=0.5)


# ## Figure level functions (`displot`, `relplot`, `catplot`)

# In[49]:


sns.displot(data=penguins, x='flipper_length_mm', hue='species', multiple='stack')


# In[51]:


sns.displot(data=penguins, x='flipper_length_mm', hue='species', multiple='stack', kind='kde', alpha=0.5)


# ## Faceting with figure level functions

# In[52]:


sns.displot(data=penguins, x='flipper_length_mm', hue='species', col='species')


# ## Axes level functioins make self contained plots

# In[57]:


f, axs = plt.subplots(1, 2, figsize=(8,4), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=penguins, x='flipper_length_mm', y='bill_length_mm', hue='species', ax=axs[0])
sns.histplot(data=penguins, x='species', hue='species', shrink=0.8, alpha=0.8, legend=False, ax=axs[1])


# In[70]:


f, axs = plt.subplots(1, 2, figsize=(8,4), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=penguins, x='flipper_length_mm', y='bill_length_mm', hue='species', ax=axs[0])
sns.histplot(data=penguins, x='species', hue='species', shrink=0.8, alpha=0.8, legend=False, ax=axs[1])
f.tight_layout()


# ## Figure level functions own the figure

# In[68]:


tips = sns.load_dataset('tips')
g = sns.relplot(data=tips, x='total_bill', y='tip')
g.ax.axline(xy1=(10, 2), slope=0.2, color='r', dashes=(5, 2))


# ## Customize plot from a figure level function

# In[72]:


g = sns.relplot(data=penguins, x='flipper_length_mm', y='bill_length_mm', col='sex')
g.set_axis_labels('Flipper length (mm)', 'Bill length (mm)')


# ## Specify figure size

# * Axes level functions
#     * size determined by axes layout of the figure
# * Figure level functions
#     * matplotlib: set `figsize` in `plt.subplots` or `mpl.Figure.set_size_inches()`
#     * seaborn: `height`, `aspect` parameters (width = height * aspect)
#         * Parameters correspond to size of each subplot

# Matplotlib

# In[73]:


f, ax = plt.subplots()


# In[74]:


f, ax = plt.subplots(1, 2, sharey=True)


# Seaborn FacetGrid

# In[75]:


g = sns.FacetGrid(penguins)


# In[76]:


g = sns.FacetGrid(penguins, col='sex')


# In[79]:


g = sns.FacetGrid(penguins, col='sex', height=3.5, aspect=1.2)


# ## Combine multiple views on the data

# `jointplot` and `pairplot`

# In[80]:


sns.jointplot(data=penguins, x='flipper_length_mm', y='bill_length_mm', hue='species')


# In[81]:


sns.pairplot(data=penguins, hue='species')


# In[82]:


sns.jointplot(data=penguins, x='flipper_length_mm', y='bill_length_mm', hue='species', kind='hist')


# # Data structures accepted by Seaborn

# https://seaborn.pydata.org/tutorial/data_structure.html

# ## Clean up messy data

# In[83]:


anagrams = sns.load_dataset('anagrams')
anagrams


# In[87]:


anagrams_long = anagrams.melt(id_vars=['subidr', 'attnr'], var_name='solutions', value_name='score')
anagrams_long.head()


# Plot the average score as a function of attention and number of solutions

# In[94]:


sns.catplot(data=anagrams_long, x='solutions', y='score', hue='attnr', kind='point')


# ## Options for visualizing long form data

# In[96]:


flights = sns.load_dataset('flights')
flights_dict = flights.to_dict()
sns.relplot(data=flights_dict, x="year", y="passengers", hue="month", kind="line")


# In[97]:


flights.head()


# In[100]:


type(flights_dict['year'])


# In[102]:


flights_avg = flights.groupby('year').mean()
flights_avg


# In[104]:


sns.relplot(data=flights_avg, x='year', y='passengers', kind='line')


# Or, pass vectors

# In[105]:


year = flights_avg.index
passengers = flights_avg['passengers']
sns.relplot(x=year, y=passengers, kind='line')


# ### Collections with different length

# In[107]:


flights_wide = flights.pivot(index="year", columns="month", values="passengers")

two_series = [flights_wide.loc[:1955, 'Jan'], flights_wide.loc[1952:, 'Aug']]
sns.relplot(data=two_series, kind='line')


# ### remove index info

# In[108]:


two_series = [s.to_numpy() for s in two_series]
sns.relplot(data=two_series, kind='line')


# # Visualizing statistical relationships

# https://seaborn.pydata.org/tutorial/relational.html

# `relplot` (figure level)
# 
# * `scatterplot()` (with `kind='scatter'`)
# * `lineplot()` (with `kind='line'`)

# ## Relate variables with scatter plots

# In[109]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


# In[110]:


tips = sns.load_dataset('tips')
sns.relplot(x='total_bill', y='tip', data=tips)


# ### Add a 3rd dimension using `hue` (change colors of points)

# In[112]:


sns.relplot(x='total_bill', y='tip', data=tips, hue='smoker')


# ### Different marker style for each class

# In[113]:


sns.relplot(x='total_bill', y='tip', data=tips, hue='smoker', style='smoker')


# ### 4D using hue and style

# In[114]:


sns.relplot(x='total_bill', y='tip', data=tips, hue='smoker', style='time')


# ### If hue is a numeric value, uses a sequential palette

# In[115]:


sns.relplot(x="total_bill", y="tip", hue="size", data=tips);


# ### Customize palette

# In[116]:


sns.relplot(x="total_bill", y="tip", hue="size", data=tips, palette="ch:r=-0.5, l=0.75");


# ### Change size of points

# In[118]:


sns.relplot(x="total_bill", y="tip", size="size", data=tips);


# In[123]:


sns.relplot(x="total_bill", y="tip", size="size", sizes=(15,200), data=tips, alpha=0.7);


# ## Emphasize continuity with line plots

# by default, the data is sorted by `x` before plotting with `lineplot`

# In[124]:


df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()


# #### Aggregation and representing uncertainty

# In[125]:


fmri = sns.load_dataset('fmri')
fmri


# Confidence intervals are computed using bootstrapping.
# 
# May be time-intensive for larger datasets

# In[132]:


sns.relplot(x='timepoint', y='signal', kind='line', data=fmri)


# In[135]:


fmri.query("timepoint == 5").agg(['mean', 'std'])


# #### No confidence interval
# 
# plots the mean of each point

# In[128]:


sns.relplot(x='timepoint', y='signal', ci=None, kind='line', data=fmri)


# #### Show standard deviation

# In[129]:


sns.relplot(x='timepoint', y='signal', ci='sd', kind='line', data=fmri)


# #### Turn off aggregation

# In[130]:


sns.relplot(x='timepoint', y='signal', estimator=None, kind='line', data=fmri)


# ## Plotting subsets of data with semantic mappings

# In[136]:


sns.relplot(x='timepoint', y='signal', hue='event', kind='line', data=fmri)


# In[137]:


sns.relplot(x='timepoint', y='signal', hue='region', style='event', kind='line', data=fmri)


# In[139]:


sns.relplot(x='timepoint', y='signal', hue='region', style='event', kind='line', markers=True, dashes=False, data=fmri)


# ## Plotting with date data

# In[146]:


df = pd.DataFrame(dict(time=pd.date_range("2021-01-01", periods=500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()
g.ax.grid(False)


# ## Show multiple relationships with facets (`col` variable)

# In[147]:


sns.relplot(x='total_bill', y='tip', hue='smoker', col='time', data=tips)


# In[148]:


sns.relplot(x="timepoint", y="signal", hue="subject",
            col="region", row="event", height=3,
            kind="line", estimator=None, data=fmri);


# In[149]:


sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            col="subject", col_wrap=5,
            height=3, aspect=.75, linewidth=2.5,
            kind="line", data=fmri.query("region == 'frontal'"));


# # Visualizing distributions of data

# The distributions module contains several functions designed to answer questions such as these. The axes-level functions are `histplot()`, `kdeplot()`, `ecdfplot()`, and `rugplot()`. They are grouped together within the figure-level `displot()`, `jointplot()`, and `pairplot()` functions.

# ## Plot univariate histograms

# In[151]:


penguins = sns.load_dataset('penguins')
sns.displot(penguins, x='flipper_length_mm');


# In[152]:


sns.displot(penguins, x='flipper_length_mm', binwidth=3);


# In[154]:


sns.displot(penguins, x='flipper_length_mm', bins=25);


# In[155]:


tips = sns.load_dataset('tips')
sns.displot(tips, x='size')


# ### specify the precise bin breaks by passing an array to bins

# In[156]:


sns.displot(tips, x="size", bins=[1, 2, 3, 4, 5, 6, 7])


# ### setting discrete=True, which chooses bin breaks that represent the unique values in a dataset with bars that are centered on their corresponding value

# In[163]:


sns.displot(tips, x="size", discrete=True, shrink=0.8, alpha=0.5)


# In[167]:


sns.displot(penguins, x="flipper_length_mm", col='sex', )


# ## `FacetGrid.map`

# In[169]:


with sns.axes_style("white"):
    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, height=2.5)
g.map(sns.scatterplot, "total_bill", "tip", color="#334488")
g.set_axis_labels("Total bill (US Dollars)", "Tip")
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])
g.fig.subplots_adjust(wspace=.02, hspace=.02)


# In[168]:


g = sns.FacetGrid(tips, col="smoker", margin_titles=True, height=4)
g.map(plt.scatter, "total_bill", "tip", color="#338844", edgecolor="white", s=50, lw=1)
for ax in g.axes.flat:
    ax.axline((0, 0), slope=.2, c=".2", ls="--", zorder=0)
g.set(xlim=(0, 60), ylim=(0, 14))


# ### Custom function to map to `FacetGrid`

# In[170]:


from scipy import stats
def quantile_plot(x, **kwargs):
    quantiles, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, quantiles, **kwargs)

g = sns.FacetGrid(tips, col="sex", height=4)
g.map(quantile_plot, "total_bill")


# In[ ]:




