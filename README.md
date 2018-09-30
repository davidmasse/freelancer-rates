## Drivers of Freelancer Success on Upwork.com


### Data Source
Upwork.com allows freelance workers to connect with potential employers.  Here I focus on the freelancer side of the business, but jobs available on Upwork could be an area for future study.  Freelancers can propose an hourly wage/rate, list their skills, write a headline and a description of their abilities (up to several paragraphs), post a photo of themselves, list education and work experience, and display results of Upwork-supplied proficiency tests in a variety of subjects.  For the benefit of potential employers, Upwork populates profiles with tallies of hours worked, jobs done and total earnings; information on each freelancer's country, state and town; and employer ratings and reviews of past job performance.  I stuck to US freelancers, but variations across freelancer country could be an are for future study.

### Research Question and Use Case

One basic question faced by Upwork freelancers is how much to charge per hour of work. As a guide, I wanted try to make a model to predict a freelancer's hourly rate, given the other details of their profile listed above (except the total earnings, which are function of the target variable).  Another goal of this model would be to examine the freelance labor market overall and show what skills are or are not in high demand/short supply.

A second question faced by Upwork freelancers is what it takes to get hired for assignments.  To the extent that a freelancer controls the contents of their profile, what should they prioritize changing?  Upwork also has an economic interest in the question, i.e. in predicting which freelancers will reliably take on an complete work through the site.  The target variable for this question is hours worked/jobs done/earnings, but in a categorical sense: did this freelancer ever actually get hired, do work and get paid, or not?  For this question the stated hourly rate can serve as an input feature for the model.

### Data Extraction and Exploration

The first step was scraping data from as many profiles as possible.  Because Python get requests are forbidden by Upwork and becuase my API application was never approved, I used Selenium to access pages and click on and off elements.  Search results were limited to 500 pages of 10 users each, but I gradually built up groomed CSV files that can be loaded into one large dataframe of several thousand rows, each representing a freelancer on Upwork.  I removed duplicates of course and rows with no rate listed.

Hourly rates had a mean of around $45/hour with standard deviation of $39/hour.  About 0.5% of freelancers commanded over $200/hour.  Without these outliers, standard deviation falls to about $31/hour.  The right tail of the distribution is much fatter than the left, which was also true for hours worked (peaking around 100, rarely over 10,000) and jobs done (peaking under 50, rarely over 250) among those with non-zero values for these fields.

### Feature Engineering (Using Unsupervised Learning and NLP)

numeric features:
* number of skills
* hours worked
* jobs done
* number of tests
* length of headline
* number of words in headline

categorical features:
* skills cluster
* skills
* country
* city
* state

text:
* headline
* bio

Meta-features such as length of headline in characters and number of words in headline are extracted and included at this point.  NLP could be used to provide even more informative features.

Each freelancer lists zero to ten skills (ten being most common by far), chosen from a list of about 2,000 of them.  I used the 2,000 skills as column names and encoded skills possessed by a freelancer as "1" and those not possessed as "0."  I also made a restricted version of the dataframe, dropping columns for skills possessed by fewer than a specified number of freelancers (leading to around 100 skill columns instead of 2,000).

For both versions of the dataframe, I performed a k-modes clustering on the skills columns, using several different values of k.  For more information k-modes, an unsupervised learning technique analogous to k-modes, please see [my blog post](https://medium.com/@davidmasse8/unsupervised-learning-for-categorical-data-dd7e497033ae).

Using clusters as (categorical, dummified) features reduced dimensionality drastically, a necessary step given the limited supply of training data.  In addition, the reality of the domain (also discovered by the clustering algorithm) is that skills cluster together: a vector with zeros except in columns for "bookkeeping," "accounting" and "Quickbooks" is the center of a group of freelancers who do accounting work.  Similarly, there is a distinct group of freelancers who do graphic design and list various Adobe programs as skills.  Freelancers who belong to both groups are rare.

### Regression

In order to predict a freelancer's hourly rate, I first turned to a simple arithmetic procedure: for any new observation, I assigned it to a training cluster (closest centroid by Hamming distance) and then predicted a value for its rate equal to the mean rate within the training cluster.  This performed better than a null model guessing the overall training mean rate for any new observation, but root mean squared error (RMSE) was close to the original standard deviation while R2 was a positive value but less than 0.1.  

After that I used a Ridge linear regression as initial non-Ridge linear regression overfit to the training data and gave very large, variable coefficients.  Since the skills data was non-linear, and the distributions of other variables were clearly not Gaussian, I then turned to decision trees, with random forests performing better than linear regression.  The regressions with tree boosting methods (adaboost) produced discouraging negative R-squared.  K-nearest-neighbors produced similar results to the linear models.

The highest- and lowest-earning clusters did show up as the most important features (most extreme coefficients) for the linear models.

### Classification

### Results and Their Limitations

Omitted variable bias: the most crucial variable missing for each freelancer is time since joining upwork. This would allow hours worked and jobs done to become hours worked per week or per month. These activity rates would also be good regression targets to predict.
