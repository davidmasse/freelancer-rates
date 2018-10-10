## Drivers of Freelancer Success on Upwork.com


### Data Source

![upwork_landing](/img/upwork_landing.png)

Upwork.com allows freelance workers to connect with potential employers.  Here I focus on the freelancer side of the business, but jobs available on Upwork (and a database relating them to freelancers and employers) could be an area for future study.  On the site, Upwork freelancers can propose an hourly wage/rate, list their skills, write a headline and a description of their abilities (up to several paragraphs), post a photo of themselves, list education and work experience, and display results of Upwork-supplied proficiency tests in a variety of subjects.  For the benefit of potential employers, Upwork populates profiles with tallies of hours worked, jobs done and total earnings; information on each freelancer's country, state and town; and employer ratings and reviews of past job performance.  I stuck to US freelancers, but variations related to freelancer country could be an area for future study.


### Research Question and Use Case

One basic question faced by Upwork freelancers is what it takes to get hired for assignments.  To the extent that a freelancer controls the contents of their profile, what should they prioritize changing?  Upwork also has an economic interest in the question, i.e. in predicting which freelancers will reliably take on and complete work through the site.  The target variable for this question is hours worked/jobs done/earnings, but in a categorical sense: did this freelancer ever actually get hired, do work and get paid, or not?  For this question the stated hourly rate can serve as an input feature for the model.

A second question faced by Upwork freelancers is how much to charge per hour of work. As a guide, I wanted try to make a model to predict a freelancer's hourly rate, given the other details of their profile listed above (except the total earnings, which are function of the target variable).  Another goal of this model would be to examine the freelance labor market overall and show what skills are or are not in high demand/short supply (also useful to freelancers in filling out their profiles).


### Data Extraction and Storage

The first step was scraping data from as many profiles as possible.  Because Python `get` requests are forbidden by Upwork and becuase my API application was never approved, I used Selenium to access pages and click on and off elements.  Search results were limited to 500 pages of 10 users each, but I gradually built up groomed CSV files that can be loaded into one large dataframe of several thousand rows, each representing a freelancer on Upwork.  I removed duplicates of course and rows with no rate listed.

For classifying freelancers and having worked or not, I used the entire dataset.  For the regression section (estimating hourly rate), I used only experienced freelancers as hourly rates stated by those who have never billed for hours on Upwork are not necessarily realistic.  

Hourly rates of experienced freelancers had a mean of around $47/hour with standard deviation of $39/hour.  About 0.5% of freelancers commanded over $200/hour.  Without these outliers, standard deviation falls to about $31/hour.  The right tail of the distribution is much fatter than the left, which was also true for hours worked (peaking around 100, rarely over 10,000) and jobs done (peaking under 50, rarely over 250) among those with non-zero values for these fields.

Top skills (rates in dollars/hour):

![top_skills](/img/top_skills.png)


### Feature Engineering (Using Unsupervised Learning and NLP)

Numeric features:
* hourly rate (dollars/hour)
* number of skills listed
* hours worked
* jobs done
* number of tests
* length of headline (in characters)
* average length of words in headline
* length of bio (in characters)
* average length of words in bio

Categorical features:
* skills
* skills cluster

Meta-features such as length of headline in characters are extracted and included at this point.  NLP could be used to provide even more informative features.

Each freelancer lists zero to ten skills (ten being most common by far), chosen from a list of about 2,000 of them.  I used the 2,000 skills as column names and encoded skills possessed by a freelancer as "1" and those not possessed as "0."  I also made a restricted version of the dataframe, dropping columns for skills possessed by fewer than a specified number of freelancers (leading to around 100 skill columns instead of 2,000).

![df](/img/df.png)

For both versions of the dataframe, I performed a k-modes clustering on the skills columns, using several different values of k.  For more information k-modes, an unsupervised learning technique analogous to k-modes, please see [my blog post](https://medium.com/@davidmasse8/unsupervised-learning-for-categorical-data-dd7e497033ae).

Using clusters as (categorical, dummified) features reduced dimensionality drastically, a necessary step given the limited supply of training data.  In addition, the reality of the domain (also discovered by the clustering algorithm) is that skills cluster together in the data: a vector with zeros except in columns for "bookkeeping," "accounting" and "Quickbooks" is the center of a group of freelancers who do accounting work.  Similarly, there is a distinct group of freelancers who do graphic design and list various Adobe programs as skills.  Freelancers who belong to both groups are rare, making the clusters separable.

Cluster sizes from 3 to 20 were tried, 10-15 clusters being the point where clusters with very similar centroids start to appear.  I used 15 clusters for most models, and 10 Huang initializations, which seemed to outperform Cao initializations.  Once a larger dataset is obtained, cluster number can be optimized further.

![clusters1](/img/clusters1.png)
![clusters2](/img/clusters2.png)

The predictor matrices (X) were also normalized and subjected to PCA.  All components were significant (eigenvalues from 1.8 down to 0.7) and therefore kept for training models.  


### Regression

In order to predict a freelancer's hourly rate, I first looked at box-and-whiskers plots of each cluster's rates, showing significantly varying means and variances.   I then turned to a simple arithmetic procedure: for any new observation, I assigned it to a training cluster (closest centroid by Hamming distance) and then predicted a value for its rate equal to the mean rate within the training cluster.  This performed better than a null model (guessing the overall training mean rate for any new observation), but root mean squared error (RMSE) was at best $1.33/hour less the null model's $39/hour.  R2 (variance in hourly rate explained by the model) was at best about 6.5%.

![cluster_boxes](/img/rate_by_cluster.png)

(All R2, RMSE, accuracy and ROC/AUC metrics given below are 4-fold cross validated.)

After that I tried a variety of machine-learning techniques to predict hourly rates, which did better but still did not establish a strong relationship.  Random forests (on the non-normalized data) performed the best, with R2 of 10.5% and improvement in RMSE over the null model of $1.97/hour.  The most important feature was the number of employment-history items that a freelancer listed on their profile, followed at a distance by average length of words in bio and headline.  Regular decision trees were only half as good while Adaboost produced very low or even negative improvements over the null model.

Random-forest feature importances (black lines show standard deviation among 500 estimators/trees):

![rf_feat_imp](/img/rf_feat_imp.png)

Ridge linear regression (alpha = 100) on the normalized, PCA data did  almost as well as random forests, achieving 9.2% R2 and an improvement in RMSE over the null model of $1.85/hour.  In contrast to random forests, clusters had the most significance in terms of Ridge regression coefficients, with high-paying clusters pushing the output up and low-paying clusters pushing it down, as expected. Regular linear regression models are only half as good, so regularization is in fact needed.

![ridge_coef](/img/ridge_coef.png)

K-nearest-neighbors on the cluster columns (grid-searched across various values for k and bit-wise distance metrics like Jaccard) unsurprisingly comes close to the 6.5% R2 and $1.33/hour improvement in RMSE of the initial method of guessing the mean of the cluster assigned.

Sequential neural nets seem to provide superior performance with RMSE in the $30-35/hour range for validation sets, but R2 is generally zero or negative, indicating there may not be enough data for the network to learn in a way generalizable to test data.

![nn_reg_acc](/img/nn_reg_acc.png)

### Classification

First a preliminary examination of marginal distribution to assess the separation of our two target values ("0" if no hours billed on Upwork, "1" if at least one hour of work has been billed).  64.2% of the observations are "1."  Looking at charts and calculating Cohen's d for each variable, I saw that the number of Upwork tests passed on the profile was the most separated (Cohen's d of 0.80).  Number of employment experiences listed came in next at Cohen's d of 0.57:

![tests_histogram](/img/tests.png)
![exp_histogram](/img/exp.png)

Random forests (used on non-normalized data) shines at classification as well as regression with an accuracy of 78% (null model is 64%) and an AUC of 84%.  But a single decision tree (depth 5) gets to 76% accuracy and 82% AUC, and Logistic regression gets the same results.  SVM (on normalized PCA data) is a tiny bit better at 77% accuracy and 82% AUC.  Naive Bayes (Gaussian) was inferior with accuracy of 68% and AUC of 73%.

![ROC_AUC](/img/ROC_AUC.png)

Unsurprisingly, the most important features had the highest values for Cohen's d: number of Upwork tests above all, followed by number of employment experiences listed, headline length and bio length.  

Random forests feature importances (black lines show standard deviation among 500 estimators/trees):

![rf_class_feat_imp](/img/rf_class_feat_imp.png)

Neural nets seem to overfit even with aggressive dropout layers.  Validation accuracy never touches the null model's 64%.

![nn_class_acc](/img/nn_class_acc.png)

### Results and Their Limitations

I still need more data and to optimize parameters systematically, but here is my preliminary conclusion: having certain skills will not get get you a first gig on Upwork as much as having taken Upwork-supplied tests and other freelancer-controlled profile content.  Both shallow and deep learning techniques struggle to estimate hourly rates based on other factors, but skill cluster seems to matter most in determining how much a freelancer can charge.

A possible explanation for the mediocre performance of the regression models I have produced so far is omitted-variable bias.  I think the most crucial variable missing for each freelancer (not public on Upwork) is time since joining Upwork, or, preferably, cumulative time spent reading and interacting with the site.  The longer a freelancer's proven track record and the more experience they have with negotiating pay, the higher their hourly rate should be.
