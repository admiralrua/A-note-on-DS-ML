# \_notyet\_ Statistics


## Briefing 
[Statistics](https://en.wikipedia.org/wiki/Statistics) is the discipline that concerns the collection, organization, displaying, analysis, interpretation and presentation of data.

Two main statistical methods are used in data analysis: **descriptive statistics**, which summarize data from a sample using indexes such as the mean or standard deviation, and **inferential statistics**, which draw conclusions from data that are subject to random variation (e.g., observational errors, sampling variation). Recently, **exploratory data analysis** analyzing data sets to summarize their main characteristics (often with visual methods) also becomes popular. 

This section will give a quick introduction to the basis of statistics. The section is heavily based on the following comprehensive references:

- [wiki](https://en.wikipedia.org/wiki/Statistics)
- [The elements of statistical learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576) [or here](https://web.stanford.edu/~hastie/ElemStatLearn/)


## Statistical data
### Data preparation
The process of data preparation includes several works:

- **data collection** by experimental study or observational study
- **data summarising** (aka descriptive statistics) is considered as a problem of defining what aspects of statistical samples need to be described and how well they can be described from a typically limited sample of data; this work includes:
  + choosing summary statistics to describe a sample
  + summarising probability distributions of sample data 
  + summarising the relationships between different quantities measured on the same items 
- **data interpreting** (aka inferential statistics) finds the philosophy underlying statistical inference by using different data analytic techniques; this work includes:
  + summarising populations in the form of a fitted distribution or probability density function
  + summarising the relationship between variables using some type of regression analysis
  + providing ways of predicting the outcome of a random quantity given other related variables
  + examining the possibility of reducing the number of variables being considered within a problem (the task of Dimension Reduction Analysis)


### Data types
In statistics, groups of individual data points may be classified as belonging to any of various [statistical data types](https://en.wikipedia.org/wiki/Statistical_data_type). In the most general form, there are two types of data/variables:

- **categorical** variables: variables conform only to nominal or ordinal measurements which cannot be reasonably measured numerically; and
- **quantitative** variables: variables conform ratio and interval measurements which have numerical nature, either discrete or continuous.

[Level of measurement](https://en.wikipedia.org/wiki/Level_of_measurement)

| Incremental progress | Measure property | Mathematical operators | Central tendency |
| ---                  | ---              | ---                    | ---              |
| Nominal  | Classification, membership   | =, \><   | Mode            |  
| Ordinal  | Comparison, level            | \>, <    | Median          |
| Interval | Difference, affinity         | \+, \-   | Mean, deviation |
| Ratio    | Magnitude, amount            | \*, /    | Geometric mean, coefficient of variation |

Examples of simple data types:

| Data type                  | Possible values        | Example usage  | Level of measurement |
| ---                        | ---                    | ---            | ---                  |
| Binary                     | 0, 1                   | binary outcome | nominal              |
| Categorical                | 1, 2,... , n           | categorical outcome | nominal |
| Ordinal                    | integer or real number | relative score      | ordinal |
| Binomial                   | 0, 1,..., $$n$$        | number of success out of $$n$$ possibility | interval |
| Count                      | non-negative integer   | number of items in given interval/area/volume | ratio |
| Real-valued additive       | real number            | anything not varying over a large scale (temperature, relative distance...) | interval |
| Real-valued multiplicative | positive real number   | anything varying over a large scale (price, income, size...)  | ratio    |

Examples of multivariate data types:

- random vectors
- random matrices
- random sequences
- random processes
- random fields
- [Bayes networks](https://en.wikipedia.org/wiki/Bayesian_network) i.e. multilevel models or random trees


## Descriptive statistics
A descriptive statistic is a summary statistic that quantitatively describes or summarizes features of a data set. Descriptive statistics is distinguished from inferential statistics in that descriptive statistics aims to summarize a sample, rather than use the data to learn about the population that the sample of data is thought to represent. Numerical descriptors include mean and standard deviation for continuous data types, while frequency and percentage are more useful in terms of describing categorical data. The drawing of the sample has been subject to an element of randomness, hence the established numerical descriptors from the sample are also due to uncertainty. 

Numerical descriptors in descriptive statistics can be listed as:

| Property type | Descriptors |
| ---           | ---         |
| Center        | **Mean**, **Median**, **Mode**, **Mid-range** |
| Despersion    | **Variance**, **Standard deviation**, **Coefficient of variation**, **Percentile**, **Range/Interquartile range** |


### Mean / Median / Mode / Mid-range
Mean is also called the expected value of a data set. There are several kinds of means in statistics. Let's consider a set of numbers $$x_1, x_2,... , x_n$$, then the expected value of that data set is $$ \textbf{E}[X] $$.

**Pythagorean means** include three classical means

| Pythagorean means | Formulation |
| ---               | ---         |
| Arithmetic        | $$ \overline{x}_A = \frac{x_1 + x_2 + ... + x_n}{n} = \sum_{i=1}^n x_i $$ |
| Geometric         | $$ \overline{x}_G = (x_1 x_2 ... x_n)^\frac{1}{n} = \left( \prod_{i=1}^n x_i \right )^\frac{1}{n} $$ |
| Harmonic          | $$ \overline{x}_H = n \left( \sum_{i=1}^n \frac{1}{x_i} \right )^{-1} $$ |

In general, $$ \overline{x}_A \ge \overline{x}_G \ge \overline{x}_H $$.

**The generalized mean**, also known as the power mean, is an abstraction of the quadratic, arithmetic, geometric and harmonic means as follows:

$$ \overline{x} = \left( \frac{1}{n} \sum_{i=1}^n x_i^m \right )^\frac{1}{m} $$

By choosing different values for the parameter $$m$$, the following types of means are obtained:

| $$m$$ | Type of means |
| ---   | ---           |
| $$ m \rightarrow \infty $$  | maximum of $$x_i$$ |
| $$ m = 2 $$                 | quadratic mean     |
| $$ m = 1 $$                 | arithmetic mean    |
| $$ m \rightarrow 0 $$       | geometric mean     |
| $$ m = -1 $$                | harmonic mean      |
| $$ m \rightarrow -\infty $$ | minimum of $$x_i$$ |


**Weighted arithmetic mean**
The weighted arithmetic mean (or weighted average) is useful if one wants to combine average values from samples of the same population with different sample sizes:

$$ \overline{x} = \frac{\sum_{i=1}^n w_i x_i}{\sum_{i=1}^n w_i} $$

The **truncated mean** is the arithmetic mean of data values after a certain number or proportion of the highest and lowest data values have been discarded.


---
The **median** is the middle value that separates the higher half from the lower half of the data set.

The most frequently occurring values in a data set is called the **mode**.

The **mid-range** of a set of statistical data values is the arithmetic mean of the maximum and minimum values in a data set:

$$ M = \frac{\max(x_i) + \min(x_i)}{2} $$


### Standard deviation / Variance / Coefficient of variation
In statistics, the [**standard deviation**](https://en.wikipedia.org/wiki/Standard_deviation) denoted as $$\sigma$$ is a measure of the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean of the set, while a high standard deviation indicates that the values are spread out over a wider range.

Let $$X$$ be a random variable with the mean value $$\mu$$:

$$ \textbf{E}[X] = \mu $$

Then the standard deviation:

$$ \sigma = \sqrt{\textbf{E}[(X - \mu)^2]} = \sqrt{\textbf{E}[X^2] - (\textbf{E}[X])^2} $$

In the case where $$X$$ takes random values from a finite data set $$x_1, x_2,..., x_n$$ with each $$x_i$$ has probability $$p_i$$, the standard deviation will be:

$$ \sigma = \sqrt{\sum_{i=1}^n p_i (x_i - \mu)^2} = \sqrt{\sum_{i=1}^n p_i x_i^2 - \mu^2} $$ where $$ \mu = \sum_{i=1}^n p_i x_i $$. 

Apparently when each $$x_i$$ has the same probability, $$ p_i = \frac{1}{n} $$ then:

$$ \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2} = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 - \mu^2} $$ with $$ \mu = \frac{1}{n} \sum_{i=1}^n x_i $$.

The standard deviation of a continuous real-valued random variable $$X$$ with probability density function $$f(x)$$ is:

$$ \sigma = \sqrt{\int_\mathbb{X} (x - \mu)^2 f(x) d(x)} = \sqrt{\int_\mathbb{X} x^2 f(x) d(x)} - \mu^2 $$ where $$ \mu = \int_\mathbb{X} x f(x) dx $$.

Basic properties of the standard deviation:

- $$ \sigma(X+a) = \sigma(X) $$
- $$ \sigma(a X) = |a| \cdot \sigma(X) $$
- $$ \sigma(X+Y) = \sqrt{\textbf{var}(X) + \textbf{var}(Y) + \textbf{cov}(XY)} $$ in which $$ \textbf{var} $$ and $$ \textbf{cov} $$ stand for variance and covariance, respectively.


---
[**Variance**](https://en.wikipedia.org/wiki/Variance) is the expectation of the squared deviation of a random variable from its mean. The variance is the square of the standard deviation, the second central moment of a distribution, and the covariance of the random variable with itself.

From the definition: $$ \textbf{var}(X) = \textbf{cov}(X,X) = \textbf{E}[(X - \mu)^2] $$

The variance is also equivalent to the second cumulant of a probability distribution that generates $$X$$:

$$ \textbf{var}(X) = \textbf{E}[(X - \textbf{E}[X])^2] = \textbf{E}[X^2] - \textbf{E}[X]^2$$

In other words, the variance of X is equal to the mean of the square of X minus the square of the mean of X.

Basic properties of the variance:

- $$ \textbf{var}(X+a) = \textbf{var}(X) $$
- $$ \textbf{var}(a X) = a^2 \cdot \textbf{var}(X) $$
- $$ \textbf{var}(aX + bY) = a^2 \cdot \textbf{var}(X) + 2ab \cdot \textbf{cov}(XY) + b^2 \cdot \textbf{var}(Y) $$


---
[**Covariance**](https://en.wikipedia.org/wiki/Covariance) is a measure of the joint variability of two random variables, in other word, shows the tendency in the linear relationship between the variables. The normalized version of the covariance, the [**correlation coefficient**](https://en.wikipedia.org/wiki/Covariance_and_correlation), however, shows by its magnitude the strength of the linear relation.

$$ \textbf{cov}(X, Y) = \sigma_{XY} = \textbf{E}\left[ (X - \textbf{E}[X])(Y - \textbf{E}[Y]) \right] $$

$$ \textbf{cor}(X, Y) = \rho_{XY} = \frac{\textbf{E}\left[ (X - \textbf{E}[X])(Y - \textbf{E}[Y]) \right]}{\sigma_X \sigma_Y} = \frac{\sigma_{XY}}{\sigma_X \sigma_Y}$$

If the random variable pair $$(X,Y)$$ can take on the values $$(x_{i},y_{i})$$ for $$i=1,\ldots ,n$$, with equal probabilities $$p_{i}=1/n$$, then the covariance reads:

$$ \textbf{cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \textbf{E}[X])(y_i - \textbf{E}[Y]) $$

More generally, if there are $$n$$ possible realizations of $$(X,Y)$$ namely $$(x_{i},y_{i})$$ but with possibly equal probabilities $$p_{i}$$ for $$i=1,\ldots ,n$$, then the covariance is:

$$ \textbf{cov}(X, Y) = \sum_{i=1}^n p_i (x_i - \textbf{E}[X])(y_i - \textbf{E}[Y]) $$

Basic properties of the covariance:

- $$ \textbf{cov}(X, a) = 0 $$
- $$ \textbf{cov}(X, X) = \textbf{var}(X) $$
- $$ \textbf{cov}(X, Y) = \textbf{cov}(Y, X) $$
- $$ \textbf{cov}(aX, bY) = ab \cdot \textbf{cov}(X, Y) $$
- $$ \textbf{cov}(X + a, Y + b) = \textbf{cov}(X, Y) $$
- $$ \textbf{cov}(aX + bY, cU + dV) = ac \cdot \textbf{cov}(X, U) + ad \cdot \textbf{cov}(X, V) + bc \cdot \textbf{cov}(Y, U) + bd \cdot \textbf{cov}(Y, V) $$

If $$X, Y$$ are independent random variables then their covariance and correlation is zero. 


### Sample mean & sample covariance
The [**sample mean** and the **sample covariance**](https://en.wikipedia.org/wiki/Sample_mean_and_covariance) are statistics computed from a collection (the sample) of data on one or more random variables. The sample mean and sample covariance are estimators of the population mean and population covariance, where the term population refers to the set from which the sample was taken.

Let $$x_{ij}$$ be the $$i$$-th independently drawn observation ($$i = (1,2,...,n)$$) on the $$j$$-th random variable ($$j = (1,2,...,k)$$). These observations can be arranged into $$n$$ column vectors, each with $$k$$ entries.

The sample mean vector $$\bar{\mathbf{x}}$$ is a column vector whose $$j$$-th element $$\bar{x}_j$$ is the average value of the $$n$$ observations of the $$j$$-th variable as follows:

$$\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n}x_{ij}$$ with $$j = (1,2,...,k)$$

Thus, the sample mean vector contains the average of the observations for each variable as follows:

$$ \bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_{i} = \begin{bmatrix} \bar{x}_1 \\ \vdots \\ \bar{x}_j\\ \vdots \\ \bar{x}_n \end{bmatrix} $$

---
The **sample covariance** is a $$k$$-by-$$k$$ matrix $$ \mathbf{Q} = [q_{jm}] $$ with entries:

$$ q_{jm} = \frac{1}{n-1} \sum_{i=1}^n (x_{ij} - \bar{x}_j)(x_{im} - \bar{x}_m) $$

where $$q_{jm}$$ is an estimate of the covariance between the $$j$$-th variable and the $$m$$-th variable of the population underlying the data. 

If the observation vectors are arranged as the columns of a matrix $$\mathbf{X} = [\mathbf{x}_1 \mathbf{x}_2 ... \mathbf{x}_n]$$ which is a matrix of $$k \times n$$ then the sample covariance matrix can be computed as:

$$ \mathbf{Q} = \frac{1}{n-1} \left( \mathbf{X} - \bar{\mathbf{x}} \mathbf{1}_n^T \right) \left( \mathbf{X} - \bar{\mathbf{x}} \mathbf{1}_n^T \right)^T $$

where $$\mathbf{1}_n$$ is an $$n \times 1$$ vector of ones. 

---
In a **weighted sample**, each vector $$\mathbf{x}_i$$ is assigned a weight $$w_i \ge 0$$. Assumed that the weights are normalised as $$\sum_{i=1}^n w_i = 1$$, then the weighted mean vector is given by:

$$ \bar{\mathbf{x}} = \sum_{i=1}^n w_i \mathbf{x}_i $$

and the elements $$q_{jm}$$ of the weighted covariance matrix $$\mathbf{Q}$$ are:

$$ q_{jm} = \frac{1}{1 - \sum_{i=1}^n w_i^2} \sum_{i=1}^n w_i (x_{ij} - \bar{x}_j) (x_{im} - \bar{x}_m) $$

Noted that the sample mean and sample covariance are not robust statistics, meaning that they are sensitive to outliers. 


### Python example

{% tabs %} {% tab title="Intro" %}
Python implementation of basic statistic functions with naive Python and verification against numpy package.
{% endtab %}
{% tab title="Python code" %}
```python
import numpy as np

size = 10
top  = 20

x = np.array([np.random.randint(0,top) for i in range(size)])
y = np.array([np.random.randint(0,top) for i in range(size)])

n = len(x)

xsum = 0
var  = 0
cov  = 0
xmean = np.mean(x)
ymean = np.mean(y)


for i in range(n):
    xsum += x[i]
    var  += (x[i] - xmean)**2
    cov  += (x[i] - xmean)*(y[i] - ymean)
    
xsum /= n
var  /= n
cov  /= (n-1)
std   = var**0.5

xt = sorted(x)
if (n % 2 == 1):
    xmed = xt[n//2 + 1]
else:
    xmed = (xt[n//2] + xt[n//2-1])/2

npcov = np.cov(x,y)[0][1]    #  np.cov(x,y) = [[(x,x) (x,y)],[(y,x) (y,y)]]


print(' Mean   = {:.2f} >< {:.2f}'.format(np.mean(x)  ,xsum))
print(' Median = {:.2f} >< {:.2f}'.format(np.median(x),xmed))
print(' Std    = {:.2f} >< {:.2f}'.format(np.std(x)   ,std ))
print(' Var    = {:.2f} >< {:.2f}'.format(np.var(x)   ,var ))
print(' Cov_xy = {:.2f} >< {:.2f}'.format(npcov       ,cov ))
```
{% endtab %} {% endtabs %}



## Inferential statistics
Statistical inference is the process of using data analysis to deduce properties of an underlying probability distribution. Inferential statistical analysis infers properties of a population. It is assumed that the observed data set is sampled from a larger population. It uses patterns in the sample data to draw inferences about the population represented, accounting for randomness. These inferences may take the form of: answering yes/no questions about the data (hypothesis testing), estimating numerical characteristics of the data (estimation), describing associations within the data (correlation) and modeling relationships within the data (for example, using regression analysis). Inference can extend to forecasting, prediction and estimation of unobserved values either in or associated with the population being studied; it can include extrapolation and interpolation of time series or spatial data, and can also include data mining.

### Statistics, estimators and pivotal quantities

### Null hypothesis and alternative hypothesis

### Error

### Interval estimation

### Significance

### Examples
Some well-known statistical tests and procedures are:

- Regression analysis
- Time series analysis 
- Chi-squared test
- Correlation and dependence

## Exploratory data analysis
Exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. A very nice example of EDA can be found [here](https://en.wikipedia.org/wiki/Exploratory_data_analysis).
