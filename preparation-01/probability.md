# \_notyet\_ Probability

## Briefing

[Probability](https://en.wikipedia.org/wiki/Probability) is a measure quantifying the likelihood that events will occur. Probability quantifies as a number between 0 and 1, where 0 indicates impossibility and 1 indicates certainty. The higher the probability of an event, the more likely it is that the event will occur.

## Mathematical treatment

### Probability

#### Independent events

#### Mutually exclusive events

#### Not mutually exclusive events

#### Conditional probability

#### Inverse probability

#### Summary of probabilities

### Probability distributions

#### Discrete probability distribution

#### Continuous probability distribution

#### Classical probability distributions

#### Cumulative distribution function

## Basic terms

### Expected value

In probability theory, the **expected value** \(aka. **mean**, the **first moment**\) of a random variable is a key aspect of its probability distribution.

Let $$X$$ be a random variable with a finite number of finite outcomes $$x_1, x_2,..., x_n$$ occurring with probabilities $$p_1, p_2,..., p_n$$ respectively. The expectation of $$X$$ is defined as:

$$\textbf{E}[X] = \sum_{i=1}^n x_i p_i$$

As all probabilities $$p_i$$ add up to 1 $$\sum_{i=1}^n p_i = 1$$, the expected value is the weighted average with $$p_i$$ as the weights.

If $$X$$ is a random variable whose cumulative distribution function admits a density $$f(x)$$, then the expected value is defined as follows \(if the integral exists\):

$$\textbf{E}[X] = \int_\mathbb{R} x f(x) dx$$

The expected value of a random variable may be undefined, if the integral does not exist. An example of such a random variable is one with the Cauchy distribution, due to its large "tails".

Basic properties of the expected value include:

* Linearity: $$\textbf{E}[X+Y] = \textbf{E}[X] + \textbf{E}[Y]$$
* Linearity: $$\textbf{E}[a X] = a \textbf{E}[X]$$
* If $$X$$ and $$Y$$ are independent then $$\textbf{E}[XY] = \textbf{E}[X] \textbf{E}[Y]$$, otherwise it is not necessary.
* If $$X$$ has a probability density function $$f(x)$$ then $$\textbf{E}[g(X)] = \int_\mathbb{R} g(x) f(x) dx$$

For any probability distribution with cumulative distribution function $$F(x)$$, regardless of whether it is any kind of continuous probability distribution or a discrete probability distribution, a **median** is by definition any real number m that satisfies the inequalities:

$$\textbf{P}(X \le m) \ge \frac{1}{2}$$ and $$\textbf{P}(X \ge m) \ge \frac{1}{2}$$

For an absolutely continuous probability distribution with probability density function $$f(x)$$, the median satisfies:

$$\textbf{P}(X \le m) = \textbf{P}(X \ge m) = \int_{-\infty}^m f(x) dx = \frac{1}{2}$$

The **mode** of a set of data values $$x_k$$ is the value that appears most often, i.e. $$p_k$$ is the maximum or the probability mass function takes its maximum value at $$x_k$$.

