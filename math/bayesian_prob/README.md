# Bayesian Probability

Bayesian probability is a way of interpreting and managing uncertainty. It is a framework in which probabilities are used to represent degrees of belief in different hypotheses. Unlike classical (frequentist) probability, which interprets probability as the long-run frequency of events, Bayesian probability interprets it as a measure of belief or confidence, which can be updated as new evidence is obtained.

## Bayes' Rule

Bayes' Rule, also known as Bayes' Theorem, is a fundamental theorem in Bayesian probability that describes how to update the probability of a hypothesis based on new evidence. The formula for Bayes' Rule is:

\[ P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)} \]

where:

- \( P(H|E) \) is the posterior probability, the probability of the hypothesis \( H \) given the evidence \( E \).
- \( P(E|H) \) is the likelihood, the probability of the evidence \( E \) given the hypothesis \( H \).
- \( P(H) \) is the prior probability, the initial probability of the hypothesis \( H \) before seeing the evidence.
- \( P(E) \) is the marginal likelihood or evidence, the total probability of the evidence under all possible hypotheses.

## Base Rate

The base rate (or prior probability) is the probability of a hypothesis or event before any new evidence is taken into account. It represents the initial degree of belief in a particular outcome based on general knowledge or previous information.

## Prior

The prior is the initial probability assigned to a hypothesis before any new evidence is considered. It reflects the initial degree of belief in the hypothesis based on existing knowledge or information. In Bayes' Rule, it is represented by \( P(H) \).

## Posterior

The posterior is the updated probability of a hypothesis after considering new evidence. It reflects the revised degree of belief in the hypothesis after incorporating the new information. In Bayes' Rule, it is represented by \( P(H|E) \).

## Likelihood

The likelihood is the probability of the evidence given that a particular hypothesis is true. It reflects how probable the observed evidence is under the assumption that the hypothesis is correct. In Bayes' Rule, it is represented by \( P(E|H) \).

## Application of Bayes' Rule

To apply Bayes' Rule, you follow these steps:

1. **Specify the prior probability** (\( P(H) \)): Determine the initial probability of the hypothesis based on prior knowledge.
2. **Determine the likelihood** (\( P(E|H) \)): Calculate the probability of observing the evidence assuming the hypothesis is true.
3. **Compute the marginal likelihood** (\( P(E) \)): This is the total probability of the evidence under all possible hypotheses. It can be calculated as:
   \[ P(E) = \sum_{i} P(E|H_i) \cdot P(H_i) \]
   where \( H_i \) are all possible hypotheses.
4. **Update the prior to get the posterior** (\( P(H|E) \)): Use Bayes' Rule to calculate the posterior probability.

### Example

Suppose you want to diagnose a disease. Let:
- \( H \) be the hypothesis that a person has the disease.
- \( E \) be the evidence from a positive test result.

Given:
- Prior probability (\( P(H) \)): Probability of having the disease without any test results (e.g., 0.01 or 1%).
- Likelihood (\( P(E|H) \)): Probability of a positive test result if the person has the disease (e.g., 0.99 or 99%).
- Probability of a positive test result (\( P(E) \)): This includes both true positives and false positives.

Using Bayes' Rule, you can update your belief about the person having the disease after getting a positive test result.

\[ P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)} \]

This updated probability (\( P(H|E) \)) is the posterior probability, reflecting the revised belief after considering the test result.
