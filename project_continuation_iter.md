## What's New

1. More model comparisons: between GMM, Parzen Window (and Kn-NN?)

   - yep. Comparison w.r.t. kn-NN would be a plus. You can stick with standard
     PW and kn-NN (that is, h1 = 1 and k1 = 1, respectively).

2. More Probability Density Function: compare not only exponential PDF but also multimodal PDF (with 3 Logistics or 3 Fisher-Tippett)

   - yep. Comparisons in terms of both graphical and quantitative results.

3. Data on multiple dimensions (2 and 3)

   - I suggest you setup a script for re-running gthe experiments on
     multivariate data for d -= 1 to (say) 4 or 5. This will allow us to plot a
     curve of the performance as a function of the dimensionality w.r.t. some
     established tehcniques.

4. Vary the number of training data (100, 1000, 10'000)

   - I suggest you start with n = 10, then you move on to 100, 1000 and
     (possibly but not necessarily) 10000.

5. do the same experiments several times in order to obtain the mean and the variance of the results.
   - Maybe not necessary, although it would be a plus. It is unnecessary for
     graphical comparisons, it might be useful for quantitative assessments
     (e.g., in terms of average ISE and the like).

## Roadmap:

- New Code:

  - More Model to predict PDF:

    - Parzen Window (h1=1) (maybe not implemented in SKlearn)
    - kn-NN (k1=1) (it's already implemented in SKLEARN)

  - More PDFs:

    - Multimodal:
      - Combining Logistic or Fischer Tippett PDFs (start with 3)

  - More Dimensions:

    - maybe just add a command option

  - More training data (already implemented):
    - add a command option to set up the number of training samples

- Do the experiments:

  - (repeat several times to get more precise results)

- Write the Report

# Update 1:

"Do not stress too much the experiments just to gain a 0.1% or the like :)

> 2. implemented multimodal PDFs

A few considerations on multidimensional pdfs:
i) keep the multidimensional experiments on hold for as long as needed to
complete successfully the experiments on multimodal estimation (or, the
other way around. i.e. focus on one task at a time)
ii) the genration of data you describe (generating individual 1-dim
components one at a time) is plausible (in so doing you are implicitly
restricting te family of pdfs underlyi8ng the data to multi-dim
distributions whose components are mutually independent, but neither the
GMM nor the MLP do know that!). Moreover, you could use the same principle
during the estimation, i.e. estimate each component individually using
1-dim GMM or MLP and eventually reconstruct the multivariate estimated
pdf by multiplying over the individual component-wise 1-dim estimates.
iii) the fact that the GMM cannot yield good results over the multi-dim
data may be OK with, insofar that our aim is to prove the MLP can perform
better than the plain GMM.

> 3. Experimented on multimodal logistics PDFs

i) your data are spread over a way too large interval (from minus
something to 20 ...) whihc is definitely difficult for the MLP to cope
with. Data should be in (say) ranges like (-1,1 or (0,1) or (0,19)
maximum.
ii) a mountain-shaped mixture pdf is to be used, not a "trident"-like.
Still, the 4 peaks of the mounts shall be visually evident from the
figures.
iii) how did you see that the MLP is overfitting/non-overfitting? (that is
an issue you mentioned a lot)

As a general point, the nature of the 3 logistic mixture seems to be too
easy task for the GMM. Combining exp and logistic mught be more eloquent.
As soon as I can (bu tomorrow tops) I will propose a mixture myself that
combines the two kinds of pdfs in a promising manner."

# Update 2:

## The professor sent a .txt file containing the sampling of 400 points for a multimodal one-dimensional pdf with a mix of exponential and logistic pdfs:

"as promised, here is a suitable estimation task involving an interesting
multi-modal mixture density.

The attached dataset ("randomized_dataset.txt") is composed of 400 random
values over the [0,10] interval drawn from a mixture of 4 component
densities.

The mixture pdf is plotted in the attached figure ("mixture-density.pdf").

The mixture combines an exponential exp(-x) with x > 0, a reflected
exponential exp(x - 10) with x < 10, and two Fisher-Tippett pdfs FT1(x,
mu1, sigma1) and FT2(x, mu2, sigma2) where:

mu1 = 4.0 sigma1 = 0.8
mu2 = 5.5 sigma2 = 0.7

such that the equation of the mixture p(x) is defined as

p(x) = 0.20 exp(-x) + 0.25 FT1(x) + 0.3 FT2(x) + 0.25 exp(x - 10)

having mixing parameters 0.20, 0.25, 0.3, and 0.25, respectively.

The Fisher-Tippet densities FT1(x) and FT2(x) are defined and computed as:

        z = exp(-(x-mu1)/sigma1)
        FT1(x) =  (z/sigma1)*exp(-z) )

and

        z = exp(-(x-mu2)/sigma2)
        FT2(x) =  (z/sigma2)*exp(-z) )

respectively.

I suggest you prepare and run an experimental setup with the present data.
Do not use too complex MLP architectures. Also, fix a proper number c of
Gaussian components in the model based on the performance of the MLP+GMM
(i.e., optimize the neural estimator first, only later check out the
results you get from a GMM with the same number c of components). Do not
worry if the Gaussian components per se have a "strange" behavior in
proximity of the borders of the deifnition domain (i.e., in proximity of 0
and 10), just focus on the final model MLP + GMM."

# Update 3 (28/11):

- Aggiustare architetture più facili con tanh. Relu iniziale.
- Print della loss sia generalizzata e sia della loss del training.
- Componenti tra 10/20
- Mescola un po' i seed
- Quando faccio gli esperimenti segnarmi gli iperparametri in modo più leggibile.
- Segnarsi gli esperimenti più interessanti e inviargli al massimo 1/2 immagini al prof ogni volta che c'è qualcosa

- fare l'optuna dei parametri della gmm