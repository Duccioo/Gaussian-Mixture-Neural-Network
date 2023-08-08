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
