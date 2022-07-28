"""
We want to evaluate

$$ \int_{\mathbb{R}} \frac{1}{(2\pi\sigma^2)^{n/2}} \frac{1}{\sqrt{2
\pi \sigma^2_{\mu}}} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^n
(R_i-\mu)^2 - \frac{1}{2 \sigma^2_{\mu}}(\mu-\mu_0)^2 \right) \; d\mu
$$


Clearly depends on $R$ only through $\bar{R}$ and $\|R-\bar{R}\|^2_2$
$$ = \frac{\exp \left(-\frac{1}{2\sigma^2}
\|R-\bar{R}\|^2_2\right)}{(2\pi\sigma^2)^{n/2}} \int_{\mathbb{R}}
\frac{1}{\sqrt{2 \pi \sigma^2_{\mu}}} \exp\left(-\frac{n}{2\sigma^2}
(\bar{R}-\mu)^2 - \frac{1}{2 \sigma^2_{\mu}}(\mu-\mu_0)^2 \right) \;
d\mu $$


Expanding: $$ =\frac{\exp\left(-\frac{1}{2\sigma^2}
\|R-\bar{R}\|^2_2-\frac{n}{2\sigma^2}\bar{R}^2-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \int_{\mathbb{R}}
\exp\left(-\left(\frac{n}{2\sigma^2} + \frac{1}{2
\sigma^2_{\mu}}\right)\mu^2 + \mu \left(\frac{\bar{R} \cdot
n}{\sigma^2} + \frac{\mu_0}{\sigma^2_{\mu}}\right)\right) \; d\mu $$


Setting $$ \bar{\sigma}^{2} = \left(\frac{n}{2 \sigma^2} + \frac{1}{2
\sigma^2_{\mu}}\right)^{-1} $$ this is $$
=\frac{\exp\left(-\frac{1}{2\sigma^2}
\|R\|^2_2-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \cdot \sqrt{2 \pi \bar{\sigma}^2}
\int_{\mathbb{R}} \frac{1}{\sqrt{2\pi\bar{\sigma}^2}}
\exp\left(-\frac{1}{2 \bar{\sigma}^2} \mu^2 + \mu \left(\frac{\bar{R}
\cdot n}{\sigma^2} + \frac{\mu_0}{\sigma^2_{\mu}}\right)\right) \;
d\mu $$


Finally, this is $$ =\frac{\exp\left(-\frac{1}{2\sigma^2}
\|R\|^2_2-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \cdot \sqrt{2 \pi \bar{\sigma}^2}
\exp\left(\frac{\bar{\sigma}^2}{2} \left(\frac{\bar{R} \cdot
n}{\sigma^2} + \frac{\mu_0}{\sigma^2_{\mu}}\right)^2 \right) $$


Or, setting $$ \bar{\mu} = \frac{\frac{\bar{R} \cdot n}{\sigma^2} +
\frac{\mu_0}{\sigma^2_{\mu}}}{\frac{n}{\sigma^2} +
\frac{1}{\sigma^2_{\mu}}} $$ this is $$
=\frac{\exp\left(-\frac{1}{2\sigma^2} \|R\|^2_2
-\frac{\mu_0^2}{2\sigma^2_{\mu}}\right)}{(2\pi\sigma^2)^{n/2}\sqrt{2
\pi \sigma^2_{\mu}}} \cdot \sqrt{2 \pi \bar{\sigma}^2}
\exp\left(\frac{1}{2\bar{\sigma}^2} \bar{\mu}^2 \right) $$

"""

import numpy as np

def marginal_loglikelihood(response,
                           sigmasq,
                           mu_prior_mean,
                           mu_prior_var,
                           incremental=False):
    response_mean = response.mean()

    n = response.shape[0]

    sigmasq_bar = 1 / (n / sigmasq + 1 / mu_prior_var)
    mu_bar = (n * response_mean / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar

    logL = (0.5 * np.log(sigmasq_bar / mu_prior_var) +
            0.5 * (mu_bar**2 / sigmasq_bar))
    logL -= 0.5 * mu_prior_mean**2 / mu_prior_var
    if not incremental:
        logL -= n * 0.5 * np.log(sigmasq)
        logL -= 0.5 * (response**2).sum() / sigmasq
                
    return logL

