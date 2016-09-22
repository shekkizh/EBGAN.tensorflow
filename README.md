# EBGAN.tensorflow
Tensorflow implementation of [Energy Based Generative Adversarial Networks (EBGAN)](http://arxiv.org/pdf/1609.03126v2.pdf).

- [Results](#results)
- [Observations](#observations)
- [Useful links](#useful-links)

*Disclaimer: Still experimenting with higher resolution GAN* :confused: *and the repo is still under edit ...*

##Results
Randomly generated images - no cherry picking here:exclamation: 

Both results are after 8 epochs - will update results for longer epochs later.

**With Pull Away Loss term**

![](logs/images/EBGAN_with_pt/gen0pred_image.png)
![](logs/images/EBGAN_with_pt/gen1pred_image.png)
![](logs/images/EBGAN_with_pt/gen2pred_image.png)
![](logs/images/EBGAN_with_pt/gen3pred_image.png)
![](logs/images/EBGAN_with_pt/gen4pred_image.png)
![](logs/images/EBGAN_with_pt/gen5pred_image.png)
![](logs/images/EBGAN_with_pt/gen6pred_image.png)
![](logs/images/EBGAN_with_pt/gen7pred_image.png)
![](logs/images/EBGAN_with_pt/gen9pred_image.png)
![](logs/images/EBGAN_with_pt/gen8pred_image.png)

![](logs/images/EBGAN_with_pt/gen10pred_image.png)
![](logs/images/EBGAN_with_pt/gen11pred_image.png)
![](logs/images/EBGAN_with_pt/gen12pred_image.png)
![](logs/images/EBGAN_with_pt/gen13pred_image.png)
![](logs/images/EBGAN_with_pt/gen14pred_image.png)
![](logs/images/EBGAN_with_pt/gen15pred_image.png)
![](logs/images/EBGAN_with_pt/gen16pred_image.png)
![](logs/images/EBGAN_with_pt/gen17pred_image.png)
![](logs/images/EBGAN_with_pt/gen18pred_image.png)
![](logs/images/EBGAN_with_pt/gen19pred_image.png)

**Without Pull Away Loss term**

![](logs/images/EBGAN_without_pt/gen0pred_image.png)
![](logs/images/EBGAN_without_pt/gen1pred_image.png)
![](logs/images/EBGAN_without_pt/gen2pred_image.png)
![](logs/images/EBGAN_without_pt/gen3pred_image.png)
![](logs/images/EBGAN_without_pt/gen4pred_image.png)
![](logs/images/EBGAN_without_pt/gen5pred_image.png)
![](logs/images/EBGAN_without_pt/gen6pred_image.png)
![](logs/images/EBGAN_without_pt/gen7pred_image.png)
![](logs/images/EBGAN_without_pt/gen9pred_image.png)
![](logs/images/EBGAN_without_pt/gen8pred_image.png)

![](logs/images/EBGAN_without_pt/gen10pred_image.png)
![](logs/images/EBGAN_without_pt/gen11pred_image.png)
![](logs/images/EBGAN_without_pt/gen12pred_image.png)
![](logs/images/EBGAN_without_pt/gen13pred_image.png)
![](logs/images/EBGAN_without_pt/gen14pred_image.png)
![](logs/images/EBGAN_without_pt/gen15pred_image.png)
![](logs/images/EBGAN_without_pt/gen16pred_image.png)
![](logs/images/EBGAN_without_pt/gen17pred_image.png)
![](logs/images/EBGAN_without_pt/gen18pred_image.png)
![](logs/images/EBGAN_without_pt/gen19pred_image.png)

##Observations
- Setting up a energy based objective did not make training GAN easier or any better by my observation. I felt the same way after reading the paper as well - The idea of using energy was the only novel idea that was presented, the comparitive results, details on the experimentation all seemed weak.
- In fact with margin values, I had no idea how my model was doing looking at the loss term - original GAN had a nice probablistic interpretation.
- Also the model seemed to collapse suddenly and what was more interesting was it was able to recover later - this was suprising but this doesn't really mean we can train GANs in a easier manner.
- The margin term introduced in the loss was important to avoid the GAN from collapsing. I believe the folks who wrote the paper started with low margin and went in steps of 10 to avoid model failure. The loss value of the discriminator fake when the model collapses also helps in choosing the margin.
- One more thing the intoduction of margin does is, it doesn't allow the autoencoder to achieve zero reconstruction loss which can be noticed with the imperfections in the decoded image.
- Pull away term with weight 0.1 seems to affect the model minimally. 

**Failed Attempts**

- Low Margin values and very low learning rate results in model failure

![](logs/images/failed_fake1.png)    ![](logs/images/failed_real1.png)

![](logs/images/failed_real2.png)    ![](logs/images/failed_fake2.png)

**Successful Attempts**

- Results for without pullaway term and with pull away term. (The graphs seem a lot different because of the y-scale)

![](logs/images/margin20_fake.png)    ![](logs/images/margin20_real.png)

![](logs/images/pullaway_fake.png)    ![](logs/images/pullaway_real.png)

 - Example of autoencoder reconstruction - the margin term pulls the autoencoder from achieving zero loss. Also the number of layers we use for autoencoder is limited by the choice for generator which is another problem for not getting good decoded image.
 
![](logs/images/decoded.png)


##Useful links
 - [Are Energy-Based GANs any more energy-based than normal GANs?](http://www.inference.vc/are-energy-based-gans-actually-energy-based/)
 
