
# Use cosine annealing (as in the paper).
# Decay rate is total number of data*epochs. Change this to your desired value. 
# The decay rate value below is arbitrary chosen for the NH-HAZE dehazing.

cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 0.0001, decay_steps=45*80, alpha=0.0)

# Adam SGD.

sgd = tf.keras.optimizers.Adam(
    learning_rate=cosine_decay_scheduler,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None
)
