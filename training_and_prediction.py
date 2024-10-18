
# Fill in the relevant image arrays and hyperparameters here.

history = FFANet.fit('hazy train images array', 'clear train images array', batch_size='fill in here', epochs='fill in here', 
                     validation_data = ['hazy valid images array', 'clear valid images array'],shuffle=True)

data_test_dehazed = FFANet.predict('hazy test images array')
