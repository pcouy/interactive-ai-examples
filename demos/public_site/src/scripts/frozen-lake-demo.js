const demo = document.querySelector('drl-frozen-lake');
const replayElement = document.querySelector('drl-frozen-lake-replay');

const PARAM_PROPERTIES = {
  'learning-rate': 'learningRate',
  epsilon: 'epsilon',
  'discount-factor': 'discountFactor',
  'slip-probability': 'slipProbability',
  'initial-value': 'initialValue',
  'candy-reward': 'candyReward',
  'replay-batch-size': 'replayBatchSize',
};

if (demo) {
  demo.addEventListener('drl-replay-state', (event) => {
    if (replayElement) {
      replayElement.replayState = event.detail;
    }
  });
}

for (const slider of document.querySelectorAll('[data-param]')) {
  const property = PARAM_PROPERTIES[slider.dataset.param];
  if (!property) {
    continue;
  }
  slider.addEventListener('drl-input', (event) => {
    if (demo) {
      demo[property] = event.detail;
    }
  });
}
