# Phase 10 review: the Keras boundary before training

## Scope

This phase connects the tested game/search records to a Keras model.  It does
not tune playing strength.  Its purpose is to make policy orientation, legal
masking, augmentation, value signs, loss computation, and serialization fail
loudly before we spend meaningful GPU time.

## Architecture

The baseline trunk is four residual blocks with 64 channels.  There is no
batch normalization in the first model: the small network does not need
running training/inference statistics, and omitting them removes a common
source of student-project discrepancies.  This is an explicit ablation point,
not a claim that normalization is universally harmful.

The policy head is a 1x1 convolution with three output planes.  Flattening an
8x8x3 tensor in row, column, plane order exactly matches
`source_square * 3 + relative_direction`.  This makes forward-left, forward,
and forward-right spatial and readable instead of hiding them in a dense head.

The value head has one tanh output.  It always predicts Player 1's result.  The
model receives the existing absolute-player plane and neither the loader nor
the learner converts values to the mover's point of view.

## Data and augmentation boundary

Raw files remain unaugmented and absolute.  Each training draw selects one of
the four exact symmetries, transforms the position and every stored root move,
and negates absolute values only when the players are swapped.  Validation is
identity-only for a stable first metric; exhaustive symmetry correctness is a
unit-test obligation rather than something inferred from validation loss.

Policy targets are normalized child visit counts.  Priors are used only for a
zero-child-visit record.  The loss masks illegal logits before softmax because
the rules engine already knows legality; unmasked illegal probability is still
reported as a diagnostic.

The data split is by complete game, never by position.

## Smoke gate

The first GPU job must:

1. pass the complete test suite with TensorFlow installed;
2. generate a tiny checksummed mini-game dataset;
3. train a deliberately tiny 8-channel, one-block model;
4. report policy loss, value loss, illegal mass, and value error;
5. save and reload a native `.keras` model.

Only after this gate will we run the small 2x2 architecture/value-target
experiment.  A low training loss alone is not a playing-strength result.
