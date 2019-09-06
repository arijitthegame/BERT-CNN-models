"""Combination of glyph and BiLSTM-CRF model"""

import tensorflow as tf
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn as BiRNN
from ner.optim import get_training_op


def _get_transition_params(num_tags):
    with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
        transition_params = tf.get_variable(
            "transition_params",
            [num_tags, num_tags],
            trainable=True,
            initializer=tf.random_uniform_initializer)
    return transition_params


class BiLSTMCRFModel():
    """
    BiLSTM Model that gets embedding from BERT and GLYPH-CNN
    """

    def __init__(self, hparams):

        self.hp = hparams
        self.scaffold = None
        self.transition_params = None

    def embed(self, *, inputs, is_training):
        raise NotImplementedError

    def body(self, *, inputs, mode):
        """ Return token-level logits """

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Get token embeddings
        token_embeddings = self.embed(inputs=inputs, is_training=is_training)

        # Build up sentence encoder
        rnn_keep_prob = 0.5 if is_training else 1.

        def build_rnn_cell():
            cell = tf.nn.rnn_cell.LSTMCell(256)

            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=rnn_keep_prob,
                input_keep_prob=rnn_keep_prob)
            return cell

        fw_cells = [build_rnn_cell()]
        bw_cells = [build_rnn_cell()]

        # encode inputs using RNN encoder
        seq_length = tf.reshape(
            inputs["INPUT_SEQUENCE_LENGTH"], [-1])
        outputs, _, _ = BiRNN(fw_cells, bw_cells, token_embeddings,
                              sequence_length=seq_length, dtype=tf.float32)

        # convert encoded inputs to logits
        logits = tf.layers.dense(
            outputs,
            self.hp.output_dim,
            use_bias=True)

        return logits

    def loss(self, *, predictions, features, targets, is_training):
        """ For a CRF, predictions should be token-level logits and
        targets should be indexed labels.
        """
        del is_training
        seq_lens = tf.reshape(
            features["INPUT_SEQUENCE_LENGTH"], [-1])
        transition_params = _get_transition_params(self.hp.output_dim)
        with tf.control_dependencies(
                [tf.compat.v1.assert_less(targets, tf.cast(self.hp.output_dim, tf.int64))]):

            likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                predictions,
                targets,
                seq_lens,
                transition_params=transition_params)

        return tf.reduce_mean(-likelihood)


    def predict_from_logits(self, *, logits, features):
        """ Do CRF decoding starting from logits, rather than raw input """
        seq_lens = tf.reshape(
            features["INPUT_SEQUENCE_LENGTH"], [-1])
        transition_params = _get_transition_params(self.hp.output_dim)
        predictions, _ = tf.contrib.crf.crf_decode(
            logits,
            transition_params,
            seq_lens)

        return {
            "PREDICTED_TAGS": predictions,
        }

    def get_model_fn(self, model_dir=None):


        def fn(features, labels, mode, params):
            del params
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            logits = self.body(inputs=features, mode=mode)

            if is_training:
                loss = self.loss(predictions=logits,
                                 features=features, targets=labels,
                                 is_training=is_training)
                train_op = get_training_op(loss, self.hp)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  train_op=train_op,
                                                  scaffold=self.scaffold)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = self.predict_from_logits(
                    logits=logits, features=features)
                predictions["PREDICTION_LENGTH"] = tf.reshape(
                    features["INPUT_SEQUENCE_LENGTH"], [-1]
                )
                est = tf.estimator.EstimatorSpec(mode, predictions=predictions)
                return est

            if mode == tf.estimator.ModeKeys.EVAL:
                loss = self.loss(predictions=logits, features=features,
                                 targets=labels, is_training=is_training)

                predictions = self.predict_from_logits(
                    logits=logits, features=features)

                seq_lens = tf.reshape(features["INPUT_SEQUENCE_LENGTH"], [-1])
                weights = tf.sequence_mask(seq_lens, dtype=tf.float32)

                predicted_labels = predictions["PREDICTED_TAGS"]

                eval_metrics = {
                    'accuracy': tf.metrics.accuracy(labels, predicted_labels, weights=weights)
                }

                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metric_ops=eval_metrics)

        return fn


class GlyphCRF(BiLSTMCRFModel):
    def __init__(self, hparams):
        BiLSTMCRFModel.__init__(self, hparams)

        self.bert = BERT(hparams)
        self.glyph_embedding_initializer = None

    def embed(self, *, inputs, is_training):
        features = []
        raw_images = inputs["GLYPH_FEATURE_SEQUENCE"]
        batch_size = tf.shape(raw_images)[0]
        batch_len = tf.shape(raw_images)[1]
        reshaped_images = tf.reshape(raw_images, [batch_size * batch_len, 64, 64, 1])
        reshaped_images /= 255.0
        if self.hp.glyph_encoder == 'strided':
            encoder = strided_glyph_encoder(self.hp)
        elif self.hp.glyph_encoder == 'glyce_cnn':
            encoder = cnn_glyph_encoder(self.hp)
        else:
            raise ValueError(self.hparams.glyph_encoder)
        codes = encoder(reshaped_images, is_training)
        output_dim = codes.get_shape().as_list()[-1]
        reshaped_codes = tf.reshape(codes, [batch_size, batch_len, output_dim])
        features.append(reshaped_codes)

        features += [self.bert.embed(inputs=inputs,
                                        is_training=False)]

        if len(features) > 1:
            return tf.concat(features, axis=-1)
        else:
            return features[0]
