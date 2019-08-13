"""Combination of glyph and BiRNN model"""

#WIP

#TODO: Fix enum string value
#TODO: Remove registry
#TODO: Clone BERT

import tensorflow as tf


class BiCRFModel():
    """ Standard CRF model, using BiRNNs to encode at the token level.
    Override embed method to use special embeddings, i.e. BERT features.
    """

    def __init__(self, hparams):

        self._hparams = hparams
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
        rnn_keep_prob = self.hp.dropout_keep_prob if is_training else 1.

        def build_rnn_cell():
            cell_type = self.hp.get("cell_type", "lstm")
            if cell_type == "lstm":
                cell = tf.nn.rnn_cell.LSTMCell(self.hp.hidden_size)
            elif cell_type == "sru":
                cell = tf.contrib.rnn.SRUCell(self.hp.hidden_size)
            else:
                raise ValueError(f"Unknown cell type: {cell_type}")

            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=rnn_keep_prob,
                input_keep_prob=rnn_keep_prob)
            return cell

        fw_cells = [build_rnn_cell() for _ in range(self.hp.birnn_layers)]
        bw_cells = [build_rnn_cell() for _ in range(self.hp.birnn_layers)]

        # encode inputs using RNN encoder
        seq_length = tf.reshape(
            inputs[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
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
            features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
        transition_params = _get_transition_params(self.hp.output_dim)
        with tf.control_dependencies(
                [tf.compat.v1.assert_less(targets, tf.cast(self.hp.output_dim, tf.int64))]):

            likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                predictions,
                targets,
                seq_lens,
                transition_params=transition_params)

        return tf.reduce_mean(-likelihood)

    def predict(self, *, inputs, params):
        """ Get model predictions """
        del params
        logits = self.body(inputs=inputs, mode=tf.estimator.ModeKeys.PREDICT)
        seq_lens = tf.reshape(
            inputs[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
        transition_params = _get_transition_params(self.hp.output_dim)
        predictions, best_score = tf.contrib.crf.crf_decode(
            logits,
            transition_params,
            seq_lens)
        return {
            Predictions.TAGS.value: predictions,
            Predictions.SEQUENCE_SCORE.value: best_score,
            Predictions.TOKEN_SCORES.value: tf.zeros_like(logits),
            Predictions.RAW_TOKEN_SCORES.value: logits
        }

    def predict_from_logits(self, *, logits, features):
        """ Do CRF decoding starting from logits, rather than raw input """
        seq_lens = tf.reshape(
            features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
        transition_params = _get_transition_params(self.hp.output_dim)
        predictions, best_score = tf.contrib.crf.crf_decode(
            logits,
            transition_params,
            seq_lens)
        trans_matrix = tf.convert_to_tensor(transition_params)
        trans_matrix = tf.tile(tf.expand_dims(trans_matrix, 0), [tf.shape(logits)[0], 1, 1])
        return {
            Predictions.TAGS.value: predictions,
            Predictions.SEQUENCE_SCORE.value: best_score,
            Predictions.TOKEN_SCORES.value: tf.zeros_like(logits),
            Predictions.RAW_TOKEN_SCORES.value: logits,
            Predictions.TRANSITION_PARAMS.value: trans_matrix
        }

    def get_model_fn(self, model_dir=None):


        def fn(features, labels, mode, params):
            del params
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            logits = self.body(inputs=features, mode=mode)

            star_vars = None
            if is_training:
                if "l2_vars" in self.hp:
                    star_vars = []
                    for var in tf.trainable_variables(self.hp.l2_vars):
                        name = var.name[:-2]
                        star_vars.append(tf.get_variable("star/"+name, shape=var.shape))
                    init_vars_from_checkpoint(self.hp.init_ckpt, "star")
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
                predictions[Predictions.LENGTH.value] = tf.reshape(
                    features[Features.INPUT_SEQUENCE_LENGTH.value], [-1]
                )
                est = tf.estimator.EstimatorSpec(mode, predictions=predictions)
                return est

            if mode == tf.estimator.ModeKeys.EVAL:
                loss = self.loss(predictions=logits, features=features,
                                 targets=labels, is_training=is_training)

                predictions = self.predict_from_logits(
                    logits=logits, features=features)

                seq_lens = tf.reshape(features[Features.INPUT_SEQUENCE_LENGTH.value], [-1])
                weights = tf.sequence_mask(seq_lens, dtype=tf.float32)

                predicted_labels = predictions[Predictions.TAGS.value]

                eval_metrics = {
                    'accuracy': tf.metrics.accuracy(labels, predicted_labels, weights=weights)
                }

                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metric_ops=eval_metrics)

        return fn

    @property
    def hp(self):
        return self._hparams


class GlyphCRF(BiCRFModel):
    def __init__(self, hparams):
        BiCRFModel.__init__(self, hparams)

        if self.hp.use_bert:
            self.bert = BERT(hparams)

        self.glyph_embedding_initializer = None


    def embed(self, *, inputs, is_training):
        features = []
        if Features.GLYPH_FEATURE_SEQUENCE.value in inputs:
            tf.logging.info("variables in checkpoint: "+str(tf.train.list_variables("/export/fs01/scale19/models/Autoencoder/autoencoder.ckpt")))
            raw_images = inputs[Features.GLYPH_FEATURE_SEQUENCE.value]
            batch_size = tf.shape(raw_images)[0]
            batch_len = tf.shape(raw_images)[1]
            reshaped_images = tf.reshape(raw_images, [batch_size * batch_len, 64, 64, 1])
            reshaped_images /= 255.0
            #Warning! Normalization factor will be different
            #for different image sizes.
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

        if Features.BERT_INPUT_SEQUENCE_LENGTH.value in inputs:

            features += [self.bert.embed(inputs=inputs,
                                            is_training=False)]

        if Features.GLYPH_ID.value in inputs:
            id_feats = \
                self.get_embedding(namespace='simple_embedding',
                                    inputs=inputs[Features.GLYPH_ID.value], input_dim=8106,
                                    output_dim=256,
                                    # regularizer=None,
                                    initializer=self.glyph_embedding_initializer)
            features.append(id_feats)

        if not features:
            raise ValueError("No supported features")

        if len(features) > 1:
            return tf.concat(features, axis=-1)
        else:
            return features[0]

    def get_embedding(self, *, namespace, inputs, input_dim=8106, output_dim=256, initializer):
                #8106 is the default size of my Chinese dictionary.
                #Change this number if you are using a different dictionary.
                #Change this if you want to embedding of subword_id to for example 22000.
        with tf.variable_scope(namespace):
            initializer = tf.initializers.truncated_normal(0.0, 0.001)
            embed = Embedding(input_dim, output_dim,
                                embeddings_initializer=initializer)
            embedding = embed.apply(inputs)
            return embedding