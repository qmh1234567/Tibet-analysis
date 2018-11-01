import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("jsonfile", "./../../Resources/jsonfiles/data_train.json", "Data source for the json file.")
tf.flags.DEFINE_string("cutwordfile", "./../../Resources/CutWordPath/data_train.txt", "Data source for the cutword save file.")
tf.flags.DEFINE_string("labelfile", "./../../Resources/labels/data_train_label.txt", "label save file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


# Misc Parameters
# 指定的设备不存在，允许TF自动分配设备
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
#是否打印设备分配日志
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#FLAGS保存命令行参数的数据
FLAGS = tf.flags.FLAGS

def preprocess():
    # Load data
    print("Loading data...")
    # 第一次调用
    x_text,y=data_helpers.load_data_and_labels(FLAGS.cutwordfile,FLAGS.jsonfile,FLAGS.labelfile)
    # 第二次调用
    # x_text=list(open(FLAGS.cutwordfile,"r",encoding='utf-8').read().splitlines())
    # y=np.loadtxt(FLAGS.labelfile)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("max_document_length=",max_document_length)
    # 创建词汇表
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #文本转为词ID序列，未知或填充用的词ID为0,直接全文输出
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    
    # Randomly shuffle data
    # 随机打乱数据
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
   

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # 新生成的图作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        # 创建一个默认会话
        with sess.as_default():
            cnn = TextCNN(
                # shape[1]代表取列数
                sequence_length=x_train.shape[1], # 句子长度
                num_classes=y_train.shape[1],  # 标签数
                vocab_size=len(vocab_processor.vocabulary_), # 句子数量
                embedding_size=FLAGS.embedding_dim,  # 词向量维度
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), # 卷积核大小
                num_filters=FLAGS.num_filters,  # 卷积核数量
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            # global_step 记录全局训练步骤  trainable=False 在梯度传播时不会修改global_step的值 
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 优化器  学习率=1e-3
            optimizer = tf.train.AdamOptimizer(1e-3)
            # 计算梯度
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # 进行参数更新
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    # 输出直方图
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    # 标量
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            # 进行合并
            grad_summaries_merged = tf.summary.merge(grad_summaries)
    
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            # 定义一个写入summary的目标文件，dev_summary_dir为写入文件地址
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            # 保存检查点
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # 调用sess.run 运行图，生成一步的训练过程数据
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                # 显示系统当前时间
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # 调用add_summary方法将训练过程以及训练步数保存
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0  # 测试时需要将dropout关闭
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

            # Generate batches
            # 调用脚本
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step =  tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev=preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
   
        