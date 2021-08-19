import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from tqdm import tqdm
from utils import *
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_cifar100(batch_size, valid_size, shuffle=True, random_seed=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train, x_test = preprocess_data(x_train), norm_images(x_test)
    x_train, x_test = np.transpose(
        x_train, (0, 3, 1, 2)), np.transpose(x_test, (0, 3, 1, 2))
    x_train, y_train, x_test, y_test = x_train.astype('float32'), y_train.astype(
        'float32'), x_test.astype('float32'), y_test.astype('float32')

    # Train validation split
    num_train = len(x_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    x_valid = x_train[valid_idx, :, :, :]
    y_valid = y_train[valid_idx, :]
    x_train = x_train[train_idx, :, :, :]
    y_train = y_train[train_idx, :]

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(batch_size)

    valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_data = valid_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(batch_size)
    return train_data, valid_data, test_data



@tf.function
def train_for_epoch(model, inputs, targets, opt, loss):
    with tf.GradientTape() as tape:
        y_predict = model(inputs)
        loss_val = loss(y_true=targets, y_pred=y_predict)
    opt.apply_gradients(zip(tape.gradient(
        loss_val, model.trainable_variables), model.trainable_variables))
    return y_predict, loss_val


@tf.function
def loss(model, inputs, targets, loss_obj):
    y_predict = model(inputs)
    loss_val = loss_obj(y_true=targets, y_pred=y_predict)
    return y_predict, loss_val


def train(args):
    # create the model
    num_epochs, batch_size = args.epoch, args.batch_size
    model = resnet50(num_classes=100,sparse_level= args.sparse_level)

    # load data
    train_data, valid_data, test_data = load_cifar100(
        batch_size, 1- args.frac_train_data, random_seed=args.seed)
    # loss
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # optimizer
    opt = tf.keras.optimizers.SGD(
        learning_rate=0.1, momentum=0.9, nesterov=False, name='SGD')

    run_name = ("resnet50" + '_' + "cifar100" + "_1")

    writer_name = 'runs/' + run_name
    writer = tf.summary.create_file_writer(writer_name)
    print("Build model")
    model.build((batch_size, 3, 32, 32))

    if(args.sparse_level != 0):
        print("Load weight from mtx")
        restore_from_mtx(model,args.mtx_path)
        print("sparsity", global_sparsity(model))

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_time = tf.keras.metrics.Sum()
        print("Epoch {:03d} Training".format(epoch))
        for x, y in tqdm(train_data):
            start = time.perf_counter()
            y_predict, loss_value = train_for_epoch(model, x, y, opt, loss_obj)
            end = time.perf_counter()
            epoch_time.update_state((end - start))
            train_loss_avg.update_state(loss_value)
            train_accuracy.update_state(y, y_predict)

        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        valid_loss_avg = tf.keras.metrics.Mean()
        print("Epoch {:03d} Validation".format(epoch))
        for x, y in tqdm(valid_data):
            y_predict, loss_value = loss(model, x, y, loss_obj)

            valid_loss_avg.update_state(loss_value)
            valid_accuracy.update_state(y, y_predict)

        # for x,y in tqdm(test_data):
        #     y_predict,loss_value = loss(model, x,y, loss_obj)

        #     valid_loss_avg.update_state(loss_value)
        #     valid_accuracy.update_state(y, y_predict)

        print("Epoch {:03d} Training: loss {:.3f} accuracy {:.3%}".format(
            epoch, train_loss_avg.result(), train_accuracy.result()))
        print("Epoch {:03d} Validation: loss {:.3f} accuracy {:.3%}".format(
            epoch, valid_loss_avg.result(), valid_accuracy.result()))
        with writer.as_default():
            tf.summary.scalar(
                'train/loss', train_loss_avg.result(), step=epoch)
            tf.summary.scalar('train/accuracy',
                              train_accuracy.result(), step=epoch)
            tf.summary.scalar('valid/accuracy',
                              valid_accuracy.result(), step=epoch)
            tf.summary.scalar(
                'valid/loss', valid_loss_avg.result(), step=epoch)
            tf.summary.scalar('epoch_time', epoch_time.result(), step=epoch)

    if epoch % 50 == 0:
        print("Epoch {:03d} loss {:.3f} accuracy {:.3%}".format(
            epoch, train_loss_avg.result(), train_accuracy.result()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training Sparse Resnet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='CIFAR100',
                        dest="dataset_name", help='Dataset to train on')
    parser.add_argument("--epoch", type=int, default=5,
                        help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set seed for training")
    parser.add_argument("--opt", type=str, default='sgd',
                        dest="optimiser",
                        help='Choice of optimisation algorithm')
    parser.add_argument("--batch_size", type=int, default=128,
                    help='Batch size')
    parser.add_argument("--in_planes", type=int, default=64,
                        help='''Number of input planes in Resnet. Afterwards they duplicate after
                        each conv with stride 2 as usual.''')
    parser.add_argument("--sparse_level", help="sparse operation level 0:dense, 1:sparse+dense kernel, 2:partial sparse, 3:fully sparse", type=int, default=3)
    parser.add_argument("--mtx_path", type=str, default=None,
                        help='''Whether to load from mtx file.''')
    parser.add_argument("--frac_train_data", type=float, default=0.9, 
                        help='Fraction of data used for training (only applied in CIFAR)')
    args = parser.parse_args()

    train(args)
