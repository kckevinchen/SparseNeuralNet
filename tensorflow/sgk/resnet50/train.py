import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from resnet import *
import numpy as np
from sparse_matrix import *
from tqdm import tqdm
import time
import scipy.io as sio

def test_resnet():
	opt = tf.keras.optimizers.SGD(learning_rate=0.01)
	b = tf.constant(np.random.randn(64,3,32,32).astype('float32'))
	r = resnet50()
	with tf.GradientTape(persistent=True) as tape:
		f = r(b)
		loss = tf.reduce_mean(f)
	for var in r.trainable_variables:
		if("sparse_conv1x1" in var.name):
			print(var)
			break
	grads = tape.gradient(loss,r.trainable_variables)
	opt.apply_gradients(zip(grads, r.trainable_variables))
	for grad,var in zip(grads, r.trainable_variables):
		if("sparse_conv1x1" in var.name):
			print(var)
			print(grad)
			break

def flatten_layers(model):
    edge_layer = []
    if(hasattr(model, "layers")):
        for i in model.layers:
            edge_layer += flatten_layers(i)
    else:
        edge_layer += [model]
    return edge_layer

def restore_from_mtx(model, mtx_dir):
    edge_layers = flatten_layers(model)
    prunable_layers = filter(
        lambda layer: isinstance(layer, GeneralConv2D) or isinstance(
            layer, SparseLinear), edge_layers)
    mtx_files = sorted(os.listdir( mtx_dir ), key= lambda x: int(x.split("_")[0]))
    for layer, mtx_f in zip(prunable_layers, mtx_files):
        matrix = np.array(sio.mmread(os.path.join(mtx_dir,mtx_f)).todense()).astype(np.float32)
        layer.load_from_np(matrix)
def global_sparsity(model):
    edge_layers = flatten_layers(model)
    prunable_layers = filter(
        lambda layer: isinstance(layer, GeneralConv2D) or isinstance(
            layer, SparseLinear), edge_layers)
    nnz = 0
    size = 0
    for layer in prunable_layers:
        if(isinstance(layer,GeneralConv2D)):
            if(isinstance(layer.kernel,SparseMatrix)):
                size += layer.kernel.size
                nnz += layer.kernel.nnz
            else:
                m = layer.kernel.numpy()
                nnz += np.count_nonzero(m)
                size += np.prod(m.shape)
        else:
            size += layer.w.size
            nnz += layer.w.nnz
    return round(nnz/size,2)



def train(num_epochs, batch_size):
    #create the model
    model = resnet50(num_classes=100)
    #load data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    train_data,valid_data, test_data = load_cifar100(batch_size,0.1,random_seed=229)
    #loss
    loss_obj =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def loss(model, x, y, loss_function):
        y_predict = model(x)
        return y_predict,loss_function(y_true=y, y_pred=y_predict)

    #gradient
    @tf.function
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            y_predict,loss_val = loss(model, inputs, targets, loss_obj)
        return y_predict,loss_val, tape.gradient(loss_val, model.trainable_variables)

    #optimizer
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name='SGD')

    run_name = ("resnet50" + '_' + "cifar100" +"_sparse")

    writer_name = 'runs/' + run_name
    writer = tf.summary.create_file_writer(writer_name)
    print("Build model")
    model.build((batch_size,3,32,32))


    print("Load weight from mtx")
    restore_from_mtx(model,"./mtx_229_0.1")   
    print("sparsity", global_sparsity(model))

    

    for epoch in range(num_epochs):
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_time = tf.keras.metrics.Sum()
        print("Epoch {:03d} Training".format(epoch))
        for x, y in tqdm(train_data):
            start = time.perf_counter()
            y_predict,loss_value, gradients = grad(model, x, y)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            end = time.perf_counter()
            epoch_time.update_state((end - start))
            train_loss_avg.update_state(loss_value)
            train_accuracy.update_state(y, y_predict)


        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        valid_loss_avg = tf.keras.metrics.Mean()
        print("Epoch {:03d} Validation".format(epoch))
        for x,y in tqdm(valid_data):
            y_predict,loss_value = loss(model, x,y, loss_obj)

            valid_loss_avg.update_state(loss_value)
            valid_accuracy.update_state(y, y_predict)

        # for x,y in tqdm(test_data):
        #     y_predict,loss_value = loss(model, x,y, loss_obj)

        #     valid_loss_avg.update_state(loss_value)
        #     valid_accuracy.update_state(y, y_predict)


            
        print("Epoch {:03d} Training: loss {:.3f} accuracy {:.3%}".format(epoch, train_loss_avg.result(), train_accuracy.result()))
        print("Epoch {:03d} Validation: loss {:.3f} accuracy {:.3%}".format(epoch, valid_loss_avg.result(), valid_accuracy.result()))
        with writer.as_default():
            tf.summary.scalar('train/loss', train_loss_avg.result(), step=epoch)
            tf.summary.scalar('train/accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('valid/accuracy', valid_accuracy.result(), step=epoch)
            tf.summary.scalar('valid/loss', valid_loss_avg.result(), step=epoch)
            tf.summary.scalar('epoch_time', epoch_time.result(), step=epoch)
        
    if epoch % 50 == 0:
        print("Epoch {:03d} loss {:.3f} accuracy {:.3%}".format(epoch, train_loss_avg.result(), train_accuracy.result()))
    

def compute_mean_var(image):
    # image.shape: [image_num, w, h, c]
    mean = []
    var  = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var

def norm_images(image):
    # image.shape: [image_num, w, h, c]
    image = image.astype('float32')
    mean, var = compute_mean_var(image)
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image

def preprocess_train(x):
    paddings = [[0,0],[4,4],[4,4],[0,0]]
    x = tf.pad(x, paddings)
    x = tf.image.random_crop(x,[x.shape[0],32,32,x.shape[3]])
    x = tf.image.random_flip_left_right(x).numpy()
    return norm_images(x)

def load_cifar100(batch_size,valid_size,shuffle=True,random_seed=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train,x_test = preprocess_train(x_train),norm_images(x_test)
    x_train, x_test = np.transpose(x_train, (0, 3,1,2)), np.transpose(x_test, (0,3,1,2))
    x_train, y_train, x_test, y_test = x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')

    # Train validation split
    num_train = len(x_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    x_valid = x_train[valid_idx,:,:,:]
    y_valid = y_train[valid_idx,:]
    x_train = x_train[train_idx,:,:,:]
    y_train = y_train[train_idx,:]


    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(batch_size)

    valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_data = valid_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(batch_size)
    return train_data, valid_data, test_data

def test_sparse_matrix():
	A = np.random.randn(4,4)
	A = SparseMatrix("haha", matrix=A)
	print(A._values)


if __name__ == "__main__":
    train(10, 128)
    # model = resnet50(num_classes=100)
    # # for i in model
    # model.build((128,3,32,32))
    # restore_from_mtx(model,"./mtx_229_0.1")    
    # for i in model.layers:
    #     # if(isinstance(i,tf.keras.Sequential)):
    #         for j in i.layers:
    #             if(isinstance(j,Bottleneck)):
    #                 for k in j.layers:
    #                     print(k.name)
    #             print(j.name)
        # else:
        #     print(i.name)
    # print(model.layers)
    # batch_size = 128
    # train_data, test_data = load_cifar100(batch_size,0.2)
    # print(len(train_data))
    # for x, y in train_data:
    #     print(x.shape)
    #     exit(0)