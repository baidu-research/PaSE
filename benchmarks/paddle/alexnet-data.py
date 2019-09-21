import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import math

class DataReader():
    def __init__(self, dataset_size, batch_size, num_classes):
        self.batch_size=batch_size
        self.images = np.reshape(np.random.uniform(-1, 1,
            size=dataset_size*227*227*3), [-1, 3, 227, 227])
        self.labels = np.reshape(np.random.randint(num_classes,
            size=dataset_size, dtype='int64'), [-1, 1])

    def __call__(self, feeder):
        def DataGenerator():
            for i in range(dataset_size):
                yield [self.images[i], self.labels[i]]

        reader = paddle.batch(paddle.reader.buffered(DataGenerator, 4096),
                batch_size=self.batch_size)
        reader = feeder.decorate_reader(reader, multi_devices=True)
        return reader

def Alexnet(img, label, num_classes):
    keep_prob = 0.5
    learning_rate = 0.01

    conv1 = layers.conv2d(img, 96, 11, stride=4, padding=0, act="relu",
            bias_attr=None)
    pool1 = layers.pool2d(conv1, 3, pool_stride=2)

    conv2 = layers.conv2d(pool1, 256, 5, stride=1, padding=2, act="relu",
            bias_attr=None)
    pool2 = layers.pool2d(conv2, 3, pool_stride=2)

    conv3 = layers.conv2d(pool2, 384, 3, stride=1, padding=1, act="relu",
            bias_attr=None)
    conv4 = layers.conv2d(conv3, 384, 3, stride=1, padding=1, act="relu",
            bias_attr=None)
    conv5 = layers.conv2d(conv4, 256, 3, stride=1, padding=1, act="relu",
            bias_attr=None)
    pool5 = layers.pool2d(conv5, 3, pool_stride=2)

    fc6 = layers.fc(pool5, 4096, act="relu", num_flatten_dims=1)
    fc7 = layers.fc(fc6, 4096, act="relu")
    fc8 = layers.fc(fc7, num_classes, act="relu")

    loss = layers.softmax_with_cross_entropy(logits=fc8, label=label)
    return layers.mean(loss)

def main():
    parser = argparse.ArgumentParser(formatter_class =
            argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', type=int, required=False,
            default=128, help="Batch size.")
    parser.add_argument('-t', '--epochs', type=int, required=False, default=30,
            help="No. of epochs")
    parser.add_argument('--display_steps', type=int, required=False, default=10,
            help="No. of epochs")
    parser.add_argument('--dataset_size', type=int, required=False,
            default=1000, help='Labels filename')
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    dataset_size = args.dataset_size
    num_classes = 1000
    place = fluid.CUDAPlace(0)

    # Model
    img = layers.data(name='img', shape=[3, 227, 227], dtype='float32')
    label = layers.data(name='label', shape=[1], dtype='int64');
    loss = Alexnet(img, label, num_classes)
    fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

    # Program
    startup_program = fluid.default_startup_program()
    train_program = fluid.default_main_program()
    #test_program = main_program.clone(for_test=True)
    data_feeder = fluid.DataFeeder([img, label], place)

    # Run startup program
    exe = fluid.Executor(place)
    exe.run(startup_program)

    # Compile train and test programs
    num_gpus = 8
    places =[fluid.CUDAPlace(i) for i in range(num_gpus)]
    compiled_train_prog = fluid.compiler.CompiledProgram(
            train_program).with_data_parallel(
                    loss_name=loss.name,
                    places=places)
    #compiled_test_prog = fluid.compiler.CompiledProgram(
    #        test_program).with_data_parallel(
    #                loss_name=loss.name,
    #                places=places)

    # Run train and test programs
    train_data_reader = DataReader(dataset_size, batch_size, num_classes)
    #test_data_reader = DataReader(int(dataset_size/10), batch_size, num_classes)
    for _ in range(num_epochs):
        for steps, data in enumerate(train_data_reader(data_feeder)):
            loss = exe.run(compiled_train_prog,
                    feed=data_feeder,
                    fetch_list=[loss.name])

            if steps != 0 and steps % display_steps == 0:
                print(f"Step: {step}, Loss: {loss}")
    
if __name__ == '__main__':
    main()

