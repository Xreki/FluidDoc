# 单机训练最佳实践

PaddlePaddle Fluid可以支持在现代CPU、GPU平台上进行训练。如果您发现Fluid在进行单机训练的速度较慢，您可以根据这篇文档的建议对您的Fluid程序进行优化。

神经网络训练代码，通常由3个部分组成：网络构建、数据准备、执行。这篇文档将分别从这3个方向讲述PaddlePaddle Fluid训练中常用的优化方法。

- 数据准备优化
  - 分析数据准备部分的耗时
  - 优化数据准备速度的方法
- 网络配置优化
  - cuDNN的选择
  - 使用融合功能的API
- 执行优化
  - 执行器介绍
  - ParallelExecutor构建策略（BuildStrategy）介绍
  - ParallelExecutor执行策略（ExecutionStrategy）介绍
  - 运行时FLAGS设置
  - CPU训练设置
- Profile工具

## 数据准备优化
### 分析数据准备部分的耗时

数据准备部分通常又分为两个部分：数据读取部分和预处理部分。
- 数据读取部分：用户需要在Python端从磁盘中加载数据，然后将数据feed到Fluid的执行器中。
- 数据预处理部分：用户需要在Python端进行数据预处理，比如图像任务通常需要进行数据增强、裁剪等。

Fluid提供了两种数据读取方式：**同步数据读取**和**异步数据读取**，详情请参考文档[准备数据](http://paddlepaddle.org/documentation/docs/zh/1.4/user_guides/howto/prepare_data/index_cn.html)。

#### 同步数据读取

**同步数据读取**是一种简单而直观的数据准备方式，一个简单的代码示例如下所示。用户通过调用自己编写的`reader`函数，将数据完全准备好之后，再传送给执行器。因此数据准备和执行是顺序进行的，用户可通过加入Python计时函数`time.time()`来统计数据准备部分和执行部分所占用的时间，以判断数据准备部分是否是整体训练性能的瓶颈。

```python
# 读取数据
end = time.time()
for batch_id, batch in enumerate(train_reader):
    data_time = time.time() - end
    # 训练网络
    executor.run(feed=[...], fetch_list=[...])
    batch_time = time.time() - end
    end = time.time()
```

#### 异步数据读取

**异步数据读取**，在Paddle里面是通过`py_reader`接口来实现的，一个简单的代码示例如下所示。使用异步数据读取时，Paddle的C++后端会维护一个数据队列，Python端通过单独的线程向C++端的数据队列传入数据。用户可以在训练过程中打印一下数据队列中数据的个数，如果queue size始终不为空，表明Python端数据准备的速度比模型执行的速度快，这种情况下Python端的数据读取可能不是瓶颈。

```python
# 启动py_reader
train_py_reader.start()
batch_id = 0
try:
    end = time.time()
    while True:
        print("queue size: ", train_py_reader.queue.size())
        loss, = executor.run(fetch_list=[...])
        # ...
        batch_time = time.time() - end
        end = time.time()
        batch_id += 1
except fluid.core.EOFException:
    train_py_reader.reset()
```

此外，Paddle提供的一些FLAGS也能很好的帮助分析性能，比如通过设置`export FLAGS_reader_queue_speed_test_mode=True`，数据队列中的训练数据在被读取之后，不会从数据队列中弹出，这样能够保证数据队列始终不为空，这样就能够很好的评估出数据读取所占的开销。**注意**，`FLAGS_reader_queue_speed_test_mode`只能在分析的时候打开，正常训练模型时需要关闭。

### 优化数据准备速度的方法

1. 为降低训练的整体时间，建议用户使用异步数据读取的方式，并开启`use_double_buffer`。此外，用户可根据模型的实际情况设置数据队列的大小。
2. 如果数据准备的时间大于模型执行的时间，或者出现了数据队列为空的情况，这时候需要考虑对Python的用户`reader`进行加速。常用的方法为：**使用Python多进程准备数据**。一个简单的使用多进程准备数据的示例，请参考[YOLOv3](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/yolov3/reader.py)。
3. Python端的数据预处理，都是使用CPU完成。如果Paddle提供了相应功能的API，可将这部分预处理功能写到模型配置中，如此Paddle就可以使用GPU来完成该预处理功能，这样也可以减轻CPU预处理数据的负担，提升总体训练速度。

## 网络配置优化

这部分优化跟具体的网络有关，很难一概而论。在这里，我们列举出一些优化过程中遇到过的一些示例。

### cuDNN的选择

cuDNN是NVIDIA公司专门为其GPU优化过的深度神经网络计算库，其中提供了很多神经网络中常用算子的接口，Paddle的op也对其进行了封装，例如：

```python
paddle.fluid.layers.conv2d(input,
                           num_filters,
                           filter_size,
                           stride=1,
                           padding=0,
                           dilation=1,
                           groups=None,
                           param_attr=None,
                           bias_attr=None,
                           use_cudnn=True,
                           act=None,
                           name=None)
```

很多cuDNN函数具体很好的性能表现，其实现性能明显优于Paddle原生的CUDA实现，比如`conv2d`。但也不是所有的cuDNN实现都是最优，比如：
- `conv2d_transpose`，在`batch_size=1`时
- `pool2d`，在`global_pooling=True`时

这些情况下，cuDNN实现的性能差于Paddle的CUDA实现，建议手动设置`use_cudnn=False`。

### 使用融合功能的API

Paddle提供一些粗粒度的API，这些API融合了多个细粒度API的计算，比如：

```python
logits = fluid.layers.softmax(logits)
loss = fluid.layers.cross_entropy(logits, label, ignore_index=255)
```

和

```python
loss = fluid.layers.softmax_with_cross_entropy(logits, label, ignore_index=255, numeric_stable_mode=True)
```

用户网络配置中使用融合功能的API，通常能取得更好的计算性能。

## 执行优化
### 执行器介绍
目前Paddle C++后端实现了两个执行器，`Executor`和`ParallelExecutor`，这两个执行器的特征和区别如下表所示。

| 执行器 | 执行对象 | 执行策略 |
| -- | -- | -- |
| Executor | Program | 按照Program中Operator定义的先后顺序执行所有Operators。
| ParallelExecutor | SSA Graph | 根据Graph中各个节点之间的依赖关系，可设置多个线程乱序调度和执行Operators。

为了更好的分析模型中数据和op算子之间的依赖关系，`ParallelExecutor`内部首先会将输入的`Program`转为`SSA Graph`。由于模型计算的速度与模型的结构有关，不同的网络结构的最优运行时设置各不相同。为更好优化速度，Paddle提供一组构建策略（BuildStrategy）来对对Graph进行变换和优化，以及一组执行策略（ExecutionStrategy）来优化执行器的执行和调度方案。

此外，ParallelExecutor支持支持数据并行，即单进程多卡和多进程多卡，关于ParallelExecutor的具体介绍请参考[文档](http://www.paddlepaddle.org/documentation/docs/en/1.4/api_guides/low_level/parallel_executor_en.html)。

为了统一ParallelExecutor接口和Executor接口，Paddle提供了fluid.compiler.CompiledProgram接口，在数据并行模式下，该接口底层调用的是ParallelExecutor。

### ParallelExecutor构建策略（BuildStrategy）介绍

选项 | 类型 | 默认值 | 说明
-- | -- | -- | --
reduce_strategy | fluid.BuildStrategy.ReduceStrategy | fluid.BuildStrategy.ReduceStrategy.AllReduce | ParallelExecutor对于数据并行支持两种参数更新模式：AllReduce和Reduce。在AllReduce模式下，各个节点上计算得到梯度之后，调用AllReduce操作，梯度在各个节点上聚合，然后各个节点分别进行参数更新。在Reduce模式下，参数的更新操作被均匀的分配到各个节点上，即各个节点计算得到梯度之后，将梯度在指定的节点上进行Reduce，然后在该节点上，最后将更新之后的参数Broadcast到其他节点。即：如果模型中有100个参数需要更新，训练时使用的是4个节点，在AllReduce模式下，各个节点需要分别对这100个参数进行更新；在Reduce模式下，各个节点需要分别对这25个参数进行更新，最后对更新的参数Broadcast到其他节点上。
enable_backward_optimizer_op_deps | bool | FALSE | 在反向操作和参数更新操作之间添加依赖，保证在所有的反向操作都运行结束之后才开始运行参数更新操作。在多卡训练时，打开该选项可能会提升训练速度。
fuse_all_optimizer_ops | bool | FALSE | 对模型中的参数更新算法进行融合。目前只支持SGD、Adam和Momentum算法，使用该选项时，参数的梯度不能是sparse类型。
fuse_all_reduce_ops | bool | FALSE | 多GPU训练时，可以对AllReduce操作进行融合，以减少AllReduce的调用次数。默认情况下会将同一layer中参数的梯度的AllReduce操作合并成一个，比如对于fluid.layers.fc中有Weight和Bias两个参数，打开该选项之后，原本需要两次AllReduce操作，现在只用一次AllReduce操作。为支持更大粒度的Fuse，Paddle提供了FLAGS_fuse_parameter_memory_size选项，用户可以指定Fuse AllReduce操作之后，每个AllReduce操作的梯度字节数，比如每次AllReduce调用传输128MB的梯度。目前不支持sparse参数梯度。
fuse_relu_depthwise_conv | bool | FALSE | 如果模型中relu和depthwise_conv，并且是连接的，即relu->depthwise_conv，该选项可以将这两个操作合并为一个。
fuse_broadcast_ops | bool | FALSE | 在Reduce模式下，对最后的多个broadcast操作融合为一个。
mkldnn_enabled_op_types | list | {} | 如果是CPU训练，可以用mkldnn_enabled_op_types指明模型中的那些操作可以使用MKLDNN库，如果不进行设置，模型可以使用MKLDNN库的所有操作都会使用MKLDNN库。

### ParallelExecutor执行策略（ExecutionStrategy）介绍

选项 | 类型 | 默认值 | 说明
-- | -- | -- | --
num_iteration_per_drop_scope | INT | 1 | 框架在运行过程中会产生一些临时变量，这些变量被放在local   execution scope中。通常每经过一个batch就要清理一下local execution   scope中的变量，但是由于GPU是异步设备，在清理local execution   scope之前需要对所有的GPU调用一次同步操作，因此耗费的时间较长。为此我们在execution_strategy中添加了num_iteration_per_drop_scope选项。用户可以指定经过多少次迭代之后清理一次local   execution scope。
num_threads | INT | CPU：2\*dev_count；GPU：4\*dev_count | ParallelExecutor中根据Op之间的依赖关系确定Op的执行顺序的，即Op的输入都已经变为ready状态之后，该Op会被放到一个队列中，等待被执行。ParallelExecutor内部有一个任务调度线程和一个线程池，任务调度线程从队列中取出所有Ready的Op，并将其放到线程队列中。num_threads表示线程池的大小。注意：线程池不是越大越好。

执行策略配置推荐：
- 在显存足够的前提下，建议将`exec_strategy.num_iteration_per_drop_scope`设置成一个较大的值，比如设置`exec_strategy.num_iteration_per_drop_scope=100`，这样可以避免反复地申请和释放内存。该配置对小模型的优化效果非常明显。
- 对于一些较小的模型，比如mnist、[language_model](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/language_model)等，多个线程乱序调度op的开销大于其收益，因此推荐设置`exec_strategy.num_threads=1`。

### 运行时FLAGS设置

- `FLAGS_fraction_of_gpu_memory_to_use`表示每次分配GPU显存的最小单位，取值范围为0 ~ 1。由于CUDA原生的显存分配`cuMalloc`和释放`cuFree`操作均是同步操作，非常耗时，因此将`FLAGS_fraction_of_gpu_memory_to_use`设置成一个较大的值，比如0.92（默认值），可以显著地加速训练的速度。
- `FLAGS_cudnn_exhaustive_search`表示cuDNN在选取conv实现算法时采取穷举搜索策略，因此往往能选取到一个更快的conv实现算法，这对于CNN网络通常都是有加速的。但穷举搜索往往也会增加cuDNN的显存需求，因此用户可根据模型的实际情况选择是否设置该变量。
- `FLAGS_enable_cublas_tensor_op_math`表示是否使用TensorCore加速计算cuBLAS。这个环境变量只在Tesla V100以及更新的GPU上适用，且可能会带来一定的精度损失。

### CPU训练设置

- 如果使用CPU做数据并行训练，需要指定环境变量CPU_NUM，这个环境变量指定程序运行过程中使用的CPUPlace的个数。
- 如果使用CPU进行数据并行训练，并且`build_strategy.reduce_strategy=fluid.BuildStrategy.ReduceStrategy.Reduce`，所有CPUPlace上的参数是共享的，因此对于一些使用CPU进行数据并行训练的模型，选用Reduce模式可能会更快一些。

## Profile工具

为方便用户更好的发现程序中的性能瓶颈，Paddle提供了多种Profile工具，这些工具的详细介绍和使用说明请参考[性能调优](http://paddlepaddle.org/documentation/docs/zh/1.4/advanced_usage/development/profiling/index_cn.html)。