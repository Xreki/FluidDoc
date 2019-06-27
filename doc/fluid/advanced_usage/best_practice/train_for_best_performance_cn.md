# 单机训练最佳实践

PaddlePaddle Fluid可以支持在CPU和GPU上进行单机和多机训练和预测。如果您发现Fluid在进行单机训练的速度较慢，显存占用较多等性能问题，您可以根据这篇文档的建议对您的Fluid程序进行优化。
- 数据读取部分
  - 判断读数据是否慢
  - 如何优化
- 选择合适的PE构建和执行策略
- cudnn最优实践

### 分析程序数据慢的原因
训练神经网络分为两个步骤：数据准备，参数更新。所以需要分析这两个部分分别占用的时间。 

在数据准备部分，使用者需要在Python端从磁盘中加载数据，然后将数据feed到Fluid的执行器中。有些情况下需要将数据在Python端进行预处理，比如在图像任务中，需要进行数据增强。Fluid提供了两种数据读取方式：同步读取和异步读取，关于这里中数据读取方式的详细介绍请参考[准备数据](http://paddlepaddle.org/documentation/docs/zh/1.4/user_guides/howto/prepare_data/index_cn.html)。为降低数据读取的时间，建议用户使用异步读取的方式。   

在参数更新部分，也就是进行模型训练，Fluid首先将用户定义的模型转化为Graph，然后根据输入的数据以及Graph计算出一个损失值（loss），之后根据loss计算模型中所有参数的梯度，如果是数据并行模式，需要对梯度进行聚合，最后更新模型中的参数。该过程完全在C++端的执行器中进行。由于模型计算的速度与模型的结构有关，为更好优化速度，模型转化为Graph模块和执行器模块暴露出了一组参数，可以通过调整这组参数来优化速度。这部分会在下面详细介绍。

为方便用户更好的发现程序中的性能瓶颈，Paddle提供了多种Profile工具，这些工具的详细介绍和使用说明请参考[性能调优](http://paddlepaddle.org/documentation/docs/zh/1.4/advanced_usage/development/profiling/index_cn.html)。这些工具都是经过实际验证，而且Paddle的核心开发人员通常也是使用这些工具对框架进行优化的。

### 数据读取相关优化 
1. 确认数据读取是否是性能瓶颈。对于同步数据读取，执行器需要等待Python端产生数之后才能计算。对于异步数据读取，C++部分会维护一个数据队列，Python端通过单独的线程向C++端的数据队列传入数据。

如果你的模型使用的是异步数据方式，即py_reader，你可以在训练过程中打印一下数据队列中数据的个数，即：
```
    train_py_reader.start()
    batch_id = 0
    try:
        while True:
            t1 = time.time()
            print("queue size: ", train_py_reader.queue.size())
            loss, = train_exe.run(fetch_list=[loss])
            t2 = time.time()
            # ...
            batch_id += 1
                
        except fluid.core.EOFException:
            train_py_reader.reset()
```
如果queue size始终不为空，表明Python端数据加速的速度比模型训练的速度快，这种情况下Python端的数据读取可能不是瓶颈。

此外，Paddle提供的一些FLAGS也能很好的帮助分析性能，比如对于异步数据读取，如果FLAGS_reader_queue_speed_test_mode打开，数据队列中的训练数据在被读取之后，不会从数据队列中弹出，这样能够保证数据队列始终不为空，这样就能够很好的评估出数据读取所占的开销。

### 模型训练相关优化
#### 执行器介绍
目前Paddle中有两个执行器，Executor和ParallelExecutor，这两个执行器的区别。

执行器 | 执行对象 | 执行策略
-- | -- | --
Executor | Program | 根据Program中operator定义的先后顺序依次运行.
ParallelExecutor | SSA Graph | 根据Graph中各个节点之间的依赖关系，通过多线程运行.

为了更好的分析模型，ParallelExecutor内部首先会将输入的Program转为SSA Graph，然后根据build_strategy中的配置，通过一系列的Pass对Graph进行优化，比如memory optimize，operator fuse等优化。最后根据execution_strategy中的配置执行训练任务。

此外，ParallelExecutor支持支持数据并行，即单进程多卡和多进程多卡，关于ParallelExecutor的具体介绍请参考[文档](http://www.paddlepaddle.org/documentation/docs/en/1.4/api_guides/low_level/parallel_executor_en.html).

为了统一ParallelExecutor接口和Executor接口，Paddle提供了fluid.compiler.CompiledProgram接口，在数据并行模式下，该接口底层调用的是ParallelExecutor。

##### BuildStrategy中参数配置说明

选项 | 类型 | 默认值 | 说明
-- | -- | -- | --
reduce_strategy | fluid.BuildStrategy.ReduceStrategy | fluid.BuildStrategy.ReduceStrategy.AllReduce | ParallelExecutor对于数据并行支持两种参数更新模式：AllReduce和Reduce。在AllReduce模式下，各个节点上计算得到梯度之后，调用AllReduce操作，梯度在各个节点上聚合，然后各个节点分别进行参数更新。在Reduce模式下，参数的更新操作被均匀的分配到各个节点上，即各个节点计算得到梯度之后，将梯度在指定的节点上进行Reduce，然后在该节点上，最后将更新之后的参数Broadcast到其他节点。即：如果模型中有100个参数需要更新，训练时使用的是4个节点，在AllReduce模式下，各个节点需要分别对这100个参数进行更新；在Reduce模式下，各个节点需要分别对这25个参数进行更新，最后对更新的参数Broadcast到其他节点上.
enable_backward_optimizer_op_deps | bool | FALSE | 在反向操作和参数更新操作之间添加依赖，保证在所有的反向操作都运行结束之后才开始运行参数更新操作。在多卡训练时，打开该选项可能会提升训练速度.
fuse_all_optimizer_ops | bool | FALSE | 对模型中的参数更新算法进行融合。目前只支持SGD、Adam和Momentum算法，使用该选项时，参数的梯度不能是sparse类型.
fuse_all_reduce_ops | bool | FALSE | 多GPU训练时，可以对AllReduce操作进行融合，以减少AllReduce的调用次数。默认情况下会将同一layer中参数的梯度的AllReduce操作合并成一个，比如对于fluid.layers.fc中有Weight和Bias两个参数，打开该选项之后，原本需要两次AllReduce操作，现在只用一次AllReduce操作。为支持更大粒度的Fuse，Paddle提供了FLAGS_fuse_parameter_memory_size选项，用户可以指定Fuse AllReduce操作之后，每个AllReduce操作的梯度字节数，比如每次AllReduce调用传输128MB的梯度。目前不支持sparse参数梯度.
fuse_relu_depthwise_conv | bool | FALSE | 如果模型中relu和depthwise_conv，并且是连接的，即relu->depthwise_conv，该选项可以将这两个操作合并为一个.
fuse_broadcast_ops | bool | FALSE | 在Reduce模式下，对最后的多个broadcast操作融合为一个.
mkldnn_enabled_op_types | list | {} | 如果是CPU训练，可以用mkldnn_enabled_op_types指明模型中的那些操作可以使用MKLDNN库，如果不进行设置，模型可以使用MKLDNN库的所有操作都会使用MKLDNN库.

##### ExecutionStrategy中的配置参数

选项 | 类型 | 默认值 | 说明
-- | -- | -- | --
num_iteration_per_drop_scope | INT | 1 | 框架在运行过程中会产生一些临时变量，这些变量被放在local   execution scope中。通常每经过一个batch就要清理一下local execution   scope中的变量，但是由于GPU是异步设备，在清理local execution   scope之前需要对所有的GPU调用一次同步操作，因此耗费的时间较长。为此我们在execution_strategy中添加了num_iteration_per_drop_scope选项。用户可以指定经过多少次迭代之后清理一次local   execution scope。
num_threads | INT | 对于CPU：2*dev_count；对于GPU：4*dev_count | ParallelExecutor中根据Op之间的依赖关系确定Op的执行顺序的，即Op的输入都已经变为ready状态之后，该Op会被放到一个队列中，等待被执行。ParallelExecutor内部有一个任务调度线程和一个线程池，任务调度线程从队列中取出所有Ready的Op，并将其放到线程队列中。num_threads表示线程池的大小。注意：线程池不是越大越好。


#### 模型训练

- 对于一些较小的模型，比如mnist等，推荐将exec_strategy.num_threads设为1.

#### 使用CPU进行模型训练
- 如果使用CPU做数据并行训练，需要指定环境变量CPU_NUM，这个环境变量指定程序运行过程中使用的CPUPlace的个数。

- 如果使用CPU进行数据并行训练，并且build_strategy.reduce_strategy=fluid.BuildStrategy.ReduceStrategy.Reduce，所有CPUPlace上的参数是共享的，因此对于一些使用CPU进行数据并行训练的模型，选用Reduce模式可能会更快一些。

