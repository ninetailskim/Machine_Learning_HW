import tensorflow as tf
import os
import utils
import collections
import time
import math
import glob
import numpy as np
import random

#must provided
mode = "train" #train or test
direction = "AtoB"

#deploy parameters
input_dir = "FP/train"
output_dir = "RE_output"
checkpoint = None

#Hyper parameters
learning_rate = 0.001
batch_size = 1
dis_fir_chan = 64
gen_fir_chan = 64
EPS = 1e-12
alpha_L1 = 1000.0
alpha_GAN = 1.0
beta1 = 0.5
epoch_num = 200
#resize the image before cropping 
scale_size = 286
crop_size = 256

#Summary frequcence
summary_freq = 100
progress_freq = 50
trace_freq = 0
display_freq = 0
save_freq = 5000


Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

######################################################
######################################################
######################################################
def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

######################################################
######################################################
######################################################
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def save_images(fetches, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    filesaveds = []

    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        filesaved = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            filesaved[kind] = filename
            out_path = os.path.join(image_dir, filename)
            ################what's this? 
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesaveds.append(filesaved)
    return filesaveds

#################show results
def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

def generator(inputs, out_channels):
    layers = []
    with tf.variable_scope("en_layer1"):
        layer1 = utils.conv(inputs, gen_fir_chan, stride=2)
        layers.append(layer1)

    encoder_channel = [min(gen_fir_chan * 2 ** (i + 1), gen_fir_chan * 8) for i in range(7)]

    for channel in encoder_channel:
        with tf.variable_scope("en_layer%d" % (len(layers) + 1)):
            afterlrelu = utils.lrelu(layers[-1], 0.2)
            afterconv = utils.conv(afterlrelu, channel, stride=2)
            afterbn = utils.batchnorm(afterconv)
            layers.append(afterbn)
    
    chan_drop = [(gen_fir_chan * 8, 0.5), 
                (gen_fir_chan * 8, 0.5), 
                (gen_fir_chan * 8, 0.5),
                (gen_fir_chan * 8, 0.0),
                (gen_fir_chan * 4, 0.0),
                (gen_fir_chan * 2, 0.0),
                (gen_fir_chan, 0.0),
                ]

    num_encoder = len(layers)
    for dest, (out_chan, dropout) in enumerate(chan_drop):
        crops_layer = num_encoder - dest - 1
        with tf.variable_scope("de_layer%d" % (crops_layer + 1)):
            if dest == 0:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[crops_layer]], axis=3)
            afterrelu = tf.nn.relu(input)
            afterdeconv = utils.deconv(afterrelu, out_chan)
            afterbn = utils.batchnorm(afterdeconv)
            
            if dropout > 0.0:
                afterbn = tf.nn.dropout(afterbn, keep_prob=1 - dropout)
            layers.append(afterbn)
    
    with tf.variable_scope("de_layer1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        afterrelu = tf.nn.relu(input)
        afterdeconv = utils.deconv(afterrelu, out_channels)
        aftertanh = tf.tanh(afterdeconv)
        layers.append(aftertanh) 

    return layers[-1]


def model_fn(inputs, targets):
    ############discriminator#################
    def discriminator(dis_images, dis_targets):

        input = tf.concat([dis_images, dis_targets], axis = 3)
        layers = []
        layer_num = 3

        with tf.variable_scope("layer1"):
            afterconv = utils.conv(input, dis_fir_chan, stride=2)
            afterlrelu = utils.lrelu(afterconv, 0.2)
            layers.append(afterlrelu)

        for i in range(layer_num):
            with tf.variable_scope("layer%d" % (len(layers) + 1)):
                out_channels = dis_fir_chan * min(2**(i+1), 8)
                stride = 1 if i == layer_num - 1 else 2  # last layer here has stride 1
                afterconv = utils.conv(layers[-1], out_channels, stride=stride)
                afterbn = utils.batchnorm(afterconv)
                afterlrelu = utils.lrelu(afterbn, 0.2)
                layers.append(afterlrelu)

        with tf.variable_scope("layer%d" % (len(layers) + 1)):
            afterconv = utils.conv(layers[-1], out_channels=1, stride=1)
            output = tf.sigmoid(afterconv)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = generator(inputs, out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = discriminator(inputs, targets)
    
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = discriminator(inputs, outputs)

    with tf.name_scope("Dis_loss"):
        loss_dis = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    
    with tf.name_scope("Gen_loss"):
        loss_GAN_gen = tf.reduce_mean(-tf.log(predict_fake + EPS))
        loss_L1_gen = tf.reduce_mean(tf.abs(targets - outputs))
        loss_gen = loss_GAN_gen * alpha_GAN + loss_L1_gen * alpha_L1

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(learning_rate, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(loss_dis, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(learning_rate, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(loss_gen, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
    

    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    update_loss = ema.apply([loss_dis, loss_GAN_gen, loss_L1_gen])

    global_step = tf.contrib.framework.get_or_create_global_step()
    ####################what's this?
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real = predict_real,
        predict_fake = predict_fake,
        ###ema.average
        discrim_loss = ema.average(loss_dis),
        discrim_grads_and_vars = discrim_grads_and_vars,
        gen_loss_GAN = loss_GAN_gen,
        gen_loss_L1 = loss_L1_gen,
        gen_grads_and_vars = gen_grads_and_vars,
        outputs = outputs,
        train = tf.group(update_loss, incr_global_step, gen_train),
    )

#create z with a fixed seed
seed = random.randint(0, 2 ** 31 - 1)

tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if mode == "TEST":
    if checkpoint is None:
        raise Exception("No Checkpoint")
    '''
    
    '''
    '''
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(key, val)
        # disable these features in test mode
        #a.scale_size = CROP_SIZE
        #a.flip = False
    '''
#####used for 
'''
for k, v in a._get_kwargs():
    print(k, "=", v)

with open(os.path.join(a.output_dir, "options.json"), "w") as f:
    f.write(json.dumps(vars(a), sort_keys=True, indent=4))
'''
input_paths = glob.glob(os.path.join(input_dir, "*.png"))

if len(input_paths) == 0:
    raise Exception("No images in input dir")

input_paths = sorted(input_paths)

with tf.name_scope("pre_input"):
    input_queue = tf.train.string_input_producer(input_paths, shuffle = mode == "train")
    reader = tf.WholeFileReader()
    paths, contents = reader.read(input_queue)
    raw_image = tf.image.decode_png(contents)
    raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(raw_image)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_image = tf.identity(raw_image)
        print("3")


    raw_image.set_shape([None, None, 3])
    width = tf.shape(raw_image)[1]
    print(width)
    #why transfer to [-1, 1]
    ################################################
    #preprocess undefined
    #because origin and GT are concated
    print("split here")
    a_images = preprocess(raw_image[:,:width//2,:])
    b_images = preprocess(raw_image[:,width//2:,:])
    ################################################
    if direction == "AtoB":
        origin, target = [a_images, b_images]
    elif direction == "BtoA":
        origin, target = [b_images, a_images]
    else:
        raise Exception("Wrong direction")
    ################################################
    #Flip, crop, resize function
    if (width > crop_size) == True:
        seed = random.randint(0, 2**31 - 1)
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, width - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        origin = tf.image.crop_to_bounding_box(origin, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        target = tf.image.crop_to_bounding_box(target, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
    else:
        origin = tf.image.resize_images(origin, [crop_size, crop_size], method=tf.image.ResizeMethod.AREA)
        target = tf.image.resize_images(target, [crop_size, crop_size], method=tf.image.ResizeMethod.AREA)
    ################################################
    #transform origin
    #transform target

    path_batch, origin_batch, target_batch = tf.train.batch([paths, origin, target], batch_size = batch_size)

    epoch_size = int(math.ceil(len(input_paths) / batch_size))

    print(len(input_paths))
    ################################################
    '''
    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )
    '''
    ################################################

model = model_fn(origin_batch, target_batch)

inputs = deprocess(origin_batch)
targets = deprocess(target_batch)
outputs = deprocess(model.outputs)

################################################
#convert? why?
conv_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
conv_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
conv_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)
################################################

with tf.name_scope("encode_images"):
    display_fetches = {
        "paths": path_batch,
        "inputs": tf.map_fn(tf.image.encode_png, conv_inputs, dtype=tf.string, name="input_images"),
        "targets": tf.map_fn(tf.image.encode_png, conv_targets, dtype=tf.string, name="target_images"),
        "outputs": tf.map_fn(tf.image.encode_png, conv_outputs, dtype=tf.string, name="input_images"),
    }
    
################################################
#summary image
with tf.name_scope("inputs_summary"):
    tf.summary.image("inputs", origin_batch)

with tf.name_scope("targets_summary"):
    tf.summary.image("targets", target_batch)

with tf.name_scope("outputs_summary"):
    tf.summary.image("outputs", outputs)
#what ? again ? deprocess
with tf.name_scope("real_summary"):
    tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

with tf.name_scope("fake_summary"):
    tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

###################################################
#summmary scalar
tf.summary.scalar("loss_dis", model.discrim_loss)
tf.summary.scalar("loss_GAN_gen", model.gen_loss_GAN)
tf.summary.scalar("loss_L1_gen", model.gen_loss_L1)

###################################################
#summary histogram
for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

#for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
#    tf.summary.histogram(var.op.name + "/gradients", grad)

with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep=1)

if trace_freq > 0 or summary_freq > 0:
    logdir = output_dir 
else:
    logdir = None

sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, saver=None)
with sv.managed_session() as sess:

    if checkpoint is not None:
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        saver.restore(sess, checkpoint)

    max_steps = 2 ** 32
    max_steps = epoch_size * epoch_num

    if mode == "test":
        max_steps = min(epoch_size, max_steps)
        for step in range(max_steps):
            results = sess.run(display_fetches)
            ###################################
            #save_images
            filesave = save_images(results)
            for i, f in enumerate(filesave):
                print("Test", f["name"])
            ###################################
            #append_index
            index_path = append_index(filesave)
        ################
        print("???????")

    else:
        start = time.time()
        ############train########################
        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            #############trace_part################
            #############important#################
            options = None
            run_metadata = None
            if should(trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.discrim_loss
                fetches["gen_loss_L1"] = model.discrim_loss

            if should(summary_freq):
                fetches["summary"] = sv.summary_op
            
            if should(display_freq):
                fetches["display"] = display_fetches

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            if should(summary_freq):
                sv.summary_writer.add_summary(results["summary"], results["global_step"])
            
            if should(display_freq):
                filesave = save_images(results["display"], step=results["global_step"])
                append_index(filesave, step=True)
            
            if should(trace_freq):
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(progress_freq):
                train_epoch = math.ceil(results["global_step"] / epoch_size)
                train_step = (results["global_step"] - 1) % epoch_size + 1
                rate = (step + 1) * batch_size / (time.time() - start)
                remaining = (max_steps - step) * batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])

            if should(save_freq):
                print("saving model")
                saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break;