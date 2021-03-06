# Project Write-Up

There are various models present at [Model_Detection_Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

I am going to use **Faster_rcnn_inception_v2_coco_2018_01_28** model in my project.

* Download the model from here 

```
wget
http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

* Extracting the tar.gz file by the following command:

```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
* Changing the directory to the extracted folder of the downloaded model:

```
cd faster_rcnn_inception_v2_coco_2018_01_28
```
* The model can't be the existing models provided by Intel. So, converting the TensorFlow model to Intermediate Representation (IR) or OpenVINO IR format. The command used is given below:

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
## Explaining Custom Layers

* The process behind converting custom layers depends on frameworks we use either it is tensorflow, caffee or kaldi. For this project Tensorflow was used. [Details](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)

In both TensorFlow and Caffe :
First,register the custom layers as extensions to the Model Optimizer.

* For Caffe :

Second, register the layers as Custom, then use Caffe to calculate the output shape of the layer.
You’ll need Caffe on your system to do this option.

* For TensorFlow :

Second,replace the unsupported subgraph with a different subgraph.
The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

Some of the potential reasons for handling custom layers are :

* If any Layer which is not present in list of Supported Layer of any particular model Framework,The Model Optimizer automatically classifies it as Custom Layer.
* Custom Layers are mainly used for handling UnSupported Layers.
* Some of the potential reasons for handling custom layers is to optimize our pre-trained models and convert them to a intermediate representation without a lot of loss of accuracy and of course shrink and speed up the Performance.
* OpenVino toolkit use model optimizer to reduce the size of the model. Model optimizer searches for the list of known layers for each layer in the model. The inference engine loads the layers from the model IR into the specified device plugin, which will search a list of known layer implementations for the device. If your model architecure contains layer or layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. To use the model having the unsupported layer we need to use the custom layer feature of OpenVino Toolkit.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Ssd_inception_v2_coco]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - Converted the model to intermediate representation using the following command. Further, this model lacked accuracy as it didn't detect people correctly in the video. 
  - The model was insufficient for the app because when i tested it failed on intervals and it didn't found the bounding boxes around the person and for next person.
  - I converted the model to an Intermediate Representation with the following arguments :
    - ```tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz```
    - ```cd ssd_inception_v2_coco_2018_01_28```
    - ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
- Model 2: [Faster_rcnn_inception_v2_coco_2018_01_28]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - Converted the model to intermediate representation using the following command. it performed really well in the output video. The model works better than the previous approaches.
  - After managing the shape attribute it worked quite well.
  - I converted the model to an Intermediate Representation with the following arguments :
    - ```tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz```
    - ```cd faster_rcnn_inception_v2_coco_2018_01_28.tar.gz```
    - ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json```
  
- Model 3: [person-detection-retail-0013]
  -[https://docs.openvinotoolkit.org/2019_R3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html]
  - So eagerly wanted to try this out, but as i started late, i was behind my project submission.
  - Will be trying it after the project submission deadline ends and provide my findings.
  
## Comparing Model Performance

Comparing the two models i.e. ssd_inception_v2_coco and faster_rcnn_inception_v2_coco in terms of latency and memory, several insights were drawn. It could be clearly seen that the Latency (microseconds) and Memory (Mb) decreases in case of OpenVINO as compared to plain Tensorflow model which is very useful in case of OpenVINO applications.

| Model/Framework                             | Latency (microseconds)            | Memory (Mb) |
| -----------------------------------         |:---------------------------------:| -------:|
| ssd_inception_v2_coco (plain TF)            | 229                               | 538    |
| ssd_inception_v2_coco (OpenVINO)            | 150                               | 329    |
| faster_rcnn_inception_v2_coco (plain TF)    | 1279                              | 562    |
| faster_rcnn_inception_v2_coco (OpenVINO)    | 891                              | 281    |


The difference between model accuracy pre- and post-conversion was :
* The model accuracy before convertion is slightly higher than the model which is converted into IR but it totally overrules the faster speed given for detection with respect to the bulky models.
* The size of the model pre- and post-conversion was :
  * The size of the model after converting into IR was smaller than the model before convertion.
When a model is converted into IR It'll convert into .xml file which contains model architecture and .bin file which contains weights and baises.
* The average inference time of the model pre- and post-conversion was :
  * The average inference time of the model after converting to IR is shorter than the model before converting.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

* It helps to improve in-store operations and individual monitoring for certain tasks.

* Customer Traffic inside warehouses could help in mitigating risk factors. Also it can help to provide visitor analytics.

* Each of these use cases would be useful in a any Shopping Center, Retail Chain, Library, Bank etc. Counting data will help you make well-informed decisions about your business and your data analytics team to make informed decision.

* Controlling the number of people present in a particular area. Further, with some updations, this could also prove helpful in the current COVID-19 scenario i.e. to keep a check on the number of people in the frame.

* Also for COVID scenario, with some modifications it can also help in managing safe distance between individuals.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
* Determining various models and their accuracy based on the frame size.
* Better be the model accuracy more are the chances to obtain the desired results through an app deployed at edge.
* Focal length/image also have a effect as better be the pixel quality of image or better the camera focal length,more clear results we will obtain.
