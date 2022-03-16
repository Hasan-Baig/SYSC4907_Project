#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==2.8.0 tensorflow-gpu==2.8.0 opencv-python matplotlib')


# In[ ]:


get_ipython().system('pip3 install opencv-python')


# In[ ]:


get_ipython().system('pip3 install matplotlib')


# In[ ]:


get_ipython().system('pip3 install tensorflow ')


# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


get_ipython().system('pip3 install datasets')


# In[330]:


get_ipython().system('pip3 install tf-nightly')


# In[10]:


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

import keras

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.preprocessing.image import ImageDataGenerator


# In[11]:


import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt



# In[12]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
tf.__version__


# In[13]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[14]:


# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# In[15]:


POS_PATH #example to check if the directory for data is set up


# In[16]:


with ZipFile('data.zip', 'r') as zip:
   # Extract all the contents of zip file in current directory
   zip.extractall()


# In[17]:


import uuid


# In[18]:


os.path.join(ANC_PATH, '{}.png'.format(uuid.uuid1()))


# In[688]:


cap = cv2.VideoCapture(0)
while cap.isOpened(): # Check if camera is open
    ret, frame = cap.read()
   
    # Cut down frame to 250x250px
    frame = frame[120:120+1790,200:200+1194, :]
    
    # Collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.png'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.png'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()


# Capture Method 2

# In[11]:


from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.png', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/png', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  filename = os.path.join(POS_PATH, '{}.png'.format(uuid.uuid1()))
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

def take_photo1(filename='photo.png', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/png', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  filename = os.path.join(ANC_PATH, '{}.png'.format(uuid.uuid1()))
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

  


# In[19]:


#with tf.device("cpu:0"): 

anchor = tf.data.Dataset.list_files(ANC_PATH+'*/*.png').take(22)
positive = tf.data.Dataset.list_files(POS_PATH+'*/*.png').take(22)
negative = tf.data.Dataset.list_files(NEG_PATH+'*/*.png').take(22)


# In[20]:


dir_test = anchor.as_numpy_iterator()


# In[21]:


dir_test.next()


# In[22]:


def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


# In[25]:


img = preprocess(b'data/anchor/Photo on 2022-03-15 at 9.01 PM #2.png')


# In[26]:


img.numpy().max() 


# In[27]:


plt.imshow(img)


# In[28]:


#dataset.map(preprocess)


# In[29]:


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# In[30]:


samples = data.as_numpy_iterator()


# In[31]:


example = samples.next()


# In[32]:


example


# In[33]:


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# In[34]:


res = preprocess_twin(*example)


# In[35]:


plt.imshow(res[1])


# In[36]:


res[2]


# In[37]:


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)


# ## Testing anchor image, negative image and label given

# In[38]:


samples = data.as_numpy_iterator()


# In[39]:


samp = samples.next()


# In[40]:


plt.imshow(samp[0])


# In[41]:


plt.imshow(samp[1])


# In[42]:


samp[2]


# In[43]:


# Training partition
train_data = data.take(round(len(data)*.7))
#train_data = tf.data.Dataset.from_generator(gen, tf.int64, tf.TensorShape([None]))
#train_data = train_data.unbatch()
train_data = train_data.batch(10)
train_data = train_data.prefetch(2)


# In[44]:


# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(10)
test_data = test_data.prefetch(2)


# ## Model Engineering - Build Embedding Layer

# In[45]:


inp = Input(shape=(100,100,3), name='input_image')


# In[46]:


inp


# In[47]:


c1 = Conv2D(64, (10,10), activation='relu')(inp)


# In[48]:


m1 = MaxPooling2D(64, (2,2), padding='same')(c1)


# In[49]:


c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)


# In[50]:


c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)


# In[51]:


c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)


# In[52]:


mod = Model(inputs=[inp], outputs=[d1], name='embedding')


# In[53]:


mod.summary()


# In[54]:


def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


# In[55]:


embedding = make_embedding()


# In[56]:


embedding.summary()


# ## L1 Siamese Distance Layer 
# Comparing the anchor and either positive or negative images by subtracting, to find the similarities in the images 

# In[57]:


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Similarity calculation
    def call(self, input_embedding, validation_embedding): 
        return tf.math.abs(input_embedding - validation_embedding) #Anchor - Positive/Negative layer


# In[58]:


#l1 = L1Dist()


# In[59]:


#l1(input_embedding, validation_embedding)


# ## Siamese Model 

# In[60]:


input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))


# In[61]:


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


# In[62]:


siamese_layer = L1Dist()


# In[63]:


distances = siamese_layer(inp_embedding, val_embedding)


# In[64]:


classifier = Dense(1, activation='sigmoid')(distances)


# In[65]:


classifier


# In[66]:


siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[67]:


siamese_network.summary()


# In[68]:


def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[69]:


siamese_model = make_siamese_model()


# In[70]:


siamese_model.summary()


# ## Training

# ## Loss Function and Optimizer

# In[71]:


binary_cross_loss = tf.losses.BinaryCrossentropy()


# In[72]:


opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


# In[73]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# In[ ]:





# In[74]:


test_batch = train_data.as_numpy_iterator()


# In[75]:


batch_1 = test_batch.next()


# In[76]:


batch_1[2]


# In[77]:


X = batch_1[:2]


# In[78]:


y = batch_1[2]


# In[79]:


y


# In[80]:


get_ipython().run_line_magic('pinfo2', 'tf.losses.BinaryCrossentropy')


# In[93]:


@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass 
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


# ## Training Loop

# In[94]:


# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# In[95]:


def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


# In[96]:


#train_data = tf.reshape(train_data, [1])


# ## Training the Model 

# In[97]:


EPOCHS = 50


# In[98]:


train(train_data, EPOCHS)


# ## Measuring how accurate the model is 

# In[99]:


# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall


# ## Making Predictions

# In[117]:


# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[118]:


y_hat = siamese_model.predict([test_input, test_val])


# In[119]:


# Post processing the results 
[1 if prediction > 0.5 else 0 for prediction in y_hat ]


# In[120]:


y_true


# In[123]:


# Creating a metric object 
m = Recall()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()


# In[124]:


# Creating a metric object 
m = Precision()

# Calculating the recall value 
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()


# In[125]:


r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())


# ## Visaulize results from the prediction from the 3rd line. Result is correct label is 0 as the validation image is not one of the allowable gestures

# In[126]:


# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1) # number of rows, number of columns and the index 
plt.imshow(test_input[1]) # control first plot 

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[1]) # control second plot 

# Renders cleanly
plt.show()


# ## Saving the Model 

# In[122]:


# Save the weights in the current directory 
siamese_model.save('siamese_model.h5')


# In[128]:


# Reload model, using custom layer
siamese_model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[129]:


# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])


# In[130]:


# View model summary
siamese_model.summary()


# ## Testing the Model

# In[134]:


from zipfile import ZipFile
with ZipFile('SNN_data.zip', 'r') as zip:
   # Extract all the contents of zip file in current directory
   zip.extractall()


# In[147]:


os.listdir(os.path.join('SNN_data', 'Positive_images'))


# In[184]:


os.path.join('SNN_data', 'Anchor_image', 'input_image.png')


# In[183]:


os.listdir(os.path.join('SNN_data', 'Anchor_image'))


# In[143]:


for image in os.listdir(os.path.join('SNN_data', 'Positive_images')):
    validation_img = os.path.join('SNN_data', 'Postive_images', image)
    print(validation_img) # Need to loop through all of the positive images to make sure to obtain an accurate positive example 


# In[185]:


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('SNN_data', 'Positive_images')): #Listing all images in the directory
        input_img = preprocess(os.path.join('SNN_data', 'Anchor_image', 'input_image.png')) #Parse and load image from the file path, resize and scale image
        validation_img = preprocess(os.path.join('SNN_data', 'Positive_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('SNN_data', 'Positive_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


# In[190]:


cap = cv2.VideoCapture(0) #Camera device number on Mac
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+1790,200:200+1194, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)

#         lim = 255 - 10
#         v[v > lim] = 255
#         v[v <= lim] -= 10
        
#         final_hsv = cv2.merge((h, s, v))
#         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('SNN_data', 'Anchor_image', 'input_image.png'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.9, 0.7) #Adjust detection threshold and verification threshold to obtain a higher accuracy  
        print(verified) #Print the result and verify whether the input image is positive or negative (0, 1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[191]:


np.sum(np.squeeze(results) > 0.9)


# In[ ]:




