# EIP_Session3
  
  Session3 - Hit the Wall
##  Base Network Accurarcy 
    
    Accuracy on test data is: 82.44

##  Model 
 
      #Define the model
    model = Sequential()
    drop_out_val=0.25
    learning_rate_val=0.004

    model.add(SeparableConv2D(64, kernel_size=(3, 3), strides=(1,1), padding="same", input_shape=(32, 32, 3), activation='relu',data_format="channels_last")) #Output=32x32x64 |RF=3
    model.add(BatchNormalization())#Output=32x32x64 |RF=3
    #model.add(Dropout(drop_out_val))


    #model.add(AveragePooling2D())
    model.add(SeparableConv2D(64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu',data_format="channels_last")) #Output=32x32x64 |RF=3
    model.add(BatchNormalization())#Output=32x32x64 |RF=3
    model.add(Dropout(drop_out_val))#Output=32x32x64 |RF=3

    model.add(SeparableConv2D(64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu',data_format="channels_last")) #Output=32x32x64 |RF=7
    model.add(BatchNormalization())#Output=32x32x64 |RF=7
    model.add(Dropout(drop_out_val))#Output=32x32x64 |RF=7

    #model.add(SeparableConv2D(64, kernel_size=(3, 3), strides=(2,2),    activation='relu',data_format="channels_last"))
    #model.add(BatchNormalization())
    #model.add(Dropout(drop_out_val))

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(AveragePooling2D())                   #Output=16x16x64 |RF=8

    model.add(SeparableConv2D(128, kernel_size=(3, 3), strides=(1,1), padding="same",  activation='relu',data_format="channels_last")) #Output=16x16x128 |RF=12
    model.add(BatchNormalization())#Output=16x16x128 |RF=12
    model.add(Dropout(drop_out_val))#Output=16x16x128 |RF=12

    model.add(SeparableConv2D(128, kernel_size=(3, 3), strides=(1,1),  padding="same", activation='relu',data_format="channels_last")) #Output=16x16x128 |RF=16
    model.add(BatchNormalization())#Output=16x16x128 |RF=16
    model.add(Dropout(drop_out_val))#Output=16x16x128 |RF=16

    model.add(SeparableConv2D(128, kernel_size=(3, 3), strides=(1,1),  padding="same", activation='relu',data_format="channels_last")) #Output=16x16x128 |RF=20
    model.add(BatchNormalization())#Output=16x16x128 |RF=20
    model.add(Dropout(drop_out_val))#Output=16x16x128 |RF=20


    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(AveragePooling2D())
    model.add(SeparableConv2D(10, kernel_size=(3, 3), strides=(2,2),   activation='relu',data_format="channels_last")) #Output=7x7x10 |RF=24
    model.add(BatchNormalization())#Output=7x7x10 |RF=24
    model.add(Dropout(drop_out_val))#Output=7x7x10 |RF=24


    model.add(AveragePooling2D()) #Output=3x3x10 |RF=28
    model.add(SeparableConv2D(10, kernel_size=(3, 3), strides=(1,1),   activation='relu',data_format="channels_last")) #Output=1x1x10 |RF=44
    #model.add(BatchNormalization())



    #model.add(BatchNormalization())
    #model.add(Dropout(drop_out_val))

    #model.add(AveragePooling2D())
    #model.add(AveragePooling2D())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(10, 1, activation='relu'))
    #model.add(Convolution2D(10, 30, activation='relu'))

    model.add(Flatten()) #Output 10
    model.add(Activation('softmax'))#Output 10
    # Compile the model
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=learning_rate_val), loss='categorical_crossentropy', metrics=['accuracy'])


#   Final Accuracy 


    Epoch 00048: LearningRateScheduler setting learning rate to 0.000274292.
    50000/50000 [==============================] - 13s 259us/step - loss: 0.3522 - acc: 0.8769 - val_loss: 0.4613 - val_acc: 0.8515
# 


##   Epoch Logs
    
    
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      This is separate from the ipykernel package so we can avoid doing imports until
    Train on 50000 samples, validate on 10000 samples
    Epoch 1/50

    Epoch 00001: LearningRateScheduler setting learning rate to 0.004.
    50000/50000 [==============================] - 15s 299us/step - loss: 1.5152 - acc: 0.4472 - val_loss: 1.5122 - val_acc: 0.5209
    Epoch 2/50

    Epoch 00002: LearningRateScheduler setting learning rate to 0.0030557678.
    50000/50000 [==============================] - 13s 255us/step - loss: 1.0020 - acc: 0.6439 - val_loss: 1.0505 - val_acc: 0.6405
    Epoch 3/50

    Epoch 00003: LearningRateScheduler setting learning rate to 0.0024721879.
    50000/50000 [==============================] - 13s 258us/step - loss: 0.8451 - acc: 0.7025 - val_loss: 0.9287 - val_acc: 0.6874
    Epoch 4/50

    Epoch 00004: LearningRateScheduler setting learning rate to 0.0020757654.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.7637 - acc: 0.7350 - val_loss: 0.8040 - val_acc: 0.7280
    Epoch 5/50

    Epoch 00005: LearningRateScheduler setting learning rate to 0.0017889088.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.7007 - acc: 0.7557 - val_loss: 0.6833 - val_acc: 0.7684
    Epoch 6/50

    Epoch 00006: LearningRateScheduler setting learning rate to 0.0015717092.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.6550 - acc: 0.7719 - val_loss: 0.6527 - val_acc: 0.7794
    Epoch 7/50

    Epoch 00007: LearningRateScheduler setting learning rate to 0.0014015417.
    50000/50000 [==============================] - 13s 258us/step - loss: 0.6263 - acc: 0.7809 - val_loss: 0.6478 - val_acc: 0.7789
    Epoch 8/50

    Epoch 00008: LearningRateScheduler setting learning rate to 0.0012646222.
    50000/50000 [==============================] - 13s 253us/step - loss: 0.5947 - acc: 0.7935 - val_loss: 0.5957 - val_acc: 0.7977
    Epoch 9/50

    Epoch 00009: LearningRateScheduler setting learning rate to 0.0011520737.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.5689 - acc: 0.8025 - val_loss: 0.5671 - val_acc: 0.8095
    Epoch 10/50

    Epoch 00010: LearningRateScheduler setting learning rate to 0.0010579212.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.5528 - acc: 0.8059 - val_loss: 0.6113 - val_acc: 0.7970
    Epoch 11/50

    Epoch 00011: LearningRateScheduler setting learning rate to 0.0009779951.
    50000/50000 [==============================] - 13s 260us/step - loss: 0.5370 - acc: 0.8125 - val_loss: 0.5526 - val_acc: 0.8129
    Epoch 12/50

    Epoch 00012: LearningRateScheduler setting learning rate to 0.0009092976.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.5217 - acc: 0.8182 - val_loss: 0.5427 - val_acc: 0.8167
    Epoch 13/50

    Epoch 00013: LearningRateScheduler setting learning rate to 0.0008496177.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.5094 - acc: 0.8219 - val_loss: 0.5235 - val_acc: 0.8261
    Epoch 14/50

    Epoch 00014: LearningRateScheduler setting learning rate to 0.0007972892.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4924 - acc: 0.8278 - val_loss: 0.5249 - val_acc: 0.8276
    Epoch 15/50

    Epoch 00015: LearningRateScheduler setting learning rate to 0.0007510327.
    50000/50000 [==============================] - 13s 257us/step - loss: 0.4846 - acc: 0.8316 - val_loss: 0.6634 - val_acc: 0.7839
    Epoch 16/50

    Epoch 00016: LearningRateScheduler setting learning rate to 0.0007098492.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4763 - acc: 0.8341 - val_loss: 0.5336 - val_acc: 0.8246
    Epoch 17/50

    Epoch 00017: LearningRateScheduler setting learning rate to 0.0006729475.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4632 - acc: 0.8375 - val_loss: 0.5055 - val_acc: 0.8312
    Epoch 18/50

    Epoch 00018: LearningRateScheduler setting learning rate to 0.0006396929.
    50000/50000 [==============================] - 13s 259us/step - loss: 0.4593 - acc: 0.8394 - val_loss: 0.5018 - val_acc: 0.8342
    Epoch 19/50

    Epoch 00019: LearningRateScheduler setting learning rate to 0.0006095703.
    50000/50000 [==============================] - 13s 257us/step - loss: 0.4454 - acc: 0.8444 - val_loss: 0.5025 - val_acc: 0.8324
    Epoch 20/50

    Epoch 00020: LearningRateScheduler setting learning rate to 0.0005821569.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4462 - acc: 0.8436 - val_loss: 0.4988 - val_acc: 0.8352
    Epoch 21/50

    Epoch 00021: LearningRateScheduler setting learning rate to 0.0005571031.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4368 - acc: 0.8456 - val_loss: 0.5196 - val_acc: 0.8322
    Epoch 22/50

    Epoch 00022: LearningRateScheduler setting learning rate to 0.0005341167.
    50000/50000 [==============================] - 13s 259us/step - loss: 0.4327 - acc: 0.8484 - val_loss: 0.4828 - val_acc: 0.8400
    Epoch 23/50

    Epoch 00023: LearningRateScheduler setting learning rate to 0.000512952.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4282 - acc: 0.8497 - val_loss: 0.4957 - val_acc: 0.8382
    Epoch 24/50

    Epoch 00024: LearningRateScheduler setting learning rate to 0.0004934008.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.4179 - acc: 0.8533 - val_loss: 0.4840 - val_acc: 0.8401
    Epoch 25/50

    Epoch 00025: LearningRateScheduler setting learning rate to 0.0004752852.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4140 - acc: 0.8550 - val_loss: 0.4738 - val_acc: 0.8452
    Epoch 26/50

    Epoch 00026: LearningRateScheduler setting learning rate to 0.0004584527.
    50000/50000 [==============================] - 13s 257us/step - loss: 0.4094 - acc: 0.8564 - val_loss: 0.4972 - val_acc: 0.8399
    Epoch 27/50

    Epoch 00027: LearningRateScheduler setting learning rate to 0.0004427718.
    50000/50000 [==============================] - 13s 257us/step - loss: 0.4072 - acc: 0.8588 - val_loss: 0.4766 - val_acc: 0.8444
    Epoch 28/50

    Epoch 00028: LearningRateScheduler setting learning rate to 0.000428128.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4038 - acc: 0.8584 - val_loss: 0.4836 - val_acc: 0.8421
    Epoch 29/50

    Epoch 00029: LearningRateScheduler setting learning rate to 0.0004144219.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.4000 - acc: 0.8590 - val_loss: 0.4805 - val_acc: 0.8433
    Epoch 30/50

    Epoch 00030: LearningRateScheduler setting learning rate to 0.0004015661.
    50000/50000 [==============================] - 13s 258us/step - loss: 0.3922 - acc: 0.8636 - val_loss: 0.4706 - val_acc: 0.8464
    Epoch 31/50

    Epoch 00031: LearningRateScheduler setting learning rate to 0.0003894839.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3905 - acc: 0.8632 - val_loss: 0.4645 - val_acc: 0.8457
    Epoch 32/50

    Epoch 00032: LearningRateScheduler setting learning rate to 0.0004016468.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.3881 - acc: 0.8617 - val_loss: 0.4841 - val_acc: 0.8412
    Epoch 33/50

    Epoch 00033: LearningRateScheduler setting learning rate to 0.0003903201.
    50000/50000 [==============================] - 13s 258us/step - loss: 0.3877 - acc: 0.8633 - val_loss: 0.4765 - val_acc: 0.8472
    Epoch 34/50

    Epoch 00034: LearningRateScheduler setting learning rate to 0.0003796147.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3827 - acc: 0.8656 - val_loss: 0.4775 - val_acc: 0.8456
    Epoch 35/50

    Epoch 00035: LearningRateScheduler setting learning rate to 0.0003694809.
    50000/50000 [==============================] - 13s 260us/step - loss: 0.3834 - acc: 0.8653 - val_loss: 0.4766 - val_acc: 0.8468
    Epoch 36/50

    Epoch 00036: LearningRateScheduler setting learning rate to 0.000359874.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.3809 - acc: 0.8662 - val_loss: 0.4754 - val_acc: 0.8484
    Epoch 37/50

    Epoch 00037: LearningRateScheduler setting learning rate to 0.0003507541.
    50000/50000 [==============================] - 13s 258us/step - loss: 0.3783 - acc: 0.8677 - val_loss: 0.4679 - val_acc: 0.8493
    Epoch 38/50

    Epoch 00038: LearningRateScheduler setting learning rate to 0.000342085.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3714 - acc: 0.8680 - val_loss: 0.5011 - val_acc: 0.8384
    Epoch 39/50

    Epoch 00039: LearningRateScheduler setting learning rate to 0.0003338341.
    50000/50000 [==============================] - 13s 257us/step - loss: 0.3683 - acc: 0.8700 - val_loss: 0.4832 - val_acc: 0.8436
    Epoch 40/50

    Epoch 00040: LearningRateScheduler setting learning rate to 0.0003259718.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.3707 - acc: 0.8693 - val_loss: 0.4655 - val_acc: 0.8480
    Epoch 41/50

    Epoch 00041: LearningRateScheduler setting learning rate to 0.0003184713.
    50000/50000 [==============================] - 13s 259us/step - loss: 0.3638 - acc: 0.8731 - val_loss: 0.4840 - val_acc: 0.8438
    Epoch 42/50

    Epoch 00042: LearningRateScheduler setting learning rate to 0.0003113083.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3599 - acc: 0.8721 - val_loss: 0.4675 - val_acc: 0.8497
    Epoch 43/50

    Epoch 00043: LearningRateScheduler setting learning rate to 0.0003044603.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3624 - acc: 0.8724 - val_loss: 0.4578 - val_acc: 0.8488
    Epoch 44/50

    Epoch 00044: LearningRateScheduler setting learning rate to 0.0002979072.
    50000/50000 [==============================] - 13s 257us/step - loss: 0.3584 - acc: 0.8733 - val_loss: 0.4584 - val_acc: 0.8505
    Epoch 45/50

    Epoch 00045: LearningRateScheduler setting learning rate to 0.0002916302.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3564 - acc: 0.8737 - val_loss: 0.4587 - val_acc: 0.8483
    Epoch 46/50

    Epoch 00046: LearningRateScheduler setting learning rate to 0.0002856123.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3517 - acc: 0.8750 - val_loss: 0.4714 - val_acc: 0.8490
    Epoch 47/50

    Epoch 00047: LearningRateScheduler setting learning rate to 0.0002798377.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3517 - acc: 0.8768 - val_loss: 0.4927 - val_acc: 0.8402
    Epoch 48/50
### Max Accuracy reached here.
    Epoch 00048: LearningRateScheduler setting learning rate to 0.000274292.
    50000/50000 [==============================] - 13s 259us/step - loss: 0.3522 - acc: 0.8769 - val_loss: 0.4613 - val_acc: 0.8515 
    Epoch 49/50

    Epoch 00049: LearningRateScheduler setting learning rate to 0.0002689618.
    50000/50000 [==============================] - 13s 255us/step - loss: 0.3506 - acc: 0.8766 - val_loss: 0.4806 - val_acc: 0.8440
    Epoch 50/50

    Epoch 00050: LearningRateScheduler setting learning rate to 0.0002638348.
    50000/50000 [==============================] - 13s 256us/step - loss: 0.3470 - acc: 0.8775 - val_loss: 0.4809 - val_acc: 0.8441
    This Model took 644.40 seconds to train

    Accuracy on test data is: 84.41
### Additional Information
    Total params: 58,965
    Trainable params: 57,793
    Non-trainable params: 1,172
    This Model took 644.40 seconds to train
    


