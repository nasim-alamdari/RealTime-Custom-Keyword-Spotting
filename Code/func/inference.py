import pyaudio
import numpy as np
import struct
import time


def stream_proc_audio (duration: int, model):
    p = pyaudio.PyAudio()

    def _find_input_device():
        device_index = None            
        for i in range( p.get_device_count() ):     
            devinfo = p.get_device_info_by_index(i)   
            print( "Device %d: %s"%(i,devinfo["name"]) )

            for record_name in ["mic","input"]:
                if record_name in devinfo["name"].lower():
                    print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
                    device_index = i
                    return device_index

        if device_index == None:
            print( "No preferred input found; using default input device." )

        return device_index

    device_index = _find_input_device()


    CHANNELS = 1
    RATE = 16000
    INPUT_BLOCK_TIME = 0.0125
    CHUNK = int(RATE*INPUT_BLOCK_TIME) # FRAMES_PER_BUFFER

    def _eval_stream (KEYWORD, model,frames):
        model_settings = input_data.standard_microspeech_model_settings(label_count=3)

        #===========================================
        # Test the trained FS-KWS model on test sets
        test_spectrograms = input_data.to_micro_spectrogram(model_settings, frames)
        #print(np.asarray(test_spectrograms).shape)
        test_spectrograms = test_spectrograms[np.newaxis, :, :, np.newaxis]
        #print(np.asarray(test_spectrograms).shape)

        # fetch softmax predictions from the finetuned model:
        pred = model.predict(test_spectrograms)
        categorical_pred = np.argmax(pred, axis=1)
        return pred,categorical_pred

    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    input=True,
                    input_device_index = device_index,
                    frames_per_buffer=CHUNK)

    print("Start Streaming...")
    frames = np.array([])
    for i in range(int(duration*RATE/CHUNK)): #go for a LEN seconds
        block = np.fromstring(stream.read(CHUNK),dtype=np.float32)
        frames = np.append(frames, block)
        
        
        if len(frames) == RATE: # 1-sec audio captures
            t = time.time()
            #pred, categorical_pred = kws_train.eval_stream(KEYWORD, model,frames)
            pred, categorical_pred = _eval_stream (KEYWORD, model,frames)
            
            if categorical_pred == 1:
                if pred[0][categorical_pred] >= 0.8:
                    print( "Other words")
            elif categorical_pred == 2:
                if pred[0][categorical_pred] >= 0.8:
                    print( "KEYWORD")
            elif categorical_pred == 0:
                if pred[0][categorical_pred] >= 0.55:
                    print("Background Noise/Silence")
            frames = []
            print("processing time for a chunk:", time.time() - t)



    stream.stop_stream()
    stream.close()
    p.terminate()
    #write(os.path.join(record_save_path,record_name),  sample_rate, wav)
    print("Sreaming Completed")
    return False



#if __name__ == "__main__":
#    duration = 30 # inference for 30 seconds
#    stream_flag = stream_proc_audio(duration, model)