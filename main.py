import nemo.collections.asr as nemo_asr


asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from("/home/ilhom/Documents/stt_m_nv/stt_uz.nemo")

def stt_nv(audio_path):
    
    generate = asr_model.transcribe([audio_path])
    return generate





# Audio Path
audio = "/home/ilhom/Documents/stt_m_nv/audio_2024-11-17_20-42-53.ogg"
gen = stt_nv(audio)

# Natijani chiqarish
print(gen[0][0])