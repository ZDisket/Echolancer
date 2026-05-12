"""
Gradio app for Echolancer Zero-Shot TTS.
"""
import os
import glob
import torch
torch.backends.cudnn.benchmark = False
import torchaudio
import gradio as gr
from speechbrain.inference.speaker import EncoderClassifier
from echolancerfe import EcholancerFE
from neucodecfe import NeuCodecFE
from brontes_inference import BrontesInference

# ============== Configuration ==============
MODEL_CONFIG_PATH = "config/model_stage3_zs.yaml"
CHECKPOINT_PATH = "./checkpoints/checkpoint_step_127500_model_only.pt"
BRONTES_MODEL_PATH = "./checkpoints/general_cpu.pt"
SPEAKERS_DIR = "speakers"
SAMPLE_RATE = 24000
BRONTES_SAMPLE_RATE = 48000
USE_REFINEMENT = True

# ============== Load Models ==============
print("Loading models...")

echolancer = EcholancerFE(model_config_path=MODEL_CONFIG_PATH)
echolancer.load_checkpoint(CHECKPOINT_PATH)

neu_codec = NeuCodecFE(is_cuda=True, offset=echolancer.get_vocab_offset())

speaker_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

brontes = None
if USE_REFINEMENT:
    brontes = BrontesInference(
        model_path=BRONTES_MODEL_PATH,
        device='cpu',
        sample_rate=BRONTES_SAMPLE_RATE,
    )

print("Models loaded successfully!")


# ============== Helper Functions ==============
def get_speaker_list():
    """Get list of speaker files from the speakers directory."""
    if not os.path.exists(SPEAKERS_DIR):
        os.makedirs(SPEAKERS_DIR)
        return []
    
    wav_files = glob.glob(os.path.join(SPEAKERS_DIR, "*.wav"))
    # Return just the basenames without extension
    return [os.path.splitext(os.path.basename(f))[0] for f in sorted(wav_files)]


def extract_speaker_embedding(audio_path):
    """Extract speaker embedding from an audio file."""
    signal, fs = torchaudio.load(audio_path)
    
    # Force mono
    signal = signal.mean(dim=0, keepdim=True)
    
    # Force 16 kHz (required by ECAPA-TDNN)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
    
    # Extract embedding
    embedding = speaker_encoder.encode_batch(signal).squeeze(0)
    return embedding


def tts_inference(text, speaker_embedding, temperature, top_p, max_length, use_refinement):
    """Perform TTS inference and return audio."""
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        codes = echolancer.infer(
            text=text,
            speaker_id=speaker_embedding,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
        )
        
        # Decode tokens to waveform
        codes_for_codec = codes.unsqueeze(1)
        waveform = neu_codec.decode_codes(codes_for_codec)
        waveform = waveform[0, 0, :].cpu()
    
    # Apply Brontes refinement if enabled
    if use_refinement and brontes is not None:
        refined = brontes.process(waveform, input_sr=SAMPLE_RATE)
        refined_waveform = refined.squeeze().cpu().numpy()
        return (BRONTES_SAMPLE_RATE, refined_waveform)
    else:
        return (SAMPLE_RATE, waveform.numpy())


def generate_speech(
    text,
    speaker_source,
    preset_speaker,
    uploaded_audio,
    recorded_audio,
    temperature,
    top_p,
    max_length,
    use_refinement
):
    """Main generation function for Gradio."""
    if not text.strip():
        return None, "Please enter some text to synthesize."
    
    # Determine which audio source to use for speaker embedding
    audio_path = None
    
    if speaker_source == "Preset Speaker":
        if not preset_speaker:
            return None, "Please select a speaker from the dropdown."
        audio_path = os.path.join(SPEAKERS_DIR, f"{preset_speaker}.wav")
        if not os.path.exists(audio_path):
            return None, f"Speaker file not found: {audio_path}"
    
    elif speaker_source == "Upload Audio":
        if uploaded_audio is None:
            return None, "Please upload a reference audio file."
        audio_path = uploaded_audio
    
    elif speaker_source == "Record Audio":
        if recorded_audio is None:
            return None, "Please record a reference audio."
        audio_path = recorded_audio
    
    else:
        return None, "Please select a speaker source."
    
    try:
        # Extract speaker embedding
        speaker_embedding = extract_speaker_embedding(audio_path)
        
        # Generate speech
        audio_output = tts_inference(
            text=text,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            top_p=top_p,
            max_length=int(max_length),
            use_refinement=use_refinement
        )
        
        return audio_output, "Generation successful!"
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def update_speaker_source(source):
    """Update visibility of input components based on speaker source selection."""
    return (
        gr.update(visible=(source == "Preset Speaker")),
        gr.update(visible=(source == "Upload Audio")),
        gr.update(visible=(source == "Record Audio")),
    )


# ============== Gradio Interface ==============
def create_app():
    speaker_list = get_speaker_list()
    
    with gr.Blocks(title="Echolancer Zero-Shot TTS", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎤 Echolancer Zero-Shot TTS
            Generate speech in any voice using a reference audio sample.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Speaker Reference")
                
                speaker_source = gr.Radio(
                    choices=["Preset Speaker", "Upload Audio", "Record Audio"],
                    value="Preset Speaker",
                    label="Speaker Source"
                )
                
                preset_dropdown = gr.Dropdown(
                    choices=speaker_list,
                    label="Select Preset Speaker",
                    visible=True
                )
                
                upload_audio = gr.Audio(
                    label="Upload Reference Audio",
                    type="filepath",
                    visible=False
                )
                
                record_audio = gr.Audio(
                    label="Record Reference Audio",
                    sources=["microphone"],
                    type="filepath",
                    visible=False
                )
                
                # Update visibility based on speaker source
                speaker_source.change(
                    fn=update_speaker_source,
                    inputs=[speaker_source],
                    outputs=[preset_dropdown, upload_audio, record_audio]
                )
                
                gr.Markdown("### Generation Settings")
                
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.05,
                    label="Temperature",
                    info="Higher = more variation"
                )
                
                top_p = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.92, step=0.01,
                    label="Top-p (Nucleus Sampling)"
                )
                
                max_length = gr.Slider(
                    minimum=256, maximum=2048, value=1024, step=64,
                    label="Max Length (tokens)"
                )
                
                use_refinement = gr.Checkbox(
                    value=USE_REFINEMENT,
                    label="Apply Brontes Refinement",
                    info="Enhances audio quality (48kHz output)"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Text Input")
                
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5
                )
                
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                
                gr.Markdown("### Output")
                
                output_audio = gr.Audio(label="Generated Speech", type="numpy")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                speaker_source,
                preset_dropdown,
                upload_audio,
                record_audio,
                temperature,
                top_p,
                max_length,
                use_refinement
            ],
            outputs=[output_audio, status_text]
        )
        
        # Examples
        if speaker_list:
            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    ["I can't believe it's already Friday! This week just flew by so fast.", speaker_list[0] if speaker_list else None],
                    ["Would you like to grab some coffee later? I know this great place downtown.", speaker_list[0] if speaker_list else None],
                    ["The weather forecast says it might rain tomorrow, so don't forget your umbrella.", speaker_list[0] if speaker_list else None],
                    ["Thanks for helping me out with that project yesterday. I really appreciate it!", speaker_list[0] if speaker_list else None],
                ],
                inputs=[text_input, preset_dropdown],
            )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
