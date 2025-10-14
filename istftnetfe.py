import os, json
from tstft import TorchSTFT
import torch
MAX_WAV_VALUE = 32768.0

class ISTFTNetFE(torch.nn.Module):
    def __init__(self, gen, stft):
        super(ISTFTNetFE, self).__init__()
        self.gen = gen
        self.stft = stft
    
    def infer(self, x):
        wav_predictions = self(x)
        audio = wav_predictions.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        return audio
        
    def infer_cpuistft(self, x):
        self.stft = self.stft.cpu()
        self.stft.window = self.stft.window.cpu()
        with torch.no_grad():
            spec, phase = self.gen(x)
            spec, phase = spec.cpu(), phase.cpu()
            y_g_hat = self.stft.inverse(spec, phase)

        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        return audio
    
    def forward(self, x):
        with torch.no_grad():
            spec, phase = self.gen(x)
            y_g_hat = self.stft.inverse(spec, phase)
            
        return y_g_hat
        
    def export_ts(self, out_foldername, sampling_rate, ex_devices = ["cuda", "cpu"]):
        dummy_mel = torch.randn((1, 88, 600)) # create dummy mel input
        for dev in ex_devices:
            self.stft = self.stft.to(dev)
            self.stft.window = self.stft.window.to(dev)

            self.gen = self.gen.to(dev)

            raw_fn = f"istft_{dev}.pt"
            full_fn = os.path.join(out_foldername, raw_fn)
            os.makedirs(out_foldername, exist_ok = True)
            
            traced_istft = torch.jit.trace_module(self.gen, {"forward": (dummy_mel.to(dev))}, check_trace=False, strict=True)
            torch.jit.save(traced_istft, full_fn)

        # export istft and other info
        configs_dict = {
            "gen_istft_n_fft" : self.stft.filter_length,
            "gen_istft_hop_size" : self.stft.hop_length,
            "gen_istft_n_fft" : self.stft.win_length,
            "sampling_rate" : sampling_rate
        }
        out_jsonfn = os.path.join(out_foldername, "config.json")
        with open(out_jsonfn, "w") as outfile:
            json.dump(configs_dict, outfile)

    def load_ts(self, in_foldername, in_dev = "cuda"):
        self.gen = torch.jit.load(
            os.path.join(in_foldername, f"istft_{in_dev}.pt")
        )
        json_fn = os.path.join(in_foldername, "config.json")
        
        with open(json_fn) as json_file:
            json_dat = json.load(json_file)

        self.stft = TorchSTFT(filter_length=json_dat["gen_istft_n_fft"],
                              hop_length=json_dat["gen_istft_hop_size"],
                              win_length= json_dat["gen_istft_n_fft"]).to(in_dev)
        
        self.stft.window = self.stft.window.to(in_dev)
        self.sampling_rate = json_dat["sampling_rate"]
        