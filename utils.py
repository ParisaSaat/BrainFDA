import numpy as np
import torch


def extract_ampl_phase(fft_im):
    fft_amp = fft_im.real ** 2 + fft_im.imag ** 2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im.imag, fft_im.real)
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    h, w, d = amp_trg.size()
    b = (np.floor(np.amin((h, w, d)) * L)).astype(int)
    amp_src[0:b, 0:b, 0:b] = amp_trg[0:b, 0:b, 0:b]
    amp_src[0:b, w - b:w, 0:b] = amp_trg[0:b, w - b:w, 0:b]
    amp_src[0:b, 0:b, d - b:d] = amp_trg[0:b, 0:b, d - b:d]
    amp_src[0:b, w - b:w, d - b:d] = amp_trg[0:b, w - b:w, d - b:d]
    amp_src[h - b:h, 0:b, 0:b] = amp_trg[h - b:h, 0:b, 0:b]
    amp_src[h - b:h, w - b:w, 0:b] = amp_trg[h - b:h, w - b:w, 0:b]
    amp_src[h - b:h, 0:b, d - b:d] = amp_trg[h - b:h, 0:b, d - b:d]
    amp_src[h - b:h, w - b:w, d - b:d] = amp_trg[h - b:h, w - b:w, d - b:d]
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # get fft of both source and target
    fft_src = torch.fft.rfft(src_img.clone())
    fft_trg = torch.fft.rfft(trg_img.clone())

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.complex(torch.cos(pha_src.clone()) * amp_src_.clone(),
                             torch.sin(pha_src.clone()) * amp_src_.clone())

    # get the recomposed image: source content, target style
    src_in_trg = torch.fft.irfft(fft_src_)
    return src_in_trg
