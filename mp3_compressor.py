"""
mp3_compressor.py: Contains functions to compress and decompress MP3 audio 
using FFT + top-K selection + quantization + Huffman coding.
"""

import numpy as np
from pydub import AudioSegment
from huffman import huffman_encode, huffman_decode

def compress_audio(infile, outfile, K):
    """
    Compress the input MP3 file and write to a custom binary format (outfile).
    """
    # Read MP3 file
    audio = AudioSegment.from_mp3(infile)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    channels = audio.channels
    fs = audio.frame_rate
    
    # Reshape for stereo
    if channels == 2:
        samples = samples.reshape(-1, 2)
    else:
        samples = samples.reshape(-1, 1)
        
    n, channels = samples.shape

    comp_result = {
        'fs': fs,
        'length': n,
        'channels': channels,
        'K': K,
        'channels_data': [],
        'sample_width': audio.sample_width,
        'frame_rate': audio.frame_rate,
        'frame_width': audio.frame_width
    }

    for ch in range(channels):
        # Convert samples to float for FFT
        channel_samples = samples[:, ch].astype(float)
        coeffs = np.fft.rfft(channel_samples)
        mags = np.abs(coeffs)

        # Determine actual number of coefficients to keep
        K_actual = min(K, len(mags))
        idx_top = np.argpartition(mags, -K_actual)[-K_actual:]
        idx_top = idx_top[np.argsort(-mags[idx_top])]
        idx_sorted = np.sort(idx_top)

        # Extract and quantize coefficients
        top_coeffs = coeffs[idx_sorted]
        reals = np.round(np.real(top_coeffs)).astype(int)
        imags = np.round(np.imag(top_coeffs)).astype(int)

        # Huffman encode
        codes_real, enc_real = huffman_encode(reals.tolist())
        codes_imag, enc_imag = huffman_encode(imags.tolist())

        ch_data = {
            'indices': idx_sorted,
            'codes_real': codes_real,
            'codes_imag': codes_imag,
            'encoded_real': enc_real,
            'encoded_imag': enc_imag
        }
        comp_result['channels_data'].append(ch_data)

    # Save compressed data
    import pickle
    with open(outfile, 'wb') as f:
        pickle.dump(comp_result, f)
    print(f"Compressed and wrote output to '{outfile}'")

def decompress_audio(infile, outfile):
    """
    Decompress the custom file and reconstruct the MP3 file.
    """
    import pickle
    comp_data = pickle.load(open(infile, 'rb'))
    
    n = comp_data['length']
    channels = comp_data['channels']
    fs = comp_data['fs']
    
    # Prepare output array
    reconstructed = np.zeros((n, channels))

    for ch in range(channels):
        ch_data = comp_data['channels_data'][ch]
        
        # Decode Huffman and reconstruct coefficients
        reals = np.array(huffman_decode(ch_data['encoded_real'], 
                                      ch_data['codes_real']), dtype=float)
        imags = np.array(huffman_decode(ch_data['encoded_imag'], 
                                      ch_data['codes_imag']), dtype=float)

        # Reconstruct frequency domain
        freq_arr = np.zeros(n//2 + 1, dtype=complex)
        freq_arr[ch_data['indices']] = reals + 1j * imags

        # IFFT to time domain
        time_recon = np.fft.irfft(freq_arr, n=n)
        reconstructed[:, ch] = time_recon

    # Convert to audio segment
    reconstructed = np.round(reconstructed).astype(np.int16)
    if channels == 1:
        reconstructed = reconstructed.flatten()
        
    audio = AudioSegment(
        reconstructed.tobytes(),
        frame_rate=fs,
        sample_width=comp_data['sample_width'],
        channels=channels
    )
    
    # Export as MP3
    audio.export(outfile, format="mp3")
    print(f"Decompressed and wrote MP3 to '{outfile}'")