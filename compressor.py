"""
compressor.py: Contains functions to compress and decompress WAV audio 
using FFT + top-K selection + quantization + Huffman coding.
"""

import numpy as np
import scipy.io.wavfile as wavfile
from huffman import huffman_encode, huffman_decode

def compress_audio(infile, outfile, K):
    """
    Compress the input WAV file and write to a custom binary format (outfile).
    Steps:
    - Load WAV (fs, data).
    - For each channel: do rFFT, select top-K by magnitude, quantize, Huffman encode.
    - Save compressed data (fs, length, channels, K, and channel data) to outfile.
    """
    # Read WAV file
    fs, data = wavfile.read(infile)
    # Handle mono vs stereo: ensure data has shape (N, channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, channels = data.shape

    comp_result = {
        'fs': fs,
        'length': n,
        'channels': channels,
        'K': K,
        'channels_data': []
    }

    for ch in range(channels):
        # Convert samples to float for FFT
        samples = data[:, ch].astype(float)
        coeffs = np.fft.rfft(samples)             # FFT to frequency domain
        mags = np.abs(coeffs)                    # Magnitudes

        # Determine actual number of coefficients to keep
        K_actual = min(K, len(mags))
        # Find indices of the top-K largest magnitudes
        idx_top = np.argpartition(mags, -K_actual)[-K_actual:]
        # Sort top indices by descending magnitude (optional for consistency)
        idx_top = idx_top[np.argsort(-mags[idx_top])]
        # Then sort indices in ascending order for storage
        idx_sorted = np.sort(idx_top)

        # Extract the top-K complex coefficients
        top_coeffs = coeffs[idx_sorted]
        # Quantize: round real and imaginary parts to integers
        reals = np.round(np.real(top_coeffs)).astype(int)
        imags = np.round(np.imag(top_coeffs)).astype(int)

        # Huffman encode the real and imaginary parts separately
        codes_real, enc_real = huffman_encode(reals.tolist())
        codes_imag, enc_imag = huffman_encode(imags.tolist())

        # Store this channel's compressed data
        ch_data = {
            'indices': idx_sorted,        # indices of retained FFT bins
            'codes_real': codes_real,     # Huffman code map for real parts
            'codes_imag': codes_imag,     # Huffman code map for imag parts
            'encoded_real': enc_real,     # Encoded bitstring (real parts)
            'encoded_imag': enc_imag      # Encoded bitstring (imag parts)
        }
        comp_result['channels_data'].append(ch_data)

    # Write compressed data to file (using pickle for simplicity)
    import pickle
    with open(outfile, 'wb') as f:
        pickle.dump(comp_result, f)
    print(f"Compressed and wrote output to '{outfile}'")

def decompress_audio(infile, outfile):
    """
    Decompress the custom file and reconstruct the WAV file (outfile).
    Steps:
    - Load compressed data (fs, length, channels, K, channel data).
    - For each channel: decode Huffman to get quantized values, reconstruct FFT array, ifft.
    - Write reconstructed audio to WAV.
    """
    import pickle
    comp_data = pickle.load(open(infile, 'rb'))
    fs = comp_data['fs']
    n = comp_data['length']
    channels = comp_data['channels']

    # Prepare output array
    reconstructed = np.zeros((n, channels))

    for ch in range(channels):
        ch_data = comp_data['channels_data'][ch]
        indices = ch_data['indices']
        codes_real = ch_data['codes_real']
        codes_imag = ch_data['codes_imag']
        enc_real = ch_data['encoded_real']
        enc_imag = ch_data['encoded_imag']

        # Decode the Huffman bitstrings
        reals = np.array(huffman_decode(enc_real, codes_real), dtype=float)
        imags = np.array(huffman_decode(enc_imag, codes_imag), dtype=float)

        # Reconstruct the frequency-domain array (complex), setting retained bins
        freq_arr = np.zeros(n//2 + 1, dtype=complex)
        freq_arr[indices] = reals + 1j * imags

        # Inverse FFT to time domain
        time_recon = np.fft.irfft(freq_arr, n=n)
        reconstructed[:, ch] = time_recon

    # Cast to original dtype (16-bit PCM) with rounding/clipping
    reconstructed = np.round(reconstructed).astype(np.int16)
    wavfile.write(outfile, fs, reconstructed)
    print(f"Decompressed and wrote WAV to '{outfile}'")
