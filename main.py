"""
main.py: Command-line interface for the FFT+Huffman audio compressor.
Usage:
    python main.py compress <input.wav> <output.bin> --K <K>
    python main.py decompress <input.bin> <output.wav>
"""
import argparse
from compressor import compress_audio, decompress_audio

def main():
    parser = argparse.ArgumentParser(description="FFT + Huffman Audio Compressor")
    sub = parser.add_subparsers(dest='mode', required=True)

    # Compress command
    p_comp = sub.add_parser('compress', help='Compress a WAV file')
    p_comp.add_argument('input', help='Input WAV file')
    p_comp.add_argument('output', help='Output compressed file')
    p_comp.add_argument('--K', type=int, default=1000,
                        help='Number of top FFT components to retain (default: 1000)')

    # Decompress command
    p_decomp = sub.add_parser('decompress', help='Decompress to WAV')
    p_decomp.add_argument('input', help='Input compressed file')
    p_decomp.add_argument('output', help='Output WAV file')

    args = parser.parse_args()

    if args.mode == 'compress':
        compress_audio(args.input, args.output, args.K)
    elif args.mode == 'decompress':
        decompress_audio(args.input, args.output)

if __name__ == "__main__":
    main()
