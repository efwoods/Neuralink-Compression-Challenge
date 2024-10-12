#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// WAV file header structure
struct Twavheader
{
    char chunk_ID[4];
    uint32_t chunk_size;
    char format[4];

    char sub_chunk1_ID[4];
    uint32_t sub_chunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;

    char sub_chunk2_ID[4];
    uint32_t sub_chunk2_size;
};

void read_wav_file(std::string fname){
    // Open the WAV file
    std::ifstream wavfile(fname, std::ios::binary);

    if(wavfile.is_open())
    {
        // Read the WAV header
        Twavheader wav;
        wavfile.read(reinterpret_cast<char*>(&wav), sizeof(Twavheader));

        // If the file is a valid WAV file
        if(std::string(wav.format, 4) != "WAVE" || std::string(wav.chunk_ID, 4) != "RIFF")
        {
            wavfile.close();
            std::cerr << "Not a WAVE or RIFF!" << std::endl;
            return;
        }

        // Properties of a WAV file
        std::cout << "Filename:" << fname << std::endl;
        std::cout << "File size:" << wav.chunk_size+8 << std::endl;
        std::cout << "Resource Exchange File Mark:" << std::string(wav.chunk_ID, 4) << std::endl;
        std::cout << "Format:" << std::string(wav.format, 4) << std::endl;
        std::cout << "Channels: " << wav.num_channels << std::endl;
        std::cout << "Sample Rate: " << wav.sample_rate << " Hz" << std::endl;
        std::cout << "Bits Per Sample: " << wav.bits_per_sample << " bits" << std::endl;

        // Read wave data
        std::vector<int16_t> audio_data( wav.sub_chunk2_size / sizeof(int16_t));
        wavfile.read(reinterpret_cast<char*>(audio_data.data()), wav.sub_chunk2_size);
        wavfile.close(); // Close audio file

        // Display audio samples
        const size_t numofsample = 20;
        std::cout << "Listen first " << numofsample << " Samples:" << std::endl;
        for (size_t i = 0; i < numofsample && i < audio_data.size(); ++i){
            std::cout << i << ":" << audio_data[i] << std::endl;
        }
        std::cout << std::endl;
    }
}

int main(void){
    read_wav_file("../../data/0aefe960-43fd-41cc-97c8-bf9d2d64efd3.wav");

    system("pause");
    return 0;
}