//
// Created for NeuralNote BPM Detection
//

#ifndef BpmDetector_h
#define BpmDetector_h

#include <JuceHeader.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class BpmDetector
{
public:
    /**
     * Estimates BPM of audio buffer.
     * @param inAudio Mono audio buffer
     * @param inSampleRate Sample rate of inAudio
     * @param minBpm Minimum BPM to consider (default: 60.0)
     * @param maxBpm Maximum BPM to consider (default: 200.0)
     * @return Estimated BPM (rounded to 1 decimal place), or 0.0 if estimation fails
     */
    static double detectBpm(const juce::AudioBuffer<float>& inAudio,
                            double inSampleRate,
                            double minBpm = 60.0,
                            double maxBpm = 200.0)
    {
        if (inAudio.getNumChannels() == 0 || inAudio.getNumSamples() < static_cast<int>(inSampleRate * 1.5)) {
            return 0.0;
        }

        const float* readPtr = inAudio.getReadPointer(0);
        const int numSamples = inAudio.getNumSamples();

        // 1. Calculate energy envelope using a sliding window
        // Hop size ~ 10ms, Window size ~ 20ms
        const int hopSize = std::max(1, static_cast<int>(inSampleRate * 0.010)); // 10 ms
        const int winSize = std::max(1, static_cast<int>(inSampleRate * 0.020)); // 20 ms
        const double envelopeSampleRate = inSampleRate / hopSize;

        const int numHops = (numSamples - winSize) / hopSize;
        if (numHops <= 0) {
            return 0.0;
        }

        std::vector<float> energy(numHops, 0.0f);
        for (int i = 0; i < numHops; ++i) {
            const int start = i * hopSize;
            float sumSq = 0.0f;
            for (int j = 0; j < winSize; ++j) {
                const float s = readPtr[start + j];
                sumSq += s * s;
            }
            energy[i] = std::sqrt(sumSq / winSize);
        }

        // 2. Compute onset novelty curve (half-wave rectified first difference)
        std::vector<float> onset(numHops, 0.0f);
        for (int i = 1; i < numHops; ++i) {
            const float diff = energy[i] - energy[i - 1];
            onset[i] = (diff > 0.0f) ? diff : 0.0f;
        }

        // Subtract local average (adaptive thresholding) to accentuate onset spikes
        const int localWin = static_cast<int>(envelopeSampleRate * 0.2); // 200 ms
        std::vector<float> filteredOnset(numHops, 0.0f);
        float runningSum = 0.0f;
        int count = 0;

        for (int i = 0; i < numHops; ++i) {
            runningSum += onset[i];
            count++;
            if (i >= localWin) {
                runningSum -= onset[i - localWin];
                count--;
            }
            const float mean = (count > 0) ? (runningSum / count) : 0.0f;
            filteredOnset[i] = std::max(0.0f, onset[i] - mean);
        }

        // 3. Autocorrelation of onset novelty curve over lag range corresponding to minBpm..maxBpm
        const int minLag = std::max(1, static_cast<int>(envelopeSampleRate * 60.0 / maxBpm));
        const int maxLag = std::max(minLag + 1, static_cast<int>(envelopeSampleRate * 60.0 / minBpm));

        if (maxLag >= numHops) {
            return 0.0;
        }

        // Limit search to first 45 seconds of onset to keep computation fast and accurate
        const int maxAnalysisHops = std::min(numHops, static_cast<int>(envelopeSampleRate * 45.0));
        const int acfLength = maxAnalysisHops - maxLag;

        if (acfLength <= 0) {
            return 0.0;
        }

        int bestLag = -1;
        float bestCorr = -1.0f;

        for (int lag = minLag; lag <= maxLag; ++lag) {
            float corr = 0.0f;
            for (int i = 0; i < acfLength; ++i) {
                corr += filteredOnset[i] * filteredOnset[i + lag];
            }

            // Slight weighting towards typical pop/electronic tempo center (~120 BPM)
            const double candidateBpm = (60.0 * envelopeSampleRate) / lag;
            const double distance = std::log2(candidateBpm / 120.0);
            const float centerWeight = static_cast<float>(std::exp(-0.5 * distance * distance));

            const float weightedCorr = corr * (0.8f + 0.2f * centerWeight);

            if (weightedCorr > bestCorr) {
                bestCorr = weightedCorr;
                bestLag = lag;
            }
        }

        if (bestLag <= 0) {
            return 0.0;
        }

        double detectedBpm = (60.0 * envelopeSampleRate) / bestLag;

        // Check if double-tempo or half-tempo is much stronger
        const int halfLag = bestLag / 2;
        if (halfLag >= minLag) {
            float corrHalf = 0.0f;
            for (int i = 0; i < acfLength; ++i) {
                corrHalf += filteredOnset[i] * filteredOnset[i + halfLag];
            }
            // If the half-lag (double tempo) correlation is nearly as strong, prefer standard 90-160 range
            if (corrHalf > bestCorr * 0.85f && detectedBpm < 90.0) {
                detectedBpm *= 2.0;
            }
        }

        // Round to 1 decimal place
        return std::round(detectedBpm * 10.0) / 10.0;
    }
};

#endif // BpmDetector_h
