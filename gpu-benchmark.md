Big batch == bad
https://arxiv.org/pdf/1609.04836.pdf

Deep Learning GPU Benchmarks - Tesla V100 vs RTX 2080 Ti vs GTX 1080 Ti vs Titan V
https://lambdalabs.com/blog/best-gpu-tensorflow-2080-ti-vs-v100-vs-titan-v-vs-1080-ti-benchmark/

On Scalable Deep Learning and Parallelizing Gradient Descent (Master Thesis Joeri R. Hermans)

https://raw.githubusercontent.com/JoeriHermans/master-thesis/master/thesis/master_thesis_joeri_hermans.pdf

5 tips for multi-GPU training with Keras
https://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/




## Literature

## Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour

https://arxiv.org/pdf/1706.02677.pdf

scale learning rate linearly with batch size 
test on ResNet-50
comparable results up to 8k batch size
0.9 scaling efficiency from 8 to 256 GPUs


minibatch size auslastung vs. stochastizität / convergence
acc vergleich minibatch 1 vs 128

run to completion vs. langsam rechnen


task parallel training

was wenn das modell nicht auf eine gpu passt


deep learning workload verschiedene netztypen

cpu auslastung


2 netzte gleichzeitig auf einer gpu: ja
mit https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
https://cs.stanford.edu/~matei/papers/2018/sysml_modelbatch.pdf << ensemble w/ shared preprocessing
https://www.usenix.org/system/files/osdi18-moritz.pdf
43, 42



modell parallel training bsp: https://arxiv.org/abs/1609.08144

multiple workers per gpu: http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf

epochs to accuracy

small batch:
less redundance
noisier gradients = more exploration

hyper parameter tuning to allow for larger batch sizes

SMA syn-chronous model averaging
https://github.com/lsds/Crossbow


NATURALGRA-DIENT ANDPARAMETERAVERAGING?

read: [energy efficency]
http://proceedings.mlr.press/v77/cai17a/cai17a.pdf
http://zpascal.net/cvpr2017/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.pdf







## 2020-01-14

stromverbrauch vom restlichen system

Next: powercap + linreg compare

These checkpointing verursacht spikes
-- erster anschein : nein

These2: validation verusacht spikes
--> validation ausschalten
--> test checkpoints auf ramdisk
sampling-rate erhöhen

nvidia-smi mit mehreren prozessen parallel 


2 modelle in einem kernel oder 2 kernel parallel?

wie viele cuda cores benutzt ein kernel


speicher pro prozess beschränken
(speicher von tensorflow beschränken)

wie viel gpu kann ich mir freiräumen


auslastung bei verschiedenen modellen

-ecg
-simples toy set
-großes modell

# calsification

-#layer
-breite der layer
-batchsize
-typ der layer lstm vs. 2d conv


# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



Fragestellungen
Experimente
Discussion
Roadmap


Nächste woche:
- mit power limits
- 

wann lohnt sich was; bei inference ?
wie groß muss mein netz sein
lohnt sich gpu bei transferlearning
wie viel energie spart transferlearning

### Fragen
- Spikes ?
- höhere sampling rate von smi daten
- small batch with multi tenant 


### Experiments
- run to completion vs. langsam rechnen (powercap + linreg compare) eigentlich auch für mehrere versch. netze
- Batchsize vs. Power
- Tensorflow speicher begrenzen vs. Theoretischer speicher bedarf
    - stromverbrauch und zeit bei weniger speicher
- cpu vs. gpu inference (mit VGG)
- deep learning workload verschiedene netztypen  toy vs. telemed vs. complex
  

### Störeinflüsse
- System grundrauschen verringern ?
- Variabilität Initial Gewichte, Dropout >> random seed festlegen

### Messtechnik
- IPMI funktioniert
- NVIDIA SMI funktioniert auch ; aber vlt höhere SamplingRate
- Rapl geht theoretisch; muss noch verstanden werden
  - via perf stat 
  - via powerstat

### Future Work
#### Multi Tennancy GPU
Can we reuse data procressing on one gpu for multiple Networks? yes; but how much can they differ
Can we train two independet networks on one gpu?
Can we train one network and do something else?
Can we train two networks interleaved for gpu sharing? (Shared memory for two users but only one computes at a time)

#### 

- Powerlimits auf dem Host
- training multi gpu effizienz (power, auslastung)
- energie sparen durch transferlearning




## Vortrag

Stromverbauch von DeepLearning
Wie messe ich den Stromverbrauch?
Rapl Counter (powerstat kann 2Hz eigentlich 1000Hz) citation needed << wrapper um rapl counter
powecap clock freqs mit messen

stellschrauben an der Hardware:
- takt freq, speicher freq
- power limits
- speicherlimit
- powerstates ?



idle vs vollast cpu

benchmark 


Workloads / Benchmarks beschreiben << seed rng
parameter am modell:
- layer größe
- batchsize



einfluss der messung auf laufzeit
auf kleinerem modell

powercap auf durchschnittswert

2 epochen laufen lassen stromverbrauch extrapolieren

stromverbrauch abbruchbedingung vs. ohne abbruch