# Prerequisites, innan del 1

I del 1 kommer vi bygga vidare på implementationen av en enkel artificiell neuron från lektion 1, m.a.o. följande startläge: *(copy-paste från lektion 1)*

# Kravspecifikation för del 1

Scope för **del 1** av denna inlämningsuppgift är att implementera följande versioner av en OCR-perceptron för handskrivna siffror:

## (A) 🧮 Optional: En enda neuron

- A1: Neuron-implementation: for-loopar och Python-listor (som på tavlan lektion 1), ***alternativt***
- A2: Neuron-implementation: NumPy vektor-multiplikation internt i varje Neuron-objekt

## (B) ✅ ANN-lager: NumPy version

Det betyder att vi nu inte längre behöver någon klass Neuron, eftersom vi kommer beräkna ett helt lager som en enda stor matris-multiplikation:

- Alla input till ett lager = NumPy-vektor
- Alla vikter för alla neuroner i ett lager = en NumPy-matris
- Observera att vi inte kommer att träna nätverket som är implementerat som en NumPy-beräkning - eftersom det blir mycket enklare i (C) när vi övergår till PyTorch.

## (C) ✅ ANN-lager: PyTorch version:

- Använd PyTorch 2.1 (eller bättre). Använd helst Python 3.10 (eller bättre).
- Kopplas först ihop alla lager i perceptronen så att du får en PyTorch-modell (a.k.a. module). Denna definierar i detalj compute-grafen för din perceptron.
- Använd därefter din perceptron via PyTorch. Googla själv för att få information om hur detta går till rent praktiskt. Det finns gott om information på webben kring PyTorch!
- I denna version ska även träning av nätverket ske, d.v.s. vi ska loopa över epochs, och applicera back-prop. En vidareutveckling av back-prop som kallas ADAM brukar användas eftersom den är både snabb och inte lika ofta fastnar i dåliga lokala minima, jämfört med ren back-prop.
- Se avsnittet “Tips för (C)” nedan.

## (D) ✅ Samma som (C), men exekverad på en CUDA GPU

- GPU:n behöver stöda CUDA v11.6 eller högre, vilket motsvarar en GPU från NVIDIA’s Pascal-generation eller senare (Exempel på Pascal-kort: GeForce GTX-1080, Quadro P5000, Tesla P100). (Senare generationer: Volta, Turing, Ampère, Ada, Hopper, Blackwell).
- Google Colab har billiga/gratis notebook-instanser med NVIDIA T4 GPU, vilket är en enkel type av Turing-GPU. Denna fungerar utmärkt för uppgiften, men har du en modern NVIDIA-GPU i din dator är den troligen snabbare än en T4.
- Tips kring att använda GPU för beräkningarna:
    
    ```python
    # Move the model to the GPU# (Otherwise the GPU will not be used!)device = torch.device(f"cuda:0")  # Select the first CUDA GPU in the computermodel  = [model.to](http://model.to/)(device)
    # If you want to list all CUDA-capable GPUs:import os, torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")
    ```
    

---

## Tips för (C)

För uppgift (C) krävs att ni själva söker information kring hur man använder PyTorch. Det finns gott om resurser och dokumentation på webben. Om ni väljer att använda ChatGPT, använd endast ChatGPT-4 eller bättre, aldrig GPT-3.5. Det är viktigt att ni använder ChatGPT på rätt sätt, d.v.s:

- Använd gärna Chat som hjälpmedel för att se exempel på kod, men använd inte Chat för att producera kod ni inte förstår. Det är viktigt att ni förstår koden som ni lämnar in, men det är fritt fram att plocka delar från Chat, Google, StackOverflow, kollegor, etc.
- Om ni låter ChatGPT producera kod för att lösa delar av uppgiften, så är ett tips att läsa koden tills ni verkligen förstår den, och därefter radera koden och göra om det m.h.a. googling o dyl istället. Annars riskerar ni att lura er själva.
- Ett riktigt bra sätt att använda ChatGPT för att lära sig saker, är att berätta för Chat att du vill lära dig att göra X. Be Chat skriva korta svar. Försök att vara specific, och fråga om exakt vad du fastnat på. ***Berätta för Chat att du vill göra programmeringen själv*** och att Chat inte ska göra allt åt dig. Chat är en bra lärare, när du ber den vara det. Berätta för Chat ***hur*** du vill att den ska stödja dig, så gör den det!

Här är en mappning från den matematik vi har gått igenom (d.v.s. vad du implementerat i uppgift B) till ett par användbara klassnamn i PyTorch (nn.Sigmoid & nn.Linear). Observera att man typiskt använder två ihopkopplade moduler för att få vad vi brukar kalla för ett neuron-lager. PyTorch ser matrismultiplikation (och addition av bias) som ett första separat steg, och sedan komponent-vis applicering av activation function (t.ex. Sigmoid) som ett efterföljande steg:
screenshot.png

# Assignment-1-part-1: ANN & CNN

<aside>
💡

Glöm ej att läsa hela uppgiften innan ni börjar. Det finns användbar information i slutet av detta dokument.

</aside>

# Prerequisites, innan del 1

I del 1 kommer vi bygga vidare på implementationen av en enkel artificiell neuron från lektion 1, m.a.o. följande startläge: *(copy-paste från lektion 1)*

- 💻 Implementera en egen neuron (med for-loop över alla inputs) i er personliga Colab-miljö.
- 💻 Kapsla in funktionaliteten i en Python-klass.
- 🚀 Optional: Implementera några olika activation functions, t.ex. Sigmoid, ReLU, Leaky-ReLU och Tanh.

---

# Kravspecifikation för del 1

Scope för **del 1** av denna inlämningsuppgift är att implementera följande versioner av en OCR-perceptron för handskrivna siffror:

## (A) 🧮 Optional: En enda neuron

- A1: Neuron-implementation: for-loopar och Python-listor (som på tavlan lektion 1), ***alternativt***
- A2: Neuron-implementation: NumPy vektor-multiplikation internt i varje Neuron-objekt

## (B) ✅ ANN-lager: NumPy version

Det betyder att vi nu inte längre behöver någon klass Neuron, eftersom vi kommer beräkna ett helt lager som en enda stor matris-multiplikation:

- Alla input till ett lager = NumPy-vektor
- Alla vikter för alla neuroner i ett lager = en NumPy-matris
- Observera att vi inte kommer att träna nätverket som är implementerat som en NumPy-beräkning - eftersom det blir mycket enklare i (C) när vi övergår till PyTorch.

## (C) ✅ ANN-lager: PyTorch version:

- Använd PyTorch 2.1 (eller bättre). Använd helst Python 3.10 (eller bättre).
- Kopplas först ihop alla lager i perceptronen så att du får en PyTorch-modell (a.k.a. module). Denna definierar i detalj compute-grafen för din perceptron.
- Använd därefter din perceptron via PyTorch. Googla själv för att få information om hur detta går till rent praktiskt. Det finns gott om information på webben kring PyTorch!
- I denna version ska även träning av nätverket ske, d.v.s. vi ska loopa över epochs, och applicera back-prop. En vidareutveckling av back-prop som kallas ADAM brukar användas eftersom den är både snabb och inte lika ofta fastnar i dåliga lokala minima, jämfört med ren back-prop.
- Se avsnittet “Tips för (C)” nedan.

## (D) ✅ Samma som (C), men exekverad på en CUDA GPU

- GPU:n behöver stöda CUDA v11.6 eller högre, vilket motsvarar en GPU från NVIDIA’s Pascal-generation eller senare (Exempel på Pascal-kort: GeForce GTX-1080, Quadro P5000, Tesla P100). (Senare generationer: Volta, Turing, Ampère, Ada, Hopper, Blackwell).
- Google Colab har billiga/gratis notebook-instanser med NVIDIA T4 GPU, vilket är en enkel type av Turing-GPU. Denna fungerar utmärkt för uppgiften, men har du en modern NVIDIA-GPU i din dator är den troligen snabbare än en T4.
- Tips kring att använda GPU för beräkningarna:
    
    ```python
    # Move the model to the GPU# (Otherwise the GPU will not be used!)device = torch.device(f"cuda:0")  # Select the first CUDA GPU in the computermodel  = [model.to](http://model.to/)(device)
    # If you want to list all CUDA-capable GPUs:import os, torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")
    ```
    

---

## Tips för (C)

För uppgift (C) krävs att ni själva söker information kring hur man använder PyTorch. Det finns gott om resurser och dokumentation på webben. Om ni väljer att använda ChatGPT, använd endast ChatGPT-4 eller bättre, aldrig GPT-3.5. Det är viktigt att ni använder ChatGPT på rätt sätt, d.v.s:

- Använd gärna Chat som hjälpmedel för att se exempel på kod, men använd inte Chat för att producera kod ni inte förstår. Det är viktigt att ni förstår koden som ni lämnar in, men det är fritt fram att plocka delar från Chat, Google, StackOverflow, kollegor, etc.
- Om ni låter ChatGPT producera kod för att lösa delar av uppgiften, så är ett tips att läsa koden tills ni verkligen förstår den, och därefter radera koden och göra om det m.h.a. googling o dyl istället. Annars riskerar ni att lura er själva.
- Ett riktigt bra sätt att använda ChatGPT för att lära sig saker, är att berätta för Chat att du vill lära dig att göra X. Be Chat skriva korta svar. Försök att vara specific, och fråga om exakt vad du fastnat på. ***Berätta för Chat att du vill göra programmeringen själv*** och att Chat inte ska göra allt åt dig. Chat är en bra lärare, när du ber den vara det. Berätta för Chat ***hur*** du vill att den ska stödja dig, så gör den det!

Här är en mappning från den matematik vi har gått igenom (d.v.s. vad du implementerat i uppgift B) till ett par användbara klassnamn i PyTorch (nn.Sigmoid & nn.Linear). Observera att man typiskt använder två ihopkopplade moduler för att få vad vi brukar kalla för ett neuron-lager. PyTorch ser matrismultiplikation (och addition av bias) som ett första separat steg, och sedan komponent-vis applicering av activation function (t.ex. Sigmoid) som ett efterföljande steg:

![](attachment:b8e36811-d9fa-44f1-9bba-92bb1fa41f23:Export-5048730e-6604-4e7e-be29-180733c20e03AI-1_368d73ebc2ac492292ae1213953d00a9_Uppgift_2_Perceptron_for_OCR_9ae6ad9fa43f4c209864974d32cf75a6Screenshot_2024-04-19_at_10.35.13.png)

Screenshot 2024-04-19 at 10.35.13.png

---

## Data för delmoment 1

Vi kommer att använda ett välkänt kostnadsfritt dataset med 70,000 bilder på handskrivna siffror, som redan är “labelled”, vilket betyder att varje bild har en tillhörande klassificering av rätt svar (alltså vilken siffra 0-9 bilden faktiskt visar). Datasettet heter “MNIST” (”Modified National Institute of Standards and Technology database”) och är väldigt välkänt som “AI:ns Hello World”.

### Kickstart för att komma igång med nedladdning av data

MNIST kan laddas ner på flera sätt:

1. *Att **träna** version (A) och (B) ingår INTE i uppgiften*, men OM ni vill testa det, så kan man ladda ner datan såhär via ett bash-kommando:
    
    ```bash
    pip install get-mnist
    mnist --dataset mnist --cache [YourDesiredDirectory]
    ```
    
2. För **version (C) och (D)** - Nedan finns *ungefärlig* Python-kod, för att ladda ner datamängden som ett PyTorch dataset. Eftersom MNIST motsvarar AI-världens “Hello World”, och används på praktiskt taget alla intro-kurser till AI i hela världen, så finns MNIST som förberedd metod i *torchvision.datasets*:
    
    ```python
    # filename: mnist_loader.py
    #!/usr/bin/env python3
    
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Skapa en transform för att redan vid load...
    transform = transforms.Compose([
        transforms.ToTensor(),  # ... 1) konvertera data till [0.0-1.0], och
        transforms.Normalize((0.5,), (0.5,))  # ... 2) normalisera gråskalan
    ])
    
    # MNIST är så vanligt att det finns som funktion i torchvision.datasets
    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Skapa en DataLoader för träningsdata och en för testdata
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    ```
    

---