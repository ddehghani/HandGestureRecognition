# Hand Gesture Recognition

Hand gesture recognition is an emerging field within computer vision that focuses on identifying and interpreting human hand movements using algorithms and computational methods. This technology has gained traction due to its potential to enable more natural and intuitive interactions between humans and machines. Recent advancements in deep learning, particularly with Convolutional Neural Networks (CNNs), have driven significant progress in this area.
The relevance of hand gesture recognition extends beyond enhancing user interfaces. It offers solutions for seamless control in environments where traditional input devices are impractical, such as medical rehabilitation, remote device control, and interactive learning environments. By bridging the gap between complex commands and natural gestures, this technology can improve accessibility and efficiency across various domains.
Our project aims to contribute to this growing field by leveraging an existing dataset from Kaggle, supplemented with user-generated data, to train a customizable CNN model for hand gesture recognition. Implemented in Python, this application is designed to be accurate, efficient, and adaptable to specific use cases, addressing real-world challenges through innovative application of this technology.

## Installation

Please clone the repository and install the requirements like below:

```<pre>
git clone https://github.com/ddehghani/HandGestureRecognition.git
cd HandGestureRecognition
pip install -r requirements.txt
```

## Usage

Format your training video repository like the below:

```<pre>
├── data
│   ├── gesture1       # first hand-gesture
│   │   ├── video1     # first video of first hand-gesture
│   │   ├── video2     # second video of first hand-gesture
│   │   └── ...     
│   ├── gesture2       # second hand-gesture
│   │   ├── video1    
│   │   ├── video2
│   │   └── ...
│   └── ...
└── ...
```
and your test videos like the below:
```<pre>
├── test 
│   ├── video1     # first test video
│   ├── video2     # second test video
│   └── ...     
└── ...
```

then run:

```<pre>
python Train.py path/to/data -o ./output   
   
```
wait until the model trains on your training data. Then run:

```<pre>
python Classify.py path/to/test --class-file ./output/classes.dict -m ./output/model.pth
```
To classify and markup any number of videos.

Learn more about arguments and configurations by running:
```<pre>
python Classify.py -h

python Train.py -h
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
