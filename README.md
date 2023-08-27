# Named Entity Recogntition Extension

An implementation of the NER model described in <i>Template-Based Named Entity Recognition Using BART</i> (Cui, Leyang et al., 2021) as well as a NER Chrome browser extension and server.

## Getting Started

### Prerequisites

* Python 3 (Tested with 3.11)
* Node.js (Tested with Node 18)

### Training the Model

In `template-ner`:

Steps 1-2 are optional but recommended to avoid package conficts

1. Create a python virtual environment by running:
    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment. 

    For Linux (bash):
    ```bash
    . ./.venv/bin/activate
    ```

    For Windows:
    ```bash
    . ./.venv/Scripts/activate
    ```

3. Install dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

4. Download dataset by running:
    ```bash
    python dl_dataset.py
    ```

5. Train model by running:
    ```bash
    python train.py
    ```

    Model will be saved in folder named `best_model`

6. Test model and calculate F-1 by running:
    ```bash
    python test.py
    ```


### Building The Extension

In `ner-browser-extension`:

1. Install dependencies by running:
    ```bash
    npm install
    ```

2. Build extension by running:
    ```bash
    npm run build
    ```

    Note for development: `npm run dev` can be used to watch files and continuously rebuild extension on update

3. The extension is now built in the `dist` directory. Add the extension to Chrome

### Running the Backend Server

In `ner-backend`:

Steps 1-2 are optional but recommended to avoid package conficts

1. Create a python virtual environment by running:
    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment. 

    For Linux (bash):
    ```bash
    . ./.venv/bin/activate
    ```

    For Windows:
    ```bash
    . ./.venv/Scripts/activate
    ```

3. Install dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure a bart model is avaliable in a directory named `best_model`. Copying over the model you just trained is is recommended.

5. Start the server by running:
    ```bash
    python server.py
    ```

    The server will now be listening on `localhost:8080`


### Usage 

With the server running and extension installed in Chrome:

1. Open a webpage

2. Click `Run` when NER prompt popups up

3. Wait for labels to be generated. See console for errors if they don't show up.

    * Keep in mind this might take a significant amount of time to label the entire page