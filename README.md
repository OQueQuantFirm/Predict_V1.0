
1. **Install Dependencies:**
   - Make sure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/).
   - Navigate to the project directory (`Quant_Ai_V1.0`) in the terminal.

2. **Create Virtual Environment (Optional):**
   - It's good practice to use a virtual environment to isolate project dependencies. You can create a virtual environment by running:
     ```bash
     python -m venv venv
     ```
     Activate the virtual environment:
     - On Windows:
       ```bash
       .\venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

3. **Install Dependencies:**
   - Install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

4. **Set Up Environment Variables:**
   - Copy the content of `example.env` into a new file named `.env`.
   - Replace the placeholder values with your actual API key, secret key, and passphrase.

5. **Run the Script:**
   - Run the trading bot script:
     ```bash
     python bot.py
     ```

   - To interrupt the bot on the terminal:
     ```bash
     ctrl + c
     ```
   

The script will start executing the main trading loop. Ensure that your environment is set up correctly and that you have the necessary permissions and API keys for the KuCoin Futures API.

Note: The script might run indefinitely in the main trading loop, and you should manually stop it when needed. It's also important to thoroughly understand and test any trading strategy before deploying it in a live environment.

