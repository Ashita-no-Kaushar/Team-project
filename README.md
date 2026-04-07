# **AI/ML- Based Cryptographic Algorithm Identification**

Welcome to the repository for the **AI/ML-Based Cryptographic Algorithm Identification** project. This repository is maintained primarily for learning and applying practical knowledge across cryptography, machine learning, and full-stack development. The system analyzes encrypted/hash-like data patterns to estimate which cryptographic algorithm may have produced them, while also exposing educational tooling for encryption samples, prediction metadata, and benchmark diagnostics.


# **Real-World Problem Statement: AI/ML-Based Cryptographic Algorithm Identification**


## 🚨 The Challenge

In today’s digital age, cryptographic algorithms are the backbone of secure communication, data protection, and privacy. However, as cyber threats evolve, attackers often exploit weaknesses in cryptographic implementations or use outdated algorithms to breach systems. Identifying the cryptographic algorithm used in a given dataset or communication stream is a critical step in assessing security vulnerabilities, but this process is **highly complex** and **time-consuming** when done manually.

### Why Is This a Real Issue?
1. **Rising Cyberattacks**: Attackers frequently use weak or deprecated cryptographic algorithms to exploit systems. Identifying these algorithms in real-time is crucial for preventing breaches.
2. **Lack of Automation**: Security analysts often rely on manual analysis or heuristic methods to identify cryptographic algorithms, which is error-prone and inefficient.
3. **Complexity of Modern Cryptography**: With the proliferation of cryptographic standards (e.g., AES, RSA, ECC, ChaCha20) and custom implementations, it’s nearly impossible for humans to analyze and identify algorithms at scale.
4. **Hidden Weaknesses**: Even strong algorithms can have weak implementations or configurations, which are difficult to detect without advanced pattern recognition.

## 🎯 The Goal

This project aims to address these challenges by developing an **AI/ML-based system** that can automatically identify cryptographic algorithms from modern cryptographic datasets. By leveraging machine learning, the system will:
- Analyze data patterns and features to determine the algorithm used.
- Automate the identification process, reducing manual effort and human error.
- Provide insights into potential weaknesses in cryptographic implementations.
- Enhance the ability of security teams to respond to threats in real-time.

## 💡 Why This Is Hard

1. **Data Complexity**: Cryptographic datasets are highly complex, with patterns that are difficult to discern without advanced ML techniques.
2. **Algorithm Diversity**: Modern cryptography involves a wide range of algorithms, each with unique characteristics and implementations.
3. **Real-Time Requirements**: Identifying algorithms in real-world scenarios requires high accuracy and low latency, which is challenging to achieve.
4. **Adversarial Environments**: Attackers often obfuscate or modify cryptographic implementations to evade detection, making the problem even more complex.

## 🌍 Real-World Impact

This project has the potential to revolutionize cryptographic security by:
- Enabling faster and more accurate identification of cryptographic algorithms.
- Helping organizations detect and mitigate vulnerabilities in their systems.
- Providing a foundation for building smarter, AI-driven security tools.
- Contributing to the global effort to combat cybercrime and protect sensitive data.

---

By tackling this problem, we aim to bridge the gap between cryptography, machine learning, and cybersecurity, creating a tool that is both innovative and impactful in the real world.



# 📂 Presentation  
You can view the project presentation, some key points are discussed in this ppt regarding the project here:  

[View Presentation](https://docs.google.com/presentation/d/1HG2yGzrDRv5D98kzMdwgv0mOmoPEIMLZ/edit?usp=sharing&ouid=113623327821126759756&rtpof=true&sd=true)

## 🚀 Tech Stack

Our project uses a practical full-stack architecture that combines frontend UI, backend APIs, and ML inference.

### 🖥 Frontend
- React 19 + Vite 6 + TypeScript
- Tailwind CSS 4, Framer Motion, and Lucide React
- Redux Toolkit for auth state management
- Axios for API integration

### 🧠 Machine Learning
- Python-based prediction pipeline under `Backend/src/main/resources/scripts`
- Hybrid model artifacts (`model.pickle`, `label_map.pickle`)
- Core libs: `numpy`, `scipy`, `scikit-learn`, `pycryptodome`

### 🔐 Backend
- Spring Boot 3.4 (Java 21)
- Spring Security + JWT-based auth
- Spring Data JPA
- H2 default for local dev, MySQL-ready configuration for production

# Prerequisites

Install these before running locally:
- Java 21
- Maven (or use `./mvnw` wrapper)
- Node.js 20+ and npm
- Python 3.10+
- Git

------------------------------------------------------------
## Step 1: Clone the Repository
```bash
git clone https://github.com/Ashita-no-Kaushar/Team-project.git
cd Team-project
```

------------------------------------------------------------
## Step 2: Setup Backend (Spring Boot)

### Navigate to backend
```bash
cd Backend
```

### Setup Python environment for ML scripts
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/main/resources/scripts/requirements.txt
```

### Database configuration
- Default local setup uses in-memory H2 from `application.properties`.
- For production, switch to MySQL/PostgreSQL settings in `application.properties` or environment variables.

### Run backend (Java 21)
```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:$PATH ./mvnw spring-boot:run
```

------------------------------------------------------------
## Step 3: Setup Frontend

### Navigate to frontend
```bash
cd ../Frontend
```

### Install dependencies
```bash
npm ci
```

### Run frontend
```bash
npm run dev -- --host 0.0.0.0 --port 5173
```

------------------------------------------------------------
## Step 4: Verify Local Setup

### Frontend
```text
http://localhost:5173
```

### Backend quick checks
```bash
curl http://localhost:8080/api/encryption/aes
curl -X POST http://localhost:8080/api/ml/predict -H "Content-Type: application/json" -d '{"input_hex":"00112233445566778899aabbccddeeff"}'
```

### Auth note
- `POST /api/ml/predict` is public.
- `/api/ml/model-info`, `/api/ml/benchmark`, and `/api/ml/benchmark/history` require JWT authentication.

------------------------------------------------------------
## Installation and setup complete

```plaintext
Team-project/
├── Backend/
│   ├── pom.xml
│   ├── mvnw
│   ├── src/main/java/com/project/backend/
│   │   ├── Controllers/
│   │   │   ├── AuthController.java
│   │   │   ├── EncryptionController.java
│   │   │   ├── MLController.java
│   │   ├── Services/
│   │   │   ├── MLService.java
│   │   │   ├── GeneratedSampleRegistryService.java
│   │   │   ├── BenchmarkHistoryService.java
│   │   ├── Entities/
│   │   │   ├── PredictionResult.java
│   ├── src/main/resources/
│   │   ├── application.properties
│   │   ├── scripts/
│   │   │   ├── predict.py
│   │   │   ├── train_model.py
│   │   │   ├── requirements.txt
├── Frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionPage.tsx
│   │   │   ├── SignUp.tsx
```
# RestEndpoints & their description 

# Cryptographic Data Controller

## Description
The `CryptographicDataController` provides REST endpoints to manage cryptographic data.  
It allows retrieving cryptographic data by ID, fetching the last 20 cryptographic data entries,  
and updating the `correctedData` status of a specific entry.

## REST Endpoints

### 1. Get Last 20 Cryptographic Data
- **Endpoint:** `/api/cryptographic-data/last20`
- **Method:** `GET`
- **Description:** Fetches the last 20 cryptographic data entries for the current user.

### 2. Update Corrected Data
- **Endpoint:** `/api/cryptographic-data/correctedData/{id}`
- **Method:** `PUT`
- **Description:** Updates the `correctedData` status of a cryptographic entry.
- **Request Parameters:**  
  - `id` (Path Variable) → ID of the cryptographic data.  
  - `correctedData` (Query Parameter) → Boolean value to update the corrected data status.

### 3. Get Cryptographic Data by ID
- **Endpoint:** `/api/cryptographic-data/{id}`
- **Method:** `GET`
- **Description:** Retrieves cryptographic data based on the given ID.
- **Request Parameters:**  
  - `id` (Path Variable) → ID of the cryptographic data.


# Encryption Controller

## Description
The `EncryptionController` provides REST endpoints for encrypting data using various cryptographic algorithms.  
It supports symmetric encryption (AES, DES, Triple DES, Blowfish, RC2, RC4, ChaCha20),  
asymmetric encryption (RSA), digital signatures (DSA, ECDSA), key exchanges (Diffie-Hellman, ECDH),  
and hashing algorithms (MD5, SHA-1, SHA-256, SHA-512, SHA-3-256).

## REST Endpoints

### Symmetric Encryption
1. **AES Encryption**
   - **Endpoint:** `/api/encryption/aes`
   - **Method:** `GET`
   - **Description:** Encrypts data using AES encryption.

2. **DES Encryption**
   - **Endpoint:** `/api/encryption/des`
   - **Method:** `GET`
   - **Description:** Encrypts data using DES encryption.

3. **Triple DES Encryption**
   - **Endpoint:** `/api/encryption/3des`
   - **Method:** `GET`
   - **Description:** Encrypts data using Triple DES encryption.

4. **Blowfish Encryption**
   - **Endpoint:** `/api/encryption/blowfish`
   - **Method:** `GET`
   - **Description:** Encrypts data using Blowfish encryption.

5. **RC2 Encryption**
   - **Endpoint:** `/api/encryption/rc2`
   - **Method:** `GET`
   - **Description:** Encrypts data using RC2 encryption.

6. **RC4 Encryption**
   - **Endpoint:** `/api/encryption/rc4`
   - **Method:** `GET`
   - **Description:** Encrypts data using RC4 encryption.
   - **Query Parameter:**  
     - `plaintext` (Optional) → The text to encrypt.

7. **ChaCha20 Encryption**
   - **Endpoint:** `/api/encryption/chacha20`
   - **Method:** `GET`
   - **Description:** Encrypts data using ChaCha20 encryption.

### Asymmetric Encryption
8. **RSA Encryption**
   - **Endpoint:** `/api/encryption/rsa`
   - **Method:** `GET`
   - **Description:** Encrypts data using RSA encryption.

### Digital Signatures
9. **DSA Signature Generation**
   - **Endpoint:** `/api/encryption/dsa`
   - **Method:** `GET`
   - **Description:** Generates a digital signature using DSA.

10. **ECDSA Signature Generation**
   - **Endpoint:** `/api/encryption/ecdsa`
   - **Method:** `GET`
   - **Description:** Generates a digital signature using ECDSA.

### Key Exchange
11. **Diffie-Hellman Key Exchange**
    - **Endpoint:** `/api/encryption/diffe`
    - **Method:** `GET`
    - **Description:** Performs a key exchange using Diffie-Hellman.

12. **ECDH Key Exchange**
    - **Endpoint:** `/api/encryption/ecdh`
    - **Method:** `GET`
    - **Description:** Performs a key exchange using ECDH.

### Hashing Algorithms
13. **MD5 Hash Generation**
    - **Endpoint:** `/api/encryption/md5`
    - **Method:** `GET`
    - **Description:** Generates an MD5 hash.

14. **SHA-1 Hash Generation**
    - **Endpoint:** `/api/encryption/sha1`
    - **Method:** `GET`
    - **Description:** Generates a SHA-1 hash.

15. **SHA-256 Hash Generation**
    - **Endpoint:** `/api/encryption/sha256`
    - **Method:** `GET`
    - **Description:** Generates a SHA-256 hash.

16. **SHA-512 Hash Generation**
   - **Endpoint:** `/api/encryption/sha512`
   - **Method:** `GET`
   - **Description:** Generates a SHA-512 hash.

17. **SHA-3-256 Hash Generation**
    - **Endpoint:** `/api/encryption/sha3-256`
    - **Method:** `GET`
    - **Description:** Generates a SHA-3-256 hash.


# ML Controller

## Description
The `MLController` provides REST endpoints for predicting cryptographic algorithms used in encryption.  
It processes input hexadecimal strings and determines the corresponding algorithm using machine learning techniques.

## REST Endpoints

### Predict Cryptographic Algorithm
- **Endpoint:** `/api/ml/predict`  
- **Method:** `POST`  
- **Description:** Predicts the cryptographic algorithm used for encryption based on the provided hexadecimal string.

### Model Info (Protected)
- **Endpoint:** `/api/ml/model-info`
- **Method:** `GET`
- **Description:** Returns ML runtime metadata (model version, confidence threshold, artifact availability, timeout settings).
- **Auth:** `Authorization: Bearer <token>` required.

### Quick Benchmark (Protected)
- **Endpoint:** `/api/ml/benchmark`
- **Method:** `GET`
- **Query Parameter:** `mode` (optional) = `hash-only` | `mixed` | `strict` (default: `mixed`)
- **Description:** Runs benchmark cases and returns summary metrics.
- **Auth:** `Authorization: Bearer <token>` required.

### Benchmark History (Protected)
- **Endpoint:** `/api/ml/benchmark/history`
- **Method:** `GET`
- **Query Parameters:**
   - `mode` (optional): filter by `hash-only`, `mixed`, `strict`
   - `limit` (optional): number of records to return
- **Description:** Returns recent benchmark runs for trend analysis.
- **Auth:** `Authorization: Bearer <token>` required.

### Benchmark Response Fields (Summary)
- `mode`: benchmark mode used for the run.
- `run_at`: timestamp when benchmark executed.
- `total_cases`, `passed_cases`, `pass_rate`: benchmark accuracy summary.
- `duration_ms`: runtime for the benchmark.
- `avg_confidence`: average model confidence across benchmark cases (when available).
- `results`: per-case details including expected algorithm, predicted algorithm, source, confidence, and match status.

### Access Control Note
- `POST /api/ml/predict` remains public for prediction requests.
- `/api/ml/model-info`, `/api/ml/benchmark`, and `/api/ml/benchmark/history` require JWT authentication.


# Authentication Controller

## Description
The `AuthController` provides REST endpoints for user authentication, including signup, login, token refresh, and logout.  
It integrates with authentication services to manage user access securely.

## REST Endpoints

### User Signup
- **Endpoint:** `/api/auth/signup`  
- **Method:** `POST`  
- **Description:** Registers a new user with the provided credentials.

### User Login
- **Endpoint:** `/api/auth/login`  
- **Method:** `POST`  
- **Description:** Authenticates the user and returns an access token.

### Refresh Token
- **Endpoint:** `/api/auth/refresh`  
- **Method:** `POST`  
- **Description:** Refreshes the access token using a valid refresh token.

### User Logout
- **Endpoint:** `/api/auth/logout`  
- **Method:** `POST`  
- **Description:** Logs out the user by invalidating the current session tokens.


# Algorithm Controller

## Description
The `AlgorithmController` provides endpoints to retrieve cryptographic algorithms, including fetching all algorithms in random order, retrieving a specific algorithm by name, and getting random algorithms excluding a specific one.

## REST Endpoints

### Get All Algorithms in Random Order
- **Endpoint:** `/api/algorithms/random`  
- **Method:** `GET`  
- **Description:** Returns a list of all algorithms in a randomized order.

### Get Algorithm by Name
- **Endpoint:** `/api/algorithms/{algo}`  
- **Method:** `GET`  
- **Description:** Fetches details of a specific algorithm by its name.

### Get 4 Random Algorithms Excluding a Specific One
- **Endpoint:** `/api/algorithms/random/exclude`  
- **Method:** `GET`  
- **Description:** Retrieves four random algorithms while excluding a specified algorithm.  
- **Query Parameter:** `exclude` – The name of the algorithm to exclude from the results.


# User Controller

## Description
The `UserController` provides endpoints for managing user profiles, including updating user details and retrieving authenticated user information.

## REST Endpoints

### Update User Profile
- **Endpoint:** `/api/users/update`  
- **Method:** `PUT`  
- **Description:** Allows authenticated users to update their profile information.  
- **Request Body:** `UpdateUserRequest` – Contains updated user details.  
- **Authentication:** Required.

### Get Authenticated User Details
- **Endpoint:** `/api/users/me`  
- **Method:** `GET`  
- **Description:** Retrieves the details of the currently authenticated user based on the provided JWT token.  
- **Request Header:** `Authorization: Bearer <token>` – JWT token for authentication.  
- **Response:** Returns the username, first name, and last name of the authenticated user.




# Relevant Diagrams for the project

### Use Case Diagram for the project
![Use case](https://github.com/razasoneji/CryptML/blob/main/Backend/images/UseCaseDiagram_final.png?raw=true)

### State Diagram for the project

![]()
![Class Diagram](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_Class_Diagram.png?raw=true)


### Activity Diagram for the project

### Authentication Activity Diagram
![actovity Diagram](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_Authentication_Activity.png?raw=true)


### Encryption Activity Diagram
![dsfsdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_Encryption_Activity.png?raw=true)



### Prediction Activity Diagram 
![dsfsdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_Prediction_Activity.png?raw=true)




### Sequence Diagram for the project


### Login Sequence Diagram 

![dsfdsfsdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_login.png?raw=true)


### Refresh Sequence Diagram
![actovsdfity Diagram](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_Refresh.png?raw=true)


### Signup Sequence Diagram
![dsfssdfdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Final_Signup.png?raw=true)




### State Diagram for the project


### Prediction State Diagram
![dsfsdfssdfdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/State_Prediction_final.png?raw=true)

### Encrypting State Diagram 
![dsfdsdsffdsfsdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/State_Final_Encrypting.png?raw=true)



### User profile State Diagram
![dsfsdfssdfsdfsdfdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/State_User_Final_profile.png?raw=true)


### User Authentication State Diagram
![actovsdfitdsfyasdf Diagram](https://github.com/razasoneji/CryptML/blob/main/Backend/images/State_Final_UserAuthentication.png?raw=true)


### Random Forest ML Architecture
![sdfasdfasdfasdfasd](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Screenshot%202025-02-23%20233110.png?raw=true)


#Images of Project

![sdfadsfsdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Screenshot%202025-02-23%20234340.png?raw=true)


![sdfasdfadsewfrw](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Screenshot%202025-02-23%20234227.png?raw=true)


![sdfasdfadsewfrwsd](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Screenshot%202025-02-23%20234347.png?raw=true)


![dsfasdfewdfsdv](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Screenshot%202025-02-23%20234359.png?raw=true)


![asefsdfsdf](https://github.com/razasoneji/CryptML/blob/main/Backend/images/Screenshot%202025-02-23%20234407.png?raw=true)











