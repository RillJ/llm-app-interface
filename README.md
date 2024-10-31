# Flexible Context-Aware Navigation Interface for Accessibitiy Apps

This project is a **flexible and intelligent context-aware navigation interface** designed to enhance interoperability with accessibility applications for vision-impaired individuals. Utilizing **LangChain** and **LLMs (Large Language Models)**, it provides a seamless API that supports both human and system inputs, enabling back-and-forth communication and clarifying interactions. It is designed to work with accesibility apps such as the [Natural Language Interface](https://github.com/StudyProject-NLI/NLInterface).

### Features

- **API Interface**: Exposed via `LangServe` for interaction with other apps.
- **Context-aware**: Uses a `LangGraph` based approach for efficient state management and future-proofing.
- **Interoperability**: Designed to enhance accessibility apps and facilitate communication with LLMs.
- **Support for OAuth**: User authentication using `OAuth2`.
- **Back-and-Forth Communication**: Enables real-time clarifying questions and data exchange between apps and users.
- **Customizable for Accessibility Apps**: Currently configured for a single accessibility app but designed with ambitions for broader app support.

---

## Getting Started

### Prerequisites

- **Python 3.x** installed
- API keys from **LangChain** and **OpenAI**.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gitfoe/llm-app-interface
   cd your-repo-url
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by creating a `.env` file in the project root and filling in the required fields:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_here
   ```

   To generate a secure `SECRET_KEY`, you can run:
   ```bash
   openssl rand -hex 32
   ```

### Usage

1. Start the server locally:
   ```bash
   python serve.py
   ```

   By default, the API will be accessible at:
   ```
   http://localhost:8000/llm-app-interface
   ```

2. The application currently supports one user:
   - **Username**: `johndoe`
   - **Password**: `secret`

3. Use the `/token` endpoint to generate a JWT token for authentication:
   ```bash
   curl -X POST "http://localhost:8000/token" -H "Content-Type: application/x-www-form-urlencoded" -d "username=johndoe&password=secret"
   ```

---

## API Endpoints

- **`/llm-app-interface/invoke`**  
  The primary endpoint for invoking LLM interactions, supporting both system and human input.

  **Example Request**:
  ```bash
  curl -X POST "http://localhost:8000/llm-app-interface/invoke" \
       -H "Authorization: Bearer <your_jwt_token>" \
       -H "Content-Type: application/json" \
       -d '{"messages": "Hello, can you assist me?"}'
  ```

  **Note**: Ensure you include a valid JWT token for authorization.

---

## Integrating your Accessibility App

You can extend the model to accommodate additional accessibility apps by adding descriptions of your app functionalities to the documents within the `tools.py` file.

---

## Security

- **OAuth2** is implemented for user authentication. The `SECRET_KEY` is used to sign and validate JWT tokens.
- Currently, one user (`johndoe`) is supported. For more users, you can extend the `fake_users_db` or integrate with a proper database.
  
---

## Future Plans

- **Multi-App Support**: The current version only supports one accessibility app at a time, but we aim to generalize this for multiple apps.
- **LLM Flexibility**: While this system is configured for OpenAI models, it can easily be adapted to other LLMs by swapping out the model initialization.
- **Increasing Reliability**: A base of the model has been implemented, but reliability can be improved by i.e. utilizing additional nodes within the graph.

---

## Contributing

We welcome contributions! Please submit a pull request or open an issue for discussion.