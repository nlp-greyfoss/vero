from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str             
    openai_base_url: str = "https://api.openai.com/v1"  
    debug: bool = False
    timeout: int = 60
    model_name: str = "Qwen/Qwen3-32B"
    temperature: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
