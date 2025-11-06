"""
Main VLM analyzer for fraud detection using LangChain
"""
import base64
import json
import time
import io
from pathlib import Path
from typing import Union, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import VLMConfig, OUTPUT_DIR
from .models import VLMFraudAnalysis
from .prompts import FRAUD_DETECTION_PROMPT, FORMAT_REASONING_INSTRUCTIONS


REASONING_MODELS = {
    "gemini-2.5-pro",
    "gpt-5",
    "o1-preview",
    "o1-mini",
    "o3-mini"
}

class VLMAnalysisResult(BaseModel):
    """Results from VLM fraud detection analysis"""
    
    image_path: str = Field(..., description="Path to the analyzed image")
    model_provider: str = Field(..., description="VLM model used for analysis")
    analysis: VLMFraudAnalysis = Field(..., description="Structured fraud analysis")
    processing_time: float = Field(..., description="Time taken for analysis in seconds")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "image_path": self.image_path,
            "model_provider": self.model_provider,
            "processing_time": self.processing_time,
            "analysis": self.analysis.to_dict()
        }


class VLMAnalyzer(BaseModel):
    """
    Main analyzer for VLM-based fraud detection
    Uses LangChain to handle different VLM providers with consistent interface
    """
    config: VLMConfig = Field(default_factory=VLMConfig)
    save_dir: Path = OUTPUT_DIR
    
    def _initialize_llm(self) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]:
        """Initialize the LangChain LLM based on configuration"""
        model_id = self.config.get_openrouter_model_id()
        
        # Use OpenRouter for all models
        if self.config.openrouter_api_key.get_secret_value():
            return ChatOpenAI(
                model=model_id,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=self.config.openrouter_api_key.get_secret_value(),
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/Sphinx-HQ/referenced-pablo",
                    "X-Title": "Check Fraud Detection"
                }
            )
        
        # Fallback to native APIs if OpenRouter key is not available
        elif self.config.model_provider.startswith("gpt") and self.config.openai_api_key.get_secret_value():
            return ChatOpenAI(
                model=self.config.model_provider,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.openai_api_key.get_secret_value()
            )
        
        elif self.config.model_provider.startswith("claude") and self.config.anthropic_api_key.get_secret_value():
            return ChatAnthropic(
                model=self.config.model_provider,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.anthropic_api_key.get_secret_value()
            )
        
        elif self.config.model_provider.startswith("gemini") and self.config.google_api_key.get_secret_value():
            return ChatGoogleGenerativeAI(
                model=self.config.model_provider,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                google_api_key=self.config.google_api_key.get_secret_value()
            )
        
        else:
            raise ValueError(
                f"No valid API key found for model: {self.config.model_provider}. "
                "Please set OPENROUTER_API_KEY or the appropriate provider key."
            )
    
    def _is_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model"""
        return self.config.model_provider in REASONING_MODELS
    
    def _initialize_formatter_llm(self) -> ChatOpenAI:
        """Initialize a default LLM for formatting reasoning model outputs"""
        # Use GPT-4o-mini as the formatter (fast and cheap)
        formatter_model = "gpt-4o"
        
        if self.config.openrouter_api_key.get_secret_value():
            return ChatOpenAI(
                model=formatter_model,
                temperature=0,  # Deterministic formatting
                max_tokens=4096,
                openai_api_key=self.config.openrouter_api_key.get_secret_value(),
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/Sphinx-HQ/referenced-pablo",
                    "X-Title": "Check Fraud Detection Formatter"
                }
            )
        elif self.config.openai_api_key.get_secret_value():
            return ChatOpenAI(
                model=formatter_model,
                temperature=0,
                max_tokens=4096,
                api_key=self.config.openai_api_key.get_secret_value()
            )
        else:
            raise ValueError("No API key available for formatter LLM")
        
    def _encode_image(self, image_path: Union[str, Path], max_size_mb: float = 4.5) -> str:
        """
        Encode image to base64 string for API with automatic compression if needed
        
        Args:
            image_path: Path to image file
            max_size_mb: Maximum size in MB (default 4.5 to stay under 5MB limit)
            
        Returns:
            Base64 encoded image with data URI prefix
        """
        image_path = Path(image_path)
        
        # Check original file size
        original_size_mb = image_path.stat().st_size / (1024 * 1024)

        expected_base64_size_mb = (4/3) * original_size_mb 
        
        if expected_base64_size_mb <= max_size_mb:
            # File is small enough, encode directly
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Determine MIME type
            suffix = image_path.suffix.lower()
            mime_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
        else:
            # File exceeds limit, compress it
            print(f"⚠️  Image size ({original_size_mb:.2f} MB) exceeds limit. Compressing...")
            image_data, mime_type = self._compress_image(image_path, max_size_mb / 1.33)
            final_size_mb = len(image_data) / (1024 * 1024)
            final_base64_size_mb = (4/3) * final_size_mb
            print(f"✅ Compressed to {final_size_mb:.2f} MB - (~{final_base64_size_mb:.2f} MB base64)")
        
        # Encode to base64
        base64_image = base64.b64encode(image_data).decode("utf-8")
        
        return f"data:{mime_type};base64,{base64_image}"
    
    def _compress_image(self, image_path: Path, max_size_mb: float) -> tuple[bytes, str]:
        """
        Compress image to meet size requirements
        
        Args:
            image_path: Path to image file
            max_size_mb: Maximum size in MB
            
        Returns:
            Tuple of (compressed image bytes, mime_type)
        """
        # Open image
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if necessary
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Try progressive quality reduction first
        quality = 95
        while quality > 20:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            size_mb = buffer.tell() / (1024 * 1024)
            
            if size_mb <= max_size_mb:
                return buffer.getvalue(), "image/jpeg"
            
            quality -= 10
        
        # If still too large, resize the image
        max_dimension = 2048
        while max_dimension >= 512:
            # Resize maintaining aspect ratio
            img_copy = img.copy()
            img_copy.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img_copy.save(buffer, format='JPEG', quality=85, optimize=True)
            size_mb = buffer.tell() / (1024 * 1024)
            
            if size_mb <= max_size_mb:
                print(f"   Resized to {img_copy.size[0]}x{img_copy.size[1]}")
                return buffer.getvalue(), "image/jpeg"
            
            max_dimension -= 256
        
        # Last resort: use lowest settings
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=60, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    
    def analyze_image(self, image_path: Union[str, Path]) -> VLMAnalysisResult:
        """
        Analyze a single check image for fraud
        
        Args:
            image_path: Path to the check image
            
        Returns:
            VLMAnalysisResult with structured fraud analysis
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"🔍 Analyzing: {image_path.name}")
        print(f"Using model: {self.config.model_provider}")

        is_reasoning = self._is_reasoning_model()
        if is_reasoning:
            print("Reasoning model detected, will use secondary formatter LLM.")

        # Initialize LLM
        llm = self._initialize_llm()
        
        # Encode image
        image_data = self._encode_image(image_path)

        # Parser for structured output
        parser = PydanticOutputParser(pydantic_object=VLMFraudAnalysis)
        
        values = {
            "image_data": image_data,
            "format_instructions": parser.get_format_instructions()
        }
    
        try:
            if is_reasoning:
                # First pass: get raw reasoning output
                reasoning_chain = FRAUD_DETECTION_PROMPT | llm
                raw_output = reasoning_chain.invoke(values)
                
                print("Initial reasoning output obtained, formatting...")
                raw_text = raw_output.content if hasattr(raw_output, 'content') else str(raw_output)

                # Second pass: format the output into structured JSON
                formatter_llm = self._initialize_formatter_llm()
                formatting_chain = FORMAT_REASONING_INSTRUCTIONS | formatter_llm | parser
                analysis = formatting_chain.invoke({
                    "raw_output": raw_output,
                    "format_instructions": parser.get_format_instructions()
                })
            
            else:
                # Directly get structured output
                chain = FRAUD_DETECTION_PROMPT | llm | parser
                analysis = chain.invoke(values)
                
        except Exception as e:
            print(f"⚠️  LLM output failed, error: {e}")
            raise

        processing_time = time.time() - start_time
        
        # Create result
        result = VLMAnalysisResult(
            image_path=str(image_path),
            model_provider=self.config.model_provider,
            analysis=analysis,
            processing_time=processing_time
        )
        
        # Save results
        if self.config.save_results:
            self._save_results(result)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _save_results(self, result: VLMAnalysisResult):
        """Save analysis results to disk"""
        image_path = Path(result.image_path)
        output_dir = self.save_dir / self.config.model_provider / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / "_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Copy original image
        if self.config.save_visualizations:
            import shutil
            shutil.copy2(image_path, output_dir / "original.jpg")
    
    def _print_summary(self, result: VLMAnalysisResult):
        """Print analysis summary to console"""
        analysis = result.analysis
        
        print(f"\n{'='*60}")
        print(f"🎯 FRAUD ANALYSIS RESULT")
        print(f"{'='*60}")
        print(f"Image: {Path(result.image_path).name}")
        print(f"Model: {result.model_provider}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"\n{'─'*60}")
        print(f"Fraudulent: {'YES' if analysis.is_fraudulent else 'NO'}")
        print(f"Confidence: {analysis.overall_confidence}")
        print(f"Fraud Score: {analysis.fraud_likelihood_score}/100")
        print(f"Recommendation: {analysis.recommendation.upper()}")
        
        if analysis.primary_fraud_types:
            print(f"\n{'─'*60}")
            print(f"Fraud Types:")
            for fraud_type in analysis.primary_fraud_types:
                print(f"  • {fraud_type}")
        
        if analysis.suspicious_regions:
            print(f"\n{'─'*60}")
            print(f"Suspicious Regions ({len(analysis.suspicious_regions)}):")
            for i, region in enumerate(analysis.suspicious_regions, 1):
                print(f"  {i}. {region.location} ({region.confidence})")
                print(f"     {region.description[:80]}...")
        
        print(f"\n{'─'*60}")
        print(f"Analysis:")
        print(f"{analysis.detailed_analysis[:300]}...")
        
        print(f"\n{'='*60}\n")
    
    def analyze_batch(self, image_paths: List[Union[str, Path]]) -> List[VLMAnalysisResult]:
        """
        Analyze multiple images
        
        Args:
            image_paths: List of paths to check images
            
        Returns:
            List of VLMAnalysisResult objects
        """
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{total}")
            print(f"{'='*60}")
            
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
        
        return results
    
    def analyze_batch_parallel(
        self, 
        image_paths: List[Union[str, Path]], 
        max_workers: Optional[int] = None
    ) -> List[VLMAnalysisResult]:
        """
        Analyze multiple images in parallel using thread pool
        
        Args:
            image_paths: List of paths to check images
            max_workers: Maximum number of parallel workers. If None, defaults to min(32, len(image_paths))
            
        Returns:
            List of VLMAnalysisResult objects in the same order as input paths
        """
        total = len(image_paths)
        
        if max_workers is None:
            max_workers = min(32, total)
        
        print(f"\n{'='*60}")
        print(f"Starting parallel analysis of {total} images")
        print(f"Using {max_workers} parallel workers")
        print(f"{'='*60}\n")
        
        # Dictionary to store results with their original index
        results_dict = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and map them to their index
            future_to_index = {
                executor.submit(self._analyze_single_image_safe, image_path, i, total): i 
                for i, image_path in enumerate(image_paths)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_dict[index] = result
                except Exception as e:
                    print(f"Unexpected error in parallel processing: {e}")
        
        # Return results in original order
        results = [results_dict[i] for i in range(len(image_paths)) if i in results_dict]
        
        print(f"\n{'='*60}")
        print(f"Parallel analysis complete: {len(results)}/{total} successful")
        print(f"{'='*60}\n")
        
        return results
    
    def _analyze_single_image_safe(
        self, 
        image_path: Union[str, Path], 
        index: int, 
        total: int
    ) -> Optional[VLMAnalysisResult]:
        """
        Safely analyze a single image with error handling for parallel processing
        
        Args:
            image_path: Path to the check image
            index: Index of this image in the batch
            total: Total number of images being processed
            
        Returns:
            VLMAnalysisResult or None if error occurs
        """
        try:
            print(f"[{index + 1}/{total}] 🔍 Analyzing: {Path(image_path).name}")
            result = self.analyze_image(image_path)
            print(f"[{index + 1}/{total}] ✅ Completed: {Path(image_path).name}")
            return result
        except Exception as e:
            print(f"[{index + 1}/{total}] ❌ Error analyzing {image_path}: {e}")
            return None
    
    def analyze_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.[jp][pn]g"
    ) -> List[VLMAnalysisResult]:
        """
        Analyze all images in a directory
        
        Args:
            directory: Path to directory containing check images
            pattern: Glob pattern for image files
            
        Returns:
            List of VLMAnalysisResult objects
        """
        directory = Path(directory)
        image_paths = list(directory.glob(pattern))
        
        print(f"\n Found {len(image_paths)} images in {directory}")
        
        return self.analyze_batch(image_paths)
    
    def analyze_directory_parallel(
        self,
        directory: Union[str, Path],
        pattern: str = "*.[jp][pn]g",
        max_workers: Optional[int] = None
    ) -> List[VLMAnalysisResult]:
        """
        Analyze all images in a directory in parallel
        
        Args:
            directory: Path to directory containing check images
            pattern: Glob pattern for image files
            max_workers: Maximum number of parallel workers. If None, defaults to min(32, num_images)
            
        Returns:
            List of VLMAnalysisResult objects
        """
        directory = Path(directory)
        image_paths = list(directory.glob(pattern))
        
        print(f"\n📁 Found {len(image_paths)} images in {directory}")
        
        return self.analyze_batch_parallel(image_paths, max_workers=max_workers)
