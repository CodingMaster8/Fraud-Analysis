"""
Pydantic models for VLM output validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal


FraudType = Literal[
    "digital_manipulation",
    "physical_alteration", 
    "tipex_whiteout",
    "pen_modification",
    "photoshop_edit",
    "suspicious_artifacts",
    "no_fraud",
    "screen_photo"
]

ConfidenceLevel = Literal[
    "very_low",
    "low", 
    "medium",
    "high",
    "very_high"
]


class SuspiciousRegion(BaseModel):
    """A region in the image that appears suspicious"""
    
    description: str = Field(
        ...,
        description="Description of what appears suspicious in this region",
        min_length=10,
        max_length=500
    )
    
    fraud_type: FraudType = Field(
        ...,
        description="Type of fraud suspected in this region"
    )
    
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level for this suspicious region"
    )
    
    location: str = Field(
        ...,
        description="Location description (e.g., 'top-right corner', 'near the payee field', 'amount area')",
        min_length=5,
        max_length=200
    )


class VLMFraudAnalysis(BaseModel):
    """
    Structured output from VLM fraud detection analysis
    This model ensures the VLM provides consistent, validated responses
    """
    
    is_fraudulent: bool = Field(
        ...,
        description="Whether the check appears to be fraudulent"
    )
    
    overall_confidence: ConfidenceLevel = Field(
        ...,
        description="Overall confidence in the fraud determination"
    )
    
    fraud_likelihood_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Numerical fraud likelihood score from 0-100"
    )
    
    primary_fraud_types: List[FraudType] = Field(
        default_factory=list,
        description="Primary types of fraud detected",
        min_length=0,
        max_length=5
    )
    
    suspicious_regions: List[SuspiciousRegion] = Field(
        default_factory=list,
        description="Specific regions in the image that appear suspicious",
        max_length=10
    )
    
    detailed_analysis: str = Field(
        ...,
        description="Detailed explanation of the analysis and findings",
        min_length=50,
        max_length=2000
    )
    
    key_indicators: List[str] = Field(
        default_factory=list,
        description="Key visual indicators that led to the conclusion",
        min_length=1,
        max_length=10
    )
    
    pixel_level_observations: Optional[str] = Field(
        default=None,
        description="Specific pixel-level observations (compression artifacts, noise patterns, edge inconsistencies)",
        max_length=1000
    )
    
    recommendation: Literal["accept", "review", "reject"] = Field(
        ...,
        description="Recommended action based on the analysis"
    )
    
    @field_validator("fraud_likelihood_score")
    @classmethod
    def validate_score_consistency(cls, v, info):
        """Ensure fraud likelihood score is consistent with is_fraudulent flag"""
        is_fraudulent = info.data.get("is_fraudulent")
        
        if is_fraudulent and v < 30:
            raise ValueError(
                f"Fraud likelihood score ({v}) is too low for a fraudulent classification"
            )
        if not is_fraudulent and v > 70:
            raise ValueError(
                f"Fraud likelihood score ({v}) is too high for a non-fraudulent classification"
            )
        
        return v
    
    @field_validator("primary_fraud_types")
    @classmethod
    def validate_fraud_types(cls, v, info):
        """Ensure fraud types are consistent with is_fraudulent flag"""
        is_fraudulent = info.data.get("is_fraudulent")
        
        if is_fraudulent:
            # Remove NO_FRAUD if other fraud types are present
            fraud_types = [ft for ft in v if ft != "no_fraud"]
            if not fraud_types:
                raise ValueError("Fraudulent checks must have at least one fraud type specified")
            return fraud_types
        else:
            # Non-fraudulent should only have NO_FRAUD or empty list
            return ["no_fraud"] if v else []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "is_fraudulent": self.is_fraudulent,
            "overall_confidence": self.overall_confidence,
            "fraud_likelihood_score": self.fraud_likelihood_score,
            "primary_fraud_types": self.primary_fraud_types,
            "suspicious_regions": [
                {
                    "description": sr.description,
                    "fraud_type": sr.fraud_type,
                    "confidence": sr.confidence,
                    "location": sr.location
                }
                for sr in self.suspicious_regions
            ],
            "detailed_analysis": self.detailed_analysis,
            "key_indicators": self.key_indicators,
            "pixel_level_observations": self.pixel_level_observations,
            "recommendation": self.recommendation
        }