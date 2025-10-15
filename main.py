"""
AI Lead Enrichment Engine - Backend API
Built for Caprae Capital Challenge
Time: 5 hours | Quality-First Approach
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
# Initialize logging early so fallback messages are visible
logging.basicConfig(level=logging.INFO)

# Avoid using pydantic's EmailStr here to prevent pydantic from importing
# the optional `email_validator` package at runtime. Use plain str and
# log a note for developers.
EmailStr = str  # type: ignore
logging.warning("Using plain 'str' for emails. For strict validation install: pip install 'pydantic[email]'")

from typing import List, Optional, Dict
import httpx
import asyncio
from datetime import datetime
import os
import functools

app = FastAPI(title="Lead Enrichment API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client with compatibility for different openai package versions
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_SDK = "unknown"
try:
    # Newer OpenAI SDK exposes OpenAI class
    from openai import OpenAI as _OpenAI

    client = _OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_SDK = "new"
    logging.info("Using new OpenAI SDK wrapper")
except Exception:
    # Fallback to legacy openai package interface
    try:
        import openai as client  # type: ignore
        if OPENAI_API_KEY:
            client.api_key = OPENAI_API_KEY
        OPENAI_SDK = "legacy"
        logging.info("Using legacy openai package")
    except Exception:
        client = None
        OPENAI_SDK = "none"
        logging.warning("OpenAI package not available; AI features will be disabled")

# Data Models
class Lead(BaseModel):
    company: str
    contact: str
    title: str
    email: EmailStr
    website: str
    industry: Optional[str] = None
    employees: Optional[str] = None
    revenue: Optional[str] = None

class ValidationResult(BaseModel):
    email_valid: bool
    company_exists: bool
    recent_activity: bool
    tech_stack: List[str]
    funding_round: Optional[str]
    confidence: int

class EmailDraft(BaseModel):
    subject: str
    body: str
    personalization_factors: List[str]

class EnrichedLead(BaseModel):
    lead: Lead
    validation: ValidationResult
    score: int
    email_draft: EmailDraft
    enriched_at: datetime

# Multi-Source Data Validation
class DataValidator:
    """
    Multi-source validation engine that cross-references data
    from multiple APIs to ensure accuracy
    """
    
    def __init__(self):
        self.sources = {
            'hunter': 'https://api.hunter.io/v2',
            'clearbit': 'https://company.clearbit.com/v2',
            'builtwith': 'https://api.builtwith.com/v20'
        }
    
    async def validate_email(self, email: str) -> Dict:
        """Validate email using Hunter.io API"""
        # Simulated validation - replace with actual API call
        # In production: Use Hunter.io Email Verifier
        domain = email.split('@')[1]
        
        return {
            'valid': '@' in email and '.' in domain,
            'format_valid': True,
            'mx_found': True,
            'smtp_valid': True
        }
    
    async def enrich_company(self, domain: str) -> Dict:
        """Enrich company data using Clearbit"""
        # Simulated enrichment - replace with actual API call
        # In production: Use Clearbit Enrichment API
        
        return {
            'name': domain.split('.')[0].title(),
            'domain': domain,
            'category': {
                'industry': 'Technology',
                'sector': 'Software'
            },
            'metrics': {
                'employees': '50-200',
                'estimated_revenue': '$5M-$10M',
                'raised': '$5M'
            },
            'tech': ['React', 'AWS', 'Salesforce', 'HubSpot']
        }
    
    async def check_web_activity(self, domain: str) -> Dict:
        """Check recent web activity and technologies"""
        # Simulated check - replace with actual scraping
        # In production: Use BuiltWith or custom scraper
        
        return {
            'last_updated': datetime.now().isoformat(),
            'active': True,
            'technologies': ['React', 'Next.js', 'Vercel'],
            'social_presence': True
        }
    
    async def validate_lead(self, lead: Lead) -> ValidationResult:
        """
        Perform multi-source validation with confidence scoring
        """
        # Run all validations in parallel
        email_result, company_result, activity_result = await asyncio.gather(
            self.validate_email(lead.email),
            self.enrich_company(lead.website),
            self.check_web_activity(lead.website)
        )
        
        # Calculate confidence score
        confidence = 0
        if email_result['valid']: confidence += 25
        if email_result['mx_found']: confidence += 15
        if company_result: confidence += 30
        if activity_result['active']: confidence += 30
        
        return ValidationResult(
            email_valid=email_result['valid'],
            company_exists=bool(company_result),
            recent_activity=activity_result['active'],
            tech_stack=company_result.get('tech', []),
            funding_round=self._extract_funding(company_result),
            confidence=min(confidence, 100)
        )
    
    def _extract_funding(self, company_data: Dict) -> str:
        """Extract funding information from company data"""
        raised = company_data.get('metrics', {}).get('raised', '0')
        
        if 'M' in raised:
            amount = float(raised.replace('$', '').replace('M', ''))
            if amount < 2:
                return 'Seed'
            elif amount < 10:
                return 'Series A'
            elif amount < 30:
                return 'Series B'
            else:
                return 'Series C+'
        return 'Not Available'

# Lead Scoring Algorithm
class LeadScorer:
    """
    Advanced lead scoring using firmographic, technographic,
    and behavioral signals
    """
    
    def __init__(self):
        self.weights = {
            'company_size': 0.25,
            'revenue': 0.20,
            'industry_fit': 0.15,
            'data_quality': 0.20,
            'tech_stack': 0.10,
            'recent_activity': 0.10
        }
    
    def calculate_score(self, lead: Lead, validation: ValidationResult) -> int:
        """
        Calculate composite lead score (0-100)
        """
        score = 0
        
        # Company size scoring
        size_score = self._score_company_size(lead.employees or '')
        score += size_score * self.weights['company_size'] * 100
        
        # Revenue scoring
        revenue_score = self._score_revenue(lead.revenue or '')
        score += revenue_score * self.weights['revenue'] * 100
        
        # Industry fit scoring
        industry_score = self._score_industry(lead.industry or '')
        score += industry_score * self.weights['industry_fit'] * 100
        
        # Data quality scoring
        quality_score = validation.confidence / 100
        score += quality_score * self.weights['data_quality'] * 100
        
        # Tech stack scoring
        tech_score = self._score_tech_stack(validation.tech_stack)
        score += tech_score * self.weights['tech_stack'] * 100
        
        # Recent activity scoring
        activity_score = 1.0 if validation.recent_activity else 0.3
        score += activity_score * self.weights['recent_activity'] * 100
        
        return min(int(score), 100)
    
    def _score_company_size(self, employees: str) -> float:
        """Score based on company size"""
        if '200-500' in employees or '500+' in employees:
            return 1.0
        elif '50-200' in employees:
            return 0.8
        elif '10-50' in employees:
            return 0.6
        return 0.4
    
    def _score_revenue(self, revenue: str) -> float:
        """Score based on revenue"""
        if '$10M' in revenue or '$50M' in revenue:
            return 1.0
        elif '$5M' in revenue:
            return 0.8
        elif '$1M' in revenue:
            return 0.6
        return 0.4
    
    def _score_industry(self, industry: str) -> float:
        """Score based on industry fit"""
        high_fit = ['SaaS', 'Technology', 'Software', 'Fintech']
        medium_fit = ['E-commerce', 'Healthcare', 'Consulting']
        
        if any(i in industry for i in high_fit):
            return 1.0
        elif any(i in industry for i in medium_fit):
            return 0.7
        return 0.5
    
    def _score_tech_stack(self, tech_stack: List[str]) -> float:
        """Score based on technology stack alignment"""
        relevant_tech = ['React', 'Salesforce', 'HubSpot', 'AWS', 'Python']
        matches = sum(1 for tech in tech_stack if tech in relevant_tech)
        return min(matches / 3, 1.0)

# AI Email Generator
class EmailGenerator:
    """
    GPT-4 powered email personalization engine
    """
    
    def __init__(self):
        self.templates = [
            'problem_solution',
            'social_proof',
            'value_proposition'
        ]
    
    async def generate_email(
        self, 
        lead: Lead, 
        validation: ValidationResult,
        score: int
    ) -> EmailDraft:
        """
        Generate personalized email using GPT-4
        """
        
        # Build context for AI
        context = self._build_context(lead, validation, score)
        
        # Call GPT-4 API if available
        if client is None:
            logging.info("OpenAI client not configured; using template fallback for email")
            subject, body = self._generate_template_email(lead, validation)
        else:
            try:
                # Support both new and legacy SDK shapes
                if OPENAI_SDK == "new":
                    # new SDK: client.chat.completions.create
                    # run in thread executor because it's blocking
                    loop = asyncio.get_running_loop()
                    fn = functools.partial(
                        client.chat.completions.create,
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert B2B sales email writer. Create highly personalized, concise emails that reference company details, lead with value, include a clear CTA, are under 150 words, and feel human."
                            },
                            {
                                "role": "user",
                                "content": f"Create a personalized sales email for:\n\nCompany: {lead.company}\nContact: {lead.contact} ({lead.title})\nIndustry: {lead.industry}\nTech Stack: {', '.join(validation.tech_stack[:3])}\nFunding: {validation.funding_round}\nCompany Size: {lead.employees}\n\nProduct: SaaSquatch Leads - AI-powered lead generation tool\nValue Prop: 40% more leads, 60% lower cost than ZoomInfo\nPrice: $19-199/month vs $995+/month competitors\n\nMake it specific to their company and tech stack."
                            }
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )

                    response = await loop.run_in_executor(None, fn)
                    # new SDK response parsing
                    email_content = None
                    try:
                        email_content = response.choices[0].message.content
                    except Exception:
                        # fallback: some SDKs return text directly
                        email_content = getattr(response, 'text', str(response))

                    subject, body = self._parse_email(email_content)
                else:
                    # legacy openai package
                    # openai.ChatCompletion.create(...) returns dict-like
                    resp = client.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert B2B sales email writer. Create concise, personalized emails."},
                            {"role": "user", "content": f"Create a personalized sales email for {lead.company} (Contact: {lead.contact})."}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    # extract text
                    if isinstance(resp, dict):
                        # legacy dict response
                        email_content = resp.get('choices', [])[0].get('message', {}).get('content') if resp.get('choices') else ''
                    else:
                        email_content = getattr(resp, 'choices', [])[0].get('message', {}).get('content', '')

                    subject, body = self._parse_email(email_content)
            except Exception as e:
                logging.exception("OpenAI request failed, falling back to template")
                subject, body = self._generate_template_email(lead, validation)
        
        return EmailDraft(
            subject=subject,
            body=body,
            personalization_factors=self._get_personalization_factors(lead, validation)
        )
    
    def _build_context(self, lead: Lead, validation: ValidationResult, score: int) -> str:
        """Build context string for AI"""
        return f"""
        Company: {lead.company}
        Contact: {lead.contact}
        Title: {lead.title}
        Industry: {lead.industry}
        Size: {lead.employees}
        Tech: {', '.join(validation.tech_stack)}
        Score: {score}/100
        """
    
    def _parse_email(self, content: str) -> tuple:
        """Parse AI response into subject and body"""
        lines = content.strip().split('\n')
        
        # Find subject line
        subject = ""
        body_start = 0
        
        for i, line in enumerate(lines):
            if line.lower().startswith('subject:'):
                subject = line.replace('Subject:', '').replace('subject:', '').strip()
                body_start = i + 1
                break
        
        if not subject and lines:
            subject = lines[0]
            body_start = 1
        
        body = '\n'.join(lines[body_start:]).strip()
        
        return subject, body
    
    def _generate_template_email(self, lead: Lead, validation: ValidationResult) -> tuple:
        """Fallback template-based email generation"""
        subject = f"{lead.company} + SaaSquatch: Boost Lead Gen by 40%"
        
        first_name = lead.contact.split()[0]
        tech = validation.tech_stack[0] if validation.tech_stack else "modern tech stack"
        
        body = f"""Hi {first_name},

I noticed {lead.company} is using {tech}. Given your {validation.funding_round} stage, you're likely scaling your sales team.

At SaaSquatch, we help {lead.industry} companies generate 40% more qualified leads while cutting costs by 60% vs ZoomInfo.

Quick question: Happy with your current lead data accuracy?

Worth a 15-min chat?

Best,
[Your Name]

P.S. We helped a similar company increase response rates by 6x using AI personalization."""
        
        return subject, body
    
    def _get_personalization_factors(self, lead: Lead, validation: ValidationResult) -> List[str]:
        """Extract personalization factors used"""
        factors = [
            f"Company name: {lead.company}",
            f"Contact name: {lead.contact}",
            f"Industry: {lead.industry}"
        ]
        
        if validation.tech_stack:
            factors.append(f"Tech stack: {', '.join(validation.tech_stack[:2])}")
        
        if validation.funding_round:
            factors.append(f"Funding stage: {validation.funding_round}")
        
        return factors

# API Endpoints
validator = DataValidator()
scorer = LeadScorer()
email_gen = EmailGenerator()

@app.post("/api/enrich", response_model=EnrichedLead)
async def enrich_lead(lead: Lead):
    """
    Main enrichment endpoint:
    1. Validates lead data from multiple sources
    2. Calculates lead score
    3. Generates personalized email
    """
    try:
        # Step 1: Multi-source validation
        validation = await validator.validate_lead(lead)
        
        # Step 2: Calculate lead score
        score = scorer.calculate_score(lead, validation)
        
        # Step 3: Generate personalized email
        email = await email_gen.generate_email(lead, validation, score)
        
        return EnrichedLead(
            lead=lead,
            validation=validation,
            score=score,
            email_draft=email,
            enriched_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-enrich")
async def batch_enrich(leads: List[Lead]):
    """
    Batch enrichment for multiple leads
    """
    tasks = [enrich_lead(lead) for lead in leads]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        'total': len(leads),
        'successful': sum(1 for r in results if not isinstance(r, Exception)),
        'failed': sum(1 for r in results if isinstance(r, Exception)),
        'results': [r for r in results if not isinstance(r, Exception)]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'validator': 'operational',
            'scorer': 'operational',
            'email_generator': 'operational'
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)