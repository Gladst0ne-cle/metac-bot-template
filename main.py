import argparse
import asyncio
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    is_real_env,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)
from free_research import build_free_research_report

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

_PAID_LLM_ENV_VARS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")

_FORECAST_LENSES: tuple[str, str, str] = (
    "Analytical Lens (Balanced Synthesis): Give equal weight to outside-view base rates and inside-view primary evidence, explicitly checking status-quo inertia.",
    "Analytical Lens (Skeptical Red-Team): Anchor strictly on historical base rates and institutional friction; challenge hype and detail concrete reasons consensus headlines could disappoint or slip past the deadline.",
    "Analytical Lens (Signal Update): Prioritize the freshest primary facts, official data trends, and aligned prediction-market benchmarks (strictly after checking wording and closing-date alignment).",
)


def _clip_text(text: str | None, max_chars: int) -> str:
    """Trim oversized prompt text while keeping both opening and closing rules."""
    s = (text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    marker = "\n...[truncated]...\n"
    if max_chars <= len(marker) + 10:
        return s[:max_chars]
    head = max_chars * 2 // 3
    tail = max_chars - head - len(marker)
    return f"{s[:head]}{marker}{s[-tail:]}" if tail > 0 else s[:max_chars]


_BINARY_PROB_TAIL_RE = re.compile(
    r"(?i)probability\s*[:=]\s*\*{0,2}\s*(\d+(?:\.\d+)?)\s*%\s*\*{0,2}"
)


def _extract_binary_probability_from_tail(reasoning: str) -> float | None:
    """Parse an unambiguous final 'Probability: ZZ%' value near the end of reasoning."""
    tail = (reasoning or "").strip()[-260:]
    matches = _BINARY_PROB_TAIL_RE.findall(tail)
    if len(matches) != 1:
        return None
    val = float(matches[0])
    if 0.0 <= val <= 100.0:
        return val / 100.0
    return None


def _aggregate_binary_log_odds(
    predictions: list[float],
    clip_min: float = 0.015,
    clip_max: float = 0.985,
) -> float:
    """Average clipped logits across samples to calibrate log score and prevent 0/1 overconfidence."""
    if not predictions:
        raise ValueError("Cannot aggregate empty list of binary predictions")
    logits = []
    for raw_p in predictions:
        p = max(clip_min, min(clip_max, float(raw_p)))
        logits.append(math.log(p / (1.0 - p)))
    mean_logit = sum(logits) / len(logits)
    pooled = 1.0 / (1.0 + math.exp(-mean_logit))
    return max(clip_min, min(clip_max, pooled))


def _aggregate_multiple_choice_log_pool(
    predictions: list[PredictedOptionList],
) -> PredictedOptionList:
    """Geometric-mean (logarithmic opinion pool) aggregation across multiple-choice samples."""
    if not predictions:
        raise ValueError("Cannot aggregate empty list of multiple-choice predictions")
    option_names = [opt.option_name for opt in predictions[0].predicted_options]
    if not option_names:
        raise ValueError("Multiple-choice prediction has no options")
    expected_set = set(option_names)
    if len(expected_set) != len(option_names):
        raise ValueError("Duplicate option names in multiple-choice prediction")

    maps: list[dict[str, float]] = []
    for pred_list in predictions:
        if len(pred_list.predicted_options) != len(option_names):
            raise ValueError("Inconsistent option counts across samples")
        opt_map = {opt.option_name: opt.probability for opt in pred_list.predicted_options}
        if set(opt_map.keys()) != expected_set:
            raise ValueError("Inconsistent option names across samples")
        maps.append(opt_map)

    n_opts = len(option_names)
    eps = min(0.005, 0.25 / max(1, n_opts))
    geo_means = [
        math.exp(sum(math.log(max(eps, float(m[name]))) for m in maps) / len(maps))
        for name in option_names
    ]
    total = sum(geo_means)
    if total <= 0.0 or not math.isfinite(total):
        geo_means = [1.0 / n_opts] * n_opts

    floor = 0.001
    if floor * n_opts < 1.0:
        probs = PredictedOptionList._normalize_with_minimum(
            geo_means, minimum_probability=floor
        )
    else:
        s = sum(geo_means)
        probs = [g / s for g in geo_means]

    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=name, probability=p)
            for name, p in zip(option_names, probs)
        ]
    )


def _build_runtime_config() -> tuple[
    dict[str, str | GeneralLlm | None] | None, int, int, bool
]:
    """
    Select LLMs and sampling parameters based on available credentials.
    Returns (llms, predictions_per_research_report, validation_samples, enable_summarize).
    """
    if any(is_real_env(k) for k in _PAID_LLM_ENV_VARS):
        return (None, 5, 2, True)

    if is_real_env("ASKNEWS_CLIENT_ID") and is_real_env("ASKNEWS_SECRET"):
        researcher: str | GeneralLlm = "asknews/news-summaries"
    elif is_real_env("PERPLEXITY_API_KEY"):
        researcher = GeneralLlm(model="perplexity/sonar-pro", temperature=0.1)
    else:
        researcher = "free-news-rss"

    if is_real_env("GEMINI_API_KEY"):
        has_groq = is_real_env("GROQ_API_KEY")
        default_fallbacks = [
            "gemini/gemini-3-flash-preview",
            "gemini/gemini-3.8-flash",
            "gemini/gemini-3.5-flash-lite",
            "gemini/gemini-3.1-flash-lite",
        ]
        lite_fallbacks = ["gemini/gemini-3.1-flash-lite"]
        if has_groq:
            default_fallbacks.extend(
                ["groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"]
            )
            lite_fallbacks.extend(
                ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"]
            )
        return (
            {
                "default": GeneralLlm(
                    model="gemini/gemini-3.6-flash",
                    timeout=90,
                    allowed_tries=4,
                    fallbacks=default_fallbacks,
                ),
                "summarizer": GeneralLlm(
                    model="gemini/gemini-3.5-flash-lite",
                    timeout=60,
                    allowed_tries=4,
                    fallbacks=lite_fallbacks,
                ),
                "researcher": researcher,
                "parser": GeneralLlm(
                    model="gemini/gemini-3.5-flash-lite",
                    timeout=60,
                    allowed_tries=4,
                    fallbacks=list(lite_fallbacks),
                ),
            },
            3,
            1,
            False,
        )

    if is_real_env("GROQ_API_KEY"):
        return (
            {
                "default": GeneralLlm(
                    model="groq/openai/gpt-oss-120b",
                    timeout=90,
                    allowed_tries=4,
                    fallbacks=["groq/openai/gpt-oss-20b"],
                ),
                "summarizer": GeneralLlm(
                    model="groq/openai/gpt-oss-20b",
                    timeout=60,
                    allowed_tries=4,
                    fallbacks=["groq/openai/gpt-oss-120b"],
                ),
                "researcher": researcher,
                "parser": GeneralLlm(
                    model="groq/openai/gpt-oss-20b",
                    timeout=60,
                    allowed_tries=4,
                    fallbacks=["groq/openai/gpt-oss-120b"],
                ),
            },
            3,
            1,
            False,
        )

    return (None, 5, 2, True)


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2
    as_of_date: str | None = None
    _min_llm_call_interval: float = 3.0

    def _today_str(self) -> str:
        return self.as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_prediction_limiter(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        limiters = getattr(self, "_loop_limiters", None)
        if limiters is None:
            limiters = {}
            self._loop_limiters = limiters
        sem = limiters.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(1)
            limiters[loop] = sem
        return sem

    def _next_lens_instruction(self, question: MetaculusQuestion) -> str:
        counters = getattr(self, "_lens_counters", None)
        if counters is None:
            counters = {}
            self._lens_counters = counters
        key = str(getattr(question, "id_of_question", None) or question.question_text)
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        return _FORECAST_LENSES[idx % len(_FORECAST_LENSES)]

    async def _invoke_default_llm(self, prompt: str) -> str:
        async with self._get_prediction_limiter():
            interval = max(0.0, float(getattr(self, "_min_llm_call_interval", 0.0)))
            last_ts = float(getattr(self, "_last_default_call_ts", 0.0))
            if last_ts > 0.0 and interval > 0.0:
                wait = interval - (time.monotonic() - last_ts)
                if wait > 0.0:
                    await asyncio.sleep(wait)
            try:
                return await self.get_llm("default", "llm").invoke(prompt)
            finally:
                self._last_default_call_ts = time.monotonic()

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if isinstance(question, BinaryQuestion):
            return _aggregate_binary_log_odds([float(p) for p in predictions])
        if isinstance(question, MultipleChoiceQuestion):
            try:
                return _aggregate_multiple_choice_log_pool(predictions)  # type: ignore[arg-type]
            except ValueError as exc:
                logger.warning(
                    f"Falling back to default MC aggregation for {question.page_url}: {exc}"
                )
        return await super()._aggregate_predictions(predictions, question)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = self._get_research_prompt(question, researcher)

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif researcher == "free-news-rss":
                research = await build_free_research_report(
                    question_text=question.question_text,
                    resolution_criteria=question.resolution_criteria or "",
                    fine_print=question.fine_print or "",
                    background_info=question.background_info or "",
                    planner_llm=self.get_llm("parser", "llm"),
                    as_of_date=self.as_of_date,
                )
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        lens_note = self._next_lens_instruction(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {_clip_text(question.background_info, 1600)}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {_clip_text(question.resolution_criteria, 2000)}

            {_clip_text(question.fine_print, 1000)}


            Your research assistant says:
            {_clip_text(research, 5200)}

            Today is {self._today_str()}.

            {lens_note}

            Follow disciplined superforecasting methodology before giving your probability:
            (a) Resolution & time horizon: summarize exact resolution criteria, edge cases, and time left until resolution.
            (b) Outside view & market anchor: estimate the historical base rate for this class of event. If external prediction markets appear in the research dossier, use them as a rough starting anchor ONLY after adjusting for differences in wording, resolution criteria, or closing dates.
            (c) Inside view update: weigh the latest primary data, official macro figures, and recent news headlines above (distinguishing confirmed facts from speculation).
            (d) Pre-mortem: state the strongest realistic case for No and the strongest realistic case for Yes.
            (e) Calibration check: avoid extreme probabilities (<3% or >97%) unless mechanically or legally near-certain, remembering the status quo carries inertia.
            {self._get_conditional_disclaimer_if_necessary(question)}

            Keep your reasoning structured and concise (under ~350 words).
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._invoke_default_llm(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        raw_pred = _extract_binary_probability_from_tail(reasoning)
        if raw_pred is None:
            parsing_instructions = clean_indents(
                f"""
                The text given to you is trying to give a probability forecast for a binary question.
                {self._create_resolved_question_parsing_message()}
                """
            )
            try:
                binary_prediction: BinaryPrediction = await structure_output(
                    reasoning,
                    BinaryPrediction,
                    model=self.get_llm("parser", "llm"),
                    num_validation_samples=self._structure_output_validation_samples,
                    additional_instructions=parsing_instructions,
                )
                raw_pred = float(binary_prediction.prediction_in_decimal)
            except ValueError:
                valid_vals = [
                    float(m)
                    for m in _BINARY_PROB_TAIL_RE.findall(reasoning or "")
                    if 0.0 <= float(m) <= 100.0
                ]
                if not valid_vals:
                    raise
                logger.warning(
                    f"structure_output rejected binary reasoning for {question.page_url}; "
                    "falling back to explicit Probability match."
                )
                raw_pred = valid_vals[-1] / 100.0

        decimal_pred = max(0.01, min(0.99, raw_pred))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        lens_note = self._next_lens_instruction(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {_clip_text(question.background_info, 1600)}

            {_clip_text(question.resolution_criteria, 2000)}

            {_clip_text(question.fine_print, 1000)}


            Your research assistant says:
            {_clip_text(research, 5200)}

            Today is {self._today_str()}.

            {lens_note}

            Follow disciplined superforecasting methodology before assigning probabilities:
            (a) Resolution & time horizon: audit exact option definitions, mutual exclusivity, and time remaining until resolution.
            (b) Outside view & market anchor: estimate base rates across options; if external prediction markets appear in the research dossier, anchor loosely only after checking wording and scope alignment.
            (c) Inside view update: weigh the primary sources, official data, and recent headlines, plus describe an unexpected scenario.
            (d) Calibration check: weight status quo continuity, reserve nonzero tail probability on plausible upset/residual options, and make sure the probabilities across all options {question.options} sum cleanly to 100%.

            {self._get_conditional_disclaimer_if_necessary(question)}

            Keep your reasoning structured and concise (under ~350 words).
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self._invoke_default_llm(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        lens_note = self._next_lens_instruction(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {_clip_text(question.background_info, 1600)}

            {_clip_text(question.resolution_criteria, 2000)}

            {_clip_text(question.fine_print, 1000)}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {_clip_text(research, 5200)}

            Today is {self._today_str()}.

            {lens_note}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase strictly monotonically (`Percentile 10 < Percentile 20 < Percentile 40 < Percentile 60 < Percentile 80 < Percentile 90`).

            Follow disciplined quantitative superforecasting before setting percentiles:
            (a) Baseline & hard floor/anchor: identify the latest observed value or cumulative year-to-date count from primary sources or FRED data above, plus time remaining until resolution.
            (b) Status quo & trend extrapolation: what happens if current rate/trend persists over the remaining horizon?
            (c) Expert / market expectations and historical variance over similar horizons.
            (d) Tail scenarios & calibration: describe a concrete downside surprise (driving P10/P20) and upside shock (driving P80/P90), ensuring genuinely wide 90/10 intervals to protect log score against unexpected tails.

            {self._get_conditional_disclaimer_if_necessary(question)}

            Keep your reasoning structured and concise (under ~350 words).
            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_default_llm(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        lens_note = self._next_lens_instruction(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {_clip_text(question.background_info, 1600)}

            {_clip_text(question.resolution_criteria, 2000)}

            {_clip_text(question.fine_print, 1000)}

            Your research assistant says:
            {_clip_text(research, 5200)}

            Today is {self._today_str()}.

            {lens_note}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase strictly (`Percentile 10 < Percentile 20 < Percentile 40 < Percentile 60 < Percentile 80 < Percentile 90`).

            Follow disciplined chronological superforecasting before setting percentiles:
            (a) Baseline & calendar anchor: note current progress/milestones from primary sources or news above, scheduled deadlines, and time elapsed.
            (b) Outside view: historical timelines and base rates for comparable milestones or regulatory/political processes.
            (c) Expert / market expectations and potential acceleration vs delay bottlenecks.
            (d) Tail scenarios & calibration: describe an early-resolution surprise (P10/P20) and a prolonged-delay scenario (P80/P90), keeping a wide 90/10 interval against unknown delays.

            {self._get_conditional_disclaimer_if_necessary(question)}

            Keep your reasoning structured and concise (under ~350 words).
            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_default_llm(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. When OPENROUTER/OPENAI/ANTHROPIC keys are absent,
    # _build_runtime_config() selects free Gemini or Groq + RSS.
    llms_cfg, preds_per_report, validation_samples, enable_summarize = (
        _build_runtime_config()
    )
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=preds_per_report,
        use_research_summary_to_forecast=False,
        enable_summarize_research=enable_summarize,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms_cfg,
    )
    template_bot._structure_output_validation_samples = validation_samples
    template_bot._min_llm_call_interval = 0.0 if llms_cfg is None else 3.0

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
