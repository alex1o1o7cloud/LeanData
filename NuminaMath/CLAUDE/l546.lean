import Mathlib

namespace NUMINAMATH_CALUDE_speed_limit_representation_l546_54644

-- Define the speed limit
def speed_limit : ℝ := 70

-- Define a vehicle's speed
variable (v : ℝ)

-- Theorem stating that v ≤ speed_limit correctly represents the speed limit instruction
theorem speed_limit_representation : 
  (v ≤ speed_limit) ↔ (v ≤ speed_limit ∧ ¬(v > speed_limit)) :=
by sorry

end NUMINAMATH_CALUDE_speed_limit_representation_l546_54644


namespace NUMINAMATH_CALUDE_three_propositions_true_l546_54678

-- Define the properties of functions
def IsConstant (f : ℝ → ℝ) : Prop := ∃ C : ℝ, ∀ x : ℝ, f x = C
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def HasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Define the propositions
def Prop1 (f : ℝ → ℝ) : Prop := IsConstant f → (IsOdd f ∧ IsEven f)
def Prop2 (f : ℝ → ℝ) : Prop := IsOdd f → HasInverse f
def Prop3 (f : ℝ → ℝ) : Prop := IsOdd f → IsOdd (λ x => Real.sin (f x))
def Prop4 (f g : ℝ → ℝ) : Prop := IsOdd f → IsEven g → IsEven (g ∘ f)

-- The main theorem
theorem three_propositions_true :
  ∃ (f g : ℝ → ℝ),
    (Prop1 f ∧ ¬Prop2 f ∧ Prop3 f ∧ Prop4 f g) ∨
    (Prop1 f ∧ Prop2 f ∧ Prop3 f ∧ ¬Prop4 f g) ∨
    (Prop1 f ∧ Prop2 f ∧ ¬Prop3 f ∧ Prop4 f g) ∨
    (¬Prop1 f ∧ Prop2 f ∧ Prop3 f ∧ Prop4 f g) :=
  sorry

end NUMINAMATH_CALUDE_three_propositions_true_l546_54678


namespace NUMINAMATH_CALUDE_hilt_family_fitness_l546_54617

-- Define conversion rates
def yards_per_mile : ℝ := 1760
def miles_per_km : ℝ := 0.621371

-- Define Mrs. Hilt's activities
def mrs_hilt_running : List ℝ := [3, 2, 7]
def mrs_hilt_swimming : List ℝ := [1760, 0, 1000]
def mrs_hilt_biking : List ℝ := [0, 6, 3, 10]

-- Define Mr. Hilt's activities
def mr_hilt_biking : List ℝ := [5, 8]
def mr_hilt_running : List ℝ := [4]
def mr_hilt_swimming : List ℝ := [2000]

-- Theorem statement
theorem hilt_family_fitness :
  (mrs_hilt_running.sum = 12) ∧
  (mrs_hilt_swimming.sum / yards_per_mile + 1000 / yards_per_mile * miles_per_km = 2854 / yards_per_mile) ∧
  (mrs_hilt_biking.sum = 19) ∧
  (mr_hilt_biking.sum = 13) ∧
  (mr_hilt_running.sum = 4) ∧
  (mr_hilt_swimming.sum = 2000) :=
by sorry

end NUMINAMATH_CALUDE_hilt_family_fitness_l546_54617


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l546_54656

/-- Given a quadratic equation 3x^2 = 5x - 1, prove that its standard form coefficients are a = 3 and b = -5 --/
theorem quadratic_equation_coefficients :
  let original_eq : ℝ → Prop := λ x ↦ 3 * x^2 = 5 * x - 1
  let standard_form : ℝ → ℝ → ℝ → ℝ → Prop := λ a b c x ↦ a * x^2 + b * x + c = 0
  ∃ (a b c : ℝ), (∀ x, original_eq x ↔ standard_form a b c x) ∧ a = 3 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l546_54656


namespace NUMINAMATH_CALUDE_number_of_lineups_l546_54607

/-- Represents the number of players in the team -/
def total_players : ℕ := 15

/-- Represents the number of players in the starting lineup -/
def lineup_size : ℕ := 4

/-- Represents the number of players that must be in the starting lineup -/
def fixed_players : ℕ := 3

/-- Calculates the number of possible starting lineups -/
def possible_lineups : ℕ := Nat.choose (total_players - fixed_players) (lineup_size - fixed_players)

/-- Theorem stating that the number of possible starting lineups is 12 -/
theorem number_of_lineups : possible_lineups = 12 := by sorry

end NUMINAMATH_CALUDE_number_of_lineups_l546_54607


namespace NUMINAMATH_CALUDE_money_division_l546_54634

theorem money_division (a b c : ℚ) : 
  a = (1/3 : ℚ) * b → 
  b = (1/4 : ℚ) * c → 
  b = 270 → 
  a + b + c = 1440 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l546_54634


namespace NUMINAMATH_CALUDE_sophia_age_in_three_years_l546_54647

/-- Represents the current ages of Jeremy, Sebastian, and Sophia --/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  sophia : ℕ

/-- The sum of their ages in three years is 150 --/
def sum_ages_in_three_years (ages : Ages) : Prop :=
  ages.jeremy + 3 + ages.sebastian + 3 + ages.sophia + 3 = 150

/-- Sebastian is 4 years older than Jeremy --/
def sebastian_older (ages : Ages) : Prop :=
  ages.sebastian = ages.jeremy + 4

/-- Jeremy's current age is 40 --/
def jeremy_age (ages : Ages) : Prop :=
  ages.jeremy = 40

/-- Sophia's age three years from now --/
def sophia_future_age (ages : Ages) : ℕ :=
  ages.sophia + 3

/-- Theorem stating Sophia's age three years from now is 60 --/
theorem sophia_age_in_three_years (ages : Ages) 
  (h1 : sum_ages_in_three_years ages) 
  (h2 : sebastian_older ages) 
  (h3 : jeremy_age ages) : 
  sophia_future_age ages = 60 := by
  sorry

end NUMINAMATH_CALUDE_sophia_age_in_three_years_l546_54647


namespace NUMINAMATH_CALUDE_range_of_a_l546_54646

theorem range_of_a (a : ℝ) : (∀ x > 0, x^2 + a*x + 1 ≥ 0) → a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l546_54646


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l546_54602

theorem quadratic_equation_solution (a b c x₁ x₂ y₁ y₂ : ℝ) 
  (hb : b ≠ 0)
  (h1 : x₁^2 + a*x₂^2 = b)
  (h2 : x₂*y₁ - x₁*y₂ = a)
  (h3 : x₁*y₁ + a*x₂*y₂ = c) :
  y₁^2 + a*y₂^2 = (a^3 + c^2) / b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l546_54602


namespace NUMINAMATH_CALUDE_fraction_simplification_l546_54632

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l546_54632


namespace NUMINAMATH_CALUDE_mitch_weekday_hours_l546_54630

/-- Represents the weekly work schedule and earnings of Mitch, a freelancer -/
structure MitchSchedule where
  weekdayHours : ℕ
  weekendHours : ℕ
  weekdayRate : ℕ
  weekendRate : ℕ
  totalEarnings : ℕ

/-- Theorem stating that Mitch works 25 hours from Monday to Friday -/
theorem mitch_weekday_hours (schedule : MitchSchedule) :
  schedule.weekendHours = 6 ∧
  schedule.weekdayRate = 3 ∧
  schedule.weekendRate = 6 ∧
  schedule.totalEarnings = 111 →
  schedule.weekdayHours = 25 := by
  sorry

end NUMINAMATH_CALUDE_mitch_weekday_hours_l546_54630


namespace NUMINAMATH_CALUDE_football_team_progress_l546_54650

/-- Calculates the net progress of a football team given yards lost and gained -/
def netProgress (yardsLost : Int) (yardsGained : Int) : Int :=
  yardsGained - yardsLost

/-- Proves that when a team loses 5 yards and gains 10 yards, their net progress is 5 yards -/
theorem football_team_progress : netProgress 5 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l546_54650


namespace NUMINAMATH_CALUDE_unique_function_exists_l546_54628

-- Define the positive rationals
def PositiveRationals := {q : ℚ // q > 0}

-- Define the function type
def FunctionType := PositiveRationals → PositiveRationals

-- Define the conditions
def Condition1 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, 0 < q.val ∧ q.val < 1/2 →
    f q = ⟨1 + (f ⟨q.val / (1 - 2*q.val), sorry⟩).val, sorry⟩

def Condition2 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, 1 < q.val ∧ q.val ≤ 2 →
    f q = ⟨1 + (f ⟨q.val + 1, sorry⟩).val, sorry⟩

def Condition3 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, (f q).val * (f ⟨1/q.val, sorry⟩).val = 1

-- State the theorem
theorem unique_function_exists :
  ∃! f : FunctionType, Condition1 f ∧ Condition2 f ∧ Condition3 f :=
sorry

end NUMINAMATH_CALUDE_unique_function_exists_l546_54628


namespace NUMINAMATH_CALUDE_team_size_l546_54627

theorem team_size (best_score : ℕ) (hypothetical_score : ℕ) (hypothetical_average : ℕ) (total_score : ℕ) :
  best_score = 85 →
  hypothetical_score = 92 →
  hypothetical_average = 84 →
  total_score = 497 →
  ∃ n : ℕ, n = 6 ∧ n * hypothetical_average - (hypothetical_score - best_score) = total_score :=
by sorry

end NUMINAMATH_CALUDE_team_size_l546_54627


namespace NUMINAMATH_CALUDE_tan_alpha_value_l546_54654

theorem tan_alpha_value (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l546_54654


namespace NUMINAMATH_CALUDE_round_trip_completion_l546_54671

/-- Represents a round trip with equal outbound and inbound journeys -/
structure RoundTrip where
  total_distance : ℝ
  outbound_distance : ℝ
  inbound_distance : ℝ
  equal_journeys : outbound_distance = inbound_distance
  total_is_sum : total_distance = outbound_distance + inbound_distance

/-- Theorem stating that completing the outbound journey and 20% of the inbound journey
    results in completing 60% of the total trip -/
theorem round_trip_completion (trip : RoundTrip) :
  trip.outbound_distance + 0.2 * trip.inbound_distance = 0.6 * trip.total_distance := by
  sorry

end NUMINAMATH_CALUDE_round_trip_completion_l546_54671


namespace NUMINAMATH_CALUDE_sin_inequalities_l546_54663

theorem sin_inequalities (x : ℝ) (h : x > 0) :
  (Real.sin x ≤ x) ∧
  (Real.sin x ≥ x - x^3 / 6) ∧
  (Real.sin x ≤ x - x^3 / 6 + x^5 / 120) ∧
  (Real.sin x ≥ x - x^3 / 6 + x^5 / 120 - x^7 / 5040) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequalities_l546_54663


namespace NUMINAMATH_CALUDE_cases_needed_l546_54614

theorem cases_needed (total_boxes : Nat) (boxes_per_case : Nat) : 
  total_boxes = 20 → boxes_per_case = 4 → total_boxes / boxes_per_case = 5 := by
  sorry

end NUMINAMATH_CALUDE_cases_needed_l546_54614


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150_l546_54689

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_five (n : ℕ) : Prop := n % 10 = 5

theorem smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150
  (a b : ℕ)
  (ha1 : is_two_digit a)
  (ha2 : a % 5 = 0)
  (ha3 : ends_in_five a)
  (ha4 : ∀ n, is_two_digit n → n % 5 = 0 → ends_in_five n → a ≤ n)
  (hb1 : is_three_digit b)
  (hb2 : b % 7 = 0)
  (hb3 : b > 150)
  (hb4 : ∀ n, is_three_digit n → n % 7 = 0 → n > 150 → b ≤ n) :
  a + b = 176 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_five_ending_in_five_plus_smallest_three_digit_multiple_of_seven_above_150_l546_54689


namespace NUMINAMATH_CALUDE_paper_mill_inspection_theorem_l546_54604

/-- Represents the number of paper mills -/
def num_mills : ℕ := 5

/-- Probability of passing initial inspection -/
def prob_pass_initial : ℚ := 1/2

/-- Probability of passing after rectification -/
def prob_pass_rectification : ℚ := 4/5

/-- Probability of exactly two mills needing rectification -/
def prob_two_rectified : ℚ := 5/16

/-- Probability of at least one mill being shut down -/
def prob_at_least_one_shutdown : ℚ := 1 - (9/10)^5

/-- Average number of mills needing rectification -/
def avg_mills_rectified : ℚ := 5/2

theorem paper_mill_inspection_theorem :
  (prob_two_rectified = Nat.choose num_mills 2 * (1 - prob_pass_initial)^2 * prob_pass_initial^3) ∧
  (prob_at_least_one_shutdown = 1 - (1 - (1 - prob_pass_initial) * (1 - prob_pass_rectification))^num_mills) ∧
  (avg_mills_rectified = num_mills * (1 - prob_pass_initial)) :=
by sorry

end NUMINAMATH_CALUDE_paper_mill_inspection_theorem_l546_54604


namespace NUMINAMATH_CALUDE_paint_usage_l546_54685

theorem paint_usage (total_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) : 
  total_paint = 360 →
  first_week_fraction = 1/6 →
  total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_paint_usage_l546_54685


namespace NUMINAMATH_CALUDE_coin_toss_probability_l546_54648

def coin_toss_events : ℕ := 2^4

def favorable_events : ℕ := 11

theorem coin_toss_probability : 
  (favorable_events : ℚ) / coin_toss_events = 11 / 16 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l546_54648


namespace NUMINAMATH_CALUDE_base5_arithmetic_l546_54672

/-- Converts a base 5 number to base 10 --/
def base5_to_base10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5 + c

/-- Converts a base 10 number to base 5 --/
noncomputable def base10_to_base5 (n : ℕ) : ℕ × ℕ × ℕ :=
  let d₂ := n / 25
  let r₂ := n % 25
  let d₁ := r₂ / 5
  let d₀ := r₂ % 5
  (d₂, d₁, d₀)

/-- Theorem stating that 142₅ + 324₅ - 213₅ = 303₅ --/
theorem base5_arithmetic : 
  let x := base5_to_base10 1 4 2
  let y := base5_to_base10 3 2 4
  let z := base5_to_base10 2 1 3
  base10_to_base5 (x + y - z) = (3, 0, 3) := by sorry

end NUMINAMATH_CALUDE_base5_arithmetic_l546_54672


namespace NUMINAMATH_CALUDE_wildflower_color_difference_l546_54629

theorem wildflower_color_difference :
  let total_flowers : ℕ := 44
  let yellow_and_white : ℕ := 13
  let red_and_yellow : ℕ := 17
  let red_and_white : ℕ := 14
  let flowers_with_red : ℕ := red_and_yellow + red_and_white
  let flowers_with_white : ℕ := yellow_and_white + red_and_white
  flowers_with_red - flowers_with_white = 4 :=
by sorry

end NUMINAMATH_CALUDE_wildflower_color_difference_l546_54629


namespace NUMINAMATH_CALUDE_athlete_formation_problem_l546_54664

theorem athlete_formation_problem :
  ∃ n : ℕ,
    200 ≤ n ∧ n ≤ 300 ∧
    (n + 4) % 10 = 0 ∧
    (n + 5) % 11 = 0 ∧
    n = 226 := by
  sorry

end NUMINAMATH_CALUDE_athlete_formation_problem_l546_54664


namespace NUMINAMATH_CALUDE_greatest_ratio_three_digit_number_l546_54682

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The ratio of a three-digit number to the sum of its digits -/
def ratio (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digit_sum n : Rat)

theorem greatest_ratio_three_digit_number :
  (∀ n : ThreeDigitNumber, ratio n ≤ 100) ∧
  (∃ n : ThreeDigitNumber, ratio n = 100) :=
sorry

end NUMINAMATH_CALUDE_greatest_ratio_three_digit_number_l546_54682


namespace NUMINAMATH_CALUDE_largest_t_value_for_60_degrees_l546_54626

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 12*t + 50

-- Define the theorem
theorem largest_t_value_for_60_degrees :
  let t := 6 + Real.sqrt 46
  (∀ s ≥ 0, temperature s = 60 → s ≤ t) ∧ temperature t = 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_t_value_for_60_degrees_l546_54626


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l546_54675

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + 3*x₁ - 1 = 0) → 
  (x₂^2 + 3*x₂ - 1 = 0) → 
  (x₁^2 - 3*x₂ + 1 = 11) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l546_54675


namespace NUMINAMATH_CALUDE_liars_count_l546_54640

/-- Represents the type of inhabitant: Knight or Liar -/
inductive InhabitantType
| Knight
| Liar

/-- Represents an island in the Tenth Kingdom -/
structure Island where
  population : Nat
  knights : Nat

/-- Represents the Tenth Kingdom -/
structure TenthKingdom where
  islands : List Island
  total_islands : Nat
  inhabitants_per_island : Nat

/-- Predicate for islands where everyone answered "Yes" to the first question -/
def first_question_yes (i : Island) : Prop :=
  i.knights = i.population / 2

/-- Predicate for islands where everyone answered "No" to the first question -/
def first_question_no (i : Island) : Prop :=
  i.knights ≠ i.population / 2

/-- Predicate for islands where everyone answered "No" to the second question -/
def second_question_no (i : Island) : Prop :=
  i.knights ≥ i.population / 2

/-- Predicate for islands where everyone answered "Yes" to the second question -/
def second_question_yes (i : Island) : Prop :=
  i.knights < i.population / 2

/-- Main theorem: The number of liars in the Tenth Kingdom is 1013 -/
theorem liars_count (k : TenthKingdom) : Nat := by
  sorry

/-- The Tenth Kingdom setup -/
def tenth_kingdom : TenthKingdom := {
  islands := [],  -- Placeholder for the list of islands
  total_islands := 17,
  inhabitants_per_island := 119
}

#check liars_count tenth_kingdom

end NUMINAMATH_CALUDE_liars_count_l546_54640


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l546_54699

/-- Represents the number of magical herbs -/
def num_herbs : ℕ := 4

/-- Represents the number of mystical stones -/
def num_stones : ℕ := 6

/-- Represents the number of herbs incompatible with one specific stone -/
def incompatible_herbs : ℕ := 3

/-- Calculates the number of valid combinations for the wizard's elixir -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_herbs

/-- Proves that the number of valid combinations for the wizard's elixir is 21 -/
theorem wizard_elixir_combinations : valid_combinations = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l546_54699


namespace NUMINAMATH_CALUDE_smallest_S_value_l546_54659

/-- Represents a standard 6-sided die -/
def Die := Fin 6

/-- The number of dice rolled -/
def n : ℕ := 342

/-- The sum we're comparing against -/
def target_sum : ℕ := 2052

/-- Function to calculate the probability of obtaining a specific sum -/
noncomputable def prob_of_sum (sum : ℕ) : ℝ := sorry

/-- The smallest sum S that has the same probability as the target sum -/
def S : ℕ := 342

theorem smallest_S_value :
  (prob_of_sum target_sum > 0) ∧ 
  (∀ s : ℕ, s < S → prob_of_sum s ≠ prob_of_sum target_sum) ∧
  (prob_of_sum S = prob_of_sum target_sum) := by sorry

end NUMINAMATH_CALUDE_smallest_S_value_l546_54659


namespace NUMINAMATH_CALUDE_morning_rowers_count_l546_54653

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowers : ℕ := 7

/-- The total number of campers who went rowing that day -/
def total_rowers : ℕ := 60

/-- The number of campers who went rowing in the morning -/
def morning_rowers : ℕ := total_rowers - afternoon_rowers

theorem morning_rowers_count : morning_rowers = 53 := by
  sorry

end NUMINAMATH_CALUDE_morning_rowers_count_l546_54653


namespace NUMINAMATH_CALUDE_water_consumed_last_mile_is_three_l546_54637

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  totalDistance : ℝ
  initialWater : ℝ
  hikeDuration : ℝ
  remainingWater : ℝ
  leakRate : ℝ
  consumptionRateFirst6Miles : ℝ

/-- Calculates the water consumed in the last mile of the hike -/
def waterConsumedLastMile (h : HikeScenario) : ℝ :=
  h.initialWater - h.remainingWater - (h.leakRate * h.hikeDuration) - 
  (h.consumptionRateFirst6Miles * (h.totalDistance - 1))

/-- Theorem stating that given the specific conditions, Harry drank 3 cups of water in the last mile -/
theorem water_consumed_last_mile_is_three (h : HikeScenario) 
  (h_totalDistance : h.totalDistance = 7)
  (h_initialWater : h.initialWater = 11)
  (h_hikeDuration : h.hikeDuration = 3)
  (h_remainingWater : h.remainingWater = 2)
  (h_leakRate : h.leakRate = 1)
  (h_consumptionRateFirst6Miles : h.consumptionRateFirst6Miles = 0.5) :
  waterConsumedLastMile h = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_consumed_last_mile_is_three_l546_54637


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l546_54683

/-- The function representing the curve y = x^3 + x^2 -/
def f (x : ℝ) : ℝ := x^3 + x^2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 2*x

theorem tangent_point_coordinates :
  ∀ a : ℝ, f' a = 4 → (a = 1 ∧ f a = 2) ∨ (a = -1 ∧ f a = -2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l546_54683


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l546_54639

/-- Given a quadratic expression 3x^2 + 9x + 17, when written in the form a(x-h)^2 + k, h = -3/2 -/
theorem quadratic_form_h_value : 
  ∃ (a k : ℝ), ∀ x : ℝ, 3*x^2 + 9*x + 17 = a*(x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l546_54639


namespace NUMINAMATH_CALUDE_sunshine_cost_per_mile_is_correct_l546_54610

/-- The cost per mile for Sunshine Car Rentals -/
def sunshine_cost_per_mile : ℝ := 0.18

/-- The daily rate for Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The daily rate for City Rentals -/
def city_daily_rate : ℝ := 18.95

/-- The cost per mile for City Rentals -/
def city_cost_per_mile : ℝ := 0.16

/-- The number of miles at which the costs are equal -/
def equal_cost_miles : ℝ := 48.0

theorem sunshine_cost_per_mile_is_correct :
  sunshine_daily_rate + equal_cost_miles * sunshine_cost_per_mile =
  city_daily_rate + equal_cost_miles * city_cost_per_mile :=
by sorry

end NUMINAMATH_CALUDE_sunshine_cost_per_mile_is_correct_l546_54610


namespace NUMINAMATH_CALUDE_cubic_function_constant_term_l546_54677

/-- Given a cubic function with specific properties, prove that the constant term is 16 -/
theorem cubic_function_constant_term (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f : ℤ → ℤ := λ x => x^3 + a*x^2 + b*x + c
  (f a = a^3) ∧ (f b = b^3) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_constant_term_l546_54677


namespace NUMINAMATH_CALUDE_erik_money_left_l546_54638

theorem erik_money_left (initial_money : ℕ) (bread_quantity : ℕ) (juice_quantity : ℕ) 
  (bread_price : ℕ) (juice_price : ℕ) (h1 : initial_money = 86) 
  (h2 : bread_quantity = 3) (h3 : juice_quantity = 3) (h4 : bread_price = 3) 
  (h5 : juice_price = 6) : 
  initial_money - (bread_quantity * bread_price + juice_quantity * juice_price) = 59 := by
  sorry

end NUMINAMATH_CALUDE_erik_money_left_l546_54638


namespace NUMINAMATH_CALUDE_billy_strategy_l546_54679

def FencePainting (n : ℕ) :=
  ∃ (strategy : ℕ → ℕ),
    (∀ k, k ≤ n → strategy k ≤ n) ∧
    (∀ k, k < n → strategy k ≠ k) ∧
    (∀ k, k < n - 1 → strategy k ≠ strategy (k + 1))

theorem billy_strategy (n : ℕ) (h : n > 10) :
  FencePainting n ∧ (n % 2 = 1 → ∃ (winning_strategy : ℕ → ℕ), FencePainting n) :=
sorry

#check billy_strategy

end NUMINAMATH_CALUDE_billy_strategy_l546_54679


namespace NUMINAMATH_CALUDE_max_distance_ellipse_point_l546_54601

/-- The maximum distance between any point on the ellipse x²/36 + y²/27 = 1 and the point (3,0) is 9 -/
theorem max_distance_ellipse_point : 
  ∃ (M : ℝ × ℝ), 
    (M.1^2 / 36 + M.2^2 / 27 = 1) ∧ 
    (∀ (N : ℝ × ℝ), (N.1^2 / 36 + N.2^2 / 27 = 1) → 
      ((N.1 - 3)^2 + N.2^2)^(1/2) ≤ ((M.1 - 3)^2 + M.2^2)^(1/2)) ∧
    ((M.1 - 3)^2 + M.2^2)^(1/2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_point_l546_54601


namespace NUMINAMATH_CALUDE_chess_team_arrangement_l546_54603

/-- The number of boys on the chess team -/
def num_boys : ℕ := 3

/-- The number of girls on the chess team -/
def num_girls : ℕ := 2

/-- The total number of students on the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange the team with girls at the ends and boys in the middle -/
def num_arrangements : ℕ := (Nat.factorial num_girls) * (Nat.factorial num_boys)

theorem chess_team_arrangement :
  num_arrangements = 12 :=
sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_l546_54603


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l546_54616

/-- In a parallelogram, if one angle exceeds the other by 50 degrees,
    and the smaller angle is 65 degrees, then the larger angle is 115 degrees. -/
theorem parallelogram_angle_measure (smaller_angle larger_angle : ℝ) : 
  smaller_angle = 65 →
  larger_angle = smaller_angle + 50 →
  larger_angle = 115 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l546_54616


namespace NUMINAMATH_CALUDE_fraction_seven_twentynine_repetend_l546_54624

/-- The repetend of a rational number is the repeating part of its decimal representation. -/
def repetend (n d : ℕ) : ℕ := sorry

/-- A number is a valid repetend for a fraction if it repeats infinitely in the decimal representation. -/
def is_valid_repetend (r n d : ℕ) : Prop := sorry

theorem fraction_seven_twentynine_repetend :
  let r := 241379
  is_valid_repetend r 7 29 ∧ repetend 7 29 = r :=
sorry

end NUMINAMATH_CALUDE_fraction_seven_twentynine_repetend_l546_54624


namespace NUMINAMATH_CALUDE_first_half_speed_l546_54643

/-- Proves that given a 6-hour journey where the second half is traveled at 48 kmph
    and the total distance is 324 km, the speed during the first half must be 60 kmph. -/
theorem first_half_speed (total_time : ℝ) (second_half_speed : ℝ) (total_distance : ℝ)
    (h1 : total_time = 6)
    (h2 : second_half_speed = 48)
    (h3 : total_distance = 324) :
    let first_half_time := total_time / 2
    let second_half_time := total_time / 2
    let second_half_distance := second_half_speed * second_half_time
    let first_half_distance := total_distance - second_half_distance
    let first_half_speed := first_half_distance / first_half_time
    first_half_speed = 60 := by
  sorry

#check first_half_speed

end NUMINAMATH_CALUDE_first_half_speed_l546_54643


namespace NUMINAMATH_CALUDE_technician_salary_l546_54676

theorem technician_salary (total_workers : ℕ) (total_avg_salary : ℝ) 
  (num_technicians : ℕ) (non_tech_avg_salary : ℝ) :
  total_workers = 18 →
  total_avg_salary = 8000 →
  num_technicians = 6 →
  non_tech_avg_salary = 6000 →
  (total_workers * total_avg_salary - (total_workers - num_technicians) * non_tech_avg_salary) / num_technicians = 12000 := by
  sorry

end NUMINAMATH_CALUDE_technician_salary_l546_54676


namespace NUMINAMATH_CALUDE_no_common_point_implies_skew_l546_54660

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define geometric relations
variable (parallel : Line → Line → Prop)
variable (determine_plane : Line → Line → Plane → Prop)
variable (coplanar : Point → Point → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (has_common_point : Line → Line → Prop)

-- Axioms and definitions
axiom parallel_determine_plane (a b : Line) (p : Plane) :
  parallel a b → determine_plane a b p

axiom non_coplanar_non_collinear (p q r s : Point) :
  ¬coplanar p q r s → ¬collinear p q r ∧ ¬collinear p q s ∧ ¬collinear p r s ∧ ¬collinear q r s

axiom skew_perpendicular (l₁ l₂ : Line) (p : Plane) :
  skew l₁ l₂ → ¬(perpendicular l₁ p ∧ perpendicular l₂ p)

-- The statement to be proved false
theorem no_common_point_implies_skew (l₁ l₂ : Line) :
  ¬has_common_point l₁ l₂ → skew l₁ l₂ := by sorry

end NUMINAMATH_CALUDE_no_common_point_implies_skew_l546_54660


namespace NUMINAMATH_CALUDE_pencil_cost_is_two_l546_54641

/-- Represents the cost of school supplies for Mary --/
structure SchoolSuppliesCost where
  num_classes : ℕ
  folders_per_class : ℕ
  pencils_per_class : ℕ
  erasers_per_pencils : ℕ
  folder_cost : ℚ
  eraser_cost : ℚ
  paint_cost : ℚ
  total_spent : ℚ

/-- Calculates the cost of a single pencil given the school supplies cost structure --/
def pencil_cost (c : SchoolSuppliesCost) : ℚ :=
  let total_folders := c.num_classes * c.folders_per_class
  let total_pencils := c.num_classes * c.pencils_per_class
  let total_erasers := (total_pencils + c.erasers_per_pencils - 1) / c.erasers_per_pencils
  let non_pencil_cost := total_folders * c.folder_cost + total_erasers * c.eraser_cost + c.paint_cost
  let pencil_total_cost := c.total_spent - non_pencil_cost
  pencil_total_cost / total_pencils

/-- Theorem stating that the cost of each pencil is $2 --/
theorem pencil_cost_is_two (c : SchoolSuppliesCost) 
  (h1 : c.num_classes = 6)
  (h2 : c.folders_per_class = 1)
  (h3 : c.pencils_per_class = 3)
  (h4 : c.erasers_per_pencils = 6)
  (h5 : c.folder_cost = 6)
  (h6 : c.eraser_cost = 1)
  (h7 : c.paint_cost = 5)
  (h8 : c.total_spent = 80) :
  pencil_cost c = 2 := by
  sorry


end NUMINAMATH_CALUDE_pencil_cost_is_two_l546_54641


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l546_54612

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, x^2 + (x + y)^2 = (x + 9)^2 ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l546_54612


namespace NUMINAMATH_CALUDE_library_seating_l546_54622

theorem library_seating (x : ℕ) : 
  (∃ (y : ℕ), x + y = 16) →  -- Total number of chairs and stools is 16
  (4 * x + 3 * (16 - x) = 60) -- Equation representing the situation
  :=
by
  sorry

end NUMINAMATH_CALUDE_library_seating_l546_54622


namespace NUMINAMATH_CALUDE_smallest_with_144_divisors_and_10_consecutive_l546_54621

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has 10 consecutive divisors -/
def has_10_consecutive_divisors (n : ℕ) : Prop := sorry

/-- The theorem stating that 110880 is the smallest number satisfying the conditions -/
theorem smallest_with_144_divisors_and_10_consecutive : 
  num_divisors 110880 = 144 ∧ 
  has_10_consecutive_divisors 110880 ∧ 
  ∀ m : ℕ, m < 110880 → (num_divisors m ≠ 144 ∨ ¬has_10_consecutive_divisors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_144_divisors_and_10_consecutive_l546_54621


namespace NUMINAMATH_CALUDE_area_units_order_l546_54684

/-- An enumeration of area units -/
inductive AreaUnit
  | SquareKilometer
  | Hectare
  | SquareMeter
  | SquareDecimeter
  | SquareCentimeter

/-- A function to compare two area units -/
def areaUnitLarger (a b : AreaUnit) : Prop :=
  match a, b with
  | AreaUnit.SquareKilometer, _ => a ≠ b
  | AreaUnit.Hectare, AreaUnit.SquareKilometer => False
  | AreaUnit.Hectare, _ => a ≠ b
  | AreaUnit.SquareMeter, AreaUnit.SquareKilometer => False
  | AreaUnit.SquareMeter, AreaUnit.Hectare => False
  | AreaUnit.SquareMeter, _ => a ≠ b
  | AreaUnit.SquareDecimeter, AreaUnit.SquareCentimeter => True
  | AreaUnit.SquareDecimeter, _ => False
  | AreaUnit.SquareCentimeter, _ => False

/-- Theorem stating the correct order of area units from largest to smallest -/
theorem area_units_order :
  areaUnitLarger AreaUnit.SquareKilometer AreaUnit.Hectare ∧
  areaUnitLarger AreaUnit.Hectare AreaUnit.SquareMeter ∧
  areaUnitLarger AreaUnit.SquareMeter AreaUnit.SquareDecimeter ∧
  areaUnitLarger AreaUnit.SquareDecimeter AreaUnit.SquareCentimeter :=
sorry

end NUMINAMATH_CALUDE_area_units_order_l546_54684


namespace NUMINAMATH_CALUDE_simplify_expression_l546_54636

theorem simplify_expression : (12^0.6) * (12^0.4) * (8^0.2) * (8^0.8) = 96 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l546_54636


namespace NUMINAMATH_CALUDE_h_not_prime_l546_54633

def h (n : ℕ+) : ℤ := n^4 - 380 * n^2 + 600

theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by
  sorry

end NUMINAMATH_CALUDE_h_not_prime_l546_54633


namespace NUMINAMATH_CALUDE_length_of_AE_l546_54681

/-- Given a coordinate grid where:
    - A is at (0,4)
    - B is at (7,0)
    - C is at (3,0)
    - D is at (5,3)
    - Line segment AB meets line segment CD at point E
    Prove that the length of segment AE is (7√65)/13 -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (7, 0) →
  C = (3, 0) →
  D = (5, 3) →
  E ∈ Set.Icc A B →
  E ∈ Set.Icc C D →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = (7 * Real.sqrt 65) / 13 :=
by sorry

end NUMINAMATH_CALUDE_length_of_AE_l546_54681


namespace NUMINAMATH_CALUDE_flash_catches_ace_l546_54696

/-- The time it takes for Flash to catch up to Ace in a race -/
theorem flash_catches_ace (v a y : ℝ) (hv : v > 0) (ha : a > 0) (hy : y > 0) :
  let t := (v + Real.sqrt (v^2 + 2*a*y)) / a
  2 * (v * t + y) = a * t^2 := by sorry

end NUMINAMATH_CALUDE_flash_catches_ace_l546_54696


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l546_54613

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ ((x - 1)^2 = 4) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l546_54613


namespace NUMINAMATH_CALUDE_not_divisible_by_three_and_four_l546_54619

theorem not_divisible_by_three_and_four (n : ℤ) : 
  ¬(∃ k : ℤ, n^2 + 1 = 3 * k) ∧ ¬(∃ m : ℤ, n^2 + 1 = 4 * m) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_and_four_l546_54619


namespace NUMINAMATH_CALUDE_right_triangle_7_24_25_l546_54686

theorem right_triangle_7_24_25 (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_7_24_25_l546_54686


namespace NUMINAMATH_CALUDE_train_speed_proof_l546_54697

/-- Proves that a train with given crossing times has a specific speed -/
theorem train_speed_proof (platform_length : ℝ) (platform_cross_time : ℝ) (man_cross_time : ℝ) :
  platform_length = 280 →
  platform_cross_time = 32 →
  man_cross_time = 18 →
  ∃ (train_speed : ℝ), train_speed = 72 ∧ 
    (train_speed * man_cross_time = train_speed * platform_cross_time - platform_length) :=
by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_train_speed_proof_l546_54697


namespace NUMINAMATH_CALUDE_no_prime_valued_polynomial_l546_54691

theorem no_prime_valued_polynomial : ¬ ∃ (P : ℕ → ℤ), (∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > n → P k = 0)) ∧ (∀ k : ℕ, Nat.Prime (P k).natAbs) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_valued_polynomial_l546_54691


namespace NUMINAMATH_CALUDE_common_points_on_line_l546_54680

-- Define the curves and line
def C1 (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 1)^2 = a^2}
def C2 : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 4}
def C3 : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1}

theorem common_points_on_line (a : ℝ) (h : a > 0) : 
  (∀ p, p ∈ C1 a ∩ C2 → p ∈ C3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_common_points_on_line_l546_54680


namespace NUMINAMATH_CALUDE_cake_shop_problem_l546_54605

theorem cake_shop_problem :
  ∃ (N n K : ℕ+), 
    (N - n * K = 6) ∧ 
    (N = (n - 1) * 8 + 1) ∧ 
    (N = 97) := by
  sorry

end NUMINAMATH_CALUDE_cake_shop_problem_l546_54605


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a4_l546_54625

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem arithmetic_geometric_sequence_a4
  (a : ℕ → ℝ)
  (h_seq : ArithmeticGeometricSequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a4_l546_54625


namespace NUMINAMATH_CALUDE_alcohol_mixture_exists_l546_54642

theorem alcohol_mixture_exists : ∃ (x y z : ℕ), 
  x + y + z = 560 ∧ 
  (70 * x + 64 * y + 50 * z : ℚ) = 60 * 560 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_exists_l546_54642


namespace NUMINAMATH_CALUDE_conic_section_types_l546_54618

/-- The equation y^4 - 6x^4 = 3y^2 - 4 represents the union of a hyperbola and an ellipse -/
theorem conic_section_types (x y : ℝ) : 
  y^4 - 6*x^4 = 3*y^2 - 4 → 
  (∃ (a b : ℝ), y^2 - a*x^2 = b ∧ a > 0 ∧ b > 0) ∧ 
  (∃ (c d : ℝ), y^2 + c*x^2 = d ∧ c > 0 ∧ d > 0) := by
sorry

end NUMINAMATH_CALUDE_conic_section_types_l546_54618


namespace NUMINAMATH_CALUDE_not_P_necessary_not_sufficient_for_not_Q_l546_54645

-- Define the propositions P and Q as functions from ℝ to Prop
def P (x : ℝ) : Prop := |2*x - 3| > 1
def Q (x : ℝ) : Prop := x^2 - 3*x + 2 ≥ 0

-- Define the relationship between ¬P and ¬Q
theorem not_P_necessary_not_sufficient_for_not_Q :
  (∀ x, ¬(Q x) → ¬(P x)) ∧ 
  ¬(∀ x, ¬(P x) → ¬(Q x)) :=
sorry

end NUMINAMATH_CALUDE_not_P_necessary_not_sufficient_for_not_Q_l546_54645


namespace NUMINAMATH_CALUDE_no_good_tetrahedron_inside_good_parallelepiped_l546_54657

-- Define a good polyhedron
def is_good_polyhedron (volume : ℝ) (surface_area : ℝ) : Prop :=
  volume = surface_area

-- Define a tetrahedron
structure Tetrahedron where
  volume : ℝ
  surface_area : ℝ

-- Define a parallelepiped
structure Parallelepiped where
  volume : ℝ
  surface_area : ℝ
  face_areas : Fin 3 → ℝ
  heights : Fin 3 → ℝ

-- Define the property of a tetrahedron being inside a parallelepiped
def tetrahedron_inside_parallelepiped (t : Tetrahedron) (p : Parallelepiped) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  t.volume = (1/3) * t.surface_area * r ∧
  p.heights 0 > 2 * r

-- Theorem statement
theorem no_good_tetrahedron_inside_good_parallelepiped :
  ¬ ∃ (t : Tetrahedron) (p : Parallelepiped),
    is_good_polyhedron t.volume t.surface_area ∧
    is_good_polyhedron p.volume p.surface_area ∧
    tetrahedron_inside_parallelepiped t p :=
sorry

end NUMINAMATH_CALUDE_no_good_tetrahedron_inside_good_parallelepiped_l546_54657


namespace NUMINAMATH_CALUDE_common_tangent_range_l546_54652

/-- The range of parameter a for which the curves y = ln x + 1 and y = x² + x + 3a have a common tangent line -/
theorem common_tangent_range :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧
    (1 / x₁ = 2 * x₂ + 1) ∧
    (Real.log x₁ + 1 = x₂^2 + x₂ + 3 * a) ∧
    (Real.log x₁ + x₂^2 = 3 * a)) ↔
  a ≥ (1 - 4 * Real.log 2) / 12 :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_range_l546_54652


namespace NUMINAMATH_CALUDE_amy_ticket_cost_l546_54673

/-- The total cost of tickets purchased by Amy at the fair -/
theorem amy_ticket_cost (initial_tickets : ℕ) (additional_tickets : ℕ) (price_per_ticket : ℚ) :
  initial_tickets = 33 →
  additional_tickets = 21 →
  price_per_ticket = 3/2 →
  (initial_tickets + additional_tickets : ℚ) * price_per_ticket = 81 := by
sorry

end NUMINAMATH_CALUDE_amy_ticket_cost_l546_54673


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l546_54611

theorem quadratic_inequality_range :
  ∃ a : ℝ, a ∈ Set.Icc 1 3 ∧ ∀ x : ℝ, a * x^2 + (a - 2) * x - 2 > 0 →
    x < -1 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l546_54611


namespace NUMINAMATH_CALUDE_eight_by_eight_tiling_ten_by_ten_no_tiling_l546_54609

-- Define a chessboard
structure Chessboard :=
  (size : Nat)
  (total_squares : Nat)
  (black_squares : Nat)
  (white_squares : Nat)

-- Define a pedestal shape
structure Pedestal :=
  (squares_covered : Nat)

-- Define the tiling property
def can_tile (b : Chessboard) (p : Pedestal) : Prop :=
  b.total_squares % p.squares_covered = 0

-- Define the color coverage property for 10x10 board
def color_coverage_property (b : Chessboard) (p : Pedestal) : Prop :=
  ∃ (k : Nat), 3 * k + k = b.black_squares ∧ 3 * k + k = b.white_squares

-- Theorem for 8x8 chessboard
theorem eight_by_eight_tiling :
  ∀ (b : Chessboard) (p : Pedestal),
    b.size = 8 →
    b.total_squares = 64 →
    p.squares_covered = 4 →
    can_tile b p :=
sorry

-- Theorem for 10x10 chessboard
theorem ten_by_ten_no_tiling :
  ∀ (b : Chessboard) (p : Pedestal),
    b.size = 10 →
    b.total_squares = 100 →
    b.black_squares = 50 →
    b.white_squares = 50 →
    p.squares_covered = 4 →
    ¬(can_tile b p ∧ color_coverage_property b p) :=
sorry

end NUMINAMATH_CALUDE_eight_by_eight_tiling_ten_by_ten_no_tiling_l546_54609


namespace NUMINAMATH_CALUDE_prob_white_second_given_red_first_l546_54666

/-- The probability of drawing a white ball on the second draw, given that the first ball drawn is red -/
theorem prob_white_second_given_red_first
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (h_total : total_balls = red_balls + white_balls)
  (h_red : red_balls = 5)
  (h_white : white_balls = 3) :
  (white_balls : ℚ) / (total_balls - 1) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_second_given_red_first_l546_54666


namespace NUMINAMATH_CALUDE_mean_home_runs_l546_54674

def player_count : ℕ := 6 + 4 + 3 + 1

def total_home_runs : ℕ := 6 * 6 + 7 * 4 + 8 * 3 + 10 * 1

theorem mean_home_runs : (total_home_runs : ℚ) / player_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l546_54674


namespace NUMINAMATH_CALUDE_congruence_problem_l546_54600

theorem congruence_problem (N : ℕ) (h1 : N > 1) 
  (h2 : 69 % N = 90 % N) (h3 : 90 % N = 125 % N) : 81 % N = 4 % N := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l546_54600


namespace NUMINAMATH_CALUDE_min_value_of_expression_l546_54687

theorem min_value_of_expression :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l546_54687


namespace NUMINAMATH_CALUDE_smallest_4digit_base7_divisible_by_7_l546_54623

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 7 number --/
def is4DigitBase7 (n : ℕ) : Prop := sorry

/-- The smallest 4-digit base 7 number --/
def smallestBase7_4Digit : ℕ := 1000

theorem smallest_4digit_base7_divisible_by_7 :
  (is4DigitBase7 smallestBase7_4Digit) ∧
  (base7ToDecimal smallestBase7_4Digit % 7 = 0) ∧
  (∀ n : ℕ, is4DigitBase7 n ∧ n < smallestBase7_4Digit → base7ToDecimal n % 7 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_4digit_base7_divisible_by_7_l546_54623


namespace NUMINAMATH_CALUDE_power_sum_inequality_l546_54631

theorem power_sum_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_eq_three : a + b + c = 3) : 
  a^a + b^b + c^c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l546_54631


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l546_54608

/-- A cylinder with a square axial cross-section of area 5 has a lateral surface area of 5π. -/
theorem cylinder_lateral_surface_area (h : ℝ) (r : ℝ) : 
  h * h = 5 → 2 * r = h → 2 * π * r * h = 5 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l546_54608


namespace NUMINAMATH_CALUDE_proportional_function_ratio_l546_54670

theorem proportional_function_ratio (k a b : ℝ) : 
  k ≠ 0 →
  b ≠ 0 →
  3 = k * 1 →
  b = k * a →
  a / b = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_proportional_function_ratio_l546_54670


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l546_54694

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon (nonagon) contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l546_54694


namespace NUMINAMATH_CALUDE_coloring_books_per_shelf_l546_54669

theorem coloring_books_per_shelf 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (num_shelves : ℕ) 
  (h1 : initial_stock = 120)
  (h2 : books_sold = 39)
  (h3 : num_shelves = 9)
  (h4 : num_shelves > 0) :
  (initial_stock - books_sold) / num_shelves = 9 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_per_shelf_l546_54669


namespace NUMINAMATH_CALUDE_robot_capacity_theorem_l546_54620

/-- Represents the material handling capacity of robots A and B --/
structure RobotCapacity where
  A : ℝ
  B : ℝ

/-- The conditions given in the problem --/
def satisfiesConditions (c : RobotCapacity) : Prop :=
  c.A = c.B + 30 ∧ 1000 / c.A = 800 / c.B

/-- The theorem to prove --/
theorem robot_capacity_theorem :
  ∃ c : RobotCapacity, satisfiesConditions c ∧ c.A = 150 ∧ c.B = 120 := by
  sorry

end NUMINAMATH_CALUDE_robot_capacity_theorem_l546_54620


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l546_54615

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y ≤ Real.sqrt 202 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l546_54615


namespace NUMINAMATH_CALUDE_alternating_series_sum_l546_54651

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

theorem alternating_series_sum (a b : ℕ) :
  (S a + S b + S (a + b) = 1) ↔ (Odd a ∧ Odd b) :=
sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l546_54651


namespace NUMINAMATH_CALUDE_opposite_of_2xyz_l546_54668

theorem opposite_of_2xyz (x y z : ℝ) 
  (h : Real.sqrt (2*x - 1) + Real.sqrt (1 - 2*x) + |x - 2*y| + |z + 4*y| = 0) : 
  -(2*x*y*z) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2xyz_l546_54668


namespace NUMINAMATH_CALUDE_intersection_point_is_two_one_l546_54661

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first line: x - 2y = 0 -/
def line1 (p : IntersectionPoint) : Prop :=
  p.x - 2 * p.y = 0

/-- Definition of the second line: x + y - 3 = 0 -/
def line2 (p : IntersectionPoint) : Prop :=
  p.x + p.y - 3 = 0

/-- Theorem stating that (2, 1) is the unique intersection point of the two lines -/
theorem intersection_point_is_two_one :
  ∃! p : IntersectionPoint, line1 p ∧ line2 p ∧ p.x = 2 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_two_one_l546_54661


namespace NUMINAMATH_CALUDE_compound_interest_rate_l546_54667

/-- Given an initial investment of $7000, invested for 2 years with annual compounding,
    resulting in a final amount of $8470, prove that the annual interest rate is 0.1 (10%). -/
theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (r : ℝ) : 
  P = 7000 → A = 8470 → t = 2 → n = 1 → 
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_compound_interest_rate_l546_54667


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l546_54662

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 9 * x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 33 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l546_54662


namespace NUMINAMATH_CALUDE_circle_rotation_l546_54695

theorem circle_rotation (r : ℝ) (d : ℝ) (h1 : r = 1) (h2 : d = 11 * Real.pi) :
  (d / (2 * Real.pi * r)) % 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_rotation_l546_54695


namespace NUMINAMATH_CALUDE_quadrilateral_I_greater_than_II_l546_54690

/-- Quadrilateral I with vertices at (0,0), (2,0), (2,1), and (0,1) -/
def quadrilateral_I : List (ℝ × ℝ) := [(0,0), (2,0), (2,1), (0,1)]

/-- Quadrilateral II with vertices at (0,0), (1,0), (1,1), (0,2) -/
def quadrilateral_II : List (ℝ × ℝ) := [(0,0), (1,0), (1,1), (0,2)]

/-- Calculate the area of a quadrilateral given its vertices -/
def area (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Calculate the perimeter of a quadrilateral given its vertices -/
def perimeter (vertices : List (ℝ × ℝ)) : ℝ := sorry

theorem quadrilateral_I_greater_than_II :
  area quadrilateral_I > area quadrilateral_II ∧
  perimeter quadrilateral_I > perimeter quadrilateral_II :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_I_greater_than_II_l546_54690


namespace NUMINAMATH_CALUDE_center_value_is_27_l546_54665

/-- Represents a 7x7 array where each row and column is an arithmetic sequence -/
def ArithmeticArray := Fin 7 → Fin 7 → ℤ

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : Fin 7 → ℤ) : ℤ :=
  (seq 6 - seq 0) / 6

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : Fin 7 → ℤ) : Prop :=
  ∀ i j : Fin 7, seq j - seq i = (j - i : ℤ) * commonDifference seq

/-- Theorem: The center value of the arithmetic array is 27 -/
theorem center_value_is_27 (A : ArithmeticArray) 
  (h_rows : ∀ i : Fin 7, isArithmeticSequence (λ j ↦ A i j))
  (h_cols : ∀ j : Fin 7, isArithmeticSequence (λ i ↦ A i j))
  (h_first_row : A 0 0 = 3 ∧ A 0 6 = 39)
  (h_last_row : A 6 0 = 10 ∧ A 6 6 = 58) :
  A 3 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_center_value_is_27_l546_54665


namespace NUMINAMATH_CALUDE_marcel_total_cost_l546_54698

def calculate_total_cost (pen_price : ℝ) : ℝ :=
  let briefcase_price := 5 * pen_price
  let notebook_price := 2 * pen_price
  let calculator_price := 3 * notebook_price
  let briefcase_discount := 0.15 * briefcase_price
  let discounted_briefcase_price := briefcase_price - briefcase_discount
  let total_before_tax := pen_price + discounted_briefcase_price + notebook_price + calculator_price
  let tax := 0.10 * total_before_tax
  total_before_tax + tax

theorem marcel_total_cost :
  calculate_total_cost 4 = 58.30 := by sorry

end NUMINAMATH_CALUDE_marcel_total_cost_l546_54698


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l546_54693

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the theorem
theorem triangle_ABC_properties 
  (A B C : ℝ) 
  (h_triangle : triangle_ABC A B C)
  (h_cos_A : Real.cos A = -5/13)
  (h_cos_B : Real.cos B = 3/5)
  (h_BC : 5 = 5) :
  Real.sin C = 16/65 ∧ 
  5 * 5 * Real.sin C / 2 = 8/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l546_54693


namespace NUMINAMATH_CALUDE_second_loan_amount_l546_54649

def initial_loan : ℝ := 40
def final_debt : ℝ := 30

theorem second_loan_amount (half_paid_back : ℝ) (second_loan : ℝ) 
  (h1 : half_paid_back = initial_loan / 2)
  (h2 : final_debt = initial_loan - half_paid_back + second_loan) : 
  second_loan = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_loan_amount_l546_54649


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l546_54688

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  12 * 3 -- Each original edge is divided into 3 segments

/-- Theorem stating that a cube of side length 4 with cubes of side length 2 removed from each corner has 36 edges -/
theorem modified_cube_edge_count :
  ∀ (cube : ModifiedCube),
    cube.originalSideLength = 4 →
    cube.removedCubeSideLength = 2 →
    edgeCount cube = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l546_54688


namespace NUMINAMATH_CALUDE_calculate_expression_l546_54658

theorem calculate_expression (a : ℝ) : a * a^2 - 2 * a^3 = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l546_54658


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l546_54692

def ends_in (n : ℕ) (m : ℕ) : Prop := n % 100 = m

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_number_with_conditions : 
  ∀ n : ℕ, 
    ends_in n 56 ∧ 
    n % 56 = 0 ∧ 
    digit_sum n = 56 →
    n ≥ 29899856 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l546_54692


namespace NUMINAMATH_CALUDE_a_range_l546_54655

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

/-- f(x) is a decreasing function -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The theorem statement -/
theorem a_range (a : ℝ) :
  is_decreasing (f a) → a ∈ Set.Icc (1/2) 1 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l546_54655


namespace NUMINAMATH_CALUDE_trig_equation_solution_l546_54606

open Real

theorem trig_equation_solution (x : ℝ) : 
  (sin (x + 15 * π / 180) + sin (x + 45 * π / 180) + sin (x + 75 * π / 180) = 
   sin (15 * π / 180) + sin (45 * π / 180) + sin (75 * π / 180)) ↔ 
  (∃ k : ℤ, x = k * 2 * π ∨ x = π / 2 + k * 2 * π) :=
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l546_54606


namespace NUMINAMATH_CALUDE_complete_square_constant_l546_54635

theorem complete_square_constant (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 8*x = a*(x - h)^2 + k ∧ k = -16 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_constant_l546_54635
