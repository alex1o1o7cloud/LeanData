import Mathlib

namespace NUMINAMATH_CALUDE_soldier_rearrangement_l3482_348257

theorem soldier_rearrangement (n : Nat) (h : n = 20 ∨ n = 21) :
  ∃ (d : ℝ), d = 10 * Real.sqrt 2 ∧
  (∀ (rearrangement : Fin n × Fin n → Fin n × Fin n),
    (∀ (i j : Fin n), (rearrangement (i, j) ≠ (i, j)) →
      Real.sqrt ((i.val - (rearrangement (i, j)).1.val)^2 +
                 (j.val - (rearrangement (i, j)).2.val)^2) ≥ d) →
    (∀ (i j : Fin n), ∃ (k l : Fin n), rearrangement (k, l) = (i, j))) ∧
  (∀ (d' : ℝ), d' > d →
    ¬∃ (rearrangement : Fin n × Fin n → Fin n × Fin n),
      (∀ (i j : Fin n), (rearrangement (i, j) ≠ (i, j)) →
        Real.sqrt ((i.val - (rearrangement (i, j)).1.val)^2 +
                   (j.val - (rearrangement (i, j)).2.val)^2) ≥ d') ∧
      (∀ (i j : Fin n), ∃ (k l : Fin n), rearrangement (k, l) = (i, j))) :=
by sorry

end NUMINAMATH_CALUDE_soldier_rearrangement_l3482_348257


namespace NUMINAMATH_CALUDE_cosine_identity_proof_l3482_348228

theorem cosine_identity_proof : 2 * (Real.cos (15 * π / 180))^2 - Real.cos (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_proof_l3482_348228


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_angle_in_range_same_terminal_side_750_l3482_348285

theorem same_terminal_side_angle : ℤ → ℝ → ℝ
  | k, α => k * 360 + α

theorem angle_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

theorem same_terminal_side_750 :
  ∃ (θ : ℝ), angle_in_range θ ∧ ∃ (k : ℤ), same_terminal_side_angle k θ = 750 ∧ θ = 30 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_angle_in_range_same_terminal_side_750_l3482_348285


namespace NUMINAMATH_CALUDE_correct_operation_l3482_348232

theorem correct_operation (x : ℝ) : 4 * x^2 * (3 * x) = 12 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3482_348232


namespace NUMINAMATH_CALUDE_binary_remainder_by_eight_l3482_348283

/-- The remainder when 110111100101₂ is divided by 8 is 5 -/
theorem binary_remainder_by_eight : Nat.mod 0b110111100101 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_remainder_by_eight_l3482_348283


namespace NUMINAMATH_CALUDE_business_profit_calculation_l3482_348268

def business_profit (a_investment b_investment total_profit : ℚ) : ℚ :=
  let total_investment := a_investment + b_investment
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let a_share_ratio := a_investment / total_investment
  let a_share := a_share_ratio * remaining_profit
  management_fee + a_share

theorem business_profit_calculation :
  business_profit 3500 1500 9600 = 7008 :=
by sorry

end NUMINAMATH_CALUDE_business_profit_calculation_l3482_348268


namespace NUMINAMATH_CALUDE_johns_final_push_time_l3482_348274

/-- The time of John's final push in a race, given specific conditions --/
theorem johns_final_push_time (john_initial_lag : ℝ) (john_speed : ℝ) (steve_speed : ℝ) (john_final_lead : ℝ)
  (h1 : john_initial_lag = 15)
  (h2 : john_speed = 4.2)
  (h3 : steve_speed = 3.7)
  (h4 : john_final_lead = 2) :
  (john_initial_lag + john_final_lead) / john_speed = 17 / 4.2 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_push_time_l3482_348274


namespace NUMINAMATH_CALUDE_x_lower_bound_l3482_348215

def x : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => 3
  | (n + 3) => 4 * x (n + 2) - 2 * x (n + 1) - 3 * x n

theorem x_lower_bound : ∀ n : ℕ, n ≥ 3 → x n > (3/2) * (1 + 3^(n-2)) := by
  sorry

end NUMINAMATH_CALUDE_x_lower_bound_l3482_348215


namespace NUMINAMATH_CALUDE_cos_zeros_range_l3482_348255

theorem cos_zeros_range (ω : ℝ) (h_pos : ω > 0) : 
  (∃ (z₁ z₂ z₃ : ℝ), z₁ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₂ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₃ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
                      Real.cos (ω * z₁ - Real.pi / 6) = 0 ∧
                      Real.cos (ω * z₂ - Real.pi / 6) = 0 ∧
                      Real.cos (ω * z₃ - Real.pi / 6) = 0 ∧
                      (∀ z ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi), 
                        Real.cos (ω * z - Real.pi / 6) = 0 → z = z₁ ∨ z = z₂ ∨ z = z₃)) →
  11 / 6 ≤ ω ∧ ω < 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cos_zeros_range_l3482_348255


namespace NUMINAMATH_CALUDE_range_of_a_l3482_348213

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 2 4, x^2 - a*x - 8 > 0) ∧ 
  (∃ θ : ℝ, a - 1 ≤ Real.sin θ - 2) → 
  a < -2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3482_348213


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l3482_348265

theorem cafeteria_red_apples :
  ∀ (red_apples green_apples students_wanting_fruit extra_apples : ℕ),
    green_apples = 15 →
    students_wanting_fruit = 5 →
    extra_apples = 16 →
    red_apples + green_apples = students_wanting_fruit + extra_apples →
    red_apples = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l3482_348265


namespace NUMINAMATH_CALUDE_departure_interval_is_six_l3482_348204

/-- Represents the tram system with a person riding along the route -/
structure TramSystem where
  tram_speed : ℝ
  person_speed : ℝ
  overtake_time : ℝ
  approach_time : ℝ

/-- The interval between tram departures from the station -/
def departure_interval (sys : TramSystem) : ℝ :=
  6

/-- Theorem stating that the departure interval is 6 minutes -/
theorem departure_interval_is_six (sys : TramSystem) 
  (h1 : sys.tram_speed > sys.person_speed) 
  (h2 : sys.overtake_time = 12)
  (h3 : sys.approach_time = 4) :
  departure_interval sys = 6 := by
  sorry

end NUMINAMATH_CALUDE_departure_interval_is_six_l3482_348204


namespace NUMINAMATH_CALUDE_frisbee_sales_minimum_receipts_l3482_348256

theorem frisbee_sales_minimum_receipts :
  ∀ (x y : ℕ),
  x + y = 64 →
  y ≥ 8 →
  3 * x + 4 * y ≥ 200 :=
by
  sorry

end NUMINAMATH_CALUDE_frisbee_sales_minimum_receipts_l3482_348256


namespace NUMINAMATH_CALUDE_curve_point_when_a_is_one_curve_passes_through_fixed_point_l3482_348262

-- Define the curve equation
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0

-- Theorem for case a = 1
theorem curve_point_when_a_is_one :
  ∀ x y : ℝ, curve x y 1 ↔ x = 1 ∧ y = 1 :=
sorry

-- Theorem for case a ≠ 1
theorem curve_passes_through_fixed_point :
  ∀ a : ℝ, a ≠ 1 → curve 1 1 a :=
sorry

end NUMINAMATH_CALUDE_curve_point_when_a_is_one_curve_passes_through_fixed_point_l3482_348262


namespace NUMINAMATH_CALUDE_automobile_distance_l3482_348244

/-- Proves that an automobile traveling a/4 feet in r seconds will cover 20a/r yards in 4 minutes if it maintains the same rate. -/
theorem automobile_distance (a r : ℝ) (ha : a > 0) (hr : r > 0) : 
  let rate_feet_per_second : ℝ := a / (4 * r)
  let rate_yards_per_second : ℝ := rate_feet_per_second / 3
  let time_in_seconds : ℝ := 4 * 60
  rate_yards_per_second * time_in_seconds = 20 * a / r :=
by sorry

end NUMINAMATH_CALUDE_automobile_distance_l3482_348244


namespace NUMINAMATH_CALUDE_probability_problem_l3482_348259

theorem probability_problem (p_biology : ℚ) (p_no_chemistry : ℚ)
  (h1 : p_biology = 5/8)
  (h2 : p_no_chemistry = 1/2) :
  let p_no_biology := 1 - p_biology
  let p_neither := p_no_biology * p_no_chemistry
  (p_no_biology = 3/8) ∧ (p_neither = 3/16) := by
  sorry

end NUMINAMATH_CALUDE_probability_problem_l3482_348259


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l3482_348209

/-- 
Given a parabola y = ax^2 + bx + c with vertex (p, -p) and y-intercept (0, p), 
where p ≠ 0, the value of b is -4.
-/
theorem parabola_coefficient_b (a b c p : ℝ) : 
  p ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 - p) →
  (a * 0^2 + b * 0 + c = p) →
  b = -4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l3482_348209


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3482_348258

def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3482_348258


namespace NUMINAMATH_CALUDE_simplify_expression_l3482_348281

theorem simplify_expression (x y : ℝ) : 7*x + 3*y + 4 - 2*x + 9 + 5*y = 5*x + 8*y + 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3482_348281


namespace NUMINAMATH_CALUDE_petals_per_rose_correct_petals_per_rose_l3482_348218

theorem petals_per_rose (petals_per_ounce : ℕ) (roses_per_bush : ℕ) (bushes_harvested : ℕ) 
  (bottles_produced : ℕ) (ounces_per_bottle : ℕ) : ℕ :=
  let total_ounces := bottles_produced * ounces_per_bottle
  let total_petals := total_ounces * petals_per_ounce
  let petals_per_bush := total_petals / bushes_harvested
  petals_per_bush / roses_per_bush

theorem correct_petals_per_rose :
  petals_per_rose 320 12 800 20 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_petals_per_rose_correct_petals_per_rose_l3482_348218


namespace NUMINAMATH_CALUDE_garlic_cloves_remaining_l3482_348234

theorem garlic_cloves_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 237 → used = 184 → remaining = initial - used → remaining = 53 := by
sorry

end NUMINAMATH_CALUDE_garlic_cloves_remaining_l3482_348234


namespace NUMINAMATH_CALUDE_cosine_graph_minimum_l3482_348241

theorem cosine_graph_minimum (c : ℝ) (h1 : c > 0) : 
  (∀ x : ℝ, 3 * Real.cos (5 * x + c) ≥ 3 * Real.cos c) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo 0 ε, 3 * Real.cos (5 * x + c) > 3 * Real.cos c) → 
  c = Real.pi := by
sorry

end NUMINAMATH_CALUDE_cosine_graph_minimum_l3482_348241


namespace NUMINAMATH_CALUDE_stock_price_increase_l3482_348227

theorem stock_price_increase (initial_price : ℝ) (first_year_increase : ℝ) : 
  initial_price > 0 →
  first_year_increase > 0 →
  initial_price * (1 + first_year_increase / 100) * 0.75 * 1.2 = initial_price * 1.08 →
  first_year_increase = 20 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l3482_348227


namespace NUMINAMATH_CALUDE_aarons_playground_area_l3482_348296

/-- Represents a rectangular playground with fence posts. -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Calculates the area of the playground given its specifications. -/
def playground_area (p : Playground) : ℕ :=
  (p.short_side_posts - 1) * p.post_spacing * (p.long_side_posts - 1) * p.post_spacing

/-- Theorem stating the area of Aaron's playground is 400 square yards. -/
theorem aarons_playground_area :
  ∃ (p : Playground),
    p.total_posts = 24 ∧
    p.post_spacing = 5 ∧
    p.long_side_posts = 3 * p.short_side_posts - 2 ∧
    playground_area p = 400 := by
  sorry


end NUMINAMATH_CALUDE_aarons_playground_area_l3482_348296


namespace NUMINAMATH_CALUDE_no_real_solution_l3482_348267

theorem no_real_solution : ¬∃ (x : ℝ), 
  (x + 5 > 0) ∧ 
  (x - 3 > 0) ∧ 
  (x^2 - 8*x + 7 > 0) ∧ 
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 7)) := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l3482_348267


namespace NUMINAMATH_CALUDE_company_production_l3482_348266

/-- Calculates the total number of parts produced by a company given specific production conditions. -/
def totalPartsProduced (initialPartsPerDay : ℕ) (initialDays : ℕ) (increasedPartsPerDay : ℕ) (extraParts : ℕ) : ℕ :=
  let totalInitialParts := initialPartsPerDay * initialDays
  let increasedProduction := initialPartsPerDay + increasedPartsPerDay
  let additionalDays := extraParts / increasedPartsPerDay
  let totalIncreasedParts := increasedProduction * additionalDays
  totalInitialParts + totalIncreasedParts

/-- Theorem stating that under given conditions, the company produces 1107 parts. -/
theorem company_production : 
  totalPartsProduced 40 3 7 150 = 1107 := by
  sorry

#eval totalPartsProduced 40 3 7 150

end NUMINAMATH_CALUDE_company_production_l3482_348266


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l3482_348239

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 0.2

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 720 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l3482_348239


namespace NUMINAMATH_CALUDE_case_cost_is_nine_l3482_348224

/-- The cost of a case of paper towels -/
def case_cost (num_rolls : ℕ) (individual_roll_cost : ℚ) (savings_percent : ℚ) : ℚ :=
  num_rolls * (individual_roll_cost * (1 - savings_percent / 100))

/-- Theorem stating the cost of a case of 12 rolls is $9 -/
theorem case_cost_is_nine :
  case_cost 12 1 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_case_cost_is_nine_l3482_348224


namespace NUMINAMATH_CALUDE_valid_systematic_sample_l3482_348203

/-- Represents a systematic sample of student numbers -/
def SystematicSample (n : ℕ) (k : ℕ) (sample : Finset ℕ) : Prop :=
  ∃ (start : ℕ) (step : ℕ), 
    sample = Finset.image (fun i => start + i * step) (Finset.range k) ∧
    start ≤ n ∧
    ∀ i ∈ Finset.range k, start + i * step ≤ n

/-- The given sample is a valid systematic sample -/
theorem valid_systematic_sample :
  SystematicSample 50 5 {5, 15, 25, 35, 45} :=
by sorry

end NUMINAMATH_CALUDE_valid_systematic_sample_l3482_348203


namespace NUMINAMATH_CALUDE_lily_paint_cans_l3482_348297

/-- Given the initial paint coverage, lost cans, and remaining coverage, 
    calculate the number of cans used for the remaining rooms --/
def paint_cans_used (initial_coverage : ℕ) (lost_cans : ℕ) (remaining_coverage : ℕ) : ℕ :=
  (remaining_coverage * lost_cans) / (initial_coverage - remaining_coverage)

/-- Theorem stating that under the given conditions, 16 cans were used for 32 rooms --/
theorem lily_paint_cans : paint_cans_used 40 4 32 = 16 := by
  sorry

end NUMINAMATH_CALUDE_lily_paint_cans_l3482_348297


namespace NUMINAMATH_CALUDE_ratio_to_thirteen_l3482_348294

theorem ratio_to_thirteen : ∃ x : ℚ, (5 : ℚ) / 1 = x / 13 ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_thirteen_l3482_348294


namespace NUMINAMATH_CALUDE_not_mapping_A_to_B_l3482_348272

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {y | 1 ≤ y ∧ y ≤ 4}

def f (x : ℝ) : ℝ := 4 - x^2

theorem not_mapping_A_to_B :
  ¬(∀ x ∈ A, f x ∈ B) :=
by sorry

end NUMINAMATH_CALUDE_not_mapping_A_to_B_l3482_348272


namespace NUMINAMATH_CALUDE_equation_solutions_l3482_348231

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 25 = 0 ↔ x = 5/2 ∨ x = -5/2) ∧
  (∀ x : ℝ, (x + 1)^3 = -27 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3482_348231


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l3482_348271

theorem six_digit_divisibility (a b c : ℕ) 
  (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  ∃ k : ℤ, (a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + c : ℤ) = 1001 * k :=
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l3482_348271


namespace NUMINAMATH_CALUDE_cards_given_to_friends_l3482_348279

theorem cards_given_to_friends (initial_cards : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 13 → remaining_cards = 4 → initial_cards - remaining_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_friends_l3482_348279


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l3482_348295

theorem decimal_arithmetic : 0.5 - 0.03 + 0.007 = 0.477 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l3482_348295


namespace NUMINAMATH_CALUDE_christen_peeled_24_l3482_348290

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initialPile : ℕ
  homerRate : ℕ
  christenJoinTime : ℕ
  christenRate : ℕ
  alexExtra : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledCount (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 24 potatoes in the given scenario -/
theorem christen_peeled_24 (scenario : PotatoPeeling) 
  (h1 : scenario.initialPile = 60)
  (h2 : scenario.homerRate = 4)
  (h3 : scenario.christenJoinTime = 6)
  (h4 : scenario.christenRate = 6)
  (h5 : scenario.alexExtra = 2) :
  christenPeeledCount scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_24_l3482_348290


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3482_348233

-- Define the inequality
def inequality (x m : ℝ) : Prop :=
  |x + 1| + |x - 2| + m - 7 > 0

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  {x : ℝ | inequality x m}

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  solution_set m = Set.univ → m > 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3482_348233


namespace NUMINAMATH_CALUDE_max_value_sum_sqrt_l3482_348221

theorem max_value_sum_sqrt (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ Real.sqrt 63 ∧
  (Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) = Real.sqrt 63 ↔ x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_sqrt_l3482_348221


namespace NUMINAMATH_CALUDE_inequality_relation_l3482_348242

theorem inequality_relation : 
  ∃ (x : ℝ), (x^2 - x - 6 > 0 ∧ x ≥ -5) ∧ 
  ∀ (y : ℝ), y < -5 → y^2 - y - 6 > 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l3482_348242


namespace NUMINAMATH_CALUDE_average_of_fifths_and_tenths_l3482_348229

/-- The average of two rational numbers -/
def average (a b : ℚ) : ℚ := (a + b) / 2

/-- Theorem: If the average of 1/5 and 1/10 is 1/x, then x = 20/3 -/
theorem average_of_fifths_and_tenths (x : ℚ) :
  average (1/5) (1/10) = 1/x → x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_fifths_and_tenths_l3482_348229


namespace NUMINAMATH_CALUDE_distance_to_line_is_sqrt_17_l3482_348278

/-- The distance from a point to a line in 3D space --/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating that the distance from (2, 0, -1) to the line passing through (1, 3, 1) and (3, -1, 5) is √17 --/
theorem distance_to_line_is_sqrt_17 :
  distance_point_to_line (2, 0, -1) (1, 3, 1) (3, -1, 5) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_is_sqrt_17_l3482_348278


namespace NUMINAMATH_CALUDE_direction_vector_y_component_l3482_348202

/-- Given a line determined by two points in 2D space, prove that if the direction vector
    has a specific x-component, then its y-component has a specific value. -/
theorem direction_vector_y_component 
  (p1 p2 : ℝ × ℝ) 
  (h1 : p1 = (-1, -1)) 
  (h2 : p2 = (3, 4)) 
  (direction : ℝ × ℝ) 
  (h_x_component : direction.1 = 3) : 
  direction.2 = 15/4 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_y_component_l3482_348202


namespace NUMINAMATH_CALUDE_x_value_proof_l3482_348264

theorem x_value_proof (x : ℝ) : -(-(-(-x))) = -4 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3482_348264


namespace NUMINAMATH_CALUDE_bottle_caps_per_box_l3482_348253

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) 
  (h1 : total_caps = 316) (h2 : num_boxes = 79) : 
  total_caps / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_per_box_l3482_348253


namespace NUMINAMATH_CALUDE_regular_scoop_cost_l3482_348220

/-- The cost of ice cream scoops for the Martin family --/
def ice_cream_cost (regular_scoop : ℚ) : Prop :=
  let kiddie_scoop : ℚ := 3
  let double_scoop : ℚ := 6
  let total_cost : ℚ := 32
  let num_regular : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie : ℕ := 2   -- Two children
  let num_double : ℕ := 3   -- Three teenagers
  (num_regular * regular_scoop + 
   num_kiddie * kiddie_scoop + 
   num_double * double_scoop) = total_cost

theorem regular_scoop_cost : 
  ∃ (regular_scoop : ℚ), ice_cream_cost regular_scoop ∧ regular_scoop = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_scoop_cost_l3482_348220


namespace NUMINAMATH_CALUDE_complex_sum_problem_l3482_348269

theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 5 → 
  t = -p - r → 
  (p + q * I) + (r + s * I) + (t + u * I) = 4 * I → 
  s + u = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l3482_348269


namespace NUMINAMATH_CALUDE_class_average_l3482_348200

theorem class_average (total_students : ℕ) 
                      (top_scorers : ℕ) 
                      (zero_scorers : ℕ) 
                      (top_score : ℕ) 
                      (rest_average : ℕ) : 
  total_students = 25 →
  top_scorers = 3 →
  zero_scorers = 5 →
  top_score = 95 →
  rest_average = 45 →
  (top_scorers * top_score + 
   (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l3482_348200


namespace NUMINAMATH_CALUDE_debt_installments_l3482_348261

theorem debt_installments (first_payment : ℕ) (additional_amount : ℕ) (average_payment : ℕ) : 
  let n := (12 * first_payment + 780) / 15
  let remaining_payment := first_payment + additional_amount
  12 * first_payment + (n - 12) * remaining_payment = n * average_payment →
  n = 52 :=
by
  sorry

#check debt_installments 410 65 460

end NUMINAMATH_CALUDE_debt_installments_l3482_348261


namespace NUMINAMATH_CALUDE_no_solution_floor_plus_x_l3482_348205

theorem no_solution_floor_plus_x :
  ¬ ∃ x : ℝ, ⌊x⌋ + x = 15.3 := by sorry

end NUMINAMATH_CALUDE_no_solution_floor_plus_x_l3482_348205


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3482_348280

theorem trigonometric_identity (α β γ : Real) 
  (h : (Real.sin (β + γ) * Real.sin (γ + α)) / (Real.cos α * Real.cos γ) = 4/9) :
  (Real.sin (β + γ) * Real.sin (γ + α)) / (Real.cos (α + β + γ) * Real.cos γ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3482_348280


namespace NUMINAMATH_CALUDE_books_sum_l3482_348291

/-- The number of books Sam has -/
def sam_books : ℕ := 110

/-- The number of books Joan has -/
def joan_books : ℕ := 102

/-- The total number of books Sam and Joan have together -/
def total_books : ℕ := sam_books + joan_books

theorem books_sum :
  total_books = 212 :=
by sorry

end NUMINAMATH_CALUDE_books_sum_l3482_348291


namespace NUMINAMATH_CALUDE_colored_graph_color_bound_l3482_348270

/-- A graph with colored edges satisfying certain properties -/
structure ColoredGraph where
  n : ℕ  -- number of vertices
  c : ℕ  -- number of colors
  edge_count : ℕ  -- number of edges
  edge_count_lower_bound : edge_count ≥ n^2 / 10
  no_incident_same_color : Bool  -- property that no two incident edges have the same color
  no_same_color_10_cycle : Bool  -- property that no cycles of size 10 have the same set of colors

/-- Main theorem: There exists a constant k such that c ≥ k * n^(8/5) for any colored graph satisfying the given properties -/
theorem colored_graph_color_bound (G : ColoredGraph) :
  ∃ (k : ℝ), G.c ≥ k * G.n^(8/5) := by
  sorry

end NUMINAMATH_CALUDE_colored_graph_color_bound_l3482_348270


namespace NUMINAMATH_CALUDE_units_digit_of_sum_is_seven_l3482_348225

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_less_than_10 : hundreds < 10
  tens_less_than_10 : tens < 10
  units_less_than_10 : units < 10
  hundreds_not_zero : hundreds ≠ 0

/-- The condition that the hundreds digit is 3 less than twice the units digit -/
def hundreds_units_relation (n : ThreeDigitNumber) : Prop :=
  n.hundreds = 2 * n.units - 3

/-- The value of the three-digit number -/
def number_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed number -/
def reversed_number (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The theorem to be proved -/
theorem units_digit_of_sum_is_seven (n : ThreeDigitNumber) 
  (h : hundreds_units_relation n) : 
  (number_value n + reversed_number n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_is_seven_l3482_348225


namespace NUMINAMATH_CALUDE_system_solution_l3482_348277

theorem system_solution (x y : ℝ) : 
  x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97 ↔ 
  ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = -3 ∧ y = -2) ∨ (x = -2 ∧ y = -3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3482_348277


namespace NUMINAMATH_CALUDE_soccer_team_probability_l3482_348249

theorem soccer_team_probability (total_players defenders : ℕ) 
  (h1 : total_players = 12)
  (h2 : defenders = 6) :
  (Nat.choose defenders 2 : ℚ) / (Nat.choose total_players 2) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_probability_l3482_348249


namespace NUMINAMATH_CALUDE_skating_minutes_proof_l3482_348252

/-- The number of minutes Gage skated per day for the first 4 days -/
def minutes_per_day_first_4 : ℕ := 70

/-- The number of minutes Gage skated per day for the next 4 days -/
def minutes_per_day_next_4 : ℕ := 100

/-- The total number of days Gage skated -/
def total_days : ℕ := 9

/-- The desired average number of minutes skated per day -/
def desired_average : ℕ := 100

/-- The number of minutes Gage must skate on the ninth day to achieve the desired average -/
def minutes_on_ninth_day : ℕ := 220

theorem skating_minutes_proof :
  minutes_on_ninth_day = 
    total_days * desired_average - 
    (4 * minutes_per_day_first_4 + 4 * minutes_per_day_next_4) := by
  sorry

end NUMINAMATH_CALUDE_skating_minutes_proof_l3482_348252


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3482_348230

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem sum_of_repeating_decimals :
  SingleDigitRepeatingDecimal 0 1 + TwoDigitRepeatingDecimal 0 1 = 4 / 33 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3482_348230


namespace NUMINAMATH_CALUDE_exists_question_with_different_answers_l3482_348287

/-- Represents a person who always tells the truth -/
structure TruthTeller where
  name : String
  always_truthful : Bool

/-- Represents a question that can be asked -/
inductive Question where
  | count_questions : Question

/-- Represents the state of the conversation -/
structure ConversationState where
  questions_asked : Nat

/-- The answer given by a TruthTeller to a Question in a given ConversationState -/
def answer (person : TruthTeller) (q : Question) (state : ConversationState) : Nat :=
  match q with
  | Question.count_questions => state.questions_asked

/-- Theorem stating that there exists a question that can have different truthful answers when asked twice -/
theorem exists_question_with_different_answers (ilya : TruthTeller) 
    (h_truthful : ilya.always_truthful = true) :
    ∃ (q : Question) (s1 s2 : ConversationState), 
      s1 ≠ s2 ∧ answer ilya q s1 ≠ answer ilya q s2 := by
  sorry


end NUMINAMATH_CALUDE_exists_question_with_different_answers_l3482_348287


namespace NUMINAMATH_CALUDE_share_calculation_l3482_348292

theorem share_calculation (total_amount : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) : 
  total_amount = 15800 → 
  ratio_a = 5 →
  ratio_b = 9 →
  ratio_c = 6 →
  ratio_d = 5 →
  (ratio_a * total_amount / (ratio_a + ratio_b + ratio_c + ratio_d) + 
   ratio_c * total_amount / (ratio_a + ratio_b + ratio_c + ratio_d)) = 6952 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l3482_348292


namespace NUMINAMATH_CALUDE_quadruple_base_triple_exponent_l3482_348210

theorem quadruple_base_triple_exponent (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) :
  (4 * a) ^ (3 * b) = a ^ b * x ^ b → x = 64 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadruple_base_triple_exponent_l3482_348210


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3482_348299

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 5) 
  (h2 : x * y = 4) : 
  x^2 + y^2 = 33 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3482_348299


namespace NUMINAMATH_CALUDE_power_function_increasing_condition_l3482_348238

theorem power_function_increasing_condition (m : ℝ) : 
  (m^2 - m - 1 = 1) ∧ (m^2 + m - 3 > 0) ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_increasing_condition_l3482_348238


namespace NUMINAMATH_CALUDE_find_a_range_of_m_l3482_348288

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 2
theorem find_a (a : ℝ) : 
  (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 := by sorry

-- Theorem 2: Prove the range of m
theorem range_of_m : 
  ∀ x, f 2 x + f 2 (x + 5) ≥ 5 ∧ 
  ∀ ε > 0, ∃ x, f 2 x + f 2 (x + 5) < 5 + ε := by sorry

end NUMINAMATH_CALUDE_find_a_range_of_m_l3482_348288


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3482_348216

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 4) :
  π * r₁^2 - π * r₂^2 = 84 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3482_348216


namespace NUMINAMATH_CALUDE_triangle_area_tripled_sides_l3482_348289

/-- Given a triangle with sides a and b and included angle θ,
    if we triple the sides to 3a and 3b while keeping θ unchanged,
    then the new area A' is 9 times the original area A. -/
theorem triangle_area_tripled_sides (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let A := (a * b * Real.sin θ) / 2
  let A' := (3 * a * 3 * b * Real.sin θ) / 2
  A' = 9 * A := by sorry

end NUMINAMATH_CALUDE_triangle_area_tripled_sides_l3482_348289


namespace NUMINAMATH_CALUDE_min_omega_value_l3482_348246

theorem min_omega_value (f : ℝ → ℝ) (ω φ : ℝ) : 
  (∀ x, f x = Real.sin (ω * x + φ)) →
  ω > 0 →
  abs φ < π / 2 →
  f 0 = 1 / 2 →
  (∀ x, f x ≤ f (π / 12)) →
  ω ≥ 4 ∧ (∀ ω', ω' > 0 ∧ ω' < 4 → 
    ∃ φ', abs φ' < π / 2 ∧ 
    Real.sin φ' = 1 / 2 ∧ 
    ∃ x, Real.sin (ω' * x + φ') > Real.sin (ω' * π / 12 + φ')) :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l3482_348246


namespace NUMINAMATH_CALUDE_population_change_l3482_348243

theorem population_change (initial_population : ℕ) 
  (increase_rate : ℚ) (decrease_rate : ℚ) : 
  initial_population = 10000 →
  increase_rate = 20 / 100 →
  decrease_rate = 20 / 100 →
  (initial_population * (1 + increase_rate) * (1 - decrease_rate)).floor = 9600 := by
sorry

end NUMINAMATH_CALUDE_population_change_l3482_348243


namespace NUMINAMATH_CALUDE_min_days_to_plant_100_trees_l3482_348275

def trees_planted (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem min_days_to_plant_100_trees :
  (∃ n : ℕ, trees_planted n ≥ 100) ∧
  (∀ n : ℕ, trees_planted n ≥ 100 → n ≥ 6) ∧
  trees_planted 6 ≥ 100 :=
sorry

end NUMINAMATH_CALUDE_min_days_to_plant_100_trees_l3482_348275


namespace NUMINAMATH_CALUDE_production_scaling_l3482_348298

theorem production_scaling (x z : ℝ) (h : x > 0) :
  let production (n : ℝ) := n * n * n * (2 / n)
  production x = 2 * x^2 →
  production z = 2 * z^3 / x :=
by sorry

end NUMINAMATH_CALUDE_production_scaling_l3482_348298


namespace NUMINAMATH_CALUDE_division_remainder_l3482_348223

theorem division_remainder (x : ℕ) (h : 23 / x = 7) : 23 % x = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3482_348223


namespace NUMINAMATH_CALUDE_car_average_speed_l3482_348284

/-- Given a car that travels 80 km in the first hour and 40 km in the second hour,
    prove that its average speed is 60 km/h. -/
theorem car_average_speed (distance_first_hour : ℝ) (distance_second_hour : ℝ)
    (h1 : distance_first_hour = 80)
    (h2 : distance_second_hour = 40) :
    (distance_first_hour + distance_second_hour) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l3482_348284


namespace NUMINAMATH_CALUDE_exam_score_problem_l3482_348273

theorem exam_score_problem (mean : ℝ) (high_score : ℝ) (std_dev : ℝ) :
  mean = 74 ∧ high_score = 98 ∧ high_score = mean + 3 * std_dev →
  mean - 2 * std_dev = 58 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3482_348273


namespace NUMINAMATH_CALUDE_shop_profit_percentage_l3482_348245

/-- Calculates the total profit percentage for a shop selling two types of items -/
theorem shop_profit_percentage
  (cost_price_ratio_A : ℝ)
  (cost_price_ratio_B : ℝ)
  (quantity_A : ℕ)
  (quantity_B : ℕ)
  (price_A : ℝ)
  (price_B : ℝ)
  (h1 : cost_price_ratio_A = 0.95)
  (h2 : cost_price_ratio_B = 0.90)
  (h3 : quantity_A = 100)
  (h4 : quantity_B = 150)
  (h5 : price_A = 50)
  (h6 : price_B = 60) :
  let profit_A := quantity_A * price_A * (1 - cost_price_ratio_A)
  let profit_B := quantity_B * price_B * (1 - cost_price_ratio_B)
  let total_profit := profit_A + profit_B
  let total_cost := quantity_A * price_A * cost_price_ratio_A + quantity_B * price_B * cost_price_ratio_B
  let profit_percentage := (total_profit / total_cost) * 100
  ∃ ε > 0, |profit_percentage - 8.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_shop_profit_percentage_l3482_348245


namespace NUMINAMATH_CALUDE_square_difference_49_16_l3482_348263

theorem square_difference_49_16 : 49^2 - 16^2 = 2145 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_49_16_l3482_348263


namespace NUMINAMATH_CALUDE_complex_division_result_l3482_348247

theorem complex_division_result : ∃ (i : ℂ), i * i = -1 ∧ (2 : ℂ) / (1 - i) = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l3482_348247


namespace NUMINAMATH_CALUDE_cistern_leak_time_l3482_348276

/-- Proves that if a cistern can be filled by pipe A in 16 hours and both pipes A and B together fill the cistern in 80.00000000000001 hours, then pipe B alone can leak out the full cistern in 80 hours. -/
theorem cistern_leak_time (fill_time_A : ℝ) (fill_time_both : ℝ) (leak_time_B : ℝ) : 
  fill_time_A = 16 →
  fill_time_both = 80.00000000000001 →
  (1 / fill_time_A) - (1 / leak_time_B) = 1 / fill_time_both →
  leak_time_B = 80 := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l3482_348276


namespace NUMINAMATH_CALUDE_original_number_proof_l3482_348282

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3482_348282


namespace NUMINAMATH_CALUDE_polynomial_property_l3482_348219

-- Define the polynomial Q(x)
def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

-- Define the properties of the polynomial
theorem polynomial_property (d e f : ℝ) :
  -- The y-intercept is 5
  Q d e f 0 = 5 →
  -- The mean of zeros equals the product of zeros
  -d/3 = -f →
  -- The mean of zeros equals the sum of coefficients
  -d/3 = 1 + d + e + f →
  -- Conclusion: e = -26
  e = -26 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l3482_348219


namespace NUMINAMATH_CALUDE_no_solutions_abs_x_eq_3_abs_x_plus_2_l3482_348212

theorem no_solutions_abs_x_eq_3_abs_x_plus_2 :
  ∀ x : ℝ, ¬(|x| = 3 * (|x| + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solutions_abs_x_eq_3_abs_x_plus_2_l3482_348212


namespace NUMINAMATH_CALUDE_john_ate_three_cookies_l3482_348254

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John has left -/
def cookies_left : ℕ := 21

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := dozens_bought * dozen - cookies_left

theorem john_ate_three_cookies : cookies_eaten = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_ate_three_cookies_l3482_348254


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3482_348250

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ × ℤ) := {(x, y, z) | 5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30}
  S = {(1, 5, 0), (1, -5, 0), (-1, 5, 0), (-1, -5, 0)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3482_348250


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l3482_348236

theorem cos_four_arccos_two_fifths : 
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l3482_348236


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3482_348240

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 2700 →
  width = 30 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3482_348240


namespace NUMINAMATH_CALUDE_gcf_360_180_l3482_348237

theorem gcf_360_180 : Nat.gcd 360 180 = 180 := by
  sorry

end NUMINAMATH_CALUDE_gcf_360_180_l3482_348237


namespace NUMINAMATH_CALUDE_negative_inequality_l3482_348260

theorem negative_inequality (x y : ℝ) (h : x < y) : -x > -y := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l3482_348260


namespace NUMINAMATH_CALUDE_smallest_w_l3482_348201

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_w : 
  ∃ (w : ℕ), w > 0 ∧ 
  is_factor (2^6) (1916 * w) ∧
  is_factor (3^4) (1916 * w) ∧
  is_factor (5^3) (1916 * w) ∧
  is_factor (7^3) (1916 * w) ∧
  is_factor (11^3) (1916 * w) ∧
  ∀ (x : ℕ), x > 0 ∧ 
    is_factor (2^6) (1916 * x) ∧
    is_factor (3^4) (1916 * x) ∧
    is_factor (5^3) (1916 * x) ∧
    is_factor (7^3) (1916 * x) ∧
    is_factor (11^3) (1916 * x) →
    w ≤ x ∧
  w = 74145392000 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l3482_348201


namespace NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l3482_348248

/-- Given that x varies inversely as the square of y and directly as the cube of z,
    prove that x = 64/243 when y = 6 and z = 4, given the initial conditions x = 1, y = 2, and z = 3. -/
theorem inverse_square_direct_cube_relation
  (k : ℚ)
  (h : ∀ (x y z : ℚ), x = k * z^3 / y^2)
  (h_init : 1 = k * 3^3 / 2^2) :
  k * 4^3 / 6^2 = 64/243 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l3482_348248


namespace NUMINAMATH_CALUDE_vaishalis_hats_l3482_348211

/-- The number of hats with three stripes each that Vaishali has -/
def hats_with_three_stripes : ℕ := sorry

/-- The number of hats with four stripes each that Vaishali has -/
def hats_with_four_stripes : ℕ := 3

/-- The number of hats with no stripes that Vaishali has -/
def hats_with_no_stripes : ℕ := 6

/-- The number of hats with five stripes each that Vaishali has -/
def hats_with_five_stripes : ℕ := 2

/-- The total number of stripes on all of Vaishali's hats -/
def total_stripes : ℕ := 34

/-- Theorem stating that the number of hats with three stripes is 4 -/
theorem vaishalis_hats : hats_with_three_stripes = 4 := by
  sorry

end NUMINAMATH_CALUDE_vaishalis_hats_l3482_348211


namespace NUMINAMATH_CALUDE_regression_estimate_l3482_348293

/-- Represents a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents a data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Parameters of the regression problem -/
structure RegressionProblem where
  original_regression : LinearRegression
  original_mean_x : ℝ
  removed_points : List DataPoint
  new_slope : ℝ

theorem regression_estimate (problem : RegressionProblem) :
  let new_intercept := problem.original_regression.intercept +
    problem.original_regression.slope * problem.original_mean_x -
    problem.new_slope * problem.original_mean_x
  let new_regression := LinearRegression.mk problem.new_slope new_intercept
  let estimate_at_6 := new_regression.slope * 6 + new_regression.intercept
  problem.original_regression = LinearRegression.mk 1.5 1 →
  problem.original_mean_x = 2 →
  problem.removed_points = [DataPoint.mk 2.6 2.8, DataPoint.mk 1.4 5.2] →
  problem.new_slope = 1.4 →
  estimate_at_6 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_regression_estimate_l3482_348293


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_smallest_integer_is_2395_l3482_348214

theorem smallest_integer_with_remainder_one (k : ℕ) : 
  (k > 1) ∧ 
  (k % 19 = 1) ∧ 
  (k % 14 = 1) ∧ 
  (k % 9 = 1) → 
  k ≥ 2395 :=
by sorry

theorem smallest_integer_is_2395 : 
  (2395 > 1) ∧ 
  (2395 % 19 = 1) ∧ 
  (2395 % 14 = 1) ∧ 
  (2395 % 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_smallest_integer_is_2395_l3482_348214


namespace NUMINAMATH_CALUDE_public_swimming_pool_attendance_l3482_348206

/-- Proves the total number of people who used the public swimming pool -/
theorem public_swimming_pool_attendance 
  (child_price : ℚ) 
  (adult_price : ℚ) 
  (total_receipts : ℚ) 
  (num_children : ℕ) : 
  child_price = 3/2 →
  adult_price = 9/4 →
  total_receipts = 1422 →
  num_children = 388 →
  ∃ (num_adults : ℕ), 
    num_adults * adult_price + num_children * child_price = total_receipts ∧
    num_adults + num_children = 761 := by
  sorry

end NUMINAMATH_CALUDE_public_swimming_pool_attendance_l3482_348206


namespace NUMINAMATH_CALUDE_roots_location_l3482_348208

theorem roots_location (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ (x₁ x₂ : ℝ), 
    (a < x₁ ∧ x₁ < b) ∧ 
    (b < x₂ ∧ x₂ < c) ∧ 
    (∀ x, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_roots_location_l3482_348208


namespace NUMINAMATH_CALUDE_restaurant_group_size_l3482_348251

theorem restaurant_group_size (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) (children : ℕ) : 
  adults = 2 →
  meal_cost = 3 →
  total_bill = 21 →
  children * meal_cost + adults * meal_cost = total_bill →
  children = 5 := by
sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l3482_348251


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l3482_348217

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l3482_348217


namespace NUMINAMATH_CALUDE_propositions_truth_l3482_348235

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem propositions_truth : 
  (∀ a : ℝ, a^2 ≥ 0) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ ∧ f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l3482_348235


namespace NUMINAMATH_CALUDE_roxy_plants_l3482_348286

def plants_problem (initial_flowering : ℕ) (initial_fruiting_ratio : ℕ) 
  (bought_fruiting : ℕ) (given_away_flowering : ℕ) (given_away_fruiting : ℕ) 
  (final_total : ℕ) : Prop :=
  ∃ (bought_flowering : ℕ),
    let initial_fruiting := initial_flowering * initial_fruiting_ratio
    let initial_total := initial_flowering + initial_fruiting
    let after_buying := initial_total + bought_flowering + bought_fruiting
    let after_giving := after_buying - given_away_flowering - given_away_fruiting
    after_giving = final_total ∧ bought_flowering = 3

theorem roxy_plants : plants_problem 7 2 2 1 4 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_plants_l3482_348286


namespace NUMINAMATH_CALUDE_white_area_of_sign_l3482_348226

/-- Represents a block letter in the sign --/
structure BlockLetter where
  width : ℕ
  height : ℕ
  stroke_width : ℕ
  covered_area : ℕ

/-- Represents the sign --/
structure Sign where
  width : ℕ
  height : ℕ
  letters : List BlockLetter

def m_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def a_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def t_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 24
}

def h_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def math_sign : Sign := {
  width := 28,
  height := 8,
  letters := [m_letter, a_letter, t_letter, h_letter]
}

theorem white_area_of_sign (s : Sign) : 
  s.width * s.height - (s.letters.map BlockLetter.covered_area).sum = 80 :=
by sorry

end NUMINAMATH_CALUDE_white_area_of_sign_l3482_348226


namespace NUMINAMATH_CALUDE_burgers_remaining_l3482_348207

theorem burgers_remaining (total_burgers : ℕ) (slices_per_burger : ℕ) 
  (friend1 friend2 friend3 friend4 friend5 : ℚ) : 
  total_burgers = 5 →
  slices_per_burger = 8 →
  friend1 = 3 / 8 →
  friend2 = 8 / 8 →
  friend3 = 5 / 8 →
  friend4 = 11 / 8 →
  friend5 = 6 / 8 →
  (total_burgers * slices_per_burger : ℚ) - (friend1 + friend2 + friend3 + friend4 + friend5) * slices_per_burger = 7 := by
  sorry

end NUMINAMATH_CALUDE_burgers_remaining_l3482_348207


namespace NUMINAMATH_CALUDE_quadratic_max_l3482_348222

theorem quadratic_max (a b c : ℝ) (x₀ : ℝ) (h1 : a < 0) (h2 : 2 * a * x₀ + b = 0) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  ∀ x : ℝ, f x ≤ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_l3482_348222
