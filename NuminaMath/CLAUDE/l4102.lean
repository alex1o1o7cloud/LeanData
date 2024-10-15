import Mathlib

namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l4102_410290

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 2| > 3
def q (x : ℝ) : Prop := x > 5

-- Statement to prove
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l4102_410290


namespace NUMINAMATH_CALUDE_min_digit_sum_of_sum_l4102_410278

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the sum of digits of a natural number -/
def digitSum (n : Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem min_digit_sum_of_sum (a b : ThreeDigitNumber) 
  (h1 : a.hundreds < 5)
  (h2 : a.hundreds ≠ b.hundreds ∧ a.hundreds ≠ b.tens ∧ a.hundreds ≠ b.ones ∧
        a.tens ≠ b.hundreds ∧ a.tens ≠ b.tens ∧ a.tens ≠ b.ones ∧
        a.ones ≠ b.hundreds ∧ a.ones ≠ b.tens ∧ a.ones ≠ b.ones)
  (h3 : (a.value + b.value) < 1000) :
  15 ≤ digitSum (a.value + b.value) :=
sorry

end NUMINAMATH_CALUDE_min_digit_sum_of_sum_l4102_410278


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l4102_410259

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l4102_410259


namespace NUMINAMATH_CALUDE_amount_ratio_l4102_410275

theorem amount_ratio (total : ℚ) (b_amt : ℚ) (a_fraction : ℚ) :
  total = 1440 →
  b_amt = 270 →
  a_fraction = 1/3 →
  ∃ (c_amt : ℚ),
    total = a_fraction * b_amt + b_amt + c_amt ∧
    b_amt / c_amt = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_amount_ratio_l4102_410275


namespace NUMINAMATH_CALUDE_N_is_composite_l4102_410208

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Nat.Prime N := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l4102_410208


namespace NUMINAMATH_CALUDE_power_equation_solution_l4102_410289

theorem power_equation_solution (n : ℕ) : 3^n = 3 * 9^5 * 81^3 → n = 23 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l4102_410289


namespace NUMINAMATH_CALUDE_inequality_with_gcd_l4102_410209

theorem inequality_with_gcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) :
  (a + 1) / (b + 1 : ℝ) ≤ Nat.gcd a b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_gcd_l4102_410209


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4102_410287

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  2 * X^2 - 21 * X + 55 = (X + 3) * q + 136 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4102_410287


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l4102_410254

/-- The total number of bottles and fruits in a grocery store -/
def total_items (regular_soda diet_soda sparkling_water orange_juice cranberry_juice apples oranges bananas pears : ℕ) : ℕ :=
  regular_soda + diet_soda + sparkling_water + orange_juice + cranberry_juice + apples + oranges + bananas + pears

/-- Theorem stating the total number of items in the grocery store -/
theorem grocery_store_inventory : 
  total_items 130 88 65 47 27 102 88 74 45 = 666 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l4102_410254


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l4102_410270

/-- Given a current speed and a speed against the current, calculates the speed with the current -/
def speed_with_current (current_speed : ℝ) (speed_against_current : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem: Given the specified conditions, the man's speed with the current is 15 km/hr -/
theorem mans_speed_with_current :
  let current_speed : ℝ := 2.8
  let speed_against_current : ℝ := 9.4
  speed_with_current current_speed speed_against_current = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l4102_410270


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l4102_410247

theorem fixed_point_on_line (m : ℝ) : 
  let x : ℝ := -1
  let y : ℝ := -1/2
  (m^2 + 6*m + 3) * x - (2*m^2 + 18*m + 2) * y - 3*m + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l4102_410247


namespace NUMINAMATH_CALUDE_young_in_specific_sample_l4102_410232

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  young_population : ℕ
  sample_size : ℕ

/-- Calculates the number of young people in a stratified sample -/
def young_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.young_population) / s.total_population

/-- Theorem stating the number of young people in the specific stratified sample -/
theorem young_in_specific_sample :
  let s : StratifiedSample := {
    total_population := 108,
    young_population := 51,
    sample_size := 36
  }
  young_in_sample s = 17 := by
  sorry

end NUMINAMATH_CALUDE_young_in_specific_sample_l4102_410232


namespace NUMINAMATH_CALUDE_adams_age_problem_l4102_410243

theorem adams_age_problem :
  ∃! x : ℕ,
    x > 0 ∧
    ∃ m : ℕ, x - 2 = m ^ 2 ∧
    ∃ n : ℕ, x + 2 = n ^ 3 ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_adams_age_problem_l4102_410243


namespace NUMINAMATH_CALUDE_cos_plus_sin_implies_cos_double_angle_l4102_410236

theorem cos_plus_sin_implies_cos_double_angle 
  (θ : ℝ) (h : Real.cos θ + Real.sin θ = 7/5) : 
  Real.cos (2 * θ) = -527/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_plus_sin_implies_cos_double_angle_l4102_410236


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l4102_410241

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 10 = 36 → Nat.gcd n 10 = 5 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l4102_410241


namespace NUMINAMATH_CALUDE_food_rent_ratio_l4102_410244

/-- Esperanza's monthly financial situation -/
structure EsperanzaFinances where
  rent : ℝ
  food : ℝ
  mortgage : ℝ
  savings : ℝ
  taxes : ℝ
  salary : ℝ

/-- Conditions for Esperanza's finances -/
def validFinances (e : EsperanzaFinances) : Prop :=
  e.rent = 600 ∧
  e.mortgage = 3 * e.food ∧
  e.savings = 2000 ∧
  e.taxes = 2/5 * e.savings ∧
  e.salary = 4840 ∧
  e.salary = e.rent + e.food + e.mortgage + e.savings + e.taxes

/-- The theorem to prove -/
theorem food_rent_ratio (e : EsperanzaFinances) 
  (h : validFinances e) : e.food / e.rent = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_food_rent_ratio_l4102_410244


namespace NUMINAMATH_CALUDE_three_turns_sufficient_l4102_410285

/-- Represents a five-digit number with distinct digits -/
structure FiveDigitNumber where
  digits : Fin 5 → Fin 10
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- Represents a turn where positions are selected and digits are revealed -/
structure Turn where
  positions : Set (Fin 5)
  revealed_digits : Set (Fin 10)

/-- Represents the process of guessing the number -/
def guess_number (n : FiveDigitNumber) (turns : List Turn) : Prop :=
  ∀ m : FiveDigitNumber, 
    (∀ t ∈ turns, {n.digits i | i ∈ t.positions} = t.revealed_digits) →
    (∀ t ∈ turns, {m.digits i | i ∈ t.positions} = t.revealed_digits) →
    n = m

/-- The main theorem stating that 3 turns are sufficient -/
theorem three_turns_sufficient :
  ∃ strategy : List Turn, 
    strategy.length ≤ 3 ∧ 
    ∀ n : FiveDigitNumber, guess_number n strategy :=
sorry

end NUMINAMATH_CALUDE_three_turns_sufficient_l4102_410285


namespace NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l4102_410268

theorem man_son_age_difference : ℕ → ℕ → Prop :=
  fun son_age man_age =>
    (son_age = 22) →
    (man_age + 2 = 2 * (son_age + 2)) →
    (man_age - son_age = 24)

-- The proof is omitted
theorem man_son_age_difference_proof : ∃ (son_age man_age : ℕ), man_son_age_difference son_age man_age :=
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l4102_410268


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l4102_410211

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 30*|x| = 500

/-- The bounded region created by the curve -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2}

/-- The area of the bounded region -/
noncomputable def area : ℝ := sorry

/-- Theorem stating that the area of the bounded region is 5000/3 -/
theorem area_of_bounded_region : area = 5000/3 := by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l4102_410211


namespace NUMINAMATH_CALUDE_contestant_A_score_l4102_410262

def speech_contest_score (content_score : ℕ) (skills_score : ℕ) (effects_score : ℕ) : ℚ :=
  (4 * content_score + 2 * skills_score + 4 * effects_score) / 10

theorem contestant_A_score :
  speech_contest_score 90 80 90 = 88 := by
  sorry

end NUMINAMATH_CALUDE_contestant_A_score_l4102_410262


namespace NUMINAMATH_CALUDE_scores_with_two_ways_exist_l4102_410213

/-- Represents a scoring configuration for a test -/
structure ScoringConfig where
  total_questions : ℕ
  correct_points : ℕ
  unanswered_points : ℕ
  incorrect_points : ℕ

/-- Represents a possible answer combination -/
structure AnswerCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given answer combination -/
def calculate_score (config : ScoringConfig) (answers : AnswerCombination) : ℕ :=
  answers.correct * config.correct_points + 
  answers.unanswered * config.unanswered_points +
  answers.incorrect * config.incorrect_points

/-- Checks if an answer combination is valid for a given configuration -/
def is_valid_combination (config : ScoringConfig) (answers : AnswerCombination) : Prop :=
  answers.correct + answers.unanswered + answers.incorrect = config.total_questions

/-- Defines the existence of scores with exactly two ways to achieve them -/
def exists_scores_with_two_ways (config : ScoringConfig) : Prop :=
  ∃ S : ℕ, 
    0 ≤ S ∧ S ≤ 175 ∧
    (∃ (a b : AnswerCombination),
      a ≠ b ∧
      is_valid_combination config a ∧
      is_valid_combination config b ∧
      calculate_score config a = S ∧
      calculate_score config b = S ∧
      ∀ c : AnswerCombination, 
        is_valid_combination config c ∧ calculate_score config c = S → (c = a ∨ c = b))

/-- The main theorem to prove -/
theorem scores_with_two_ways_exist : 
  let config : ScoringConfig := {
    total_questions := 25,
    correct_points := 7,
    unanswered_points := 3,
    incorrect_points := 0
  }
  exists_scores_with_two_ways config := by
  sorry

end NUMINAMATH_CALUDE_scores_with_two_ways_exist_l4102_410213


namespace NUMINAMATH_CALUDE_x_plus_two_equals_seven_implies_x_equals_five_l4102_410233

theorem x_plus_two_equals_seven_implies_x_equals_five :
  ∀ x : ℝ, x + 2 = 7 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_two_equals_seven_implies_x_equals_five_l4102_410233


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l4102_410260

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_negative_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l4102_410260


namespace NUMINAMATH_CALUDE_fraction_of_66_l4102_410279

theorem fraction_of_66 (x : ℚ) (h : x = 22.142857142857142) : 
  ((((x + 5) * 7) / 5) - 5) = 66 * (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_66_l4102_410279


namespace NUMINAMATH_CALUDE_hcd_6432_132_minus_8_l4102_410267

theorem hcd_6432_132_minus_8 : Nat.gcd 6432 132 - 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hcd_6432_132_minus_8_l4102_410267


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4102_410246

-- Define the quadratic function
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + p*x - 6

-- Define the solution set
def solution_set (p : ℝ) : Set ℝ := {x : ℝ | f p x < 0}

-- Theorem statement
theorem quadratic_inequality_solution (p : ℝ) :
  solution_set p = {x : ℝ | -3 < x ∧ x < 2} → p = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4102_410246


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l4102_410292

-- Define the contents of each box
def box_contents : ℕ := 3

-- Define the number of red balls in each box
def red_balls : ℕ := 2

-- Define the number of white balls in each box
def white_balls : ℕ := 1

-- Define the total number of possible outcomes
def total_outcomes : ℕ := box_contents * box_contents

-- Define the number of outcomes with no red balls
def no_red_outcomes : ℕ := white_balls * white_balls

-- State the theorem
theorem probability_at_least_one_red :
  (1 : ℚ) - (no_red_outcomes : ℚ) / (total_outcomes : ℚ) = 8 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l4102_410292


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_sequence_sum_ratio_l4102_410206

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) :
  geometric_sequence a₁ q (n + 1) = q * geometric_sequence a₁ q n := by sorry

theorem geometric_sequence_sum_ratio (a₁ : ℝ) :
  let q := -1/3
  (geometric_sequence a₁ q 1 + geometric_sequence a₁ q 3 + geometric_sequence a₁ q 5 + geometric_sequence a₁ q 7) /
  (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 + geometric_sequence a₁ q 6 + geometric_sequence a₁ q 8) = -3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_sequence_sum_ratio_l4102_410206


namespace NUMINAMATH_CALUDE_range_of_m_l4102_410201

/-- Given sets A and B, where B is a subset of A, prove that m ≤ 0 -/
theorem range_of_m (m : ℝ) : 
  let A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m + 3}
  B ⊆ A → m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l4102_410201


namespace NUMINAMATH_CALUDE_cook_is_innocent_l4102_410295

-- Define the type for individuals
def Individual : Type := String

-- Define the property of stealing pepper
def stole_pepper (x : Individual) : Prop := sorry

-- Define the property of lying
def always_lies (x : Individual) : Prop := sorry

-- Define the property of knowing who stole the pepper
def knows_thief (x : Individual) : Prop := sorry

-- The cook
def cook : Individual := "Cook"

-- Axiom: Individuals who steal pepper always lie
axiom pepper_thieves_lie : ∀ x : Individual, stole_pepper x → always_lies x

-- Axiom: The cook stated they know who stole the pepper
axiom cook_statement : knows_thief cook

-- Theorem: The cook is innocent (did not steal the pepper)
theorem cook_is_innocent : ¬(stole_pepper cook) := by sorry

end NUMINAMATH_CALUDE_cook_is_innocent_l4102_410295


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_focus_l4102_410237

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the origin to the right focus is 6 and 
    the asymptote forms an equilateral triangle with the origin and the right focus,
    then a = 3 and b = 3√3 -/
theorem hyperbola_equilateral_focus (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := 6  -- distance from origin to right focus
  let slope := b / a  -- slope of asymptote
  c^2 = a^2 + b^2 →  -- focus property of hyperbola
  slope = Real.sqrt 3 →  -- equilateral triangle condition
  (a = 3 ∧ b = 3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_focus_l4102_410237


namespace NUMINAMATH_CALUDE_remainder_theorem_l4102_410221

def polynomial (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 4*x^6 - 9*x^4 + 3*x^3 - 5*x^2 + 8

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * (q x) + polynomial (2 : ℝ) ∧
    polynomial (2 : ℝ) = 1020 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4102_410221


namespace NUMINAMATH_CALUDE_rectangle_area_relationship_l4102_410235

/-- Theorem: For a rectangle with area 4 and side lengths x and y, y = 4/x where x > 0 -/
theorem rectangle_area_relationship (x y : ℝ) (h1 : x > 0) (h2 : x * y = 4) : y = 4 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relationship_l4102_410235


namespace NUMINAMATH_CALUDE_house_expansion_l4102_410202

/-- Given two houses with areas 5200 and 7300 square feet, if their total area
    after expanding the smaller house is 16000 square feet, then the expansion
    size is 3500 square feet. -/
theorem house_expansion (small_house large_house expanded_total : ℕ)
    (h1 : small_house = 5200)
    (h2 : large_house = 7300)
    (h3 : expanded_total = 16000)
    (h4 : expanded_total = small_house + large_house + expansion_size) :
    expansion_size = 3500 := by
  sorry

end NUMINAMATH_CALUDE_house_expansion_l4102_410202


namespace NUMINAMATH_CALUDE_downstream_distance_l4102_410214

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : time = 2) : 
  boat_speed * time + stream_speed * time = 56 := by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l4102_410214


namespace NUMINAMATH_CALUDE_smallest_number_l4102_410284

def number_set : Set ℤ := {1, 0, -2, -6}

theorem smallest_number :
  ∀ x ∈ number_set, -6 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l4102_410284


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4102_410264

/-- A hyperbola sharing a focus with a parabola and having a specific eccentricity -/
def HyperbolaWithSharedFocus (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  ∃ (x₀ y₀ : ℝ), (x₀ = 2 ∧ y₀ = 0) ∧  -- Focus of parabola y² = 8x
  ∃ (c : ℝ), c = 2 ∧  -- Distance from center to focus
  ∃ (e : ℝ), e = 2 ∧ e = c / a  -- Eccentricity

/-- Theorem stating the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) 
  (h : HyperbolaWithSharedFocus a b) : 
  a = 1 ∧ b^2 = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4102_410264


namespace NUMINAMATH_CALUDE_annie_cookies_total_l4102_410297

/-- The number of cookies Annie ate on Monday -/
def monday_cookies : ℕ := 5

/-- The number of cookies Annie ate on Tuesday -/
def tuesday_cookies : ℕ := 2 * monday_cookies

/-- The number of cookies Annie ate on Wednesday -/
def wednesday_cookies : ℕ := tuesday_cookies + (tuesday_cookies * 40 / 100)

/-- The total number of cookies Annie ate over three days -/
def total_cookies : ℕ := monday_cookies + tuesday_cookies + wednesday_cookies

/-- Theorem stating that Annie ate 29 cookies in total over three days -/
theorem annie_cookies_total : total_cookies = 29 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_total_l4102_410297


namespace NUMINAMATH_CALUDE_guessing_game_scores_l4102_410212

-- Define the players and their scores
def Hajar : ℕ := 42
def Farah : ℕ := Hajar + 24
def Sami : ℕ := Farah + 18

-- Theorem statement
theorem guessing_game_scores :
  Hajar = 42 ∧ Farah = 66 ∧ Sami = 84 ∧
  Farah - Hajar = 24 ∧ Sami - Farah = 18 ∧
  Farah > Hajar ∧ Sami > Hajar :=
by
  sorry


end NUMINAMATH_CALUDE_guessing_game_scores_l4102_410212


namespace NUMINAMATH_CALUDE_makeup_set_cost_l4102_410271

theorem makeup_set_cost (initial_money mom_contribution additional_needed : ℕ) :
  initial_money = 35 →
  mom_contribution = 20 →
  additional_needed = 10 →
  initial_money + mom_contribution + additional_needed = 65 := by
  sorry

end NUMINAMATH_CALUDE_makeup_set_cost_l4102_410271


namespace NUMINAMATH_CALUDE_car_journey_distance_l4102_410280

theorem car_journey_distance :
  ∀ (v : ℝ),
  v > 0 →
  v * 7 = (v + 12) * 5 →
  v * 7 = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_car_journey_distance_l4102_410280


namespace NUMINAMATH_CALUDE_complex_power_difference_l4102_410263

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^24 - (1 - i)^24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l4102_410263


namespace NUMINAMATH_CALUDE_employee_distribution_percentage_difference_l4102_410239

theorem employee_distribution_percentage_difference :
  let total_degrees : ℝ := 360
  let manufacturing_degrees : ℝ := 162
  let sales_degrees : ℝ := 108
  let research_degrees : ℝ := 54
  let admin_degrees : ℝ := 36
  let manufacturing_percent := (manufacturing_degrees / total_degrees) * 100
  let sales_percent := (sales_degrees / total_degrees) * 100
  let research_percent := (research_degrees / total_degrees) * 100
  let admin_percent := (admin_degrees / total_degrees) * 100
  let max_percent := max manufacturing_percent (max sales_percent (max research_percent admin_percent))
  let min_percent := min manufacturing_percent (min sales_percent (min research_percent admin_percent))
  (max_percent - min_percent) = 35 := by
  sorry

end NUMINAMATH_CALUDE_employee_distribution_percentage_difference_l4102_410239


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4102_410200

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x^2 - 3*x - 4 = 0 → x ≠ -1 →
  (2 - (x - 1) / (x + 1)) / ((x^2 + 6*x + 9) / (x^2 - 1)) = (x - 1) / (x + 3) ∧
  (x - 1) / (x + 3) = 3 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4102_410200


namespace NUMINAMATH_CALUDE_tanning_salon_revenue_l4102_410215

/-- Revenue calculation for a tanning salon --/
theorem tanning_salon_revenue :
  let first_visit_cost : ℕ := 10
  let subsequent_visit_cost : ℕ := 8
  let total_customers : ℕ := 100
  let second_visit_customers : ℕ := 30
  let third_visit_customers : ℕ := 10
  
  let first_visit_revenue := first_visit_cost * total_customers
  let second_visit_revenue := subsequent_visit_cost * second_visit_customers
  let third_visit_revenue := subsequent_visit_cost * third_visit_customers
  
  first_visit_revenue + second_visit_revenue + third_visit_revenue = 1320 :=
by
  sorry


end NUMINAMATH_CALUDE_tanning_salon_revenue_l4102_410215


namespace NUMINAMATH_CALUDE_susna_class_f_fraction_l4102_410273

/-- Represents the fractions of students getting each grade in Mrs. Susna's class -/
structure GradeDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  f : ℚ

/-- The conditions of the problem -/
def susna_class : GradeDistribution where
  a := 1/4
  b := 1/2
  c := 1/8
  d := 1/12
  f := 0 -- We'll prove this is actually 1/24

theorem susna_class_f_fraction :
  let g := susna_class
  (g.a + g.b + g.c = 7/8) →
  (g.a + g.b + g.c = 0.875) →
  (g.a + g.b + g.c + g.d + g.f = 1) →
  g.f = 1/24 := by sorry

end NUMINAMATH_CALUDE_susna_class_f_fraction_l4102_410273


namespace NUMINAMATH_CALUDE_magnets_ratio_l4102_410258

theorem magnets_ratio (adam_initial : ℕ) (peter : ℕ) (adam_final : ℕ) : 
  adam_initial = 18 →
  peter = 24 →
  adam_final = peter / 2 →
  adam_final = adam_initial - (adam_initial - adam_final) →
  (adam_initial - adam_final : ℚ) / adam_initial = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_magnets_ratio_l4102_410258


namespace NUMINAMATH_CALUDE_lindsey_owns_four_more_cars_l4102_410282

/-- The number of cars owned by each person --/
structure CarOwnership where
  cathy : ℕ
  carol : ℕ
  susan : ℕ
  lindsey : ℕ

/-- The conditions of the car ownership problem --/
def carProblemConditions (co : CarOwnership) : Prop :=
  co.cathy = 5 ∧
  co.carol = 2 * co.cathy ∧
  co.susan = co.carol - 2 ∧
  co.lindsey > co.cathy ∧
  co.cathy + co.carol + co.susan + co.lindsey = 32

/-- The theorem stating that Lindsey owns 4 more cars than Cathy --/
theorem lindsey_owns_four_more_cars (co : CarOwnership) 
  (h : carProblemConditions co) : co.lindsey - co.cathy = 4 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_owns_four_more_cars_l4102_410282


namespace NUMINAMATH_CALUDE_conical_flask_height_l4102_410234

/-- The height of a conical flask given water depths in two positions -/
theorem conical_flask_height (h : ℝ) : 
  (h > 0) →  -- The height is positive
  (h^3 - (h-1)^3 = 8) →  -- Volume equation derived from the two water depths
  h = 1/2 + Real.sqrt 93 / 6 := by
sorry

end NUMINAMATH_CALUDE_conical_flask_height_l4102_410234


namespace NUMINAMATH_CALUDE_dot_product_is_two_l4102_410249

/-- A rhombus with side length 2 and angle BAC of 60° -/
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (is_rhombus : sorry)
  (side_length : sorry)
  (angle_BAC : sorry)

/-- The dot product of vectors BC and AC in the given rhombus -/
def dot_product_BC_AC (r : Rhombus) : ℝ :=
  sorry

/-- Theorem: The dot product of vectors BC and AC in the given rhombus is 2 -/
theorem dot_product_is_two (r : Rhombus) : dot_product_BC_AC r = 2 :=
  sorry

end NUMINAMATH_CALUDE_dot_product_is_two_l4102_410249


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l4102_410299

/-- Given that the solution set of ax² + bx + c > 0 is {x | -3 < x < 2} -/
def solution_set (a b c : ℝ) : Set ℝ :=
  {x | -3 < x ∧ x < 2 ∧ a * x^2 + b * x + c > 0}

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x | -3 < x ∧ x < 2}) :
  a < 0 ∧
  a + b + c > 0 ∧
  {x | c * x^2 + b * x + a < 0} = {x | -1/3 < x ∧ x < 1/2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l4102_410299


namespace NUMINAMATH_CALUDE_expression_simplification_l4102_410218

theorem expression_simplification (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a * b^2 = c / a - b) :
  let expr := (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c)) /
               (2 / (a * b) - 2 * a * b / c) /
               (101 / c)
  expr = -1 / 202 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4102_410218


namespace NUMINAMATH_CALUDE_circle_circumference_l4102_410293

theorem circle_circumference (r : ℝ) (h : r = 4) : 
  2 * Real.pi * r = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l4102_410293


namespace NUMINAMATH_CALUDE_distance_from_bogula_to_bolifoyn_l4102_410205

/-- The distance from Bogula to Bolifoyn in miles -/
def total_distance : ℝ := 10

/-- The time in hours at which they approach Pigtown -/
def time_to_pigtown : ℝ := 1

/-- The additional distance traveled after Pigtown in miles -/
def additional_distance : ℝ := 5

/-- The total travel time in hours -/
def total_time : ℝ := 4

theorem distance_from_bogula_to_bolifoyn :
  ∃ (distance_to_pigtown : ℝ),
    /- After 20 minutes (1/3 hour), half of the remaining distance to Pigtown is covered -/
    (1/3 * (total_distance / time_to_pigtown)) = distance_to_pigtown / 2 ∧
    /- The distance covered is twice less than the remaining distance to Pigtown -/
    (1/3 * (total_distance / time_to_pigtown)) = distance_to_pigtown / 3 ∧
    /- They took another 5 miles after approaching Pigtown -/
    total_distance = distance_to_pigtown + additional_distance ∧
    /- The total travel time is 4 hours -/
    total_time * (total_distance / total_time) = total_distance := by
  sorry


end NUMINAMATH_CALUDE_distance_from_bogula_to_bolifoyn_l4102_410205


namespace NUMINAMATH_CALUDE_hit_probability_random_gun_selection_l4102_410219

/-- The probability of hitting a target when randomly selecting a gun from a set of calibrated and uncalibrated guns. -/
theorem hit_probability_random_gun_selection 
  (total_guns : ℕ) 
  (calibrated_guns : ℕ) 
  (uncalibrated_guns : ℕ) 
  (calibrated_accuracy : ℝ) 
  (uncalibrated_accuracy : ℝ) 
  (h1 : total_guns = 5)
  (h2 : calibrated_guns = 3)
  (h3 : uncalibrated_guns = 2)
  (h4 : calibrated_guns + uncalibrated_guns = total_guns)
  (h5 : calibrated_accuracy = 0.9)
  (h6 : uncalibrated_accuracy = 0.4) :
  (calibrated_guns : ℝ) / total_guns * calibrated_accuracy + 
  (uncalibrated_guns : ℝ) / total_guns * uncalibrated_accuracy = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_hit_probability_random_gun_selection_l4102_410219


namespace NUMINAMATH_CALUDE_hexagon_area_fraction_l4102_410274

/-- Represents a tiling pattern of the plane -/
structure TilingPattern where
  /-- The number of smaller units in one side of a large square -/
  units_per_side : ℕ
  /-- The number of units occupied by hexagons in a large square -/
  hexagon_units : ℕ

/-- The fraction of the plane enclosed by hexagons -/
def hexagon_fraction (pattern : TilingPattern) : ℚ :=
  pattern.hexagon_units / (pattern.units_per_side ^ 2 : ℚ)

/-- The specific tiling pattern described in the problem -/
def problem_pattern : TilingPattern :=
  { units_per_side := 4
  , hexagon_units := 8 }

theorem hexagon_area_fraction :
  hexagon_fraction problem_pattern = 1/2 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_fraction_l4102_410274


namespace NUMINAMATH_CALUDE_number_of_factors_of_n_l4102_410288

def n : ℕ := 2^5 * 3^4 * 5^6 * 6^3

theorem number_of_factors_of_n : (Finset.card (Nat.divisors n)) = 504 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_n_l4102_410288


namespace NUMINAMATH_CALUDE_board_numbers_proof_l4102_410256

def pairwise_sums (a b c d e : ℤ) : Finset ℤ :=
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}

theorem board_numbers_proof :
  ∃ (a b c d e : ℤ),
    pairwise_sums a b c d e = {5, 9, 10, 11, 12, 16, 16, 17, 21, 23} ∧
    Finset.toList {a, b, c, d, e} = [2, 3, 7, 9, 14] ∧
    a * b * c * d * e = 5292 :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_proof_l4102_410256


namespace NUMINAMATH_CALUDE_ninety_nine_in_third_column_l4102_410277

/-- Represents the column number (1 to 5) in the arrangement -/
inductive Column
  | one
  | two
  | three
  | four
  | five

/-- Function that determines the column for a given odd number -/
def columnForOddNumber (n : ℕ) : Column :=
  match n % 5 with
  | 1 => Column.one
  | 2 => Column.four
  | 3 => Column.two
  | 4 => Column.five
  | 0 => Column.three
  | _ => Column.one  -- This case should never occur for odd numbers

theorem ninety_nine_in_third_column :
  columnForOddNumber 99 = Column.three :=
sorry

end NUMINAMATH_CALUDE_ninety_nine_in_third_column_l4102_410277


namespace NUMINAMATH_CALUDE_fourth_root_cubed_l4102_410248

theorem fourth_root_cubed (x : ℝ) : (x^(1/4))^3 = 729 → x = 6561 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_cubed_l4102_410248


namespace NUMINAMATH_CALUDE_gcd_108_45_l4102_410203

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by sorry

end NUMINAMATH_CALUDE_gcd_108_45_l4102_410203


namespace NUMINAMATH_CALUDE_two_digit_multiplication_swap_l4102_410227

theorem two_digit_multiplication_swap (a b c d : Nat) : 
  (a ≥ 1 ∧ a ≤ 9) →
  (b ≥ 0 ∧ b ≤ 9) →
  (c ≥ 1 ∧ c ≤ 9) →
  (d ≥ 0 ∧ d ≤ 9) →
  ((10 * a + b) * (10 * c + d) - (10 * b + a) * (10 * c + d) = 4248) →
  ((10 * a + b) * (10 * c + d) = 5369 ∨ (10 * a + b) * (10 * c + d) = 4720) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_swap_l4102_410227


namespace NUMINAMATH_CALUDE_total_goals_in_five_matches_l4102_410242

/-- A football player's goal scoring record over 5 matches -/
structure FootballPlayer where
  /-- The average number of goals per match before the fifth match -/
  initial_average : ℝ
  /-- The number of goals scored in the fifth match -/
  fifth_match_goals : ℕ
  /-- The increase in average goals after the fifth match -/
  average_increase : ℝ

/-- Theorem stating the total number of goals scored over 5 matches -/
theorem total_goals_in_five_matches (player : FootballPlayer)
    (h1 : player.fifth_match_goals = 2)
    (h2 : player.average_increase = 0.3) :
    (player.initial_average * 4 + player.fifth_match_goals : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_goals_in_five_matches_l4102_410242


namespace NUMINAMATH_CALUDE_min_real_roots_2010_l4102_410240

/-- A polynomial of degree 2010 with real coefficients -/
def RealPolynomial2010 : Type := { p : Polynomial ℝ // p.degree = 2010 }

/-- The roots of a polynomial -/
def roots (p : RealPolynomial2010) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial2010) : ℕ := sorry

/-- The theorem statement -/
theorem min_real_roots_2010 (p : RealPolynomial2010) 
  (h : distinctAbsValues p = 1010) : 
  realRootCount p ≥ 10 := sorry

end NUMINAMATH_CALUDE_min_real_roots_2010_l4102_410240


namespace NUMINAMATH_CALUDE_fraction_inequality_l4102_410238

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < 0) (h3 : c < d) (h4 : d < 0) : 
  d / a < c / a :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l4102_410238


namespace NUMINAMATH_CALUDE_find_divisor_l4102_410269

theorem find_divisor (n : ℕ) (s : ℕ) (d : ℕ) : 
  n = 724946 →
  s = 6 →
  d ∣ (n - s) →
  (∀ k < s, ¬(d ∣ (n - k))) →
  d = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l4102_410269


namespace NUMINAMATH_CALUDE_angle_theorem_l4102_410229

theorem angle_theorem (α β : Real) (P : Real × Real) :
  P = (3, 4) → -- Point P is (3,4)
  (∃ r : Real, r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) → -- P is on terminal side of α
  Real.cos β = 5/13 → -- cos β = 5/13
  β ∈ Set.Icc 0 (Real.pi / 2) → -- β ∈ [0, π/2]
  Real.sin α = 4/5 ∧ Real.cos (α - β) = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_angle_theorem_l4102_410229


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_two_l4102_410224

def i : ℂ := Complex.I

theorem imaginary_sum_equals_two :
  i^15 + i^20 + i^25 + i^30 + i^35 + i^40 = (2 : ℂ) :=
by sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_two_l4102_410224


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l4102_410204

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 3 * (b + c))
  (second_eq : b = 5 * c) :
  a * b * c = 176 := by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l4102_410204


namespace NUMINAMATH_CALUDE_complement_B_equals_M_intersection_A_B_l4102_410283

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x - a > 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | (x+a)*(x+b) > 0}

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1
theorem complement_B_equals_M (a b : ℝ) (h : a ≠ b) :
  (U \ B a b = M) ↔ ((a = -3 ∧ b = 1) ∨ (a = 1 ∧ b = -3)) :=
sorry

-- Theorem 2
theorem intersection_A_B (a b : ℝ) (h : -1 < b ∧ b < a ∧ a < 1) :
  A a ∩ B a b = {x | x < -a ∨ x > 1} :=
sorry

end NUMINAMATH_CALUDE_complement_B_equals_M_intersection_A_B_l4102_410283


namespace NUMINAMATH_CALUDE_triangle_base_length_l4102_410251

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) :
  height = 10 →
  area = 50 →
  area = (base * height) / 2 →
  base = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l4102_410251


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l4102_410207

theorem students_taking_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 20) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 470 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l4102_410207


namespace NUMINAMATH_CALUDE_dan_picked_more_apples_l4102_410226

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- Theorem: Dan picked 7 more apples than Benny -/
theorem dan_picked_more_apples : dan_apples - benny_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_more_apples_l4102_410226


namespace NUMINAMATH_CALUDE_stock_price_increase_l4102_410210

/-- Calculate the percent increase in stock price -/
theorem stock_price_increase (opening_price closing_price : ℝ) 
  (h1 : opening_price = 25)
  (h2 : closing_price = 28) :
  (closing_price - opening_price) / opening_price * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l4102_410210


namespace NUMINAMATH_CALUDE_team_a_win_probability_l4102_410245

theorem team_a_win_probability (p : ℝ) (h : p = 2/3) : 
  p^2 + p^2 * (1 - p) = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l4102_410245


namespace NUMINAMATH_CALUDE_value_difference_reciprocals_l4102_410298

theorem value_difference_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x - y = x / y) : 1 / x - 1 / y = -1 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_value_difference_reciprocals_l4102_410298


namespace NUMINAMATH_CALUDE_existence_of_k_with_n_prime_factors_l4102_410250

theorem existence_of_k_with_n_prime_factors 
  (m n : ℕ+) : 
  ∃ k : ℕ+, ∃ p : Finset ℕ, 
    (∀ x ∈ p, Nat.Prime x) ∧ 
    (Finset.card p ≥ n) ∧ 
    (∀ x ∈ p, x ∣ (2^(k:ℕ) - m)) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_k_with_n_prime_factors_l4102_410250


namespace NUMINAMATH_CALUDE_max_area_triangle_OPQ_l4102_410217

/-- The maximum area of triangle OPQ given the specified conditions -/
theorem max_area_triangle_OPQ :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let O : ℝ × ℝ := (0, 0)
  ∃ (M P Q : ℝ × ℝ),
    (M.1 ≠ -2 ∧ M.1 ≠ 2) →  -- M is not on the same vertical line as A or B
    (M.2 / (M.1 + 2)) * (M.2 / (M.1 - 2)) = -3/4 →  -- Product of slopes AM and BM
    (P.2 - Q.2) / (P.1 - Q.1) = 1 →  -- PQ has slope 1
    (M.1^2 / 4 + M.2^2 / 3 = 1) →  -- M is on the locus
    (P.1^2 / 4 + P.2^2 / 3 = 1) →  -- P is on the locus
    (Q.1^2 / 4 + Q.2^2 / 3 = 1) →  -- Q is on the locus
    (∀ R : ℝ × ℝ, R.1^2 / 4 + R.2^2 / 3 = 1 →  -- For all points R on the locus
      abs ((P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2)) / 2 ≥
      abs ((P.1 - O.1) * (R.2 - O.2) - (R.1 - O.1) * (P.2 - O.2)) / 2) →
    abs ((P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2)) / 2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_OPQ_l4102_410217


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l4102_410266

theorem cos_2alpha_plus_4pi_3 (α : ℝ) (h : Real.sqrt 3 * Real.sin α + Real.cos α = 1/2) :
  Real.cos (2 * α + 4 * Real.pi / 3) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l4102_410266


namespace NUMINAMATH_CALUDE_children_ages_exist_l4102_410261

theorem children_ages_exist :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 33 ∧
    (a - 3) + (b - 3) + (c - 3) + (d - 3) = 22 ∧
    (a - 7) + (b - 7) + (c - 7) + (d - 7) = 11 ∧
    (a - 13) + (b - 13) + (c - 13) + (d - 13) = 1 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
by sorry

end NUMINAMATH_CALUDE_children_ages_exist_l4102_410261


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l4102_410222

theorem largest_divisor_of_difference_of_squares (m n : ℕ) : 
  Even m → Even n → n < m → 
  (∃ (k : ℕ), ∀ (a b : ℕ), Even a → Even b → b < a → 
    k ∣ (a^2 - b^2) ∧ k = 16 ∧ ∀ (l : ℕ), (∀ (x y : ℕ), Even x → Even y → y < x → l ∣ (x^2 - y^2)) → l ≤ k) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l4102_410222


namespace NUMINAMATH_CALUDE_cos_4theta_from_exp_l4102_410231

theorem cos_4theta_from_exp (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (1 - Complex.I * Real.sqrt 8) / 3 → 
  Real.cos (4 * θ) = 17 / 81 := by
sorry

end NUMINAMATH_CALUDE_cos_4theta_from_exp_l4102_410231


namespace NUMINAMATH_CALUDE_apple_weight_l4102_410225

theorem apple_weight (total_weight orange_weight grape_weight strawberry_weight : ℝ) 
  (h1 : total_weight = 10)
  (h2 : orange_weight = 1)
  (h3 : grape_weight = 3)
  (h4 : strawberry_weight = 3) :
  total_weight - (orange_weight + grape_weight + strawberry_weight) = 3 :=
by sorry

end NUMINAMATH_CALUDE_apple_weight_l4102_410225


namespace NUMINAMATH_CALUDE_carol_first_six_prob_l4102_410216

/-- The probability of rolling a number other than 6 on a fair six-sided die. -/
def prob_not_six : ℚ := 5/6

/-- The probability of rolling a 6 on a fair six-sided die. -/
def prob_six : ℚ := 1/6

/-- The number of players before Carol. -/
def players_before_carol : ℕ := 2

/-- The total number of players. -/
def total_players : ℕ := 4

/-- The probability that Carol is the first to roll a six in the dice game. -/
theorem carol_first_six_prob : 
  (prob_not_six^players_before_carol * prob_six) / (1 - prob_not_six^total_players) = 125/671 := by
  sorry

end NUMINAMATH_CALUDE_carol_first_six_prob_l4102_410216


namespace NUMINAMATH_CALUDE_max_annual_profit_l4102_410294

/-- Represents the annual profit function in million yuan -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  if x < 80 then
    50 * x - (1/3 * x^2 + 10 * x) / 100 - 250
  else
    50 * x - (51 * x + 10000 / x - 1450) / 100 - 250

/-- The maximum annual profit is 1000 million yuan -/
theorem max_annual_profit :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → annual_profit x ≥ annual_profit y ∧ annual_profit x = 1000 :=
sorry

end NUMINAMATH_CALUDE_max_annual_profit_l4102_410294


namespace NUMINAMATH_CALUDE_car_trip_mpg_l4102_410228

/-- Calculates the average miles per gallon for a trip given odometer readings and gas fill-ups --/
def average_mpg (initial_reading : ℕ) (final_reading : ℕ) (gas_used : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / gas_used

theorem car_trip_mpg : 
  let initial_reading : ℕ := 48500
  let second_reading : ℕ := 48800
  let final_reading : ℕ := 49350
  let first_fillup : ℕ := 8
  let second_fillup : ℕ := 10
  let third_fillup : ℕ := 15
  let total_gas_used : ℕ := second_fillup + third_fillup
  average_mpg initial_reading final_reading total_gas_used = 34 := by
  sorry

#eval average_mpg 48500 49350 25

end NUMINAMATH_CALUDE_car_trip_mpg_l4102_410228


namespace NUMINAMATH_CALUDE_bottle_caps_per_visit_l4102_410296

def store_visits : ℕ := 5
def total_bottle_caps : ℕ := 25

theorem bottle_caps_per_visit :
  total_bottle_caps / store_visits = 5 :=
by sorry

end NUMINAMATH_CALUDE_bottle_caps_per_visit_l4102_410296


namespace NUMINAMATH_CALUDE_power_function_through_point_l4102_410253

/-- A power function passing through a specific point -/
def isPowerFunctionThroughPoint (m : ℝ) : Prop :=
  ∃ (y : ℝ → ℝ), (∀ x, y x = (m^2 - 3*m + 3) * x^m) ∧ y 2 = 4

/-- The value of m for which the power function passes through (2, 4) -/
theorem power_function_through_point :
  ∃ (m : ℝ), isPowerFunctionThroughPoint m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l4102_410253


namespace NUMINAMATH_CALUDE_fold_paper_l4102_410272

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is perpendicular bisector of two points -/
def isPerpBisector (l : Line) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  (midpoint.y = l.slope * midpoint.x + l.intercept) ∧
  (l.slope * (p2.x - p1.x) = -(p2.y - p1.y))

/-- Function to check if two points are symmetric about a line -/
def areSymmetric (l : Line) (p1 p2 : Point) : Prop :=
  isPerpBisector l p1 p2

/-- Main theorem -/
theorem fold_paper (l : Line) (p1 p2 p3 : Point) (p q : ℝ) :
  areSymmetric l ⟨1, 3⟩ ⟨5, 1⟩ →
  areSymmetric l ⟨8, 4⟩ ⟨p, q⟩ →
  p + q = 8 := by
  sorry

end NUMINAMATH_CALUDE_fold_paper_l4102_410272


namespace NUMINAMATH_CALUDE_relationship_uvwt_l4102_410276

theorem relationship_uvwt (m p r s : ℝ) (u v w t : ℝ) 
  (h1 : m^u = p^v) (h2 : p^v = r) (h3 : p^w = m^t) (h4 : m^t = s) :
  u * v = w * t := by
  sorry

end NUMINAMATH_CALUDE_relationship_uvwt_l4102_410276


namespace NUMINAMATH_CALUDE_cubic_equation_proof_l4102_410230

theorem cubic_equation_proof :
  let f : ℝ → ℝ := fun x ↦ x^3 - 5*x - 2
  ∃ (x₁ x₂ x₃ : ℝ),
    (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (x₁ * x₂ * x₃ = x₁ + x₂ + x₃ + 2) ∧
    (x₁^2 + x₂^2 + x₃^2 = 10) ∧
    (x₁^3 + x₂^3 + x₃^3 = 6) :=
by sorry

#check cubic_equation_proof

end NUMINAMATH_CALUDE_cubic_equation_proof_l4102_410230


namespace NUMINAMATH_CALUDE_longest_piece_length_l4102_410255

theorem longest_piece_length (rope1 rope2 rope3 : ℕ) 
  (h1 : rope1 = 75)
  (h2 : rope2 = 90)
  (h3 : rope3 = 135) : 
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_longest_piece_length_l4102_410255


namespace NUMINAMATH_CALUDE_correct_fraction_proof_l4102_410281

theorem correct_fraction_proof (x y : ℚ) : 
  (5 / 6 : ℚ) * 288 = x / y * 288 + 150 → x / y = 5 / 32 := by
sorry

end NUMINAMATH_CALUDE_correct_fraction_proof_l4102_410281


namespace NUMINAMATH_CALUDE_cube_surface_coverage_l4102_410291

/-- Represents a cube -/
structure Cube where
  vertices : ℕ
  angle_sum_at_vertex : ℕ

/-- Represents a triangle -/
structure Triangle where
  angle_sum : ℕ

/-- The problem statement -/
theorem cube_surface_coverage (c : Cube) (t : Triangle) : 
  c.vertices = 8 → 
  c.angle_sum_at_vertex = 270 → 
  t.angle_sum = 180 → 
  ¬ (3 * t.angle_sum ≥ c.vertices * 90) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_coverage_l4102_410291


namespace NUMINAMATH_CALUDE_f_unique_solution_l4102_410220

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

theorem f_unique_solution :
  ∃! x, f x = 1/4 ∧ x ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_f_unique_solution_l4102_410220


namespace NUMINAMATH_CALUDE_compound_has_one_Al_l4102_410265

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- A compound with Aluminium and Iodine -/
structure Compound where
  Al_count : ℕ
  I_count : ℕ
  molecular_weight : ℝ

/-- The compound in question -/
def our_compound : Compound where
  Al_count := 1
  I_count := 3
  molecular_weight := 408

/-- Theorem stating that our compound has exactly 1 Aluminium atom -/
theorem compound_has_one_Al : 
  our_compound.Al_count = 1 ∧
  our_compound.I_count = 3 ∧
  our_compound.molecular_weight = 408 ∧
  (our_compound.Al_count : ℝ) * atomic_weight_Al + (our_compound.I_count : ℝ) * atomic_weight_I = our_compound.molecular_weight :=
by sorry

end NUMINAMATH_CALUDE_compound_has_one_Al_l4102_410265


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l4102_410286

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem third_term_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 1 = 3 → q = -2 → a 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l4102_410286


namespace NUMINAMATH_CALUDE_child_height_calculation_l4102_410223

/-- Calculates a child's current height given their previous height and growth. -/
def current_height (previous_height growth : Float) : Float :=
  previous_height + growth

theorem child_height_calculation :
  let previous_height : Float := 38.5
  let growth : Float := 3.0
  current_height previous_height growth = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_child_height_calculation_l4102_410223


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_l4102_410252

/-- A point on a hyperbola with a specific distance to a line has a specific distance to another line --/
theorem hyperbola_point_distance (m n : ℝ) : 
  m^2 - n^2 = 9 →                        -- P(m, n) is on the hyperbola x^2 - y^2 = 9
  (|m + n| / Real.sqrt 2) = 2016 →       -- Distance from P to y = -x is 2016
  (|m - n| / Real.sqrt 2) = 448 :=       -- Distance from P to y = x is 448
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_distance_l4102_410252


namespace NUMINAMATH_CALUDE_good_apples_count_l4102_410257

theorem good_apples_count (total : ℕ) (unripe : ℕ) (h1 : total = 14) (h2 : unripe = 6) :
  total - unripe = 8 := by
sorry

end NUMINAMATH_CALUDE_good_apples_count_l4102_410257
