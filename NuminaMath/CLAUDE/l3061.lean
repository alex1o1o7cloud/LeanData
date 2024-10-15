import Mathlib

namespace NUMINAMATH_CALUDE_log_319_approximation_l3061_306166

-- Define the logarithm values for 0.317 and 0.318
def log_317 : ℝ := 0.33320
def log_318 : ℝ := 0.3364

-- Define the approximation function for log 0.319
def approx_log_319 : ℝ := log_318 + (log_318 - log_317)

-- Theorem statement
theorem log_319_approximation : 
  abs (approx_log_319 - 0.3396) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_log_319_approximation_l3061_306166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3061_306183

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3061_306183


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3061_306156

/-- The total surface area of a hemisphere with radius 9 cm, including its circular base, is 243π cm². -/
theorem hemisphere_surface_area :
  let r : ℝ := 9
  let base_area : ℝ := π * r^2
  let curved_area : ℝ := 2 * π * r^2
  let total_area : ℝ := base_area + curved_area
  total_area = 243 * π := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3061_306156


namespace NUMINAMATH_CALUDE_point_on_circle_l3061_306142

theorem point_on_circle (t : ℝ) : 
  let x := (3 - t^3) / (3 + t^3)
  let y := 3*t / (3 + t^3)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_circle_l3061_306142


namespace NUMINAMATH_CALUDE_women_average_age_l3061_306198

theorem women_average_age 
  (n : Nat) 
  (initial_avg : ℝ) 
  (age_increase : ℝ) 
  (man1_age : ℝ) 
  (man2_age : ℝ) 
  (h1 : n = 7) 
  (h2 : age_increase = 4) 
  (h3 : man1_age = 26) 
  (h4 : man2_age = 30) 
  (h5 : n * (initial_avg + age_increase) = n * initial_avg - man1_age - man2_age + (women_avg * 2)) : 
  women_avg = 42 := by
  sorry

#check women_average_age

end NUMINAMATH_CALUDE_women_average_age_l3061_306198


namespace NUMINAMATH_CALUDE_concentric_circles_theorem_l3061_306125

/-- Given two concentric circles where the area between them is equal to twice the area of the smaller circle -/
theorem concentric_circles_theorem (a b : ℝ) (h : a > 0) (h' : b > 0) (h_concentric : a < b)
  (h_area : π * b^2 - π * a^2 = 2 * π * a^2) :
  (a / b = 1 / Real.sqrt 3) ∧ (π * a^2 / (π * b^2) = 1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_theorem_l3061_306125


namespace NUMINAMATH_CALUDE_candy_bar_profit_l3061_306123

def candy_bars_bought : ℕ := 1500
def buying_price : ℚ := 3 / 8
def selling_price : ℚ := 2 / 3
def booth_setup_cost : ℚ := 50

def total_cost : ℚ := candy_bars_bought * buying_price
def total_revenue : ℚ := candy_bars_bought * selling_price
def net_profit : ℚ := total_revenue - total_cost - booth_setup_cost

theorem candy_bar_profit : net_profit = 387.5 := by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l3061_306123


namespace NUMINAMATH_CALUDE_partner_q_investment_time_l3061_306126

/-- Represents the investment and profit information for a business partnership --/
structure Partnership where
  investmentRatio : Fin 3 → ℚ
  profitRatio : Fin 3 → ℚ
  investmentTime : Fin 3 → ℚ

/-- Theorem stating the investment time for partner Q given the conditions --/
theorem partner_q_investment_time (p : Partnership) :
  p.investmentRatio 0 = 3 ∧
  p.investmentRatio 1 = 4 ∧
  p.investmentRatio 2 = 5 ∧
  p.profitRatio 0 = 9 ∧
  p.profitRatio 1 = 16 ∧
  p.profitRatio 2 = 25 ∧
  p.investmentTime 0 = 4 ∧
  p.investmentTime 2 = 10 →
  p.investmentTime 1 = 8 :=
by sorry

end NUMINAMATH_CALUDE_partner_q_investment_time_l3061_306126


namespace NUMINAMATH_CALUDE_ratio_problem_l3061_306185

theorem ratio_problem (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 1 / 2)
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : c ≠ 0) (h8 : d ≠ 0) (h9 : e ≠ 0) : 
  e / a = 8 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3061_306185


namespace NUMINAMATH_CALUDE_egg_production_increase_l3061_306167

theorem egg_production_increase (last_year_production this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) :
  this_year_production - last_year_production = 3220 :=
by sorry

end NUMINAMATH_CALUDE_egg_production_increase_l3061_306167


namespace NUMINAMATH_CALUDE_income_difference_l3061_306151

-- Define the incomes of A and B
def A (B : ℝ) : ℝ := 0.75 * B

-- Theorem statement
theorem income_difference (B : ℝ) (h : B > 0) : 
  (B - A B) / (A B) = 1/3 := by sorry

end NUMINAMATH_CALUDE_income_difference_l3061_306151


namespace NUMINAMATH_CALUDE_batting_highest_score_l3061_306175

-- Define the given conditions
def total_innings : ℕ := 46
def overall_average : ℚ := 60
def score_difference : ℕ := 180
def average_excluding_extremes : ℚ := 58
def min_half_centuries : ℕ := 15
def min_centuries : ℕ := 10

-- Define the function to calculate the highest score
def highest_score : ℕ := 194

-- Theorem statement
theorem batting_highest_score :
  (total_innings : ℚ) * overall_average = 
    (total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + (highest_score - score_difference) ∧
  highest_score ≥ 100 ∧
  min_half_centuries + min_centuries ≤ total_innings - 2 :=
by sorry

end NUMINAMATH_CALUDE_batting_highest_score_l3061_306175


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3061_306115

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3061_306115


namespace NUMINAMATH_CALUDE_equation_system_solution_l3061_306129

theorem equation_system_solution (x y : ℝ) :
  (2 * x^2 + 6 * x + 4 * y + 2 = 0) →
  (3 * x + y + 4 = 0) →
  (y^2 + 17 * y - 11 = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3061_306129


namespace NUMINAMATH_CALUDE_bananas_permutations_l3061_306112

/-- The number of distinct permutations of a word with repeated letters -/
def distinct_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (List.prod (List.map Nat.factorial repetitions))

/-- Theorem: The number of distinct permutations of BANANAS is 420 -/
theorem bananas_permutations :
  distinct_permutations 7 [3, 2] = 420 := by
  sorry

#eval distinct_permutations 7 [3, 2]

end NUMINAMATH_CALUDE_bananas_permutations_l3061_306112


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l3061_306188

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: In a 200 × 325 × 376 rectangular solid, an internal diagonal passes through 868 unit cubes -/
theorem internal_diagonal_cubes : cubes_passed 200 325 376 = 868 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l3061_306188


namespace NUMINAMATH_CALUDE_quadrilateral_numbers_multiple_of_14_l3061_306160

def quadrilateral_number (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

def is_multiple_of_14 (n : ℕ) : Prop := ∃ k : ℕ, n = 14 * k

theorem quadrilateral_numbers_multiple_of_14 (t : ℤ) :
  (∀ n : ℤ, (n = 28 * t ∨ n = 28 * t + 6 ∨ n = 28 * t + 7 ∨ n = 28 * t + 12 ∨ 
             n = 28 * t + 14 ∨ n = 28 * t - 9 ∨ n = 28 * t - 8 ∨ n = 28 * t - 2 ∨ 
             n = 28 * t - 1) → 
    is_multiple_of_14 (quadrilateral_number n.toNat)) ∧
  (∀ n : ℕ, is_multiple_of_14 (quadrilateral_number n) → 
    ∃ t : ℤ, n = (28 * t).toNat ∨ n = (28 * t + 6).toNat ∨ n = (28 * t + 7).toNat ∨ 
              n = (28 * t + 12).toNat ∨ n = (28 * t + 14).toNat ∨ n = (28 * t - 9).toNat ∨ 
              n = (28 * t - 8).toNat ∨ n = (28 * t - 2).toNat ∨ n = (28 * t - 1).toNat) :=
by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_numbers_multiple_of_14_l3061_306160


namespace NUMINAMATH_CALUDE_simplify_expression_l3061_306190

theorem simplify_expression : (324 : ℝ)^(1/4) * (98 : ℝ)^(1/2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3061_306190


namespace NUMINAMATH_CALUDE_ratio_of_21_to_reversed_l3061_306131

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem ratio_of_21_to_reversed : 
  let original := 21
  let reversed := reverse_digits original
  (original : ℚ) / reversed = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_21_to_reversed_l3061_306131


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3061_306114

/-- The solution set of the system of equations {x - 2y = 1, x^3 - 6xy - 8y^3 = 1} 
    is equivalent to the line y = (x-1)/2 -/
theorem solution_set_equivalence (x y : ℝ) : 
  (x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1) ↔ y = (x - 1) / 2 := by
  sorry

#check solution_set_equivalence

end NUMINAMATH_CALUDE_solution_set_equivalence_l3061_306114


namespace NUMINAMATH_CALUDE_set_intersection_proof_l3061_306110

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem set_intersection_proof : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_proof_l3061_306110


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3061_306139

theorem quadratic_factorization (t : ℝ) : t^2 - 10*t + 25 = (t - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3061_306139


namespace NUMINAMATH_CALUDE_fill_with_corners_l3061_306170

/-- A type representing a box with integer dimensions -/
structure Box where
  m : ℕ
  n : ℕ
  k : ℕ
  m_gt_one : m > 1
  n_gt_one : n > 1
  k_gt_one : k > 1

/-- A type representing a 1 × 1 × 3 bar -/
structure Bar

/-- A type representing a corner made from three 1 × 1 × 1 cubes -/
structure Corner

/-- A function that checks if a box can be filled with bars and corners -/
def canFillWithBarsAndCorners (b : Box) : Prop :=
  ∃ (bars : ℕ) (corners : ℕ), bars * 3 + corners * 3 = b.m * b.n * b.k

/-- A function that checks if a box can be filled with only corners -/
def canFillWithOnlyCorners (b : Box) : Prop :=
  ∃ (corners : ℕ), corners * 3 = b.m * b.n * b.k

/-- The main theorem to be proved -/
theorem fill_with_corners (b : Box) :
  canFillWithBarsAndCorners b → canFillWithOnlyCorners b :=
by sorry

end NUMINAMATH_CALUDE_fill_with_corners_l3061_306170


namespace NUMINAMATH_CALUDE_equation_solutions_l3061_306192

theorem equation_solutions :
  (∀ x, x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) ∧
  (∀ x, 3*x*(x - 1) = 2 - 2*x ↔ x = 1 ∨ x = -2/3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3061_306192


namespace NUMINAMATH_CALUDE_negative_of_negative_five_greater_than_negative_five_l3061_306144

theorem negative_of_negative_five_greater_than_negative_five : -(-5) > -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_five_greater_than_negative_five_l3061_306144


namespace NUMINAMATH_CALUDE_equation_roots_imply_sum_l3061_306187

/-- Given two equations with constants a and b, prove that 100a + b = 156 -/
theorem equation_roots_imply_sum (a b : ℝ) : 
  (∃! x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + a) * (x + b) * (x + 12) = 0 ∧
    (y + a) * (y + b) * (y + 12) = 0 ∧
    (z + a) * (z + b) * (z + 12) = 0 ∧
    x ≠ -3 ∧ y ≠ -3 ∧ z ≠ -3) →
  (∃! w, (w + 2*a) * (w + 3) * (w + 6) = 0 ∧ 
    w + b ≠ 0 ∧ w + 12 ≠ 0) →
  100 * a + b = 156 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_imply_sum_l3061_306187


namespace NUMINAMATH_CALUDE_abs_difference_of_roots_l3061_306140

theorem abs_difference_of_roots (α β : ℝ) (h1 : α + β = 17) (h2 : α * β = 70) : 
  |α - β| = 3 := by
sorry

end NUMINAMATH_CALUDE_abs_difference_of_roots_l3061_306140


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l3061_306194

/-- The quadrant in which an angle falls -/
inductive Quadrant
| I
| II
| III
| IV

/-- Determines the quadrant of an angle in degrees -/
def angle_quadrant (angle : Int) : Quadrant :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle && normalized_angle < 90 then Quadrant.I
  else if 90 ≤ normalized_angle && normalized_angle < 180 then Quadrant.II
  else if 180 ≤ normalized_angle && normalized_angle < 270 then Quadrant.III
  else Quadrant.IV

theorem terminal_side_quadrant :
  angle_quadrant (-1060) = Quadrant.I :=
sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l3061_306194


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3061_306105

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = (1 : ℝ) / 3 ∧ x₂ = (3 : ℝ) / 2 ∧ 
  (∀ x : ℝ, -6 * x^2 + 11 * x - 3 = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3061_306105


namespace NUMINAMATH_CALUDE_circle_configuration_theorem_l3061_306145

/-- Represents a configuration of three circles tangent to each other and a line -/
structure CircleConfiguration where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < c
  h2 : c < b
  h3 : b < a

/-- The relation between radii of mutually tangent circles according to Descartes' theorem -/
def descartes_relation (config : CircleConfiguration) : Prop :=
  ((1 / config.a + 1 / config.b + 1 / config.c) ^ 2 : ℝ) = 
  2 * ((1 / config.a ^ 2 + 1 / config.b ^ 2 + 1 / config.c ^ 2) : ℝ)

/-- A configuration is nice if all radii are integers -/
def is_nice (config : CircleConfiguration) : Prop :=
  ∃ (i j k : ℕ), (config.a = i) ∧ (config.b = j) ∧ (config.c = k)

theorem circle_configuration_theorem :
  ∀ (config : CircleConfiguration),
  descartes_relation config →
  (∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    (config.a = 16 ∧ config.b = 4 → |config.c - 40| < ε)) ∧
  (∀ (nice_config : CircleConfiguration),
    is_nice nice_config → descartes_relation nice_config → 
    nice_config.c ≥ 2) ∧
  (∃ (nice_config : CircleConfiguration),
    is_nice nice_config ∧ descartes_relation nice_config ∧ nice_config.c = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_configuration_theorem_l3061_306145


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_power_2017_l3061_306109

theorem last_four_digits_of_5_power_2017 (h1 : 5^5 % 10000 = 3125) 
                                         (h2 : 5^6 % 10000 = 5625) 
                                         (h3 : 5^7 % 10000 = 8125) : 
  5^2017 % 10000 = 3125 := by
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_power_2017_l3061_306109


namespace NUMINAMATH_CALUDE_A_more_likely_to_win_prob_at_least_one_wins_l3061_306100

-- Define the probabilities for A and B in each round
def prob_A_first : ℚ := 3/5
def prob_A_second : ℚ := 2/3
def prob_B_first : ℚ := 3/4
def prob_B_second : ℚ := 2/5

-- Define the probability of winning for each participant
def prob_A_win : ℚ := prob_A_first * prob_A_second
def prob_B_win : ℚ := prob_B_first * prob_B_second

-- Theorem 1: A has a greater probability of winning than B
theorem A_more_likely_to_win : prob_A_win > prob_B_win := by sorry

-- Theorem 2: The probability that at least one of A and B wins is 29/50
theorem prob_at_least_one_wins : 1 - (1 - prob_A_win) * (1 - prob_B_win) = 29/50 := by sorry

end NUMINAMATH_CALUDE_A_more_likely_to_win_prob_at_least_one_wins_l3061_306100


namespace NUMINAMATH_CALUDE_gcd_282_470_l3061_306111

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_CALUDE_gcd_282_470_l3061_306111


namespace NUMINAMATH_CALUDE_bumper_car_line_joiners_l3061_306113

/-- The number of new people who joined a line for bumper cars at a fair -/
theorem bumper_car_line_joiners (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 12 → left = 10 → final = 17 → final - (initial - left) = 15 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_joiners_l3061_306113


namespace NUMINAMATH_CALUDE_courtyard_diagonal_length_l3061_306122

/-- Represents the length of the diagonal of a rectangular courtyard -/
def diagonal_length (side_ratio : ℚ) (paving_cost : ℚ) (cost_per_sqm : ℚ) : ℚ :=
  let longer_side := 4 * (paving_cost / cost_per_sqm / (12 * side_ratio)).sqrt
  let shorter_side := 3 * (paving_cost / cost_per_sqm / (12 * side_ratio)).sqrt
  (longer_side^2 + shorter_side^2).sqrt

/-- Theorem: The diagonal length of the courtyard is 50 meters -/
theorem courtyard_diagonal_length :
  diagonal_length (4/3) 600 0.5 = 50 := by
  sorry

#eval diagonal_length (4/3) 600 0.5

end NUMINAMATH_CALUDE_courtyard_diagonal_length_l3061_306122


namespace NUMINAMATH_CALUDE_arrangement_count_10_l3061_306118

/-- The number of ways to choose a president, vice-president, and committee from a group. -/
def arrangementCount (n : ℕ) : ℕ :=
  let presidentChoices := n
  let vicePresidentChoices := n - 1
  let officerArrangements := presidentChoices * vicePresidentChoices
  let remainingPeople := n - 2
  let committeeArrangements := remainingPeople.choose 3
  officerArrangements * committeeArrangements

/-- Theorem stating the number of arrangements for a group of 10 people. -/
theorem arrangement_count_10 : arrangementCount 10 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_10_l3061_306118


namespace NUMINAMATH_CALUDE_century_park_weed_removal_l3061_306172

/-- Represents the weed growth and removal scenario in Century Park --/
structure WeedScenario where
  weed_growth_rate : ℝ
  worker_removal_rate : ℝ
  day1_duration : ℕ
  day2_workers : ℕ
  day2_duration : ℕ

/-- Calculates the finish time for day 3 given a WeedScenario --/
def day3_finish_time (scenario : WeedScenario) : ℕ :=
  sorry

/-- The theorem states that given the specific scenario, 8 workers will finish at 8:38 AM on day 3 --/
theorem century_park_weed_removal 
  (scenario : WeedScenario)
  (h1 : scenario.day1_duration = 60)
  (h2 : scenario.day2_workers = 10)
  (h3 : scenario.day2_duration = 30) :
  day3_finish_time scenario = 38 :=
sorry

end NUMINAMATH_CALUDE_century_park_weed_removal_l3061_306172


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3061_306161

theorem scientific_notation_equivalence :
  686530000 = 6.8653 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3061_306161


namespace NUMINAMATH_CALUDE_num_women_is_sixteen_l3061_306174

-- Define the number of men
def num_men : ℕ := 24

-- Define the daily wage of a man
def man_wage : ℕ := 350

-- Define the total daily wage
def total_wage : ℕ := 11600

-- Define the number of women in the second condition
def women_in_second_condition : ℕ := 37

-- Define the function to calculate the number of women
def calculate_women : ℕ := 16

-- Theorem statement
theorem num_women_is_sixteen :
  ∃ (women_wage : ℕ),
    -- Condition 1: Total wage equation
    num_men * man_wage + calculate_women * women_wage = total_wage ∧
    -- Condition 2: Half men and 37 women earn the same as all men and all women
    (num_men / 2) * man_wage + women_in_second_condition * women_wage = num_men * man_wage + calculate_women * women_wage :=
by
  sorry


end NUMINAMATH_CALUDE_num_women_is_sixteen_l3061_306174


namespace NUMINAMATH_CALUDE_auction_sale_total_l3061_306150

/-- Calculate the total amount received from selling a TV and a phone at an auction -/
theorem auction_sale_total (tv_initial_cost phone_initial_cost : ℚ) 
  (tv_price_increase phone_price_increase : ℚ) : ℚ :=
  by
  -- Define the initial costs and price increases
  have h1 : tv_initial_cost = 500 := by sorry
  have h2 : tv_price_increase = 2 / 5 := by sorry
  have h3 : phone_initial_cost = 400 := by sorry
  have h4 : phone_price_increase = 40 / 100 := by sorry

  -- Calculate the final prices
  let tv_final_price := tv_initial_cost + tv_initial_cost * tv_price_increase
  let phone_final_price := phone_initial_cost + phone_initial_cost * phone_price_increase

  -- Calculate the total amount received
  let total_amount := tv_final_price + phone_final_price

  -- Prove that the total amount is equal to 1260
  sorry

end NUMINAMATH_CALUDE_auction_sale_total_l3061_306150


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l3061_306184

/-- Represents a square pattern of tiles -/
structure TilePattern :=
  (black : ℕ)
  (white : ℕ)

/-- Represents the extended pattern with a black border -/
def extendPattern (p : TilePattern) : TilePattern :=
  let side := Nat.sqrt (p.black + p.white)
  let newBlack := p.black + 4 * side + 4
  { black := newBlack, white := p.white }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern) :
  p.black = 13 ∧ p.white = 23 →
  let ep := extendPattern p
  (ep.black : ℚ) / ep.white = 41 / 23 := by
  sorry


end NUMINAMATH_CALUDE_extended_pattern_ratio_l3061_306184


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l3061_306128

theorem joshua_bottle_caps (initial : ℕ) (final : ℕ) (bought : ℕ) : 
  initial = 40 → final = 47 → final = initial + bought → bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l3061_306128


namespace NUMINAMATH_CALUDE_base12Addition_l3061_306121

/-- Converts a base 12 number represented as a list of digits to its decimal equivalent -/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal number to its base 12 representation -/
def decimalToBase12 (n : Nat) : List Nat :=
  if n < 12 then [n]
  else (n % 12) :: decimalToBase12 (n / 12)

/-- Represents the base 12 number 857₁₂ -/
def num1 : List Nat := [7, 5, 8]

/-- Represents the base 12 number 296₁₂ -/
def num2 : List Nat := [6, 9, 2]

/-- Represents the base 12 number B31₁₂ -/
def result : List Nat := [1, 3, 11]

theorem base12Addition :
  decimalToBase12 (base12ToDecimal num1 + base12ToDecimal num2) = result := by
  sorry

#eval base12ToDecimal num1
#eval base12ToDecimal num2
#eval base12ToDecimal result
#eval decimalToBase12 (base12ToDecimal num1 + base12ToDecimal num2)

end NUMINAMATH_CALUDE_base12Addition_l3061_306121


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l3061_306199

theorem complex_square_plus_self (z : ℂ) (h : z = -1/2 + (Real.sqrt 3)/2 * Complex.I) : z^2 + z = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l3061_306199


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l3061_306180

/-- The number of shirts -/
def num_shirts : ℕ := 8

/-- The number of ties -/
def num_ties : ℕ := 6

/-- The number of shirts that can be paired with the specific tie -/
def specific_shirts : ℕ := 2

/-- The number of different shirt-and-tie combinations -/
def total_combinations : ℕ := (num_shirts - specific_shirts) * (num_ties - 1) + specific_shirts

theorem shirt_tie_combinations : total_combinations = 32 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l3061_306180


namespace NUMINAMATH_CALUDE_angle_value_l3061_306106

theorem angle_value (θ : Real) (h : Real.tan θ = 2) : Real.sin (2 * θ + Real.pi / 2) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l3061_306106


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3061_306181

theorem sin_cos_identity : 
  Real.sin (34 * π / 180) * Real.sin (26 * π / 180) - 
  Real.cos (34 * π / 180) * Real.cos (26 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3061_306181


namespace NUMINAMATH_CALUDE_smallest_shift_l3061_306193

-- Define a periodic function g with period 25
def g (x : ℝ) : ℝ := sorry

-- Define the period of g
def period : ℝ := 25

-- State the periodicity of g
axiom g_periodic (x : ℝ) : g (x - period) = g x

-- Define the property we want to prove
def property (a : ℝ) : Prop :=
  ∀ x, g ((x - a) / 4) = g (x / 4)

-- State the theorem
theorem smallest_shift :
  (∃ a > 0, property a) ∧ (∀ a > 0, property a → a ≥ 100) :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l3061_306193


namespace NUMINAMATH_CALUDE_tan_pi_over_a_equals_sqrt_three_l3061_306108

theorem tan_pi_over_a_equals_sqrt_three (a : ℝ) (h : a ^ 3 = 27) : 
  Real.tan (π / a) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_pi_over_a_equals_sqrt_three_l3061_306108


namespace NUMINAMATH_CALUDE_night_rides_calculation_wills_ferris_wheel_rides_l3061_306148

/-- Calculates the number of night rides on a Ferris wheel -/
def night_rides (total_rides day_rides : ℕ) : ℕ :=
  total_rides - day_rides

/-- Theorem: The number of night rides is equal to the total rides minus the day rides -/
theorem night_rides_calculation (total_rides day_rides : ℕ) 
  (h : day_rides ≤ total_rides) : 
  night_rides total_rides day_rides = total_rides - day_rides := by
  sorry

/-- Given Will's specific scenario -/
theorem wills_ferris_wheel_rides : 
  night_rides 13 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_night_rides_calculation_wills_ferris_wheel_rides_l3061_306148


namespace NUMINAMATH_CALUDE_equilateral_roots_l3061_306163

/-- Given complex numbers p and q, and z₁ and z₂ being the roots of z² + pz + q = 0
    such that 0, z₁, and z₂ form an equilateral triangle in the complex plane,
    prove that p²/q = 1 -/
theorem equilateral_roots (p q z₁ z₂ : ℂ) : 
  z₁^2 + p*z₁ + q = 0 ∧ 
  z₂^2 + p*z₂ + q = 0 ∧ 
  ∃ ω : ℂ, ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁ →
  p^2 / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_roots_l3061_306163


namespace NUMINAMATH_CALUDE_square_difference_cube_and_sixth_power_equation_l3061_306149

theorem square_difference_cube_and_sixth_power_equation :
  (∀ m : ℕ, m > 1 → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3) ∧
  (∀ x y : ℕ, x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 → x = 4 ∧ y = 63) :=
by
  sorry

#check square_difference_cube_and_sixth_power_equation

end NUMINAMATH_CALUDE_square_difference_cube_and_sixth_power_equation_l3061_306149


namespace NUMINAMATH_CALUDE_bubble_sort_probability_bubble_sort_probability_proof_l3061_306152

/-- The probability that the 10th element in a random sequence of 50 distinct elements 
    will end up in the 25th position after one bubble pass -/
theorem bubble_sort_probability (n : ℕ) (h : n = 50) : ℝ :=
  24 / 25

/-- Proof of the bubble_sort_probability theorem -/
theorem bubble_sort_probability_proof (n : ℕ) (h : n = 50) : 
  bubble_sort_probability n h = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_bubble_sort_probability_proof_l3061_306152


namespace NUMINAMATH_CALUDE_rocky_total_miles_l3061_306103

/-- Rocky's training schedule for the first three days -/
def rocky_training : Fin 3 → ℕ
| 0 => 4  -- Day 1: 4 miles
| 1 => 4 * 2  -- Day 2: Double day 1
| 2 => 4 * 2 * 3  -- Day 3: Triple day 2

/-- The total miles Rocky ran in the first three days of training -/
theorem rocky_total_miles :
  (Finset.sum Finset.univ rocky_training) = 36 := by
  sorry

end NUMINAMATH_CALUDE_rocky_total_miles_l3061_306103


namespace NUMINAMATH_CALUDE_coprime_iterations_exists_coprime_polynomial_l3061_306195

/-- The polynomial f(x) = x^2007 - x^2006 + 1 -/
def f (x : ℤ) : ℤ := x^2007 - x^2006 + 1

/-- The m-th iteration of f -/
def f_iter (m : ℕ) (x : ℤ) : ℤ :=
  match m with
  | 0 => x
  | m+1 => f (f_iter m x)

theorem coprime_iterations (n : ℤ) (m : ℕ) : Int.gcd n (f_iter m n) = 1 := by
  sorry

/-- The main theorem stating that the polynomial f satisfies the required property -/
theorem exists_coprime_polynomial :
  ∃ (f : ℤ → ℤ), (∀ (x : ℤ), ∃ (a b c : ℤ), f x = a * x^2007 + b * x^2006 + c) ∧
                 (∀ (n : ℤ) (m : ℕ), Int.gcd n (f_iter m n) = 1) := by
  sorry

end NUMINAMATH_CALUDE_coprime_iterations_exists_coprime_polynomial_l3061_306195


namespace NUMINAMATH_CALUDE_isabel_paper_count_l3061_306189

/-- Given that Isabel bought some paper, used some, and has some left, 
    prove that the initial amount is the sum of used and left amounts. -/
theorem isabel_paper_count (initial used left : ℕ) 
  (h1 : used = 156)
  (h2 : left = 744)
  (h3 : initial = used + left) : 
  initial = 900 := by sorry

end NUMINAMATH_CALUDE_isabel_paper_count_l3061_306189


namespace NUMINAMATH_CALUDE_computer_price_increase_l3061_306178

theorem computer_price_increase (d : ℝ) : 
  2 * d = 560 →
  ((364 - d) / d) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3061_306178


namespace NUMINAMATH_CALUDE_eric_erasers_friends_l3061_306179

theorem eric_erasers_friends (total_erasers : ℕ) (erasers_per_friend : ℕ) (h1 : total_erasers = 9306) (h2 : erasers_per_friend = 94) :
  total_erasers / erasers_per_friend = 99 := by
sorry

end NUMINAMATH_CALUDE_eric_erasers_friends_l3061_306179


namespace NUMINAMATH_CALUDE_three_tangent_lines_imply_a_8_symmetry_of_circle_C_l3061_306186

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x + x + 2*y - 1 + m = 0

-- Define the curve (another circle)
def curve (a x y : ℝ) : Prop := x^2 + y^2 - 2*x + 8*y + a = 0

-- Theorem 1: Three common tangent lines imply a = 8
theorem three_tangent_lines_imply_a_8 :
  (∃ (a : ℝ), (∀ x y : ℝ, curve a x y) ∧ 
  (∃! (l₁ l₂ l₃ : ℝ → ℝ → Prop), 
    (∀ x y : ℝ, (l₁ x y ∨ l₂ x y ∨ l₃ x y) → (curve a x y ∨ circle_C x y)) ∧
    (∀ x y : ℝ, (l₁ x y ∨ l₂ x y ∨ l₃ x y) → 
      (∃ ε > 0, ∀ x' y' : ℝ, ((x' - x)^2 + (y' - y)^2 < ε^2) → 
        ¬(curve a x' y' ∧ circle_C x' y'))))) →
  a = 8 :=
sorry

-- Theorem 2: Symmetry of circle C with respect to line l when m = 1
theorem symmetry_of_circle_C :
  ∀ x y : ℝ, line_l 1 x y → 
  (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x + x')/2 = x ∧ (y + y')/2 = y) ∧ 
    (x^2 + (y-2)^2 = 4)) :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_imply_a_8_symmetry_of_circle_C_l3061_306186


namespace NUMINAMATH_CALUDE_town_distance_proof_l3061_306124

/-- Given a map distance and a scale, calculates the actual distance between two towns. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem stating that for a map distance of 7.5 inches and a scale of 1 inch = 8 miles,
    the actual distance between two towns is 60 miles. -/
theorem town_distance_proof :
  let map_distance : ℝ := 7.5
  let scale : ℝ := 8
  actual_distance map_distance scale = 60 := by
  sorry

end NUMINAMATH_CALUDE_town_distance_proof_l3061_306124


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l3061_306138

/-- Represents the amount of fiber a koala eats and absorbs in a day. -/
structure KoalaFiber where
  eaten : ℝ
  absorbed : ℝ
  absorption_rate : ℝ
  absorption_equation : absorbed = absorption_rate * eaten

/-- Theorem: If a koala absorbs 20% of the fiber it eats and absorbed 12 ounces
    of fiber in one day, then it ate 60 ounces of fiber that day. -/
theorem koala_fiber_intake (k : KoalaFiber) 
    (h1 : k.absorption_rate = 0.20)
    (h2 : k.absorbed = 12) : 
    k.eaten = 60 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l3061_306138


namespace NUMINAMATH_CALUDE_projection_result_l3061_306158

/-- Given two vectors a and b, if both are projected onto the same vector v
    resulting in the same vector p, then p is equal to (15/58, 35/58). -/
theorem projection_result (a b v p : ℝ × ℝ) : 
  a = (-3, 2) →
  b = (4, -1) →
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ p = k₂ • v) →
  p = (15/58, 35/58) :=
sorry

end NUMINAMATH_CALUDE_projection_result_l3061_306158


namespace NUMINAMATH_CALUDE_equation_solutions_l3061_306196

theorem equation_solutions :
  (∀ x : ℝ, 9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3061_306196


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_minus_3x_plus_4_l3061_306173

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The discriminant of the quadratic equation 5x² - 3x + 4 is -71 -/
theorem discriminant_of_5x2_minus_3x_plus_4 :
  discriminant 5 (-3) 4 = -71 := by sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_minus_3x_plus_4_l3061_306173


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3061_306119

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≤ 0}

-- State the theorem
theorem intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3061_306119


namespace NUMINAMATH_CALUDE_abs_difference_implies_abs_inequality_l3061_306191

theorem abs_difference_implies_abs_inequality (a_n l : ℝ) :
  |a_n - l| > 1 → |a_n| > 1 - |l| := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_implies_abs_inequality_l3061_306191


namespace NUMINAMATH_CALUDE_quadratic_max_condition_l3061_306133

/-- Quadratic function f(x) = x^2 + (2-a)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2-a)*x + 5

theorem quadratic_max_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ f a 1) →
  a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_condition_l3061_306133


namespace NUMINAMATH_CALUDE_birds_on_fence_l3061_306165

/-- Given an initial number of birds and a final number of birds on a fence,
    calculate the number of additional birds that joined. -/
def additional_birds (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that given 6 initial birds and 10 final birds on a fence,
    the number of additional birds that joined is 4. -/
theorem birds_on_fence : additional_birds 6 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3061_306165


namespace NUMINAMATH_CALUDE_box_side_length_l3061_306132

/-- Proves that the length of one side of a cubic box is approximately 18.17 inches
    given the cost per box, total volume needed, and total cost. -/
theorem box_side_length (cost_per_box : ℝ) (total_volume : ℝ) (total_cost : ℝ)
  (h1 : cost_per_box = 1.30)
  (h2 : total_volume = 3.06 * 1000000)
  (h3 : total_cost = 663)
  : ∃ (side_length : ℝ), abs (side_length - 18.17) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_box_side_length_l3061_306132


namespace NUMINAMATH_CALUDE_pentagon_star_area_theorem_l3061_306155

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The star formed by connecting every second vertex of the pentagon -/
def star (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set of points in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The intersection point of two line segments -/
def intersect (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The quadrilateral APQD -/
def quadrilateral_APQD (p : RegularPentagon) : Set (ℝ × ℝ) :=
  let A := p.vertices 0
  let B := p.vertices 1
  let C := p.vertices 2
  let D := p.vertices 3
  let E := p.vertices 4
  let P := intersect A C B E
  let Q := intersect B D C E
  sorry

theorem pentagon_star_area_theorem (p : RegularPentagon) 
  (h : area (star p) = 1) : 
  area (quadrilateral_APQD p) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_star_area_theorem_l3061_306155


namespace NUMINAMATH_CALUDE_fraction_value_l3061_306116

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3061_306116


namespace NUMINAMATH_CALUDE_arithmetic_harmonic_geometric_proportion_l3061_306137

theorem arithmetic_harmonic_geometric_proportion (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / ((a + b) / 2) = (2 * a * b / (a + b)) / b := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_harmonic_geometric_proportion_l3061_306137


namespace NUMINAMATH_CALUDE_count_multiples_eq_16_l3061_306164

/-- The number of positive multiples of 3 less than 150 with units digit 3 or 9 -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => n % 10 = 3 ∨ n % 10 = 9)
    (Finset.filter (fun n => n % 3 = 0) (Finset.range 150))).card

theorem count_multiples_eq_16 : count_multiples = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_eq_16_l3061_306164


namespace NUMINAMATH_CALUDE_jane_mean_score_l3061_306130

def jane_scores : List ℝ := [98, 97, 92, 85, 93, 88, 82]

theorem jane_mean_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 90.71428571428571 := by
sorry

end NUMINAMATH_CALUDE_jane_mean_score_l3061_306130


namespace NUMINAMATH_CALUDE_exactly_fourteen_numbers_l3061_306107

/-- A function that reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- The property that a number satisfies the given condition -/
def satisfies_condition (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ k : ℕ, (reverse_two_digit n - n) = k^2

/-- The theorem stating that there are exactly 14 numbers satisfying the condition -/
theorem exactly_fourteen_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 14 :=
sorry

end NUMINAMATH_CALUDE_exactly_fourteen_numbers_l3061_306107


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l3061_306154

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the relationships between jelly sales based on the given conditions -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 2 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.raspberry * 3 = s.grape ∧
  s.plum = 6

/-- Theorem stating that given the conditions, 18 jars of strawberry jelly were sold -/
theorem strawberry_jelly_sales (s : JellySales) (h : valid_jelly_sales s) : s.strawberry = 18 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_jelly_sales_l3061_306154


namespace NUMINAMATH_CALUDE_distribute_spots_correct_l3061_306135

/-- The number of ways to distribute 8 spots among 6 classes with at least one spot per class -/
def distribute_spots : ℕ := 21

/-- The number of senior classes -/
def num_classes : ℕ := 6

/-- The total number of spots to be distributed -/
def total_spots : ℕ := 8

/-- The minimum number of spots per class -/
def min_spots_per_class : ℕ := 1

theorem distribute_spots_correct :
  distribute_spots = 
    (num_classes.choose 2) + num_classes ∧
  num_classes * min_spots_per_class ≤ total_spots ∧
  total_spots - num_classes * min_spots_per_class = 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_spots_correct_l3061_306135


namespace NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_l3061_306176

theorem no_solution_iff_m_geq_two (m : ℝ) :
  (∀ x : ℝ, ¬(x < m + 1 ∧ x > 2*m - 1)) ↔ m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_l3061_306176


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l3061_306143

/-- A line passing through two points intersects the x-axis --/
theorem line_intersection_x_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 8 ∧ y₁ = 2) 
  (h_point2 : x₂ = 4 ∧ y₂ = 6) :
  ∃ x : ℝ, x = 10 ∧ 
    (y₂ - y₁) * (x - x₁) = (x₂ - x₁) * (0 - y₁) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l3061_306143


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_square_expression_l3061_306157

theorem absolute_value_sqrt_square_expression : |-7| + Real.sqrt 16 - (-3)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_square_expression_l3061_306157


namespace NUMINAMATH_CALUDE_set_equality_l3061_306117

theorem set_equality : Set ℝ := by
  have h1 : Set ℝ := {x | x = -2 ∨ x = 1}
  have h2 : Set ℝ := {x | (x - 1) * (x + 2) = 0}
  sorry

#check set_equality

end NUMINAMATH_CALUDE_set_equality_l3061_306117


namespace NUMINAMATH_CALUDE_josh_marbles_l3061_306127

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := sorry

/-- The number of marbles Josh initially had -/
def initial_marbles : ℕ := 7

/-- The number of new marbles Josh found -/
def new_marbles : ℕ := 10

/-- The difference between marbles found and marbles lost -/
def difference : ℕ := 2

theorem josh_marbles : marbles_lost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3061_306127


namespace NUMINAMATH_CALUDE_expression_simplification_l3061_306146

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -a - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3061_306146


namespace NUMINAMATH_CALUDE_range_of_a_l3061_306169

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, Real.exp x - a ≥ 0

-- State the theorem
theorem range_of_a (a : ℝ) : p a ↔ a ∈ Set.Iic (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3061_306169


namespace NUMINAMATH_CALUDE_reciprocal_product_theorem_l3061_306159

theorem reciprocal_product_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_product_theorem_l3061_306159


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3061_306182

/-- A rectangular solid with prime edge lengths and volume 1001 has surface area 622 -/
theorem rectangular_solid_surface_area :
  ∀ (a b c : ℕ),
  Prime a → Prime b → Prime c →
  a * b * c = 1001 →
  2 * (a * b + b * c + c * a) = 622 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3061_306182


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3061_306162

-- Define the triangles and their properties
structure Triangle :=
  (X Y Z : ℝ × ℝ)

def XYZ : Triangle := sorry
def PQR : Triangle := sorry

-- Define the lengths of the sides
def XY : ℝ := 9
def YZ : ℝ := 21
def XZ : ℝ := 15
def PQ : ℝ := 3
def QR : ℝ := 7

-- Define the angles
def angle_XYZ : ℝ := sorry
def angle_PQR : ℝ := sorry

-- State the theorem
theorem similar_triangles_side_length :
  angle_XYZ = angle_PQR →
  XY = 9 →
  XZ = 15 →
  PQ = 3 →
  ∃ (PR : ℝ), PR = 5 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3061_306162


namespace NUMINAMATH_CALUDE_alice_wrong_questions_l3061_306171

/-- Represents the number of questions a person got wrong in the test. -/
structure TestResult where
  wrong : ℕ

/-- Represents the test results for Alice, Beth, Charlie, Daniel, and Ellen. -/
structure TestResults where
  alice : TestResult
  beth : TestResult
  charlie : TestResult
  daniel : TestResult
  ellen : TestResult

/-- The theorem stating that Alice got 9 questions wrong given the conditions. -/
theorem alice_wrong_questions (results : TestResults) : results.alice.wrong = 9 :=
  by
  have h1 : results.alice.wrong + results.beth.wrong = results.charlie.wrong + results.daniel.wrong + results.ellen.wrong :=
    sorry
  have h2 : results.alice.wrong + results.daniel.wrong = results.beth.wrong + results.charlie.wrong + 3 :=
    sorry
  have h3 : results.charlie.wrong = 6 :=
    sorry
  have h4 : results.daniel.wrong = 8 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_alice_wrong_questions_l3061_306171


namespace NUMINAMATH_CALUDE_namjoon_marbles_l3061_306120

def marble_problem (sets : ℕ) (marbles_per_set : ℕ) (boxes : ℕ) (marbles_per_box : ℕ) : ℕ :=
  boxes * marbles_per_box - sets * marbles_per_set

theorem namjoon_marbles : marble_problem 3 7 6 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_marbles_l3061_306120


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_edge_sum_l3061_306147

theorem rectangular_parallelepiped_edge_sum (a b c : ℕ) (V : ℕ) : 
  V = a * b * c → 
  V.Prime → 
  V > 2 → 
  Odd (a + b + c) := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_edge_sum_l3061_306147


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3061_306168

theorem degree_to_radian_conversion (π : Real) :
  (180 : Real) = π → (300 : Real) * π / 180 = 5 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3061_306168


namespace NUMINAMATH_CALUDE_root_sum_zero_l3061_306197

theorem root_sum_zero (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_zero_l3061_306197


namespace NUMINAMATH_CALUDE_first_complete_shading_l3061_306177

def board_width : ℕ := 10

def shaded_square (n : ℕ) : ℕ := n * n

def is_shaded (square : ℕ) : Prop :=
  ∃ n : ℕ, shaded_square n = square

def column_of_square (square : ℕ) : ℕ :=
  (square - 1) % board_width + 1

theorem first_complete_shading :
  (∀ col : ℕ, col ≤ board_width → 
    ∃ square : ℕ, is_shaded square ∧ column_of_square square = col) ∧
  (∀ smaller : ℕ, smaller < 100 → 
    ¬(∀ col : ℕ, col ≤ board_width → 
      ∃ square : ℕ, square ≤ smaller ∧ is_shaded square ∧ column_of_square square = col)) :=
by sorry

end NUMINAMATH_CALUDE_first_complete_shading_l3061_306177


namespace NUMINAMATH_CALUDE_xy_equals_four_l3061_306141

theorem xy_equals_four (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y) 
  (h_eq : x + 4 / x = y + 4 / y) : x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_four_l3061_306141


namespace NUMINAMATH_CALUDE_reciprocal_of_2016_l3061_306101

theorem reciprocal_of_2016 : (2016⁻¹ : ℚ) = 1 / 2016 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_2016_l3061_306101


namespace NUMINAMATH_CALUDE_new_person_weight_l3061_306136

/-- Given a group of 6 people, if replacing one person weighing 75 kg with a new person
    increases the average weight by 4.5 kg, then the weight of the new person is 102 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 6 →
  weight_increase = 4.5 →
  replaced_weight = 75 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 102 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3061_306136


namespace NUMINAMATH_CALUDE_person_age_puzzle_l3061_306104

theorem person_age_puzzle : ∃ (A : ℕ), 4 * (A + 3) - 4 * (A - 3) = A ∧ A = 24 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l3061_306104


namespace NUMINAMATH_CALUDE_geometric_progression_terms_l3061_306102

-- Define the geometric progression
def geometric_progression (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

-- Define the sum of a geometric progression
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

theorem geometric_progression_terms (a : ℚ) :
  (geometric_progression a (1/3) 4 = 1/54) →
  (geometric_sum a (1/3) 5 = 121/162) →
  ∃ n : ℕ, geometric_sum a (1/3) n = 121/162 ∧ n = 5 :=
by
  sorry

#check geometric_progression_terms

end NUMINAMATH_CALUDE_geometric_progression_terms_l3061_306102


namespace NUMINAMATH_CALUDE_cos_165_degrees_l3061_306153

theorem cos_165_degrees : 
  Real.cos (165 * π / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_165_degrees_l3061_306153


namespace NUMINAMATH_CALUDE_smallest_number_with_property_l3061_306134

theorem smallest_number_with_property : ∃! n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 2 = 1 ∧
    m % 3 = 2 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 → m ≥ n) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_property_l3061_306134
