import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2380_238065

-- Problem 1
theorem problem_1 : 12 - (-1) + (-7) = 6 := by sorry

-- Problem 2
theorem problem_2 : -3.5 * (-3/4) / (7/8) = 3 := by sorry

-- Problem 3
theorem problem_3 : (1/3 - 1/6 - 1/12) * (-12) = -1 := by sorry

-- Problem 4
theorem problem_4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2380_238065


namespace NUMINAMATH_CALUDE_lianliang_run_distance_l2380_238094

/-- The length of the playground in meters -/
def playground_length : ℕ := 110

/-- The difference between the length and width of the playground in meters -/
def length_width_difference : ℕ := 15

/-- The width of the playground in meters -/
def playground_width : ℕ := playground_length - length_width_difference

/-- The perimeter of the playground in meters -/
def playground_perimeter : ℕ := (playground_length + playground_width) * 2

theorem lianliang_run_distance : playground_perimeter = 230 := by
  sorry

end NUMINAMATH_CALUDE_lianliang_run_distance_l2380_238094


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2380_238025

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2380_238025


namespace NUMINAMATH_CALUDE_dave_initial_apps_l2380_238018

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 8

/-- The number of apps Dave had left after deleting -/
def remaining_apps : ℕ := 8

/-- The initial number of apps Dave had -/
def initial_apps : ℕ := deleted_apps + remaining_apps

theorem dave_initial_apps : initial_apps = 16 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l2380_238018


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2380_238093

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

def B : Set ℤ := {x | ∃ k : ℕ, k < 3 ∧ x = 2 * k + 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2380_238093


namespace NUMINAMATH_CALUDE_equation_solution_l2380_238050

theorem equation_solution (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) 
  (h : 3 / x + 2 / y = 1 / 3) : 
  x = 9 * y / (y - 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2380_238050


namespace NUMINAMATH_CALUDE_total_toy_count_l2380_238071

/-- The number of toys each person has -/
structure ToyCount where
  jaxon : ℕ
  gabriel : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def problem_conditions (tc : ToyCount) : Prop :=
  tc.jaxon = 15 ∧
  tc.gabriel = 2 * tc.jaxon ∧
  tc.jerry = tc.gabriel + 8

/-- The theorem to prove -/
theorem total_toy_count (tc : ToyCount) 
  (h : problem_conditions tc) : tc.jaxon + tc.gabriel + tc.jerry = 83 := by
  sorry


end NUMINAMATH_CALUDE_total_toy_count_l2380_238071


namespace NUMINAMATH_CALUDE_prob_five_diamond_three_l2380_238091

-- Define a standard deck of cards
def standard_deck : Nat := 52

-- Define the number of 5s in a deck
def num_fives : Nat := 4

-- Define the number of diamonds in a deck
def num_diamonds : Nat := 13

-- Define the number of 3s in a deck
def num_threes : Nat := 4

-- Define our specific event
def event_probability : ℚ :=
  (num_fives : ℚ) / standard_deck *
  (num_diamonds : ℚ) / (standard_deck - 1) *
  (num_threes : ℚ) / (standard_deck - 2)

-- Theorem statement
theorem prob_five_diamond_three :
  event_probability = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_diamond_three_l2380_238091


namespace NUMINAMATH_CALUDE_madeline_score_is_28_l2380_238069

/-- Represents the score and mistakes in a Geometry exam -/
structure GeometryExam where
  totalQuestions : ℕ
  scorePerQuestion : ℕ
  madelineMistakes : ℕ
  leoMistakes : ℕ
  brentMistakes : ℕ
  brentScore : ℕ

/-- Calculates Madeline's score in the Geometry exam -/
def madelineScore (exam : GeometryExam) : ℕ :=
  exam.totalQuestions * exam.scorePerQuestion - exam.madelineMistakes * exam.scorePerQuestion

/-- Theorem: Given the conditions, Madeline's score in the Geometry exam is 28 -/
theorem madeline_score_is_28 (exam : GeometryExam)
  (h1 : exam.madelineMistakes = 2)
  (h2 : exam.leoMistakes = 2 * exam.madelineMistakes)
  (h3 : exam.brentMistakes = exam.leoMistakes + 1)
  (h4 : exam.brentScore = 25)
  (h5 : exam.totalQuestions = exam.brentScore + exam.brentMistakes)
  (h6 : exam.scorePerQuestion = 1) :
  madelineScore exam = 28 := by
  sorry


end NUMINAMATH_CALUDE_madeline_score_is_28_l2380_238069


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2380_238062

def digit_product (n : ℕ) : ℕ := 
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n % 10) + digit_sum (n / 10)

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem unique_number_satisfying_conditions : 
  ∃! x : ℕ, digit_product x = 44 * x - 86868 ∧ is_cube (digit_sum x) ∧ x = 1989 :=
sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2380_238062


namespace NUMINAMATH_CALUDE_optimization_problem_l2380_238083

theorem optimization_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 2 * x₀ * y₀ = 1/4) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 2 * x₁ * y₁ ≤ 1/4) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 4 * x₀^2 + y₀^2 = 1/2) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 4 * x₁^2 + y₁^2 ≥ 1/2) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2 * Real.sqrt 2) ∧
  (∀ (x₁ y₁ : ℝ), x₁ > 0 → y₁ > 0 → 2 * x₁ + y₁ = 1 → 1/x₁ + 1/y₁ ≥ 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_optimization_problem_l2380_238083


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l2380_238057

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 37

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

theorem fgh_supermarkets_count :
  us_supermarkets = 37 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l2380_238057


namespace NUMINAMATH_CALUDE_playground_children_count_l2380_238058

/-- Calculate the final number of children on the playground --/
theorem playground_children_count 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (additional_girls : ℕ) 
  (additional_boys : ℕ) 
  (children_leaving : ℕ) 
  (h1 : initial_girls = 28)
  (h2 : initial_boys = 35)
  (h3 : additional_girls = 5)
  (h4 : additional_boys = 7)
  (h5 : children_leaving = 15) : 
  (initial_girls + initial_boys + additional_girls + additional_boys) - children_leaving = 60 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l2380_238058


namespace NUMINAMATH_CALUDE_water_level_rise_l2380_238047

/-- The rise in water level when a cube is immersed in a rectangular vessel --/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 10)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 10 ^ 3 / (20 * 15) :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l2380_238047


namespace NUMINAMATH_CALUDE_integral_always_positive_l2380_238076

-- Define a continuous function f that is always positive
variable (f : ℝ → ℝ)
variable (hf : Continuous f)
variable (hfpos : ∀ x, f x > 0)

-- Define the integral bounds
variable (a b : ℝ)
variable (hab : a < b)

-- Theorem statement
theorem integral_always_positive :
  ∫ x in a..b, f x > 0 := by sorry

end NUMINAMATH_CALUDE_integral_always_positive_l2380_238076


namespace NUMINAMATH_CALUDE_min_value_problem_l2380_238097

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_constraint : x + Real.sqrt 3 * y + z = 6) :
  ∃ (min_val : ℝ), min_val = 37/4 ∧ 
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
  x' + Real.sqrt 3 * y' + z' = 6 → 
  x'^3 + y'^2 + 3*z' ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2380_238097


namespace NUMINAMATH_CALUDE_clothing_discount_problem_l2380_238086

theorem clothing_discount_problem (discount_rate : ℝ) (savings : ℝ) 
  (h1 : discount_rate = 0.2)
  (h2 : savings = 10)
  (h3 : ∀ x, (1 - discount_rate) * (x + savings) = x) :
  ∃ x, x = 40 ∧ (1 - discount_rate) * (x + savings) = x := by
sorry

end NUMINAMATH_CALUDE_clothing_discount_problem_l2380_238086


namespace NUMINAMATH_CALUDE_total_results_l2380_238088

theorem total_results (average : ℝ) (first_12_avg : ℝ) (last_12_avg : ℝ) (result_13 : ℝ) :
  average = 24 →
  first_12_avg = 14 →
  last_12_avg = 17 →
  result_13 = 228 →
  ∃ (n : ℕ), n = 25 ∧ (12 * first_12_avg + result_13 + 12 * last_12_avg) / n = average :=
by
  sorry


end NUMINAMATH_CALUDE_total_results_l2380_238088


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2380_238028

theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 4*x^2 + 14*x + a = (2*x + b)^2) → a = 49/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l2380_238028


namespace NUMINAMATH_CALUDE_fraction_simplification_l2380_238008

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (3*x^3 - 2*x^2 - 5*x + 1) / ((x+1)*(x-2)) - (2*x^2 - 7*x + 3) / ((x+1)*(x-2)) =
  (x-1)*(3*x^2 - x + 2) / ((x+1)*(x-2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2380_238008


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2380_238019

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3*x + 2) / Real.log (1/3)

theorem f_strictly_increasing :
  ∀ x y, x < y ∧ y < 1 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2380_238019


namespace NUMINAMATH_CALUDE_triangle_side_length_l2380_238085

theorem triangle_side_length 
  (a b c : ℝ) 
  (B : ℝ) 
  (h1 : b = 3) 
  (h2 : c = Real.sqrt 6) 
  (h3 : B = π / 3) 
  (h4 : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B) : 
  a = (Real.sqrt 6 + 3 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2380_238085


namespace NUMINAMATH_CALUDE_intersection_M_N_l2380_238027

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x * Real.log x > 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2380_238027


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2380_238054

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inequality_solution_set (x : ℝ) :
  (f (2*x) + f (x-1) < 0) ↔ (x < 1/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2380_238054


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2380_238068

theorem average_of_three_numbers (x : ℝ) : 
  (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2380_238068


namespace NUMINAMATH_CALUDE_international_data_daily_cost_l2380_238092

def regular_plan_cost : ℚ := 175
def total_charges : ℚ := 210
def stay_duration : ℕ := 10

theorem international_data_daily_cost : 
  (total_charges - regular_plan_cost) / stay_duration = (35 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_international_data_daily_cost_l2380_238092


namespace NUMINAMATH_CALUDE_product_sum_fractions_l2380_238059

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l2380_238059


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l2380_238026

/-- Proves that adding 3 liters of pure alcohol to a 6-liter solution that is 25% alcohol 
    will result in a solution that is 50% alcohol. -/
theorem alcohol_concentration_proof 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_alcohol : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.25)
  (h3 : added_alcohol = 3)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_concentration_proof_l2380_238026


namespace NUMINAMATH_CALUDE_total_amount_is_70000_l2380_238007

/-- The total amount of money divided -/
def total_amount : ℕ := sorry

/-- The amount given at 10% interest -/
def amount_10_percent : ℕ := 60000

/-- The amount given at 20% interest -/
def amount_20_percent : ℕ := sorry

/-- The interest rate for the first part (10%) -/
def interest_rate_10 : ℚ := 1/10

/-- The interest rate for the second part (20%) -/
def interest_rate_20 : ℚ := 1/5

/-- The total profit after one year -/
def total_profit : ℕ := 8000

/-- Theorem stating that the total amount divided is 70,000 -/
theorem total_amount_is_70000 :
  total_amount = 70000 ∧
  amount_10_percent + amount_20_percent = total_amount ∧
  amount_10_percent * interest_rate_10 + amount_20_percent * interest_rate_20 = total_profit :=
sorry

end NUMINAMATH_CALUDE_total_amount_is_70000_l2380_238007


namespace NUMINAMATH_CALUDE_data_grouping_l2380_238036

theorem data_grouping (max min interval : ℕ) (h1 : max = 145) (h2 : min = 50) (h3 : interval = 10) :
  (max - min + interval - 1) / interval = 10 := by
  sorry

end NUMINAMATH_CALUDE_data_grouping_l2380_238036


namespace NUMINAMATH_CALUDE_evaluate_expression_l2380_238048

theorem evaluate_expression : (2^(2+1) - 4*(2-1)^2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2380_238048


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l2380_238079

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), 4 * x^3 + 6 * x^2 + 11 * x - 6 = (x - 1/2) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l2380_238079


namespace NUMINAMATH_CALUDE_number_difference_l2380_238056

theorem number_difference (L S : ℕ) (h1 : L = 1631) (h2 : L = 6 * S + 35) : L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2380_238056


namespace NUMINAMATH_CALUDE_min_sum_at_24_l2380_238061

/-- Arithmetic sequence with general term a_n = 2n - 49 -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

theorem min_sum_at_24 :
  ∀ k : ℕ, k ≠ 0 → S 24 ≤ S k :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_24_l2380_238061


namespace NUMINAMATH_CALUDE_wristbands_per_spectator_l2380_238040

theorem wristbands_per_spectator (total_wristbands : ℕ) (total_spectators : ℕ) 
  (h1 : total_wristbands = 290) 
  (h2 : total_spectators = 145) :
  total_wristbands / total_spectators = 2 := by
  sorry

end NUMINAMATH_CALUDE_wristbands_per_spectator_l2380_238040


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2380_238096

-- Problem 1
theorem problem_1 (z : ℂ) (h : z = (Complex.I - 1) / Real.sqrt 2) :
  z^20 + z^10 + 1 = -Complex.I := by sorry

-- Problem 2
theorem problem_2 (z : ℂ) (h : Complex.abs (z - (3 + 4*Complex.I)) = 1) :
  4 ≤ Complex.abs z ∧ Complex.abs z ≤ 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2380_238096


namespace NUMINAMATH_CALUDE_frog_probability_l2380_238015

/-- Represents the probability of ending on a vertical side from a given position -/
def P (x y : ℝ) : ℝ := sorry

/-- The square's dimensions -/
def squareSize : ℝ := 6

theorem frog_probability :
  /- Starting position -/
  let startX : ℝ := 2
  let startY : ℝ := 2

  /- Conditions -/
  (∀ x y, 0 ≤ x ∧ x ≤ squareSize ∧ 0 ≤ y ∧ y ≤ squareSize →
    P x y = (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1)) / 4) →
  (∀ y, 0 ≤ y ∧ y ≤ squareSize → P 0 y = 1 ∧ P squareSize y = 1) →
  (∀ x, 0 ≤ x ∧ x ≤ squareSize → P x 0 = 0 ∧ P x squareSize = 0) →
  (∀ x y, P x y = P (squareSize - x) (squareSize - y)) →

  /- Conclusion -/
  P startX startY = 2/3 := by sorry

end NUMINAMATH_CALUDE_frog_probability_l2380_238015


namespace NUMINAMATH_CALUDE_exists_32_chinese_l2380_238041

/-- Represents the seating arrangement of businessmen at a round table. -/
structure Seating :=
  (japanese : ℕ)
  (korean : ℕ)
  (chinese : ℕ)
  (total : ℕ)
  (total_eq : japanese + korean + chinese = total)
  (japanese_positive : japanese > 0)

/-- The condition that between any two nearest Japanese, there are exactly as many Chinese as Koreans. -/
def equal_distribution (s : Seating) : Prop :=
  ∃ k : ℕ, s.chinese = k * s.japanese ∧ s.korean = k * s.japanese

/-- The main theorem stating that it's possible to have 32 Chinese in a valid seating arrangement. -/
theorem exists_32_chinese : 
  ∃ s : Seating, s.total = 50 ∧ equal_distribution s ∧ s.chinese = 32 :=
sorry


end NUMINAMATH_CALUDE_exists_32_chinese_l2380_238041


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2380_238010

theorem exponent_multiplication (m : ℝ) : 5 * m^2 * m^3 = 5 * m^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2380_238010


namespace NUMINAMATH_CALUDE_stating_pond_population_species_c_l2380_238042

/-- Represents the number of fish initially tagged for each species -/
def initial_tagged : ℕ := 40

/-- Represents the total number of fish caught in the second catch -/
def second_catch : ℕ := 180

/-- Represents the number of tagged fish of Species C found in the second catch -/
def tagged_species_c : ℕ := 2

/-- Represents the total number of fish of Species C in the pond -/
def total_species_c : ℕ := 3600

/-- 
Theorem stating that given the conditions from the problem, 
the total number of fish for Species C in the pond is 3600 
-/
theorem pond_population_species_c : 
  initial_tagged * second_catch / tagged_species_c = total_species_c := by
  sorry

end NUMINAMATH_CALUDE_stating_pond_population_species_c_l2380_238042


namespace NUMINAMATH_CALUDE_burger_cost_l2380_238077

/-- Represents the cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ
  fry : ℕ

/-- Alice's purchase -/
def alice_purchase (c : Cost) : ℕ :=
  4 * c.burger + 2 * c.soda + 3 * c.fry

/-- Bill's purchase -/
def bill_purchase (c : Cost) : ℕ :=
  3 * c.burger + c.soda + 2 * c.fry

theorem burger_cost :
  ∃ (c : Cost), alice_purchase c = 480 ∧ bill_purchase c = 360 ∧ c.burger = 80 :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_l2380_238077


namespace NUMINAMATH_CALUDE_advanced_purchase_ticket_price_l2380_238003

/-- Given information about ticket sales for an art exhibition, prove the price of advanced-purchase tickets. -/
theorem advanced_purchase_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℚ)
  (door_price : ℚ)
  (advanced_tickets : ℕ)
  (h_total_tickets : total_tickets = 140)
  (h_total_revenue : total_revenue = 1720)
  (h_door_price : door_price = 14)
  (h_advanced_tickets : advanced_tickets = 100) :
  ∃ (advanced_price : ℚ),
    advanced_price * advanced_tickets + door_price * (total_tickets - advanced_tickets) = total_revenue ∧
    advanced_price = 11.60 :=
by sorry

end NUMINAMATH_CALUDE_advanced_purchase_ticket_price_l2380_238003


namespace NUMINAMATH_CALUDE_find_y_value_l2380_238006

theorem find_y_value (x y : ℝ) 
  (h1 : (100 + 200 + 300 + x) / 4 = 250)
  (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : 
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l2380_238006


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2380_238070

/-- Given a circle with radius R and four smaller circles with radius R/2 constructed
    as described in the problem, this theorem states that the sum of the areas of the
    overlapping parts of the smaller circles equals the area of the original circle
    minus the areas of the non-overlapping parts of the smaller circles. -/
theorem circle_area_theorem (R : ℝ) (h : R > 0) :
  let big_circle_area := π * R^2
  let small_circle_area := π * (R/2)^2
  let segment_area := (π/4 - 1/2) * R^2
  let overlap_area := 2 * segment_area
  overlap_area = big_circle_area - 4 * (small_circle_area - segment_area) := by
  sorry


end NUMINAMATH_CALUDE_circle_area_theorem_l2380_238070


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l2380_238014

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 250

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 50

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, d > 0 ∧ d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l2380_238014


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2380_238031

/-- The number of apples handed out to students -/
def apples_handed_out : ℕ := 5

/-- The number of pies made -/
def pies_made : ℕ := 9

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 5

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := apples_handed_out + pies_made * apples_per_pie

theorem cafeteria_apples : initial_apples = 50 := by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2380_238031


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2380_238051

/-- The length of the real axis of a hyperbola with equation x²/3 - y²/6 = 1 is 2√3 -/
theorem hyperbola_real_axis_length : 
  ∃ (f : ℝ × ℝ → ℝ), 
    (∀ x y, f (x, y) = x^2 / 3 - y^2 / 6) ∧ 
    (∃ a : ℝ, a > 0 ∧ (∀ x y, f (x, y) = 1 → x^2 / a^2 - y^2 / (2*a^2) = 1) ∧ 2*a = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2380_238051


namespace NUMINAMATH_CALUDE_multiple_of_seven_l2380_238038

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def six_digit_number (d : ℕ) : ℕ := 567800 + d * 10 + 2

theorem multiple_of_seven (d : ℕ) (h : is_single_digit d) : 
  (six_digit_number d) % 7 = 0 ↔ d = 0 ∨ d = 7 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_seven_l2380_238038


namespace NUMINAMATH_CALUDE_banana_bunch_count_l2380_238032

theorem banana_bunch_count (x : ℕ) : 
  (6 * x + 5 * 7 = 83) → x = 8 := by
sorry

end NUMINAMATH_CALUDE_banana_bunch_count_l2380_238032


namespace NUMINAMATH_CALUDE_football_group_size_l2380_238005

/-- The proportion of people who like football -/
def like_football_ratio : ℚ := 24 / 60

/-- The proportion of people who play football among those who like it -/
def play_football_ratio : ℚ := 1 / 2

/-- The number of people expected to play football -/
def expected_players : ℕ := 50

/-- The total number of people in the group -/
def total_people : ℕ := 250

theorem football_group_size :
  (↑expected_players : ℚ) = like_football_ratio * play_football_ratio * total_people :=
sorry

end NUMINAMATH_CALUDE_football_group_size_l2380_238005


namespace NUMINAMATH_CALUDE_rectangle_area_l2380_238090

theorem rectangle_area (a b : ℝ) (h1 : a + b = 8) (h2 : 2*a^2 + 2*b^2 = 68) :
  a * b = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2380_238090


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2380_238001

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2380_238001


namespace NUMINAMATH_CALUDE_cube_properties_l2380_238078

/-- Given a cube with volume 343 cubic centimeters, prove its surface area and internal space diagonal --/
theorem cube_properties (V : ℝ) (h : V = 343) : 
  ∃ (s : ℝ), s > 0 ∧ s^3 = V ∧ 6 * s^2 = 294 ∧ s * Real.sqrt 3 = 7 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_properties_l2380_238078


namespace NUMINAMATH_CALUDE_x_range_theorem_l2380_238098

-- Define the points and line
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def l (a : ℝ) : ℝ → ℝ := fun x => a - x

-- Define the condition for points relative to the line
def point_line_condition (a : ℝ) : Prop :=
  (l a O.1 = O.2 ∨ l a A.1 = A.2) ∨ (l a O.1 - O.2) * (l a A.1 - A.2) < 0

-- Define the function h
def h (a : ℝ) : ℝ := a^2 + 2*a + 3

-- State the theorem
theorem x_range_theorem :
  ∀ x : ℝ, (∀ a : ℝ, point_line_condition a → x^2 + 4*x - 2 ≤ h a) ↔ -5 ≤ x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_x_range_theorem_l2380_238098


namespace NUMINAMATH_CALUDE_fraction_calculation_l2380_238033

theorem fraction_calculation (x y : ℚ) (hx : x = 2/3) (hy : y = 5/2) :
  (1/3) * x^7 * y^6 = 125/261 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2380_238033


namespace NUMINAMATH_CALUDE_max_books_borrowed_l2380_238017

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (two_books : ℕ) (three_books : ℕ) 
  (h1 : total_students = 50)
  (h2 : no_books = 10)
  (h3 : two_books = 18)
  (h4 : three_books = 8)
  (h5 : (total_students - no_books - two_books - three_books) * 7 ≤ 
        total_students * 4 - no_books * 0 - two_books * 2 - three_books * 3) :
  ∃ (max_books : ℕ), max_books = 49 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l2380_238017


namespace NUMINAMATH_CALUDE_square_root_equation_l2380_238045

theorem square_root_equation (N : ℝ) : 
  Real.sqrt (0.05 * N) * Real.sqrt 5 = 0.25 → N = 0.25 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_l2380_238045


namespace NUMINAMATH_CALUDE_walk_distance_proof_l2380_238075

def walk_duration : ℝ := 5
def min_speed : ℝ := 3
def max_speed : ℝ := 4

def possible_distance (d : ℝ) : Prop :=
  ∃ (speed : ℝ), min_speed ≤ speed ∧ speed ≤ max_speed ∧ d = speed * walk_duration

theorem walk_distance_proof :
  possible_distance 19 ∧
  ¬ possible_distance 12 ∧
  ¬ possible_distance 14 ∧
  ¬ possible_distance 24 ∧
  ¬ possible_distance 35 :=
by sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l2380_238075


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l2380_238016

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the property of being even
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the domain of the function
def domain (b : ℝ) : Set ℝ := {x | -2*b ≤ x ∧ x ≤ 3*b - 1}

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  (∀ x ∈ domain b, f a b x = f a b (-x)) →
  {y | ∃ x ∈ domain b, f a b x = y} = {y | 1 ≤ y ∧ y ≤ 5} :=
sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l2380_238016


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2380_238089

theorem division_remainder_problem (R Q D : ℕ) : 
  D = 3 * Q →
  D = 3 * R + 3 →
  251 = D * Q + R →
  R = 8 := by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2380_238089


namespace NUMINAMATH_CALUDE_count_valid_functions_l2380_238044

def polynomial_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x * f (-x) = f (x^3)

theorem count_valid_functions :
  ∃! (valid_functions : Finset (ℝ → ℝ)),
    (∀ f ∈ valid_functions, ∃ a b c d : ℝ, 
      (∀ x : ℝ, f x = polynomial_function a b c d x) ∧
      satisfies_condition f) ∧
    (Finset.card valid_functions = 12) :=
sorry

end NUMINAMATH_CALUDE_count_valid_functions_l2380_238044


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2380_238011

theorem trigonometric_identity (t : ℝ) : 
  1 + Real.sin (t/2) * Real.sin t - Real.cos (t/2) * (Real.sin t)^2 = 
  2 * (Real.cos (π/4 - t/2))^2 ↔ 
  ∃ k : ℤ, t = k * π := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2380_238011


namespace NUMINAMATH_CALUDE_distinct_numbers_count_l2380_238055

/-- Represents a two-sided card with distinct numbers on each side -/
structure Card where
  side1 : ℕ
  side2 : ℕ
  distinct : side1 ≠ side2

/-- The set of four cards as described in the problem -/
def card_set : Finset Card := sorry

/-- A function that generates all possible three-digit numbers from the card set -/
def generate_numbers (cards : Finset Card) : Finset ℕ := sorry

/-- The main theorem stating that the number of distinct three-digit numbers is 192 -/
theorem distinct_numbers_count : 
  (generate_numbers card_set).card = 192 := by sorry

end NUMINAMATH_CALUDE_distinct_numbers_count_l2380_238055


namespace NUMINAMATH_CALUDE_ricks_books_l2380_238030

theorem ricks_books (N : ℕ) : (N / 2 / 2 / 2 / 2 = 25) → N = 400 := by
  sorry

end NUMINAMATH_CALUDE_ricks_books_l2380_238030


namespace NUMINAMATH_CALUDE_felicity_lollipop_collection_l2380_238099

/-- The number of sticks needed to finish the fort -/
def total_sticks : ℕ := 400

/-- The number of times Felicity's family goes to the store per week -/
def store_visits_per_week : ℕ := 3

/-- The percentage of completion of the fort -/
def fort_completion_percentage : ℚ := 60 / 100

/-- The number of weeks Felicity has been collecting lollipops -/
def collection_weeks : ℕ := 80

theorem felicity_lollipop_collection :
  collection_weeks = (fort_completion_percentage * total_sticks) / store_visits_per_week := by
  sorry

end NUMINAMATH_CALUDE_felicity_lollipop_collection_l2380_238099


namespace NUMINAMATH_CALUDE_discount_calculation_l2380_238004

theorem discount_calculation (cost_price : ℝ) (profit_with_discount : ℝ) (profit_without_discount : ℝ) :
  cost_price = 100 ∧ profit_with_discount = 20 ∧ profit_without_discount = 25 →
  (cost_price + cost_price * profit_without_discount / 100) - (cost_price + cost_price * profit_with_discount / 100) = 5 := by
sorry

end NUMINAMATH_CALUDE_discount_calculation_l2380_238004


namespace NUMINAMATH_CALUDE_not_or_false_implies_and_or_l2380_238066

theorem not_or_false_implies_and_or (p q : Prop) :
  ¬(¬p ∨ ¬q) → (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_or_false_implies_and_or_l2380_238066


namespace NUMINAMATH_CALUDE_original_number_proof_l2380_238034

theorem original_number_proof (x : ℝ) : 
  (((x + 5) - (x - 5)) / (x + 5)) * 100 = 76.92 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2380_238034


namespace NUMINAMATH_CALUDE_amount_C_is_correct_l2380_238023

/-- The amount C receives when $5000 is divided among A, B, C, and D in the ratio 1:3:5:7 -/
def amount_C : ℚ :=
  let total_amount : ℚ := 5000
  let ratio_A : ℚ := 1
  let ratio_B : ℚ := 3
  let ratio_C : ℚ := 5
  let ratio_D : ℚ := 7
  let total_ratio : ℚ := ratio_A + ratio_B + ratio_C + ratio_D
  (total_amount / total_ratio) * ratio_C

theorem amount_C_is_correct : amount_C = 1562.50 := by
  sorry

end NUMINAMATH_CALUDE_amount_C_is_correct_l2380_238023


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2380_238024

/-- The distance between the foci of a hyperbola defined by xy = 4 is 2√10. -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 4 → (x - f₁.1)^2 / f₁.1^2 - (y - f₁.2)^2 / f₁.2^2 = 1) ∧
    (∀ (x y : ℝ), x * y = 4 → (x - f₂.1)^2 / f₂.1^2 - (y - f₂.2)^2 / f₂.2^2 = 1) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2380_238024


namespace NUMINAMATH_CALUDE_original_selling_price_l2380_238009

theorem original_selling_price (P : ℝ) : 
  (P + 0.1 * P) - ((0.9 * P) + 0.3 * (0.9 * P)) = 70 → 
  P + 0.1 * P = 1100 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l2380_238009


namespace NUMINAMATH_CALUDE_largest_quotient_from_set_l2380_238012

theorem largest_quotient_from_set : ∃ (a b : ℤ), 
  a ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) ∧ 
  b ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) ∧ 
  b ≠ 0 ∧
  (∀ (x y : ℤ), x ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) → 
                y ∈ ({-32, -5, 1, 2, 12, 15} : Set ℤ) → 
                y ≠ 0 → 
                (x : ℚ) / y ≤ (a : ℚ) / b) ∧
  (a : ℚ) / b = 32 := by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_from_set_l2380_238012


namespace NUMINAMATH_CALUDE_james_driving_distance_l2380_238063

/-- Calculates the total distance driven given multiple segments of a trip -/
def total_distance (speeds : List ℝ) (times : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) speeds times)

/-- James' driving problem -/
theorem james_driving_distance :
  let speeds : List ℝ := [30, 60, 75, 60]
  let times : List ℝ := [0.5, 0.75, 1.5, 2]
  total_distance speeds times = 292.5 := by
  sorry

end NUMINAMATH_CALUDE_james_driving_distance_l2380_238063


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_5x5x5_l2380_238046

/-- Represents a cube composed of smaller unit cubes --/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ
  painted_surface : Bool

/-- Calculates the number of unpainted cubes in a large cube --/
def count_unpainted_cubes (c : LargeCube) : ℕ :=
  if c.painted_surface then (c.side_length - 2)^3 else c.total_cubes

/-- Theorem stating that a 5x5x5 cube with painted surface has 27 unpainted cubes --/
theorem unpainted_cubes_in_5x5x5 :
  let c : LargeCube := { side_length := 5, total_cubes := 125, painted_surface := true }
  count_unpainted_cubes c = 27 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_5x5x5_l2380_238046


namespace NUMINAMATH_CALUDE_decreasing_quadratic_l2380_238064

theorem decreasing_quadratic (m : ℝ) : 
  (∀ x₁ x₂ : ℤ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → 
    x₁^2 + (m-1)*x₁ + 1 > x₂^2 + (m-1)*x₂ + 1) ↔ 
  m ≤ -8 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_l2380_238064


namespace NUMINAMATH_CALUDE_one_correct_meal_servings_l2380_238037

def number_of_people : ℕ := 10
def number_of_meal_choices : ℕ := 3
def beef_orders : ℕ := 2
def chicken_orders : ℕ := 4
def fish_orders : ℕ := 4

theorem one_correct_meal_servings :
  (∃ (ways : ℕ), 
    ways = number_of_people * 
      (((beef_orders - 1) * (chicken_orders * fish_orders)) + 
       ((chicken_orders - 1) * beef_orders * fish_orders) + 
       ((fish_orders - 1) * beef_orders * chicken_orders)) ∧
    ways = 180) :=
by sorry

end NUMINAMATH_CALUDE_one_correct_meal_servings_l2380_238037


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2380_238087

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 8) = x^2 + 5*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2380_238087


namespace NUMINAMATH_CALUDE_allan_bought_three_balloons_l2380_238074

/-- The number of balloons Allan bought at the park -/
def balloons_bought_by_allan : ℕ := 3

/-- Allan's initial number of balloons -/
def allan_initial_balloons : ℕ := 2

/-- Jake's initial number of balloons -/
def jake_initial_balloons : ℕ := 6

theorem allan_bought_three_balloons :
  balloons_bought_by_allan = 3 ∧
  allan_initial_balloons = 2 ∧
  jake_initial_balloons = 6 ∧
  jake_initial_balloons = (allan_initial_balloons + balloons_bought_by_allan + 1) :=
by sorry

end NUMINAMATH_CALUDE_allan_bought_three_balloons_l2380_238074


namespace NUMINAMATH_CALUDE_pyramid_top_value_l2380_238072

/-- Represents a three-level pyramid of numbers -/
structure NumberPyramid where
  bottomLeft : ℕ
  bottomRight : ℕ
  middleLeft : ℕ
  middleRight : ℕ
  top : ℕ

/-- Checks if a NumberPyramid is valid according to the sum rule -/
def isValidPyramid (p : NumberPyramid) : Prop :=
  p.middleLeft = p.bottomLeft ∧
  p.middleRight = p.bottomRight ∧
  p.top = p.middleLeft + p.middleRight

theorem pyramid_top_value (p : NumberPyramid) 
  (h1 : p.bottomLeft = 35)
  (h2 : p.bottomRight = 47)
  (h3 : isValidPyramid p) : 
  p.top = 82 := by
  sorry

#check pyramid_top_value

end NUMINAMATH_CALUDE_pyramid_top_value_l2380_238072


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2380_238002

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2380_238002


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2380_238053

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (5/12) * x + 43/12

-- Theorem statement
theorem circle_and_line_properties :
  -- Circle C passes through (0, 2) and (2, -2)
  circle_C 0 2 ∧ circle_C 2 (-2) ∧
  -- Center of C lies on x - y + 1 = 0
  (∃ t : ℝ, circle_C t (t + 1)) ∧
  -- Line m passes through (1, 4)
  line_m 1 4 ∧
  -- Chord length on C is 6
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →
  -- The standard equation of C is correct
  (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ circle_C x y) ∧
  -- The slope-intercept equation of m is correct
  (∀ x y : ℝ, y = (5/12) * x + 43/12 ↔ line_m x y) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2380_238053


namespace NUMINAMATH_CALUDE_certain_number_problem_l2380_238082

theorem certain_number_problem (N : ℚ) :
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 150 → N = 288 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2380_238082


namespace NUMINAMATH_CALUDE_max_type_a_accessories_l2380_238029

/-- Represents the cost and quantity of drone accessories. -/
structure DroneAccessories where
  costA : ℕ  -- Cost of type A accessory
  costB : ℕ  -- Cost of type B accessory
  totalQuantity : ℕ  -- Total number of accessories
  maxCost : ℕ  -- Maximum total cost

/-- Calculates the maximum number of type A accessories that can be purchased. -/
def maxTypeA (d : DroneAccessories) : ℕ :=
  let m := (d.maxCost - d.costB * d.totalQuantity) / (d.costA - d.costB)
  min m d.totalQuantity

/-- Theorem stating the maximum number of type A accessories that can be purchased. -/
theorem max_type_a_accessories (d : DroneAccessories) : 
  d.costA = 230 ∧ d.costB = 100 ∧ d.totalQuantity = 30 ∧ d.maxCost = 4180 ∧
  d.costA + 3 * d.costB = 530 ∧ 3 * d.costA + 2 * d.costB = 890 →
  maxTypeA d = 9 := by
  sorry

#eval maxTypeA { costA := 230, costB := 100, totalQuantity := 30, maxCost := 4180 }

end NUMINAMATH_CALUDE_max_type_a_accessories_l2380_238029


namespace NUMINAMATH_CALUDE_quadrilateral_k_value_l2380_238067

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral formed by two lines and the positive semi-axes -/
structure Quadrilateral where
  l₁ : Line
  l₂ : Line

/-- Predicate to check if a quadrilateral has a circumscribed circle -/
def has_circumscribed_circle (q : Quadrilateral) : Prop :=
  sorry

/-- The quadrilateral formed by the given lines and axes -/
def quad (k : ℝ) : Quadrilateral :=
  { l₁ := { a := 1, b := 3, c := -7 },
    l₂ := { a := k, b := 1, c := -2 } }

theorem quadrilateral_k_value :
  ∀ k : ℝ, has_circumscribed_circle (quad k) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_k_value_l2380_238067


namespace NUMINAMATH_CALUDE_perpendicular_vectors_cos2theta_l2380_238000

theorem perpendicular_vectors_cos2theta (θ : ℝ) : 
  let a : ℝ × ℝ := (1, Real.cos θ)
  let b : ℝ × ℝ := (-1, 2 * Real.cos θ)
  (a.1 * b.1 + a.2 * b.2 = 0) → Real.cos (2 * θ) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_cos2theta_l2380_238000


namespace NUMINAMATH_CALUDE_lottery_probability_l2380_238081

theorem lottery_probability (max_number : ℕ) 
  (prob_1_to_15 : ℚ) (prob_1_or_larger : ℚ) :
  max_number ≥ 15 →
  prob_1_to_15 = 1 / 3 →
  prob_1_or_larger = 2 / 3 →
  (∀ n : ℕ, n ≤ 15 → n ≥ 1) →
  (∀ n : ℕ, n ≤ max_number → n ≥ 1) →
  (probability_less_equal_15 : ℚ) →
  probability_less_equal_15 = prob_1_or_larger :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l2380_238081


namespace NUMINAMATH_CALUDE_Q_value_at_negative_one_l2380_238095

/-- The cubic polynomial P(x) -/
def P (x : ℝ) : ℝ := x^3 + 8*x^2 - x + 3

/-- The roots of P(x) -/
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

/-- Q is a monic polynomial with roots ab - c^2, ac - b^2, bc - a^2 -/
def Q (x : ℝ) : ℝ := x^3 + 67*x^2 + 67*x + 1537

theorem Q_value_at_negative_one :
  P a = 0 → P b = 0 → P c = 0 → Q (-1) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_Q_value_at_negative_one_l2380_238095


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l2380_238013

/-- The maximum distance a point can be from the origin, given the constraints --/
def max_distance : ℝ := 10

/-- The coordinates of the post where the dog is tied --/
def post : ℝ × ℝ := (6, 8)

/-- The length of the rope --/
def rope_length : ℝ := 15

/-- The x-coordinate of the wall's end --/
def wall_end : ℝ := 10

/-- Theorem stating the maximum distance from the origin --/
theorem max_distance_from_origin :
  ∀ (p : ℝ × ℝ), 
    (p.1 ≤ wall_end) → -- point is not beyond the wall
    (p.2 ≥ 0) → -- point is not below the wall
    ((p.1 - post.1)^2 + (p.2 - post.2)^2 ≤ rope_length^2) → -- point is within or on the circle
    (p.1^2 + p.2^2 ≤ max_distance^2) := -- distance from origin is at most max_distance
by
  sorry


end NUMINAMATH_CALUDE_max_distance_from_origin_l2380_238013


namespace NUMINAMATH_CALUDE_modulus_of_z_l2380_238022

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2380_238022


namespace NUMINAMATH_CALUDE_james_age_when_thomas_reaches_current_l2380_238021

theorem james_age_when_thomas_reaches_current (T : ℕ) : 
  let shay_age := T + 13
  let james_current_age := T + 18
  james_current_age = 42 →
  james_current_age + (james_current_age - T) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_james_age_when_thomas_reaches_current_l2380_238021


namespace NUMINAMATH_CALUDE_divisor_calculation_l2380_238052

theorem divisor_calculation (dividend : Float) (quotient : Float) (h1 : dividend = 0.0204) (h2 : quotient = 0.0012000000000000001) :
  dividend / quotient = 17 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l2380_238052


namespace NUMINAMATH_CALUDE_M_remainder_mod_55_l2380_238049

def M : ℕ := sorry

theorem M_remainder_mod_55 : M % 55 = 44 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_55_l2380_238049


namespace NUMINAMATH_CALUDE_f_neg_three_l2380_238073

def f (x : ℝ) : ℝ := x^2 + x

theorem f_neg_three : f (-3) = 6 := by sorry

end NUMINAMATH_CALUDE_f_neg_three_l2380_238073


namespace NUMINAMATH_CALUDE_rose_needs_more_l2380_238080

def paintbrush_cost : ℚ := 2.40
def paints_cost : ℚ := 9.20
def easel_cost : ℚ := 6.50
def rose_has : ℚ := 7.10

theorem rose_needs_more : 
  paintbrush_cost + paints_cost + easel_cost - rose_has = 11 :=
by sorry

end NUMINAMATH_CALUDE_rose_needs_more_l2380_238080


namespace NUMINAMATH_CALUDE_heathers_oranges_l2380_238084

/-- The total number of oranges Heather has after receiving more from Russell -/
def total_oranges (initial : Float) (received : Float) : Float :=
  initial + received

/-- Theorem stating that Heather's total oranges is the sum of her initial oranges and those received from Russell -/
theorem heathers_oranges (initial : Float) (received : Float) :
  total_oranges initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_heathers_oranges_l2380_238084


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l2380_238039

/-- Given a quadratic function f(x) = 3x^2 + 5x - 2, prove that when it's shifted 5 units to the left,
    resulting in a new quadratic function g(x) = ax^2 + bx + c, then a + b + c = 136. -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 + 5 * x - 2) →
  (∀ x, g x = f (x + 5)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 136 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l2380_238039


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l2380_238043

/-- Calculates the discounted price of a fruit purchase -/
def discountedPrice (quantity : ℕ) (pricePerKg : ℚ) (discountPercent : ℚ) : ℚ :=
  quantity * pricePerKg * (1 - discountPercent / 100)

/-- Represents Harkamal's fruit purchases -/
def fruitPurchases : List (ℕ × ℚ × ℚ) := [
  (10, 70, 10),  -- grapes
  (9, 55, 0),    -- mangoes
  (12, 80, 5),   -- apples
  (7, 45, 15),   -- papayas
  (15, 30, 0),   -- oranges
  (5, 25, 0)     -- bananas
]

/-- Calculates the total cost of Harkamal's fruit purchases -/
def totalCost : ℚ :=
  fruitPurchases.foldr (fun (purchase : ℕ × ℚ × ℚ) (acc : ℚ) =>
    acc + discountedPrice purchase.1 purchase.2.1 purchase.2.2
  ) 0

/-- Theorem stating that the total cost of Harkamal's fruit purchases is $2879.75 -/
theorem harkamal_fruit_purchase_cost :
  totalCost = 2879.75 := by sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l2380_238043


namespace NUMINAMATH_CALUDE_higher_rate_fewer_attendees_possible_l2380_238020

/-- Represents a workshop with attendees and total capacity -/
structure Workshop where
  attendees : ℕ
  capacity : ℕ
  attendance_rate : ℚ
  attendance_rate_def : attendance_rate = attendees / capacity

/-- Theorem stating that it's possible for a workshop to have a higher attendance rate
    but fewer attendees than another workshop -/
theorem higher_rate_fewer_attendees_possible :
  ∃ (A B : Workshop), A.attendance_rate > B.attendance_rate ∧ A.attendees < B.attendees := by
  sorry


end NUMINAMATH_CALUDE_higher_rate_fewer_attendees_possible_l2380_238020


namespace NUMINAMATH_CALUDE_two_pairs_exist_l2380_238035

/-- A function that checks if a number consists of three identical digits -/
def has_three_identical_digits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 100 + d * 10 + d

/-- The main theorem stating the existence of two distinct pairs of numbers
    satisfying the given conditions -/
theorem two_pairs_exist : ∃ (a b c d : ℕ),
  has_three_identical_digits (a * b) ∧
  has_three_identical_digits (a + b) ∧
  has_three_identical_digits (c * d) ∧
  has_three_identical_digits (c + d) ∧
  (a ≠ c ∨ b ≠ d) :=
sorry

end NUMINAMATH_CALUDE_two_pairs_exist_l2380_238035


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l2380_238060

theorem cyclist_speed_problem (x y : ℝ) :
  y = x + 5 ∧  -- Y's speed is 5 mph faster than X's
  100 / y + 1/6 + 20 / y = 80 / x + 1/4 ∧  -- Time equality equation
  x > 0 ∧ y > 0  -- Positive speeds
  → x = 10 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l2380_238060
