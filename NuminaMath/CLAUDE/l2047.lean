import Mathlib

namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l2047_204789

theorem cricket_bat_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 228) : 
  ∃ (cost_price_A : ℝ), cost_price_A = 152 ∧ 
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l2047_204789


namespace NUMINAMATH_CALUDE_decreasing_power_function_l2047_204738

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_power_function_l2047_204738


namespace NUMINAMATH_CALUDE_total_followers_after_one_month_l2047_204736

/-- Represents the number of followers on various social media platforms -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ
  pinterest : ℕ
  snapchat : ℕ

/-- Calculates the total number of followers across all platforms -/
def total_followers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube + f.pinterest + f.snapchat

/-- Represents the initial number of followers -/
def initial_followers : Followers := {
  instagram := 240,
  facebook := 500,
  twitter := (240 + 500) / 2,
  tiktok := 3 * ((240 + 500) / 2),
  youtube := 3 * ((240 + 500) / 2) + 510,
  pinterest := 120,
  snapchat := 120 / 2
}

/-- Represents the number of followers after one month -/
def followers_after_one_month : Followers := {
  instagram := initial_followers.instagram + (initial_followers.instagram * 15 / 100),
  facebook := initial_followers.facebook + (initial_followers.facebook * 20 / 100),
  twitter := initial_followers.twitter + 30,
  tiktok := initial_followers.tiktok + 45,
  youtube := initial_followers.youtube,
  pinterest := initial_followers.pinterest,
  snapchat := initial_followers.snapchat - 10
}

/-- Theorem stating that the total number of followers after one month is 4221 -/
theorem total_followers_after_one_month : 
  total_followers followers_after_one_month = 4221 := by
  sorry


end NUMINAMATH_CALUDE_total_followers_after_one_month_l2047_204736


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l2047_204723

/-- Given a triangle with vertices (4, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 24 square units, then x = -16. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * abs x * 3 = 24 → x = -16 := by sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l2047_204723


namespace NUMINAMATH_CALUDE_pen_pencil_price_ratio_l2047_204740

theorem pen_pencil_price_ratio :
  ∀ (pen_price pencil_price total_price : ℚ),
    pencil_price = 8 →
    total_price = 12 →
    total_price = pen_price + pencil_price →
    pen_price / pencil_price = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_price_ratio_l2047_204740


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l2047_204763

/-- An arithmetic sequence with given first and third terms -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum (a : ℕ → ℤ) :
  arithmetic_seq a → a 1 = 1 → a 3 = -3 →
  a 1 - a 2 - a 3 - a 4 - a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l2047_204763


namespace NUMINAMATH_CALUDE_opposite_zero_l2047_204772

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines an opposite -/
axiom opposite_def (x : ℝ) : x + opposite x = 0

/-- Theorem: The opposite of 0 is 0 -/
theorem opposite_zero : opposite 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_zero_l2047_204772


namespace NUMINAMATH_CALUDE_chloe_score_l2047_204711

/-- The score for each treasure found in the game -/
def points_per_treasure : ℕ := 9

/-- The number of treasures found on the first level -/
def treasures_level_1 : ℕ := 6

/-- The number of treasures found on the second level -/
def treasures_level_2 : ℕ := 3

/-- Chloe's total score in the game -/
def total_score : ℕ := points_per_treasure * (treasures_level_1 + treasures_level_2)

/-- Theorem stating that Chloe's total score is 81 points -/
theorem chloe_score : total_score = 81 := by
  sorry

end NUMINAMATH_CALUDE_chloe_score_l2047_204711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11_terms_l2047_204756

theorem arithmetic_sequence_11_terms (a₁ : ℕ) (d : ℕ) (n : ℕ) (aₙ : ℕ) :
  a₁ = 12 →
  d = 6 →
  n = 11 →
  aₙ = a₁ + (n - 1) * d →
  aₙ = 72 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11_terms_l2047_204756


namespace NUMINAMATH_CALUDE_marbles_distribution_l2047_204712

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 80 →
  num_boys = 8 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2047_204712


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2047_204718

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := a + 2 • (b x)
def v (x : ℝ) : ℝ × ℝ := 2 • a - b x

theorem parallel_vectors_x_value :
  ∃ x : ℝ, (∃ k : ℝ, u x = k • (v x)) ∧ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2047_204718


namespace NUMINAMATH_CALUDE_function_evaluation_l2047_204779

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 for any real number a. -/
theorem function_evaluation (a : ℝ) : (fun x : ℝ => x^2 + 1) (a + 1) = a^2 + 2*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l2047_204779


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l2047_204720

theorem max_value_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (2 * x + 27) + Real.sqrt (17 - x) + Real.sqrt (3 * x) ≤ 14.951 ∧
  ∃ x₀, x₀ = 17 ∧ Real.sqrt (2 * x₀ + 27) + Real.sqrt (17 - x₀) + Real.sqrt (3 * x₀) = 14.951 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l2047_204720


namespace NUMINAMATH_CALUDE_tape_division_l2047_204798

theorem tape_division (total_tape : ℚ) (num_packages : ℕ) :
  total_tape = 7 / 12 ∧ num_packages = 5 →
  total_tape / num_packages = 7 / 60 := by
  sorry

end NUMINAMATH_CALUDE_tape_division_l2047_204798


namespace NUMINAMATH_CALUDE_fraction_calculation_l2047_204739

theorem fraction_calculation : 
  let f1 := 531 / 135
  let f2 := 579 / 357
  let f3 := 753 / 975
  let f4 := 135 / 531
  (f1 + f2 + f3) * (f2 + f3 + f4) - (f1 + f2 + f3 + f4) * (f2 + f3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2047_204739


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l2047_204702

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  ({0, 1, a^2} : Set ℝ) = ({1, 0, 2*a + 3} : Set ℝ) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_three_l2047_204702


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l2047_204783

/-- Proves that the price of each apple is $0.80 given the conditions of the fruit stand problem -/
theorem fruit_stand_problem (total_cost : ℝ) (total_fruits : ℕ) (banana_price : ℝ) 
  (h1 : total_cost = 5.60)
  (h2 : total_fruits = 9)
  (h3 : banana_price = 0.60) :
  ∃ (apple_price : ℝ) (num_apples : ℕ),
    apple_price = 0.80 ∧
    num_apples ≤ total_fruits ∧
    apple_price * num_apples + banana_price * (total_fruits - num_apples) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l2047_204783


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2047_204732

/-- Represents a repeating decimal with a 4-digit repetend -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999

/-- The problem statement -/
theorem repeating_decimal_subtraction :
  RepeatingDecimal 4 5 6 7 - RepeatingDecimal 1 2 3 4 - RepeatingDecimal 2 3 4 5 = 988 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2047_204732


namespace NUMINAMATH_CALUDE_blue_flower_percentage_l2047_204792

theorem blue_flower_percentage (total_flowers : ℕ) (green_flowers : ℕ) (yellow_flowers : ℕ)
  (h1 : total_flowers = 96)
  (h2 : green_flowers = 9)
  (h3 : yellow_flowers = 12)
  : (total_flowers - (green_flowers + 3 * green_flowers + yellow_flowers)) / total_flowers * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blue_flower_percentage_l2047_204792


namespace NUMINAMATH_CALUDE_problem_statement_l2047_204714

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem statement
theorem problem_statement : (p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2047_204714


namespace NUMINAMATH_CALUDE_baseball_runs_proof_l2047_204767

theorem baseball_runs_proof (sequence : Fin 6 → ℕ) 
  (h1 : ∃ i, sequence i = 1)
  (h2 : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ sequence i = 5 ∧ sequence j = 5 ∧ sequence k = 5)
  (h3 : ∃ i j, i ≠ j ∧ sequence i = sequence j)
  (h4 : (Finset.sum Finset.univ (λ i => sequence i)) / 6 = 4) :
  ∃ i j, i ≠ j ∧ sequence i = sequence j ∧ sequence i = 4 := by
  sorry

end NUMINAMATH_CALUDE_baseball_runs_proof_l2047_204767


namespace NUMINAMATH_CALUDE_p_less_than_negative_one_l2047_204799

theorem p_less_than_negative_one (x y p : ℝ) 
  (eq1 : 3 * x - 2 * y = 4 - p)
  (eq2 : 4 * x - 3 * y = 2 + p)
  (ineq : x > y) : 
  p < -1 := by
sorry

end NUMINAMATH_CALUDE_p_less_than_negative_one_l2047_204799


namespace NUMINAMATH_CALUDE_money_distribution_l2047_204701

/-- Given three people A, B, and C with a total of 600 Rs between them,
    where B and C together have 450 Rs, and C has 100 Rs,
    prove that A and C together have 250 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 600 →
  B + C = 450 →
  C = 100 →
  A + C = 250 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2047_204701


namespace NUMINAMATH_CALUDE_root_equation_coefficient_l2047_204768

theorem root_equation_coefficient (a : ℝ) : (2 : ℝ)^2 + a * 2 - 2 = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_coefficient_l2047_204768


namespace NUMINAMATH_CALUDE_modulus_of_complex_l2047_204729

theorem modulus_of_complex (z : ℂ) : (1 - Complex.I) * z = 3 - Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l2047_204729


namespace NUMINAMATH_CALUDE_expression_simplification_l2047_204734

theorem expression_simplification (y : ℝ) : 
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2047_204734


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2047_204704

theorem complex_equation_solution (z : ℂ) : (z - Complex.I) * Complex.I = 2 + Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2047_204704


namespace NUMINAMATH_CALUDE_workshop_average_salary_l2047_204751

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 42)
  (h2 : technicians = 7)
  (h3 : avg_salary_technicians = 18000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l2047_204751


namespace NUMINAMATH_CALUDE_div_mul_sqrt_three_reciprocal_result_equals_one_l2047_204778

theorem div_mul_sqrt_three_reciprocal (x : ℝ) (h : x > 0) : 3 / Real.sqrt x * (1 / Real.sqrt x) = 3 / x :=
by sorry

theorem result_equals_one : 3 / Real.sqrt 3 * (1 / Real.sqrt 3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_div_mul_sqrt_three_reciprocal_result_equals_one_l2047_204778


namespace NUMINAMATH_CALUDE_initial_staples_count_l2047_204707

/-- The number of staples used per report -/
def staples_per_report : ℕ := 1

/-- The number of reports in a dozen -/
def reports_per_dozen : ℕ := 12

/-- The number of dozens of reports Stacie staples -/
def dozens_of_reports : ℕ := 3

/-- The number of staples remaining after stapling -/
def remaining_staples : ℕ := 14

/-- Theorem: The initial number of staples in the stapler is 50 -/
theorem initial_staples_count : 
  dozens_of_reports * reports_per_dozen * staples_per_report + remaining_staples = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_staples_count_l2047_204707


namespace NUMINAMATH_CALUDE_min_sum_pqrs_l2047_204777

theorem min_sum_pqrs (p q r s : ℕ) : 
  p > 1 → q > 1 → r > 1 → s > 1 →
  31 * (p + 1) = 37 * (q + 1) →
  41 * (r + 1) = 43 * (s + 1) →
  p + q + r + s ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_pqrs_l2047_204777


namespace NUMINAMATH_CALUDE_total_questions_is_100_l2047_204703

/-- Represents the scoring system and test results for a student. -/
structure TestResult where
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  total_questions : ℕ

/-- Defines the properties of a valid test result based on the given conditions. -/
def is_valid_test_result (tr : TestResult) : Prop :=
  tr.score = tr.correct_responses - 2 * tr.incorrect_responses ∧
  tr.total_questions = tr.correct_responses + tr.incorrect_responses

/-- Theorem stating that given the conditions, the total number of questions is 100. -/
theorem total_questions_is_100 (tr : TestResult) 
  (h1 : is_valid_test_result tr) 
  (h2 : tr.score = 64) 
  (h3 : tr.correct_responses = 88) : 
  tr.total_questions = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_is_100_l2047_204703


namespace NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l2047_204771

/-- Given the conversion rates between bahs, rahs, and yahs, 
    prove that 500 yahs are equal in value to 100 bahs. -/
theorem yahs_to_bahs_conversion 
  (bah_to_rah : ℚ) (rah_to_yah : ℚ)
  (h1 : bah_to_rah = 30 / 10)  -- 10 bahs = 30 rahs
  (h2 : rah_to_yah = 10 / 6)   -- 6 rahs = 10 yahs
  : 500 * (1 / rah_to_yah) * (1 / bah_to_rah) = 100 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l2047_204771


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2047_204735

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (2 : ℝ)^2 = a^2 + b^2 →
  (∀ (x y : ℝ), (b*x = a*y ∨ b*x = -a*y) → (x - 2)^2 + y^2 = 3) →
  (∀ (x y : ℝ), x^2 - y^2 / 3 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2047_204735


namespace NUMINAMATH_CALUDE_function_properties_l2047_204766

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a 1 > 0) →
  (∀ x, f a x > b ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2047_204766


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l2047_204747

/-- Given a quadratic function f(x) = ax^2 - 2x + c with x ∈ ℝ and range [0, +∞),
    the maximum value of 1/(c+1) + 4/(a+4) is 4/3 -/
theorem max_value_quadratic_function (a c : ℝ) : 
  (∀ x, a * x^2 - 2*x + c ≥ 0) →  -- Range is [0, +∞)
  (∃ x, a * x^2 - 2*x + c = 0) →  -- Minimum value is 0
  (∃ M, M = (1 / (c + 1) + 4 / (a + 4)) ∧ 
   ∀ a' c', (∀ x, a' * x^2 - 2*x + c' ≥ 0) → 
             (∃ x, a' * x^2 - 2*x + c' = 0) → 
             M ≥ (1 / (c' + 1) + 4 / (a' + 4))) →
  (1 / (c + 1) + 4 / (a + 4)) ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l2047_204747


namespace NUMINAMATH_CALUDE_stratified_sampling_grade10_l2047_204705

theorem stratified_sampling_grade10 (total_students : ℕ) (sample_size : ℕ) (grade10_in_sample : ℕ) :
  total_students = 1800 →
  sample_size = 90 →
  grade10_in_sample = 42 →
  (grade10_in_sample : ℚ) / (sample_size : ℚ) = (840 : ℚ) / (total_students : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_grade10_l2047_204705


namespace NUMINAMATH_CALUDE_min_value_of_cosine_sum_l2047_204797

theorem min_value_of_cosine_sum (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : 0 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : 0 ≤ z ∧ z ≤ Real.pi / 2) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_cosine_sum_l2047_204797


namespace NUMINAMATH_CALUDE_perfect_square_sequence_l2047_204726

theorem perfect_square_sequence (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sequence_l2047_204726


namespace NUMINAMATH_CALUDE_parabola_intersection_l2047_204769

-- Define the two parabolas
def f (x : ℝ) : ℝ := 4 * x^2 + 6 * x - 7
def g (x : ℝ) : ℝ := 2 * x^2 + 5

-- Define the intersection points
def p1 : ℝ × ℝ := (-4, 33)
def p2 : ℝ × ℝ := (1.5, 11)

-- Theorem statement
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p1 ∨ (x, y) = p2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2047_204769


namespace NUMINAMATH_CALUDE_ratio_of_powers_compute_power_ratio_l2047_204708

theorem ratio_of_powers (a b : ℕ) (n : ℕ) (h : b ≠ 0) :
  (a ^ n) / (b ^ n) = (a / b) ^ n :=
sorry

theorem compute_power_ratio :
  (90000 ^ 5) / (30000 ^ 5) = 243 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_powers_compute_power_ratio_l2047_204708


namespace NUMINAMATH_CALUDE_average_score_is_two_l2047_204746

/-- Represents the score distribution of a test --/
structure ScoreDistribution where
  three_points : ℝ
  two_points : ℝ
  one_point : ℝ
  zero_points : ℝ
  sum_to_one : three_points + two_points + one_point + zero_points = 1

/-- Calculates the average score given a score distribution --/
def average_score (sd : ScoreDistribution) : ℝ :=
  3 * sd.three_points + 2 * sd.two_points + 1 * sd.one_point + 0 * sd.zero_points

/-- The main theorem stating that the average score is 2 points --/
theorem average_score_is_two (sd : ScoreDistribution)
  (h1 : sd.three_points = 0.3)
  (h2 : sd.two_points = 0.5)
  (h3 : sd.one_point = 0.1)
  (h4 : sd.zero_points = 0.1) :
  average_score sd = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_two_l2047_204746


namespace NUMINAMATH_CALUDE_jerry_added_two_figures_l2047_204733

/-- The number of action figures Jerry added to his shelf -/
def action_figures_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem: Jerry added 2 action figures to his shelf -/
theorem jerry_added_two_figures : action_figures_added 8 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_two_figures_l2047_204733


namespace NUMINAMATH_CALUDE_f_composed_three_roots_l2047_204717

/-- A quadratic function f(x) = x^2 - 4x + c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^2 - 4*x + c

/-- The composition of f with itself -/
def f_composed (c : ℝ) : ℝ → ℝ := fun x ↦ f c (f c x)

/-- Predicate for a function having exactly three distinct real roots -/
def has_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g x = 0 ∧ g y = 0 ∧ g z = 0 ∧
    ∀ w, g w = 0 → w = x ∨ w = y ∨ w = z

theorem f_composed_three_roots :
  ∀ c : ℝ, has_three_distinct_real_roots (f_composed c) ↔ c = 8 :=
sorry

end NUMINAMATH_CALUDE_f_composed_three_roots_l2047_204717


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l2047_204794

theorem greatest_integer_inequality : 
  (∀ x : ℤ, (1 / 4 : ℚ) + (x : ℚ) / 9 < 7 / 8 → x ≤ 5) ∧ 
  ((1 / 4 : ℚ) + (5 : ℚ) / 9 < 7 / 8) := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l2047_204794


namespace NUMINAMATH_CALUDE_convex_polygon_three_obtuse_sides_l2047_204761

/-- A convex polygon with n sides and exactly 3 obtuse angles -/
structure ConvexPolygon (n : ℕ) :=
  (sides : ℕ)
  (is_convex : Bool)
  (obtuse_angles : ℕ)
  (sides_eq : sides = n)
  (convex : is_convex = true)
  (obtuse : obtuse_angles = 3)

/-- The theorem stating that a convex polygon with exactly 3 obtuse angles can only have 5 or 6 sides -/
theorem convex_polygon_three_obtuse_sides (n : ℕ) (p : ConvexPolygon n) : n = 5 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_three_obtuse_sides_l2047_204761


namespace NUMINAMATH_CALUDE_first_box_nonempty_count_l2047_204782

def total_boxes : Nat := 4
def total_balls : Nat := 3

def ways_with_first_box_nonempty : Nat :=
  total_boxes ^ total_balls - (total_boxes - 1) ^ total_balls

theorem first_box_nonempty_count :
  ways_with_first_box_nonempty = 37 := by
  sorry

end NUMINAMATH_CALUDE_first_box_nonempty_count_l2047_204782


namespace NUMINAMATH_CALUDE_cafeteria_cottage_pies_l2047_204742

/-- The number of lasagnas made by the cafeteria -/
def num_lasagnas : ℕ := 100

/-- The amount of ground mince used per lasagna (in pounds) -/
def mince_per_lasagna : ℕ := 2

/-- The amount of ground mince used per cottage pie (in pounds) -/
def mince_per_cottage_pie : ℕ := 3

/-- The total amount of ground mince used (in pounds) -/
def total_mince : ℕ := 500

/-- The number of cottage pies made by the cafeteria -/
def num_cottage_pies : ℕ := (total_mince - num_lasagnas * mince_per_lasagna) / mince_per_cottage_pie

theorem cafeteria_cottage_pies :
  num_cottage_pies = 100 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_cottage_pies_l2047_204742


namespace NUMINAMATH_CALUDE_inequality_proof_l2047_204730

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a / b < c / d) : 
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2047_204730


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2047_204724

/-- Given that i is the imaginary unit, prove that (3 + 2i) / (1 - i) = 1/2 + 5/2 * i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 2 * i) / (1 - i) = 1/2 + 5/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2047_204724


namespace NUMINAMATH_CALUDE_health_codes_suitable_for_comprehensive_survey_other_options_not_suitable_for_comprehensive_survey_l2047_204780

/-- Represents a survey option --/
inductive SurveyOption
  | MovieViewing
  | SeedGermination
  | WaterQuality
  | HealthCodes

/-- Determines if a survey option is suitable for a comprehensive survey --/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.HealthCodes => true
  | _ => false

/-- Theorem stating that the health codes survey is suitable for a comprehensive survey --/
theorem health_codes_suitable_for_comprehensive_survey :
  isSuitableForComprehensiveSurvey SurveyOption.HealthCodes = true :=
sorry

/-- Theorem stating that other survey options are not suitable for a comprehensive survey --/
theorem other_options_not_suitable_for_comprehensive_survey (option : SurveyOption) :
  option ≠ SurveyOption.HealthCodes →
  isSuitableForComprehensiveSurvey option = false :=
sorry

end NUMINAMATH_CALUDE_health_codes_suitable_for_comprehensive_survey_other_options_not_suitable_for_comprehensive_survey_l2047_204780


namespace NUMINAMATH_CALUDE_all_roots_real_l2047_204731

/-- The polynomial x^4 - 4x^3 + 6x^2 - 4x + 1 -/
def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x + 1

/-- Theorem stating that all roots of the polynomial are real -/
theorem all_roots_real : ∀ x : ℂ, p x.re = 0 → x.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_roots_real_l2047_204731


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l2047_204715

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the condition for points on the intersection line
def on_intersection_line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x - 2

-- Define the relation between points O, M, N, and D
def point_relation (xm ym xn yn xd yd t : ℝ) : Prop :=
  xm + xn = t * xd ∧ ym + yn = t * yd

theorem hyperbola_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_real_axis : 2 * a = 4 * Real.sqrt 3)
  (h_focus_asymptote : b * Real.sqrt (b^2 + a^2) / Real.sqrt (b^2 + a^2) = Real.sqrt 3) :
  (∃ (xm ym xn yn xd yd t : ℝ),
    hyperbola a b xm ym ∧
    hyperbola a b xn yn ∧
    hyperbola a b xd yd ∧
    on_intersection_line xm ym ∧
    on_intersection_line xn yn ∧
    point_relation xm ym xn yn xd yd t ∧
    a^2 = 12 ∧
    b^2 = 3 ∧
    t = 4 ∧
    xd = 4 * Real.sqrt 3 ∧
    yd = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l2047_204715


namespace NUMINAMATH_CALUDE_first_customer_payment_l2047_204776

/-- The cost of one MP3 player -/
def mp3_cost : ℕ := sorry

/-- The cost of one set of headphones -/
def headphone_cost : ℕ := 30

/-- The total cost for the second customer -/
def second_customer_total : ℕ := 480

/-- The number of MP3 players bought by the first customer -/
def first_customer_mp3 : ℕ := 5

/-- The number of headphones bought by the first customer -/
def first_customer_headphones : ℕ := 8

/-- The number of MP3 players bought by the second customer -/
def second_customer_mp3 : ℕ := 3

/-- The number of headphones bought by the second customer -/
def second_customer_headphones : ℕ := 4

theorem first_customer_payment :
  second_customer_mp3 * mp3_cost + second_customer_headphones * headphone_cost = second_customer_total →
  first_customer_mp3 * mp3_cost + first_customer_headphones * headphone_cost = 840 :=
by sorry

end NUMINAMATH_CALUDE_first_customer_payment_l2047_204776


namespace NUMINAMATH_CALUDE_students_not_in_biology_l2047_204775

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 840)
  (h2 : biology_percentage = 35 / 100) :
  total_students - (total_students * biology_percentage).floor = 546 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l2047_204775


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2047_204753

open Set

universe u

def U : Set (Fin 6) := {1,2,3,4,5,6}
def A : Set (Fin 6) := {2,4,6}
def B : Set (Fin 6) := {1,2,3,5}

theorem intersection_with_complement : A ∩ (U \ B) = {4,6} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2047_204753


namespace NUMINAMATH_CALUDE_trapezoid_median_l2047_204781

/-- Given a triangle and a trapezoid with the same altitude, prove that the median of the trapezoid is 24 inches -/
theorem trapezoid_median (h : ℝ) : 
  let triangle_base : ℝ := 24
  let trapezoid_base1 : ℝ := 12
  let trapezoid_base2 : ℝ := 36
  let triangle_area : ℝ := (1/2) * triangle_base * h
  let trapezoid_area : ℝ := (1/2) * (trapezoid_base1 + trapezoid_base2) * h
  let trapezoid_median : ℝ := (1/2) * (trapezoid_base1 + trapezoid_base2)
  triangle_area = trapezoid_area → trapezoid_median = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_l2047_204781


namespace NUMINAMATH_CALUDE_chords_for_full_rotation_l2047_204790

/-- The number of chords needed to complete a full rotation when drawing chords on a larger circle
    tangent to a smaller concentric circle, given that the angle between consecutive chords is 60°. -/
def numChords : ℕ := 3

theorem chords_for_full_rotation (angle : ℝ) (h : angle = 60) :
  (numChords : ℝ) * angle = 360 := by
  sorry

end NUMINAMATH_CALUDE_chords_for_full_rotation_l2047_204790


namespace NUMINAMATH_CALUDE_apples_given_to_neighbor_l2047_204727

theorem apples_given_to_neighbor (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : remaining_apples = 39) :
  initial_apples - remaining_apples = 88 := by
sorry

end NUMINAMATH_CALUDE_apples_given_to_neighbor_l2047_204727


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2047_204786

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, mx - n > 0 ↔ x < 1/3) →
  (∀ x, (m + n) * x < n - m ↔ x > -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2047_204786


namespace NUMINAMATH_CALUDE_max_pots_is_ten_l2047_204709

/-- Represents the number of items Susan can buy -/
structure Purchase where
  pins : ℕ
  pans : ℕ
  pots : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  3 * p.pins + 4 * p.pans + 9 * p.pots

/-- Checks if a purchase is valid according to the problem constraints -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pins ≥ 1 ∧ p.pans ≥ 1 ∧ p.pots ≥ 1 ∧ totalCost p = 100

/-- Theorem stating that the maximum number of pots Susan can buy is 10 -/
theorem max_pots_is_ten :
  ∀ p : Purchase, isValidPurchase p → p.pots ≤ 10 ∧ 
  ∃ q : Purchase, isValidPurchase q ∧ q.pots = 10 :=
sorry

end NUMINAMATH_CALUDE_max_pots_is_ten_l2047_204709


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l2047_204757

theorem greatest_integer_fraction (x : ℤ) : 
  (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l2047_204757


namespace NUMINAMATH_CALUDE_segment_rectangle_configurations_l2047_204785

/-- Represents a rectangle made of segments -/
structure SegmentRectangle where
  m : ℕ  -- number of segments on one side
  n : ℕ  -- number of segments on the other side

/-- The total number of segments in a rectangle -/
def total_segments (rect : SegmentRectangle) : ℕ :=
  rect.m * (rect.n + 1) + rect.n * (rect.m + 1)

/-- Possible configurations of a rectangle with 1997 segments -/
def is_valid_configuration (rect : SegmentRectangle) : Prop :=
  total_segments rect = 1997 ∧
  (rect.m = 2 ∧ rect.n = 399) ∨
  (rect.m = 8 ∧ rect.n = 117) ∨
  (rect.m = 23 ∧ rect.n = 42)

/-- Main theorem: The only valid configurations are 2×399, 8×117, and 23×42 -/
theorem segment_rectangle_configurations :
  ∀ rect : SegmentRectangle, total_segments rect = 1997 → is_valid_configuration rect :=
by sorry

end NUMINAMATH_CALUDE_segment_rectangle_configurations_l2047_204785


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2047_204741

/-- The volume of a sphere inscribed in a cube with side length 8 inches -/
theorem volume_of_inscribed_sphere (π : ℝ) : ℝ := by
  -- Define the side length of the cube
  let cube_side : ℝ := 8

  -- Define the radius of the inscribed sphere
  let sphere_radius : ℝ := cube_side / 2

  -- Define the volume of the sphere
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3

  -- Prove that the volume equals (256/3)π cubic inches
  have : sphere_volume = (256 / 3) * π := by sorry

  -- Return the result
  exact (256 / 3) * π

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l2047_204741


namespace NUMINAMATH_CALUDE_shielas_drawing_distribution_l2047_204748

/-- Represents the number of animal drawings each neighbor receives. -/
def drawings_per_neighbor (total_drawings : ℕ) (num_neighbors : ℕ) : ℕ :=
  total_drawings / num_neighbors

/-- Proves that Shiela's neighbors each receive 9 animal drawings. -/
theorem shielas_drawing_distribution :
  let total_drawings : ℕ := 54
  let num_neighbors : ℕ := 6
  drawings_per_neighbor total_drawings num_neighbors = 9 := by
  sorry

end NUMINAMATH_CALUDE_shielas_drawing_distribution_l2047_204748


namespace NUMINAMATH_CALUDE_notebook_cost_l2047_204787

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 50 ∧
  total_cost = 2739 ∧
  buying_students > total_students / 2 ∧
  notebooks_per_student % 2 = 1 ∧
  notebooks_per_student > 1 ∧
  cost_per_notebook > notebooks_per_student ∧
  buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
  cost_per_notebook = 7 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2047_204787


namespace NUMINAMATH_CALUDE_repair_cost_is_288_l2047_204764

/-- The amount spent on repairs for a scooter, given the purchase price, selling price, and gain percentage. -/
def repair_cost (purchase_price selling_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  selling_price * (1 - gain_percentage / 100) - purchase_price

/-- Theorem stating that the repair cost is $288 given the specific conditions. -/
theorem repair_cost_is_288 :
  repair_cost 900 1320 10 = 288 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_288_l2047_204764


namespace NUMINAMATH_CALUDE_equal_perimeter_parallel_sections_l2047_204759

/-- A tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A plane that intersects a tetrahedron -/
structure IntersectingPlane where
  plane : Plane
  tetrahedron : Tetrahedron

/-- The perimeter of the intersection between a plane and a tetrahedron -/
def intersectionPerimeter (p : IntersectingPlane) : ℝ := sorry

/-- Two edges of a tetrahedron are disjoint -/
def disjointEdges (t : Tetrahedron) (e1 e2 : Segment) : Prop := sorry

/-- A plane is parallel to two edges of a tetrahedron -/
def parallelToEdges (p : IntersectingPlane) (e1 e2 : Segment) : Prop := sorry

/-- The length of a segment -/
def length (s : Segment) : ℝ := sorry

theorem equal_perimeter_parallel_sections (t : Tetrahedron) 
  (e1 e2 : Segment) (p1 p2 : IntersectingPlane) :
  disjointEdges t e1 e2 →
  length e1 = length e2 →
  parallelToEdges p1 e1 e2 →
  parallelToEdges p2 e1 e2 →
  intersectionPerimeter p1 = intersectionPerimeter p2 := by
  sorry

end NUMINAMATH_CALUDE_equal_perimeter_parallel_sections_l2047_204759


namespace NUMINAMATH_CALUDE_complex_fraction_cube_l2047_204744

theorem complex_fraction_cube (i : ℂ) (h : i^2 = -1) :
  ((1 + i) / (1 - i))^3 = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_cube_l2047_204744


namespace NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l2047_204721

/-- The area of a convex cyclic quadrilateral -/
theorem area_cyclic_quadrilateral 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_convex : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) 
  (h_cyclic : ∃ (r : ℝ), r > 0 ∧ 
    a * c = (r + (a^2 / (4*r))) * (r + (c^2 / (4*r))) ∧ 
    b * d = (r + (b^2 / (4*r))) * (r + (d^2 / (4*r)))) :
  let p := (a + b + c + d) / 2
  ∃ (area : ℝ), area = Real.sqrt ((p-a)*(p-b)*(p-c)*(p-d)) := by
  sorry


end NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l2047_204721


namespace NUMINAMATH_CALUDE_pizza_time_is_ten_minutes_l2047_204749

/-- Represents the pizza-making scenario --/
structure PizzaScenario where
  totalTime : ℕ        -- Total time in hours
  initialFlour : ℕ     -- Initial flour in kg
  flourPerPizza : ℚ    -- Flour required per pizza in kg
  remainingPizzas : ℕ  -- Number of pizzas that can be made with remaining flour

/-- Calculates the time taken to make each pizza --/
def timeTakenPerPizza (scenario : PizzaScenario) : ℚ :=
  let totalMinutes := scenario.totalTime * 60
  let usedFlour := scenario.initialFlour - (scenario.remainingPizzas * scenario.flourPerPizza)
  let pizzasMade := usedFlour / scenario.flourPerPizza
  totalMinutes / pizzasMade

/-- Theorem stating that the time taken per pizza is 10 minutes --/
theorem pizza_time_is_ten_minutes (scenario : PizzaScenario) 
    (h1 : scenario.totalTime = 7)
    (h2 : scenario.initialFlour = 22)
    (h3 : scenario.flourPerPizza = 1/2)
    (h4 : scenario.remainingPizzas = 2) :
    timeTakenPerPizza scenario = 10 := by
  sorry


end NUMINAMATH_CALUDE_pizza_time_is_ten_minutes_l2047_204749


namespace NUMINAMATH_CALUDE_billy_is_45_l2047_204774

/-- Billy's age -/
def B : ℕ := sorry

/-- Joe's age -/
def J : ℕ := sorry

/-- Billy's age is three times Joe's age -/
axiom billy_age : B = 3 * J

/-- The sum of their ages is 60 -/
axiom total_age : B + J = 60

/-- Prove that Billy is 45 years old -/
theorem billy_is_45 : B = 45 := by sorry

end NUMINAMATH_CALUDE_billy_is_45_l2047_204774


namespace NUMINAMATH_CALUDE_race_distance_proof_l2047_204750

/-- The distance of a race where:
  * A covers the distance in 36 seconds
  * B covers the distance in 45 seconds
  * A beats B by 22 meters
-/
def race_distance : ℝ := 110

theorem race_distance_proof (A_time B_time : ℝ) (beat_distance : ℝ) 
  (h1 : A_time = 36)
  (h2 : B_time = 45)
  (h3 : beat_distance = 22)
  (h4 : A_time * (race_distance / B_time) + beat_distance = race_distance) :
  race_distance = 110 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_proof_l2047_204750


namespace NUMINAMATH_CALUDE_chess_game_probability_l2047_204796

theorem chess_game_probability (prob_draw prob_B_win : ℝ) 
  (h1 : prob_draw = 1/2)
  (h2 : prob_B_win = 1/3) :
  prob_draw + prob_B_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2047_204796


namespace NUMINAMATH_CALUDE_largest_x_value_largest_x_exists_l2047_204788

theorem largest_x_value (x y : ℝ) : 
  (|x - 3| = 15 ∧ x + y = 10) → x ≤ 18 := by
  sorry

theorem largest_x_exists : 
  ∃ x y : ℝ, |x - 3| = 15 ∧ x + y = 10 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_largest_x_exists_l2047_204788


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_properties_l2047_204719

theorem geometric_and_arithmetic_properties :
  ∀ (s r h : ℝ) (a b : ℝ) (x : ℝ),
  s > 0 → r > 0 → h > 0 → b ≠ 0 →
  (2 * s)^2 = 4 * s^2 ∧
  (π * r^2 * (2 * h)) = 2 * (π * r^2 * h) ∧
  (2 * s)^3 = 8 * s^3 ∧
  (2 * a) / (b / 2) = 4 * (a / b) ∧
  x + 0 = x :=
by sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_properties_l2047_204719


namespace NUMINAMATH_CALUDE_pizza_slices_l2047_204745

theorem pizza_slices (total_slices : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) : 
  total_slices = 16 → 
  num_pizzas = 2 → 
  total_slices = num_pizzas * slices_per_pizza → 
  slices_per_pizza = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l2047_204745


namespace NUMINAMATH_CALUDE_max_utilization_rate_square_plate_l2047_204773

/-- Given a square steel plate with side length 4 and a rusted corner defined by AF = 2 and BF = 1,
    prove that the maximum utilization rate is 50%. -/
theorem max_utilization_rate_square_plate (side_length : ℝ) (af bf : ℝ) :
  side_length = 4 ∧ af = 2 ∧ bf = 1 →
  ∃ (rect_area : ℝ),
    rect_area ≤ side_length * side_length ∧
    rect_area = side_length * (side_length - af) ∧
    (rect_area / (side_length * side_length)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_utilization_rate_square_plate_l2047_204773


namespace NUMINAMATH_CALUDE_jeremy_oranges_l2047_204710

theorem jeremy_oranges (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) : 
  tuesday = 3 * monday →
  wednesday = 70 →
  monday + tuesday + wednesday = 470 →
  monday = 100 := by
sorry

end NUMINAMATH_CALUDE_jeremy_oranges_l2047_204710


namespace NUMINAMATH_CALUDE_ab_range_l2047_204722

theorem ab_range (a b q : ℝ) (h1 : (1/3 : ℝ) ≤ q ∧ q ≤ 2) 
  (h2 : ∃ m : ℝ, ∃ r1 r2 r3 r4 : ℝ, 
    (r1^2 - a*r1 + 1)*(r1^2 - b*r1 + 1) = 0 ∧
    (r2^2 - a*r2 + 1)*(r2^2 - b*r2 + 1) = 0 ∧
    (r3^2 - a*r3 + 1)*(r3^2 - b*r3 + 1) = 0 ∧
    (r4^2 - a*r4 + 1)*(r4^2 - b*r4 + 1) = 0 ∧
    r1 = m ∧ r2 = m*q ∧ r3 = m*q^2 ∧ r4 = m*q^3) :
  4 ≤ a*b ∧ a*b ≤ 112/9 := by sorry

end NUMINAMATH_CALUDE_ab_range_l2047_204722


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2047_204765

/-- Given that sin(α) / (sin(α) - cos(α)) = -1, prove:
    1. tan(α) = 1/2
    2. (sin²(α) + 2sin(α)cos(α)) / (3sin²(α) + cos²(α)) = 5/7 -/
theorem trigonometric_identity (α : ℝ) 
    (h : Real.sin α / (Real.sin α - Real.cos α) = -1) : 
    Real.tan α = 1/2 ∧ 
    (Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / 
    (3 * Real.sin α ^ 2 + Real.cos α ^ 2) = 5/7 := by
  sorry


end NUMINAMATH_CALUDE_trigonometric_identity_l2047_204765


namespace NUMINAMATH_CALUDE_platform_and_train_length_l2047_204716

/-- The combined length of a platform and a train, given the speeds and passing times of two trains. -/
theorem platform_and_train_length
  (t1_platform_time : ℝ)
  (t1_man_time : ℝ)
  (t1_speed : ℝ)
  (t2_speed : ℝ)
  (t2_man_time : ℝ)
  (h1 : t1_platform_time = 16)
  (h2 : t1_man_time = 10)
  (h3 : t1_speed = 54 * 1000 / 3600)
  (h4 : t2_speed = 72 * 1000 / 3600)
  (h5 : t2_man_time = 12) :
  t1_speed * (t1_platform_time - t1_man_time) + t2_speed * t2_man_time = 330 :=
by sorry

end NUMINAMATH_CALUDE_platform_and_train_length_l2047_204716


namespace NUMINAMATH_CALUDE_no_real_roots_l2047_204706

theorem no_real_roots : ¬∃ x : ℝ, x + 2 * Real.sqrt (x - 5) = 6 := by sorry

end NUMINAMATH_CALUDE_no_real_roots_l2047_204706


namespace NUMINAMATH_CALUDE_total_capacity_l2047_204793

/-- Represents the capacity of boats -/
structure BoatCapacity where
  large : ℕ
  small : ℕ

/-- The capacity of different combinations of boats -/
def boat_combinations (c : BoatCapacity) : Prop :=
  c.large + 4 * c.small = 46 ∧ 2 * c.large + 3 * c.small = 57

/-- The theorem to prove -/
theorem total_capacity (c : BoatCapacity) :
  boat_combinations c → 3 * c.large + 6 * c.small = 96 := by
  sorry


end NUMINAMATH_CALUDE_total_capacity_l2047_204793


namespace NUMINAMATH_CALUDE_estimate_sqrt_19_l2047_204755

theorem estimate_sqrt_19 : 6 < 2 + Real.sqrt 19 ∧ 2 + Real.sqrt 19 < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_19_l2047_204755


namespace NUMINAMATH_CALUDE_prob_all_cats_before_lunch_l2047_204713

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of animals -/
def totalAnimals : ℕ := 7

/-- The number of cats -/
def numCats : ℕ := 2

/-- The number of dogs -/
def numDogs : ℕ := 5

/-- The number of animals to be groomed before lunch -/
def numGroomed : ℕ := 4

/-- The probability of grooming all cats before lunch -/
def probAllCats : ℚ := (choose numDogs (numGroomed - numCats)) / (choose totalAnimals numGroomed)

theorem prob_all_cats_before_lunch : probAllCats = 2/7 := by sorry

end NUMINAMATH_CALUDE_prob_all_cats_before_lunch_l2047_204713


namespace NUMINAMATH_CALUDE_binomial_coefficient_not_always_divisible_l2047_204754

theorem binomial_coefficient_not_always_divisible :
  ∃ k : ℕ+, ∀ n : ℕ, n > 1 → ∃ i : ℕ, 1 ≤ i ∧ i ≤ n - 1 ∧ ¬(k : ℕ) ∣ Nat.choose n i := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_not_always_divisible_l2047_204754


namespace NUMINAMATH_CALUDE_triangle_property_l2047_204752

open Real

theorem triangle_property (A B C : ℝ) (R : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  cos (2 * A) - 3 * cos (B + C) - 1 = 0 ∧
  R = 1 →
  A = π / 3 ∧ 
  (∃ (S : ℝ), S ≤ 3 * sqrt 3 / 4 ∧ 
    ∀ (S' : ℝ), (∃ (a b c : ℝ), 
      a = 2 * R * sin A ∧
      b = 2 * R * sin B ∧
      c = 2 * R * sin C ∧
      S' = 1 / 2 * a * b * sin C) → 
    S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2047_204752


namespace NUMINAMATH_CALUDE_infinite_primes_l2047_204770

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l2047_204770


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l2047_204762

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (∀ x : ℤ, 3 * |x| - 4 < 20 → y ≤ x) → y = -7 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l2047_204762


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_not_equal_not_vertical_l2047_204728

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the property of being vertical angles
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define the property of angles being equal
def are_equal (a b : Angle) : Prop := a = b

-- Theorem 1: If two angles are vertical angles, then they are equal
theorem vertical_angles_are_equal (a b : Angle) :
  are_vertical_angles a b → are_equal a b := by sorry

-- Theorem 2: If two angles are not equal, then they are not vertical angles
theorem not_equal_not_vertical (a b : Angle) :
  ¬(are_equal a b) → ¬(are_vertical_angles a b) := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_not_equal_not_vertical_l2047_204728


namespace NUMINAMATH_CALUDE_translated_function_and_triangle_area_l2047_204795

/-- A linear function f(x) = 3x + b passing through (1, 4) -/
def f (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

theorem translated_function_and_triangle_area (b : ℝ) :
  f b 1 = 4 →
  b = 1 ∧
  (1 / 2 : ℝ) * (1 / 3) * 1 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_translated_function_and_triangle_area_l2047_204795


namespace NUMINAMATH_CALUDE_soccer_game_time_proof_l2047_204737

/-- Calculates the total time in minutes for a soccer game and post-game ceremony -/
def total_time (game_hours : ℕ) (game_minutes : ℕ) (ceremony_minutes : ℕ) : ℕ :=
  game_hours * 60 + game_minutes + ceremony_minutes

/-- Proves that the total time for a 2 hour 35 minute game and 25 minute ceremony is 180 minutes -/
theorem soccer_game_time_proof :
  total_time 2 35 25 = 180 := by
  sorry

end NUMINAMATH_CALUDE_soccer_game_time_proof_l2047_204737


namespace NUMINAMATH_CALUDE_reflection_matrix_squared_is_identity_l2047_204784

/-- Reflection matrix over a non-zero vector -/
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- Theorem: The square of a reflection matrix is the identity matrix -/
theorem reflection_matrix_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = !![1, 0; 0, 1] :=
sorry

end NUMINAMATH_CALUDE_reflection_matrix_squared_is_identity_l2047_204784


namespace NUMINAMATH_CALUDE_starWars_earnings_l2047_204743

/-- Represents movie financial data in millions of dollars -/
structure MovieData where
  cost : ℝ
  boxOffice : ℝ
  profit : ℝ

/-- The Lion King's financial data -/
def lionKing : MovieData := {
  cost := 10,
  boxOffice := 200,
  profit := 200 - 10
}

/-- Star Wars' financial data -/
def starWars : MovieData := {
  cost := 25,
  profit := 2 * lionKing.profit,
  boxOffice := 25 + 2 * lionKing.profit
}

/-- Theorem stating that Star Wars earned 405 million at the box office -/
theorem starWars_earnings : starWars.boxOffice = 405 := by
  sorry

#eval starWars.boxOffice

end NUMINAMATH_CALUDE_starWars_earnings_l2047_204743


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2047_204758

open Set

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_deriv : ∀ x, f x > deriv f x) (h_init : f 0 = 2) :
  {x : ℝ | f x < 2 * Real.exp x} = {x : ℝ | x > 0} := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2047_204758


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2047_204725

theorem intersecting_squares_area_difference : 
  let s1 : ℕ := 12
  let s2 : ℕ := 9
  let s3 : ℕ := 7
  let s4 : ℕ := 3
  s1^2 + s3^2 - (s2^2 + s4^2) = 103 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2047_204725


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l2047_204700

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l2047_204700


namespace NUMINAMATH_CALUDE_kamal_age_double_son_l2047_204791

/-- The number of years after which Kamal will be twice as old as his son -/
def years_until_double_age (kamal_age : ℕ) (son_age : ℕ) : ℕ :=
  kamal_age + 8 - 2 * (son_age + 8)

/-- Kamal's current age -/
def kamal_current_age : ℕ := 40

theorem kamal_age_double_son :
  years_until_double_age kamal_current_age
    ((kamal_current_age - 8) / 4 + 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_kamal_age_double_son_l2047_204791


namespace NUMINAMATH_CALUDE_total_run_time_l2047_204760

def emma_time : ℝ := 20

theorem total_run_time (fernando_time : ℝ) 
  (h1 : fernando_time = 2 * emma_time) : 
  emma_time + fernando_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_run_time_l2047_204760
