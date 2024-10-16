import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_cube_root_system_l210_21061

theorem unique_solution_cube_root_system :
  ∃! (x y z : ℝ),
    Real.sqrt (x^3 - y) = z - 1 ∧
    Real.sqrt (y^3 - z) = x - 1 ∧
    Real.sqrt (z^3 - x) = y - 1 ∧
    x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_system_l210_21061


namespace NUMINAMATH_CALUDE_max_tau_minus_n_max_tau_minus_n_achievable_l210_21027

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The theorem states that 4τ(n) - n is at most 12 for all positive integers n -/
theorem max_tau_minus_n (n : ℕ+) : 4 * (tau n) - n.val ≤ 12 := by sorry

/-- The theorem states that there exists a positive integer n for which 4τ(n) - n equals 12 -/
theorem max_tau_minus_n_achievable : ∃ n : ℕ+, 4 * (tau n) - n.val = 12 := by sorry

end NUMINAMATH_CALUDE_max_tau_minus_n_max_tau_minus_n_achievable_l210_21027


namespace NUMINAMATH_CALUDE_exam_score_difference_l210_21044

def math_exam_problem (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ) : Prop :=
  bryan_score = 20 ∧
  jen_score = bryan_score + 10 ∧
  sammy_score < jen_score ∧
  total_points = 35 ∧
  sammy_mistakes = 7 ∧
  sammy_score = total_points - sammy_mistakes ∧
  jen_score - sammy_score = 2

theorem exam_score_difference :
  ∀ (bryan_score jen_score sammy_score total_points sammy_mistakes : ℕ),
  math_exam_problem bryan_score jen_score sammy_score total_points sammy_mistakes :=
by
  sorry

#check exam_score_difference

end NUMINAMATH_CALUDE_exam_score_difference_l210_21044


namespace NUMINAMATH_CALUDE_vikas_questions_l210_21065

theorem vikas_questions (total : ℕ) (r v a : ℕ) : 
  total = 24 →
  r + v + a = total →
  7 * v = 3 * r →
  3 * a = 2 * v →
  v = 6 := by
sorry

end NUMINAMATH_CALUDE_vikas_questions_l210_21065


namespace NUMINAMATH_CALUDE_digit_matching_equality_l210_21026

theorem digit_matching_equality : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a ≤ 99 ∧ 
  b ≤ 99 ∧ 
  a + b ≤ 9999 ∧ 
  (a + b)^2 = 100 * a + b :=
sorry

end NUMINAMATH_CALUDE_digit_matching_equality_l210_21026


namespace NUMINAMATH_CALUDE_max_distance_is_three_l210_21059

/-- A figure constructed from an equilateral triangle with semicircles on each side -/
structure TriangleWithSemicircles where
  /-- Side length of the equilateral triangle -/
  triangleSide : ℝ
  /-- Radius of the semicircles -/
  semicircleRadius : ℝ

/-- The maximum distance between any two points on the boundary of the figure -/
def maxBoundaryDistance (figure : TriangleWithSemicircles) : ℝ :=
  figure.triangleSide + 2 * figure.semicircleRadius

/-- Theorem stating the maximum distance for the specific figure described in the problem -/
theorem max_distance_is_three :
  let figure : TriangleWithSemicircles := ⟨2, 1⟩
  maxBoundaryDistance figure = 3 := by
  sorry


end NUMINAMATH_CALUDE_max_distance_is_three_l210_21059


namespace NUMINAMATH_CALUDE_negation_equivalence_l210_21046

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2016 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l210_21046


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l210_21030

theorem reciprocal_equals_self (x : ℚ) : x = x⁻¹ ↔ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l210_21030


namespace NUMINAMATH_CALUDE_same_solution_implies_m_value_l210_21008

theorem same_solution_implies_m_value : ∀ m x : ℚ,
  (8 - m = 2 * (x + 1) ∧ 2 * (2 * x - 3) - 1 = 1 - 2 * x) →
  m = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_value_l210_21008


namespace NUMINAMATH_CALUDE_number_of_male_students_l210_21011

theorem number_of_male_students 
  (total_candidates : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (selected_male : ℕ)
  (selected_female : ℕ)
  (num_camps : ℕ)
  (total_schemes : ℕ) :
  total_candidates = 10 →
  male_students + female_students = total_candidates →
  male_students > female_students →
  selected_male = 2 →
  selected_female = 2 →
  num_camps = 3 →
  total_schemes = 3240 →
  (male_students.choose selected_male * female_students.choose selected_female * 
   (selected_male + selected_female).choose num_camps * num_camps.factorial = total_schemes) →
  male_students = 6 := by
sorry

end NUMINAMATH_CALUDE_number_of_male_students_l210_21011


namespace NUMINAMATH_CALUDE_frequency_limit_theorem_l210_21028

/-- A fair coin toss experiment -/
structure CoinToss where
  /-- The number of tosses -/
  n : ℕ
  /-- The number of heads -/
  heads : ℕ
  /-- The number of heads is less than or equal to the number of tosses -/
  heads_le_n : heads ≤ n

/-- The frequency of heads in a coin toss experiment -/
def frequency (ct : CoinToss) : ℚ :=
  ct.heads / ct.n

/-- The limit of the frequency of heads as the number of tosses approaches infinity -/
theorem frequency_limit_theorem :
  ∀ ε > 0, ∃ N : ℕ, ∀ ct : CoinToss, ct.n ≥ N → |frequency ct - 1/2| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_limit_theorem_l210_21028


namespace NUMINAMATH_CALUDE_triangle_vector_ratio_l210_21015

/-- Given a triangle ABC with point E, prove that if AE = 3/4 * AB + 1/4 * AC, 
    then BE = 1/3 * EC -/
theorem triangle_vector_ratio (A B C E : ℝ × ℝ) : 
  (E - A) = 3/4 * (B - A) + 1/4 * (C - A) → 
  (E - B) = 1/3 * (C - E) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_ratio_l210_21015


namespace NUMINAMATH_CALUDE_third_derivative_x5_minus_7x3_plus_2_l210_21013

/-- The third derivative of x^5 - 7x^3 + 2 is 60x^2 - 42 -/
theorem third_derivative_x5_minus_7x3_plus_2 (x : ℝ) :
  (deriv^[3] (fun x => x^5 - 7*x^3 + 2)) x = 60*x^2 - 42 := by
  sorry

end NUMINAMATH_CALUDE_third_derivative_x5_minus_7x3_plus_2_l210_21013


namespace NUMINAMATH_CALUDE_enclosed_area_is_five_twelfths_l210_21068

noncomputable def f (x : ℝ) : ℝ := x^(1/2)
noncomputable def g (x : ℝ) : ℝ := x^3

theorem enclosed_area_is_five_twelfths :
  ∫ x in (0)..(1), (f x - g x) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_is_five_twelfths_l210_21068


namespace NUMINAMATH_CALUDE_percentage_of_l210_21038

theorem percentage_of (a b : ℝ) (h : b ≠ 0) :
  (a / b) * 100 = 250 → a = 150 ∧ b = 60 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_l210_21038


namespace NUMINAMATH_CALUDE_m_range_l210_21010

def p (m : ℝ) : Prop := ∀ x, x^2 - 2*m*x + 1 ≥ 0

def q (m : ℝ) : Prop := m * (m - 2) < 0

def range_m (m : ℝ) : Prop := (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2)

theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m → q m) → range_m m :=
sorry

end NUMINAMATH_CALUDE_m_range_l210_21010


namespace NUMINAMATH_CALUDE_tan_3_75_deg_sum_l210_21037

theorem tan_3_75_deg_sum (a b c d : ℕ+) 
  (h1 : Real.tan (3.75 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - d)
  (h2 : a ≥ b) (h3 : b ≥ c) (h4 : c ≥ d) :
  a + b + c + d = 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_75_deg_sum_l210_21037


namespace NUMINAMATH_CALUDE_pond_algae_free_day_24_l210_21039

/-- Represents the coverage of algae in the pond on a given day -/
def algae_coverage (day : ℕ) : ℝ := sorry

/-- The algae coverage triples every two days -/
axiom triple_every_two_days (d : ℕ) : algae_coverage (d + 2) = 3 * algae_coverage d

/-- The pond is completely covered on day 28 -/
axiom full_coverage_day_28 : algae_coverage 28 = 1

/-- Theorem: The pond is 88.89% algae-free on day 24 -/
theorem pond_algae_free_day_24 : algae_coverage 24 = 1 - 0.8889 := by sorry

end NUMINAMATH_CALUDE_pond_algae_free_day_24_l210_21039


namespace NUMINAMATH_CALUDE_sin_equality_proof_l210_21007

theorem sin_equality_proof (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 → Real.sin (n * π / 180) = Real.sin (721 * π / 180) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l210_21007


namespace NUMINAMATH_CALUDE_double_average_l210_21075

theorem double_average (n : ℕ) (original_avg : ℚ) (new_avg : ℚ) : 
  n = 11 → original_avg = 36 → new_avg = 2 * original_avg → new_avg = 72 := by
  sorry

end NUMINAMATH_CALUDE_double_average_l210_21075


namespace NUMINAMATH_CALUDE_candy_count_l210_21032

theorem candy_count : ∃ n : ℕ, 
  (∃ x : ℕ, x > 10 ∧ n = 3 * (x - 1) + 2) ∧ 
  (∃ y : ℕ, y < 10 ∧ n = 4 * (y - 1) + 3) ∧ 
  n = 35 := by
sorry

end NUMINAMATH_CALUDE_candy_count_l210_21032


namespace NUMINAMATH_CALUDE_arccos_sin_eight_equals_pi_half_minus_1_72_l210_21023

theorem arccos_sin_eight_equals_pi_half_minus_1_72 :
  Real.arccos (Real.sin 8) = π / 2 - 1.72 := by sorry

end NUMINAMATH_CALUDE_arccos_sin_eight_equals_pi_half_minus_1_72_l210_21023


namespace NUMINAMATH_CALUDE_max_cookies_without_ingredients_l210_21025

theorem max_cookies_without_ingredients 
  (total_cookies : ℕ) 
  (peanut_cookies : ℕ) 
  (choc_chip_cookies : ℕ) 
  (raisin_cookies : ℕ) 
  (oat_cookies : ℕ) 
  (h1 : total_cookies = 36)
  (h2 : peanut_cookies ≥ 24)
  (h3 : choc_chip_cookies ≥ 12)
  (h4 : raisin_cookies ≥ 9)
  (h5 : oat_cookies ≥ 4) :
  total_cookies - max peanut_cookies (max choc_chip_cookies (max raisin_cookies oat_cookies)) ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_without_ingredients_l210_21025


namespace NUMINAMATH_CALUDE_no_multiple_with_smaller_digit_sum_l210_21090

/-- The number composed of m digits all being ones -/
def ones_number (m : ℕ) : ℕ :=
  (10^m - 1) / 9

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number composed of m ones has no multiple with digit sum less than m -/
theorem no_multiple_with_smaller_digit_sum (m : ℕ) :
  ∀ k : ℕ, k > 0 → digit_sum (k * ones_number m) ≥ m :=
sorry

end NUMINAMATH_CALUDE_no_multiple_with_smaller_digit_sum_l210_21090


namespace NUMINAMATH_CALUDE_divisible_by_three_l210_21089

theorem divisible_by_three (n : ℕ) : 3 ∣ (5^n - 2^n) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l210_21089


namespace NUMINAMATH_CALUDE_storm_rainfall_theorem_l210_21012

/-- Represents the rainfall data for a city over three days -/
structure CityRainfall where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ

/-- Represents the rainfall data for two cities -/
structure StormData where
  cityA : CityRainfall
  cityB : CityRainfall
  X : ℝ  -- Combined rainfall on day 3
  Y : ℝ  -- Total rainfall over three days

/-- Defines the conditions of the storm and proves the results -/
theorem storm_rainfall_theorem (s : StormData) : 
  s.cityA.day1 = 4 ∧ 
  s.cityA.day2 = 5 * s.cityA.day1 ∧
  s.cityB.day2 = 3 * s.cityA.day1 ∧
  s.cityA.day3 = (s.cityA.day1 + s.cityA.day2) / 2 ∧
  s.cityB.day3 = s.cityB.day1 + s.cityB.day2 - 6 ∧
  s.X = s.cityA.day3 + s.cityB.day3 ∧
  s.Y = s.cityA.day1 + s.cityA.day2 + s.cityA.day3 + s.cityB.day1 + s.cityB.day2 + s.cityB.day3 →
  s.cityA.day3 = 12 ∧
  s.cityB.day3 = s.cityB.day1 + 6 ∧
  s.X = 18 + s.cityB.day1 ∧
  s.Y = 54 + 2 * s.cityB.day1 := by
  sorry


end NUMINAMATH_CALUDE_storm_rainfall_theorem_l210_21012


namespace NUMINAMATH_CALUDE_cube_rotation_invariance_l210_21014

-- Define a cube
structure Cube where
  position : ℕ × ℕ  -- Position on the plane
  topFace : Fin 6   -- Top face (numbered 1 to 6)
  rotation : Fin 4  -- Rotation of top face (0, 90, 180, or 270 degrees)

-- Define a roll operation
def roll (c : Cube) : Cube :=
  sorry

-- Define a sequence of rolls
def rollSequence (c : Cube) (n : ℕ) : Cube :=
  sorry

-- Theorem statement
theorem cube_rotation_invariance (c : Cube) (n : ℕ) :
  let c' := rollSequence c n
  c'.position = c.position ∧ c'.topFace = c.topFace →
  c'.rotation = c.rotation :=
sorry

end NUMINAMATH_CALUDE_cube_rotation_invariance_l210_21014


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l210_21063

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : y - x = -1) 
  (h2 : x * y = 2) : 
  -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3 = -4 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l210_21063


namespace NUMINAMATH_CALUDE_smallest_number_between_10_and_11_l210_21021

theorem smallest_number_between_10_and_11 (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (y_eq : y = 3 * x + 10)
  (z_eq : z = x^2 - 5) :
  ∃ w, w = min x (min y z) ∧ 10 < w ∧ w < 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_between_10_and_11_l210_21021


namespace NUMINAMATH_CALUDE_vector_equation_l210_21047

variable {V : Type*} [AddCommGroup V]

theorem vector_equation (A B C : V) : (C - A) - (C - B) = B - A := by sorry

end NUMINAMATH_CALUDE_vector_equation_l210_21047


namespace NUMINAMATH_CALUDE_third_row_sum_l210_21094

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → grid i j ≠ grid i' j'

theorem third_row_sum (grid : Matrix (Fin 3) (Fin 3) ℕ) 
  (h_valid : is_valid_grid grid)
  (h_row1 : (grid 0 0) * (grid 0 1) * (grid 0 2) = 60)
  (h_row2 : (grid 1 0) * (grid 1 1) * (grid 1 2) = 96) :
  (grid 2 0) + (grid 2 1) + (grid 2 2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_third_row_sum_l210_21094


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l210_21043

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l210_21043


namespace NUMINAMATH_CALUDE_min_a_value_l210_21019

/-- The minimum value of a that satisfies the given inequality for all positive x -/
theorem min_a_value (a : ℝ) : 
  (∀ x > 0, Real.log (2 * x) - (a * Real.exp x) / 2 ≤ Real.log a) → 
  a ≥ 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l210_21019


namespace NUMINAMATH_CALUDE_final_balance_approx_l210_21084

/-- Calculates the final amount in Steve's bank account after 5 years --/
def bank_account_balance : ℝ := 
  let initial_deposit : ℝ := 100
  let interest_rate_1 : ℝ := 0.1  -- 10% for first 3 years
  let interest_rate_2 : ℝ := 0.08 -- 8% for next 2 years
  let deposit_1 : ℝ := 10 -- annual deposit for first 2 years
  let deposit_2 : ℝ := 15 -- annual deposit for remaining 3 years
  let year_1 : ℝ := initial_deposit * (1 + interest_rate_1) + deposit_1
  let year_2 : ℝ := year_1 * (1 + interest_rate_1) + deposit_1
  let year_3 : ℝ := year_2 * (1 + interest_rate_1) + deposit_2
  let year_4 : ℝ := year_3 * (1 + interest_rate_2) + deposit_2
  let year_5 : ℝ := year_4 * (1 + interest_rate_2) + deposit_2
  year_5

/-- The final balance in Steve's bank account after 5 years is approximately $230.89 --/
theorem final_balance_approx : 
  ∃ ε > 0, |bank_account_balance - 230.89| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_final_balance_approx_l210_21084


namespace NUMINAMATH_CALUDE_equation_solution_l210_21082

theorem equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l210_21082


namespace NUMINAMATH_CALUDE_min_value_xy_minus_2x_l210_21005

theorem min_value_xy_minus_2x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log x + y * Real.log y = Real.exp x) :
  ∃ (m : ℝ), m = 2 - 2 * Real.log 2 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 
  y' * Real.log x' + y' * Real.log y' = Real.exp x' → x' * y' - 2 * x' ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_minus_2x_l210_21005


namespace NUMINAMATH_CALUDE_temperature_difference_l210_21060

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 2) 
  (h2 : lowest = -8) : 
  highest - lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l210_21060


namespace NUMINAMATH_CALUDE_min_value_f_l210_21087

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_period (x : ℝ) : f (x + 2) = 3 * f x

axiom f_def (x : ℝ) (h : x ∈ Set.Icc 0 2) : f x = x^2 - 2*x

-- Define the theorem
theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4) (-2) ∧
  f x = -1/9 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-4) (-2) → f y ≥ -1/9 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_l210_21087


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l210_21085

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x y z : ℤ), 72 * x + 54 * y + 36 * z > 0 → 72 * x + 54 * y + 36 * z ≥ n) ∧
  (∃ (x y z : ℤ), 72 * x + 54 * y + 36 * z = n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l210_21085


namespace NUMINAMATH_CALUDE_negation_of_forall_nonnegative_square_l210_21066

theorem negation_of_forall_nonnegative_square (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 ≥ 0) → (¬p ↔ ∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_nonnegative_square_l210_21066


namespace NUMINAMATH_CALUDE_total_groom_time_l210_21099

def poodle_groom_time : ℕ := 30
def terrier_groom_time : ℕ := poodle_groom_time / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

theorem total_groom_time :
  num_poodles * poodle_groom_time + num_terriers * terrier_groom_time = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_l210_21099


namespace NUMINAMATH_CALUDE_right_triangle_max_sum_l210_21098

theorem right_triangle_max_sum (a b c : ℝ) : 
  c = 5 →
  a ≤ 3 →
  b ≥ 3 →
  a^2 + b^2 = c^2 →
  a + b ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_sum_l210_21098


namespace NUMINAMATH_CALUDE_auditorium_seats_cost_l210_21053

theorem auditorium_seats_cost 
  (rows : ℕ) 
  (seats_per_row : ℕ) 
  (cost_per_seat : ℕ) 
  (discount_rate : ℚ) 
  (seats_per_discount_group : ℕ) : 
  rows = 5 → 
  seats_per_row = 8 → 
  cost_per_seat = 30 → 
  discount_rate = 1/10 → 
  seats_per_discount_group = 10 → 
  (rows * seats_per_row * cost_per_seat : ℚ) - 
    ((rows * seats_per_row / seats_per_discount_group : ℚ) * 
     (seats_per_discount_group * cost_per_seat * discount_rate)) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_auditorium_seats_cost_l210_21053


namespace NUMINAMATH_CALUDE_tree_height_l210_21051

/-- The height of a tree given specific conditions involving a rope and a person. -/
theorem tree_height (rope_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ) 
  (h1 : rope_ground_distance = 4)
  (h2 : person_distance = 3)
  (h3 : person_height = 1.6)
  (h4 : person_distance < rope_ground_distance) : 
  ∃ (tree_height : ℝ), tree_height = 6.4 := by
  sorry


end NUMINAMATH_CALUDE_tree_height_l210_21051


namespace NUMINAMATH_CALUDE_three_prime_divisors_special_form_l210_21079

theorem three_prime_divisors_special_form (n : ℕ) (x : ℕ) : 
  x = 2^n - 32 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
sorry

end NUMINAMATH_CALUDE_three_prime_divisors_special_form_l210_21079


namespace NUMINAMATH_CALUDE_tournament_balls_count_l210_21064

def tournament_rounds : ℕ := 7

def games_per_round : List ℕ := [64, 32, 16, 8, 4, 2, 1]

def cans_per_game : ℕ := 6

def balls_per_can : ℕ := 4

def total_balls : ℕ := (games_per_round.sum * cans_per_game * balls_per_can)

theorem tournament_balls_count :
  total_balls = 3048 :=
by sorry

end NUMINAMATH_CALUDE_tournament_balls_count_l210_21064


namespace NUMINAMATH_CALUDE_max_cookie_price_l210_21009

theorem max_cookie_price (cookie_price bun_price : ℕ) : 
  (cookie_price > 0) →
  (bun_price > 0) →
  (8 * cookie_price + 3 * bun_price < 200) →
  (4 * cookie_price + 5 * bun_price > 150) →
  cookie_price ≤ 19 ∧ 
  ∃ (max_price : ℕ), max_price = 19 ∧
    ∃ (bun_price_19 : ℕ), 
      (8 * max_price + 3 * bun_price_19 < 200) ∧
      (4 * max_price + 5 * bun_price_19 > 150) := by
  sorry

end NUMINAMATH_CALUDE_max_cookie_price_l210_21009


namespace NUMINAMATH_CALUDE_find_b_l210_21070

theorem find_b (a b c : ℤ) (eq1 : a + 5 = b) (eq2 : 5 + b = c) (eq3 : b + c = a) : b = -10 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l210_21070


namespace NUMINAMATH_CALUDE_extremum_of_f_l210_21072

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_of_f (a b : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x=2
  (f' a b 1 = -3) →  -- Tangent line at x=1 parallel to y=-3x-2
  (∃ x, f a b x = -4 ∧ ∀ y, f a b y ≥ f a b x) :=
by
  sorry

end NUMINAMATH_CALUDE_extremum_of_f_l210_21072


namespace NUMINAMATH_CALUDE_lucy_max_notebooks_l210_21091

/-- The amount of money Lucy has in cents -/
def lucys_money : ℕ := 2550

/-- The cost of each notebook in cents -/
def notebook_cost : ℕ := 240

/-- The maximum number of notebooks Lucy can buy -/
def max_notebooks : ℕ := lucys_money / notebook_cost

theorem lucy_max_notebooks :
  max_notebooks = 10 ∧
  max_notebooks * notebook_cost ≤ lucys_money ∧
  (max_notebooks + 1) * notebook_cost > lucys_money :=
by sorry

end NUMINAMATH_CALUDE_lucy_max_notebooks_l210_21091


namespace NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l210_21006

theorem sqrt_5_greater_than_2 : Real.sqrt 5 > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l210_21006


namespace NUMINAMATH_CALUDE_mean_steps_per_day_l210_21022

theorem mean_steps_per_day (total_steps : ℕ) (num_days : ℕ) (h1 : total_steps = 243000) (h2 : num_days = 30) :
  total_steps / num_days = 8100 := by
  sorry

end NUMINAMATH_CALUDE_mean_steps_per_day_l210_21022


namespace NUMINAMATH_CALUDE_mikeys_leaves_theorem_l210_21067

/-- The number of leaves that blew away given initial and remaining leaf counts -/
def leaves_blown_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that 244 leaves blew away given the initial and remaining counts -/
theorem mikeys_leaves_theorem (initial : ℕ) (remaining : ℕ)
  (h1 : initial = 356)
  (h2 : remaining = 112) :
  leaves_blown_away initial remaining = 244 := by
sorry

end NUMINAMATH_CALUDE_mikeys_leaves_theorem_l210_21067


namespace NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l210_21004

theorem tan_alpha_sqrt_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l210_21004


namespace NUMINAMATH_CALUDE_line_circle_separate_trajectory_of_P_l210_21049

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the line l
def line_l (a b α t : ℝ) (x y : ℝ) : Prop :=
  x = a + t * Real.cos α ∧ y = b + t * Real.sin α

-- Part 1: Line and circle are separate
theorem line_circle_separate :
  ∀ x y t : ℝ,
  line_l 8 0 (π/3) t x y →
  ¬ circle_C x y :=
sorry

-- Part 2: Trajectory of point P
theorem trajectory_of_P :
  ∀ a b x y : ℝ,
  (∃ α t₁ t₂ : ℝ,
    circle_C (a + t₁ * Real.cos α) (b + t₁ * Real.sin α) ∧
    circle_C (a + t₂ * Real.cos α) (b + t₂ * Real.sin α) ∧
    t₁ ≠ t₂ ∧
    (a^2 + b^2) * ((a + t₁ * Real.cos α)^2 + (b + t₁ * Real.sin α)^2) =
    ((a + t₂ * Real.cos α)^2 + (b + t₂ * Real.sin α)^2) * a^2 + b^2) →
  x^2 + y^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_line_circle_separate_trajectory_of_P_l210_21049


namespace NUMINAMATH_CALUDE_duck_pond_problem_l210_21020

theorem duck_pond_problem :
  let small_pond_total : ℕ := 30
  let small_pond_green_ratio : ℚ := 1/5
  let large_pond_green_ratio : ℚ := 3/25
  let total_green_ratio : ℚ := 3/20
  ∃ (large_pond_total : ℕ),
    (small_pond_green_ratio * small_pond_total + large_pond_green_ratio * large_pond_total : ℚ) = 
    total_green_ratio * (small_pond_total + large_pond_total) ∧
    large_pond_total = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l210_21020


namespace NUMINAMATH_CALUDE_book_purchase_equation_l210_21071

/-- Represents a book purchase scenario with two purchases -/
structure BookPurchase where
  first_cost : ℝ
  second_cost : ℝ
  quantity_difference : ℕ
  first_quantity : ℝ

/-- The equation correctly represents the book purchase scenario -/
def correct_equation (bp : BookPurchase) : Prop :=
  bp.first_cost / bp.first_quantity = bp.second_cost / (bp.first_quantity + bp.quantity_difference)

/-- Theorem stating that the given equation correctly represents the book purchase scenario -/
theorem book_purchase_equation (bp : BookPurchase) 
  (h1 : bp.first_cost = 7000)
  (h2 : bp.second_cost = 9000)
  (h3 : bp.quantity_difference = 60)
  (h4 : bp.first_quantity > 0) :
  correct_equation bp := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_equation_l210_21071


namespace NUMINAMATH_CALUDE_min_value_theorem_l210_21093

/-- The line equation ax + by - 2 = 0 --/
def line_equation (a b x y : ℝ) : Prop := a * x + b * y - 2 = 0

/-- The circle equation x^2 + y^2 - 2x - 2y = 2 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 2

/-- The line bisects the circumference of the circle --/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y, line_equation a b x y → circle_equation x y →
    ∃ c d, c^2 + d^2 = 1 ∧ line_equation a b (1 + c) (1 + d)

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_bisect : line_bisects_circle a b) :
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l210_21093


namespace NUMINAMATH_CALUDE_peach_basket_count_l210_21024

/-- The number of peaches in each original basket -/
def peaches_per_basket : ℕ := 25

/-- The number of peaches eaten by farmers -/
def peaches_eaten : ℕ := 5

/-- The number of peaches in each smaller box -/
def peaches_per_box : ℕ := 15

/-- The number of smaller boxes -/
def num_boxes : ℕ := 8

/-- The number of baskets delivered to the market -/
def baskets_delivered : ℕ := 5

theorem peach_basket_count :
  baskets_delivered * peaches_per_basket = 
    num_boxes * peaches_per_box + peaches_eaten := by
  sorry

end NUMINAMATH_CALUDE_peach_basket_count_l210_21024


namespace NUMINAMATH_CALUDE_runners_alignment_time_l210_21016

def steinLapTime : ℕ := 6
def roseLapTime : ℕ := 10
def schwartzLapTime : ℕ := 18

theorem runners_alignment_time :
  Nat.lcm steinLapTime (Nat.lcm roseLapTime schwartzLapTime) = 90 := by
  sorry

end NUMINAMATH_CALUDE_runners_alignment_time_l210_21016


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l210_21054

/-- Given a cone with base radius √2 and lateral surface that unfolds into a semicircle,
    the length of the generatrix is 2√2. -/
theorem cone_generatrix_length (r : ℝ) (l : ℝ) : 
  r = Real.sqrt 2 → 
  2 * Real.pi * r = Real.pi * l → 
  l = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l210_21054


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l210_21069

theorem max_value_theorem (A M C : ℕ) (h : A + M + C = 15) :
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 :=
by sorry

theorem max_value_achieved (A M C : ℕ) (h : A + M + C = 15) :
  ∃ A M C, A + M + C = 15 ∧ 2 * (A * M * C) + A * M + M * C + C * A = 325 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l210_21069


namespace NUMINAMATH_CALUDE_chipmunk_families_left_l210_21018

theorem chipmunk_families_left (original : ℕ) (went_away : ℕ) (h1 : original = 86) (h2 : went_away = 65) :
  original - went_away = 21 := by
  sorry

end NUMINAMATH_CALUDE_chipmunk_families_left_l210_21018


namespace NUMINAMATH_CALUDE_senior_titles_in_sample_l210_21062

/-- Represents the number of staff members with senior titles in a stratified sample -/
def seniorTitlesInSample (totalStaff : ℕ) (seniorStaff : ℕ) (sampleSize : ℕ) : ℕ :=
  (seniorStaff * sampleSize) / totalStaff

/-- Theorem: In a company with 150 staff members, including 15 with senior titles,
    a stratified sample of size 30 will contain 3 staff members with senior titles. -/
theorem senior_titles_in_sample :
  seniorTitlesInSample 150 15 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_senior_titles_in_sample_l210_21062


namespace NUMINAMATH_CALUDE_expression_value_l210_21045

theorem expression_value (a : ℚ) (h : a = 1/3) : (2 * a⁻¹ + a⁻¹ / 3) / a^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l210_21045


namespace NUMINAMATH_CALUDE_alien_eggs_count_l210_21088

-- Define a function to convert a number from base 7 to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem alien_eggs_count :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end NUMINAMATH_CALUDE_alien_eggs_count_l210_21088


namespace NUMINAMATH_CALUDE_range_of_m_l210_21052

theorem range_of_m (p q : ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2) →
  (∀ x, q x ↔ x^2 - 2*x + (1-m^2) ≤ 0) →
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x)) →
  (∃ x, ¬(p x) ∧ q x) →
  m ≥ 9 ∧ ∀ k ≥ 9, ∃ x, ¬(p x) ∧ q x :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l210_21052


namespace NUMINAMATH_CALUDE_max_value_operation_l210_21078

theorem max_value_operation (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999) :
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → (300 - m)^2 - 10 ≤ (300 - n)^2 - 10) →
  (300 - n)^2 - 10 = 39990 :=
by sorry

end NUMINAMATH_CALUDE_max_value_operation_l210_21078


namespace NUMINAMATH_CALUDE_inverse_sum_zero_l210_21042

theorem inverse_sum_zero (a b : ℝ) (h : a * b = 1) :
  a^2015 * b^2016 + a^2016 * b^2017 + a^2017 * b^2016 + a^2016 * b^2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_zero_l210_21042


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l210_21017

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 6 + x) ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l210_21017


namespace NUMINAMATH_CALUDE_cookies_per_box_type1_is_12_l210_21029

/-- Represents the number of cookies in a box of the first type -/
def cookies_per_box_type1 : ℕ := 12

/-- Represents the number of cookies in a box of the second type -/
def cookies_per_box_type2 : ℕ := 20

/-- Represents the number of cookies in a box of the third type -/
def cookies_per_box_type3 : ℕ := 16

/-- Represents the number of boxes sold of the first type -/
def boxes_sold_type1 : ℕ := 50

/-- Represents the number of boxes sold of the second type -/
def boxes_sold_type2 : ℕ := 80

/-- Represents the number of boxes sold of the third type -/
def boxes_sold_type3 : ℕ := 70

/-- Represents the total number of cookies sold -/
def total_cookies_sold : ℕ := 3320

/-- Theorem stating that the number of cookies in each box of the first type is 12 -/
theorem cookies_per_box_type1_is_12 :
  cookies_per_box_type1 * boxes_sold_type1 +
  cookies_per_box_type2 * boxes_sold_type2 +
  cookies_per_box_type3 * boxes_sold_type3 = total_cookies_sold :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_box_type1_is_12_l210_21029


namespace NUMINAMATH_CALUDE_airplane_capacity_theorem_l210_21003

/-- The total luggage weight an airplane can hold -/
def airplane_luggage_capacity 
  (num_people : ℕ) 
  (bags_per_person : ℕ) 
  (bag_weight : ℕ) 
  (additional_bags : ℕ) : ℕ :=
  (num_people * bags_per_person * bag_weight) + (additional_bags * bag_weight)

/-- Theorem stating the total luggage weight the airplane can hold -/
theorem airplane_capacity_theorem :
  airplane_luggage_capacity 6 5 50 90 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_airplane_capacity_theorem_l210_21003


namespace NUMINAMATH_CALUDE_finite_decimals_are_rational_l210_21095

theorem finite_decimals_are_rational : 
  ∀ x : ℝ, (∃ n : ℕ, ∃ m : ℤ, x = m / (10 ^ n)) → ∃ a b : ℤ, x = a / b ∧ b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_finite_decimals_are_rational_l210_21095


namespace NUMINAMATH_CALUDE_additional_license_plates_l210_21031

-- Define the original sets
def original_first_section : ℕ := 5
def original_second_section : ℕ := 3
def original_third_section : ℕ := 5

-- Define the updated sets
def updated_first_section : ℕ := original_first_section + 1
def updated_second_section : ℕ := original_second_section + 1

-- Define the function to calculate the number of license plates
def license_plate_count (first : ℕ) (second : ℕ) (third : ℕ) : ℕ :=
  first * second * third

-- Theorem statement
theorem additional_license_plates :
  license_plate_count updated_first_section updated_second_section original_third_section -
  license_plate_count original_first_section original_second_section original_third_section = 45 :=
by sorry

end NUMINAMATH_CALUDE_additional_license_plates_l210_21031


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l210_21033

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 - x) / (x - 2) - 1 = 2 / (2 - x) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l210_21033


namespace NUMINAMATH_CALUDE_expected_waiting_time_correct_l210_21055

/-- Represents the arrival time of a bus in minutes past 8:00 AM -/
def BusArrivalTime := Fin 120

/-- Represents the probability distribution of bus arrivals -/
def BusDistribution := BusArrivalTime → ℝ

/-- The first bus arrives randomly between 8:00 and 9:00 -/
def firstBusDistribution : BusDistribution := sorry

/-- The second bus arrives randomly between 9:00 and 10:00 -/
def secondBusDistribution : BusDistribution := sorry

/-- The passenger arrival time in minutes past 8:00 AM -/
def passengerArrivalTime : ℕ := 20

/-- Expected waiting time function -/
def expectedWaitingTime (firstBus secondBus : BusDistribution) (passengerTime : ℕ) : ℝ := sorry

theorem expected_waiting_time_correct :
  expectedWaitingTime firstBusDistribution secondBusDistribution passengerArrivalTime = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_waiting_time_correct_l210_21055


namespace NUMINAMATH_CALUDE_octal_to_binary_127_l210_21050

theorem octal_to_binary_127 : 
  (1 * 8^2 + 2 * 8^1 + 7 * 8^0 : ℕ) = (1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_octal_to_binary_127_l210_21050


namespace NUMINAMATH_CALUDE_lowest_number_in_range_l210_21057

/-- The probability of selecting a number greater than another randomly selected number -/
def probability : ℚ := 4995 / 10000

/-- Theorem stating that given the probability, the lowest number in the range is 999 -/
theorem lowest_number_in_range (x y : ℕ) (h : x ≤ y) :
  (((y - x) * (y - x + 1) : ℚ) / (2 * (y - x + 1)^2)) = probability → x = 999 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_in_range_l210_21057


namespace NUMINAMATH_CALUDE_apples_to_pears_ratio_l210_21034

/-- Represents the contents of a shopping cart --/
structure ShoppingCart where
  apples : ℕ
  oranges : ℕ
  pears : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Defines the relationships between fruit quantities in the shopping cart --/
def validCart (cart : ShoppingCart) : Prop :=
  cart.oranges = 2 * cart.apples ∧
  cart.pears = 5 * cart.oranges ∧
  cart.bananas = 3 * cart.pears ∧
  cart.peaches = cart.bananas / 2

/-- Theorem stating that apples are 1/10 of pears in a valid shopping cart --/
theorem apples_to_pears_ratio (cart : ShoppingCart) (h : validCart cart) :
  cart.apples = cart.pears / 10 := by
  sorry


end NUMINAMATH_CALUDE_apples_to_pears_ratio_l210_21034


namespace NUMINAMATH_CALUDE_canoe_trip_average_speed_l210_21080

/-- Proves that the average distance per day for the remaining days of a canoe trip is 32 km/day -/
theorem canoe_trip_average_speed
  (total_distance : ℝ)
  (total_days : ℕ)
  (completed_days : ℕ)
  (completed_fraction : ℚ)
  (h1 : total_distance = 168)
  (h2 : total_days = 6)
  (h3 : completed_days = 3)
  (h4 : completed_fraction = 3/7)
  : (total_distance - completed_fraction * total_distance) / (total_days - completed_days : ℝ) = 32 := by
  sorry

#check canoe_trip_average_speed

end NUMINAMATH_CALUDE_canoe_trip_average_speed_l210_21080


namespace NUMINAMATH_CALUDE_range_of_a_solution_set_l210_21002

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part I
theorem range_of_a (a : ℝ) :
  (∃ x, f x < 2 * a - 1) ↔ a > 2 :=
sorry

-- Theorem for part II
theorem solution_set :
  {x : ℝ | f x ≥ x^2 - 2*x} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 + Real.sqrt 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_solution_set_l210_21002


namespace NUMINAMATH_CALUDE_linear_equation_system_l210_21074

theorem linear_equation_system (a b c : ℝ) 
  (eq1 : a + 2*b - 3*c = 4)
  (eq2 : 5*a - 6*b + 7*c = 8) :
  9*a + 2*b - 5*c = 24 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_system_l210_21074


namespace NUMINAMATH_CALUDE_julia_tag_players_l210_21097

theorem julia_tag_players (monday_kids : ℕ) (difference : ℕ) (tuesday_kids : ℕ) 
  (h1 : monday_kids = 18)
  (h2 : difference = 8)
  (h3 : monday_kids = tuesday_kids + difference) :
  tuesday_kids = 10 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_players_l210_21097


namespace NUMINAMATH_CALUDE_defined_implies_continuous_but_not_conversely_l210_21001

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Statement: If f is defined at x₀, then f is continuous at x₀,
-- but the converse is not always true
theorem defined_implies_continuous_but_not_conversely :
  (∃ y, f x₀ = y) → ContinuousAt f x₀ ∧ 
  ¬(∀ g : ℝ → ℝ, ContinuousAt g x₀ → ∃ y, g x₀ = y) :=
sorry

end NUMINAMATH_CALUDE_defined_implies_continuous_but_not_conversely_l210_21001


namespace NUMINAMATH_CALUDE_chord_length_l210_21096

-- Define the circle and chord
def circle_radius : ℝ := 5
def center_to_chord : ℝ := 4

-- Theorem statement
theorem chord_length :
  ∀ (chord_length : ℝ),
  circle_radius = 5 ∧
  center_to_chord = 4 →
  chord_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l210_21096


namespace NUMINAMATH_CALUDE_brooklyn_annual_donation_l210_21076

/-- Brooklyn's monthly donation in dollars -/
def monthly_donation : ℕ := 1453

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Brooklyn's total donation in a year -/
def annual_donation : ℕ := monthly_donation * months_in_year

theorem brooklyn_annual_donation : annual_donation = 17436 := by
  sorry

end NUMINAMATH_CALUDE_brooklyn_annual_donation_l210_21076


namespace NUMINAMATH_CALUDE_shortest_leg_of_smallest_triangle_l210_21058

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  short_leg : ℝ
  long_leg : ℝ
  hypotenuse : ℝ
  short_leg_prop : short_leg = hypotenuse / 2
  long_leg_prop : long_leg = short_leg * Real.sqrt 3

/-- Represents a series of three 30-60-90 triangles -/
structure TriangleSeries where
  large : Triangle30_60_90
  medium : Triangle30_60_90
  small : Triangle30_60_90
  large_medium_relation : large.short_leg = medium.hypotenuse
  medium_small_relation : medium.short_leg = small.hypotenuse
  largest_hypotenuse : large.hypotenuse = 12

theorem shortest_leg_of_smallest_triangle (series : TriangleSeries) :
  series.small.short_leg = 1.5 := by sorry

end NUMINAMATH_CALUDE_shortest_leg_of_smallest_triangle_l210_21058


namespace NUMINAMATH_CALUDE_f_less_than_g_max_l210_21048

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * log x

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem f_less_than_g_max (a : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂) →
  a > log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_f_less_than_g_max_l210_21048


namespace NUMINAMATH_CALUDE_cube_sum_product_l210_21036

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_product_l210_21036


namespace NUMINAMATH_CALUDE_cricket_team_size_l210_21083

/-- Represents a cricket team with given age properties -/
structure CricketTeam where
  n : ℕ  -- number of team members
  captain_age : ℕ
  wicket_keeper_age : ℕ
  team_avg_age : ℝ
  remaining_avg_age : ℝ

/-- The cricket team satisfies the given conditions -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.captain_age = 26 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.team_avg_age = 23 ∧
  team.remaining_avg_age = team.team_avg_age - 1 ∧
  (team.n : ℝ) * team.team_avg_age = 
    (team.n - 2 : ℝ) * team.remaining_avg_age + team.captain_age + team.wicket_keeper_age

theorem cricket_team_size (team : CricketTeam) :
  valid_cricket_team team → team.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l210_21083


namespace NUMINAMATH_CALUDE_triangle_properties_l210_21056

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- Define the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : Real.sin (t.A + t.B) = 3/5)
  (h2 : Real.sin (t.A - t.B) = 1/5)
  (h3 : ∃ (AB : Real), AB = 3) :
  (Real.tan t.A = 2 * Real.tan t.B) ∧
  (∃ (height : Real), height = 2 + Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l210_21056


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_exponential_inequality_l210_21040

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, Real.exp x ≥ 1) ↔ (∃ x : ℝ, Real.exp x < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_exponential_inequality_l210_21040


namespace NUMINAMATH_CALUDE_min_magnitude_u_l210_21092

/-- The minimum magnitude of vector u -/
theorem min_magnitude_u (a b : ℝ × ℝ) (h1 : a = (Real.cos (25 * π / 180), Real.sin (25 * π / 180)))
  (h2 : b = (Real.sin (20 * π / 180), Real.cos (20 * π / 180))) :
  (∃ (t : ℝ), ∀ (s : ℝ), ‖a + s • b‖ ≥ ‖a + t • b‖) ∧
  (∃ (u : ℝ × ℝ), ∃ (t : ℝ), u = a + t • b ∧ ‖u‖ = Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_u_l210_21092


namespace NUMINAMATH_CALUDE_proposition_relationship_l210_21041

theorem proposition_relationship (x y : ℤ) :
  (∀ x y, x + y ≠ 2010 → (x ≠ 1010 ∨ y ≠ 1000)) ∧
  (∃ x y, (x ≠ 1010 ∨ y ≠ 1000) ∧ x + y = 2010) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l210_21041


namespace NUMINAMATH_CALUDE_dagger_example_l210_21000

-- Define the † operation
def dagger (m n p q : ℕ) (hm : m ≠ 0) : ℚ :=
  (m^2 * p * (q / n : ℚ)) + (p / m : ℚ)

-- Theorem statement
theorem dagger_example : dagger 5 9 6 2 (by norm_num) = 518 / 15 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l210_21000


namespace NUMINAMATH_CALUDE_twin_primes_sum_divisible_by_12_l210_21086

theorem twin_primes_sum_divisible_by_12 (p : ℕ) (h1 : p > 3) (h2 : Prime p) (h3 : Prime (p + 2)) :
  12 ∣ (p + (p + 2)) :=
sorry

end NUMINAMATH_CALUDE_twin_primes_sum_divisible_by_12_l210_21086


namespace NUMINAMATH_CALUDE_joaozinho_meeting_day_l210_21081

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to determine if Joãozinho lies on a given day
def lies_on_day (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Saturday

-- Define a function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Theorem statement
theorem joaozinho_meeting_day :
  ∀ (meeting_day : Day),
    (lies_on_day meeting_day →
      (meeting_day ≠ Day.Saturday ∧
       next_day meeting_day ≠ Day.Wednesday)) →
    meeting_day = Day.Thursday :=
by
  sorry


end NUMINAMATH_CALUDE_joaozinho_meeting_day_l210_21081


namespace NUMINAMATH_CALUDE_tan_30_degrees_l210_21073

theorem tan_30_degrees :
  let sin_30 := (1 : ℝ) / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_30 := sin_30 / cos_30
  tan_30 = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l210_21073


namespace NUMINAMATH_CALUDE_max_product_at_three_l210_21077

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

def product_of_terms (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (a₁ * r^((n-1)/2))^n

theorem max_product_at_three (a₁ r : ℝ) (h₁ : a₁ = 3) (h₂ : r = 2/5) :
  ∀ k : ℕ, k ≠ 0 → product_of_terms a₁ r 3 ≥ product_of_terms a₁ r k :=
by sorry

end NUMINAMATH_CALUDE_max_product_at_three_l210_21077


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l210_21035

theorem x_value_from_fraction_equality (x y : ℝ) :
  x ≠ 2 →
  x / (x - 2) = (y^2 + 3*y - 4) / (y^2 + 3*y - 5) →
  x = 2*y^2 + 6*y - 8 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l210_21035
