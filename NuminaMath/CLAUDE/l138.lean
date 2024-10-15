import Mathlib

namespace NUMINAMATH_CALUDE_smallest_m_for_positive_integer_solutions_l138_13898

theorem smallest_m_for_positive_integer_solutions :
  ∃ (m : ℤ), m = -1 ∧
  (∀ k : ℤ, k < m →
    ¬∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ x + y = 2*k + 7 ∧ x - 2*y = 4*k - 3) ∧
  (∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ x + y = 2*m + 7 ∧ x - 2*y = 4*m - 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_positive_integer_solutions_l138_13898


namespace NUMINAMATH_CALUDE_art_marks_calculation_l138_13878

theorem art_marks_calculation (geography : ℕ) (history_government : ℕ) (computer_science : ℕ) (modern_literature : ℕ) (average : ℚ) :
  geography = 56 →
  history_government = 60 →
  computer_science = 85 →
  modern_literature = 80 →
  average = 70.6 →
  ∃ (art : ℕ), (geography + history_government + art + computer_science + modern_literature : ℚ) / 5 = average ∧ art = 72 :=
by
  sorry

#check art_marks_calculation

end NUMINAMATH_CALUDE_art_marks_calculation_l138_13878


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l138_13803

theorem geometric_sequence_sum (a₁ r : ℝ) (n : ℕ) : 
  a₁ = 4 → r = 2 → n = 4 → 
  (a₁ * (1 - r^n)) / (1 - r) = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l138_13803


namespace NUMINAMATH_CALUDE_least_months_to_triple_l138_13874

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 1500

/-- The monthly interest rate as a decimal -/
def interest_rate : ℝ := 0.06

/-- The factor by which the borrowed amount increases each month -/
def growth_factor : ℝ := 1 + interest_rate

/-- The amount owed after t months -/
def amount_owed (t : ℕ) : ℝ := initial_amount * growth_factor ^ t

/-- Predicate that checks if the amount owed exceeds three times the initial amount -/
def exceeds_triple (t : ℕ) : Prop := amount_owed t > 3 * initial_amount

theorem least_months_to_triple :
  (∀ m : ℕ, m < 20 → ¬(exceeds_triple m)) ∧ exceeds_triple 20 :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l138_13874


namespace NUMINAMATH_CALUDE_clock_angle_at_7_clock_angle_at_7_is_150_l138_13893

/-- The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees. -/
theorem clock_angle_at_7 : ℝ :=
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let hours_at_7 : ℕ := 7
  let angle_per_hour : ℝ := total_degrees / total_hours
  let hour_hand_angle : ℝ := angle_per_hour * hours_at_7
  let smaller_angle : ℝ := total_degrees - hour_hand_angle
  smaller_angle

/-- The theorem states that the smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees. -/
theorem clock_angle_at_7_is_150 : clock_angle_at_7 = 150 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_7_clock_angle_at_7_is_150_l138_13893


namespace NUMINAMATH_CALUDE_floor_ratio_property_l138_13822

theorem floor_ratio_property (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) 
  (h : ∀ n : ℕ, Int.floor (x / y) = Int.floor (n * x) / Int.floor (n * y)) :
  x = y ∨ (∃ (a b : ℕ), x = a ∧ y = b ∧ (a ∣ b ∨ b ∣ a)) :=
sorry

end NUMINAMATH_CALUDE_floor_ratio_property_l138_13822


namespace NUMINAMATH_CALUDE_angle_C_measure_side_ratio_bounds_l138_13884

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π

variable (t : Triangle)

/-- First theorem: If sin(2C - π/2) = 1/2 and a² + b² < c², then C = 2π/3 -/
theorem angle_C_measure (h1 : sin (2 * t.C - π/2) = 1/2) (h2 : t.a^2 + t.b^2 < t.c^2) :
  t.C = 2*π/3 := by sorry

/-- Second theorem: If C = 2π/3, then 1 < (a + b)/c ≤ 2√3/3 -/
theorem side_ratio_bounds (h : t.C = 2*π/3) :
  1 < (t.a + t.b) / t.c ∧ (t.a + t.b) / t.c ≤ 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_side_ratio_bounds_l138_13884


namespace NUMINAMATH_CALUDE_sum_lent_problem_l138_13845

/-- Proves that given the conditions of the problem, the sum lent is 450 Rs. -/
theorem sum_lent_problem (P : ℝ) : 
  (P * 0.04 * 8 = P - 306) → P = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_problem_l138_13845


namespace NUMINAMATH_CALUDE_remainder_531531_mod_6_l138_13848

theorem remainder_531531_mod_6 : 531531 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_531531_mod_6_l138_13848


namespace NUMINAMATH_CALUDE_stratified_sample_size_l138_13829

theorem stratified_sample_size 
  (population_ratio_A : ℚ) 
  (population_ratio_B : ℚ) 
  (population_ratio_C : ℚ) 
  (sample_size_A : ℕ) 
  (total_sample_size : ℕ) :
  population_ratio_A = 3 / 14 →
  population_ratio_B = 4 / 14 →
  population_ratio_C = 7 / 14 →
  sample_size_A = 15 →
  population_ratio_A = sample_size_A / total_sample_size →
  total_sample_size = 70 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l138_13829


namespace NUMINAMATH_CALUDE_preimage_of_point_l138_13888

def f (x y : ℝ) : ℝ × ℝ := (2*x + y, x*y)

theorem preimage_of_point (x₁ y₁ x₂ y₂ : ℝ) :
  f x₁ y₁ = (1/6, -1/6) ∧ f x₂ y₂ = (1/6, -1/6) ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  ((x₁ = 1/4 ∧ y₁ = -1/3) ∨ (x₁ = -1/3 ∧ y₁ = 7/6)) ∧
  ((x₂ = 1/4 ∧ y₂ = -1/3) ∨ (x₂ = -1/3 ∧ y₂ = 7/6)) :=
sorry

end NUMINAMATH_CALUDE_preimage_of_point_l138_13888


namespace NUMINAMATH_CALUDE_sara_oranges_l138_13882

theorem sara_oranges (joan_oranges : ℕ) (total_oranges : ℕ) (h1 : joan_oranges = 37) (h2 : total_oranges = 47) :
  total_oranges - joan_oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_oranges_l138_13882


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l138_13820

/-- The radius of the incircle of a triangle with sides 5, 12, and 13 units is 2 units. -/
theorem incircle_radius_of_special_triangle : 
  ∀ (a b c r : ℝ), 
  a = 5 → b = 12 → c = 13 →
  r = (a * b) / (a + b + c) →
  r = 2 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l138_13820


namespace NUMINAMATH_CALUDE_least_clock_equivalent_l138_13823

def clock_equivalent (n : ℕ) : Prop :=
  24 ∣ (n^2 - n)

theorem least_clock_equivalent : 
  ∀ k : ℕ, k > 5 → clock_equivalent k → k ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_l138_13823


namespace NUMINAMATH_CALUDE_third_person_contribution_l138_13830

theorem third_person_contribution
  (total : ℕ)
  (h_total : total = 1040)
  (x : ℕ)
  (h_brittany : 3 * x = Brittany)
  (h_angela : 3 * Brittany = Angela)
  (h_sum : x + Brittany + Angela = total) :
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_third_person_contribution_l138_13830


namespace NUMINAMATH_CALUDE_factors_of_243_times_5_l138_13806

-- Define the number we're working with
def n : Nat := 243 * 5

-- Define a function to count the number of distinct positive factors
def countDistinctPositiveFactors (x : Nat) : Nat :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card

-- State the theorem
theorem factors_of_243_times_5 : countDistinctPositiveFactors n = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_243_times_5_l138_13806


namespace NUMINAMATH_CALUDE_rectangular_field_area_l138_13812

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio. -/
theorem rectangular_field_area 
  (perimeter : ℝ) 
  (width_to_length_ratio : ℝ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : width_to_length_ratio = 1/3) : 
  let width := perimeter / (2 * (1 + 1/width_to_length_ratio))
  let length := width / width_to_length_ratio
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l138_13812


namespace NUMINAMATH_CALUDE_three_digit_number_ending_in_five_divisible_by_five_l138_13849

theorem three_digit_number_ending_in_five_divisible_by_five (N : ℕ) :
  100 ≤ N ∧ N < 1000 ∧ N % 10 = 5 → N % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_ending_in_five_divisible_by_five_l138_13849


namespace NUMINAMATH_CALUDE_anniversary_number_is_counting_l138_13824

/-- Represents the categories of numbers in context --/
inductive NumberCategory
  | Label
  | MeasurementResult
  | Counting

/-- Represents the context in which the number is used --/
structure AnniversaryContext where
  years : ℕ

/-- Determines the category of a number in the anniversary context --/
def categorizeAnniversaryNumber (context : AnniversaryContext) : NumberCategory :=
  NumberCategory.Counting

/-- Theorem stating that the number used for anniversary years is a counting number --/
theorem anniversary_number_is_counting (context : AnniversaryContext) :
  categorizeAnniversaryNumber context = NumberCategory.Counting :=
by sorry

end NUMINAMATH_CALUDE_anniversary_number_is_counting_l138_13824


namespace NUMINAMATH_CALUDE_remainder_after_adding_2010_l138_13854

theorem remainder_after_adding_2010 (n : ℤ) (h : n % 6 = 1) : (n + 2010) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2010_l138_13854


namespace NUMINAMATH_CALUDE_floor_e_equals_two_l138_13880

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_e_equals_two_l138_13880


namespace NUMINAMATH_CALUDE_tv_sale_value_increase_l138_13809

theorem tv_sale_value_increase 
  (original_price original_quantity : ℝ) 
  (original_price_positive : 0 < original_price)
  (original_quantity_positive : 0 < original_quantity) :
  let price_reduction_factor := 0.8
  let sales_increase_factor := 1.8
  let new_price := price_reduction_factor * original_price
  let new_quantity := sales_increase_factor * original_quantity
  let original_sale_value := original_price * original_quantity
  let new_sale_value := new_price * new_quantity
  (new_sale_value - original_sale_value) / original_sale_value = 0.44 := by
sorry

end NUMINAMATH_CALUDE_tv_sale_value_increase_l138_13809


namespace NUMINAMATH_CALUDE_train_length_proof_l138_13855

def train_problem (length1 : ℝ) (speed1 : ℝ) (speed2 : ℝ) (clear_time : ℝ) : Prop :=
  let relative_speed : ℝ := (speed1 + speed2) * (1000 / 3600)
  let total_length : ℝ := relative_speed * clear_time
  let length2 : ℝ := total_length - length1
  length2 = 180

theorem train_length_proof :
  train_problem 110 80 65 7.199424046076314 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l138_13855


namespace NUMINAMATH_CALUDE_quadratic_root_two_l138_13801

theorem quadratic_root_two (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 2) ↔ 4 * a + 2 * b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_two_l138_13801


namespace NUMINAMATH_CALUDE_xy_value_l138_13840

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l138_13840


namespace NUMINAMATH_CALUDE_max_hardcover_books_l138_13895

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The problem statement -/
theorem max_hardcover_books :
  ∀ (hardcover paperback : ℕ),
  hardcover + paperback = 36 →
  IsComposite (paperback - hardcover) →
  hardcover ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_hardcover_books_l138_13895


namespace NUMINAMATH_CALUDE_julia_money_left_l138_13838

theorem julia_money_left (initial_amount : ℚ) : initial_amount = 40 →
  let after_game := initial_amount / 2
  let after_purchases := after_game - (after_game / 4)
  after_purchases = 15 := by
sorry

end NUMINAMATH_CALUDE_julia_money_left_l138_13838


namespace NUMINAMATH_CALUDE_function_approximation_by_additive_l138_13815

/-- Given a function f: ℝ → ℝ satisfying |f(x+y) - f(x) - f(y)| ≤ 1 for all x, y ∈ ℝ,
    there exists a function g: ℝ → ℝ such that |f(x) - g(x)| ≤ 1 and
    g(x+y) = g(x) + g(y) for all x, y ∈ ℝ. -/
theorem function_approximation_by_additive (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ 
               (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end NUMINAMATH_CALUDE_function_approximation_by_additive_l138_13815


namespace NUMINAMATH_CALUDE_melissa_games_played_l138_13847

/-- Given a player's points per game and total score, calculate the number of games played -/
def games_played (points_per_game : ℕ) (total_points : ℕ) : ℕ :=
  total_points / points_per_game

/-- Theorem: A player scoring 120 points per game with a total of 1200 points played 10 games -/
theorem melissa_games_played :
  games_played 120 1200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l138_13847


namespace NUMINAMATH_CALUDE_section_area_regular_triangular_pyramid_l138_13863

/-- The area of a section in a regular triangular pyramid -/
theorem section_area_regular_triangular_pyramid
  (a h : ℝ)
  (ha : a > 0)
  (hh : h > (a * Real.sqrt 6) / 6) :
  let area := (3 * a^2 * h) / (4 * Real.sqrt (a^2 + 3 * h^2))
  ∃ (S : ℝ), S = area ∧ S > 0 :=
by sorry

end NUMINAMATH_CALUDE_section_area_regular_triangular_pyramid_l138_13863


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l138_13832

theorem fifteenth_student_age 
  (total_students : ℕ)
  (avg_age_all : ℝ)
  (num_group1 : ℕ)
  (avg_age_group1 : ℝ)
  (num_group2 : ℕ)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + (num_group2 : ℝ) * avg_age_group2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l138_13832


namespace NUMINAMATH_CALUDE_sine_equality_theorem_l138_13867

theorem sine_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.sin (192 * π / 180)) ↔ (n = 12 ∨ n = 168) := by
sorry

end NUMINAMATH_CALUDE_sine_equality_theorem_l138_13867


namespace NUMINAMATH_CALUDE_maximize_product_l138_13872

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 50) :
  x^4 * y^3 ≤ (200/7)^4 * (150/7)^3 ∧
  (x = 200/7 ∧ y = 150/7) → x^4 * y^3 = (200/7)^4 * (150/7)^3 :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l138_13872


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l138_13814

def largest_number_divisible_by_88 : ℕ := 9944

theorem largest_number_divisible_by_88_has_4_digits :
  (largest_number_divisible_by_88 ≥ 1000) ∧ (largest_number_divisible_by_88 < 10000) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_88_has_4_digits_l138_13814


namespace NUMINAMATH_CALUDE_reciprocal_roots_iff_m_eq_p_l138_13877

/-- A quadratic equation with coefficients p, q, and m -/
structure QuadraticEquation where
  p : ℝ
  q : ℝ
  m : ℝ

/-- The roots of a quadratic equation are reciprocals -/
def has_reciprocal_roots (eq : QuadraticEquation) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ eq.p * r^2 + eq.q * r + eq.m = 0 ∧ eq.p * (1/r)^2 + eq.q * (1/r) + eq.m = 0

/-- Theorem: The roots of px^2 + qx + m = 0 are reciprocals iff m = p -/
theorem reciprocal_roots_iff_m_eq_p (eq : QuadraticEquation) :
  has_reciprocal_roots eq ↔ eq.m = eq.p :=
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_iff_m_eq_p_l138_13877


namespace NUMINAMATH_CALUDE_students_with_two_skills_l138_13860

theorem students_with_two_skills (total : ℕ) (no_poetry : ℕ) (no_paint : ℕ) (no_instrument : ℕ) 
  (h1 : total = 150)
  (h2 : no_poetry = 80)
  (h3 : no_paint = 90)
  (h4 : no_instrument = 60) :
  let poetry := total - no_poetry
  let paint := total - no_paint
  let instrument := total - no_instrument
  let two_skills := poetry + paint + instrument - total
  two_skills = 70 := by
  sorry

end NUMINAMATH_CALUDE_students_with_two_skills_l138_13860


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l138_13805

theorem polynomial_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  ((∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) ∧
   a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) →
  m = 3 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l138_13805


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l138_13875

theorem dogwood_trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_trees : ℕ) : 
  initial_trees = 7 → planted_today = 5 → final_trees = 16 → 
  final_trees - (initial_trees + planted_today) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l138_13875


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l138_13896

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m+1)*x + 25 = (x + a)^2) → 
  (m = 4 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l138_13896


namespace NUMINAMATH_CALUDE_correct_multiplication_result_l138_13817

theorem correct_multiplication_result : ∃ (n : ℕ), 
  (987 * n = 559989) ∧ 
  (∃ (a b : ℕ), 559981 = 550000 + a * 100 + b * 10 + 1 ∧ a ≠ 9 ∧ b ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_correct_multiplication_result_l138_13817


namespace NUMINAMATH_CALUDE_swimming_improvement_l138_13821

-- Define the initial performance
def initial_laps : ℕ := 15
def initial_time : ℕ := 30

-- Define the improved performance
def improved_laps : ℕ := 20
def improved_time : ℕ := 36

-- Define the improvement in lap time
def lap_time_improvement : ℚ := 
  (initial_time : ℚ) / initial_laps - (improved_time : ℚ) / improved_laps

-- Theorem statement
theorem swimming_improvement : lap_time_improvement = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_improvement_l138_13821


namespace NUMINAMATH_CALUDE_det_of_matrix_l138_13890

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, 4;
     0,  6, -3;
     3,  0,  1]

theorem det_of_matrix : Matrix.det matrix = -51 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_l138_13890


namespace NUMINAMATH_CALUDE_smallest_positive_root_comparison_l138_13811

theorem smallest_positive_root_comparison : ∃ (x₁ x₂ : ℝ), 
  (x₁ > 0 ∧ x₂ > 0) ∧ 
  (x₁^2011 + 2011*x₁ - 1 = 0) ∧
  (x₂^2011 - 2011*x₂ + 1 = 0) ∧
  (∀ y₁ > 0, y₁^2011 + 2011*y₁ - 1 = 0 → y₁ ≥ x₁) ∧
  (∀ y₂ > 0, y₂^2011 - 2011*y₂ + 1 = 0 → y₂ ≥ x₂) ∧
  (x₁ < x₂) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_root_comparison_l138_13811


namespace NUMINAMATH_CALUDE_min_value_shifted_l138_13842

/-- The function f(x) = x^2 + 4x + 5 - c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

/-- The shifted function f(x-2009) -/
def f_shifted (c : ℝ) (x : ℝ) : ℝ := f c (x - 2009)

theorem min_value_shifted (c : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ (∃ (x_0 : ℝ), f c x_0 = m) ∧ m = 2) →
  (∃ (m : ℝ), ∀ (x : ℝ), f_shifted c x ≥ m ∧ (∃ (x_0 : ℝ), f_shifted c x_0 = m) ∧ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_shifted_l138_13842


namespace NUMINAMATH_CALUDE_min_distance_and_line_equation_l138_13858

-- Define the line l: x - y + 3 = 0
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C: (x - 1)^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the distance PA
def distance_PA (x y : ℝ) : ℝ := sorry

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := 2*x - 2*y - 1 = 0

-- Theorem statement
theorem min_distance_and_line_equation :
  (∃ (x y : ℝ), point_P x y ∧ 
    (∀ (x' y' : ℝ), point_P x' y' → distance_PA x y ≤ distance_PA x' y')) ∧
  (∀ (x y : ℝ), point_P x y ∧ distance_PA x y = Real.sqrt 7 → line_AB x y) :=
sorry

end NUMINAMATH_CALUDE_min_distance_and_line_equation_l138_13858


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_with_tangent_chord_l138_13897

/-- The area between two concentric circles with a tangent chord -/
theorem area_between_concentric_circles_with_tangent_chord 
  (r : ℝ) -- radius of the smaller circle
  (c : ℝ) -- length of the chord of the larger circle
  (h1 : r = 40) -- given radius of the smaller circle
  (h2 : c = 120) -- given length of the chord
  : ∃ (A : ℝ), A = 3600 * Real.pi ∧ A = Real.pi * ((c / 2)^2 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_with_tangent_chord_l138_13897


namespace NUMINAMATH_CALUDE_T_equals_x_plus_one_to_fourth_l138_13892

theorem T_equals_x_plus_one_to_fourth (x : ℝ) : 
  (x + 2)^4 - 4*(x + 2)^3 + 6*(x + 2)^2 - 4*(x + 2) + 1 = (x + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_T_equals_x_plus_one_to_fourth_l138_13892


namespace NUMINAMATH_CALUDE_smallest_positive_n_squared_l138_13871

-- Define the circles c1 and c2
def c1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0
def c2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 175 = 0

-- Define a function to check if a point is on a line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the conditions for external and internal tangency
def externally_tangent (x y r : ℝ) : Prop := (x - 4)^2 + (y - 10)^2 = (r + 7)^2
def internally_tangent (x y r : ℝ) : Prop := (x + 4)^2 + (y - 10)^2 = (11 - r)^2

-- State the theorem
theorem smallest_positive_n_squared (n : ℝ) : 
  (∀ b : ℝ, b > 0 → b < n → 
    ¬∃ x y r : ℝ, on_line x y b ∧ externally_tangent x y r ∧ internally_tangent x y r) →
  (∃ x y r : ℝ, on_line x y n ∧ externally_tangent x y r ∧ internally_tangent x y r) →
  n^2 = 49/64 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_n_squared_l138_13871


namespace NUMINAMATH_CALUDE_prob_same_color_is_correct_l138_13837

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 3
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

def prob_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem prob_same_color_is_correct : prob_same_color = 66 / 1330 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_correct_l138_13837


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l138_13802

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (z^3 + y^3) / (x^2 + x*y + y^2) + (x^3 + z^3) / (y^2 + y*z + z^2) + (y^3 + x^3) / (z^2 + z*x + x^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l138_13802


namespace NUMINAMATH_CALUDE_best_representative_l138_13810

structure Student where
  name : String
  average_time : Float
  variance : Float

def is_better (s1 s2 : Student) : Prop :=
  (s1.average_time < s2.average_time) ∨
  (s1.average_time = s2.average_time ∧ s1.variance < s2.variance)

def is_best (s : Student) (students : List Student) : Prop :=
  ∀ other ∈ students, s ≠ other → is_better s other

theorem best_representative (students : List Student) :
  let a := { name := "A", average_time := 7.9, variance := 1.4 }
  let b := { name := "B", average_time := 8.2, variance := 2.2 }
  let c := { name := "C", average_time := 7.9, variance := 2.4 }
  let d := { name := "D", average_time := 8.2, variance := 1.4 }
  students = [a, b, c, d] →
  is_best a students :=
by sorry

end NUMINAMATH_CALUDE_best_representative_l138_13810


namespace NUMINAMATH_CALUDE_quadratic_condition_l138_13813

/-- A quadratic equation in one variable is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0

/-- The equation ax^2 - x + 2 = 0 -/
def equation (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - x + 2 = 0

theorem quadratic_condition (a : ℝ) :
  (∃ x, equation a x) → is_quadratic_equation a (-1) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_condition_l138_13813


namespace NUMINAMATH_CALUDE_third_side_length_l138_13856

/-- Triangle inequality theorem for a triangle with sides a, b, and c -/
axiom triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : 
  a + b > c ∧ b + c > a ∧ c + a > b

theorem third_side_length (x : ℝ) (hx : x > 0) : 
  (4 + x > 9 ∧ 9 + x > 4 ∧ 4 + 9 > x) → (x > 5 ∧ x < 13) := by sorry

end NUMINAMATH_CALUDE_third_side_length_l138_13856


namespace NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l138_13883

theorem tan_20_plus_4sin_20_equals_sqrt_3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_20_plus_4sin_20_equals_sqrt_3_l138_13883


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_2880_l138_13859

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_2880 :
  let factorization := prime_factorization 2880
  (factorization = [(2, 6), (3, 2), (5, 1)]) →
  count_perfect_square_factors 2880 = 8 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_2880_l138_13859


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l138_13846

theorem lcm_gcf_relation (n : ℕ) :
  n ≠ 0 ∧ Nat.lcm n 24 = 48 ∧ Nat.gcd n 24 = 8 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l138_13846


namespace NUMINAMATH_CALUDE_largest_common_term_proof_l138_13891

/-- The first arithmetic progression with common difference 5 -/
def ap1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression with common difference 11 -/
def ap2 (n : ℕ) : ℕ := 7 + 11 * n

/-- A common term of both arithmetic progressions -/
def common_term (k m : ℕ) : Prop := ap1 k = ap2 m

/-- The largest common term less than 1000 -/
def largest_common_term : ℕ := 964

theorem largest_common_term_proof :
  (∃ k m : ℕ, common_term k m ∧ largest_common_term = ap1 k) ∧
  (∀ n : ℕ, n < 1000 → (∃ k m : ℕ, common_term k m ∧ n = ap1 k) → n ≤ largest_common_term) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_proof_l138_13891


namespace NUMINAMATH_CALUDE_elephant_drinking_problem_l138_13852

/-- The number of days it takes for one elephant to drink a lake dry -/
def days_to_drink_lake (V C K : ℝ) : ℝ :=
  365

/-- Theorem stating the conditions and the result for the elephant drinking problem -/
theorem elephant_drinking_problem (V C K : ℝ) 
  (h1 : 183 * C = V + K)
  (h2 : 37 * 5 * C = V + 5 * K)
  (h3 : V > 0)
  (h4 : C > 0)
  (h5 : K > 0) :
  ∃ (t : ℝ), t * C = V + t * K ∧ t = days_to_drink_lake V C K :=
by
  sorry

#check elephant_drinking_problem

end NUMINAMATH_CALUDE_elephant_drinking_problem_l138_13852


namespace NUMINAMATH_CALUDE_power_of_fraction_l138_13828

theorem power_of_fraction : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_l138_13828


namespace NUMINAMATH_CALUDE_intersection_point_parametric_equation_l138_13894

/-- Given a triangle ABC with points D and E such that:
    - D lies on BC extended past C with BD:DC = 2:1
    - E lies on AC with AE:EC = 2:1
    - P is the intersection of BE and AD
    This theorem proves that P can be expressed as (1/7)A + (2/7)B + (4/7)C -/
theorem intersection_point_parametric_equation 
  (A B C D E P : ℝ × ℝ) : 
  (∃ t : ℝ, D = (1 - t) • B + t • C ∧ t = 2/3) →
  (∃ s : ℝ, E = (1 - s) • A + s • C ∧ s = 2/3) →
  (∃ u v : ℝ, P = (1 - u) • A + u • D ∧ P = (1 - v) • B + v • E) →
  P = (1/7) • A + (2/7) • B + (4/7) • C :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_parametric_equation_l138_13894


namespace NUMINAMATH_CALUDE_expression_simplification_l138_13862

theorem expression_simplification (x y : ℝ) : 
  4 * x + 8 * x^2 + 6 * y - (3 - 5 * x - 8 * x^2 - 2 * y) = 16 * x^2 + 9 * x + 8 * y - 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l138_13862


namespace NUMINAMATH_CALUDE_highway_mileage_calculation_l138_13899

/-- Calculates the highway mileage of a car given total distance, city distance, city mileage, and total gas used. -/
theorem highway_mileage_calculation 
  (total_highway_distance : ℝ) 
  (total_city_distance : ℝ) 
  (city_mileage : ℝ) 
  (total_gas_used : ℝ) 
  (h1 : total_highway_distance = 210)
  (h2 : total_city_distance = 54)
  (h3 : city_mileage = 18)
  (h4 : total_gas_used = 9) :
  (total_highway_distance / (total_gas_used - total_city_distance / city_mileage)) = 35 := by
sorry

end NUMINAMATH_CALUDE_highway_mileage_calculation_l138_13899


namespace NUMINAMATH_CALUDE_red_cars_count_l138_13834

/-- Given a parking lot where the ratio of red cars to black cars is 3:8
    and there are 90 black cars, prove that there are 33 red cars. -/
theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) :
  black_cars = 90 →
  ratio_red = 3 →
  ratio_black = 8 →
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 33 := by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l138_13834


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l138_13879

def blocks_west : ℕ := 9
def blocks_south : ℕ := 15
def mile_per_block : ℚ := 1/4

theorem arthur_walk_distance :
  (blocks_west + blocks_south : ℚ) * mile_per_block = 6 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l138_13879


namespace NUMINAMATH_CALUDE_daisy_sales_proof_l138_13870

/-- The number of daisies sold on the first day -/
def first_day_sales : ℕ := 45

/-- The number of daisies sold on the second day -/
def second_day_sales : ℕ := first_day_sales + 20

/-- The number of daisies sold on the third day -/
def third_day_sales : ℕ := 2 * second_day_sales - 10

/-- The number of daisies sold on the fourth day -/
def fourth_day_sales : ℕ := 120

/-- The total number of daisies sold over 4 days -/
def total_sales : ℕ := 350

theorem daisy_sales_proof :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = total_sales :=
by sorry

end NUMINAMATH_CALUDE_daisy_sales_proof_l138_13870


namespace NUMINAMATH_CALUDE_average_score_is_92_l138_13876

def brief_scores : List Int := [10, -5, 0, 8, -3]
def xiao_ming_score : Int := 90
def xiao_ming_rank : Nat := 3

def actual_scores : List Int := brief_scores.map (λ x => xiao_ming_score + x)

theorem average_score_is_92 : 
  (actual_scores.sum : ℚ) / actual_scores.length = 92 := by sorry

end NUMINAMATH_CALUDE_average_score_is_92_l138_13876


namespace NUMINAMATH_CALUDE_tan_alpha_eq_one_third_l138_13827

theorem tan_alpha_eq_one_third (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_one_third_l138_13827


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l138_13836

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => (17/3) * x^3 - 38 * x^2 - (101/3) * x + 185/3
  (q 1 = -5) ∧ (q 2 = 1) ∧ (q 3 = -1) ∧ (q 4 = 23) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l138_13836


namespace NUMINAMATH_CALUDE_candy_distribution_l138_13868

theorem candy_distribution (x : ℕ) : 
  x > 500 ∧ 
  x % 21 = 5 ∧ 
  x % 22 = 3 →
  x ≥ 509 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l138_13868


namespace NUMINAMATH_CALUDE_abs_neg_one_fourth_l138_13864

theorem abs_neg_one_fourth : |(-1 : ℚ) / 4| = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_fourth_l138_13864


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l138_13857

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l138_13857


namespace NUMINAMATH_CALUDE_right_triangle_area_l138_13816

/-- The area of a right triangle with vertices at (0, 0), (0, 7), and (-7, 0) is 24.5 square units. -/
theorem right_triangle_area : 
  let vertex1 : ℝ × ℝ := (0, 0)
  let vertex2 : ℝ × ℝ := (0, 7)
  let vertex3 : ℝ × ℝ := (-7, 0)
  let base : ℝ := 7
  let height : ℝ := 7
  let area : ℝ := (1 / 2) * base * height
  area = 24.5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l138_13816


namespace NUMINAMATH_CALUDE_symmetric_function_g_l138_13819

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1 - 2

-- Define the symmetry condition
def is_symmetric_about (g : ℝ → ℝ) (p : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, g x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

-- Theorem statement
theorem symmetric_function_g : 
  ∃ g : ℝ → ℝ, is_symmetric_about g (1, 2) f ∧ (∀ x, g x = 3 * x - 1) :=
sorry

end NUMINAMATH_CALUDE_symmetric_function_g_l138_13819


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l138_13850

/-- Given two vectors a and b in a 2D plane with specific magnitudes and angle between them,
    prove that the dot product of a and (a + b) is 12. -/
theorem dot_product_specific_vectors (a b : ℝ × ℝ) :
  ‖a‖ = 4 →
  ‖b‖ = Real.sqrt 2 →
  a • b = -4 →
  a • (a + b) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l138_13850


namespace NUMINAMATH_CALUDE_variation_problem_l138_13835

/-- Given:
  - R varies directly as the square of S and inversely as T^2
  - When R = 3, T = 2, S = 1
  Prove that when R = 75 and T = 5, S = 12.5
-/
theorem variation_problem (R S T : ℝ) (c : ℝ) : 
  (∀ R S T, R = c * S^2 / T^2) →  -- Relationship between R, S, and T
  (3 = c * 1^2 / 2^2) →           -- Given condition: R = 3, S = 1, T = 2
  (75 = c * S^2 / 5^2) →          -- Target condition: R = 75, T = 5
  S = 12.5 := by                  -- Prove S = 12.5
sorry


end NUMINAMATH_CALUDE_variation_problem_l138_13835


namespace NUMINAMATH_CALUDE_smaller_screen_diagonal_l138_13839

theorem smaller_screen_diagonal (d : ℝ) : 
  d > 0 → d^2 / 2 = 200 - 38 → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_smaller_screen_diagonal_l138_13839


namespace NUMINAMATH_CALUDE_binomial_coefficient_inequality_l138_13818

theorem binomial_coefficient_inequality (n k h : ℕ) (h1 : n ≥ k + h) :
  Nat.choose n (k + h) ≥ Nat.choose (n - k) h :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_inequality_l138_13818


namespace NUMINAMATH_CALUDE_missing_number_proof_l138_13841

theorem missing_number_proof (x y : ℝ) : 
  (12 + x + 42 + y + 104) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  y = 78 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l138_13841


namespace NUMINAMATH_CALUDE_harmonic_mean_closest_integer_l138_13885

theorem harmonic_mean_closest_integer :
  ∃ (h : ℝ), 
    (h = 2 / ((1 : ℝ)⁻¹ + (2023 : ℝ)⁻¹)) ∧ 
    (∀ n : ℤ, n ≠ 2 → |h - 2| < |h - (n : ℝ)|) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_closest_integer_l138_13885


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l138_13804

theorem square_perimeter_problem (perimeter_C : ℝ) (area_C area_D : ℝ) :
  perimeter_C = 32 →
  area_D = area_C / 8 →
  ∃ (side_D : ℝ), side_D * side_D = area_D ∧ 4 * side_D = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l138_13804


namespace NUMINAMATH_CALUDE_poster_placement_l138_13825

/-- Given a wall of width 25 feet and a centrally placed poster of width 4 feet,
    the distance from the end of the wall to the nearest edge of the poster is 10.5 feet. -/
theorem poster_placement (wall_width : ℝ) (poster_width : ℝ) 
    (h1 : wall_width = 25) 
    (h2 : poster_width = 4) :
  (wall_width - poster_width) / 2 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_poster_placement_l138_13825


namespace NUMINAMATH_CALUDE_extreme_points_when_a_is_one_extreme_points_condition_l138_13833

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x + 1 else a*x

-- Theorem 1: When a = 1, f(x) has exactly two extreme points
theorem extreme_points_when_a_is_one :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧
  (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f 1 y ≤ f 1 x) ∨
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f 1 y ≥ f 1 x)) ↔ (x = x1 ∨ x = x2)) :=
sorry

-- Theorem 2: f(x) has exactly two extreme points iff 0 < a < 2
theorem extreme_points_condition :
  ∀ (a : ℝ), (∃! (x1 x2 : ℝ), x1 ≠ x2 ∧
  (∀ (x : ℝ), (∃ (ε : ℝ), ε > 0 ∧
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f a y ≤ f a x) ∨
    (∀ (y : ℝ), y ≠ x ∧ |y - x| < ε → f a y ≥ f a x)) ↔ (x = x1 ∨ x = x2)))
  ↔ (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_when_a_is_one_extreme_points_condition_l138_13833


namespace NUMINAMATH_CALUDE_gabrielles_peaches_l138_13853

theorem gabrielles_peaches (martine benjy gabrielle : ℕ) 
  (h1 : martine = 2 * benjy + 6)
  (h2 : benjy = gabrielle / 3)
  (h3 : martine = 16) :
  gabrielle = 15 := by
  sorry

end NUMINAMATH_CALUDE_gabrielles_peaches_l138_13853


namespace NUMINAMATH_CALUDE_weightlifter_fourth_minute_l138_13843

/-- Calculates the total weight a weightlifter can lift in the 4th minute given initial weights,
    weight increments, and fatigue factor. -/
def weightLifterFourthMinute (leftInitial rightInitial leftIncrement rightIncrement fatigueDecline : ℕ) : ℕ :=
  let leftAfterThree := leftInitial + 3 * leftIncrement
  let rightAfterThree := rightInitial + 3 * rightIncrement
  let totalAfterThree := leftAfterThree + rightAfterThree
  totalAfterThree - fatigueDecline

/-- Theorem stating that the weightlifter can lift 55 pounds in the 4th minute under given conditions. -/
theorem weightlifter_fourth_minute :
  weightLifterFourthMinute 12 18 4 6 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_weightlifter_fourth_minute_l138_13843


namespace NUMINAMATH_CALUDE_sin_theta_value_l138_13869

theorem sin_theta_value (θ : Real) 
  (h1 : 6 * Real.tan θ = 5 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-3 + 2 * Real.sqrt 34) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l138_13869


namespace NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l138_13831

/-- The area of a triangle given two sides and the angle bisector between them. -/
theorem triangle_area_with_angle_bisector 
  (a b f_c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf_c : f_c > 0) 
  (h_triangle : 4 * a^2 * b^2 > (a + b)^2 * f_c^2) : 
  ∃ t : ℝ, t = ((a + b) * f_c) / (4 * a * b) * Real.sqrt (4 * a^2 * b^2 - (a + b)^2 * f_c^2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l138_13831


namespace NUMINAMATH_CALUDE_unique_perfect_cube_divisibility_l138_13887

theorem unique_perfect_cube_divisibility : ∃! X : ℕ+, 
  (∃ Y : ℕ+, X = Y^3) ∧ 
  X = (555 * 465)^2 * (555 - 465)^3 + (555 - 465)^4 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_cube_divisibility_l138_13887


namespace NUMINAMATH_CALUDE_simultaneous_integers_l138_13800

theorem simultaneous_integers (t : ℤ) : 
  let x : ℤ := 60 * t + 1
  (∃ (k₁ k₂ k₃ : ℤ), (2 * x + 1) / 3 = k₁ ∧ (3 * x + 1) / 4 = k₂ ∧ (4 * x + 1) / 5 = k₃) ∧
  (∀ (y : ℤ), y ≠ x → ¬(∃ (k₁ k₂ k₃ : ℤ), (2 * y + 1) / 3 = k₁ ∧ (3 * y + 1) / 4 = k₂ ∧ (4 * y + 1) / 5 = k₃)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_integers_l138_13800


namespace NUMINAMATH_CALUDE_net_pay_calculation_l138_13851

/-- Calculate net pay given gross pay and taxes paid -/
def net_pay (gross_pay taxes_paid : ℕ) : ℕ :=
  gross_pay - taxes_paid

/-- Theorem: Given the conditions, prove that the net pay is 315 dollars -/
theorem net_pay_calculation (gross_pay taxes_paid : ℕ) 
  (h1 : gross_pay = 450)
  (h2 : taxes_paid = 135) :
  net_pay gross_pay taxes_paid = 315 := by
  sorry

end NUMINAMATH_CALUDE_net_pay_calculation_l138_13851


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l138_13866

theorem arithmetic_expression_equals_24 : (8 * 10 - 8) / 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l138_13866


namespace NUMINAMATH_CALUDE_max_fraction_sum_l138_13881

theorem max_fraction_sum (a b c d : ℕ+) (h1 : a + c = 20) (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) :
  (∀ a' b' c' d' : ℕ+, a' + c' = 20 → (a' : ℚ) / b' + (c' : ℚ) / d' < 1 → (a : ℚ) / b + (c : ℚ) / d ≤ (a' : ℚ) / b' + (c' : ℚ) / d') →
  (a : ℚ) / b + (c : ℚ) / d = 20 / 21 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l138_13881


namespace NUMINAMATH_CALUDE_A_intersect_B_l138_13807

def A : Set (ℤ × ℤ) := {(1, 2), (2, 1)}
def B : Set (ℤ × ℤ) := {p : ℤ × ℤ | p.1 - p.2 = 1}

theorem A_intersect_B : A ∩ B = {(2, 1)} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l138_13807


namespace NUMINAMATH_CALUDE_parabola_parameter_values_l138_13889

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The point satisfies the parabola equation -/
def on_parabola (point : ParabolaPoint) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- The distance from the point to the directrix (x = -p/2) is 10 -/
def distance_to_directrix (point : ParabolaPoint) (parabola : Parabola) : Prop :=
  point.x + parabola.p / 2 = 10

/-- The distance from the point to the axis of symmetry (y-axis) is 6 -/
def distance_to_axis (point : ParabolaPoint) : Prop :=
  point.y = 6 ∨ point.y = -6

theorem parabola_parameter_values
  (parabola : Parabola)
  (point : ParabolaPoint)
  (h_on_parabola : on_parabola point parabola)
  (h_directrix : distance_to_directrix point parabola)
  (h_axis : distance_to_axis point) :
  parabola.p = 2 ∨ parabola.p = 18 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_values_l138_13889


namespace NUMINAMATH_CALUDE_max_product_divisible_by_55_l138_13886

/-- Represents a four-digit number in the form 11,0ab -/
structure Number11_0ab where
  a : Nat
  b : Nat
  a_single_digit : a < 10
  b_single_digit : b < 10

/-- Check if a number in the form 11,0ab is divisible by 55 -/
def isDivisibleBy55 (n : Number11_0ab) : Prop :=
  (11000 + 100 * n.a + n.b) % 55 = 0

/-- The maximum product of a and b for numbers divisible by 55 -/
def maxProduct : Nat :=
  25

theorem max_product_divisible_by_55 :
  ∀ n : Number11_0ab, isDivisibleBy55 n → n.a * n.b ≤ maxProduct :=
by sorry

end NUMINAMATH_CALUDE_max_product_divisible_by_55_l138_13886


namespace NUMINAMATH_CALUDE_four_in_B_iff_m_in_range_B_subset_A_iff_m_in_range_l138_13873

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part 1: 4 ∈ B iff m ∈ [5/2, 3]
theorem four_in_B_iff_m_in_range (m : ℝ) : 
  (4 ∈ B m) ↔ (5/2 ≤ m ∧ m ≤ 3) :=
sorry

-- Part 2: B ⊂ A iff m ∈ (-∞, 3]
theorem B_subset_A_iff_m_in_range (m : ℝ) :
  (B m ⊂ A) ↔ (m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_four_in_B_iff_m_in_range_B_subset_A_iff_m_in_range_l138_13873


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l138_13826

theorem square_root_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (40 - x) = 10 →
  (10 + x) * (40 - x) = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l138_13826


namespace NUMINAMATH_CALUDE_permutations_of_middle_digits_l138_13861

/-- The number of permutations of four digits with two pairs of repeated digits -/
def permutations_with_repetition : ℕ := 6

/-- The set of digits to be permuted -/
def digits : Finset ℕ := {2, 2, 3, 3}

/-- The theorem stating that the number of permutations of the given digits is 6 -/
theorem permutations_of_middle_digits :
  Finset.card (Finset.powersetCard 4 digits) = permutations_with_repetition :=
sorry

end NUMINAMATH_CALUDE_permutations_of_middle_digits_l138_13861


namespace NUMINAMATH_CALUDE_functional_equation_solution_l138_13808

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

/-- The main theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesFunctionalEquation f) :
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l138_13808


namespace NUMINAMATH_CALUDE_grid_50_25_toothpicks_l138_13865

/-- Calculates the number of toothpicks needed for a grid --/
def toothpicks_in_grid (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A grid of 50 by 25 toothpicks requires 2575 toothpicks --/
theorem grid_50_25_toothpicks :
  toothpicks_in_grid 50 25 = 2575 := by
  sorry

end NUMINAMATH_CALUDE_grid_50_25_toothpicks_l138_13865


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l138_13844

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ
  is_isosceles : True
  base1_longer : base1 > base2

-- Define the rotation of the trapezoid
def rotate_trapezoid (t : IsoscelesTrapezoid) : Solid :=
  sorry

-- Define the components of a solid
inductive SolidComponent
  | Cylinder
  | Cone
  | Frustum

-- Define a solid as a collection of components
def Solid := List SolidComponent

-- Theorem statement
theorem isosceles_trapezoid_rotation 
  (t : IsoscelesTrapezoid) : 
  rotate_trapezoid t = [SolidComponent.Cylinder, SolidComponent.Cone, SolidComponent.Cone] :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l138_13844
