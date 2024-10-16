import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l98_9837

theorem problem_solution : 
  (Real.sqrt 48 / Real.sqrt 3 - 2 * Real.sqrt (1/5) * Real.sqrt 30 + Real.sqrt 24 = 4) ∧
  ((2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l98_9837


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l98_9854

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 1 = 0) → (x₂^2 - 2*x₂ - 1 = 0) → (x₁ + x₂ - x₁*x₂ = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l98_9854


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l98_9847

theorem polygon_diagonals_sides (n : ℕ) : n > 2 →
  (n * (n - 3) / 2 = 2 * n) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l98_9847


namespace NUMINAMATH_CALUDE_difference_of_squares_l98_9819

theorem difference_of_squares (a : ℝ) : a^2 - 100 = (a + 10) * (a - 10) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l98_9819


namespace NUMINAMATH_CALUDE_initial_number_proof_l98_9824

theorem initial_number_proof : ∃ (n : ℕ), n = 427398 ∧ 
  (∃ (k : ℕ), n - 6 = 14 * k) ∧ 
  (∀ (m : ℕ), m < 6 → ¬∃ (j : ℕ), n - m = 14 * j) :=
by sorry

end NUMINAMATH_CALUDE_initial_number_proof_l98_9824


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l98_9810

-- Define the parabola
def parabola (x : ℝ) : ℝ := 4 * x^2

-- Define the tangent line at a point (a, 4a^2) on the parabola
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 8 * a * x - 4 * a^2

-- Define the condition for perpendicular tangents
def perpendicular_tangents (a b : ℝ) : Prop := 8 * a * 8 * b = -1

-- Theorem statement
theorem intersection_y_coordinate (a b : ℝ) : 
  a ≠ b → 
  perpendicular_tangents a b →
  ∃ x, tangent_line a x = tangent_line b x ∧ tangent_line a x = -1/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l98_9810


namespace NUMINAMATH_CALUDE_benny_baseball_gear_expense_l98_9815

/-- The amount Benny spent on baseball gear -/
def amount_spent (initial : ℕ) (left_over : ℕ) : ℕ :=
  initial - left_over

/-- Theorem stating that Benny spent 34 dollars on baseball gear -/
theorem benny_baseball_gear_expense :
  amount_spent 67 33 = 34 := by
  sorry

end NUMINAMATH_CALUDE_benny_baseball_gear_expense_l98_9815


namespace NUMINAMATH_CALUDE_solution_set_inequality_l98_9873

/-- Given a function f: ℝ → ℝ satisfying certain conditions,
    prove that the solution set of f(x) + 1 > 2023 * exp(x) is (-∞, 0) -/
theorem solution_set_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, (deriv f) x - f x < 1)
  (h2 : f 0 = 2022) :
  {x : ℝ | f x + 1 > 2023 * Real.exp x} = Set.Iio 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l98_9873


namespace NUMINAMATH_CALUDE_equality_condition_l98_9801

theorem equality_condition (x y z a b c : ℝ) :
  (Real.sqrt (x + a) + Real.sqrt (y + b) + Real.sqrt (z + c) =
   Real.sqrt (y + a) + Real.sqrt (z + b) + Real.sqrt (x + c)) ∧
  (Real.sqrt (y + a) + Real.sqrt (z + b) + Real.sqrt (x + c) =
   Real.sqrt (z + a) + Real.sqrt (x + b) + Real.sqrt (y + c)) →
  (x = y ∧ y = z) ∨ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equality_condition_l98_9801


namespace NUMINAMATH_CALUDE_quadratic_function_max_min_difference_l98_9838

-- Define the function f(x) = x^2 + bx + c
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the theorem
theorem quadratic_function_max_min_difference (b c : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → f b c x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ f b c x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → min ≤ f b c x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ f b c x = min) ∧
    max - min = 25) →
  b = -4 ∨ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_min_difference_l98_9838


namespace NUMINAMATH_CALUDE_remainder_108_112_mod_11_l98_9879

theorem remainder_108_112_mod_11 : (108 * 112) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_108_112_mod_11_l98_9879


namespace NUMINAMATH_CALUDE_quadratic_sum_l98_9806

/-- A quadratic function f(x) = ax^2 - bx + c passing through (1, -1) with vertex at (-1/2, -1/4) -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := λ x ↦ (a : ℝ) * x^2 - (b : ℝ) * x + (c : ℝ)

theorem quadratic_sum (a b c : ℤ) :
  (QuadraticFunction a b c 1 = -1) →
  (QuadraticFunction a b c (-1/2) = -1/4) →
  a + b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l98_9806


namespace NUMINAMATH_CALUDE_coworker_repair_ratio_l98_9849

/-- The ratio of phones a coworker fixes to the total number of damaged phones -/
theorem coworker_repair_ratio : 
  ∀ (initial_phones repaired_phones new_phones phones_per_person : ℕ),
    initial_phones = 15 →
    repaired_phones = 3 →
    new_phones = 6 →
    phones_per_person = 9 →
    (phones_per_person : ℚ) / ((initial_phones - repaired_phones + new_phones) : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_coworker_repair_ratio_l98_9849


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l98_9894

/-- Proves that for a rectangular plot with breadth 11 metres and length 10 metres more than its breadth, 
    the area of the plot divided by its breadth equals 21. -/
theorem rectangle_area_breadth_ratio : 
  ∀ (length breadth area : ℝ),
    breadth = 11 →
    length = breadth + 10 →
    area = length * breadth →
    area / breadth = 21 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l98_9894


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l98_9813

theorem polynomial_root_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2500*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  p + q + r = 0 →
  p * q + q * r + r * p = -2500 →
  p * q * r = -m →
  |p| + |q| + |r| = 100 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l98_9813


namespace NUMINAMATH_CALUDE_five_apples_ten_oranges_baskets_l98_9831

/-- Represents the number of different fruit baskets that can be made -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - 1

/-- Theorem stating that the number of different fruit baskets
    with 5 apples and 10 oranges is 65 -/
theorem five_apples_ten_oranges_baskets :
  fruitBaskets 5 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_five_apples_ten_oranges_baskets_l98_9831


namespace NUMINAMATH_CALUDE_students_voting_both_issues_l98_9876

theorem students_voting_both_issues 
  (total_students : ℕ) 
  (first_issue : ℕ) 
  (second_issue : ℕ) 
  (against_both : ℕ) 
  (h1 : total_students = 250)
  (h2 : first_issue = 171)
  (h3 : second_issue = 141)
  (h4 : against_both = 39) :
  first_issue + second_issue - (total_students - against_both) = 101 := by
sorry

end NUMINAMATH_CALUDE_students_voting_both_issues_l98_9876


namespace NUMINAMATH_CALUDE_travis_bowls_problem_l98_9898

/-- Represents the problem of calculating the number of bowls Travis initially had --/
theorem travis_bowls_problem :
  let base_fee : ℕ := 100
  let safe_bowl_pay : ℕ := 3
  let lost_bowl_fee : ℕ := 4
  let lost_bowls : ℕ := 12
  let broken_bowls : ℕ := 15
  let total_payment : ℕ := 1825

  ∃ (total_bowls safe_bowls : ℕ),
    total_bowls = safe_bowls + lost_bowls + broken_bowls ∧
    total_payment = base_fee + safe_bowl_pay * safe_bowls - lost_bowl_fee * (lost_bowls + broken_bowls) ∧
    total_bowls = 638 :=
by
  sorry

end NUMINAMATH_CALUDE_travis_bowls_problem_l98_9898


namespace NUMINAMATH_CALUDE_cubic_root_sum_l98_9864

theorem cubic_root_sum (p q r : ℝ) : 
  p + q + r = 4 →
  p * q + p * r + q * r = 1 →
  p * q * r = -6 →
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 22 - 213 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l98_9864


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l98_9827

theorem correct_quadratic_equation :
  ∃ (r₁ r₂ : ℝ), 
    (r₁ + r₂ = 9) ∧ 
    (r₁ * r₂ = 18) ∧ 
    (∃ (s₁ s₂ : ℝ), s₁ + s₂ = 5 - 1 ∧ s₁ + s₂ = 9) ∧
    (r₁ * r₂ = r₁ * r₂ - 9 * (r₁ + r₂) + 18) := by
  sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l98_9827


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l98_9851

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l98_9851


namespace NUMINAMATH_CALUDE_flour_remaining_l98_9870

theorem flour_remaining (initial_amount : ℝ) : 
  let first_removal_percent : ℝ := 60
  let second_removal_percent : ℝ := 25
  let remaining_after_first : ℝ := initial_amount * (100 - first_removal_percent) / 100
  let final_remaining : ℝ := remaining_after_first * (100 - second_removal_percent) / 100
  final_remaining = initial_amount * 30 / 100 := by sorry

end NUMINAMATH_CALUDE_flour_remaining_l98_9870


namespace NUMINAMATH_CALUDE_b_cubed_is_zero_l98_9871

theorem b_cubed_is_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_cubed_is_zero_l98_9871


namespace NUMINAMATH_CALUDE_geometric_sum_pebbles_l98_9877

theorem geometric_sum_pebbles (a : ℕ) (r : ℕ) (n : ℕ) (h1 : a = 1) (h2 : r = 2) (h3 : n = 10) :
  a * (r^n - 1) / (r - 1) = 1023 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_pebbles_l98_9877


namespace NUMINAMATH_CALUDE_solution_set_inequality_l98_9890

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (x + 1) < 0 ↔ -1 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l98_9890


namespace NUMINAMATH_CALUDE_distinct_collections_count_l98_9872

/-- Represents the collection of letters in MATHEMATICAL -/
def mathematical : Finset Char := {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

/-- Represents the vowels in MATHEMATICAL -/
def vowels : Finset Char := {'A', 'E', 'I'}

/-- Represents the consonants in MATHEMATICAL -/
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'L'}

/-- Function to count occurrences of a character in MATHEMATICAL -/
def count (c : Char) : Nat := (mathematical.filter (· = c)).card

/-- The number of distinct possible collections of letters in the bag -/
def distinct_collections : Nat := sorry

theorem distinct_collections_count : distinct_collections = 220 := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l98_9872


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l98_9858

/-- Given two points P and Q symmetric about the x-axis, prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ P Q : ℝ × ℝ, 
    P = (a - 1, 5) ∧ 
    Q = (2, b - 1) ∧ 
    P.1 = Q.1 ∧ 
    P.2 = -Q.2) →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l98_9858


namespace NUMINAMATH_CALUDE_hotel_expenditure_l98_9834

theorem hotel_expenditure (total_expenditure : ℕ) 
  (standard_spenders : ℕ) (standard_amount : ℕ) (extra_amount : ℕ) : 
  total_expenditure = 117 →
  standard_spenders = 8 →
  standard_amount = 12 →
  extra_amount = 8 →
  ∃ (n : ℕ), n = 9 ∧ 
    (standard_spenders * standard_amount + 
    (total_expenditure / n + extra_amount) = total_expenditure) :=
by sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l98_9834


namespace NUMINAMATH_CALUDE_derivative_of_cosine_linear_l98_9853

/-- Given a function f(x) = cos(2x - π/6), its derivative f'(x) = -2sin(2x - π/6) --/
theorem derivative_of_cosine_linear (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (2 * x - π / 6)
  deriv f x = -2 * Real.sin (2 * x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_cosine_linear_l98_9853


namespace NUMINAMATH_CALUDE_inequality_solution_l98_9807

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
axiom f_one_eq_zero : f 1 = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f (x - 2) ≥ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l98_9807


namespace NUMINAMATH_CALUDE_selling_price_calculation_l98_9859

def calculate_selling_price (purchase_price repair_cost transport_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_charges
  let profit := (total_cost * profit_percentage) / 100
  total_cost + profit

theorem selling_price_calculation :
  calculate_selling_price 13000 5000 1000 50 = 28500 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l98_9859


namespace NUMINAMATH_CALUDE_distance_between_points_l98_9878

/-- Prove the distance between two points given rowing speed, stream speed, and round trip time -/
theorem distance_between_points (v : ℝ) (s : ℝ) (t : ℝ) (h1 : v > s) (h2 : v > 0) (h3 : s > 0) (h4 : t > 0) :
  let d := (v * t * (v - s) * (v + s)) / (2 * v)
  d = 24 ∧ (d / (v - s) + d / (v + s) = t) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l98_9878


namespace NUMINAMATH_CALUDE_multiple_with_specific_remainder_l98_9805

theorem multiple_with_specific_remainder (x : ℕ) (hx : x > 0) 
  (hx_rem : x % 9 = 5) : 
  (∃ k : ℕ, k > 0 ∧ (k * x) % 9 = 2) ∧ 
  (∀ k : ℕ, k > 0 → (k * x) % 9 = 2 → k ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_specific_remainder_l98_9805


namespace NUMINAMATH_CALUDE_max_value_range_l98_9803

noncomputable def f (a b x : ℝ) : ℝ :=
  if x ≤ a then -(x + 1) * Real.exp x else b * x - 1

theorem max_value_range (a b : ℝ) :
  ∃ M : ℝ, (∀ x, f a b x ≤ M) ∧ (0 < M) ∧ (M ≤ Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_range_l98_9803


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l98_9850

-- Define the constants for the foci locations
def ellipse_focus : ℝ := 5
def hyperbola_focus : ℝ := 8

-- Define the theorem
theorem ellipse_hyperbola_product (c d : ℝ) : 
  (d^2 - c^2 = ellipse_focus^2) →   -- Condition for ellipse foci
  (c^2 + d^2 = hyperbola_focus^2) → -- Condition for hyperbola foci
  |c * d| = Real.sqrt ((39 * 89) / 4) := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l98_9850


namespace NUMINAMATH_CALUDE_weight_relationship_and_sum_l98_9897

/-- Given the weights of Haley, Verna, and Sherry, prove their relationship and combined weight -/
theorem weight_relationship_and_sum (haley_weight verna_weight sherry_weight : ℕ) : 
  haley_weight = 103 →
  verna_weight = haley_weight + 17 →
  verna_weight * 2 = sherry_weight →
  verna_weight + sherry_weight = 360 := by
  sorry

end NUMINAMATH_CALUDE_weight_relationship_and_sum_l98_9897


namespace NUMINAMATH_CALUDE_sum_of_three_integers_l98_9857

theorem sum_of_three_integers (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^4 →
  (a : ℕ) + b + c = 131 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_l98_9857


namespace NUMINAMATH_CALUDE_problem_statement_l98_9816

/-- s(n,m) is the number of positive integers in [n,m] that are coprime with m -/
def s (n m : ℕ) : ℕ := sorry

theorem problem_statement (m : ℕ) (hm : m ≥ 2) :
  (∀ n ∈ Finset.range (m - 1), (s n m : ℚ) / (m - n : ℚ) ≥ (s 1 m : ℚ) / m) →
  (2022^m + 1) % m^2 = 0 →
  m = 7 ∨ m = 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l98_9816


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l98_9863

theorem sum_of_x_and_y (x y : ℝ) (h : (2 : ℝ)^x = (18 : ℝ)^y ∧ (2 : ℝ)^x = (6 : ℝ)^(x*y)) :
  x + y = 0 ∨ x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l98_9863


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l98_9874

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ (b : ℝ), a + (5 * Complex.I) / (1 - 2 * Complex.I) = b * Complex.I) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l98_9874


namespace NUMINAMATH_CALUDE_john_needs_168_nails_l98_9868

/-- The number of nails needed for a house wall -/
def nails_needed (num_planks : ℕ) (nails_per_plank : ℕ) : ℕ :=
  num_planks * nails_per_plank

/-- Theorem: John needs 168 nails for the house wall -/
theorem john_needs_168_nails :
  nails_needed 42 4 = 168 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_168_nails_l98_9868


namespace NUMINAMATH_CALUDE_stratified_sampling_l98_9880

theorem stratified_sampling (first_grade : ℕ) (second_grade : ℕ) (sample_size : ℕ) (third_grade_sampled : ℕ) :
  first_grade = 24 →
  second_grade = 36 →
  sample_size = 20 →
  third_grade_sampled = 10 →
  ∃ (total_parts : ℕ) (third_grade : ℕ) (second_grade_sampled : ℕ),
    total_parts = first_grade + second_grade + third_grade ∧
    third_grade = 60 ∧
    second_grade_sampled = 6 ∧
    (third_grade : ℚ) / total_parts = (third_grade_sampled : ℚ) / sample_size ∧
    (second_grade : ℚ) / total_parts = (second_grade_sampled : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l98_9880


namespace NUMINAMATH_CALUDE_reservoir_D_largest_l98_9825

-- Define the initial amount of water (same for all reservoirs)
variable (a : ℝ)

-- Define the final amounts of water in each reservoir
def final_amount_A : ℝ := a * (1 + 0.10) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

-- Theorem stating that Reservoir D has the largest amount of water
theorem reservoir_D_largest (a : ℝ) (h : a > 0) : 
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end NUMINAMATH_CALUDE_reservoir_D_largest_l98_9825


namespace NUMINAMATH_CALUDE_rabbit_age_order_l98_9899

-- Define the rabbit types
inductive Rabbit
  | BlueEyed
  | Gray
  | Black
  | RedEyed

-- Define the age relation
def older_than (a b : Rabbit) : Prop := sorry

-- Define the conditions
axiom blue_not_eldest : ¬ (∀ r : Rabbit, older_than Rabbit.BlueEyed r ∨ r = Rabbit.BlueEyed)
axiom gray_not_youngest : ∃ r : Rabbit, older_than r Rabbit.Gray
axiom black_older_than_red : older_than Rabbit.Black Rabbit.RedEyed
axiom black_younger_than_gray : older_than Rabbit.Gray Rabbit.Black
axiom red_not_youngest : ∃ r : Rabbit, older_than Rabbit.RedEyed r

-- Theorem to prove
theorem rabbit_age_order :
  (∀ r : Rabbit, older_than Rabbit.Gray r ∨ r = Rabbit.Gray) ∧
  older_than Rabbit.Gray Rabbit.Black ∧
  older_than Rabbit.Black Rabbit.RedEyed ∧
  older_than Rabbit.RedEyed Rabbit.BlueEyed ∧
  (∀ r : Rabbit, older_than r Rabbit.BlueEyed ∨ r = Rabbit.BlueEyed) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_age_order_l98_9899


namespace NUMINAMATH_CALUDE_max_value_of_f_inequality_with_sum_constraint_l98_9865

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ (s : ℝ), s = 3 ∧ ∀ (x : ℝ), f x ≤ s := by sorry

-- Theorem for the inequality
theorem inequality_with_sum_constraint (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) : 
  a^2 + b^2 + c^2 ≥ 3 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_inequality_with_sum_constraint_l98_9865


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l98_9856

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2*n + 1) = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l98_9856


namespace NUMINAMATH_CALUDE_sequence_sum_l98_9840

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  a 1 = 1 → 
  (∀ n : ℕ, 2 * S n = a (n + 1) - 1) → 
  a 3 + a 4 + a 5 = 117 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l98_9840


namespace NUMINAMATH_CALUDE_min_value_theorem_l98_9862

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) (hab : a * b = 1/4) :
  ∃ (min_val : ℝ), min_val = 4 + 4 * Real.sqrt 2 / 3 ∧
  ∀ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 →
  1 / (1 - x) + 2 / (1 - y) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l98_9862


namespace NUMINAMATH_CALUDE_expression_value_l98_9885

theorem expression_value (x : ℝ) (h : x^2 - 2*x = 3) : 2*x^2 - 4*x + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l98_9885


namespace NUMINAMATH_CALUDE_ellipse_equation_l98_9884

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > 0, b > 0),
    if the line 2x + y - 2 = 0 passes through its upper vertex and right focus,
    then the equation of the ellipse is x²/5 + y²/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ 2*x + y = 2 ∧
   ((x = a ∧ y = 0) ∨ (x = 0 ∧ y = b))) →
  a^2 = 5 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l98_9884


namespace NUMINAMATH_CALUDE_max_trailing_zeros_1003_l98_9829

/-- Three natural numbers whose sum is 1003 -/
def SumTo1003 (a b c : ℕ) : Prop := a + b + c = 1003

/-- The number of trailing zeros in a natural number -/
def TrailingZeros (n : ℕ) : ℕ := sorry

/-- The product of three natural numbers -/
def ProductOfThree (a b c : ℕ) : ℕ := a * b * c

/-- Theorem stating that the maximum number of trailing zeros in the product of three natural numbers summing to 1003 is 7 -/
theorem max_trailing_zeros_1003 :
  ∀ a b c : ℕ, SumTo1003 a b c →
  ∀ n : ℕ, n = TrailingZeros (ProductOfThree a b c) →
  n ≤ 7 ∧ ∃ x y z : ℕ, SumTo1003 x y z ∧ TrailingZeros (ProductOfThree x y z) = 7 :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_1003_l98_9829


namespace NUMINAMATH_CALUDE_scientific_notation_of_8500_billion_l98_9866

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The value in yuan -/
def value : ℝ := 8500000000000

/-- The scientific notation representation of the value -/
def scientificForm : ScientificNotation := toScientificNotation value

/-- Theorem stating that the scientific notation of 8500 billion yuan is 8.5 × 10^11 -/
theorem scientific_notation_of_8500_billion :
  scientificForm.coefficient = 8.5 ∧ scientificForm.exponent = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8500_billion_l98_9866


namespace NUMINAMATH_CALUDE_forty_three_base7_equals_thirty_four_base9_l98_9882

/-- Converts a number from base-7 to base-10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base-10 to a given base -/
def base10ToBase (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem forty_three_base7_equals_thirty_four_base9 :
  let n := 43
  let base7Value := base7ToBase10 n
  let reversedDigits := reverseDigits n
  base10ToBase base7Value 9 = reversedDigits := by sorry

end NUMINAMATH_CALUDE_forty_three_base7_equals_thirty_four_base9_l98_9882


namespace NUMINAMATH_CALUDE_negative_one_squared_and_one_are_opposite_l98_9883

-- Define opposite numbers
def are_opposite (a b : ℤ) : Prop := a + b = 0

-- Theorem statement
theorem negative_one_squared_and_one_are_opposite : 
  are_opposite (-(1^2)) 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_squared_and_one_are_opposite_l98_9883


namespace NUMINAMATH_CALUDE_val_initial_nickels_l98_9892

/-- The number of nickels Val initially had -/
def initial_nickels : ℕ := 20

/-- The value of a nickel in cents -/
def nickel_value : ℚ := 5/100

/-- The value of a dime in cents -/
def dime_value : ℚ := 10/100

/-- The total value in dollars after tripling the number of nickels -/
def total_value : ℚ := 9

theorem val_initial_nickels :
  ∀ n : ℕ,
  (n : ℚ) * nickel_value * 3 + (3 * n : ℚ) * dime_value = total_value →
  n = initial_nickels :=
by
  sorry

end NUMINAMATH_CALUDE_val_initial_nickels_l98_9892


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l98_9848

theorem cylinder_surface_area (h : ℝ) (d : ℝ) (cylinder_height : h = 2) (sphere_diameter : d = 2 * Real.sqrt 6) :
  let r := Real.sqrt (((d / 2) ^ 2 - (h / 2) ^ 2))
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = (10 + 4 * Real.sqrt 5) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l98_9848


namespace NUMINAMATH_CALUDE_board_numbers_sum_l98_9842

theorem board_numbers_sum (a b c : ℝ) : 
  ({a, b, c} : Set ℝ) = {a^2 + 2*b*c, b^2 + 2*c*a, c^2 + 2*a*b} → 
  a + b + c = 0 ∨ a + b + c = 1 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_sum_l98_9842


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l98_9826

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (4 * a 1 = a m) →
  (a m)^2 = a 1 * a n →
  (m + n = 6) →
  (1 / m + 4 / n ≥ 3 / 2) ∧
  (∃ m₀ n₀ : ℕ, m₀ + n₀ = 6 ∧ 1 / m₀ + 4 / n₀ = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l98_9826


namespace NUMINAMATH_CALUDE_tangent_and_sin_cos_product_l98_9833

theorem tangent_and_sin_cos_product (α : Real) 
  (h : Real.tan (π / 4 + α) = 3) : 
  Real.tan α = 1 / 2 ∧ Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_sin_cos_product_l98_9833


namespace NUMINAMATH_CALUDE_rory_tank_water_l98_9811

/-- Calculates the final amount of water in Rory's tank after a rainstorm --/
def final_water_amount (initial_water : ℝ) (inflow_rate_1 inflow_rate_2 : ℝ) 
  (leak_rate : ℝ) (evap_rate_1 evap_rate_2 : ℝ) (evap_reduction : ℝ) 
  (duration_1 duration_2 : ℝ) : ℝ :=
  let total_inflow := inflow_rate_1 * duration_1 + inflow_rate_2 * duration_2
  let total_leak := leak_rate * (duration_1 + duration_2)
  let total_evap := (evap_rate_1 * duration_1 + evap_rate_2 * duration_2) * (1 - evap_reduction)
  initial_water + total_inflow - total_leak - total_evap

/-- Theorem stating the final amount of water in Rory's tank --/
theorem rory_tank_water : 
  final_water_amount 100 2 3 0.5 0.2 0.1 0.75 45 45 = 276.625 := by
  sorry

end NUMINAMATH_CALUDE_rory_tank_water_l98_9811


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l98_9844

/-- The total surface area of a pyramid formed from a cube -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) : 
  let cube_edge := a
  let base_side := a * Real.sqrt 2 / 2
  let slant_height := 3 * a * Real.sqrt 2 / 4
  let lateral_area := 4 * (1/2 * base_side * slant_height)
  let base_area := base_side ^ 2
  lateral_area + base_area = 2 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l98_9844


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l98_9846

theorem binomial_coefficient_ratio :
  ∀ a₀ a₁ a₂ a₃ a₄ a₅ : ℝ,
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -122/121 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l98_9846


namespace NUMINAMATH_CALUDE_triangle_construction_from_nagel_point_vertex_and_altitude_foot_l98_9836

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by its vertices -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- The Nagel point of a triangle -/
def nagelPoint (t : Triangle) : Point2D := sorry

/-- The foot of the altitude from a vertex -/
def altitudeFoot (t : Triangle) (v : Point2D) : Point2D := sorry

/-- Theorem: Given a Nagel point, a vertex, and the foot of the altitude from that vertex,
    a triangle can be constructed -/
theorem triangle_construction_from_nagel_point_vertex_and_altitude_foot
  (N : Point2D) (B : Point2D) (F : Point2D) :
  ∃ (t : Triangle), nagelPoint t = N ∧ t.B = B ∧ altitudeFoot t B = F :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_from_nagel_point_vertex_and_altitude_foot_l98_9836


namespace NUMINAMATH_CALUDE_average_equals_expression_l98_9869

theorem average_equals_expression (x : ℝ) : 
  (1/3) * ((3*x + 8) + (7*x + 3) + (4*x + 9)) = 5*x - 10 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_equals_expression_l98_9869


namespace NUMINAMATH_CALUDE_rectangle_length_l98_9808

/-- The length of a rectangle with width 4 cm and area equal to a square with sides 4 cm -/
theorem rectangle_length (width : ℝ) (square_side : ℝ) (length : ℝ) : 
  width = 4 →
  square_side = 4 →
  length * width = square_side * square_side →
  length = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l98_9808


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l98_9804

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 210 →
  B = 330 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l98_9804


namespace NUMINAMATH_CALUDE_flour_cost_l98_9823

/-- Represents the cost of ingredients and cake slices --/
structure CakeCost where
  flour : ℝ
  sugar : ℝ
  butter : ℝ
  eggs : ℝ
  total : ℝ
  sliceCount : ℕ
  sliceCost : ℝ
  dogAteCost : ℝ

/-- Theorem stating that given the total cost of ingredients and the cost of what the dog ate, 
    the cost of flour is $4 --/
theorem flour_cost (c : CakeCost) 
  (h1 : c.sugar = 2)
  (h2 : c.butter = 2.5)
  (h3 : c.eggs = 0.5)
  (h4 : c.total = c.flour + c.sugar + c.butter + c.eggs)
  (h5 : c.sliceCount = 6)
  (h6 : c.sliceCost = c.total / c.sliceCount)
  (h7 : c.dogAteCost = 6)
  (h8 : c.dogAteCost = 4 * c.sliceCost) :
  c.flour = 4 := by
  sorry

end NUMINAMATH_CALUDE_flour_cost_l98_9823


namespace NUMINAMATH_CALUDE_power_of_power_l98_9822

theorem power_of_power : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l98_9822


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l98_9820

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l98_9820


namespace NUMINAMATH_CALUDE_unique_intersection_l98_9800

/-- 
Given two functions f(x) = ax² + 2x + 3 and g(x) = -2x - 3, 
this theorem states that these functions intersect at exactly one point 
if and only if a = 2/3.
-/
theorem unique_intersection (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 3 = -2 * x - 3) ↔ a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l98_9800


namespace NUMINAMATH_CALUDE_cube_rotation_different_face_l98_9889

-- Define a cube face
inductive CubeFace
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

-- Define a cube position
structure CubePosition :=
  (location : ℝ × ℝ × ℝ)
  (bottom_face : CubeFace)

-- Define a cube rotation
inductive CubeRotation
  | RollForward
  | RollBackward
  | RollLeft
  | RollRight

-- Define a function that applies a rotation to a cube position
def apply_rotation (pos : CubePosition) (rot : CubeRotation) : CubePosition :=
  sorry

-- Define the theorem
theorem cube_rotation_different_face :
  ∃ (initial_pos final_pos : CubePosition) (rotations : List CubeRotation),
    (initial_pos.location = final_pos.location) ∧
    (initial_pos.bottom_face ≠ final_pos.bottom_face) ∧
    (final_pos = rotations.foldl apply_rotation initial_pos) :=
  sorry

end NUMINAMATH_CALUDE_cube_rotation_different_face_l98_9889


namespace NUMINAMATH_CALUDE_intersection_M_N_l98_9832

def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l98_9832


namespace NUMINAMATH_CALUDE_unique_pair_existence_l98_9812

theorem unique_pair_existence (n : ℕ+) :
  ∃! (k l : ℕ), 0 ≤ l ∧ l < k ∧ n = (k * (k - 1)) / 2 + l := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_existence_l98_9812


namespace NUMINAMATH_CALUDE_percentage_problem_l98_9821

theorem percentage_problem (P : ℝ) : 
  (20 / 100) * 680 = (P / 100) * 140 + 80 → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l98_9821


namespace NUMINAMATH_CALUDE_inequality_solution_set_l98_9843

theorem inequality_solution_set (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) 3) = {x | (x - 3) * (x + 2) < 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l98_9843


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l98_9852

/-- The line equation y = kx + 2 is tangent to the ellipse 2x^2 + 8y^2 = 8 exactly once if and only if k^2 = 3/4 -/
theorem line_tangent_to_ellipse (k : ℝ) : 
  (∃! x y : ℝ, y = k * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) ↔ k^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l98_9852


namespace NUMINAMATH_CALUDE_fraction_equality_l98_9860

theorem fraction_equality : (81081 : ℝ)^4 / (27027 : ℝ)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l98_9860


namespace NUMINAMATH_CALUDE_polygon_sides_l98_9887

theorem polygon_sides (n : ℕ) (h : n > 2) : 
  (2 : ℚ) / 9 * ((n - 2) * 180) = 360 → n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l98_9887


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l98_9802

theorem min_value_reciprocal_sum (x : ℝ) (h1 : 0 < x) (h2 : x < 3) :
  (1 / x) + (1 / (3 - x)) ≥ 4 / 3 ∧
  ((1 / x) + (1 / (3 - x)) = 4 / 3 ↔ x = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l98_9802


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l98_9875

theorem pencil_buyers_difference : ∀ (pencil_cost : ℕ) 
  (seventh_graders : ℕ) (sixth_graders : ℕ),
  pencil_cost > 0 →
  pencil_cost * seventh_graders = 143 →
  pencil_cost * sixth_graders = 195 →
  sixth_graders ≤ 30 →
  sixth_graders - seventh_graders = 4 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l98_9875


namespace NUMINAMATH_CALUDE_cookie_shop_problem_l98_9895

def num_cookie_flavors : ℕ := 7
def num_milk_types : ℕ := 4
def total_products : ℕ := 4

def ways_charlie_buys (k : ℕ) : ℕ := Nat.choose (num_cookie_flavors + num_milk_types) k

def ways_delta_buys_distinct (k : ℕ) : ℕ := Nat.choose num_cookie_flavors k

def ways_delta_buys_with_repeats (k : ℕ) : ℕ :=
  if k = 1 then num_cookie_flavors
  else if k = 2 then ways_delta_buys_distinct 2 + num_cookie_flavors
  else if k = 3 then ways_delta_buys_distinct 3 + num_cookie_flavors * (num_cookie_flavors - 1) + num_cookie_flavors
  else if k = 4 then ways_delta_buys_distinct 4 + num_cookie_flavors * (num_cookie_flavors - 1) + 
                     (num_cookie_flavors * (num_cookie_flavors - 1)) / 2 + num_cookie_flavors
  else 0

def total_ways : ℕ :=
  (ways_charlie_buys 4) +
  (ways_charlie_buys 3 * ways_delta_buys_with_repeats 1) +
  (ways_charlie_buys 2 * ways_delta_buys_with_repeats 2) +
  (ways_charlie_buys 1 * ways_delta_buys_with_repeats 3) +
  (ways_delta_buys_with_repeats 4)

theorem cookie_shop_problem : total_ways = 4054 := by sorry

end NUMINAMATH_CALUDE_cookie_shop_problem_l98_9895


namespace NUMINAMATH_CALUDE_system_solutions_correct_l98_9818

theorem system_solutions_correct : 
  (∃ (x y : ℝ), 3*x - y = -1 ∧ x + 2*y = 9 ∧ x = 1 ∧ y = 4) ∧
  (∃ (x y : ℝ), x/4 + y/3 = 4/3 ∧ 5*(x - 9) = 4*(y - 13/4) ∧ x = 6 ∧ y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l98_9818


namespace NUMINAMATH_CALUDE_world_cup_viewers_scientific_notation_l98_9841

/-- Expresses a number in millions as scientific notation -/
def scientific_notation_millions (x : ℝ) : ℝ × ℤ :=
  (x, 7)

theorem world_cup_viewers_scientific_notation :
  scientific_notation_millions 70.62 = (7.062, 7) := by
  sorry

end NUMINAMATH_CALUDE_world_cup_viewers_scientific_notation_l98_9841


namespace NUMINAMATH_CALUDE_tenth_term_value_l98_9855

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_property : a 1 + a 3 + a 5 = 9
  product_property : a 3 * (a 4)^2 = 27

/-- The 10th term of the arithmetic sequence is either -39 or 30 -/
theorem tenth_term_value (seq : ArithmeticSequence) : seq.a 10 = -39 ∨ seq.a 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l98_9855


namespace NUMINAMATH_CALUDE_second_equation_value_l98_9830

theorem second_equation_value (x y : ℝ) 
  (eq1 : 2 * x + y = 26) 
  (eq2 : (x + y) / 3 = 4) : 
  x + 2 * y = 10 := by
sorry

end NUMINAMATH_CALUDE_second_equation_value_l98_9830


namespace NUMINAMATH_CALUDE_molecular_weight_of_C2H5Cl2O2_l98_9817

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of C2H5Cl2O2 in g/mol -/
def molecular_weight : ℝ := 2 * carbon_weight + 5 * hydrogen_weight + 2 * chlorine_weight + 2 * oxygen_weight

/-- Theorem stating that the molecular weight of C2H5Cl2O2 is 132.96 g/mol -/
theorem molecular_weight_of_C2H5Cl2O2 : molecular_weight = 132.96 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_of_C2H5Cl2O2_l98_9817


namespace NUMINAMATH_CALUDE_aku_birthday_friends_l98_9861

/-- Given the conditions of Aku's birthday party, prove the number of friends invited. -/
theorem aku_birthday_friends (packages : Nat) (cookies_per_package : Nat) (cookies_per_child : Nat) :
  packages = 3 →
  cookies_per_package = 25 →
  cookies_per_child = 15 →
  (packages * cookies_per_package) / cookies_per_child - 1 = 4 := by
  sorry

#eval (3 * 25) / 15 - 1  -- Expected output: 4

end NUMINAMATH_CALUDE_aku_birthday_friends_l98_9861


namespace NUMINAMATH_CALUDE_card_shop_problem_l98_9814

theorem card_shop_problem :
  ∃ (x y : ℕ), 1.25 * (x : ℝ) + 1.75 * (y : ℝ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_card_shop_problem_l98_9814


namespace NUMINAMATH_CALUDE_relationship_abc_l98_9835

theorem relationship_abc : 
  let a : ℝ := (0.3 : ℝ)^3
  let b : ℝ := 3^(0.3 : ℝ)
  let c : ℝ := (0.2 : ℝ)^3
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l98_9835


namespace NUMINAMATH_CALUDE_well_depth_rope_length_l98_9893

/-- 
Given a well of unknown depth and a rope of unknown length, prove that if:
1) Folding the rope three times results in a length equal to the well's depth plus 4 feet
2) Folding the rope four times results in a length equal to the well's depth plus 1 foot
Then the relationship between the rope length (h) and well depth (x) is:
{ h/3 = x + 4
{ h/4 = x + 1
-/
theorem well_depth_rope_length (h x : ℝ) : 
  (h / 3 = x + 4 ∧ h / 4 = x + 1) ↔ 
  (∃ (depth : ℝ) (rope_length : ℝ), 
    depth = x ∧ 
    rope_length = h ∧
    rope_length / 3 = depth + 4 ∧ 
    rope_length / 4 = depth + 1) := by
sorry

end NUMINAMATH_CALUDE_well_depth_rope_length_l98_9893


namespace NUMINAMATH_CALUDE_total_games_in_season_l98_9809

def total_teams : ℕ := 200
def num_sub_leagues : ℕ := 10
def teams_per_sub_league : ℕ := 20
def regular_season_matches : ℕ := 8
def teams_to_intermediate : ℕ := 5
def teams_to_playoff : ℕ := 2

def regular_season_games (n : ℕ) : ℕ := n * (n - 1) / 2 * regular_season_matches

def intermediate_round_games (n : ℕ) : ℕ := n * (n - 1) / 2

def playoff_round_games (n : ℕ) : ℕ := (n * (n - 1) / 2 - num_sub_leagues * (num_sub_leagues - 1) / 2) * 2

theorem total_games_in_season :
  regular_season_games teams_per_sub_league * num_sub_leagues +
  intermediate_round_games (teams_to_intermediate * num_sub_leagues) +
  playoff_round_games (teams_to_playoff * num_sub_leagues) = 16715 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l98_9809


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l98_9828

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a line in polar coordinates -/
def pointOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  l.equation p.ρ p.θ

/-- Checks if a line is parallel to the polar axis -/
def parallelToPolarAxis (l : PolarLine) : Prop :=
  ∀ ρ θ, l.equation ρ θ ↔ ∃ k, ρ * Real.sin θ = k

theorem line_through_point_parallel_to_polar_axis 
  (p : PolarPoint) 
  (h_p : p.ρ = 2 ∧ p.θ = Real.pi / 6) :
  ∃ l : PolarLine, 
    pointOnLine p l ∧ 
    parallelToPolarAxis l ∧
    (∀ ρ θ, l.equation ρ θ ↔ ρ * Real.sin θ = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l98_9828


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l98_9867

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_n as the sum of the first n terms,
    prove that S_4 / a_2 = 15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  q = 2 →  -- Given condition
  S 4 / a 2 = 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l98_9867


namespace NUMINAMATH_CALUDE_condition_relationship_l98_9845

theorem condition_relationship (x : ℝ) :
  (x > 1/3 → 1/x < 3) ∧ ¬(1/x < 3 → x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l98_9845


namespace NUMINAMATH_CALUDE_bd_squared_equals_25_l98_9896

theorem bd_squared_equals_25 
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2*a - 3*b + c + 4*d = 17)
  : (b - d)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bd_squared_equals_25_l98_9896


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l98_9839

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 13*x + 40 = (x + d) * (x + e)) →
  (∀ x, x^2 - 19*x + 88 = (x - e) * (x - f)) →
  d + e + f = 24 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l98_9839


namespace NUMINAMATH_CALUDE_root_value_implies_m_l98_9886

theorem root_value_implies_m (m : ℝ) : (∃ x : ℝ, x^2 + m*x - 3 = 0 ∧ x = 1) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_value_implies_m_l98_9886


namespace NUMINAMATH_CALUDE_trajectory_intersection_fixed_point_l98_9888

/-- The trajectory of a point equidistant from a fixed point and a fixed line -/
def Trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- A line not perpendicular to the x-axis -/
structure Line where
  t : ℝ
  m : ℝ
  h : t ≠ 0

/-- The condition that a line intersects the trajectory at two distinct points -/
def intersects_trajectory (l : Line) : Prop :=
  ∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ Trajectory ∧ q ∈ Trajectory ∧
    p.1 = l.t * p.2 + l.m ∧ q.1 = l.t * q.2 + l.m

/-- The condition that the x-axis is the angle bisector of ∠PBQ -/
def x_axis_bisects (l : Line) : Prop :=
  ∀ p q : ℝ × ℝ, p ≠ q → p ∈ Trajectory → q ∈ Trajectory →
    p.1 = l.t * p.2 + l.m → q.1 = l.t * q.2 + l.m →
    p.2 / (p.1 + 3) + q.2 / (q.1 + 3) = 0

/-- The main theorem -/
theorem trajectory_intersection_fixed_point :
  ∀ l : Line, intersects_trajectory l → x_axis_bisects l →
    l.m = 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_intersection_fixed_point_l98_9888


namespace NUMINAMATH_CALUDE_power_of_64_l98_9891

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 :=
by
  have h : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l98_9891


namespace NUMINAMATH_CALUDE_sum_of_nine_zero_seven_digits_l98_9881

/-- A function that checks if a real number uses only digits 0 and 7 in base 10 --/
def uses_only_0_and_7 (a : ℝ) : Prop := sorry

/-- Theorem stating that any real number can be expressed as the sum of nine numbers,
    each using only digits 0 and 7 in base 10 --/
theorem sum_of_nine_zero_seven_digits (x : ℝ) : 
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ), 
    x = a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ ∧ 
    uses_only_0_and_7 a₁ ∧ uses_only_0_and_7 a₂ ∧ uses_only_0_and_7 a₃ ∧ 
    uses_only_0_and_7 a₄ ∧ uses_only_0_and_7 a₅ ∧ uses_only_0_and_7 a₆ ∧ 
    uses_only_0_and_7 a₇ ∧ uses_only_0_and_7 a₈ ∧ uses_only_0_and_7 a₉ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nine_zero_seven_digits_l98_9881
