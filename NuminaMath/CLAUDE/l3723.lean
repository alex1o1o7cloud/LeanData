import Mathlib

namespace NUMINAMATH_CALUDE_johns_croissants_l3723_372317

theorem johns_croissants :
  ∀ (c : ℕ) (k : ℕ),
  c + k = 5 →
  (88 * c + 44 * k) % 100 = 0 →
  c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_croissants_l3723_372317


namespace NUMINAMATH_CALUDE_sum_mod_nine_l3723_372390

theorem sum_mod_nine : (1234 + 1235 + 1236 + 1237 + 1238) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l3723_372390


namespace NUMINAMATH_CALUDE_fuel_consumption_statements_correct_l3723_372348

/-- Represents the fuel consumption data for a car journey -/
structure FuelConsumptionData where
  initial_fuel : ℝ
  distance_interval : ℝ
  fuel_decrease_per_interval : ℝ
  total_distance : ℝ

/-- Theorem stating the correctness of all fuel consumption statements -/
theorem fuel_consumption_statements_correct
  (data : FuelConsumptionData)
  (h_initial : data.initial_fuel = 45)
  (h_interval : data.distance_interval = 50)
  (h_decrease : data.fuel_decrease_per_interval = 4)
  (h_total : data.total_distance = 500) :
  (data.initial_fuel = 45) ∧
  ((data.fuel_decrease_per_interval / data.distance_interval) * 100 = 8) ∧
  (∀ x y : ℝ, y = data.initial_fuel - (data.fuel_decrease_per_interval / data.distance_interval) * x) ∧
  (data.initial_fuel - (data.fuel_decrease_per_interval / data.distance_interval) * data.total_distance = 5) :=
by sorry


end NUMINAMATH_CALUDE_fuel_consumption_statements_correct_l3723_372348


namespace NUMINAMATH_CALUDE_magic_square_sum_l3723_372339

theorem magic_square_sum (b c d e g h : ℕ) : 
  b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ g > 0 ∧ h > 0 →
  30 * b * c = d * e * 3 →
  30 * b * c = g * h * 3 →
  30 * b * c = 30 * e * 3 →
  30 * b * c = b * e * h →
  30 * b * c = c * 3 * 3 →
  30 * b * c = 30 * e * g →
  30 * b * c = c * e * 3 →
  (∃ g₁ g₂ : ℕ, g = g₁ ∨ g = g₂) →
  g₁ + g₂ = 25 :=
by sorry

end NUMINAMATH_CALUDE_magic_square_sum_l3723_372339


namespace NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l3723_372301

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_iff_b_zero (a b c : ℝ) :
  is_even (quadratic a b c) ↔ b = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_even_iff_b_zero_l3723_372301


namespace NUMINAMATH_CALUDE_mean_equality_problem_l3723_372371

theorem mean_equality_problem (y : ℝ) : 
  (5 + 8 + 17) / 3 = (12 + y) / 2 → y = 8 := by sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l3723_372371


namespace NUMINAMATH_CALUDE_smallest_w_l3723_372369

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (13^2) (936 * w) → 
  ∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (13^2) (936 * v) → 
    w ≤ v → 
  w = 156 := by sorry

end NUMINAMATH_CALUDE_smallest_w_l3723_372369


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l3723_372300

theorem max_sum_on_circle (x y : ℕ) : x^2 + y^2 = 64 → x + y ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l3723_372300


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3723_372342

theorem inequality_system_solution (x : ℝ) :
  (-9 * x^2 + 12 * x + 5 > 0) ∧ (3 * x - 1 < 0) ↔ x < -1/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3723_372342


namespace NUMINAMATH_CALUDE_vacation_cost_splitting_l3723_372360

/-- Prove that the difference between what Tom and Dorothy owe Sammy is 20 dollars -/
theorem vacation_cost_splitting (tom_paid dorothy_paid sammy_paid t d : ℚ) : 
  tom_paid = 105 →
  dorothy_paid = 125 →
  sammy_paid = 175 →
  (tom_paid + dorothy_paid + sammy_paid) / 3 = tom_paid + t →
  (tom_paid + dorothy_paid + sammy_paid) / 3 = dorothy_paid + d →
  t - d = 20 := by
sorry


end NUMINAMATH_CALUDE_vacation_cost_splitting_l3723_372360


namespace NUMINAMATH_CALUDE_mindy_emails_l3723_372337

theorem mindy_emails (e m : ℕ) (h1 : e = 9 * m - 7) (h2 : e + m = 93) : e = 83 := by
  sorry

end NUMINAMATH_CALUDE_mindy_emails_l3723_372337


namespace NUMINAMATH_CALUDE_karlson_max_candies_l3723_372396

/-- The number of vertices in the complete graph -/
def n : ℕ := 29

/-- The maximum number of candies Karlson could eat -/
def max_candies : ℕ := 406

/-- Theorem stating the maximum number of candies Karlson could eat -/
theorem karlson_max_candies :
  (n * (n - 1)) / 2 = max_candies := by
  sorry

end NUMINAMATH_CALUDE_karlson_max_candies_l3723_372396


namespace NUMINAMATH_CALUDE_min_distinct_integers_for_progressions_l3723_372329

/-- A sequence of integers forms a geometric progression of length 5 -/
def is_geometric_progression (seq : Fin 5 → ℤ) : Prop :=
  ∃ (b q : ℤ), ∀ i : Fin 5, seq i = b * q ^ (i : ℕ)

/-- A sequence of integers forms an arithmetic progression of length 5 -/
def is_arithmetic_progression (seq : Fin 5 → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ i : Fin 5, seq i = a + (i : ℕ) * d

/-- The minimum number of distinct integers needed for both progressions -/
def min_distinct_integers : ℕ := 6

/-- Theorem stating the minimum number of distinct integers needed -/
theorem min_distinct_integers_for_progressions :
  ∀ (S : Finset ℤ),
  (∃ (seq_gp : Fin 5 → ℤ), (∀ i, seq_gp i ∈ S) ∧ is_geometric_progression seq_gp) ∧
  (∃ (seq_ap : Fin 5 → ℤ), (∀ i, seq_ap i ∈ S) ∧ is_arithmetic_progression seq_ap) →
  S.card ≥ min_distinct_integers :=
sorry

end NUMINAMATH_CALUDE_min_distinct_integers_for_progressions_l3723_372329


namespace NUMINAMATH_CALUDE_correct_equations_l3723_372319

/-- Represents the money held by a person -/
structure Money where
  amount : ℚ
  deriving Repr

/-- The problem setup -/
def problem_setup (a b : Money) : Prop :=
  (a.amount + (1/2) * b.amount = 50) ∧
  ((2/3) * a.amount + b.amount = 50)

/-- The theorem to prove -/
theorem correct_equations (a b : Money) :
  problem_setup a b ↔
  (a.amount + (1/2) * b.amount = 50 ∧ (2/3) * a.amount + b.amount = 50) :=
by sorry

end NUMINAMATH_CALUDE_correct_equations_l3723_372319


namespace NUMINAMATH_CALUDE_lcm_6_15_l3723_372328

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_6_15_l3723_372328


namespace NUMINAMATH_CALUDE_max_profit_at_nine_profit_function_correct_max_profit_at_nine_explicit_l3723_372380

-- Define the profit function
def profit (x : ℝ) : ℝ := x^3 - 30*x^2 + 288*x - 864

-- Define the theorem
theorem max_profit_at_nine :
  ∀ x ∈ Set.Icc 9 11,
    profit x ≤ profit 9 ∧
    profit 9 = 27 := by
  sorry

-- Define the selling price range
def selling_price_range : Set ℝ := Set.Icc 9 11

-- Define the annual sales volume function
def annual_sales (x : ℝ) : ℝ := (12 - x)^2

-- State that the profit function is correct
theorem profit_function_correct :
  ∀ x ∈ selling_price_range,
    profit x = (x - 6) * annual_sales x := by
  sorry

-- State that the maximum profit occurs at x = 9
theorem max_profit_at_nine_explicit :
  ∃ x ∈ selling_price_range,
    ∀ y ∈ selling_price_range,
      profit y ≤ profit x ∧
      x = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_nine_profit_function_correct_max_profit_at_nine_explicit_l3723_372380


namespace NUMINAMATH_CALUDE_number_of_apricot_trees_apricot_trees_count_l3723_372354

/-- Proves that the number of apricot trees is 135, given the conditions stated in the problem. -/
theorem number_of_apricot_trees : ℕ → Prop :=
  fun n : ℕ =>
    (∃ peach_trees : ℕ,
      peach_trees = 300 ∧
      peach_trees = 2 * n + 30) →
    n = 135

/-- The main theorem stating that there are 135 apricot trees. -/
theorem apricot_trees_count : ∃ n : ℕ, number_of_apricot_trees n :=
  sorry

end NUMINAMATH_CALUDE_number_of_apricot_trees_apricot_trees_count_l3723_372354


namespace NUMINAMATH_CALUDE_matrix_power_2023_l3723_372359

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 : A ^ 2023 = !![1, 0; 4046, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l3723_372359


namespace NUMINAMATH_CALUDE_bom_watermelon_seeds_l3723_372374

/-- Given the number of watermelon seeds for Bom, Gwi, and Yeon, prove that Bom has 300 seeds. -/
theorem bom_watermelon_seeds :
  ∀ (bom gwi yeon : ℕ),
  gwi = bom + 40 →
  yeon = 3 * gwi →
  bom + gwi + yeon = 1660 →
  bom = 300 := by
sorry

end NUMINAMATH_CALUDE_bom_watermelon_seeds_l3723_372374


namespace NUMINAMATH_CALUDE_square_sum_equals_4014_l3723_372379

theorem square_sum_equals_4014 (a : ℝ) (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a)^2 + (2004 - a)^2 = 4014 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_4014_l3723_372379


namespace NUMINAMATH_CALUDE_solution_product_l3723_372393

theorem solution_product (p q : ℝ) : 
  (p - 6) * (2 * p + 10) = p^2 - 15 * p + 56 →
  (q - 6) * (2 * q + 10) = q^2 - 15 * q + 56 →
  p ≠ q →
  (p + 4) * (q + 4) = -40 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l3723_372393


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3723_372383

/-- Represents a repeating decimal where the whole number part is 7 and the repeating part is 182. -/
def repeating_decimal : ℚ := 7 + 182 / 999

/-- The fraction representation of the repeating decimal. -/
def fraction : ℚ := 7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999. -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3723_372383


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_triangle_equality_theorem_l3723_372312

/-- A triangle with sides a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  area_positive : 0 < S

/-- The inequality holds for all triangles -/
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * t.S * Real.sqrt 3 :=
sorry

/-- The equality holds if and only if the triangle is equilateral -/
theorem triangle_equality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 4 * t.S * Real.sqrt 3 ↔ t.a = t.b ∧ t.b = t.c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_triangle_equality_theorem_l3723_372312


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3723_372310

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 3 →
  a 6 * a 7 * a 8 = 21 →
  a 9 * a 10 * a 11 = 147 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3723_372310


namespace NUMINAMATH_CALUDE_less_than_preserved_subtraction_l3723_372358

theorem less_than_preserved_subtraction (a b : ℝ) : a < b → a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_less_than_preserved_subtraction_l3723_372358


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3723_372367

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3723_372367


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3723_372320

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) 
  (h2 : 3 * a 3 ^ 2 - 25 * a 3 + 27 = 0) (h3 : 3 * a 11 ^ 2 - 25 * a 11 + 27 = 0) : a 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3723_372320


namespace NUMINAMATH_CALUDE_tan_theta_is_negative_three_l3723_372332

/-- Given vectors a and b with angle θ between them, if a • b = -1, a = (-1, 2), and |b| = √2, then tan θ = -3 -/
theorem tan_theta_is_negative_three (a b : ℝ × ℝ) (θ : ℝ) :
  a = (-1, 2) →
  a • b = -1 →
  ‖b‖ = Real.sqrt 2 →
  Real.tan θ = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_is_negative_three_l3723_372332


namespace NUMINAMATH_CALUDE_lisa_total_miles_l3723_372389

/-- The total miles flown by Lisa -/
def total_miles_flown (distance_per_trip : Float) (num_trips : Float) : Float :=
  distance_per_trip * num_trips

/-- Theorem stating that Lisa's total miles flown is 8192.0 -/
theorem lisa_total_miles :
  total_miles_flown 256.0 32.0 = 8192.0 := by
  sorry

end NUMINAMATH_CALUDE_lisa_total_miles_l3723_372389


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l3723_372373

theorem inverse_expression_equals_one_fifth :
  (3 - 4 * (3 - 5)⁻¹)⁻¹ = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l3723_372373


namespace NUMINAMATH_CALUDE_prime_triplet_divisiblity_l3723_372346

theorem prime_triplet_divisiblity (p q r : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  (p ∣ 1 + q^r) ∧ (q ∣ 1 + r^p) ∧ (r ∣ 1 + p^q) →
  ((p = 2 ∧ q = 5 ∧ r = 3) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 5)) :=
sorry

end NUMINAMATH_CALUDE_prime_triplet_divisiblity_l3723_372346


namespace NUMINAMATH_CALUDE_expected_digits_is_nineteen_twelfths_l3723_372324

/-- Die numbers -/
def die_numbers : List ℕ := List.range 12 |>.map (· + 5)

/-- Count of digits in a number -/
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- Expected value calculation -/
def expected_digits : ℚ :=
  (die_numbers.map digit_count).sum / die_numbers.length

/-- Theorem: Expected number of digits is 19/12 -/
theorem expected_digits_is_nineteen_twelfths :
  expected_digits = 19 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_digits_is_nineteen_twelfths_l3723_372324


namespace NUMINAMATH_CALUDE_trouser_sale_price_l3723_372303

theorem trouser_sale_price 
  (original_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percentage = 80) : 
  original_price * (1 - discount_percentage / 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_trouser_sale_price_l3723_372303


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3723_372376

theorem triangle_angle_problem (left right top : ℝ) : 
  left + right + top = 250 →
  left = 2 * right →
  right = 60 →
  top = 70 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3723_372376


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3723_372355

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3723_372355


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l3723_372386

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 9 else sorry

def sequence_sum (n : ℕ) : ℚ :=
  sorry

theorem smallest_n_satisfying_inequality : 
  (∀ n : ℕ, n > 0 → 3 * sequence_a (n + 1) + sequence_a n = 4) →
  sequence_a 1 = 9 →
  (∀ n : ℕ, n > 0 → |sequence_sum n - n - 6| < 1 / 125 → n ≥ 7) ∧
  |sequence_sum 7 - 7 - 6| < 1 / 125 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l3723_372386


namespace NUMINAMATH_CALUDE_star_two_neg_three_l3723_372364

-- Define the new operation *
def star (a b : ℝ) : ℝ := a * b - (a + b)

-- Theorem statement
theorem star_two_neg_three : star 2 (-3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_star_two_neg_three_l3723_372364


namespace NUMINAMATH_CALUDE_math_competition_participation_l3723_372378

theorem math_competition_participation (total_students : ℕ) (non_participants : ℕ) 
  (h1 : total_students = 39) (h2 : non_participants = 26) :
  (total_students - non_participants : ℚ) / total_students = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_participation_l3723_372378


namespace NUMINAMATH_CALUDE_sum_of_roots_of_equation_l3723_372325

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_equation_l3723_372325


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l3723_372321

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l3723_372321


namespace NUMINAMATH_CALUDE_deer_cheetah_time_difference_l3723_372340

/-- Proves the time difference between a deer and cheetah passing a point, given their speeds and catch-up time. -/
theorem deer_cheetah_time_difference 
  (deer_speed : ℝ) 
  (cheetah_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : deer_speed = 50) 
  (h2 : cheetah_speed = 60) 
  (h3 : catch_up_time = 1) : 
  ∃ (time_difference : ℝ), 
    time_difference = 4 ∧ 
    deer_speed * (catch_up_time + time_difference) = cheetah_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_deer_cheetah_time_difference_l3723_372340


namespace NUMINAMATH_CALUDE_quadratic_solution_l3723_372375

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 + k * x + 2

-- Define the known root
def known_root : ℝ := -0.5

-- Theorem statement
theorem quadratic_solution :
  ∃ (k : ℝ),
    (quadratic_equation k known_root = 0) ∧
    (k = 6) ∧
    (∃ (other_root : ℝ), 
      (quadratic_equation k other_root = 0) ∧
      (other_root = -1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3723_372375


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3723_372370

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 2) = a (n + 1) * r

theorem arithmetic_geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_ag : ArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) :
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l3723_372370


namespace NUMINAMATH_CALUDE_survey_results_l3723_372365

theorem survey_results (total : ℕ) (support_a : ℕ) (support_b : ℕ) (support_both : ℕ) (support_neither : ℕ) : 
  total = 50 ∧
  support_a = (3 * total) / 5 ∧
  support_b = support_a + 3 ∧
  support_neither = support_both / 3 + 1 ∧
  total = support_a + support_b - support_both + support_neither →
  support_both = 21 ∧ support_neither = 8 := by
  sorry

end NUMINAMATH_CALUDE_survey_results_l3723_372365


namespace NUMINAMATH_CALUDE_smallest_a_l3723_372381

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the vertex condition
def vertex_condition (a b c : ℝ) : Prop :=
  parabola a b c (1/3) = -4/3

-- Define the integer condition
def integer_condition (a b c : ℝ) : Prop :=
  ∃ n : ℤ, 3*a + 2*b + c = n

-- State the theorem
theorem smallest_a (a b c : ℝ) :
  vertex_condition a b c →
  integer_condition a b c →
  a > 0 →
  (∀ a' b' c' : ℝ, vertex_condition a' b' c' → integer_condition a' b' c' → a' > 0 → a' ≥ a) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_l3723_372381


namespace NUMINAMATH_CALUDE_no_initial_values_satisfy_conditions_l3723_372395

/-- A sequence defined by the given recurrence relation -/
def RecurrenceSequence (x₀ x₁ : ℚ) : ℕ → ℚ
  | 0 => x₀
  | 1 => x₁
  | (n + 2) => (RecurrenceSequence x₀ x₁ n * RecurrenceSequence x₀ x₁ (n + 1)) /
               (3 * RecurrenceSequence x₀ x₁ n - 2 * RecurrenceSequence x₀ x₁ (n + 1))

/-- The property of a sequence containing infinitely many natural numbers -/
def ContainsInfinitelyManyNaturals (seq : ℕ → ℚ) : Prop :=
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ ∃ m : ℕ, seq n = m

/-- The main theorem stating that no initial values satisfy the conditions -/
theorem no_initial_values_satisfy_conditions :
  ¬∃ (x₀ x₁ : ℚ), ContainsInfinitelyManyNaturals (RecurrenceSequence x₀ x₁) :=
sorry

end NUMINAMATH_CALUDE_no_initial_values_satisfy_conditions_l3723_372395


namespace NUMINAMATH_CALUDE_zachary_crunches_count_l3723_372316

/-- The number of push-ups and crunches done by David and Zachary -/
def gym_class (david_pushups david_crunches zachary_pushups zachary_crunches : ℕ) : Prop :=
  (david_pushups = zachary_pushups + 40) ∧ 
  (zachary_crunches = david_crunches + 17) ∧
  (david_crunches = 45) ∧
  (zachary_pushups = 34)

theorem zachary_crunches_count :
  ∀ (david_pushups david_crunches zachary_pushups zachary_crunches : ℕ),
  gym_class david_pushups david_crunches zachary_pushups zachary_crunches →
  zachary_crunches = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_count_l3723_372316


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_XY_length_l3723_372350

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop := sorry

def longerSide (t : Triangle) (s1 s2 : ℝ × ℝ) : Prop := sorry

def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem isosceles_right_triangle_XY_length 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : longerSide t t.X t.Y) 
  (h3 : triangleArea t = 36) : 
  ‖t.X - t.Y‖ = 12 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_XY_length_l3723_372350


namespace NUMINAMATH_CALUDE_profit_maximized_at_12_marginal_profit_decreasing_l3723_372347

-- Define the revenue and cost functions
def R (x : ℕ) : ℝ := 3700 * x + 45 * x^2 - 10 * x^3
def C (x : ℕ) : ℝ := 460 * x + 5000

-- Define the profit function
def P (x : ℕ) : ℝ := R x - C x

-- Define the marginal function
def M (f : ℕ → ℝ) (x : ℕ) : ℝ := f (x + 1) - f x

-- Define the marginal profit function
def MP (x : ℕ) : ℝ := M P x

-- Theorem: Profit is maximized when 12 ships are built
theorem profit_maximized_at_12 :
  ∀ x : ℕ, 1 ≤ x → x ≤ 20 → P 12 ≥ P x :=
sorry

-- Theorem: Marginal profit function is decreasing on [1, 19]
theorem marginal_profit_decreasing :
  ∀ x y : ℕ, 1 ≤ x → x < y → y ≤ 19 → MP y < MP x :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_12_marginal_profit_decreasing_l3723_372347


namespace NUMINAMATH_CALUDE_solve_problem_l3723_372397

def problem (basketballs soccer_balls volleyballs : ℕ) : Prop :=
  (soccer_balls = basketballs + 23) ∧
  (volleyballs + 18 = soccer_balls) ∧
  (volleyballs = 40)

theorem solve_problem :
  ∃ (basketballs soccer_balls volleyballs : ℕ),
    problem basketballs soccer_balls volleyballs ∧ basketballs = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l3723_372397


namespace NUMINAMATH_CALUDE_coffee_table_price_l3723_372306

theorem coffee_table_price 
  (sofa_price : ℕ) 
  (armchair_price : ℕ) 
  (num_armchairs : ℕ) 
  (total_invoice : ℕ) 
  (h1 : sofa_price = 1250)
  (h2 : armchair_price = 425)
  (h3 : num_armchairs = 2)
  (h4 : total_invoice = 2430) :
  total_invoice - (sofa_price + num_armchairs * armchair_price) = 330 := by
sorry

end NUMINAMATH_CALUDE_coffee_table_price_l3723_372306


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_l3723_372366

/-- The line y = -x + a intersects the curve y = √(1 - x²) at two points
    if and only if a is in the range [1, √2). -/
theorem line_intersects_circle_twice (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   -x₁ + a = Real.sqrt (1 - x₁^2) ∧
   -x₂ + a = Real.sqrt (1 - x₂^2)) ↔ 
  1 ≤ a ∧ a < Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_l3723_372366


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l3723_372327

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventA (d : Distribution) : Prop := d Person.A = Card.Red
def EventB (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_contradictory :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventA d ∧ EventB d)) ∧
  -- The events are not contradictory
  (∃ d : Distribution, EventA d) ∧
  (∃ d : Distribution, EventB d) ∧
  -- There exists a distribution where neither event occurs
  (∃ d : Distribution, ¬EventA d ∧ ¬EventB d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l3723_372327


namespace NUMINAMATH_CALUDE_division_of_fraction_by_integer_l3723_372362

theorem division_of_fraction_by_integer : 
  (3 : ℚ) / 7 / 4 = (3 : ℚ) / 28 := by
sorry

end NUMINAMATH_CALUDE_division_of_fraction_by_integer_l3723_372362


namespace NUMINAMATH_CALUDE_four_integers_problem_l3723_372314

theorem four_integers_problem (x y z u n : ℤ) :
  x + y + z + u = 36 →
  x + n = y - n ∧ y - n = z * n ∧ z * n = u / n →
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_problem_l3723_372314


namespace NUMINAMATH_CALUDE_zeros_after_one_in_10000_pow_50_l3723_372335

theorem zeros_after_one_in_10000_pow_50 :
  ∃ (n : ℕ), 10000^50 = 10^n ∧ n = 200 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_10000_pow_50_l3723_372335


namespace NUMINAMATH_CALUDE_range_of_a_l3723_372344

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a : ℝ, a ≤ 0 ∨ (1/4 < a ∧ a < 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3723_372344


namespace NUMINAMATH_CALUDE_range_of_g_l3723_372356

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x) * (Real.arcsin x)

theorem range_of_g :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
    Real.arccos x + Real.arcsin x = π / 2 →
    ∃ y ∈ Set.Icc 0 (π^2 / 8), g y = g x ∧
    ∀ z ∈ Set.Icc (-1 : ℝ) 1, g z ≤ π^2 / 8 ∧ g z ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l3723_372356


namespace NUMINAMATH_CALUDE_tan_alpha_equals_one_l3723_372351

theorem tan_alpha_equals_one (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 2 * Real.sin (α - 15 * π / 180) - 1 = 0) : Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_one_l3723_372351


namespace NUMINAMATH_CALUDE_gathering_dancers_l3723_372309

theorem gathering_dancers (men : ℕ) (women : ℕ) : 
  men = 15 →
  men * 4 = women * 3 →
  women = 20 := by
sorry

end NUMINAMATH_CALUDE_gathering_dancers_l3723_372309


namespace NUMINAMATH_CALUDE_thirtieth_term_of_specific_sequence_l3723_372394

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem thirtieth_term_of_specific_sequence :
  let a₁ := 3
  let a₂ := 7
  let d := a₂ - a₁
  arithmeticSequence a₁ d 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_specific_sequence_l3723_372394


namespace NUMINAMATH_CALUDE_complete_factorization_l3723_372363

theorem complete_factorization (x : ℝ) : 
  x^6 - 64 = (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l3723_372363


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3723_372372

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2*x - 3*y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := x - y = 0

theorem intersection_and_perpendicular_line :
  -- P is on both l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧ 
  -- l is perpendicular to l₂ and passes through P
  (∀ x y : ℝ, l x y → (x - P.1) * 1 + (y - P.2) * 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3723_372372


namespace NUMINAMATH_CALUDE_square_circle_union_area_l3723_372338

/-- The area of the union of a square and a circle with specific properties -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l3723_372338


namespace NUMINAMATH_CALUDE_simplify_trig_expression_find_sin_beta_plus_pi_over_4_l3723_372392

-- Part 1
theorem simplify_trig_expression :
  Real.sin (119 * π / 180) * Real.sin (181 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by sorry

-- Part 2
theorem find_sin_beta_plus_pi_over_4 (α β : Real) 
  (h1 : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5)
  (h2 : π < β ∧ β < 3*π/2) :  -- β is in the third quadrant
  Real.sin (β + π/4) = -7*Real.sqrt 2/10 := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_find_sin_beta_plus_pi_over_4_l3723_372392


namespace NUMINAMATH_CALUDE_regions_theorem_l3723_372336

/-- The number of regions formed by n lines in a plane -/
def total_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- The number of bounded regions formed by n lines in a plane -/
def bounded_regions (n : ℕ) : ℚ :=
  (n^2 - 3*n + 2) / 2

/-- Theorem stating the formulas for total and bounded regions -/
theorem regions_theorem (n : ℕ) :
  (total_regions n = (n^2 + n + 2) / 2) ∧
  (bounded_regions n = (n^2 - 3*n + 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_regions_theorem_l3723_372336


namespace NUMINAMATH_CALUDE_average_w_x_is_half_l3723_372398

theorem average_w_x_is_half 
  (w x y : ℝ) 
  (h1 : 5 / w + 5 / x = 5 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_w_x_is_half_l3723_372398


namespace NUMINAMATH_CALUDE_veranda_area_l3723_372387

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) : 
  room_length = 21 →
  room_width = 12 →
  veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 148 := by
  sorry


end NUMINAMATH_CALUDE_veranda_area_l3723_372387


namespace NUMINAMATH_CALUDE_binomial_12_11_l3723_372313

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l3723_372313


namespace NUMINAMATH_CALUDE_division_problem_l3723_372333

theorem division_problem (remainder quotient divisor dividend : ℕ) :
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  dividend = divisor * quotient + remainder →
  dividend = 86 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3723_372333


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3723_372357

theorem min_value_sum_squares (x y z : ℝ) :
  x - 1 = 2 * (y + 1) ∧ x - 1 = 3 * (z + 2) →
  ∀ a b c : ℝ, a - 1 = 2 * (b + 1) ∧ a - 1 = 3 * (c + 2) →
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  ∃ x₀ y₀ z₀ : ℝ, x₀ - 1 = 2 * (y₀ + 1) ∧ x₀ - 1 = 3 * (z₀ + 2) ∧
                  x₀^2 + y₀^2 + z₀^2 = 293 / 49 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3723_372357


namespace NUMINAMATH_CALUDE_sum_of_f_values_l3723_372334

noncomputable def f (x : ℝ) : ℝ := (x * Real.exp x + x + 2) / (Real.exp x + 1) + Real.sin x

theorem sum_of_f_values : 
  f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l3723_372334


namespace NUMINAMATH_CALUDE_money_division_l3723_372353

theorem money_division (A B C : ℚ) (h1 : A = (1/3) * (B + C))
                                   (h2 : ∃ x, B = x * (A + C))
                                   (h3 : A = B + 15)
                                   (h4 : A + B + C = 540) :
  ∃ x, B = x * (A + C) ∧ x = 2/9 := by
sorry

end NUMINAMATH_CALUDE_money_division_l3723_372353


namespace NUMINAMATH_CALUDE_boys_in_biology_class_l3723_372399

/-- Given a Physics class with 200 students and a Biology class with half as many students,
    where the ratio of girls to boys in Biology is 3:1, prove that there are 25 boys in Biology. -/
theorem boys_in_biology_class
  (physics_students : ℕ)
  (biology_students : ℕ)
  (girls_to_boys_ratio : ℚ)
  (h1 : physics_students = 200)
  (h2 : biology_students = physics_students / 2)
  (h3 : girls_to_boys_ratio = 3)
  : biology_students / (girls_to_boys_ratio + 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_biology_class_l3723_372399


namespace NUMINAMATH_CALUDE_chantel_bracelets_l3723_372385

/-- Represents the number of bracelets Chantel makes per day in the last four days -/
def x : ℕ := sorry

/-- The total number of bracelets Chantel has at the end -/
def total_bracelets : ℕ := 13

/-- The number of bracelets Chantel makes in the first 5 days -/
def first_phase_bracelets : ℕ := 2 * 5

/-- The number of bracelets Chantel gives away after the first phase -/
def first_giveaway : ℕ := 3

/-- The number of bracelets Chantel gives away after the second phase -/
def second_giveaway : ℕ := 6

/-- The number of days in the second phase -/
def second_phase_days : ℕ := 4

theorem chantel_bracelets : 
  first_phase_bracelets - first_giveaway + x * second_phase_days - second_giveaway = total_bracelets ∧ 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_chantel_bracelets_l3723_372385


namespace NUMINAMATH_CALUDE_igneous_sedimentary_ratio_l3723_372318

/-- Represents Cliff's rock collection --/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- The properties of Cliff's rock collection --/
def isValidCollection (c : RockCollection) : Prop :=
  c.shinyIgneous = (2 * c.igneous) / 3 ∧
  c.shinySedimentary = c.sedimentary / 5 ∧
  c.shinyIgneous = 40 ∧
  c.igneous + c.sedimentary = 180

/-- The theorem stating the ratio of igneous to sedimentary rocks --/
theorem igneous_sedimentary_ratio (c : RockCollection) 
  (h : isValidCollection c) : c.igneous * 2 = c.sedimentary := by
  sorry


end NUMINAMATH_CALUDE_igneous_sedimentary_ratio_l3723_372318


namespace NUMINAMATH_CALUDE_range_equals_fixed_points_l3723_372330

theorem range_equals_fixed_points (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f (f m) + f n) : 
  {n : ℕ | ∃ k : ℕ, f k = n} = {n : ℕ | f n = n} := by
sorry

end NUMINAMATH_CALUDE_range_equals_fixed_points_l3723_372330


namespace NUMINAMATH_CALUDE_sum_of_roots_l3723_372323

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 6*x^2 + 15*x = 12) 
  (hy : y^3 - 6*y^2 + 15*y = 16) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3723_372323


namespace NUMINAMATH_CALUDE_problem_solution_l3723_372311

theorem problem_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (7 * x) * Real.sqrt (21 * x) = 42) : 
  x = Real.sqrt (21 / 47) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3723_372311


namespace NUMINAMATH_CALUDE_slower_traveler_speed_l3723_372326

/-- Proves that given two people traveling in opposite directions for 1.5 hours,
    where one travels 3 miles per hour faster than the other, and they end up 19.5 miles apart,
    the slower person's speed is 5 miles per hour. -/
theorem slower_traveler_speed
  (time : ℝ)
  (distance_apart : ℝ)
  (speed_difference : ℝ)
  (h1 : time = 1.5)
  (h2 : distance_apart = 19.5)
  (h3 : speed_difference = 3)
  : ∃ (slower_speed : ℝ), slower_speed = 5 ∧
    distance_apart = time * (slower_speed + (slower_speed + speed_difference)) :=
by sorry

end NUMINAMATH_CALUDE_slower_traveler_speed_l3723_372326


namespace NUMINAMATH_CALUDE_final_b_value_l3723_372307

def program_execution (a b c : Int) : Int :=
  let a' := b
  let b' := c
  b'

theorem final_b_value :
  ∀ (a b c : Int),
  a = 3 →
  b = -5 →
  c = 8 →
  program_execution a b c = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_b_value_l3723_372307


namespace NUMINAMATH_CALUDE_asymptote_slope_l3723_372377

-- Define the hyperbola parameters
def m : ℝ := 2

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / (m^2 + 12) - y^2 / (5*m - 1) = 1

-- Define the length of the real axis
def real_axis_length : ℝ := 8

-- Theorem statement
theorem asymptote_slope :
  hyperbola x y ∧ real_axis_length = 8 →
  ∃ (k : ℝ), k = 3/4 ∧ (y = k*x ∨ y = -k*x) :=
sorry

end NUMINAMATH_CALUDE_asymptote_slope_l3723_372377


namespace NUMINAMATH_CALUDE_proportion_solution_l3723_372322

theorem proportion_solution (x : ℝ) : (0.75 / x = 3 / 8) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3723_372322


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3723_372388

theorem unique_solution_for_equation (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3723_372388


namespace NUMINAMATH_CALUDE_min_n_for_geometric_sum_l3723_372305

theorem min_n_for_geometric_sum (n : ℕ) : 
  (∀ k : ℕ, k < n → (2^(k+1) - 1) ≤ 128) ∧ 
  (2^(n+1) - 1) > 128 → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_min_n_for_geometric_sum_l3723_372305


namespace NUMINAMATH_CALUDE_regular_hexagon_radius_l3723_372391

/-- The radius of a regular hexagon with perimeter 12a is 2a -/
theorem regular_hexagon_radius (a : ℝ) (h : a > 0) :
  let perimeter := 12 * a
  ∃ (radius : ℝ), radius = 2 * a ∧ 
    (∃ (side : ℝ), side * 6 = perimeter ∧ radius = side) := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_radius_l3723_372391


namespace NUMINAMATH_CALUDE_greg_distance_when_azarah_finishes_l3723_372382

/-- Represents the constant speed of a runner -/
structure Speed : Type :=
  (value : ℝ)
  (pos : value > 0)

/-- Calculates the distance traveled given speed and time -/
def distance (s : Speed) (t : ℝ) : ℝ := s.value * t

theorem greg_distance_when_azarah_finishes 
  (azarah_speed charlize_speed greg_speed : Speed)
  (h1 : distance azarah_speed 1 = 100)
  (h2 : distance charlize_speed 1 = 80)
  (h3 : distance charlize_speed (100 / charlize_speed.value) = 100)
  (h4 : distance greg_speed (100 / charlize_speed.value) = 90) :
  distance greg_speed (100 / azarah_speed.value) = 72 :=
sorry

end NUMINAMATH_CALUDE_greg_distance_when_azarah_finishes_l3723_372382


namespace NUMINAMATH_CALUDE_probability_two_defective_shipment_l3723_372341

/-- The probability of selecting two defective smartphones at random from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℝ :=
  let p1 := defective / total
  let p2 := (defective - 1) / (total - 1)
  p1 * p2

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_shipment :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  |probability_two_defective 250 76 - 0.0915632| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_two_defective_shipment_l3723_372341


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3723_372302

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 2*(3*x - 2) + (x + 1) = 0 ↔ x^2 - 5*x + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3723_372302


namespace NUMINAMATH_CALUDE_digit_multiplication_sum_l3723_372308

theorem digit_multiplication_sum (p q : ℕ) : 
  p < 10 → q < 10 → (40 + p) * (10 * q + 5) = 190 → p + q = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_sum_l3723_372308


namespace NUMINAMATH_CALUDE_son_age_is_eleven_l3723_372304

/-- Represents the ages of a mother and son -/
structure FamilyAges where
  son : ℕ
  mother : ℕ

/-- The conditions of the age problem -/
def AgeProblemConditions (ages : FamilyAges) : Prop :=
  (ages.son + ages.mother = 55) ∧ 
  (ages.son - 3 + ages.mother - 3 = 49) ∧
  (ages.mother = 4 * ages.son)

/-- The theorem stating that under the given conditions, the son's age is 11 -/
theorem son_age_is_eleven (ages : FamilyAges) 
  (h : AgeProblemConditions ages) : ages.son = 11 := by
  sorry

end NUMINAMATH_CALUDE_son_age_is_eleven_l3723_372304


namespace NUMINAMATH_CALUDE_max_red_socks_l3723_372368

def is_valid_sock_distribution (r b y : ℕ) : Prop :=
  let t := r + b + y
  t ≤ 2300 ∧
  (r * (r - 1) * (r - 2) + b * (b - 1) * (b - 2) + y * (y - 1) * (y - 2)) * 3 =
  t * (t - 1) * (t - 2)

theorem max_red_socks :
  ∀ r b y : ℕ, is_valid_sock_distribution r b y → r ≤ 897 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l3723_372368


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_max_l3723_372349

theorem quadratic_function_inequality_max (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = Real.sqrt 6 - 2 ∧ 
    (∀ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') → 
      b'^2 / (a'^2 + 2 * c'^2) ≤ M) ∧
    (∃ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') ∧ 
      b'^2 / (a'^2 + 2 * c'^2) = M)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_max_l3723_372349


namespace NUMINAMATH_CALUDE_binomial_odd_iff_binary_condition_l3723_372361

def has_one_at_position (m : ℕ) (pos : ℕ) : Prop :=
  (m / 2^pos) % 2 = 1

theorem binomial_odd_iff_binary_condition (n k : ℕ) :
  Nat.choose n k % 2 = 1 ↔ ∀ pos, has_one_at_position k pos → has_one_at_position n pos :=
sorry

end NUMINAMATH_CALUDE_binomial_odd_iff_binary_condition_l3723_372361


namespace NUMINAMATH_CALUDE_infinite_sum_equals_nine_eighties_l3723_372331

/-- The infinite sum of 2n / (n^4 + 16) from n=1 to infinity equals 9/80 -/
theorem infinite_sum_equals_nine_eighties :
  (∑' n : ℕ+, (2 * n : ℝ) / (n^4 + 16)) = 9 / 80 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_nine_eighties_l3723_372331


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_and_consecutive_primes_l3723_372352

/-- A function that returns the number of positive factors of a given natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if two prime numbers are consecutive -/
def are_consecutive_primes (p q : ℕ) : Prop := sorry

/-- The theorem stating that 36 is the least positive integer with exactly 12 factors
    and consecutive prime factors -/
theorem least_integer_with_12_factors_and_consecutive_primes :
  ∀ n : ℕ, n > 0 → num_factors n = 12 →
  (∃ p q : ℕ, n = p^2 * q^2 ∧ are_consecutive_primes p q) →
  n ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_and_consecutive_primes_l3723_372352


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_diff_l3723_372345

theorem sqrt_fraction_sum_diff (x : ℝ) : 
  x = Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36 - (1 : ℝ) / 100) → x = (Real.sqrt 13) / 15 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_diff_l3723_372345


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3723_372315

theorem decimal_multiplication (h : 213 * 16 = 3408) : 0.16 * 2.13 = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3723_372315


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_solutions_l3723_372384

theorem infinitely_many_primes_with_solutions : 
  ¬ (∃ (S : Finset Nat), ∀ (p : Nat), 
    (Nat.Prime p ∧ (∃ (x y : ℤ), x^2 + x + 1 = p * y)) → p ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_solutions_l3723_372384


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l3723_372343

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry for a regular polygon (in degrees) -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ (p : RegularPolygon 17),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 649 / 17 := by
  sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l3723_372343
