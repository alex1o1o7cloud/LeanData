import Mathlib

namespace NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l721_72131

theorem division_of_mixed_number_by_fraction :
  (2 + 1 / 4 : ℚ) / (2 / 3 : ℚ) = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_division_of_mixed_number_by_fraction_l721_72131


namespace NUMINAMATH_CALUDE_factorial_base_312_b3_is_zero_l721_72142

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Checks if a list of coefficients is a valid factorial base representation -/
def isValidFactorialBase (coeffs : List ℕ) : Prop :=
  ∀ (i : ℕ), i < coeffs.length → coeffs[i]! ≤ i + 1

/-- Computes the value represented by a list of coefficients in factorial base -/
def valueFromFactorialBase (coeffs : List ℕ) : ℕ :=
  coeffs.enum.foldl (fun acc (i, b) => acc + b * factorial (i + 1)) 0

/-- Theorem: The factorial base representation of 312 has b₃ = 0 -/
theorem factorial_base_312_b3_is_zero :
  ∃ (coeffs : List ℕ),
    isValidFactorialBase coeffs ∧
    valueFromFactorialBase coeffs = 312 ∧
    coeffs.length > 3 ∧
    coeffs[2]! = 0 :=
by sorry

end NUMINAMATH_CALUDE_factorial_base_312_b3_is_zero_l721_72142


namespace NUMINAMATH_CALUDE_quadratic_inequality_l721_72152

-- Define the quadratic function
def f (b c x : ℝ) := x^2 + b*x + c

-- Define the solution set condition
def solution_set (b c : ℝ) : Prop :=
  ∀ x, f b c x > 0 ↔ (x > 2 ∨ x < 1)

-- Theorem statement
theorem quadratic_inequality (b c : ℝ) (h : solution_set b c) :
  (b = -3 ∧ c = 2) ∧
  (∀ x, 2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l721_72152


namespace NUMINAMATH_CALUDE_estate_distribution_l721_72134

/-- Represents the estate distribution problem --/
theorem estate_distribution (total : ℚ) 
  (daughter_share : ℚ) (son_share : ℚ) (husband_share : ℚ) (gardener_share : ℚ) : 
  daughter_share + son_share = (3 : ℚ) / 5 * total →
  daughter_share = (3 : ℚ) / 5 * (daughter_share + son_share) →
  husband_share = 3 * son_share →
  gardener_share = 600 →
  total = daughter_share + son_share + husband_share + gardener_share →
  total = 1875 := by
  sorry

end NUMINAMATH_CALUDE_estate_distribution_l721_72134


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l721_72116

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a * b = -15) : 
  a^2 + b^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l721_72116


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l721_72130

variable (x y : ℝ)

theorem problem_1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = -x * y :=
by sorry

theorem problem_2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l721_72130


namespace NUMINAMATH_CALUDE_sqrt_three_addition_l721_72183

theorem sqrt_three_addition : 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_addition_l721_72183


namespace NUMINAMATH_CALUDE_range_of_g_l721_72115

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l721_72115


namespace NUMINAMATH_CALUDE_remainder_19_power_1999_mod_25_l721_72157

theorem remainder_19_power_1999_mod_25 : 19^1999 % 25 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_19_power_1999_mod_25_l721_72157


namespace NUMINAMATH_CALUDE_complex_trajectory_line_l721_72136

theorem complex_trajectory_line (z : ℂ) :
  Complex.abs (z + 1) = Complex.abs (1 + Complex.I * z) →
  (z.re : ℝ) + z.im = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_trajectory_line_l721_72136


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l721_72189

/-- A geometric sequence with a₂ = 8 and a₅ = 64 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence definition
  a 2 = 8 →                              -- Given condition
  a 5 = 64 →                             -- Given condition
  a 2 / a 1 = 2 :=                       -- Common ratio q = 2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l721_72189


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l721_72170

theorem purely_imaginary_complex_number (m : ℝ) :
  (((m * (m + 2)) / (m - 1) : ℂ) + (m^2 + m - 2) * I).re = 0 ∧
  (((m * (m + 2)) / (m - 1) : ℂ) + (m^2 + m - 2) * I).im ≠ 0 →
  m = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l721_72170


namespace NUMINAMATH_CALUDE_no_solution_exists_l721_72188

/-- Represents the number of books for each subject -/
structure BookCounts where
  math : ℕ
  history : ℕ
  science : ℕ
  literature : ℕ

/-- The problem constraints -/
def satisfiesConstraints (books : BookCounts) : Prop :=
  books.math + books.history + books.science + books.literature = 80 ∧
  4 * books.math + 5 * books.history + 6 * books.science + 7 * books.literature = 520 ∧
  3 * books.history = 2 * books.math ∧
  2 * books.science = books.math ∧
  4 * books.literature = books.math

theorem no_solution_exists : ¬∃ (books : BookCounts), satisfiesConstraints books := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l721_72188


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l721_72156

/-- Geometric sequence with common ratio greater than 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : GeometricSequence a q)
  (h_sum : a 1 + a 4 = 9)
  (h_prod : a 2 * a 3 = 8) :
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l721_72156


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l721_72109

theorem fraction_to_decimal : (7 : ℚ) / 32 = 0.21875 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l721_72109


namespace NUMINAMATH_CALUDE_function_zero_nonpositive_l721_72176

/-- A function satisfying the given inequality property -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≤ y * f x + f (f x)

/-- The main theorem to prove -/
theorem function_zero_nonpositive (f : ℝ → ℝ) (h : SatisfiesInequality f) :
    ∀ x : ℝ, x ≤ 0 → f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_nonpositive_l721_72176


namespace NUMINAMATH_CALUDE_problem_solution_l721_72139

/-- An arithmetic sequence with positive terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem problem_solution (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
    (h_geom : geometric_sequence b)
    (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l721_72139


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l721_72191

def fraction : ℚ := 3 / (2^7 * 5^10)

theorem zeros_before_first_nonzero_digit : 
  (∃ (n : ℕ) (d : ℚ), fraction * 10^n = d ∧ d ≥ 1 ∧ d < 10 ∧ n = 8) :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l721_72191


namespace NUMINAMATH_CALUDE_friends_money_sharing_l721_72123

theorem friends_money_sharing (A : ℝ) (h_pos : A > 0) :
  let jorge_total := 5 * A
  let jose_total := 4 * A
  let janio_total := 3 * A
  let joao_received := 3 * A
  let group_total := jorge_total + jose_total + janio_total
  (joao_received / group_total) = (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_friends_money_sharing_l721_72123


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l721_72105

/-- Given the initial total weight of strawberries collected by Marco and his dad,
    the additional weight of strawberries found by dad, and dad's final weight of strawberries,
    prove that Marco's strawberries weigh 6 pounds. -/
theorem marcos_strawberries_weight
  (initial_total : ℕ)
  (dads_additional : ℕ)
  (dads_final : ℕ)
  (h1 : initial_total = 22)
  (h2 : dads_additional = 30)
  (h3 : dads_final = 16) :
  initial_total - dads_final = 6 :=
by sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l721_72105


namespace NUMINAMATH_CALUDE_equal_values_l721_72138

-- Define the algebraic expression
def f (a : ℝ) : ℝ := a^4 - 2*a^2 + 3

-- State the theorem
theorem equal_values : f 2 = f (-2) := by sorry

end NUMINAMATH_CALUDE_equal_values_l721_72138


namespace NUMINAMATH_CALUDE_length_of_angle_bisector_l721_72182

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 8 ∧ qr = 15 ∧ pr = 17

-- Define the angle bisector PS
def AngleBisector (P Q R S : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let qs := Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2)
  let rs := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  qs / rs = pq / qr

-- Theorem statement
theorem length_of_angle_bisector 
  (P Q R S : ℝ × ℝ) 
  (h1 : Triangle P Q R) 
  (h2 : AngleBisector P Q R S) : 
  Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = Real.sqrt 87.04 :=
by
  sorry

end NUMINAMATH_CALUDE_length_of_angle_bisector_l721_72182


namespace NUMINAMATH_CALUDE_min_distance_to_line_l721_72199

-- Define the right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c > a ∧ c > b ∧ a^2 + b^2 = c^2

-- Define the point (m, n) on the line ax + by + c = 0
def point_on_line (a b c m n : ℝ) : Prop :=
  a * m + b * n + c = 0

-- Theorem statement
theorem min_distance_to_line (a b c m n : ℝ) 
  (h1 : is_right_triangle a b c) 
  (h2 : point_on_line a b c m n) : 
  m^2 + n^2 ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l721_72199


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l721_72122

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a → a 1 + a 2 = 3 → a 4 + a 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l721_72122


namespace NUMINAMATH_CALUDE_company_employees_l721_72133

theorem company_employees (december_employees : ℕ) (january_employees : ℕ) 
  (h1 : december_employees = 460) 
  (h2 : december_employees = january_employees + (january_employees * 15 / 100)) : 
  january_employees = 400 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l721_72133


namespace NUMINAMATH_CALUDE_gary_chicken_multiple_l721_72197

/-- The multiple of chickens Gary has now compared to the start -/
def chicken_multiple (initial_chickens : ℕ) (eggs_per_day : ℕ) (total_eggs_per_week : ℕ) : ℕ :=
  (total_eggs_per_week / (eggs_per_day * 7)) / initial_chickens

/-- Proof that Gary's chicken multiple is 8 -/
theorem gary_chicken_multiple :
  chicken_multiple 4 6 1344 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gary_chicken_multiple_l721_72197


namespace NUMINAMATH_CALUDE_jakes_test_average_l721_72120

theorem jakes_test_average : 
  let first_test : ℕ := 80
  let second_test : ℕ := first_test + 10
  let third_test : ℕ := 65
  let fourth_test : ℕ := third_test
  let total_marks : ℕ := first_test + second_test + third_test + fourth_test
  let num_tests : ℕ := 4
  (total_marks : ℚ) / num_tests = 75 := by
  sorry

end NUMINAMATH_CALUDE_jakes_test_average_l721_72120


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l721_72180

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l721_72180


namespace NUMINAMATH_CALUDE_log_equation_solution_l721_72198

theorem log_equation_solution (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x - 2*y > 0) :
  2 * Real.log (x - 2*y) = Real.log x + Real.log y → x / y = 4 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l721_72198


namespace NUMINAMATH_CALUDE_all_players_odd_sum_probability_l721_72194

def number_of_tiles : ℕ := 15
def number_of_players : ℕ := 5
def tiles_per_player : ℕ := 3

def probability_all_odd_sum : ℚ :=
  480 / 19019

theorem all_players_odd_sum_probability :
  (number_of_tiles = 15) →
  (number_of_players = 5) →
  (tiles_per_player = 3) →
  probability_all_odd_sum = 480 / 19019 :=
by sorry

end NUMINAMATH_CALUDE_all_players_odd_sum_probability_l721_72194


namespace NUMINAMATH_CALUDE_remainder_2027_div_28_l721_72160

theorem remainder_2027_div_28 : 2027 % 28 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2027_div_28_l721_72160


namespace NUMINAMATH_CALUDE_expenditure_difference_l721_72114

theorem expenditure_difference
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_increase_percent : ℝ)
  (purchased_quantity_percent : ℝ)
  (h1 : price_increase_percent = 25)
  (h2 : purchased_quantity_percent = 72)
  : (1 + price_increase_percent / 100) * (purchased_quantity_percent / 100) - 1 = -0.1 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_difference_l721_72114


namespace NUMINAMATH_CALUDE_sqrt_inequality_l721_72177

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l721_72177


namespace NUMINAMATH_CALUDE_range_of_a_l721_72164

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a ↔ -3 ≤ a ∧ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l721_72164


namespace NUMINAMATH_CALUDE_blackboard_multiplication_l721_72185

theorem blackboard_multiplication (a b : ℕ) (n : ℕ+) : 
  (100 ≤ a ∧ a ≤ 999) →
  (100 ≤ b ∧ b ≤ 999) →
  10000 * a + b = n * (a * b) →
  n = 73 := by sorry

end NUMINAMATH_CALUDE_blackboard_multiplication_l721_72185


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l721_72110

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem f_strictly_increasing :
  (∀ x y, x < y ∧ x < -1/3 → f x < f y) ∧
  (∀ x y, x < y ∧ 1 < x → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l721_72110


namespace NUMINAMATH_CALUDE_distinct_quotients_exist_l721_72178

/-- A function that checks if a number is composed of five twos and three ones -/
def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).count 2 = 5 ∧ (n.digits 10).count 1 = 3 ∧ (n.digits 10).length = 8

/-- The theorem statement -/
theorem distinct_quotients_exist : ∃ (a b c d e : ℕ),
  is_valid_number a ∧
  is_valid_number b ∧
  is_valid_number c ∧
  is_valid_number d ∧
  is_valid_number e ∧
  a % 7 = 0 ∧
  b % 7 = 0 ∧
  c % 7 = 0 ∧
  d % 7 = 0 ∧
  e % 7 = 0 ∧
  a / 7 ≠ b / 7 ∧
  a / 7 ≠ c / 7 ∧
  a / 7 ≠ d / 7 ∧
  a / 7 ≠ e / 7 ∧
  b / 7 ≠ c / 7 ∧
  b / 7 ≠ d / 7 ∧
  b / 7 ≠ e / 7 ∧
  c / 7 ≠ d / 7 ∧
  c / 7 ≠ e / 7 ∧
  d / 7 ≠ e / 7 :=
sorry

end NUMINAMATH_CALUDE_distinct_quotients_exist_l721_72178


namespace NUMINAMATH_CALUDE_faiths_weekly_earnings_l721_72143

/-- Faith's weekly earnings calculation --/
theorem faiths_weekly_earnings
  (hourly_rate : ℝ)
  (regular_hours_per_day : ℕ)
  (working_days_per_week : ℕ)
  (overtime_hours_per_day : ℕ)
  (h1 : hourly_rate = 13.5)
  (h2 : regular_hours_per_day = 8)
  (h3 : working_days_per_week = 5)
  (h4 : overtime_hours_per_day = 2) :
  let regular_pay := hourly_rate * regular_hours_per_day * working_days_per_week
  let overtime_pay := hourly_rate * overtime_hours_per_day * working_days_per_week
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 675 := by sorry

end NUMINAMATH_CALUDE_faiths_weekly_earnings_l721_72143


namespace NUMINAMATH_CALUDE_common_chord_intersection_l721_72102

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for circles
variable (Circle : Type)

-- Function to check if a point is on a circle
variable (on_circle : Point → Circle → Prop)

-- Function to check if two circles intersect
variable (intersect : Circle → Circle → Prop)

-- Function to create a circle passing through two points
variable (circle_through : Point → Point → Circle)

-- Function to find the common chord of two circles
variable (common_chord : Circle → Circle → Set Point)

-- Theorem statement
theorem common_chord_intersection
  (A B C D : Point)
  (h : ∀ (c1 c2 : Circle), on_circle A c1 → on_circle B c1 → 
                           on_circle C c2 → on_circle D c2 → 
                           intersect c1 c2) :
  ∃ (P : Point), ∀ (c1 c2 : Circle),
    on_circle A c1 → on_circle B c1 →
    on_circle C c2 → on_circle D c2 →
    P ∈ common_chord c1 c2 :=
sorry

end NUMINAMATH_CALUDE_common_chord_intersection_l721_72102


namespace NUMINAMATH_CALUDE_factorization_of_x4_plus_64_l721_72190

theorem factorization_of_x4_plus_64 (x : ℝ) : x^4 + 64 = (x^2 - 4*x + 8) * (x^2 + 4*x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x4_plus_64_l721_72190


namespace NUMINAMATH_CALUDE_largest_common_divisor_462_231_l721_72140

theorem largest_common_divisor_462_231 : Nat.gcd 462 231 = 231 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_462_231_l721_72140


namespace NUMINAMATH_CALUDE_pyramid_volume_transformation_l721_72162

/-- Represents a pyramid with a triangular base -/
structure Pyramid where
  volume : ℝ
  base_a : ℝ
  base_b : ℝ
  base_c : ℝ
  height : ℝ

/-- Transforms a pyramid according to the given conditions -/
def transform_pyramid (p : Pyramid) : Pyramid :=
  { volume := 0,  -- We'll prove this is 12 * p.volume
    base_a := 2 * p.base_a,
    base_b := 2 * p.base_b,
    base_c := 3 * p.base_c,
    height := 3 * p.height }

theorem pyramid_volume_transformation (p : Pyramid) :
  (transform_pyramid p).volume = 12 * p.volume := by
  sorry

#check pyramid_volume_transformation

end NUMINAMATH_CALUDE_pyramid_volume_transformation_l721_72162


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l721_72147

theorem prime_power_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  let n := (2^(2*p) - 1) / 3
  n ∣ (2^n - 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l721_72147


namespace NUMINAMATH_CALUDE_gibi_score_is_59_percent_l721_72163

/-- Represents the exam scores of four students -/
structure ExamScores where
  max_score : ℕ
  jigi_percent : ℕ
  mike_percent : ℕ
  lizzy_percent : ℕ
  average_mark : ℕ

/-- Calculates Gibi's score percentage given the exam scores -/
def gibi_score_percent (scores : ExamScores) : ℕ :=
  let total_marks := 4 * scores.average_mark
  let other_scores := (scores.jigi_percent * scores.max_score / 100) +
                      (scores.mike_percent * scores.max_score / 100) +
                      (scores.lizzy_percent * scores.max_score / 100)
  let gibi_score := total_marks - other_scores
  (gibi_score * 100) / scores.max_score

/-- Theorem stating that Gibi's score percentage is 59% given the exam conditions -/
theorem gibi_score_is_59_percent (scores : ExamScores)
  (h1 : scores.max_score = 700)
  (h2 : scores.jigi_percent = 55)
  (h3 : scores.mike_percent = 99)
  (h4 : scores.lizzy_percent = 67)
  (h5 : scores.average_mark = 490) :
  gibi_score_percent scores = 59 := by
  sorry

end NUMINAMATH_CALUDE_gibi_score_is_59_percent_l721_72163


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l721_72192

theorem smallest_lcm_with_gcd_5 (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  ∀ m n : ℕ, 1000 ≤ m ∧ m < 10000 ∧ 
             1000 ≤ n ∧ n < 10000 ∧ 
             Nat.gcd m n = 5 →
  Nat.lcm k l ≤ Nat.lcm m n ∧
  Nat.lcm k l = 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l721_72192


namespace NUMINAMATH_CALUDE_quotient_change_l721_72169

theorem quotient_change (initial_quotient : ℝ) (dividend_multiplier : ℝ) (divisor_multiplier : ℝ) :
  initial_quotient = 0.78 →
  dividend_multiplier = 10 →
  divisor_multiplier = 0.1 →
  initial_quotient * dividend_multiplier / divisor_multiplier = 78 := by
  sorry

#check quotient_change

end NUMINAMATH_CALUDE_quotient_change_l721_72169


namespace NUMINAMATH_CALUDE_rectangle_area_l721_72155

theorem rectangle_area (x y : ℝ) 
  (h1 : (x + 3.5) * (y - 1.5) = x * y)
  (h2 : (x - 3.5) * (y + 2.5) = x * y)
  (h3 : 2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y) :
  x * y = 196 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l721_72155


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l721_72121

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / (Real.exp 1 * x)

theorem f_monotonicity_and_inequality (a : ℝ) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x < y ∧ a ≤ 0 → f a x < f a y) ∧
  (a > 0 →
    (∀ x y, 0 < x ∧ x < y ∧ y < a / Real.exp 1 → f a y < f a x) ∧
    (∀ x y, a / Real.exp 1 < x ∧ x < y → f a x < f a y)) ∧
  (a = 2 → ∀ x, x > 0 → f a x > Real.exp (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l721_72121


namespace NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l721_72153

theorem exactly_two_sunny_days_probability :
  let days : ℕ := 3
  let rain_probability : ℚ := 60 / 100
  let sunny_probability : ℚ := 1 - rain_probability
  let ways_to_choose_two_days : ℕ := (days.choose 2)
  let probability_two_sunny_one_rainy : ℚ := sunny_probability^2 * rain_probability
  ways_to_choose_two_days * probability_two_sunny_one_rainy = 36 / 125 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_sunny_days_probability_l721_72153


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l721_72126

/-- The cost price of a cupboard satisfying certain conditions -/
def cost_price : ℝ := 5625

/-- The selling price of the cupboard -/
def selling_price : ℝ := 0.84 * cost_price

/-- The increased selling price that would result in a profit -/
def increased_selling_price : ℝ := 1.16 * cost_price

/-- Theorem stating that the cost price satisfies the given conditions -/
theorem cupboard_cost_price : 
  selling_price = 0.84 * cost_price ∧ 
  increased_selling_price = selling_price + 1800 :=
by sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l721_72126


namespace NUMINAMATH_CALUDE_middle_speed_calculation_l721_72106

/-- Represents the speed and duration of a part of the journey -/
structure JourneyPart where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (part : JourneyPart) : ℝ := part.speed * part.duration

theorem middle_speed_calculation (total_distance : ℝ) (first_part last_part middle_part : JourneyPart) 
  (h1 : total_distance = 800)
  (h2 : first_part.speed = 80 ∧ first_part.duration = 6)
  (h3 : last_part.speed = 40 ∧ last_part.duration = 2)
  (h4 : middle_part.duration = 4)
  (h5 : total_distance = distance first_part + distance middle_part + distance last_part) :
  middle_part.speed = 60 := by
sorry

end NUMINAMATH_CALUDE_middle_speed_calculation_l721_72106


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l721_72181

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (simple_interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : simple_interest = 176)
  (h2 : time = 4)
  (h3 : rate = 5.5 / 100) :
  simple_interest = (800 : ℝ) * rate * time := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l721_72181


namespace NUMINAMATH_CALUDE_square_root_sum_equality_l721_72119

theorem square_root_sum_equality (n : ℕ) :
  (∃ (x : ℕ), (x : ℝ) * (2018 : ℝ)^2 = (2018 : ℝ)^20) ∧
  (Real.sqrt ((x : ℝ) * (2018 : ℝ)^2) = (2018 : ℝ)^10) →
  x = 2018^18 :=
by sorry

end NUMINAMATH_CALUDE_square_root_sum_equality_l721_72119


namespace NUMINAMATH_CALUDE_equation_proof_l721_72171

theorem equation_proof : 300 * 2 + (12 + 4) * (1 / 8) = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l721_72171


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_slope_product_neg_one_l721_72112

/-- Two lines in the plane are perpendicular if and only if the product of their slopes is -1 -/
theorem lines_perpendicular_iff_slope_product_neg_one 
  (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) (hB₁ : B₁ ≠ 0) (hB₂ : B₂ ≠ 0) :
  (∀ x y : ℝ, A₁ * x + B₁ * y + C₁ = 0 → A₂ * x + B₂ * y + C₂ = 0 → 
    (A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) → 
    (A₁ * A₂) / (B₁ * B₂) = -1) ↔
  (A₁ * A₂) / (B₁ * B₂) = -1 :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_slope_product_neg_one_l721_72112


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l721_72187

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_y_intercept :
  let m := (3 : ℝ) * P.1^2  -- Slope of the tangent line
  let b := P.2 - m * P.1    -- y-intercept of the tangent line
  b = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l721_72187


namespace NUMINAMATH_CALUDE_difference_of_squares_601_599_l721_72151

theorem difference_of_squares_601_599 : 601^2 - 599^2 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_601_599_l721_72151


namespace NUMINAMATH_CALUDE_unique_number_pair_l721_72127

theorem unique_number_pair : ∃! (a b : ℕ), 
  a + b = 2015 ∧ 
  ∃ (c : ℕ), c ≤ 9 ∧ a = 10 * b + c ∧
  a = 1832 ∧ b = 183 := by
sorry

end NUMINAMATH_CALUDE_unique_number_pair_l721_72127


namespace NUMINAMATH_CALUDE_alice_win_probability_l721_72100

-- Define the game types
inductive Move
| Rock
| Paper
| Scissors

-- Define the player types
inductive Player
| Alice
| Bob
| Other

-- Define the tournament structure
def TournamentSize : Nat := 8
def NumRounds : Nat := 3

-- Define the rules of the game
def beats (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Rock, Move.Scissors => true
  | Move.Scissors, Move.Paper => true
  | Move.Paper, Move.Rock => true
  | _, _ => false

-- Define the strategy for each player
def playerMove (p : Player) : Move :=
  match p with
  | Player.Alice => Move.Rock
  | Player.Bob => Move.Paper
  | Player.Other => Move.Scissors

-- Define the probability of Alice winning
def aliceWinProbability : Rat := 6/7

-- Theorem statement
theorem alice_win_probability :
  (TournamentSize = 8) →
  (NumRounds = 3) →
  (∀ p, playerMove p = match p with
    | Player.Alice => Move.Rock
    | Player.Bob => Move.Paper
    | Player.Other => Move.Scissors) →
  (∀ m1 m2, beats m1 m2 = match m1, m2 with
    | Move.Rock, Move.Scissors => true
    | Move.Scissors, Move.Paper => true
    | Move.Paper, Move.Rock => true
    | _, _ => false) →
  aliceWinProbability = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_alice_win_probability_l721_72100


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l721_72146

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - 2*i) / i) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l721_72146


namespace NUMINAMATH_CALUDE_prob_two_tails_after_HHT_is_correct_l721_72137

/-- A fair coin flip sequence that stops when two consecutive heads or tails are obtained -/
def CoinFlipSequence : Type := List Bool

/-- The probability of getting a specific sequence of coin flips -/
def prob_sequence (s : CoinFlipSequence) : ℚ :=
  (1 / 2) ^ s.length

/-- The probability of getting two tails after HHT -/
def prob_two_tails_after_HHT : ℚ :=
  1 / 24

/-- The theorem stating that the probability of getting two tails after HHT is 1/24 -/
theorem prob_two_tails_after_HHT_is_correct :
  prob_two_tails_after_HHT = 1 / 24 := by
  sorry

#check prob_two_tails_after_HHT_is_correct

end NUMINAMATH_CALUDE_prob_two_tails_after_HHT_is_correct_l721_72137


namespace NUMINAMATH_CALUDE_correct_balanced_redox_reaction_l721_72124

/-- Represents a chemical species in a redox reaction -/
structure ChemicalSpecies where
  formula : String
  charge : Int

/-- Represents a half-reaction in a redox reaction -/
structure HalfReaction where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies
  electrons : Int

/-- Represents a complete redox reaction -/
structure RedoxReaction where
  oxidation : HalfReaction
  reduction : HalfReaction

/-- Standard conditions in an acidic solution -/
def standardAcidicConditions : Prop := sorry

/-- Salicylic acid -/
def salicylicAcid : ChemicalSpecies := ⟨"C7H6O2", 0⟩

/-- Iron (III) ion -/
def ironIII : ChemicalSpecies := ⟨"Fe", 3⟩

/-- 2,3-dihydroxybenzoic acid -/
def dihydroxybenzoicAcid : ChemicalSpecies := ⟨"C7H6O4", 0⟩

/-- Hydrogen ion -/
def hydrogenIon : ChemicalSpecies := ⟨"H", 1⟩

/-- Iron (II) ion -/
def ironII : ChemicalSpecies := ⟨"Fe", 2⟩

/-- The balanced redox reaction between iron (III) nitrate and salicylic acid under standard acidic conditions -/
def balancedRedoxReaction (conditions : Prop) : RedoxReaction := sorry

/-- Theorem stating that the given redox reaction is the correct balanced reaction under standard acidic conditions -/
theorem correct_balanced_redox_reaction :
  standardAcidicConditions →
  balancedRedoxReaction standardAcidicConditions =
    RedoxReaction.mk
      (HalfReaction.mk [salicylicAcid] [dihydroxybenzoicAcid, hydrogenIon, hydrogenIon] 2)
      (HalfReaction.mk [ironIII, ironIII] [ironII, ironII] (-2)) :=
sorry

end NUMINAMATH_CALUDE_correct_balanced_redox_reaction_l721_72124


namespace NUMINAMATH_CALUDE_log_ratio_equals_two_thirds_l721_72159

theorem log_ratio_equals_two_thirds :
  (Real.log 9 / Real.log 8) / (Real.log 3 / Real.log 2) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_equals_two_thirds_l721_72159


namespace NUMINAMATH_CALUDE_segment_length_l721_72132

-- Define the line segment AB and points P and Q
structure Segment where
  length : ℝ

structure Point where
  position : ℝ

-- Define the ratios for P and Q
def ratio_P : ℚ := 3 / 7
def ratio_Q : ℚ := 4 / 9

-- State the theorem
theorem segment_length 
  (AB : Segment) 
  (P Q : Point) 
  (h1 : P.position ≤ Q.position) -- P and Q are on the same side of the midpoint
  (h2 : P.position = ratio_P * AB.length) -- P divides AB in ratio 3:4
  (h3 : Q.position = ratio_Q * AB.length) -- Q divides AB in ratio 4:5
  (h4 : Q.position - P.position = 3) -- PQ = 3
  : AB.length = 189 := by
  sorry


end NUMINAMATH_CALUDE_segment_length_l721_72132


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l721_72186

theorem expense_increase_percentage (monthly_salary : ℝ) (initial_savings_rate : ℝ) (new_savings : ℝ) :
  monthly_salary = 6500 →
  initial_savings_rate = 0.20 →
  new_savings = 260 →
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - new_savings
  expense_increase / initial_expenses = 0.20 := by sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l721_72186


namespace NUMINAMATH_CALUDE_rat_speed_l721_72184

/-- Proves that under given conditions, the rat's speed is 36 kmph -/
theorem rat_speed (head_start : ℝ) (catch_up_time : ℝ) (cat_speed : ℝ)
  (h1 : head_start = 6)
  (h2 : catch_up_time = 4)
  (h3 : cat_speed = 90) :
  let rat_speed := (cat_speed * catch_up_time) / (head_start + catch_up_time)
  rat_speed = 36 := by
sorry

end NUMINAMATH_CALUDE_rat_speed_l721_72184


namespace NUMINAMATH_CALUDE_ages_solution_l721_72179

/-- Represents the current ages of Justin, Angelina, and Larry -/
structure Ages where
  justin : ℝ
  angelina : ℝ
  larry : ℝ

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.angelina = ages.justin + 4 ∧
  ages.angelina + 5 = 40 ∧
  ages.larry = ages.justin + 0.5 * ages.justin

/-- The theorem to be proved -/
theorem ages_solution (ages : Ages) :
  problem_conditions ages → ages.justin = 31 ∧ ages.larry = 46.5 := by
  sorry


end NUMINAMATH_CALUDE_ages_solution_l721_72179


namespace NUMINAMATH_CALUDE_product_75_360_trailing_zeros_l721_72101

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The product of 75 and 360 has 3 trailing zeros -/
theorem product_75_360_trailing_zeros :
  trailingZeros (75 * 360) = 3 := by sorry

end NUMINAMATH_CALUDE_product_75_360_trailing_zeros_l721_72101


namespace NUMINAMATH_CALUDE_kitten_weight_l721_72107

theorem kitten_weight (kitten lighter_dog heavier_dog : ℝ) 
  (h1 : kitten + lighter_dog + heavier_dog = 36)
  (h2 : kitten + heavier_dog = 3 * lighter_dog)
  (h3 : kitten + lighter_dog = (1/2) * heavier_dog) :
  kitten = 3 := by
  sorry

end NUMINAMATH_CALUDE_kitten_weight_l721_72107


namespace NUMINAMATH_CALUDE_notebook_duration_is_seven_l721_72148

/-- Represents the number of weeks John's notebooks last -/
def notebook_duration (
  num_notebooks : ℕ
  ) (pages_per_notebook : ℕ
  ) (math_pages_per_day : ℕ
  ) (math_days_per_week : ℕ
  ) (science_pages_per_day : ℕ
  ) (science_days_per_week : ℕ
  ) (history_pages_per_day : ℕ
  ) (history_days_per_week : ℕ
  ) : ℕ :=
  let total_pages := num_notebooks * pages_per_notebook
  let pages_per_week := 
    math_pages_per_day * math_days_per_week +
    science_pages_per_day * science_days_per_week +
    history_pages_per_day * history_days_per_week
  total_pages / pages_per_week

theorem notebook_duration_is_seven :
  notebook_duration 5 40 4 3 5 2 6 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_is_seven_l721_72148


namespace NUMINAMATH_CALUDE_max_intersected_edges_is_twelve_l721_72108

/-- A regular 10-sided prism -/
structure RegularDecagonalPrism where
  -- We don't need to define the internal structure,
  -- as the problem doesn't require specific properties beyond it being a regular 10-sided prism

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane

/-- The number of edges a plane intersects with a prism -/
def intersected_edges (prism : RegularDecagonalPrism) (plane : Plane) : ℕ :=
  sorry -- Definition not provided, as it's not explicitly given in the problem

/-- The maximum number of edges that can be intersected by any plane -/
def max_intersected_edges (prism : RegularDecagonalPrism) : ℕ :=
  sorry -- Definition not provided, as it's not explicitly given in the problem

/-- Theorem: The maximum number of edges of a regular 10-sided prism 
    that can be intersected by a plane is 12 -/
theorem max_intersected_edges_is_twelve (prism : RegularDecagonalPrism) :
  max_intersected_edges prism = 12 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_max_intersected_edges_is_twelve_l721_72108


namespace NUMINAMATH_CALUDE_negation_of_positive_quadratic_l721_72103

theorem negation_of_positive_quadratic (x : ℝ) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_positive_quadratic_l721_72103


namespace NUMINAMATH_CALUDE_six_quarters_around_nickel_l721_72168

/-- Represents the arrangement of coins on a table -/
structure CoinArrangement where
  nickelDiameter : ℝ
  quarterDiameter : ℝ

/-- Calculates the maximum number of quarters that can be placed around a nickel -/
def maxQuarters (arrangement : CoinArrangement) : ℕ :=
  sorry

/-- Theorem stating that for the given coin sizes, 6 quarters can be placed around a nickel -/
theorem six_quarters_around_nickel :
  let arrangement : CoinArrangement := { nickelDiameter := 2, quarterDiameter := 2.4 }
  maxQuarters arrangement = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_quarters_around_nickel_l721_72168


namespace NUMINAMATH_CALUDE_time_to_cut_one_piece_l721_72158

-- Define the total number of pieces
def total_pieces : ℕ := 146

-- Define the total time taken in seconds
def total_time : ℕ := 580

-- Define the time taken to cut one piece
def time_per_piece : ℚ := total_time / total_pieces

-- Theorem to prove
theorem time_to_cut_one_piece : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) ∧ |time_per_piece - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_time_to_cut_one_piece_l721_72158


namespace NUMINAMATH_CALUDE_smallest_integer_inequality_l721_72173

theorem smallest_integer_inequality : ∀ x : ℤ, x + 5 < 3*x - 9 → x ≥ 8 ∧ 8 + 5 < 3*8 - 9 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_inequality_l721_72173


namespace NUMINAMATH_CALUDE_average_speed_inequality_l721_72154

theorem average_speed_inequality (a b v : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a < b) 
  (hv : v = (2 * a * b) / (a + b)) : 
  a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_average_speed_inequality_l721_72154


namespace NUMINAMATH_CALUDE_decreasing_even_shifted_function_property_l721_72113

/-- A function that is decreasing on (8, +∞) and f(x+8) is even -/
def DecreasingEvenShiftedFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 8 ∧ y > 8 ∧ x > y → f x < f y) ∧
  (∀ x, f (x + 8) = f (-x + 8))

theorem decreasing_even_shifted_function_property
  (f : ℝ → ℝ) (h : DecreasingEvenShiftedFunction f) :
  f 7 > f 10 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_even_shifted_function_property_l721_72113


namespace NUMINAMATH_CALUDE_expression_evaluation_l721_72141

theorem expression_evaluation : 4^3 - 4 * 4^2 + 6 * 4 - 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l721_72141


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l721_72165

theorem larger_solution_quadratic_equation :
  ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 13*x + 36 = 0 ∧
  y^2 - 13*y + 36 = 0 ∧
  (∀ z : ℝ, z^2 - 13*z + 36 = 0 → z = x ∨ z = y) ∧
  max x y = 9 := by
sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l721_72165


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l721_72193

theorem subtraction_of_decimals : 3.75 - 2.18 = 1.57 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l721_72193


namespace NUMINAMATH_CALUDE_trapezoid_longest_diagonal_lower_bound_trapezoid_longest_diagonal_lower_bound_tight_l721_72161

/-- A trapezoid with area 1 -/
structure Trapezoid :=
  (a b h : ℝ)  -- lengths of bases and height
  (d₁ d₂ : ℝ)  -- lengths of diagonals
  (area_eq : (a + b) * h / 2 = 1)
  (d₁_ge_d₂ : d₁ ≥ d₂)

/-- The longest diagonal of a trapezoid with area 1 is at least √2 -/
theorem trapezoid_longest_diagonal_lower_bound (T : Trapezoid) : 
  T.d₁ ≥ Real.sqrt 2 := by sorry

/-- There exists a trapezoid with area 1 whose longest diagonal is exactly √2 -/
theorem trapezoid_longest_diagonal_lower_bound_tight : 
  ∃ T : Trapezoid, T.d₁ = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_longest_diagonal_lower_bound_trapezoid_longest_diagonal_lower_bound_tight_l721_72161


namespace NUMINAMATH_CALUDE_first_term_values_l721_72145

def fibonacci_like_sequence (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n + 1)

theorem first_term_values (a b : ℕ) :
  fibonacci_like_sequence a b 2 = 7 ∧
  fibonacci_like_sequence a b 2013 % 4 = 1 →
  a = 1 ∨ a = 5 := by
sorry

end NUMINAMATH_CALUDE_first_term_values_l721_72145


namespace NUMINAMATH_CALUDE_some_number_equation_l721_72128

theorem some_number_equation (y : ℝ) : 
  ∃ (n : ℝ), n * (1 + y) + 17 = n * (-1 + y) - 21 ∧ n = -19 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l721_72128


namespace NUMINAMATH_CALUDE_income_record_l721_72125

/-- Represents the recording of a financial transaction -/
def record (amount : ℤ) : ℤ := amount

/-- An expenditure is recorded as a negative number -/
axiom expenditure_record : record (-200) = -200

/-- Theorem: An income is recorded as a positive number -/
theorem income_record : record 60 = 60 := by sorry

end NUMINAMATH_CALUDE_income_record_l721_72125


namespace NUMINAMATH_CALUDE_aunt_wang_lilies_l721_72135

theorem aunt_wang_lilies (rose_cost lily_cost roses_bought total_spent : ℕ) : 
  rose_cost = 5 →
  lily_cost = 9 →
  roses_bought = 2 →
  total_spent = 55 →
  (total_spent - rose_cost * roses_bought) / lily_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_aunt_wang_lilies_l721_72135


namespace NUMINAMATH_CALUDE_alpha_value_l721_72166

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).im = 0 ∧ (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).im = 0 ∧ (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_alpha_value_l721_72166


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l721_72172

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l721_72172


namespace NUMINAMATH_CALUDE_distinct_differences_sequence_length_l721_72104

def is_valid_n (n : ℕ) : Prop :=
  ∃ k : ℕ+, (n = 4 * k ∨ n = 4 * k - 1)

theorem distinct_differences_sequence_length {n : ℕ} (h_n : n ≥ 3) :
  (∃ (a : ℕ → ℝ), (∀ i j : Fin n, i ≠ j → |a i - a (i + 1)| ≠ |a j - a (j + 1)|)) →
  is_valid_n n :=
sorry

end NUMINAMATH_CALUDE_distinct_differences_sequence_length_l721_72104


namespace NUMINAMATH_CALUDE_fraction_meaningful_l721_72150

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 2) / (x - 3)) ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l721_72150


namespace NUMINAMATH_CALUDE_evaluate_expression_l721_72167

theorem evaluate_expression : 3^(1^(2^8)) + ((3^1)^2)^4 = 6564 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l721_72167


namespace NUMINAMATH_CALUDE_not_always_valid_solution_set_l721_72195

theorem not_always_valid_solution_set (a b : ℝ) (h : b ≠ 0) :
  ¬ (∀ x, x ∈ Set.Ioi (b / a) ↔ a * x + b > 0) :=
sorry

end NUMINAMATH_CALUDE_not_always_valid_solution_set_l721_72195


namespace NUMINAMATH_CALUDE_winnie_keeps_remainder_l721_72196

/-- The number of balloons Winnie keeps for herself -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

/-- The total number of balloons Winnie has -/
def total_balloons : ℕ := 17 + 33 + 65 + 83

/-- The number of friends Winnie has -/
def num_friends : ℕ := 10

theorem winnie_keeps_remainder :
  balloons_kept total_balloons num_friends = 8 :=
sorry

end NUMINAMATH_CALUDE_winnie_keeps_remainder_l721_72196


namespace NUMINAMATH_CALUDE_polynomial_composite_l721_72175

def P (x : ℕ) : ℕ := 4*x^3 + 6*x^2 + 4*x + 1

theorem polynomial_composite : ∀ x : ℕ, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ P x = a * b :=
sorry

end NUMINAMATH_CALUDE_polynomial_composite_l721_72175


namespace NUMINAMATH_CALUDE_monomial_combination_l721_72149

theorem monomial_combination (m n : ℤ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 3 * a^4 * b^(n+2) = 5 * a^(m-1) * b^(2*n+3)) → 
  m + n = 4 := by
sorry

end NUMINAMATH_CALUDE_monomial_combination_l721_72149


namespace NUMINAMATH_CALUDE_adult_meal_cost_l721_72144

/-- Proves that the cost of each adult meal is $6 given the conditions of the restaurant bill. -/
theorem adult_meal_cost (num_adults num_children : ℕ) (child_meal_cost soda_cost total_bill : ℚ) :
  num_adults = 6 →
  num_children = 2 →
  child_meal_cost = 4 →
  soda_cost = 2 →
  total_bill = 60 →
  ∃ (adult_meal_cost : ℚ),
    adult_meal_cost * num_adults + child_meal_cost * num_children + soda_cost * (num_adults + num_children) = total_bill ∧
    adult_meal_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l721_72144


namespace NUMINAMATH_CALUDE_statement_b_statement_c_not_statement_a_not_statement_d_l721_72174

-- Statement B
theorem statement_b (a b : ℝ) : a > b → a - 1 > b - 2 := by sorry

-- Statement C
theorem statement_c (a b c : ℝ) (h : c ≠ 0) : a / c^2 > b / c^2 → a > b := by sorry

-- Disproof of Statement A
theorem not_statement_a : ¬ (∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by sorry

-- Disproof of Statement D
theorem not_statement_d : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by sorry

end NUMINAMATH_CALUDE_statement_b_statement_c_not_statement_a_not_statement_d_l721_72174


namespace NUMINAMATH_CALUDE_trigonometric_problem_l721_72117

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = 4 / 5)
  (h4 : Real.cos (α + β) = 5 / 13) :
  (Real.cos β = 63 / 65) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l721_72117


namespace NUMINAMATH_CALUDE_original_production_was_125_l721_72118

/-- Represents the clothing production problem --/
structure ClothingProduction where
  plannedDays : ℕ
  actualDailyProduction : ℕ
  daysAheadOfSchedule : ℕ

/-- Calculates the original planned daily production --/
def originalPlannedProduction (cp : ClothingProduction) : ℚ :=
  (cp.actualDailyProduction * (cp.plannedDays - cp.daysAheadOfSchedule)) / cp.plannedDays

/-- Theorem stating that the original planned production was 125 sets per day --/
theorem original_production_was_125 (cp : ClothingProduction) 
  (h1 : cp.plannedDays = 30)
  (h2 : cp.actualDailyProduction = 150)
  (h3 : cp.daysAheadOfSchedule = 5) :
  originalPlannedProduction cp = 125 := by
  sorry

#eval originalPlannedProduction ⟨30, 150, 5⟩

end NUMINAMATH_CALUDE_original_production_was_125_l721_72118


namespace NUMINAMATH_CALUDE_laura_pants_purchase_l721_72111

def pants_cost : ℕ := 54
def shirt_cost : ℕ := 33
def num_shirts : ℕ := 4
def money_given : ℕ := 250
def change_received : ℕ := 10

theorem laura_pants_purchase :
  (money_given - change_received - num_shirts * shirt_cost) / pants_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_laura_pants_purchase_l721_72111


namespace NUMINAMATH_CALUDE_tony_graduate_degree_time_l721_72129

/-- Time spent on graduate degree in physics -/
def graduate_degree_time (first_degree_time additional_degree_time number_of_additional_degrees total_school_time : ℕ) : ℕ :=
  total_school_time - (first_degree_time + additional_degree_time * number_of_additional_degrees)

/-- Theorem stating that Tony's graduate degree time is 2 years -/
theorem tony_graduate_degree_time :
  graduate_degree_time 4 4 2 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tony_graduate_degree_time_l721_72129
