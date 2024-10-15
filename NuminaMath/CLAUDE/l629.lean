import Mathlib

namespace NUMINAMATH_CALUDE_max_rectangle_area_l629_62943

/-- Given a string of length 32 cm, the maximum area of a rectangle that can be formed is 64 cm². -/
theorem max_rectangle_area (string_length : ℝ) (h : string_length = 32) : 
  (∀ w h : ℝ, w > 0 → h > 0 → 2*w + 2*h ≤ string_length → w * h ≤ 64) ∧ 
  (∃ w h : ℝ, w > 0 ∧ h > 0 ∧ 2*w + 2*h = string_length ∧ w * h = 64) :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l629_62943


namespace NUMINAMATH_CALUDE_complex_addition_point_l629_62909

/-- A complex number corresponding to a point in the complex plane -/
def complex_point (x y : ℝ) : ℂ := x + y * Complex.I

/-- The theorem stating that if z corresponds to (2,5), then 1+z corresponds to (3,5) -/
theorem complex_addition_point (z : ℂ) (h : z = complex_point 2 5) :
  1 + z = complex_point 3 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_point_l629_62909


namespace NUMINAMATH_CALUDE_c_range_l629_62995

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → (c - 1) * x + 1 < (c - 1) * y + 1

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 - x + c > 0

theorem c_range (c : ℝ) (hp : p c) (hq : q c) : c > 1 := by
  sorry

end NUMINAMATH_CALUDE_c_range_l629_62995


namespace NUMINAMATH_CALUDE_robin_gum_packages_l629_62999

/-- Represents the number of pieces of gum in each package -/
def pieces_per_package : ℕ := 23

/-- Represents the number of extra pieces of gum Robin has -/
def extra_pieces : ℕ := 8

/-- Represents the total number of pieces of gum Robin has -/
def total_pieces : ℕ := 997

/-- Represents the number of packages Robin has -/
def num_packages : ℕ := (total_pieces - extra_pieces) / pieces_per_package

theorem robin_gum_packages : num_packages = 43 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l629_62999


namespace NUMINAMATH_CALUDE_repeating_decimal_56_l629_62954

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56 : RepeatingDecimal 5 6 = 56 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_l629_62954


namespace NUMINAMATH_CALUDE_percentage_10_years_or_more_is_correct_l629_62936

/-- Represents the employment distribution at Apex Innovations -/
structure EmploymentDistribution (X : ℕ) :=
  (less_than_2_years : ℕ := 7 * X)
  (two_to_4_years : ℕ := 4 * X)
  (four_to_6_years : ℕ := 3 * X)
  (six_to_8_years : ℕ := 3 * X)
  (eight_to_10_years : ℕ := 2 * X)
  (ten_to_12_years : ℕ := 2 * X)
  (twelve_to_14_years : ℕ := X)
  (fourteen_to_16_years : ℕ := X)
  (sixteen_to_18_years : ℕ := X)

/-- Calculates the percentage of employees who have worked for 10 years or more -/
def percentage_10_years_or_more (dist : EmploymentDistribution X) : ℚ :=
  let total_employees := 23 * X
  let employees_10_years_or_more := 5 * X
  (employees_10_years_or_more : ℚ) / total_employees * 100

/-- Theorem stating that the percentage of employees who have worked for 10 years or more is (5/23) * 100 -/
theorem percentage_10_years_or_more_is_correct (X : ℕ) (dist : EmploymentDistribution X) :
  percentage_10_years_or_more dist = 5 / 23 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_10_years_or_more_is_correct_l629_62936


namespace NUMINAMATH_CALUDE_xia_shared_hundred_stickers_l629_62934

/-- The number of stickers Xia shared with her friends -/
def shared_stickers (total : ℕ) (sheets_left : ℕ) (stickers_per_sheet : ℕ) : ℕ :=
  total - (sheets_left * stickers_per_sheet)

/-- Theorem stating that Xia shared 100 stickers with her friends -/
theorem xia_shared_hundred_stickers :
  shared_stickers 150 5 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_xia_shared_hundred_stickers_l629_62934


namespace NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l629_62929

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) (h1 : a = 5) (h2 : b = 5) (h3 : c = 4) :
  let r := (a * b * c) / (4 * (1/2 * c * (a^2 - (c/2)^2).sqrt))
  π * r^2 = (13125/1764) * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l629_62929


namespace NUMINAMATH_CALUDE_doubling_function_m_range_l629_62900

/-- A function f is a "doubling function" if there exists an interval [a,b] in its domain
    such that the range of f on [a,b] is [2a,2b] -/
def DoublingFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (2*a) (2*b)) ∧
    (∀ y ∈ Set.Icc (2*a) (2*b), ∃ x ∈ Set.Icc a b, f x = y)

/-- The main theorem stating that for f(x) = ln(e^x + m) to be a doubling function,
    m must be in the range (-1/4, 0) -/
theorem doubling_function_m_range :
  ∀ m : ℝ, (DoublingFunction (fun x ↦ Real.log (Real.exp x + m))) ↔ -1/4 < m ∧ m < 0 := by
  sorry


end NUMINAMATH_CALUDE_doubling_function_m_range_l629_62900


namespace NUMINAMATH_CALUDE_stone_slab_length_l629_62916

theorem stone_slab_length (n : ℕ) (total_area : ℝ) (h1 : n = 30) (h2 : total_area = 120) :
  ∃ (slab_length : ℝ), slab_length > 0 ∧ n * slab_length^2 = total_area ∧ slab_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_length_l629_62916


namespace NUMINAMATH_CALUDE_tourists_knowing_both_languages_l629_62914

theorem tourists_knowing_both_languages 
  (total : ℕ) 
  (neither : ℕ) 
  (german : ℕ) 
  (french : ℕ) 
  (h1 : total = 100) 
  (h2 : neither = 10) 
  (h3 : german = 76) 
  (h4 : french = 83) : 
  total - neither = german + french - 69 := by
sorry

end NUMINAMATH_CALUDE_tourists_knowing_both_languages_l629_62914


namespace NUMINAMATH_CALUDE_special_matrix_determinant_l629_62904

/-- The determinant of a special n × n matrix A with elements a_{ij} = |i - j| -/
theorem special_matrix_determinant (n : ℕ) (hn : n > 0) :
  let A : Matrix (Fin n) (Fin n) ℤ := λ i j => |i.val - j.val|
  Matrix.det A = (-1 : ℤ)^(n-1) * (n - 1) * 2^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_determinant_l629_62904


namespace NUMINAMATH_CALUDE_johnnys_age_reference_l629_62989

/-- Proves that Johnny was referring to 3 years ago -/
theorem johnnys_age_reference : 
  ∀ (current_age : ℕ) (years_ago : ℕ),
  current_age = 8 →
  current_age + 2 = 2 * (current_age - years_ago) →
  years_ago = 3 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_age_reference_l629_62989


namespace NUMINAMATH_CALUDE_cube_root_unity_inverse_l629_62983

/-- Given a complex cube root of unity ω, prove that (ω - ω⁻¹)⁻¹ = -(1 + 2ω²)/5 -/
theorem cube_root_unity_inverse (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (ω - ω⁻¹)⁻¹ = -(1 + 2*ω^2)/5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_inverse_l629_62983


namespace NUMINAMATH_CALUDE_simplify_expression_l629_62930

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    c.val = 24 ∧
    a.val = 56 ∧
    b.val = 54 ∧
    (∀ (x y z : ℕ+),
      Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) =
      (x.val * Real.sqrt 6 + y.val * Real.sqrt 8) / z.val →
      z.val ≥ c.val) ∧
    Real.sqrt 6 + (1 / Real.sqrt 6) + Real.sqrt 8 + (1 / Real.sqrt 8) =
    (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l629_62930


namespace NUMINAMATH_CALUDE_intersection_point_l629_62912

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 5

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line y = 2x - 5 and the y-axis is (0, -5) -/
theorem intersection_point : 
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l629_62912


namespace NUMINAMATH_CALUDE_bart_money_theorem_l629_62973

theorem bart_money_theorem :
  ∃ m : ℕ, m > 0 ∧ ∀ n : ℕ, n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b := by
  sorry

end NUMINAMATH_CALUDE_bart_money_theorem_l629_62973


namespace NUMINAMATH_CALUDE_boxes_to_brother_l629_62967

def total_boxes : ℕ := 45
def boxes_to_sister : ℕ := 9
def boxes_to_cousin : ℕ := 7
def boxes_left : ℕ := 17

theorem boxes_to_brother :
  total_boxes - boxes_to_sister - boxes_to_cousin - boxes_left = 12 := by
  sorry

end NUMINAMATH_CALUDE_boxes_to_brother_l629_62967


namespace NUMINAMATH_CALUDE_square_root_16_divided_by_2_l629_62996

theorem square_root_16_divided_by_2 : Real.sqrt 16 / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_16_divided_by_2_l629_62996


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l629_62963

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑a - Complex.I) * (1 + Complex.I) = Complex.I * (Complex.ofReal (a - 1)) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l629_62963


namespace NUMINAMATH_CALUDE_min_p_plus_q_l629_62924

theorem min_p_plus_q (p q : ℕ+) (h : 108 * p = q^3) : 
  ∀ (p' q' : ℕ+), 108 * p' = q'^3 → p + q ≤ p' + q' :=
by sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l629_62924


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l629_62969

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (45^125 + 7^87) % 10 = n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l629_62969


namespace NUMINAMATH_CALUDE_petyas_chips_l629_62921

theorem petyas_chips (x : ℕ) (y : ℕ) : 
  y = x - 2 → -- The side of the square has 2 fewer chips than the triangle
  3 * x - 3 = 4 * y - 4 → -- Total chips are the same for both shapes
  3 * x - 3 = 24 -- The total number of chips is 24
  := by sorry

end NUMINAMATH_CALUDE_petyas_chips_l629_62921


namespace NUMINAMATH_CALUDE_inequality_solution_set_l629_62960

theorem inequality_solution_set (x : ℝ) :
  (4 * x^3 + 9 * x^2 - 6 * x < 2) ↔ ((-2 < x ∧ x < -1) ∨ (-1 < x ∧ x < 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l629_62960


namespace NUMINAMATH_CALUDE_polynomial_with_prime_roots_l629_62970

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem polynomial_with_prime_roots (s : ℕ) :
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 7 ∧ p * q = s) →
  s = 10 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_with_prime_roots_l629_62970


namespace NUMINAMATH_CALUDE_no_solution_condition_l629_62905

theorem no_solution_condition (a : ℝ) :
  (∀ x : ℝ, 9 * |x - 4*a| + |x - a^2| + 8*x - 2*a ≠ 0) ↔ (a < -26 ∨ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l629_62905


namespace NUMINAMATH_CALUDE_bin_game_expected_win_l629_62984

/-- The number of yellow balls in the bin -/
def yellow_balls : ℕ := 7

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 3

/-- The amount won when drawing a yellow ball -/
def yellow_win : ℚ := 3

/-- The amount lost when drawing a blue ball -/
def blue_loss : ℚ := 1

/-- The expected amount won from playing the game -/
def expected_win : ℚ := 1

/-- Theorem stating that the expected amount won is 1 dollar
    given the specified number of yellow and blue balls and win/loss amounts -/
theorem bin_game_expected_win :
  (yellow_balls * yellow_win + blue_balls * (-blue_loss)) / (yellow_balls + blue_balls) = expected_win :=
sorry

end NUMINAMATH_CALUDE_bin_game_expected_win_l629_62984


namespace NUMINAMATH_CALUDE_cyclic_inequality_l629_62976

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (1/3) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l629_62976


namespace NUMINAMATH_CALUDE_right_triangle_arm_square_l629_62987

theorem right_triangle_arm_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arm_square_l629_62987


namespace NUMINAMATH_CALUDE_second_rectangle_height_l629_62992

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Proves that the height of the second rectangle is 6 inches -/
theorem second_rectangle_height (r1 r2 : Rectangle) 
  (h1 : r1.width = 4)
  (h2 : r1.height = 5)
  (h3 : r2.width = 3)
  (h4 : area r1 = area r2 + 2) : 
  r2.height = 6 := by
  sorry

#check second_rectangle_height

end NUMINAMATH_CALUDE_second_rectangle_height_l629_62992


namespace NUMINAMATH_CALUDE_seven_factorial_mod_thirteen_l629_62958

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem seven_factorial_mod_thirteen : factorial 7 % 13 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_factorial_mod_thirteen_l629_62958


namespace NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l629_62942

theorem perfect_cubes_between_powers_of_three : 
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  (Finset.filter (fun n => lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) 
    (Finset.range (upper_bound + 1))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l629_62942


namespace NUMINAMATH_CALUDE_barons_claim_correct_l629_62997

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two different 10-digit numbers satisfying the Baron's claim -/
theorem barons_claim_correct : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    10^9 ≤ a ∧ a < 10^10 ∧
    10^9 ≤ b ∧ b < 10^10 ∧
    a % 10 ≠ 0 ∧
    b % 10 ≠ 0 ∧
    a + sum_of_digits (a^2) = b + sum_of_digits (b^2) :=
by sorry

end NUMINAMATH_CALUDE_barons_claim_correct_l629_62997


namespace NUMINAMATH_CALUDE_volume_of_smaller_cube_l629_62906

/-- Given that eight equal-sized cubes form a larger cube with a surface area of 1536 cm²,
    prove that the volume of each smaller cube is 512 cm³. -/
theorem volume_of_smaller_cube (surface_area : ℝ) (num_small_cubes : ℕ) :
  surface_area = 1536 →
  num_small_cubes = 8 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    surface_area = 6 * side_length^2 ∧
    (side_length / 2)^3 = 512 :=
sorry

end NUMINAMATH_CALUDE_volume_of_smaller_cube_l629_62906


namespace NUMINAMATH_CALUDE_sum_from_simple_interest_and_true_discount_l629_62993

/-- Given a sum, time, and rate, if the simple interest is 85 and the true discount is 80, then the sum is 1360 -/
theorem sum_from_simple_interest_and_true_discount 
  (P T R : ℝ) 
  (h_simple_interest : (P * T * R) / 100 = 85)
  (h_true_discount : (P * T * R) / (100 + T * R) = 80) :
  P = 1360 := by
  sorry

end NUMINAMATH_CALUDE_sum_from_simple_interest_and_true_discount_l629_62993


namespace NUMINAMATH_CALUDE_fourth_cat_weight_proof_l629_62952

/-- The weight of the fourth cat given the weights of three cats and the average weight of all four cats -/
def fourth_cat_weight (weight1 weight2 weight3 average_weight : ℝ) : ℝ :=
  4 * average_weight - (weight1 + weight2 + weight3)

/-- Theorem stating that given the specific weights of three cats and the average weight of all four cats, the weight of the fourth cat is 9.3 pounds -/
theorem fourth_cat_weight_proof :
  fourth_cat_weight 12 12 14.7 12 = 9.3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_cat_weight_proof_l629_62952


namespace NUMINAMATH_CALUDE_integral_bound_for_differentiable_function_l629_62978

open Set
open MeasureTheory
open Interval
open Real

theorem integral_bound_for_differentiable_function 
  (f : ℝ → ℝ) 
  (hf_diff : DifferentiableOn ℝ f (Icc 0 1))
  (hf_zero : f 0 = 0 ∧ f 1 = 0)
  (hf_deriv_bound : ∀ x ∈ Icc 0 1, abs (deriv f x) ≤ 1) :
  abs (∫ x in Icc 0 1, f x) < (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_integral_bound_for_differentiable_function_l629_62978


namespace NUMINAMATH_CALUDE_intersection_point_median_altitude_l629_62907

/-- Given a triangle ABC with vertices A(5,1), B(-1,-3), and C(4,3),
    the intersection point of the median CM and altitude BN
    has coordinates (5/3, -5/3). -/
theorem intersection_point_median_altitude (A B C M N : ℝ × ℝ) :
  A = (5, 1) →
  B = (-1, -3) →
  C = (4, 3) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (N.2 - B.2) * (C.1 - A.1) = (C.2 - A.2) * (N.1 - B.1) →
  (∃ t : ℝ, C + t • (M - C) = N) →
  N = (5/3, -5/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_median_altitude_l629_62907


namespace NUMINAMATH_CALUDE_set_operations_l629_62965

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}

theorem set_operations :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 2 < x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l629_62965


namespace NUMINAMATH_CALUDE_initial_candies_count_l629_62917

def candies_remaining (initial : ℕ) (day : ℕ) : ℤ :=
  match day with
  | 0 => initial
  | n + 1 => (candies_remaining initial n / 2 : ℤ) - 1

theorem initial_candies_count :
  ∃ initial : ℕ, 
    candies_remaining initial 3 = 0 ∧ 
    ∀ d : ℕ, d < 3 → candies_remaining initial d > 0 ∧ 
    initial = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_candies_count_l629_62917


namespace NUMINAMATH_CALUDE_rectangle_area_l629_62938

theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 52) (h2 : width = 11) :
  (width * (perimeter / 2 - width)) = 165 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l629_62938


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l629_62985

/-- Given an arithmetic sequence {a_n} where a₂ = 9 and a₅ = 33, 
    prove that the common difference is 8. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Definition of arithmetic sequence
  (h2 : a 2 = 9) -- Given: a₂ = 9
  (h3 : a 5 = 33) -- Given: a₅ = 33
  : a 2 - a 1 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l629_62985


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l629_62964

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 5 = 0 ∧ x₂^2 + m*x₂ - 5 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l629_62964


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l629_62991

theorem unique_function_satisfying_condition :
  ∃! f : ℝ → ℝ, (∀ x y z : ℝ, f (x * y) + f (x * z) + f x * f (y * z) ≥ 3) ∧
  (∀ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l629_62991


namespace NUMINAMATH_CALUDE_initial_group_size_l629_62933

theorem initial_group_size (initial_avg : ℝ) (new_avg : ℝ) (weight1 : ℝ) (weight2 : ℝ) :
  initial_avg = 48 →
  new_avg = 51 →
  weight1 = 78 →
  weight2 = 93 →
  ∃ n : ℕ, n * initial_avg + weight1 + weight2 = (n + 2) * new_avg ∧ n = 23 :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l629_62933


namespace NUMINAMATH_CALUDE_cindy_solution_l629_62966

def cindy_problem (x : ℝ) : Prop :=
  (x - 12) / 4 = 32 →
  round ((x - 7) / 5) = 27

theorem cindy_solution : ∃ x : ℝ, cindy_problem x := by
  sorry

end NUMINAMATH_CALUDE_cindy_solution_l629_62966


namespace NUMINAMATH_CALUDE_product_of_solutions_l629_62982

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|6 * x₁| + 5 = 47) → (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l629_62982


namespace NUMINAMATH_CALUDE_amelia_monday_distance_l629_62941

theorem amelia_monday_distance (total_distance tuesday_distance remaining_distance : ℕ) 
  (h1 : total_distance = 8205)
  (h2 : tuesday_distance = 582)
  (h3 : remaining_distance = 6716) :
  total_distance = tuesday_distance + remaining_distance + 907 := by
  sorry

end NUMINAMATH_CALUDE_amelia_monday_distance_l629_62941


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l629_62944

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l629_62944


namespace NUMINAMATH_CALUDE_money_combination_l629_62979

theorem money_combination (raquel nataly tom sam : ℝ) : 
  tom = (1/4) * nataly →
  nataly = 3 * raquel →
  sam = 2 * nataly →
  raquel = 40 →
  tom + raquel + nataly + sam = 430 :=
by sorry

end NUMINAMATH_CALUDE_money_combination_l629_62979


namespace NUMINAMATH_CALUDE_function_passes_through_point_l629_62951

theorem function_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x - 1)
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l629_62951


namespace NUMINAMATH_CALUDE_mark_bought_three_weeks_of_food_l629_62994

/-- Calculates the number of weeks of dog food purchased given the total cost,
    puppy cost, daily food consumption, bag size, and bag cost. -/
def weeks_of_food (total_cost puppy_cost daily_food_cups bag_size_cups bag_cost : ℚ) : ℚ :=
  let food_cost := total_cost - puppy_cost
  let bags_bought := food_cost / bag_cost
  let total_cups := bags_bought * bag_size_cups
  let days_of_food := total_cups / daily_food_cups
  days_of_food / 7

/-- Theorem stating that under the given conditions, Mark bought food for 3 weeks. -/
theorem mark_bought_three_weeks_of_food :
  weeks_of_food 14 10 (1/3) (7/2) 2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_mark_bought_three_weeks_of_food_l629_62994


namespace NUMINAMATH_CALUDE_prob_X_eq_three_l629_62977

/-- A random variable X following a binomial distribution B(6, 1/2) -/
def X : ℕ → ℝ := sorry

/-- The probability mass function for X -/
def pmf (k : ℕ) : ℝ := sorry

/-- Theorem: The probability of X = 3 is 5/16 -/
theorem prob_X_eq_three : pmf 3 = 5/16 := by sorry

end NUMINAMATH_CALUDE_prob_X_eq_three_l629_62977


namespace NUMINAMATH_CALUDE_triangle_sum_vertices_sides_l629_62918

/-- Definition of a triangle -/
structure Triangle where
  vertices : ℕ
  sides : ℕ

/-- The sum of vertices and sides of a triangle is 6 -/
theorem triangle_sum_vertices_sides : ∀ t : Triangle, t.vertices + t.sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_vertices_sides_l629_62918


namespace NUMINAMATH_CALUDE_candy_box_problem_l629_62901

theorem candy_box_problem (milk_chocolate dark_chocolate milk_almond : ℕ) 
  (h1 : milk_chocolate = 25)
  (h2 : dark_chocolate = 25)
  (h3 : milk_almond = 25)
  (h4 : ∀ chocolate_type, chocolate_type = milk_chocolate ∨ 
                          chocolate_type = dark_chocolate ∨ 
                          chocolate_type = milk_almond ∨ 
                          chocolate_type = white_chocolate →
        chocolate_type = (milk_chocolate + dark_chocolate + milk_almond + white_chocolate) / 4) :
  white_chocolate = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_box_problem_l629_62901


namespace NUMINAMATH_CALUDE_production_rates_satisfy_conditions_unique_solution_l629_62932

/-- The number of parts person A can make per day -/
def parts_per_day_A : ℕ := 60

/-- The number of parts person B can make per day -/
def parts_per_day_B : ℕ := 80

/-- The total number of machine parts -/
def total_parts : ℕ := 400

/-- Theorem stating that the given production rates satisfy the problem conditions -/
theorem production_rates_satisfy_conditions :
  (parts_per_day_A + 2 * parts_per_day_A + 2 * parts_per_day_B = total_parts - 60) ∧
  (3 * parts_per_day_A + 3 * parts_per_day_B = total_parts + 20) := by
  sorry

/-- Theorem proving the uniqueness of the solution -/
theorem unique_solution (a b : ℕ) 
  (h1 : a + 2 * a + 2 * b = total_parts - 60)
  (h2 : 3 * a + 3 * b = total_parts + 20) :
  a = parts_per_day_A ∧ b = parts_per_day_B := by
  sorry

end NUMINAMATH_CALUDE_production_rates_satisfy_conditions_unique_solution_l629_62932


namespace NUMINAMATH_CALUDE_jordan_weight_change_l629_62945

def weight_change (initial_weight : ℕ) (loss_first_4_weeks : ℕ) (loss_week_5 : ℕ) 
  (loss_next_7_weeks : ℕ) (gain_week_13 : ℕ) : ℕ :=
  initial_weight - (4 * loss_first_4_weeks + loss_week_5 + 7 * loss_next_7_weeks - gain_week_13)

theorem jordan_weight_change :
  weight_change 250 3 5 2 2 = 221 :=
by sorry

end NUMINAMATH_CALUDE_jordan_weight_change_l629_62945


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l629_62947

theorem trigonometric_expression_evaluation (α : Real) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l629_62947


namespace NUMINAMATH_CALUDE_rhea_count_l629_62920

theorem rhea_count (num_wombats : ℕ) (wombat_claws : ℕ) (rhea_claws : ℕ) (total_claws : ℕ) : 
  num_wombats = 9 →
  wombat_claws = 4 →
  rhea_claws = 1 →
  total_claws = 39 →
  total_claws = num_wombats * wombat_claws + (total_claws - num_wombats * wombat_claws) →
  (total_claws - num_wombats * wombat_claws) / rhea_claws = 3 := by
sorry

end NUMINAMATH_CALUDE_rhea_count_l629_62920


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l629_62935

theorem cos_pi_minus_2alpha (α : Real) (h : Real.sin α = 2/3) :
  Real.cos (π - 2*α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l629_62935


namespace NUMINAMATH_CALUDE_diamond_ratio_sixteen_two_over_two_sixteen_l629_62956

-- Define the diamond operation
def diamond (n m : ℕ) : ℕ := n^4 * m^2

-- State the theorem
theorem diamond_ratio_sixteen_two_over_two_sixteen : 
  (diamond 16 2) / (diamond 2 16) = 64 := by sorry

end NUMINAMATH_CALUDE_diamond_ratio_sixteen_two_over_two_sixteen_l629_62956


namespace NUMINAMATH_CALUDE_prob_at_least_one_one_correct_l629_62915

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one die showing a 1 when two fair 8-sided dice are rolled -/
def prob_at_least_one_one : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing a 1 
    when two fair 8-sided dice are rolled is 15/64 -/
theorem prob_at_least_one_one_correct : 
  prob_at_least_one_one = 1 - (num_sides - 1)^2 / num_sides^2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_one_correct_l629_62915


namespace NUMINAMATH_CALUDE_birth_interval_l629_62946

/-- Given 5 children born at equal intervals, with the youngest being 6 years old
    and the sum of all ages being 60 years, the interval between births is 3.6 years. -/
theorem birth_interval (n : ℕ) (youngest_age sum_ages : ℝ) (h1 : n = 5) (h2 : youngest_age = 6)
    (h3 : sum_ages = 60) : ∃ interval : ℝ,
  interval = 3.6 ∧
  sum_ages = n * youngest_age + (interval * (n * (n - 1)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_birth_interval_l629_62946


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_intercept_l629_62902

/-- Given a line that intersects a circle centered at the origin, 
    prove that the line forms an isosceles right triangle with the origin 
    if and only if the absolute value of its y-intercept equals √2. -/
theorem isosceles_right_triangle_intercept (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A.1 - A.2 + a = 0 ∧ 
    B.1 - B.2 + a = 0 ∧ 
    A.1^2 + A.2^2 = 2 ∧ 
    B.1^2 + B.2^2 = 2 ∧ 
    (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2 ∧ 
    (A.1 - 0) * (B.1 - 0) + (A.2 - 0) * (B.2 - 0) = 0) ↔ 
  |a| = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_intercept_l629_62902


namespace NUMINAMATH_CALUDE_percent_of_number_zero_point_one_percent_of_12356_l629_62955

theorem percent_of_number (x : ℝ) : x * 0.001 = 0.001 * x := by sorry

theorem zero_point_one_percent_of_12356 : (12356 : ℝ) * 0.001 = 12.356 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_zero_point_one_percent_of_12356_l629_62955


namespace NUMINAMATH_CALUDE_triangle_angle_b_l629_62908

theorem triangle_angle_b (a b c : ℝ) (A B C : ℝ) :
  c = 2 * b * Real.cos B →
  C = 2 * Real.pi / 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  B = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_b_l629_62908


namespace NUMINAMATH_CALUDE_table_runner_coverage_l629_62950

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (two_layer_area : ℝ) (four_layer_area : ℝ) 
  (h1 : total_runner_area = 360)
  (h2 : table_area = 250)
  (h3 : two_layer_area = 35)
  (h4 : four_layer_area = 15)
  (h5 : 0.9 * table_area = two_layer_area + three_layer_area + four_layer_area + one_layer_area)
  (h6 : total_runner_area = one_layer_area + 2 * two_layer_area + 3 * three_layer_area + 4 * four_layer_area) :
  three_layer_area = 65 := by
  sorry


end NUMINAMATH_CALUDE_table_runner_coverage_l629_62950


namespace NUMINAMATH_CALUDE_girls_average_height_l629_62937

theorem girls_average_height
  (num_boys : ℕ)
  (num_girls : ℕ)
  (total_students : ℕ)
  (avg_height_all : ℝ)
  (avg_height_boys : ℝ)
  (h1 : num_boys = 12)
  (h2 : num_girls = 10)
  (h3 : total_students = num_boys + num_girls)
  (h4 : avg_height_all = 103)
  (h5 : avg_height_boys = 108) :
  (total_students : ℝ) * avg_height_all - (num_boys : ℝ) * avg_height_boys = (num_girls : ℝ) * 97 :=
sorry

end NUMINAMATH_CALUDE_girls_average_height_l629_62937


namespace NUMINAMATH_CALUDE_sequence_squared_l629_62948

theorem sequence_squared (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n > 0 → 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2) ∧
  (∀ n : ℕ, n > 1 → a n > a (n - 1)) →
  ∀ n : ℕ, n > 0 → a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_squared_l629_62948


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l629_62971

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l629_62971


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l629_62980

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 462 →
  B = 150 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l629_62980


namespace NUMINAMATH_CALUDE_number_solution_l629_62981

theorem number_solution (z s : ℝ) (n : ℝ) : 
  z ≠ 0 → 
  z = Real.sqrt (n * z * s - 9 * s^2) → 
  z = 3 → 
  n = 3 + 3 * s := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l629_62981


namespace NUMINAMATH_CALUDE_two_digit_square_sum_equals_concatenation_l629_62910

theorem two_digit_square_sum_equals_concatenation : 
  {(x, y) : ℕ × ℕ | 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ (x + y)^2 = 100 * x + y} = 
  {(20, 25), (30, 25)} := by
sorry

end NUMINAMATH_CALUDE_two_digit_square_sum_equals_concatenation_l629_62910


namespace NUMINAMATH_CALUDE_cube_difference_equality_l629_62940

theorem cube_difference_equality (x y : ℝ) (h : x - y = 1) :
  x^3 - 3*x*y - y^3 = 1 := by sorry

end NUMINAMATH_CALUDE_cube_difference_equality_l629_62940


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l629_62926

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 3| > 0} = {x : ℝ | x < 2/3 ∨ x > 2} :=
sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | ∀ x, f a x < |x - 3|} = {a : ℝ | 1 < a ∧ a < 5} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l629_62926


namespace NUMINAMATH_CALUDE_raisin_count_l629_62968

theorem raisin_count (total : ℕ) (box1 : ℕ) (box345 : ℕ) (h1 : total = 437) 
  (h2 : box1 = 72) (h3 : box345 = 97) : 
  total - box1 - 3 * box345 = 74 := by
  sorry

end NUMINAMATH_CALUDE_raisin_count_l629_62968


namespace NUMINAMATH_CALUDE_cube_of_hundred_l629_62931

theorem cube_of_hundred : 99^3 + 3*(99^2) + 3*99 + 1 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_hundred_l629_62931


namespace NUMINAMATH_CALUDE_intersection_implies_range_l629_62919

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

theorem intersection_implies_range (a : ℝ) : A ∩ B a = A → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_range_l629_62919


namespace NUMINAMATH_CALUDE_biased_coin_probability_l629_62922

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  (Nat.choose 6 3 : ℝ) * p^3 * (1 - p)^3 = (1 : ℝ) / 20 →
  p = 0.125 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l629_62922


namespace NUMINAMATH_CALUDE_function_decreasing_range_l629_62939

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 * a - 1) * x + 4 * a else a / x

theorem function_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) < 0) ↔
  a ∈ Set.Icc (1 / 5 : ℝ) (1 / 2 : ℝ) ∧ a ≠ 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_range_l629_62939


namespace NUMINAMATH_CALUDE_least_prime_factor_of_p6_minus_p5_l629_62990

theorem least_prime_factor_of_p6_minus_p5 (p : ℕ) (hp : Nat.Prime p) :
  Nat.minFac (p^6 - p^5) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_p6_minus_p5_l629_62990


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l629_62928

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 3/2) : x^2 + 1/x^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l629_62928


namespace NUMINAMATH_CALUDE_family_suitcases_l629_62953

theorem family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (total_suitcases : ℕ) : 
  num_siblings = 4 →
  suitcases_per_sibling = 2 →
  total_suitcases = 14 →
  ∃ (parents_suitcases : ℕ), 
    parents_suitcases = total_suitcases - (num_siblings * suitcases_per_sibling) ∧
    parents_suitcases % 2 = 0 ∧
    parents_suitcases / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_family_suitcases_l629_62953


namespace NUMINAMATH_CALUDE_hall_of_mirrors_wall_length_l629_62927

/-- Given three walls in a hall of mirrors, where two walls have the same unknown length and are 12 feet high,
    and the third wall is 20 feet by 12 feet, if the total glass needed is 960 square feet,
    then the length of each of the two unknown walls is 30 feet. -/
theorem hall_of_mirrors_wall_length :
  ∀ (L : ℝ),
  (2 * L * 12 + 20 * 12 = 960) →
  L = 30 := by
sorry

end NUMINAMATH_CALUDE_hall_of_mirrors_wall_length_l629_62927


namespace NUMINAMATH_CALUDE_nine_students_left_l629_62959

/-- The number of students left after some were checked out early -/
def students_left (initial : ℕ) (checked_out : ℕ) : ℕ :=
  initial - checked_out

/-- Theorem: Given 16 initial students and 7 checked out early, 9 students are left -/
theorem nine_students_left :
  students_left 16 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_students_left_l629_62959


namespace NUMINAMATH_CALUDE_sum_of_decimals_l629_62913

theorem sum_of_decimals : (4.3 : ℝ) + 3.88 = 8.18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l629_62913


namespace NUMINAMATH_CALUDE_science_quiz_bowl_participation_l629_62949

/-- The Science Quiz Bowl Participation Problem -/
theorem science_quiz_bowl_participation (participants_2018 : ℕ) : 
  participants_2018 = 150 → 
  ∃ (participants_2019 participants_2020 : ℕ),
    participants_2019 = 2 * participants_2018 + 20 ∧
    participants_2020 = participants_2019 / 2 - 40 ∧
    participants_2019 - participants_2020 = 200 := by
  sorry

end NUMINAMATH_CALUDE_science_quiz_bowl_participation_l629_62949


namespace NUMINAMATH_CALUDE_f_odd_and_periodic_l629_62988

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop := ∀ x, f (10 + x) = f (10 - x)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (20 + x) = -f (20 - x)

-- State the theorem
theorem f_odd_and_periodic (h1 : condition1 f) (h2 : condition2 f) :
  (∀ x, f (x + 40) = f x) ∧ (∀ x, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_periodic_l629_62988


namespace NUMINAMATH_CALUDE_similar_right_triangles_l629_62972

theorem similar_right_triangles (y : ℝ) : 
  (15 : ℝ) / 12 = y / 10 → y = 12.5 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_l629_62972


namespace NUMINAMATH_CALUDE_quadratic_factorization_l629_62974

theorem quadratic_factorization (C D : ℤ) :
  (∀ y : ℚ, 15 * y^2 - 56 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l629_62974


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2015_l629_62961

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem smallest_positive_angle_2015 :
  ∃! θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ (-2015) ∧
  ∀ φ, 0 ≤ φ ∧ φ < 360 ∧ same_terminal_side φ (-2015) → θ ≤ φ :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2015_l629_62961


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l629_62903

/-- The set of numbers on the eight-sided die -/
def dieNumbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The product of seven numbers from the die -/
def Q (s : Finset ℕ) : ℕ :=
  if s.card = 7 ∧ s ⊆ dieNumbers then s.prod id else 0

/-- The theorem stating that 48 is the largest number certain to divide Q -/
theorem largest_certain_divisor :
  ∀ n : ℕ, (∀ s : Finset ℕ, s.card = 7 ∧ s ⊆ dieNumbers → n ∣ Q s) → n ≤ 48 :=
sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l629_62903


namespace NUMINAMATH_CALUDE_sum_number_and_square_l629_62998

/-- If a number is 16, then the sum of this number and its square is 272. -/
theorem sum_number_and_square (x : ℕ) : x = 16 → x + x^2 = 272 := by
  sorry

end NUMINAMATH_CALUDE_sum_number_and_square_l629_62998


namespace NUMINAMATH_CALUDE_product_sqrt_inequality_l629_62986

theorem product_sqrt_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) (hsum : a + b + c = 9) :
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_product_sqrt_inequality_l629_62986


namespace NUMINAMATH_CALUDE_solve_equation_l629_62957

theorem solve_equation (n : ℚ) : 
  (1 : ℚ) / (n + 1) + 2 / (n + 1) + (n + 1) / (n + 1) = 2 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l629_62957


namespace NUMINAMATH_CALUDE_building_block_length_l629_62925

theorem building_block_length 
  (box_height box_width box_length : ℝ)
  (block_height block_width : ℝ)
  (num_blocks : ℕ) :
  box_height = 8 →
  box_width = 10 →
  box_length = 12 →
  block_height = 3 →
  block_width = 2 →
  num_blocks = 40 →
  ∃ (block_length : ℝ),
    box_height * box_width * box_length = 
    num_blocks * block_height * block_width * block_length ∧
    block_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_building_block_length_l629_62925


namespace NUMINAMATH_CALUDE_inequality_solution_set_l629_62975

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l629_62975


namespace NUMINAMATH_CALUDE_derivative_condition_implies_constants_l629_62923

open Real

theorem derivative_condition_implies_constants (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x => (a*x + b) * sin x + (c*x + d) * cos x
  (∀ x, deriv f x = x * cos x) →
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_condition_implies_constants_l629_62923


namespace NUMINAMATH_CALUDE_tangent_line_equation_l629_62962

/-- The equation of the tangent line to y = xe^x + 1 at (1, e+1) -/
theorem tangent_line_equation (x y : ℝ) : 
  (∀ t, y = t * Real.exp t + 1) →  -- Curve equation
  2 * Real.exp 1 * x - y - Real.exp 1 + 1 = 0 -- Tangent line equation
  ↔ 
  (x = 1 ∧ y = Real.exp 1 + 1) -- Point of tangency
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l629_62962


namespace NUMINAMATH_CALUDE_line_symmetry_l629_62911

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The property of two lines being symmetrical about the x-axis -/
def symmetrical_about_x_axis (l1 l2 : Line) : Prop :=
  l1.slope = -l2.slope ∧ l1.intercept = -l2.intercept

/-- The given line y = 2x + 1 -/
def given_line : Line :=
  { slope := 2, intercept := 1 }

/-- The proposed symmetrical line y = -2x - 1 -/
def symmetrical_line : Line :=
  { slope := -2, intercept := -1 }

theorem line_symmetry :
  symmetrical_about_x_axis given_line symmetrical_line :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l629_62911
