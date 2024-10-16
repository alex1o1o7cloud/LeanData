import Mathlib

namespace NUMINAMATH_CALUDE_four_points_plane_count_l1930_193070

/-- A set of four points in three-dimensional space -/
structure FourPoints where
  points : Fin 4 → ℝ × ℝ × ℝ

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : ℝ × ℝ × ℝ) : Prop := sorry

/-- Predicate to check if no three points in a set of four points are collinear -/
def no_three_collinear (fp : FourPoints) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(are_collinear (fp.points i) (fp.points j) (fp.points k))

/-- The number of planes determined by four points -/
def num_planes (fp : FourPoints) : ℕ := sorry

/-- Theorem: Given four points in space where no three points are collinear, 
    the number of planes these points can determine is either 1 or 4 -/
theorem four_points_plane_count (fp : FourPoints) 
  (h : no_three_collinear fp) : 
  num_planes fp = 1 ∨ num_planes fp = 4 := by sorry

end NUMINAMATH_CALUDE_four_points_plane_count_l1930_193070


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1930_193042

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4*y) * (9 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1930_193042


namespace NUMINAMATH_CALUDE_product_of_real_parts_l1930_193041

theorem product_of_real_parts (x : ℂ) : 
  x^2 - 6*x = -8 + 2*I → 
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    x₁^2 - 6*x₁ = -8 + 2*I ∧ 
    x₂^2 - 6*x₂ = -8 + 2*I ∧
    (x₁.re * x₂.re = 9 - Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l1930_193041


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1930_193015

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 172) : 
  a * b = 85 / 6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1930_193015


namespace NUMINAMATH_CALUDE_win_sector_area_l1930_193095

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1930_193095


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l1930_193079

theorem cubic_expression_evaluation : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l1930_193079


namespace NUMINAMATH_CALUDE_special_pair_characterization_l1930_193037

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Property of natural numbers a and b -/
def special_pair (a b : ℕ) : Prop :=
  (a^2 + 1) % b = 0 ∧ (b^2 + 1) % a = 0

/-- Main theorem -/
theorem special_pair_characterization (a b : ℕ) :
  special_pair a b → (a = 1 ∧ b = 1) ∨ (∃ n : ℕ, n ≥ 1 ∧ a = fib (2*n - 1) ∧ b = fib (2*n + 1)) :=
sorry

end NUMINAMATH_CALUDE_special_pair_characterization_l1930_193037


namespace NUMINAMATH_CALUDE_sphere_surface_area_increase_l1930_193093

theorem sphere_surface_area_increase (r : ℝ) (h : r > 0) :
  let original_area := 4 * Real.pi * r^2
  let new_radius := 1.5 * r
  let new_area := 4 * Real.pi * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_increase_l1930_193093


namespace NUMINAMATH_CALUDE_fish_tank_balls_count_total_balls_in_tank_l1930_193086

theorem fish_tank_balls_count : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_goldfish, num_platyfish, red_balls_per_goldfish, white_balls_per_platyfish =>
    num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish

theorem total_balls_in_tank : fish_tank_balls_count 3 10 10 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_balls_count_total_balls_in_tank_l1930_193086


namespace NUMINAMATH_CALUDE_lidia_remaining_money_l1930_193023

/-- Calculates the remaining money after buying apps -/
def remaining_money (app_cost : ℕ) (num_apps : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - app_cost * num_apps

/-- Proves that Lidia will have $6 left after buying apps -/
theorem lidia_remaining_money :
  let app_cost : ℕ := 4
  let num_apps : ℕ := 15
  let initial_money : ℕ := 66
  remaining_money app_cost num_apps initial_money = 6 := by
sorry

end NUMINAMATH_CALUDE_lidia_remaining_money_l1930_193023


namespace NUMINAMATH_CALUDE_distribute_four_among_five_l1930_193051

/-- The number of ways to distribute n identical objects among k people,
    where each person receives at most one object and all objects must be distributed. -/
def distribute (n k : ℕ) : ℕ :=
  if n = k - 1 then k else 0

theorem distribute_four_among_five :
  distribute 4 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_among_five_l1930_193051


namespace NUMINAMATH_CALUDE_expand_expression_l1930_193089

theorem expand_expression (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4*x + 4) = x^4 + 4*x^3 - 16*x - 16 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1930_193089


namespace NUMINAMATH_CALUDE_map_scale_l1930_193017

/-- Given a map where 10 cm represents 50 km, prove that 15 cm represents 75 km -/
theorem map_scale (scale : ℝ → ℝ) : 
  (scale 10 = 50) → (scale 15 = 75) := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l1930_193017


namespace NUMINAMATH_CALUDE_polygon_angle_sums_l1930_193054

/-- For an n-sided polygon, the sum of exterior angles is 360° and the sum of interior angles is (n-2) × 180° -/
theorem polygon_angle_sums (n : ℕ) (h : n ≥ 3) :
  ∃ (exterior_sum interior_sum : ℝ),
    exterior_sum = 360 ∧
    interior_sum = (n - 2) * 180 :=
by sorry

end NUMINAMATH_CALUDE_polygon_angle_sums_l1930_193054


namespace NUMINAMATH_CALUDE_volume_ratio_l1930_193040

/-- The domain S bounded by two curves -/
structure Domain (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The volume formed by revolving the domain around the x-axis -/
noncomputable def volume_x (d : Domain a b) : ℝ := sorry

/-- The volume formed by revolving the domain around the y-axis -/
noncomputable def volume_y (d : Domain a b) : ℝ := sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (a b : ℝ) (d : Domain a b) :
  (volume_x d) / (volume_y d) = 14 / 5 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_l1930_193040


namespace NUMINAMATH_CALUDE_prec_2011_130_l1930_193085

-- Define the new operation ⪯
def prec (a b : ℕ) : ℕ := b * 10 + a * 2

-- Theorem to prove
theorem prec_2011_130 : prec 2011 130 = 5322 := by
  sorry

end NUMINAMATH_CALUDE_prec_2011_130_l1930_193085


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1930_193075

theorem negation_of_proposition (p : Prop) :
  (p ↔ ∃ x, x < -1 ∧ x^2 - x + 1 < 0) →
  (¬p ↔ ∀ x, x < -1 → x^2 - x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1930_193075


namespace NUMINAMATH_CALUDE_rational_fraction_value_l1930_193005

theorem rational_fraction_value (x y : ℝ) :
  3 < (x - y) / (x + y) →
  (x - y) / (x + y) < 4 →
  ∃ (a b : ℤ), x / y = a / b →
  x + y = 10 →
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_value_l1930_193005


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_seven_l1930_193097

/-- The largest integer whose square has exactly 3 digits when written in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def to_base_seven (n : ℕ) : ℕ :=
  if n < 7 then n
  else 10 * to_base_seven (n / 7) + n % 7

theorem largest_three_digit_square_base_seven :
  (M^2 ≥ 7^2) ∧ 
  (M^2 < 7^3) ∧ 
  (∀ n : ℕ, n > M → n^2 ≥ 7^3) ∧
  (to_base_seven M = 66) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_seven_l1930_193097


namespace NUMINAMATH_CALUDE_square_difference_equality_l1930_193012

theorem square_difference_equality : (1 + 2)^2 - (1^2 + 2^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1930_193012


namespace NUMINAMATH_CALUDE_polygon_120_sides_diagonals_l1930_193090

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 120 sides has 7020 diagonals -/
theorem polygon_120_sides_diagonals :
  num_diagonals 120 = 7020 := by
  sorry

end NUMINAMATH_CALUDE_polygon_120_sides_diagonals_l1930_193090


namespace NUMINAMATH_CALUDE_unique_solution_system_l1930_193049

theorem unique_solution_system (x y : ℝ) : 
  (2 * x + y = 3 ∧ x - y = 3) ↔ (x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1930_193049


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1930_193006

theorem geometric_sequence_common_ratio_sum (k a₂ a₃ b₂ b₃ : ℝ) (p r : ℝ) 
  (hk : k ≠ 0)
  (hp : p ≠ 1)
  (hr : r ≠ 1)
  (hp_neq_r : p ≠ r)
  (ha₂ : a₂ = k * p)
  (ha₃ : a₃ = k * p^2)
  (hb₂ : b₂ = k * r)
  (hb₃ : b₃ = k * r^2)
  (h_eq : a₃^2 - b₃^2 = 3 * (a₂^2 - b₂^2)) :
  p^2 + r^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1930_193006


namespace NUMINAMATH_CALUDE_quadratic_properties_l1930_193063

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_neg1 : a * (-1)^2 + b * (-1) + c = 0
  point_0 : c = -3
  point_1 : a * 1^2 + b * 1 + c = -4
  point_2 : a * 2^2 + b * 2 + c = -3
  point_3 : a * 3^2 + b * 3 + c = 0

/-- The theorem stating properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (∃ x y, f.a * x^2 + f.b * x + f.c = y ∧ ∀ t, f.a * t^2 + f.b * t + f.c ≥ y) ∧
  (f.a * x^2 + f.b * x + f.c = -4 ↔ x = 1) ∧
  (f.a * 5^2 + f.b * 5 + f.c = 12) ∧
  (∀ x > 1, ∀ y > x, f.a * y^2 + f.b * y + f.c > f.a * x^2 + f.b * x + f.c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1930_193063


namespace NUMINAMATH_CALUDE_sin_80_degrees_l1930_193014

theorem sin_80_degrees (k : ℝ) (h : Real.tan (100 * π / 180) = k) : 
  Real.sin (80 * π / 180) = - k / Real.sqrt (1 + k^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_80_degrees_l1930_193014


namespace NUMINAMATH_CALUDE_recurring_decimal_equals_two_fifteenths_l1930_193047

-- Define the recurring decimal 0.333...
def recurring_third : ℚ := 1/3

-- Define the recurring decimal 0.1333...
def recurring_decimal : ℚ := 0.1333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

-- State the theorem
theorem recurring_decimal_equals_two_fifteenths 
  (h : recurring_third = 1/3) : 
  recurring_decimal = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_equals_two_fifteenths_l1930_193047


namespace NUMINAMATH_CALUDE_smallest_z_value_l1930_193050

theorem smallest_z_value (x y z : ℝ) : 
  (7 < x) → (x < 9) → (9 < y) → (y < z) → 
  (∃ (n : ℕ), y - x = n ∧ ∀ (m : ℕ), y - x ≤ m → m ≤ n) →
  (∀ (w : ℝ), (7 < w) → (w < 9) → (9 < y) → (y < z) → 
    ∃ (k : ℕ), y - w ≤ k ∧ k ≤ 7) →
  z ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l1930_193050


namespace NUMINAMATH_CALUDE_unique_valid_n_l1930_193011

def is_valid_number (n : ℕ) (x : ℕ) : Prop :=
  (x.digits 10).length = n ∧
  (x.digits 10).count 7 = 1 ∧
  (x.digits 10).count 1 = n - 1

def all_numbers_prime (n : ℕ) : Prop :=
  ∀ x : ℕ, is_valid_number n x → Nat.Prime x

theorem unique_valid_n : 
  ∀ n : ℕ, (n > 0 ∧ all_numbers_prime n) ↔ (n = 1 ∨ n = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_valid_n_l1930_193011


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1930_193018

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 2) : 
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^2 + y^2 + z^2 = 2 → 2*a*b + 3*b*c ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1930_193018


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l1930_193064

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral :=
  (a : ℝ) -- Length of side a
  (b : ℝ) -- Length of side b
  (c : ℝ) -- Length of diagonal c
  (d : ℝ) -- Length of diagonal d
  (ha : a > 0) -- Side lengths are positive
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)

/-- In any cyclic quadrilateral, the sum of the squares of the sides 
    is equal to the sum of the squares of the diagonals. -/
theorem cyclic_quadrilateral_theorem (q : CyclicQuadrilateral) :
  q.c^2 + q.d^2 = 2 * (q.a^2 + q.b^2) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l1930_193064


namespace NUMINAMATH_CALUDE_complementary_angle_measure_l1930_193035

/-- Given two complementary angles A and B, where the measure of A is 3 times the measure of B,
    prove that the measure of angle A is 67.5° -/
theorem complementary_angle_measure (A B : ℝ) : 
  A + B = 90 →  -- angles A and B are complementary
  A = 3 * B →   -- measure of A is 3 times measure of B
  A = 67.5 :=   -- measure of A is 67.5°
by sorry

end NUMINAMATH_CALUDE_complementary_angle_measure_l1930_193035


namespace NUMINAMATH_CALUDE_certain_number_exists_l1930_193072

theorem certain_number_exists : ∃ x : ℝ, 5 * 1.25 * x^(1/4) * 60^(3/4) = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1930_193072


namespace NUMINAMATH_CALUDE_vasya_number_exists_l1930_193058

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = 8) ∧
  (digits.count 1 = 2) ∧ (digits.count 2 = 2) ∧ (digits.count 3 = 2) ∧ (digits.count 4 = 2) ∧
  (∃ i, digits.get? i = some 1 ∧ digits.get? (i + 2) = some 1) ∧
  (∃ i, digits.get? i = some 2 ∧ digits.get? (i + 3) = some 2) ∧
  (∃ i, digits.get? i = some 3 ∧ digits.get? (i + 4) = some 3) ∧
  (∃ i, digits.get? i = some 4 ∧ digits.get? (i + 5) = some 4)

theorem vasya_number_exists : ∃ n : ℕ, is_valid_number n := by
  sorry

end NUMINAMATH_CALUDE_vasya_number_exists_l1930_193058


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1930_193077

theorem quadratic_inequality (x : ℝ) : x^2 - 36*x + 325 ≤ 9 ↔ 16 ≤ x ∧ x ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1930_193077


namespace NUMINAMATH_CALUDE_smallest_AAB_l1930_193053

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def AB (a b : ℕ) : ℕ := 10 * a + b

def AAB (a b : ℕ) : ℕ := 100 * a + 10 * a + b

theorem smallest_AAB :
  ∃ (a b : ℕ),
    is_digit a ∧
    is_digit b ∧
    two_digit (AB a b) ∧
    three_digit (AAB a b) ∧
    AB a b = (AAB a b) / 7 ∧
    AAB a b = 996 ∧
    (∀ (x y : ℕ),
      is_digit x ∧
      is_digit y ∧
      two_digit (AB x y) ∧
      three_digit (AAB x y) ∧
      AB x y = (AAB x y) / 7 →
      AAB x y ≥ 996) :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_l1930_193053


namespace NUMINAMATH_CALUDE_four_fold_f_of_two_plus_i_l1930_193091

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

-- State the theorem
theorem four_fold_f_of_two_plus_i :
  f (f (f (f (2 + Complex.I)))) = 164833 + 354192 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_four_fold_f_of_two_plus_i_l1930_193091


namespace NUMINAMATH_CALUDE_solution_comparison_l1930_193044

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-d / c > -f / e) ↔ (f / e > d / c) :=
sorry

end NUMINAMATH_CALUDE_solution_comparison_l1930_193044


namespace NUMINAMATH_CALUDE_tower_difference_l1930_193024

/-- The number of blocks Randy used to build different structures -/
structure BlockCounts where
  total : ℕ
  house : ℕ
  tower : ℕ
  bridge : ℕ

/-- The theorem stating the difference in blocks used for the tower versus the house and bridge combined -/
theorem tower_difference (b : BlockCounts) (h1 : b.total = 250) (h2 : b.house = 65) (h3 : b.tower = 120) (h4 : b.bridge = 45) :
  b.tower - (b.house + b.bridge) = 10 := by
  sorry


end NUMINAMATH_CALUDE_tower_difference_l1930_193024


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_90_l1930_193025

theorem greatest_common_multiple_10_15_under_90 : 
  ∃ (n : ℕ), n = 60 ∧ 
  (∀ m : ℕ, m < 90 ∧ 10 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  10 ∣ n ∧ 15 ∣ n ∧ n < 90 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_90_l1930_193025


namespace NUMINAMATH_CALUDE_larger_circle_radius_l1930_193009

/-- Two concentric circles with radii in ratio 2:5 -/
structure ConcentricCircles where
  r : ℝ
  small_radius : ℝ := 2 * r
  large_radius : ℝ := 5 * r

/-- Chord of larger circle tangent to smaller circle -/
def tangent_chord (c : ConcentricCircles) (ab : ℝ) : Prop :=
  ab^2 = (c.large_radius^2 - c.small_radius^2)

theorem larger_circle_radius (c : ConcentricCircles) 
  (h : tangent_chord c 15) : c.large_radius = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l1930_193009


namespace NUMINAMATH_CALUDE_square_room_tiles_and_triangles_l1930_193081

theorem square_room_tiles_and_triangles (n : ℕ) : 
  n > 0 →  -- Ensure the room has a positive side length
  (2 * n - 1 = 57) →  -- Total tiles on diagonals
  (n^2 = 841 ∧ 4 = 4) :=  -- Total tiles and number of triangles
by sorry

end NUMINAMATH_CALUDE_square_room_tiles_and_triangles_l1930_193081


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_x_minus_one_times_exp_x_over_two_l1930_193030

theorem integral_x_squared_plus_x_minus_one_times_exp_x_over_two :
  ∫ x in (0 : ℝ)..2, (x^2 + x - 1) * Real.exp (x / 2) = 2 * (3 * Real.exp 1 - 5) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_x_minus_one_times_exp_x_over_two_l1930_193030


namespace NUMINAMATH_CALUDE_hamburgers_left_over_example_l1930_193096

/-- Given a restaurant that made some hamburgers and served some of them,
    calculate the number of hamburgers left over. -/
def hamburgers_left_over (made served : ℕ) : ℕ :=
  made - served

/-- Theorem stating that if 9 hamburgers were made and 3 were served,
    then 6 hamburgers were left over. -/
theorem hamburgers_left_over_example :
  hamburgers_left_over 9 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_example_l1930_193096


namespace NUMINAMATH_CALUDE_quadratic_function_values_l1930_193022

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(1) = 3 and f(2) = 5, then f(3) = 7 -/
theorem quadratic_function_values (a b c : ℝ) :
  f a b c 1 = 3 → f a b c 2 = 5 → f a b c 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_values_l1930_193022


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_root_l1930_193052

theorem triangle_inequality_cube_root (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (((a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b))^(1/3) : ℝ) > (a^2 + b^2 + c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_root_l1930_193052


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_40_l1930_193021

theorem ones_digit_of_8_to_40 (cycle : List Nat) (h_cycle : cycle = [8, 4, 2, 6]) :
  (8^40 : ℕ) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_40_l1930_193021


namespace NUMINAMATH_CALUDE_symmetrical_cubic_function_l1930_193008

-- Define the function f(x) with parameters a and b
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (a - 2) * x + b

-- Define the property of symmetry about the origin
def symmetrical_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem symmetrical_cubic_function
  (a b : ℝ)
  (h_symmetry : symmetrical_about_origin (f a b)) :
  (a = 1 ∧ b = 0) ∧
  (∀ x, f a b x = x^3 - 48*x) ∧
  (∀ x, -4 ≤ x ∧ x ≤ 4 → (∀ y, x < y → f a b x > f a b y)) ∧
  (∀ x, (x < -4 ∨ x > 4) → (∀ y, x < y → f a b x < f a b y)) ∧
  (f a b (-4) = 128) ∧
  (f a b 4 = -128) ∧
  (∀ x, f a b x ≤ 128) ∧
  (∀ x, f a b x ≥ -128) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_cubic_function_l1930_193008


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1930_193046

/-- The number of sides of a regular polygon whose sum of interior angles is 1080° more than
    the sum of exterior angles of a pentagon. -/
def num_sides_regular_polygon : ℕ := 10

/-- The sum of exterior angles of any polygon is always 360°. -/
axiom sum_exterior_angles : ℕ → ℝ
axiom sum_exterior_angles_def : ∀ n : ℕ, sum_exterior_angles n = 360

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180°. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem stating that the number of sides of the regular polygon is 10. -/
theorem regular_polygon_sides :
  sum_interior_angles num_sides_regular_polygon =
  sum_exterior_angles 5 + 1080 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1930_193046


namespace NUMINAMATH_CALUDE_lizard_wrinkle_eye_ratio_l1930_193076

theorem lizard_wrinkle_eye_ratio :
  ∀ (W : ℕ) (S : ℕ),
    S = 7 * W →
    3 = S + W - 69 →
    (W : ℚ) / 3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_lizard_wrinkle_eye_ratio_l1930_193076


namespace NUMINAMATH_CALUDE_min_tangent_length_l1930_193029

/-- The minimum length of a tangent from a point on y = x + 2 to (x-3)² + (y+1)² = 2 is 4 -/
theorem min_tangent_length : 
  let line := {p : ℝ × ℝ | p.2 = p.1 + 2}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 + 1)^2 = 2}
  ∃ (min_length : ℝ), 
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ circle → 
      ‖p - q‖ ≥ min_length) ∧
    (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧ ‖p - q‖ = min_length) ∧
    min_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_min_tangent_length_l1930_193029


namespace NUMINAMATH_CALUDE_soda_price_theorem_l1930_193087

/-- Calculates the price of a given number of soda cans with a discount applied to full cases. -/
def discounted_soda_price (regular_price : ℚ) (discount_percent : ℚ) (case_size : ℕ) (num_cans : ℕ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  let full_cases := num_cans / case_size
  let remaining_cans := num_cans % case_size
  full_cases * (case_size : ℚ) * discounted_price + (remaining_cans : ℚ) * discounted_price

/-- The price of 75 cans of soda purchased in 24-can cases with a 10% discount is $10.125. -/
theorem soda_price_theorem :
  discounted_soda_price (15/100) (1/10) 24 75 = 10125/1000 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_theorem_l1930_193087


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1930_193032

theorem polynomial_division_quotient : 
  let dividend : Polynomial ℚ := 8 * X^4 - 4 * X^3 + 3 * X^2 - 5 * X - 10
  let divisor : Polynomial ℚ := X^2 + 3 * X + 2
  let quotient : Polynomial ℚ := 8 * X^2 - 28 * X + 89
  dividend = divisor * quotient + (dividend.mod divisor) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1930_193032


namespace NUMINAMATH_CALUDE_exists_valid_point_distribution_l1930_193043

/-- Represents a convex pentagon --/
structure ConvexPentagon where
  -- Add necessary fields

/-- Represents a point inside the pentagon --/
structure Point where
  -- Add necessary fields

/-- Represents a triangle formed by the vertices of the pentagon --/
structure Triangle where
  -- Add necessary fields

/-- Function to check if a point is inside a triangle --/
def pointInTriangle (p : Point) (t : Triangle) : Bool :=
  sorry

/-- Function to count points inside a triangle --/
def countPointsInTriangle (points : List Point) (t : Triangle) : Nat :=
  sorry

/-- Theorem stating the existence of a valid point distribution --/
theorem exists_valid_point_distribution (pentagon : ConvexPentagon) :
  ∃ (points : List Point),
    points.length = 18 ∧
    ∀ (t1 t2 : Triangle),
      countPointsInTriangle points t1 = countPointsInTriangle points t2 :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_point_distribution_l1930_193043


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l1930_193048

-- Define a rhombus
structure Rhombus :=
  (side_length : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)
  (side_length_positive : side_length > 0)
  (diagonals_positive : diagonal1 > 0 ∧ diagonal2 > 0)

-- State the theorem
theorem rhombus_diagonals_not_always_equal :
  ∃ (r : Rhombus), r.diagonal1 ≠ r.diagonal2 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l1930_193048


namespace NUMINAMATH_CALUDE_drink_mix_to_stakes_ratio_l1930_193084

/-- Represents the number of items Rebecca bought for her camping trip -/
def total_items : ℕ := 22

/-- Represents the number of tent stakes Rebecca bought -/
def tent_stakes : ℕ := 4

/-- Represents the number of bottles of water Rebecca bought -/
def water_bottles : ℕ := tent_stakes + 2

/-- Represents the number of packets of drink mix Rebecca bought -/
def drink_mix_packets : ℕ := total_items - tent_stakes - water_bottles

/-- Proves that the ratio of drink mix packets to tent stakes is 3:1 -/
theorem drink_mix_to_stakes_ratio :
  (drink_mix_packets : ℚ) / (tent_stakes : ℚ) = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_drink_mix_to_stakes_ratio_l1930_193084


namespace NUMINAMATH_CALUDE_five_greater_than_two_sqrt_five_l1930_193007

theorem five_greater_than_two_sqrt_five : 5 > 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_five_greater_than_two_sqrt_five_l1930_193007


namespace NUMINAMATH_CALUDE_bowling_team_size_l1930_193066

theorem bowling_team_size (n : ℕ) (original_avg : ℝ) (new_avg : ℝ) 
  (new_player1_weight : ℝ) (new_player2_weight : ℝ) 
  (h1 : original_avg = 112)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_avg = 106)
  (h5 : n * original_avg + new_player1_weight + new_player2_weight = (n + 2) * new_avg) :
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_size_l1930_193066


namespace NUMINAMATH_CALUDE_complex_division_simplification_l1930_193027

theorem complex_division_simplification :
  (1 - 2 * Complex.I) / (1 + Complex.I) = -1/2 - 3/2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l1930_193027


namespace NUMINAMATH_CALUDE_min_sin6_2cos6_l1930_193071

theorem min_sin6_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_sin6_2cos6_l1930_193071


namespace NUMINAMATH_CALUDE_period_and_trigonometric_function_l1930_193065

theorem period_and_trigonometric_function (ω : ℝ) (α β : ℝ) : 
  ω > 0 →
  (∀ x, 2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x) = 
    Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)) →
  (∀ x, 2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x) = 
    2 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (2 * ω * x)) →
  Real.sqrt 2 * Real.sin (α - Real.pi / 4 + Real.pi / 4) = Real.sqrt 2 / 3 →
  Real.sqrt 2 * Real.sin (β - Real.pi / 4 + Real.pi / 4) = 2 * Real.sqrt 2 / 3 →
  α > -Real.pi / 2 →
  α < Real.pi / 2 →
  β > -Real.pi / 2 →
  β < Real.pi / 2 →
  Real.cos (α + β) = (2 * Real.sqrt 10 - 2) / 9 := by
sorry


end NUMINAMATH_CALUDE_period_and_trigonometric_function_l1930_193065


namespace NUMINAMATH_CALUDE_bills_tv_height_l1930_193036

-- Define the dimensions of the TVs
def bill_width : ℕ := 48
def bob_width : ℕ := 70
def bob_height : ℕ := 60

-- Define the weight per square inch
def weight_per_sq_inch : ℕ := 4

-- Define the weight difference in ounces
def weight_diff_oz : ℕ := 150 * 16

-- Theorem statement
theorem bills_tv_height :
  ∃ (h : ℕ),
    h * bill_width * weight_per_sq_inch =
    bob_width * bob_height * weight_per_sq_inch - weight_diff_oz ∧
    h = 75 := by
  sorry

end NUMINAMATH_CALUDE_bills_tv_height_l1930_193036


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1930_193003

theorem cos_alpha_value (α : Real) :
  (∃ P : Real × Real, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1930_193003


namespace NUMINAMATH_CALUDE_base5_product_l1930_193019

/-- Converts a base-5 number represented as a list of digits to a natural number. -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a natural number to its base-5 representation as a list of digits. -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The statement of the problem. -/
theorem base5_product : 
  let a := fromBase5 [1, 3, 1]
  let b := fromBase5 [1, 3]
  toBase5 (a * b) = [2, 3, 3, 3] := by sorry

end NUMINAMATH_CALUDE_base5_product_l1930_193019


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1930_193056

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = -19 / 16) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1930_193056


namespace NUMINAMATH_CALUDE_sticker_count_l1930_193004

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := 30

/-- The number of stickers Steven has -/
def steven_stickers : ℕ := 3 * ryan_stickers

/-- The number of stickers Terry has -/
def terry_stickers : ℕ := steven_stickers + 20

/-- The total number of stickers Ryan, Steven, and Terry have altogether -/
def total_stickers : ℕ := ryan_stickers + steven_stickers + terry_stickers

theorem sticker_count : total_stickers = 230 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l1930_193004


namespace NUMINAMATH_CALUDE_range_of_a_in_first_quadrant_l1930_193016

-- Define a complex number z with real part a and imaginary part (a-1)
def z (a : ℝ) : ℂ := Complex.mk a (a - 1)

-- Define what it means for a complex number to be in the first quadrant
def in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

-- State the theorem
theorem range_of_a_in_first_quadrant :
  ∀ a : ℝ, in_first_quadrant (z a) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_first_quadrant_l1930_193016


namespace NUMINAMATH_CALUDE_a_range_l1930_193055

def A (a : ℝ) : Set ℝ := {x | x^2 - 2*x + a > 0}

theorem a_range (a : ℝ) : (1 ∉ A a) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1930_193055


namespace NUMINAMATH_CALUDE_circle_equation_l1930_193069

/-- The general equation of a circle with specific properties -/
theorem circle_equation (x y : ℝ) : 
  ∃ (h k : ℝ), 
    (k = -4 * h) ∧ 
    ((3 - h)^2 + (-2 - k)^2 = (3 + (-2) - 1)^2) ∧
    (∀ (a b : ℝ), (a + b - 1 = 0) → ((a - h)^2 + (b - k)^2 ≥ (3 + (-2) - 1)^2)) →
    x^2 + y^2 - 2*x + 8*y + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1930_193069


namespace NUMINAMATH_CALUDE_count_large_glasses_l1930_193034

/-- The number of jelly beans needed to fill a large drinking glass -/
def large_glass_jelly_beans : ℕ := 50

/-- The number of jelly beans needed to fill a small drinking glass -/
def small_glass_jelly_beans : ℕ := 25

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jelly beans used to fill all glasses -/
def total_jelly_beans : ℕ := 325

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

theorem count_large_glasses : 
  large_glass_jelly_beans * num_large_glasses + 
  small_glass_jelly_beans * num_small_glasses = total_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_count_large_glasses_l1930_193034


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_l1930_193094

theorem height_on_hypotenuse (a b h : ℝ) (hyp : ℝ) : 
  a = 2 → b = 3 → a^2 + b^2 = hyp^2 → (a * b) / 2 = (hyp * h) / 2 → h = (6 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_l1930_193094


namespace NUMINAMATH_CALUDE_bella_stamps_count_l1930_193000

/-- The number of snowflake stamps Bella bought -/
def snowflake_stamps : ℕ := 11

/-- The number of truck stamps Bella bought -/
def truck_stamps : ℕ := snowflake_stamps + 9

/-- The number of rose stamps Bella bought -/
def rose_stamps : ℕ := truck_stamps - 13

/-- The total number of stamps Bella bought -/
def total_stamps : ℕ := snowflake_stamps + truck_stamps + rose_stamps

theorem bella_stamps_count : total_stamps = 38 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamps_count_l1930_193000


namespace NUMINAMATH_CALUDE_digit_3000_is_1_l1930_193010

/-- Represents the decimal expansion of integers from 1 to 1001 concatenated -/
def x : ℝ :=
  sorry

/-- Returns the nth digit after the decimal point in the given real number -/
def nthDigit (n : ℕ) (r : ℝ) : ℕ :=
  sorry

/-- The 3000th digit after the decimal point in x is 1 -/
theorem digit_3000_is_1 : nthDigit 3000 x = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_3000_is_1_l1930_193010


namespace NUMINAMATH_CALUDE_salad_ratio_l1930_193080

/-- Given a salad with cucumbers and tomatoes, prove the ratio of tomatoes to cucumbers -/
theorem salad_ratio (total : ℕ) (cucumbers : ℕ) (h1 : total = 280) (h2 : cucumbers = 70) :
  (total - cucumbers) / cucumbers = 3 := by
  sorry

end NUMINAMATH_CALUDE_salad_ratio_l1930_193080


namespace NUMINAMATH_CALUDE_xyz_value_l1930_193088

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x*y + x*z + y*z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1930_193088


namespace NUMINAMATH_CALUDE_min_distance_equilateral_triangles_l1930_193082

/-- The distance between vertices of equilateral triangles on AC and CB is minimized when C is at the midpoint of AB -/
theorem min_distance_equilateral_triangles (l : ℝ) (h : l > 0) :
  let f : ℝ → ℝ := λ x => l^2 + 3 * (2*x - l)^2 / 4
  ∃ x : ℝ, x = l / 2 ∧ ∀ y : ℝ, 0 ≤ y ∧ y ≤ l → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_min_distance_equilateral_triangles_l1930_193082


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_l1930_193045

/-- A line passing through point P (1, 3) intersects the coordinate axes at points A and B. 
    P is the midpoint of AB. The equation of the line is 3x + y - 6 = 0. -/
theorem line_equation_through_midpoint (A B P : ℝ × ℝ) : 
  P = (1, 3) →
  (∃ a b : ℝ, A = (a, 0) ∧ B = (0, b)) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y : ℝ, (3 * x + y - 6 = 0) ↔ (∃ t : ℝ, (x, y) = (1 - t, 3 + t * (B.2 - 3))) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_l1930_193045


namespace NUMINAMATH_CALUDE_rectangular_cube_height_l1930_193074

-- Define the dimensions of the rectangular cube
def length : ℝ := 3
def width : ℝ := 2

-- Define the side length of the reference cube
def cubeSide : ℝ := 2

-- Define the surface area of the rectangular cube
def surfaceArea (h : ℝ) : ℝ := 2 * length * width + 2 * length * h + 2 * width * h

-- Define the surface area of the reference cube
def cubeSurfaceArea : ℝ := 6 * cubeSide^2

-- Theorem statement
theorem rectangular_cube_height : 
  ∃ h : ℝ, surfaceArea h = cubeSurfaceArea ∧ h = 1.2 := by sorry

end NUMINAMATH_CALUDE_rectangular_cube_height_l1930_193074


namespace NUMINAMATH_CALUDE_product_digit_sum_theorem_l1930_193020

def is_single_digit (n : ℕ) : Prop := 1 < n ∧ n < 10

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem product_digit_sum_theorem (x y : ℕ) :
  is_single_digit x ∧ is_single_digit y ∧ x ≠ 9 ∧ y ≠ 9 ∧ digit_sum (x * y) = x →
  (x = 3 ∧ y = 4) ∨ (x = 3 ∧ y = 7) ∨ (x = 6 ∧ y = 4) ∨ (x = 6 ∧ y = 7) :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_theorem_l1930_193020


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1930_193067

-- Define repeating decimals
def repeating_decimal_3 : ℚ := 1 / 3
def repeating_decimal_27 : ℚ := 3 / 11

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_3 + repeating_decimal_27 = 20 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1930_193067


namespace NUMINAMATH_CALUDE_levi_goal_difference_l1930_193092

/-- The number of baskets Levi wants to beat his brother by -/
def basketDifference (leviInitial : ℕ) (brotherInitial : ℕ) (brotherIncrease : ℕ) (leviIncrease : ℕ) : ℕ :=
  (leviInitial + leviIncrease) - (brotherInitial + brotherIncrease)

/-- Theorem stating that Levi wants to beat his brother by 5 baskets -/
theorem levi_goal_difference : basketDifference 8 12 3 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_levi_goal_difference_l1930_193092


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l1930_193057

theorem arithmetic_mean_fractions : 
  let a := 7 / 11
  let b := 9 / 11
  let c := 8 / 11
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l1930_193057


namespace NUMINAMATH_CALUDE_library_books_count_l1930_193001

/-- Given the conditions of the library bookshelves, calculate the total number of books -/
theorem library_books_count (num_shelves : ℕ) (floors_per_shelf : ℕ) (books_after_removal : ℕ) : 
  num_shelves = 28 → 
  floors_per_shelf = 6 → 
  books_after_removal = 20 → 
  (num_shelves * floors_per_shelf * (books_after_removal + 2) = 3696) :=
by
  sorry

#check library_books_count

end NUMINAMATH_CALUDE_library_books_count_l1930_193001


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_equals_one_l1930_193059

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number in question -/
def complex_number (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 2*m⟩

/-- Theorem stating that if the complex number is purely imaginary, then m = 1 -/
theorem purely_imaginary_implies_m_equals_one :
  ∀ m : ℝ, is_purely_imaginary (complex_number m) → m = 1 := by
  sorry

#check purely_imaginary_implies_m_equals_one

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_equals_one_l1930_193059


namespace NUMINAMATH_CALUDE_probability_two_girls_l1930_193028

theorem probability_two_girls (total_members : ℕ) (girl_members : ℕ) : 
  total_members = 15 → girl_members = 6 → 
  (Nat.choose girl_members 2 : ℚ) / (Nat.choose total_members 2 : ℚ) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l1930_193028


namespace NUMINAMATH_CALUDE_unique_solution_system_l1930_193099

theorem unique_solution_system (x y : ℝ) : 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - 2*y = (x - 2) + (2*y - 2)) → 
  x = 6 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1930_193099


namespace NUMINAMATH_CALUDE_profit_fluctuation_l1930_193039

theorem profit_fluctuation (march_profit : ℝ) (april_may_decrease : ℝ) :
  let april_profit := march_profit * 1.5
  let may_profit := april_profit * (1 - april_may_decrease / 100)
  let june_profit := may_profit * 1.5
  june_profit = march_profit * 1.8 →
  april_may_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_fluctuation_l1930_193039


namespace NUMINAMATH_CALUDE_cube_diagonal_l1930_193038

theorem cube_diagonal (s : ℝ) (h1 : 6 * s^2 = 54) (h2 : 12 * s = 36) :
  ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d^2 = 3 * s^2 := by
  sorry

#check cube_diagonal

end NUMINAMATH_CALUDE_cube_diagonal_l1930_193038


namespace NUMINAMATH_CALUDE_range_of_b_l1930_193026

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * Real.log (x + 1) - x - b
def g (x : ℝ) : ℝ := Real.exp x

-- Define the point P
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- State the theorem
theorem range_of_b (a : ℝ) (x₀ : ℝ) (h1 : 0 < x₀ ∧ x₀ < Real.exp 1 - 1) :
  ∃ b : ℝ, 0 < b ∧ b < 1 - 1 / Real.exp 1 ∧
  ∃ y₀ : ℝ, 
    -- P is on the curve f
    y₀ = f a b x₀ ∧
    -- OP is the tangent line of f
    (deriv (f a b) x₀ = y₀ / x₀) ∧
    -- OP is perpendicular to a tangent line of g passing through the origin
    ∃ m : ℝ, deriv g m * (y₀ / x₀) = -1 ∧ g m = m * (deriv g m) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l1930_193026


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1930_193002

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ → Prop :=
  λ b => a + b = 0

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 (-2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1930_193002


namespace NUMINAMATH_CALUDE_daal_consumption_reduction_l1930_193013

theorem daal_consumption_reduction (old_price new_price : ℝ) 
  (hold_price : old_price = 16) 
  (hnew_price : new_price = 20) : 
  (new_price - old_price) / old_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_daal_consumption_reduction_l1930_193013


namespace NUMINAMATH_CALUDE_square_sum_xyz_l1930_193083

/-- Given real numbers x, y, and z satisfying the following conditions:
  1. 2x(y + z) = 1 + yz
  2. 1/x - 2/y = 3/2
  3. x + y + 1/2 = 0
  Prove that (x + y + z)^2 = 1/90 -/
theorem square_sum_xyz (x y z : ℝ) 
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z)^2 = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xyz_l1930_193083


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1930_193098

/-- The length of the minor axis of the ellipse 9x^2 + y^2 = 36 is 4 -/
theorem ellipse_minor_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, (x, y) ∈ ellipse ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
    2 * b = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1930_193098


namespace NUMINAMATH_CALUDE_morning_and_evening_emails_sum_l1930_193060

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- Theorem: The sum of emails Jack received in the morning and evening is 11 -/
theorem morning_and_evening_emails_sum :
  morning_emails + evening_emails = 11 := by sorry

end NUMINAMATH_CALUDE_morning_and_evening_emails_sum_l1930_193060


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1930_193033

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2 : ℝ) = π * (r₁^2 - r₂^2) :=
by sorry

/-- The area of a ring formed by concentric circles with radii 12 and 7 is 95π -/
theorem area_of_specific_ring : 
  (π * 12^2 - π * 7^2 : ℝ) = 95 * π :=
by sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l1930_193033


namespace NUMINAMATH_CALUDE_remaining_score_proof_l1930_193073

theorem remaining_score_proof (scores : List ℕ) (average : ℕ) : 
  scores = [85, 95, 75, 65] → 
  average = 80 → 
  scores.length + 1 = 5 →
  (scores.sum + (5 * average - scores.sum)) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_remaining_score_proof_l1930_193073


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l1930_193078

theorem negative_integer_equation_solution :
  ∃ (M : ℤ), (M < 0) ∧ (2 * M^2 + M = 12) → M = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l1930_193078


namespace NUMINAMATH_CALUDE_least_valid_number_l1930_193061

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 6 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  n % 9 = 2 ∧
  n % 11 = 2

theorem least_valid_number : 
  is_valid 27722 ∧ ∀ m : ℕ, m < 27722 → ¬is_valid m :=
by sorry

end NUMINAMATH_CALUDE_least_valid_number_l1930_193061


namespace NUMINAMATH_CALUDE_sugar_amount_l1930_193031

/-- The number of cups of sugar in Mary's cake recipe -/
def sugar : ℕ := sorry

/-- The total amount of flour needed for the recipe in cups -/
def total_flour : ℕ := 9

/-- The amount of flour already added in cups -/
def flour_added : ℕ := 2

/-- The remaining flour to be added is 1 cup more than the amount of sugar -/
axiom remaining_flour_sugar_relation : total_flour - flour_added = sugar + 1

theorem sugar_amount : sugar = 6 := by sorry

end NUMINAMATH_CALUDE_sugar_amount_l1930_193031


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_six_l1930_193068

/-- Box A contains ping-pong balls numbered 1 and 2 -/
def box_A : Finset ℕ := {1, 2}

/-- Box B contains ping-pong balls numbered 3, 4, 5, and 6 -/
def box_B : Finset ℕ := {3, 4, 5, 6}

/-- The set of all possible outcomes when drawing one ball from each box -/
def all_outcomes : Finset (ℕ × ℕ) :=
  box_A.product box_B

/-- The set of favorable outcomes (sum greater than 6) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 > 6)

/-- The probability of drawing balls with sum greater than 6 -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem probability_sum_greater_than_six : probability = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_six_l1930_193068


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1930_193062

theorem complex_fraction_sum (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1930_193062
