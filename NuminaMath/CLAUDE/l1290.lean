import Mathlib

namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l1290_129005

theorem smallest_divisible_by_one_to_ten : 
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
    (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l1290_129005


namespace NUMINAMATH_CALUDE_arithmetic_mean_x_y_l1290_129076

/-- Given two real numbers x and y satisfying certain conditions, 
    prove that their arithmetic mean is 3/4 -/
theorem arithmetic_mean_x_y (x y : ℝ) 
  (h1 : x * y > 0)
  (h2 : 2 * x * (1/2) + 1 * (-1/(2*y)) = 0)  -- Perpendicularity condition
  (h3 : y / x = 2 / y)  -- Geometric sequence condition
  : (x + y) / 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_x_y_l1290_129076


namespace NUMINAMATH_CALUDE_vector_sum_equals_c_l1290_129094

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (5, 1)

theorem vector_sum_equals_c : c + a + b = c := by sorry

end NUMINAMATH_CALUDE_vector_sum_equals_c_l1290_129094


namespace NUMINAMATH_CALUDE_right_triangle_from_conditions_l1290_129017

/-- Given a triangle ABC with side lengths a, b, and c satisfying certain conditions,
    prove that it is a right triangle. -/
theorem right_triangle_from_conditions (a b c : ℝ) (h1 : a + c = 2 * b) (h2 : c - a = 1 / 2 * b) :
  c^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_conditions_l1290_129017


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1290_129054

/-- Given an arithmetic sequence {aₙ}, prove that a₁₈ = 8 when a₄ + a₈ = 10 and a₁₀ = 6 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  a 4 + a 8 = 10 →
  a 10 = 6 →
  a 18 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1290_129054


namespace NUMINAMATH_CALUDE_polynomial_product_bound_l1290_129067

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that |P(x) * P(1/x)| ≤ 1 for all positive real x -/
def HasBoundedProduct (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, x > 0 → |P x * P (1/x)| ≤ 1

/-- The form c * x^n where |c| ≤ 1 and n is a non-negative integer -/
def IsMonomial (P : RealPolynomial) : Prop :=
  ∃ (c : ℝ) (n : ℕ), (|c| ≤ 1) ∧ (∀ x : ℝ, P x = c * x^n)

/-- The main theorem -/
theorem polynomial_product_bound (P : RealPolynomial) :
  HasBoundedProduct P → IsMonomial P := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_bound_l1290_129067


namespace NUMINAMATH_CALUDE_fourth_side_is_six_l1290_129004

/-- Represents a quadrilateral pyramid with a base ABCD and apex S -/
structure QuadrilateralPyramid where
  /-- Length of side AB of the base -/
  ab : ℝ
  /-- Length of side BC of the base -/
  bc : ℝ
  /-- Length of side CD of the base -/
  cd : ℝ
  /-- Length of side DA of the base -/
  da : ℝ
  /-- Predicate indicating that all dihedral angles at the base are equal -/
  equal_dihedral_angles : Prop

/-- Theorem stating that for a quadrilateral pyramid with given side lengths and equal dihedral angles,
    the fourth side of the base is 6 -/
theorem fourth_side_is_six (p : QuadrilateralPyramid)
  (h1 : p.ab = 5)
  (h2 : p.bc = 7)
  (h3 : p.cd = 8)
  (h4 : p.equal_dihedral_angles) :
  p.da = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_is_six_l1290_129004


namespace NUMINAMATH_CALUDE_unique_base_representation_l1290_129020

theorem unique_base_representation :
  ∃! (x y z b : ℕ), 
    1987 = x * b^2 + y * b + z ∧
    b > 1 ∧
    x < b ∧ y < b ∧ z < b ∧
    x + y + z = 25 ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_representation_l1290_129020


namespace NUMINAMATH_CALUDE_mango_rate_is_65_l1290_129023

/-- The rate per kg for mangoes given the following conditions:
    - Tom purchased 8 kg of apples at 70 per kg
    - Tom purchased 9 kg of mangoes
    - Tom paid a total of 1145 to the shopkeeper -/
def mango_rate (apple_weight : ℕ) (apple_rate : ℕ) (mango_weight : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_weight * apple_rate) / mango_weight

/-- Theorem stating that the rate per kg for mangoes is 65 -/
theorem mango_rate_is_65 : mango_rate 8 70 9 1145 = 65 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_is_65_l1290_129023


namespace NUMINAMATH_CALUDE_no_formula_matches_l1290_129015

def x_values : List ℕ := [1, 2, 3, 4, 5]
def y_values : List ℕ := [4, 12, 28, 52, 84]

def formula_a (x : ℕ) : ℕ := 4 * x^2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 3 * x + 1
def formula_c (x : ℕ) : ℕ := 5 * x^3 - 2 * x
def formula_d (x : ℕ) : ℕ := 4 * x^2 + 4 * x

theorem no_formula_matches : 
  ∀ (i : Fin 5), 
    (formula_a (x_values.get i) ≠ y_values.get i) ∧
    (formula_b (x_values.get i) ≠ y_values.get i) ∧
    (formula_c (x_values.get i) ≠ y_values.get i) ∧
    (formula_d (x_values.get i) ≠ y_values.get i) := by
  sorry

end NUMINAMATH_CALUDE_no_formula_matches_l1290_129015


namespace NUMINAMATH_CALUDE_playground_dimensions_l1290_129040

theorem playground_dimensions :
  ∃! n : ℕ, n = (Finset.filter (fun pair : ℕ × ℕ =>
    pair.2 > pair.1 ∧
    (pair.1 - 4) * (pair.2 - 4) = 2 * pair.1 * pair.2 / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_playground_dimensions_l1290_129040


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_eight_l1290_129027

theorem greatest_three_digit_divisible_by_eight :
  ∃ n : ℕ, n = 992 ∧ 
    (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 8 = 0 → m ≤ n) ∧
    n % 8 = 0 ∧
    100 ≤ n ∧ n < 1000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_eight_l1290_129027


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1290_129033

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 5 > Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1290_129033


namespace NUMINAMATH_CALUDE_tangent_equality_implies_x_130_l1290_129089

theorem tangent_equality_implies_x_130 (x : ℝ) :
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 130 := by sorry

end NUMINAMATH_CALUDE_tangent_equality_implies_x_130_l1290_129089


namespace NUMINAMATH_CALUDE_six_bulb_illumination_l1290_129063

/-- The number of ways to illuminate a room with n light bulbs, where at least one bulb must be on -/
def illuminationWays (n : ℕ) : ℕ :=
  2^n - 1

/-- Theorem: The number of ways to illuminate a room with 6 light bulbs, 
    where at least one bulb must be on, is 63 -/
theorem six_bulb_illumination : illuminationWays 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_six_bulb_illumination_l1290_129063


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l1290_129047

theorem smallest_prime_dividing_sum : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ (7^15 + 9^17) ∧ ∀ (q : Nat), Nat.Prime q → q ∣ (7^15 + 9^17) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l1290_129047


namespace NUMINAMATH_CALUDE_employee_age_problem_l1290_129034

theorem employee_age_problem (total_employees : Nat) 
  (group1_count : Nat) (group1_avg_age : Nat)
  (group2_count : Nat) (group2_avg_age : Nat)
  (group3_count : Nat) (group3_avg_age : Nat)
  (avg_age_29 : Nat) :
  total_employees = 30 →
  group1_count = 10 →
  group1_avg_age = 24 →
  group2_count = 12 →
  group2_avg_age = 30 →
  group3_count = 7 →
  group3_avg_age = 35 →
  avg_age_29 = 29 →
  ∃ (age_30th : Nat), age_30th = 25 := by
sorry


end NUMINAMATH_CALUDE_employee_age_problem_l1290_129034


namespace NUMINAMATH_CALUDE_omega_properties_l1290_129011

/-- The weight function ω(n) that returns the sum of binary digits of n -/
def ω (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 2 + ω (n / 2))

/-- Theorem stating the properties of the ω function -/
theorem omega_properties :
  ∀ n : ℕ,
  (ω (2 * n) = ω n) ∧
  (ω (8 * n + 5) = ω (4 * n + 3)) ∧
  (ω ((2 ^ n) - 1) = n) :=
by sorry

end NUMINAMATH_CALUDE_omega_properties_l1290_129011


namespace NUMINAMATH_CALUDE_binomial_6_choose_3_l1290_129091

theorem binomial_6_choose_3 : Nat.choose 6 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_6_choose_3_l1290_129091


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1290_129090

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + 1 = 0 ∧ 
   ∀ y : ℝ, a * y^2 - 4 * y + 1 = 0 → y = x) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1290_129090


namespace NUMINAMATH_CALUDE_min_tiles_to_cover_region_l1290_129075

/-- The number of tiles needed to cover a rectangular region -/
def tiles_needed (tile_width : ℕ) (tile_height : ℕ) (region_width : ℕ) (region_height : ℕ) : ℕ :=
  let region_area := region_width * region_height
  let tile_area := tile_width * tile_height
  (region_area + tile_area - 1) / tile_area

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem min_tiles_to_cover_region :
  tiles_needed 5 7 (3 * feet_to_inches) (7 * feet_to_inches) = 87 := by
  sorry

#eval tiles_needed 5 7 (3 * feet_to_inches) (7 * feet_to_inches)

end NUMINAMATH_CALUDE_min_tiles_to_cover_region_l1290_129075


namespace NUMINAMATH_CALUDE_sticker_distribution_l1290_129099

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- There are 8 identical stickers -/
def num_stickers : ℕ := 8

/-- There are 4 sheets of paper -/
def num_sheets : ℕ := 4

theorem sticker_distribution :
  distribute num_stickers num_sheets = 15 :=
sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1290_129099


namespace NUMINAMATH_CALUDE_curve_expression_bound_l1290_129010

theorem curve_expression_bound (x y : ℝ) : 
  4 * x^2 + y^2 = 16 → -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_expression_bound_l1290_129010


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1290_129098

theorem sin_2alpha_value (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1290_129098


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1290_129013

theorem consecutive_integers_cube_sum (n : ℕ) :
  (n > 0) →
  ((n - 1)^2 + n^2 + (n + 1)^2 = 7805) →
  ((n - 1)^3 + n^3 + (n + 1)^3 = 398259) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1290_129013


namespace NUMINAMATH_CALUDE_montoya_family_food_budget_l1290_129070

theorem montoya_family_food_budget (grocery_fraction eating_out_fraction : ℝ) 
  (h1 : grocery_fraction = 0.6)
  (h2 : eating_out_fraction = 0.2) :
  grocery_fraction + eating_out_fraction = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_montoya_family_food_budget_l1290_129070


namespace NUMINAMATH_CALUDE_equation_solution_l1290_129078

theorem equation_solution : ∃! x : ℚ, 3 * x - 4 = -6 * x + 11 ∧ x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1290_129078


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1290_129064

theorem quadratic_max_value (m : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → 
    -(x - m)^2 + m^2 + 1 ≤ 4) ∧ 
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ 
    -(x - m)^2 + m^2 + 1 = 4) → 
  m = 2 ∨ m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1290_129064


namespace NUMINAMATH_CALUDE_A_B_symmetrical_wrt_origin_l1290_129019

/-- Two points are symmetrical with respect to the origin if their coordinates are negatives of each other -/
def symmetrical_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given points A and B in the Cartesian coordinate system -/
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-2, 1)

/-- Theorem: Points A and B are symmetrical with respect to the origin -/
theorem A_B_symmetrical_wrt_origin : symmetrical_wrt_origin A B := by
  sorry

end NUMINAMATH_CALUDE_A_B_symmetrical_wrt_origin_l1290_129019


namespace NUMINAMATH_CALUDE_game_packing_l1290_129072

theorem game_packing (initial_games : Nat) (sold_games : Nat) (games_per_box : Nat) :
  initial_games = 35 →
  sold_games = 19 →
  games_per_box = 8 →
  (initial_games - sold_games) / games_per_box = 2 := by
  sorry

end NUMINAMATH_CALUDE_game_packing_l1290_129072


namespace NUMINAMATH_CALUDE_no_natural_solutions_l1290_129059

theorem no_natural_solutions : ¬∃ (x y : ℕ), (2 * x + y) * (2 * y + x) = 2017^2017 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l1290_129059


namespace NUMINAMATH_CALUDE_rectangle_area_unchanged_l1290_129085

/-- Given a rectangle with area 432 square centimeters, prove that decreasing the length by 20%
    and increasing the width by 25% results in the same area. -/
theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) :
  (0.8 * l) * (1.25 * w) = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_unchanged_l1290_129085


namespace NUMINAMATH_CALUDE_f_min_max_l1290_129079

def f (x : ℝ) : ℝ := -2 * x + 1

theorem f_min_max :
  let a : ℝ := -2
  let b : ℝ := 2
  (∀ x ∈ Set.Icc a b, f x ≥ -3) ∧
  (∃ x ∈ Set.Icc a b, f x = -3) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l1290_129079


namespace NUMINAMATH_CALUDE_hearty_blue_packages_l1290_129007

/-- The number of packages of red beads -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := 320

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := (total_beads - red_packages * beads_per_package) / beads_per_package

theorem hearty_blue_packages : blue_packages = 3 := by
  sorry

end NUMINAMATH_CALUDE_hearty_blue_packages_l1290_129007


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1290_129084

-- Define what it means for an angle to be in the third quadrant
def in_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

-- Define what it means for an angle to be in the second quadrant
def in_second_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 180

-- Define what it means for an angle to be in the fourth quadrant
def in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 360

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  in_third_quadrant α → in_second_quadrant (α/2) ∨ in_fourth_quadrant (α/2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1290_129084


namespace NUMINAMATH_CALUDE_polynomial_degree_l1290_129002

/-- The degree of the polynomial (x^5-2x^4+3x^3+x-14)(4x^11-8x^8+7x^5+40)-(2x^3+3)^7 is 21 -/
theorem polynomial_degree : ∃ p : Polynomial ℝ, 
  p = (X^5 - 2*X^4 + 3*X^3 + X - 14) * (4*X^11 - 8*X^8 + 7*X^5 + 40) - (2*X^3 + 3)^7 ∧ 
  p.degree = some 21 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l1290_129002


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1290_129053

theorem fraction_to_decimal : (51 : ℚ) / 160 = 0.31875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1290_129053


namespace NUMINAMATH_CALUDE_beaver_carrot_count_l1290_129068

/-- Represents the number of carrots stored per burrow by the beaver -/
def beaver_carrots_per_burrow : ℕ := 4

/-- Represents the number of carrots stored per burrow by the rabbit -/
def rabbit_carrots_per_burrow : ℕ := 5

/-- Represents the difference in the number of burrows between the beaver and the rabbit -/
def burrow_difference : ℕ := 3

theorem beaver_carrot_count (beaver_burrows rabbit_burrows total_carrots : ℕ) :
  beaver_burrows = rabbit_burrows + burrow_difference →
  beaver_carrots_per_burrow * beaver_burrows = total_carrots →
  rabbit_carrots_per_burrow * rabbit_burrows = total_carrots →
  total_carrots = 60 := by
  sorry

end NUMINAMATH_CALUDE_beaver_carrot_count_l1290_129068


namespace NUMINAMATH_CALUDE_complex_number_location_l1290_129029

theorem complex_number_location : ∃ (z : ℂ), z = 2 / (1 - Complex.I) - 2 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1290_129029


namespace NUMINAMATH_CALUDE_simplified_expression_l1290_129095

theorem simplified_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_l1290_129095


namespace NUMINAMATH_CALUDE_b_over_a_range_l1290_129074

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0
  sine_law : a / Real.sin A = b / Real.sin B
  B_eq_2A : B = 2 * A

-- Theorem statement
theorem b_over_a_range (t : AcuteTriangle) : Real.sqrt 2 < t.b / t.a ∧ t.b / t.a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_b_over_a_range_l1290_129074


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l1290_129055

theorem president_vice_president_selection (n : ℕ) (h : n = 6) : 
  (n * (n - 1) : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l1290_129055


namespace NUMINAMATH_CALUDE_a_eq_4_neither_sufficient_nor_necessary_l1290_129062

/-- Two lines in the real plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line l₁: ax + 8y - 8 = 0 -/
def l1 (a : ℝ) : Line :=
  { a := a, b := 8, c := -8 }

/-- The second line l₂: 2x + ay - a = 0 -/
def l2 (a : ℝ) : Line :=
  { a := 2, b := a, c := -a }

/-- The main theorem stating that a = 4 is neither sufficient nor necessary for parallelism -/
theorem a_eq_4_neither_sufficient_nor_necessary :
  ∃ a : ℝ, a ≠ 4 ∧ parallel (l1 a) (l2 a) ∧
  ∃ b : ℝ, b = 4 ∧ ¬parallel (l1 b) (l2 b) :=
sorry

end NUMINAMATH_CALUDE_a_eq_4_neither_sufficient_nor_necessary_l1290_129062


namespace NUMINAMATH_CALUDE_abs_sum_min_value_abs_sum_min_value_achieved_l1290_129045

theorem abs_sum_min_value (x : ℝ) : 
  |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5| ≥ 6 :=
sorry

theorem abs_sum_min_value_achieved : 
  ∃ x : ℝ, |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5| = 6 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_min_value_abs_sum_min_value_achieved_l1290_129045


namespace NUMINAMATH_CALUDE_denise_expenditure_l1290_129088

/-- Represents the menu items --/
inductive MenuItem
| SimpleDish
| MeatDish
| FishDish
| MilkSmoothie
| FruitSmoothie
| SpecialSmoothie

/-- Price of a menu item in reais --/
def price (item : MenuItem) : ℕ :=
  match item with
  | MenuItem.SimpleDish => 7
  | MenuItem.MeatDish => 11
  | MenuItem.FishDish => 14
  | MenuItem.MilkSmoothie => 6
  | MenuItem.FruitSmoothie => 7
  | MenuItem.SpecialSmoothie => 9

/-- Total cost of a meal (one dish and one smoothie) --/
def mealCost (dish : MenuItem) (smoothie : MenuItem) : ℕ :=
  price dish + price smoothie

/-- Denise's possible expenditures --/
def deniseExpenditure : Set ℕ :=
  {14, 17}

/-- Theorem stating Denise's possible expenditures --/
theorem denise_expenditure :
  ∀ (deniseDish deniseSmoothie julioDish julioSmoothie : MenuItem),
    mealCost julioDish julioSmoothie = mealCost deniseDish deniseSmoothie + 6 →
    mealCost deniseDish deniseSmoothie ∈ deniseExpenditure :=
by sorry

end NUMINAMATH_CALUDE_denise_expenditure_l1290_129088


namespace NUMINAMATH_CALUDE_sqrt_of_square_neg_l1290_129071

theorem sqrt_of_square_neg (a : ℝ) (h : a < 0) : Real.sqrt (a ^ 2) = -a := by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_neg_l1290_129071


namespace NUMINAMATH_CALUDE_tangent_line_constraint_l1290_129061

/-- Given a cubic function f(x) = x³ - (1/2)x² + bx + c, 
    if f has a tangent line parallel to y = 1, then b ≤ 1/12 -/
theorem tangent_line_constraint (b c : ℝ) : 
  (∃ x : ℝ, (3*x^2 - x + b) = 1) → b ≤ 1/12 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_constraint_l1290_129061


namespace NUMINAMATH_CALUDE_ball_problem_l1290_129052

theorem ball_problem (x : ℕ) : 
  (x > 0) →                                      -- Ensure x is positive
  ((x + 1) / (2 * x + 1) - x / (2 * x) = 1 / 22) →  -- Probability condition
  (2 * x = 10) :=                                -- Conclusion
by sorry

end NUMINAMATH_CALUDE_ball_problem_l1290_129052


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l1290_129030

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  m_pos : m > 0
  parallel : 1 / (-2) = 2 / n
  distance : 2 * Real.sqrt 5 = |m + 3| / Real.sqrt 5

/-- The sum of coefficients m and n for parallel lines with given properties -/
theorem parallel_lines_sum (l : ParallelLines) : l.m + l.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l1290_129030


namespace NUMINAMATH_CALUDE_stadium_seats_pattern_l1290_129008

/-- Represents the number of seats in a row of the stadium -/
def seats (n : ℕ) : ℕ := n + 49

/-- The theorem states that the number of seats in each row follows the given pattern -/
theorem stadium_seats_pattern (n : ℕ) (h : 1 ≤ n ∧ n ≤ 40) : 
  seats n = 50 + (n - 1) := by sorry

end NUMINAMATH_CALUDE_stadium_seats_pattern_l1290_129008


namespace NUMINAMATH_CALUDE_garden_area_l1290_129086

/-- The total area of a garden with a semicircle and an attached square -/
theorem garden_area (diameter : ℝ) (h : diameter = 8) : 
  let radius := diameter / 2
  let semicircle_area := π * radius^2 / 2
  let square_area := radius^2
  semicircle_area + square_area = 8 * π + 16 := by
  sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l1290_129086


namespace NUMINAMATH_CALUDE_g_of_2_eq_neg_1_l1290_129082

/-- The function g defined as g(x) = x^2 - 3x + 1 -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem stating that g(2) = -1 -/
theorem g_of_2_eq_neg_1 : g 2 = -1 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_neg_1_l1290_129082


namespace NUMINAMATH_CALUDE_tan_45_degrees_l1290_129046

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l1290_129046


namespace NUMINAMATH_CALUDE_andy_ball_count_l1290_129012

theorem andy_ball_count : ∃ (a r m : ℕ), 
  (a = 2 * r) ∧ 
  (a = m + 5) ∧ 
  (a + r + m = 35) → 
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_andy_ball_count_l1290_129012


namespace NUMINAMATH_CALUDE_vector_parallelism_l1290_129024

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

theorem vector_parallelism :
  ∃! k : ℝ, parallel ((k * a.1 + b.1, k * a.2 + b.2) : ℝ × ℝ) ((a.1 - 3 * b.1, a.2 - 3 * b.2) : ℝ × ℝ) ∧
  k = -1/3 := by sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1290_129024


namespace NUMINAMATH_CALUDE_problem_solution_l1290_129035

theorem problem_solution :
  ∀ (a b c : ℕ),
  ({a, b, c} : Set ℕ) = {0, 1, 2} →
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0)) ∨
   ((a ≠ 2) ∧ (b = 2) ∧ (c ≠ 0))) →
  10 * a + 2 * b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1290_129035


namespace NUMINAMATH_CALUDE_nth_equation_proof_l1290_129001

theorem nth_equation_proof (n : ℕ) : (((n + 3)^2 - n^2 - 9) / 2 : ℚ) = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l1290_129001


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1290_129048

theorem no_solution_for_equation : ¬ ∃ (x : ℝ), (x - 8) / (x - 7) - 8 = 1 / (7 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1290_129048


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1290_129096

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ y => y^2 + 7*y + 10 + (y + 2)*(y + 8)
  (f (-2) = 0 ∧ f (-13/2) = 0) ∧
  ∀ y : ℝ, f y = 0 → (y = -2 ∨ y = -13/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1290_129096


namespace NUMINAMATH_CALUDE_cookies_left_l1290_129042

def initial_cookies : ℕ := 32
def eaten_cookies : ℕ := 9

theorem cookies_left : initial_cookies - eaten_cookies = 23 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l1290_129042


namespace NUMINAMATH_CALUDE_optimal_feeding_program_l1290_129093

/-- Represents a feeding program for animals -/
structure FeedingProgram where
  x : ℝ  -- Amount of first feed in kg
  y : ℝ  -- Amount of second feed in kg

/-- Nutrient requirements for each animal per day -/
def nutrientRequirements : ℝ × ℝ × ℝ := (45, 60, 5)

/-- Nutrient content of first feed per kg -/
def firstFeedContent : ℝ × ℝ := (10, 10)

/-- Nutrient content of second feed per kg -/
def secondFeedContent : ℝ × ℝ × ℝ := (10, 20, 5)

/-- Cost of feeds in Ft/q -/
def feedCosts : ℝ × ℝ := (30, 120)

/-- Feeding loss percentages -/
def feedingLoss : ℝ × ℝ := (0.1, 0.2)

/-- Check if a feeding program satisfies nutrient requirements -/
def satisfiesRequirements (fp : FeedingProgram) : Prop :=
  let (reqA, reqB, reqC) := nutrientRequirements
  let (firstA, firstB) := firstFeedContent
  let (secondA, secondB, secondC) := secondFeedContent
  firstA * fp.x + secondA * fp.y ≥ reqA ∧
  firstB * fp.x + secondB * fp.y ≥ reqB ∧
  secondC * fp.y ≥ reqC

/-- Calculate the cost of a feeding program -/
def calculateCost (fp : FeedingProgram) : ℝ :=
  let (costFirst, costSecond) := feedCosts
  costFirst * fp.x + costSecond * fp.y

/-- Calculate the feeding loss of a feeding program -/
def calculateLoss (fp : FeedingProgram) : ℝ :=
  let (lossFirst, lossSecond) := feedingLoss
  lossFirst * fp.x + lossSecond * fp.y

/-- Theorem stating that (4, 1) is the optimal feeding program -/
theorem optimal_feeding_program :
  let optimalProgram := FeedingProgram.mk 4 1
  satisfiesRequirements optimalProgram ∧
  ∀ fp : FeedingProgram, satisfiesRequirements fp →
    calculateCost optimalProgram ≤ calculateCost fp ∧
    calculateLoss optimalProgram ≤ calculateLoss fp :=
by sorry

end NUMINAMATH_CALUDE_optimal_feeding_program_l1290_129093


namespace NUMINAMATH_CALUDE_probability_three_correct_out_of_seven_l1290_129000

/-- The number of derangements of n elements -/
def derangement (n : ℕ) : ℕ := sorry

/-- The probability of exactly k people receiving their correct letter when n letters are randomly distributed to n people -/
def probability_correct_letters (n k : ℕ) : ℚ :=
  (Nat.choose n k * derangement (n - k)) / n.factorial

theorem probability_three_correct_out_of_seven :
  probability_correct_letters 7 3 = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_three_correct_out_of_seven_l1290_129000


namespace NUMINAMATH_CALUDE_monkey_multiplication_l1290_129056

/-- The number of spirit monkeys created from one hair -/
def spiritsPerHair : ℕ := 3

/-- The number of new spirit monkeys created by each existing spirit monkey per second -/
def splitRate : ℕ := 3

/-- The number of hairs the Monkey King pulls out -/
def numHairs : ℕ := 10

/-- The number of seconds that pass -/
def timeElapsed : ℕ := 5

/-- The total number of monkeys after the given time -/
def totalMonkeys : ℕ := numHairs * spiritsPerHair * splitRate ^ timeElapsed + 1

theorem monkey_multiplication (spiritsPerHair splitRate numHairs timeElapsed : ℕ) :
  totalMonkeys = 7290 :=
sorry

end NUMINAMATH_CALUDE_monkey_multiplication_l1290_129056


namespace NUMINAMATH_CALUDE_max_candy_remainder_l1290_129036

theorem max_candy_remainder (x : ℕ) : 
  ∃ (q r : ℕ), x = 9 * q + r ∧ r < 9 ∧ r ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_remainder_l1290_129036


namespace NUMINAMATH_CALUDE_east_northwest_angle_is_144_degrees_l1290_129087

/-- Represents a circular garden with equally spaced radial paths -/
structure CircularGarden where
  numPaths : ℕ
  northPathIndex : ℕ
  eastPathIndex : ℕ
  northwestPathIndex : ℕ

/-- Calculates the angle between two paths in a circular garden -/
def angleBetweenPaths (garden : CircularGarden) (path1 : ℕ) (path2 : ℕ) : ℝ :=
  let angleBetweenConsecutivePaths := 360 / garden.numPaths
  let pathDifference := (path2 - path1 + garden.numPaths) % garden.numPaths
  pathDifference * angleBetweenConsecutivePaths

/-- Theorem stating that the smaller angle between East and Northwest paths is 144 degrees -/
theorem east_northwest_angle_is_144_degrees (garden : CircularGarden) :
  garden.numPaths = 10 →
  garden.northPathIndex = 0 →
  garden.eastPathIndex = 3 →
  garden.northwestPathIndex = 8 →
  min (angleBetweenPaths garden garden.eastPathIndex garden.northwestPathIndex)
      (angleBetweenPaths garden garden.northwestPathIndex garden.eastPathIndex) = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_east_northwest_angle_is_144_degrees_l1290_129087


namespace NUMINAMATH_CALUDE_students_answering_one_question_l1290_129069

/-- Represents the number of questions answered by students in each grade -/
structure GradeAnswers :=
  (g1 g2 g3 g4 g5 : Nat)

/-- The problem setup -/
structure ProblemSetup :=
  (total_students : Nat)
  (total_grades : Nat)
  (total_questions : Nat)
  (grade_answers : GradeAnswers)

/-- The conditions of the problem -/
def satisfies_conditions (setup : ProblemSetup) : Prop :=
  setup.total_students = 30 ∧
  setup.total_grades = 5 ∧
  setup.total_questions = 40 ∧
  setup.grade_answers.g1 < setup.grade_answers.g2 ∧
  setup.grade_answers.g2 < setup.grade_answers.g3 ∧
  setup.grade_answers.g3 < setup.grade_answers.g4 ∧
  setup.grade_answers.g4 < setup.grade_answers.g5 ∧
  setup.grade_answers.g1 ≥ 1 ∧
  setup.grade_answers.g2 ≥ 1 ∧
  setup.grade_answers.g3 ≥ 1 ∧
  setup.grade_answers.g4 ≥ 1 ∧
  setup.grade_answers.g5 ≥ 1

/-- The theorem to be proved -/
theorem students_answering_one_question (setup : ProblemSetup) 
  (h : satisfies_conditions setup) : 
  setup.total_students - (setup.total_questions - (setup.grade_answers.g1 + 
  setup.grade_answers.g2 + setup.grade_answers.g3 + setup.grade_answers.g4 + 
  setup.grade_answers.g5)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_students_answering_one_question_l1290_129069


namespace NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l1290_129065

theorem modulus_of_5_minus_12i : Complex.abs (5 - 12*I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l1290_129065


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l1290_129041

/-- Given a mixture of raisins and nuts, where the cost of nuts is twice that of raisins,
    prove that the cost of raisins is 3/11 of the total cost of the mixture. -/
theorem raisin_cost_fraction (raisin_cost : ℚ) : 
  let raisin_pounds : ℚ := 3
  let nut_pounds : ℚ := 4
  let nut_cost : ℚ := 2 * raisin_cost
  let total_raisin_cost : ℚ := raisin_pounds * raisin_cost
  let total_nut_cost : ℚ := nut_pounds * nut_cost
  let total_cost : ℚ := total_raisin_cost + total_nut_cost
  total_raisin_cost / total_cost = 3 / 11 := by
sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_l1290_129041


namespace NUMINAMATH_CALUDE_intersection_A_B_l1290_129081

def A : Set ℤ := {-1, 0, 1, 5, 8}
def B : Set ℤ := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1290_129081


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1290_129016

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ |x - y| = 1) →
  p = Real.sqrt (4*q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1290_129016


namespace NUMINAMATH_CALUDE_parabola_vertices_distance_l1290_129043

/-- The equation of the parabolas -/
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 4

/-- The y-coordinate of the vertex for the upper parabola (y ≥ 2) -/
def upper_vertex_y : ℝ := 3

/-- The y-coordinate of the vertex for the lower parabola (y < 2) -/
def lower_vertex_y : ℝ := -1

/-- The distance between the vertices of the parabolas -/
def vertex_distance : ℝ := |upper_vertex_y - lower_vertex_y|

theorem parabola_vertices_distance :
  vertex_distance = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertices_distance_l1290_129043


namespace NUMINAMATH_CALUDE_third_butcher_packages_l1290_129025

/-- Represents the number of packages delivered by each butcher and their delivery times -/
structure Delivery :=
  (x y z : ℕ)
  (t1 t2 t3 : ℕ)

/-- Defines the conditions of the delivery problem -/
def DeliveryProblem (d : Delivery) : Prop :=
  d.x = 10 ∧
  d.y = 7 ∧
  d.t1 = 8 ∧
  d.t2 = 10 ∧
  d.t3 = 18 ∧
  4 * d.x + 4 * d.y + 4 * d.z = 100

/-- Theorem stating that under the given conditions, the third butcher delivered 8 packages -/
theorem third_butcher_packages (d : Delivery) (h : DeliveryProblem d) : d.z = 8 := by
  sorry

end NUMINAMATH_CALUDE_third_butcher_packages_l1290_129025


namespace NUMINAMATH_CALUDE_sqrt_primes_not_arithmetic_progression_l1290_129051

theorem sqrt_primes_not_arithmetic_progression (a b c : ℕ) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  ¬∃ (d : ℝ), (Real.sqrt (a : ℝ) + d = Real.sqrt (b : ℝ) ∧ 
               Real.sqrt (b : ℝ) + d = Real.sqrt (c : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_primes_not_arithmetic_progression_l1290_129051


namespace NUMINAMATH_CALUDE_subset_probability_l1290_129060

def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def T : Finset Char := {'a', 'b', 'c'}

theorem subset_probability : 
  (Finset.filter (fun X => X ⊆ T) (Finset.powerset S)).card / (Finset.powerset S).card = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_probability_l1290_129060


namespace NUMINAMATH_CALUDE_downstream_distance_is_24km_l1290_129022

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  still_water_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (scenario : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the downstream distance is 24 km -/
theorem downstream_distance_is_24km 
  (scenario : SwimmingScenario)
  (h1 : scenario.upstream_distance = 12)
  (h2 : scenario.upstream_time = 6)
  (h3 : scenario.downstream_time = 6)
  (h4 : scenario.still_water_speed = 3) :
  downstream_distance scenario = 24 :=
sorry

end NUMINAMATH_CALUDE_downstream_distance_is_24km_l1290_129022


namespace NUMINAMATH_CALUDE_wrong_observation_value_l1290_129057

theorem wrong_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (correct_value : ℝ)
  (new_mean : ℝ)
  (h1 : n = 50)
  (h2 : initial_mean = 40)
  (h3 : correct_value = 45)
  (h4 : new_mean = 40.66)
  : ∃ (wrong_value : ℝ),
    n * new_mean - n * initial_mean = correct_value - wrong_value ∧
    wrong_value = 12 :=
by sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l1290_129057


namespace NUMINAMATH_CALUDE_sons_age_l1290_129083

theorem sons_age (son_age father_age : ℕ) : 
  father_age = 6 * son_age →
  father_age + 6 + son_age + 6 = 68 →
  son_age = 8 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1290_129083


namespace NUMINAMATH_CALUDE_tan_135_deg_l1290_129006

/-- Tangent of 135 degrees is -1 -/
theorem tan_135_deg : Real.tan (135 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_135_deg_l1290_129006


namespace NUMINAMATH_CALUDE_dave_cleaning_time_l1290_129018

/-- Proves that Dave's cleaning time is 15 minutes, given Carla's cleaning time and the ratio of Dave's to Carla's time -/
theorem dave_cleaning_time (carla_time : ℕ) (dave_ratio : ℚ) : 
  carla_time = 40 → dave_ratio = 3/8 → dave_ratio * carla_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_dave_cleaning_time_l1290_129018


namespace NUMINAMATH_CALUDE_unknown_bill_value_is_five_l1290_129092

/-- Represents the value of a US dollar bill -/
inductive USBill
| One
| Two
| Five
| Ten
| Twenty
| Fifty
| Hundred

/-- The wallet contents before purchase -/
structure Wallet where
  twenties : Nat
  unknown_bills : Nat
  unknown_bill_value : USBill
  loose_coins : Rat

def Wallet.total_value (w : Wallet) : Rat :=
  20 * w.twenties + 
  (match w.unknown_bill_value with
   | USBill.One => 1
   | USBill.Two => 2
   | USBill.Five => 5
   | USBill.Ten => 10
   | USBill.Twenty => 20
   | USBill.Fifty => 50
   | USBill.Hundred => 100) * w.unknown_bills +
  w.loose_coins

theorem unknown_bill_value_is_five (w : Wallet) (h1 : w.twenties = 2) 
  (h2 : w.loose_coins = 9/2) (h3 : Wallet.total_value w - 35/2 = 42) :
  w.unknown_bill_value = USBill.Five := by
  sorry

end NUMINAMATH_CALUDE_unknown_bill_value_is_five_l1290_129092


namespace NUMINAMATH_CALUDE_triangle_shape_l1290_129037

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem triangle_shape (t : Triangle) 
  (p : Vector2D) 
  (q : Vector2D) 
  (hp : p = ⟨t.c^2, t.a^2⟩) 
  (hq : q = ⟨Real.tan t.C, Real.tan t.A⟩) 
  (hpq : parallel p q) : 
  (t.a = t.c) ∨ (t.b^2 = t.a^2 + t.c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1290_129037


namespace NUMINAMATH_CALUDE_pen_count_problem_l1290_129049

theorem pen_count_problem :
  ∃! X : ℕ, 1 ≤ X ∧ X < 100 ∧ 
  X % 9 = 1 ∧ X % 5 = 3 ∧ X % 2 = 1 ∧ 
  X = 73 :=
by sorry

end NUMINAMATH_CALUDE_pen_count_problem_l1290_129049


namespace NUMINAMATH_CALUDE_magazine_subscription_issues_l1290_129028

/-- Proves that the number of issues in an 18-month magazine subscription is 36,
    given the normal price, promotional discount per issue, and total promotional discount. -/
theorem magazine_subscription_issues
  (normal_price : ℝ)
  (subscription_duration : ℝ)
  (discount_per_issue : ℝ)
  (total_discount : ℝ)
  (h1 : normal_price = 34)
  (h2 : subscription_duration = 18)
  (h3 : discount_per_issue = 0.25)
  (h4 : total_discount = 9) :
  (total_discount / discount_per_issue : ℝ) = 36 := by
sorry

end NUMINAMATH_CALUDE_magazine_subscription_issues_l1290_129028


namespace NUMINAMATH_CALUDE_apple_to_mango_ratio_l1290_129039

/-- Represents the total produce of fruits in kilograms -/
structure FruitProduce where
  apples : ℝ
  mangoes : ℝ
  oranges : ℝ

/-- Represents the fruit selling details -/
structure FruitSale where
  price_per_kg : ℝ
  total_amount : ℝ

/-- Theorem stating the ratio of apple to mango production -/
theorem apple_to_mango_ratio (fp : FruitProduce) (fs : FruitSale) :
  fp.mangoes = 400 ∧
  fp.oranges = fp.mangoes + 200 ∧
  fs.price_per_kg = 50 ∧
  fs.total_amount = 90000 ∧
  fs.total_amount = fs.price_per_kg * (fp.apples + fp.mangoes + fp.oranges) →
  fp.apples / fp.mangoes = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_apple_to_mango_ratio_l1290_129039


namespace NUMINAMATH_CALUDE_remaining_problems_calculation_l1290_129050

/-- Given a number of worksheets, problems per worksheet, and graded worksheets,
    calculate the number of remaining problems to grade. -/
def remaining_problems (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem remaining_problems_calculation :
  remaining_problems 16 4 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_calculation_l1290_129050


namespace NUMINAMATH_CALUDE_cosine_symmetry_l1290_129009

/-- A function f is symmetric about the origin if f(-x) = -f(x) for all x -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cosine_symmetry (φ : ℝ) :
  SymmetricAboutOrigin (fun x ↦ Real.cos (3 * x + φ)) →
  ¬ ∃ k : ℤ, φ = k * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cosine_symmetry_l1290_129009


namespace NUMINAMATH_CALUDE_mint_code_is_6785_l1290_129021

-- Define a function that maps characters to digits based on their position in GREAT MIND
def code_to_digit (c : Char) : Nat :=
  match c with
  | 'G' => 1
  | 'R' => 2
  | 'E' => 3
  | 'A' => 4
  | 'T' => 5
  | 'M' => 6
  | 'I' => 7
  | 'N' => 8
  | 'D' => 9
  | _ => 0

-- Define a function that converts a string to a number using the code
def code_to_number (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + code_to_digit c) 0

-- Theorem stating that MINT represents 6785
theorem mint_code_is_6785 : code_to_number "MINT" = 6785 := by
  sorry

end NUMINAMATH_CALUDE_mint_code_is_6785_l1290_129021


namespace NUMINAMATH_CALUDE_wall_length_calculation_l1290_129038

/-- Given a square mirror and a rectangular wall, if the mirror's area is half the wall's area,
    prove that the wall's length is approximately 27 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 24 →
  wall_width = 42 →
  (mirror_side * mirror_side) * 2 = wall_width * (27 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l1290_129038


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1290_129014

theorem quadratic_inequality (x : ℝ) : (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1290_129014


namespace NUMINAMATH_CALUDE_correct_equation_l1290_129073

theorem correct_equation (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l1290_129073


namespace NUMINAMATH_CALUDE_chocolate_division_l1290_129026

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) :
  total_chocolate = 35 / 4 ∧
  num_piles = 5 ∧
  piles_to_shaina = 2 →
  piles_to_shaina * (total_chocolate / num_piles) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1290_129026


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l1290_129066

theorem geometry_biology_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119) :
  (min geometry biology) - (geometry + biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l1290_129066


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1290_129097

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x + 3 = a * (x - h)^2 + k) → a + h + k = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1290_129097


namespace NUMINAMATH_CALUDE_largest_product_of_three_l1290_129003

def S : Finset Int := {-4, -3, -1, 3, 5}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    x * y * z ≤ 60) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x * y * z = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l1290_129003


namespace NUMINAMATH_CALUDE_venus_hall_rental_cost_l1290_129044

theorem venus_hall_rental_cost (caesars_rental : ℕ) (caesars_meal : ℕ) (venus_meal : ℕ) (guests : ℕ) :
  caesars_rental = 800 →
  caesars_meal = 30 →
  venus_meal = 35 →
  guests = 60 →
  ∃ venus_rental : ℕ, venus_rental = 500 ∧ 
    caesars_rental + guests * caesars_meal = venus_rental + guests * venus_meal :=
by sorry

end NUMINAMATH_CALUDE_venus_hall_rental_cost_l1290_129044


namespace NUMINAMATH_CALUDE_windshield_wiper_area_l1290_129032

/-- The area swept by two semicircular windshield wipers -/
theorem windshield_wiper_area (L : ℝ) (h : L > 0) :
  let area := (2 / 3 * π + Real.sqrt 3 / 4) * L^2
  area = (π * L^2) - ((1 / 3 * π - Real.sqrt 3 / 4) * L^2) :=
by sorry

end NUMINAMATH_CALUDE_windshield_wiper_area_l1290_129032


namespace NUMINAMATH_CALUDE_jackson_charity_collection_l1290_129058

/-- Proves the number of houses Jackson needs to visit per day to meet his goal -/
theorem jackson_charity_collection (total_goal : ℕ) (days_per_week : ℕ) (monday_earnings : ℕ) (tuesday_earnings : ℕ) (houses_per_collection : ℕ) (earnings_per_collection : ℕ) : 
  total_goal = 1000 →
  days_per_week = 5 →
  monday_earnings = 300 →
  tuesday_earnings = 40 →
  houses_per_collection = 4 →
  earnings_per_collection = 10 →
  ∃ (houses_per_day : ℕ), 
    houses_per_day = 88 ∧ 
    (total_goal - monday_earnings - tuesday_earnings) = 
      (days_per_week - 2) * houses_per_day * (earnings_per_collection / houses_per_collection) :=
by
  sorry

end NUMINAMATH_CALUDE_jackson_charity_collection_l1290_129058


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l1290_129077

theorem smallest_angle_solution (x : Real) : 
  (8 * Real.sin x ^ 2 * Real.cos x ^ 4 - 8 * Real.sin x ^ 4 * Real.cos x ^ 2 = 1) →
  (x ≥ 0) →
  (∀ y, y > 0 ∧ y < x → 8 * Real.sin y ^ 2 * Real.cos y ^ 4 - 8 * Real.sin y ^ 4 * Real.cos y ^ 2 ≠ 1) →
  x = 10 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l1290_129077


namespace NUMINAMATH_CALUDE_cross_section_area_is_40_div_3_l1290_129080

/-- Right prism with isosceles triangle base -/
structure RightPrism where
  -- Base triangle
  AB : ℝ
  BC : ℝ
  angleABC : ℝ
  -- Intersection points
  AD_ratio : ℝ
  EC1_ratio : ℝ
  -- Conditions
  isIsosceles : AB = BC
  baseLength : AB = 5
  angleCondition : angleABC = 2 * Real.arcsin (3/5)
  adIntersection : AD_ratio = 1/3
  ec1Intersection : EC1_ratio = 1/3

/-- The area of the cross-section of the prism -/
def crossSectionArea (p : RightPrism) : ℝ :=
  sorry -- Actual calculation would go here

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_is_40_div_3 (p : RightPrism) :
  crossSectionArea p = 40/3 := by
  sorry

#check cross_section_area_is_40_div_3

end NUMINAMATH_CALUDE_cross_section_area_is_40_div_3_l1290_129080


namespace NUMINAMATH_CALUDE_philip_intersections_l1290_129031

theorem philip_intersections (crosswalks_per_intersection : ℕ) 
                              (lines_per_crosswalk : ℕ) 
                              (total_lines : ℕ) :
  crosswalks_per_intersection = 4 →
  lines_per_crosswalk = 20 →
  total_lines = 400 →
  total_lines / (crosswalks_per_intersection * lines_per_crosswalk) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_philip_intersections_l1290_129031
