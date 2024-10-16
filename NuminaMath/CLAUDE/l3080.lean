import Mathlib

namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l3080_308065

theorem combined_fuel_efficiency (d : ℝ) (h : d > 0) :
  let efficiency1 : ℝ := 50
  let efficiency2 : ℝ := 20
  let efficiency3 : ℝ := 15
  let total_distance : ℝ := 3 * d
  let total_fuel : ℝ := d / efficiency1 + d / efficiency2 + d / efficiency3
  total_distance / total_fuel = 900 / 41 :=
by sorry

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l3080_308065


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3080_308062

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1900 →
  loss_percentage = 18 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  selling_price = 1558 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3080_308062


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1001_l3080_308018

theorem largest_gcd_of_sum_1001 :
  ∃ (a b : ℕ+), a + b = 1001 ∧
  ∀ (c d : ℕ+), c + d = 1001 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 143 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1001_l3080_308018


namespace NUMINAMATH_CALUDE_claudia_coins_l3080_308048

/-- Represents the number of different coin combinations possible with n coins -/
def combinations (n : ℕ) : ℕ := sorry

/-- Represents the number of different values that can be formed with n coins -/
def values (n : ℕ) : ℕ := sorry

theorem claudia_coins :
  ∀ x y : ℕ,
  x + y = 15 →                           -- Total number of coins is 15
  combinations (x + y) = 23 →            -- 23 different combinations possible
  (∀ n : ℕ, n ≤ 10 → values n ≥ 15) →    -- At least 15 values with no more than 10 coins
  y = 9                                  -- Claudia has 9 10-cent coins
  := by sorry

end NUMINAMATH_CALUDE_claudia_coins_l3080_308048


namespace NUMINAMATH_CALUDE_compute_expression_l3080_308007

theorem compute_expression : 8 * (1 / 4)^4 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3080_308007


namespace NUMINAMATH_CALUDE_polygon_sides_l3080_308084

theorem polygon_sides (S : ℕ) (h : S = 2160) : ∃ n : ℕ, n = 14 ∧ S = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3080_308084


namespace NUMINAMATH_CALUDE_largest_difference_l3080_308074

def A : ℕ := 3 * 1005^1006
def B : ℕ := 1005^1006
def C : ℕ := 1004 * 1005^1005
def D : ℕ := 3 * 1005^1005
def E : ℕ := 1005^1005
def F : ℕ := 1005^1004

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 1005^1006)
  (hB : B = 1005^1006)
  (hC : C = 1004 * 1005^1005)
  (hD : D = 3 * 1005^1005)
  (hE : E = 1005^1005)
  (hF : F = 1005^1004) :
  (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l3080_308074


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3080_308019

def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3080_308019


namespace NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l3080_308006

/-- The decimal expansion 0.363636... (infinitely repeating 36) is equal to 4/11 -/
theorem repeating_decimal_36_equals_4_11 : ∃ (x : ℚ), x = 4/11 ∧ x = ∑' n, 36 / (100 ^ (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l3080_308006


namespace NUMINAMATH_CALUDE_jelly_bean_matching_probability_l3080_308031

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe_jelly_beans : JellyBeans :=
  { green := 2, red := 1, blue := 1, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob_jelly_beans : JellyBeans :=
  { green := 3, red := 2, blue := 1, yellow := 2 }

/-- Calculates the probability of both people showing the same color -/
def matching_probability (jb1 jb2 : JellyBeans) : ℚ :=
  let total1 := jb1.total
  let total2 := jb2.total
  (jb1.green * jb2.green + jb1.red * jb2.red + jb1.blue * jb2.blue) / (total1 * total2)

theorem jelly_bean_matching_probability :
  matching_probability abe_jelly_beans bob_jelly_beans = 9 / 32 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_matching_probability_l3080_308031


namespace NUMINAMATH_CALUDE_reflection_symmetry_l3080_308000

/-- Represents an L-like shape with two segments --/
structure LShape :=
  (top_segment : ℝ)
  (bottom_segment : ℝ)

/-- Reflects an L-shape over a horizontal line --/
def reflect (shape : LShape) : LShape :=
  { top_segment := shape.bottom_segment,
    bottom_segment := shape.top_segment }

/-- Checks if two L-shapes are equal --/
def is_equal (shape1 shape2 : LShape) : Prop :=
  shape1.top_segment = shape2.top_segment ∧ shape1.bottom_segment = shape2.bottom_segment

theorem reflection_symmetry (original : LShape) :
  original.top_segment > original.bottom_segment →
  is_equal (reflect original) { top_segment := original.bottom_segment, bottom_segment := original.top_segment } :=
by
  sorry

#check reflection_symmetry

end NUMINAMATH_CALUDE_reflection_symmetry_l3080_308000


namespace NUMINAMATH_CALUDE_custom_mult_solution_l3080_308046

/-- Custom multiplication operation for integers -/
def customMult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if 21b = 160 under the custom multiplication, then b = 9 -/
theorem custom_mult_solution :
  ∀ b : ℤ, customMult 21 b = 160 → b = 9 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l3080_308046


namespace NUMINAMATH_CALUDE_bus_trip_distance_l3080_308072

/-- Given a bus trip with specific conditions, prove that the distance traveled is 210 miles. -/
theorem bus_trip_distance (actual_speed : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) 
  (h1 : actual_speed = 30)
  (h2 : speed_increase = 5)
  (h3 : time_reduction = 1)
  (h4 : ∀ (distance : ℝ), distance / actual_speed = distance / (actual_speed + speed_increase) + time_reduction) :
  ∃ (distance : ℝ), distance = 210 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l3080_308072


namespace NUMINAMATH_CALUDE_shannon_bought_no_gum_l3080_308051

/-- Represents the purchase made by Shannon -/
structure Purchase where
  yogurt_pints : ℕ
  gum_packs : ℕ
  shrimp_trays : ℕ
  yogurt_price : ℚ
  shrimp_price : ℚ
  total_cost : ℚ

/-- The conditions of Shannon's purchase -/
def shannon_purchase : Purchase where
  yogurt_pints := 5
  gum_packs := 0  -- We'll prove this
  shrimp_trays := 5
  yogurt_price := 6  -- Derived from the total cost
  shrimp_price := 5
  total_cost := 55

/-- The price of gum is half the price of yogurt -/
def gum_price (p : Purchase) : ℚ := p.yogurt_price / 2

/-- The total cost of the purchase -/
def total_cost (p : Purchase) : ℚ :=
  p.yogurt_pints * p.yogurt_price +
  p.gum_packs * (gum_price p) +
  p.shrimp_trays * p.shrimp_price

/-- Theorem stating that Shannon bought 0 packs of gum -/
theorem shannon_bought_no_gum :
  shannon_purchase.gum_packs = 0 ∧
  total_cost shannon_purchase = shannon_purchase.total_cost := by
  sorry


end NUMINAMATH_CALUDE_shannon_bought_no_gum_l3080_308051


namespace NUMINAMATH_CALUDE_sum_first_three_terms_l3080_308027

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_three_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth : a 5 = 7)
  (h_sixth : a 6 = 12)
  (h_seventh : a 7 = 17) :
  a 1 + a 2 + a 3 = -24 :=
sorry

end NUMINAMATH_CALUDE_sum_first_three_terms_l3080_308027


namespace NUMINAMATH_CALUDE_shadow_problem_l3080_308053

/-- Given a cube with edge length 2 cm and a light source y cm above one of its upper vertices
    casting a shadow of 324 sq cm (excluding the area beneath the cube),
    prove that the largest integer less than or equal to 500y is 8000. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ y > 0 ∧ 
  (((18 : ℝ)^2 - 2^2) = 324) ∧
  ((y / 2) = ((18 : ℝ) - 2) / 2) →
  ⌊500 * y⌋ = 8000 := by sorry

end NUMINAMATH_CALUDE_shadow_problem_l3080_308053


namespace NUMINAMATH_CALUDE_total_area_parallelogram_triangle_l3080_308035

/-- The total area of a shape consisting of a parallelogram and an adjacent right triangle -/
theorem total_area_parallelogram_triangle (angle : Real) (side1 side2 : Real) (leg : Real) : 
  angle = 150 * π / 180 →
  side1 = 10 →
  side2 = 24 →
  leg = 10 →
  (side1 * side2 * Real.sin angle) / 2 + (side2 * leg) / 2 = 170 := by sorry

end NUMINAMATH_CALUDE_total_area_parallelogram_triangle_l3080_308035


namespace NUMINAMATH_CALUDE_solve_m_n_l3080_308026

def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m*x + n = 0}

theorem solve_m_n :
  ∃ (m n : ℝ),
    (A ∪ B m n = A) ∧
    (A ∩ B m n = {5}) ∧
    m = -10 ∧
    n = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_m_n_l3080_308026


namespace NUMINAMATH_CALUDE_beach_trip_time_l3080_308017

theorem beach_trip_time :
  let drive_time_one_way : ℝ := 2
  let total_drive_time : ℝ := 2 * drive_time_one_way
  let beach_time : ℝ := 2.5 * total_drive_time
  let total_trip_time : ℝ := total_drive_time + beach_time
  total_trip_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_time_l3080_308017


namespace NUMINAMATH_CALUDE_square_and_circle_measurements_l3080_308033

/-- Given a square with side length 70√2 cm and a circle with diameter equal to the square's diagonal,
    prove the square's diagonal length and the circle's circumference. -/
theorem square_and_circle_measurements :
  let square_side : ℝ := 70 * Real.sqrt 2
  let square_diagonal : ℝ := square_side * Real.sqrt 2
  let circle_diameter : ℝ := square_diagonal
  let circle_circumference : ℝ := π * circle_diameter
  (square_diagonal = 140) ∧ (circle_circumference = 140 * π) := by sorry

end NUMINAMATH_CALUDE_square_and_circle_measurements_l3080_308033


namespace NUMINAMATH_CALUDE_pages_read_first_day_l3080_308087

theorem pages_read_first_day (total_pages : ℕ) (days : ℕ) (first_day_pages : ℕ) : 
  total_pages = 130 →
  days = 7 →
  total_pages = first_day_pages + (days - 1) * (2 * first_day_pages) →
  first_day_pages = 10 :=
by sorry

end NUMINAMATH_CALUDE_pages_read_first_day_l3080_308087


namespace NUMINAMATH_CALUDE_max_a_value_l3080_308004

theorem max_a_value (a k x₁ x₂ : ℝ) : 
  (∀ k ∈ Set.Icc 0 2, 
   ∀ x₁ ∈ Set.Icc k (k + a), 
   ∀ x₂ ∈ Set.Icc (k + 2*a) (k + 4*a), 
   (x₁^2 - (k^2 - 5*a*k + 3)*x₁ + 7) ≥ (x₂^2 - (k^2 - 5*a*k + 3)*x₂ + 7)) →
  a ≤ (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3080_308004


namespace NUMINAMATH_CALUDE_percentage_calculation_l3080_308001

theorem percentage_calculation : 
  (2 * (1/4 * (4/100))) + (3 * (15/100)) - (1/2 * (10/100)) = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3080_308001


namespace NUMINAMATH_CALUDE_monotone_decreasing_function_positivity_l3080_308090

theorem monotone_decreasing_function_positivity 
  (f : ℝ → ℝ) 
  (h_monotone : ∀ x y, x < y → f x > f y) 
  (h_inequality : ∀ x, f x / (deriv f x) + x < 1) : 
  ∀ x, f x > 0 := by
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_function_positivity_l3080_308090


namespace NUMINAMATH_CALUDE_inverse_proportional_point_l3080_308015

theorem inverse_proportional_point :
  let f : ℝ → ℝ := λ x => 6 / x
  f (-2) = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportional_point_l3080_308015


namespace NUMINAMATH_CALUDE_elephant_distribution_l3080_308092

theorem elephant_distribution (union_members non_union_members : ℕ) 
  (h1 : union_members = 28)
  (h2 : non_union_members = 37) :
  let total_elephants := 2072
  let elephants_per_union := total_elephants / union_members
  let elephants_per_non_union := total_elephants / non_union_members
  (elephants_per_union * union_members = elephants_per_non_union * non_union_members) ∧
  (elephants_per_union ≥ 1) ∧
  (elephants_per_non_union ≥ 1) ∧
  (∀ n : ℕ, n > total_elephants → 
    ¬(n / union_members * union_members = n / non_union_members * non_union_members ∧
      n / union_members ≥ 1 ∧
      n / non_union_members ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_elephant_distribution_l3080_308092


namespace NUMINAMATH_CALUDE_complex_square_sum_zero_l3080_308059

theorem complex_square_sum_zero (i : ℂ) (h : i^2 = -1) : 
  (1 + i)^2 + (1 - i)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_zero_l3080_308059


namespace NUMINAMATH_CALUDE_train_speed_clicks_l3080_308058

theorem train_speed_clicks (x : ℝ) : x > 0 →
  let t := (2400 : ℝ) / 5280
  t ≠ 0.25 ∧ t ≠ 1 ∧ t ≠ 2 ∧ t ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_clicks_l3080_308058


namespace NUMINAMATH_CALUDE_coin_distribution_l3080_308098

theorem coin_distribution (a b c d e : ℚ) : 
  -- The amounts form an arithmetic sequence
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) →
  -- The total number of coins is 5
  a + b + c + d + e = 5 →
  -- The sum of first two equals the sum of last three
  a + b = c + d + e →
  -- B receives 7/6 coins
  b = 7/6 := by sorry

end NUMINAMATH_CALUDE_coin_distribution_l3080_308098


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_eight_l3080_308075

theorem sqrt_expression_equals_eight :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_eight_l3080_308075


namespace NUMINAMATH_CALUDE_gcd_7254_156_minus_10_l3080_308052

theorem gcd_7254_156_minus_10 : Nat.gcd 7254 156 - 10 = 68 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7254_156_minus_10_l3080_308052


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l3080_308085

/-- Represents the initial ratio of costs for raw material, labor, and overheads -/
def initial_ratio : Fin 3 → ℚ
  | 0 => 4
  | 1 => 3
  | 2 => 2

/-- Represents the percentage changes in costs for raw material, labor, and overheads -/
def cost_changes : Fin 3 → ℚ
  | 0 => 110 / 100  -- 10% increase
  | 1 => 108 / 100  -- 8% increase
  | 2 => 95 / 100   -- 5% decrease

/-- Theorem stating that the overall percentage increase in cost is 6% -/
theorem cost_increase_percentage : 
  let initial_total := (Finset.sum Finset.univ initial_ratio)
  let new_total := (Finset.sum Finset.univ (λ i => initial_ratio i * cost_changes i))
  (new_total - initial_total) / initial_total * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l3080_308085


namespace NUMINAMATH_CALUDE_hexagon_square_side_ratio_l3080_308061

theorem hexagon_square_side_ratio (s_h s_s : ℝ) 
  (h_positive : s_h > 0 ∧ s_s > 0)
  (h_perimeter : 6 * s_h = 4 * s_s) : 
  s_s / s_h = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_square_side_ratio_l3080_308061


namespace NUMINAMATH_CALUDE_focus_of_symmetric_parabola_l3080_308012

/-- The focus of a parabola symmetric to x^2 = 4y with respect to x + y = 0 -/
def symmetric_parabola_focus : ℝ × ℝ :=
  (-1, 0)

/-- The original parabola equation -/
def original_parabola (x y : ℝ) : Prop :=
  x^2 = 4*y

/-- The line of symmetry equation -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

theorem focus_of_symmetric_parabola :
  symmetric_parabola_focus = (-1, 0) :=
sorry

end NUMINAMATH_CALUDE_focus_of_symmetric_parabola_l3080_308012


namespace NUMINAMATH_CALUDE_perpendicular_bisector_complex_l3080_308060

/-- The set of points equidistant from two distinct complex numbers forms a perpendicular bisector -/
theorem perpendicular_bisector_complex (z₁ z₂ : ℂ) (hz : z₁ ≠ z₂) :
  {z : ℂ | Complex.abs (z - z₁) = Complex.abs (z - z₂)} =
  {z : ℂ | (z - (z₁ + z₂) / 2) • (z₁ - z₂) = 0} :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_complex_l3080_308060


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3080_308014

theorem sum_reciprocals_bound (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c) + 1 / (1 + d) > 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3080_308014


namespace NUMINAMATH_CALUDE_employee_devices_l3080_308021

theorem employee_devices (total : ℝ) (h_total : total > 0) : 
  let cell_phone := (2/3 : ℝ) * total
  let pager := (2/5 : ℝ) * total
  let neither := (1/3 : ℝ) * total
  let both := cell_phone + pager - (total - neither)
  both / total = 2/5 := by
sorry

end NUMINAMATH_CALUDE_employee_devices_l3080_308021


namespace NUMINAMATH_CALUDE_rectangle_validity_l3080_308056

/-- A rectangle is valid if its area is less than or equal to the square of a quarter of its perimeter. -/
theorem rectangle_validity (S l : ℝ) (h_pos : S > 0 ∧ l > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = S ∧ 2 * (x + y) = l) ↔ S ≤ (l / 4)^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_validity_l3080_308056


namespace NUMINAMATH_CALUDE_expanded_volume_of_problem_box_l3080_308030

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of space inside and within one unit of a box -/
def expandedVolume (b : Box) : ℝ := sorry

/-- The specific box in the problem -/
def problemBox : Box := ⟨2, 3, 4⟩

theorem expanded_volume_of_problem_box :
  expandedVolume problemBox = (228 + 31 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_expanded_volume_of_problem_box_l3080_308030


namespace NUMINAMATH_CALUDE_fraction_problem_l3080_308077

theorem fraction_problem (x y : ℚ) (h1 : x + y = 3/4) (h2 : x * y = 1/8) : 
  min x y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3080_308077


namespace NUMINAMATH_CALUDE_price_difference_l3080_308068

/-- The price difference problem -/
theorem price_difference (discount_price : ℝ) (discount_rate : ℝ) (increase_rate : ℝ) : 
  discount_price = 68 ∧ 
  discount_rate = 0.15 ∧ 
  increase_rate = 0.25 →
  ∃ (original_price final_price : ℝ),
    original_price * (1 - discount_rate) = discount_price ∧
    final_price = discount_price * (1 + increase_rate) ∧
    final_price - original_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l3080_308068


namespace NUMINAMATH_CALUDE_business_valuation_l3080_308041

def business_value (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) : ℕ :=
  (2 * sale_price : ℕ)

theorem business_valuation (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) 
  (h1 : total_ownership = 2/3)
  (h2 : sold_fraction = 3/4)
  (h3 : sale_price = 6500) :
  business_value total_ownership sold_fraction sale_price = 13000 := by
  sorry

end NUMINAMATH_CALUDE_business_valuation_l3080_308041


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3080_308069

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3080_308069


namespace NUMINAMATH_CALUDE_billy_cherries_l3080_308020

theorem billy_cherries (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 74 → remaining = 2 → eaten = initial - remaining → eaten = 72 := by
  sorry

end NUMINAMATH_CALUDE_billy_cherries_l3080_308020


namespace NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l3080_308080

theorem pythagorean_theorem_3_4_5 : 
  ∀ (a b c : ℝ), 
    a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_3_4_5_l3080_308080


namespace NUMINAMATH_CALUDE_lemon_bags_count_l3080_308057

/-- The maximum load of the truck in kilograms -/
def max_load : ℕ := 900

/-- The mass of one bag of lemons in kilograms -/
def bag_mass : ℕ := 8

/-- The remaining capacity of the truck in kilograms -/
def remaining_capacity : ℕ := 100

/-- The number of bags of lemons on the truck -/
def num_bags : ℕ := (max_load - remaining_capacity) / bag_mass

theorem lemon_bags_count : num_bags = 100 := by
  sorry

end NUMINAMATH_CALUDE_lemon_bags_count_l3080_308057


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l3080_308040

def m : ℕ := 2010^2 + 2^2010

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ) : (m^2 + 3^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l3080_308040


namespace NUMINAMATH_CALUDE_multiply_fractions_l3080_308096

theorem multiply_fractions : (2 * (1/3)) * (3 * (1/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l3080_308096


namespace NUMINAMATH_CALUDE_solution_product_l3080_308032

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 8) = r^2 - 20 * r + 75 →
  (s - 3) * (3 * s + 8) = s^2 - 20 * s + 75 →
  r ≠ s →
  (r + 4) * (s + 4) = -119/2 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l3080_308032


namespace NUMINAMATH_CALUDE_bees_second_day_l3080_308039

def bees_first_day : ℕ := 144
def multiplier : ℕ := 3

theorem bees_second_day : bees_first_day * multiplier = 432 := by
  sorry

end NUMINAMATH_CALUDE_bees_second_day_l3080_308039


namespace NUMINAMATH_CALUDE_banana_profit_calculation_l3080_308082

/-- Calculates the profit from selling bananas given the purchase and selling rates and the total quantity purchased. -/
theorem banana_profit_calculation 
  (purchase_rate_pounds : ℚ) 
  (purchase_rate_dollars : ℚ) 
  (sell_rate_pounds : ℚ) 
  (sell_rate_dollars : ℚ) 
  (total_pounds : ℚ) : 
  purchase_rate_pounds = 3 →
  purchase_rate_dollars = 1/2 →
  sell_rate_pounds = 4 →
  sell_rate_dollars = 1 →
  total_pounds = 72 →
  (sell_rate_dollars / sell_rate_pounds * total_pounds) - 
  (purchase_rate_dollars / purchase_rate_pounds * total_pounds) = 6 := by
sorry

end NUMINAMATH_CALUDE_banana_profit_calculation_l3080_308082


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l3080_308034

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt (Real.sqrt x)

-- State the theorem
theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt 16 = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l3080_308034


namespace NUMINAMATH_CALUDE_star_diamond_relation_l3080_308067

theorem star_diamond_relation (star diamond : ℤ) 
  (h : 514 - star = 600 - diamond) : 
  star < diamond ∧ diamond - star = 86 := by
  sorry

end NUMINAMATH_CALUDE_star_diamond_relation_l3080_308067


namespace NUMINAMATH_CALUDE_max_value_cubic_ratio_l3080_308078

theorem max_value_cubic_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^3 / (x^3 + y^3) ≤ 4 ∧
  (x + y)^3 / (x^3 + y^3) = 4 ↔ x = y :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_ratio_l3080_308078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3080_308016

/-- An arithmetic sequence with given first term and third term -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 8) :
  a 10 = 29 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3080_308016


namespace NUMINAMATH_CALUDE_quadratic_maximum_l3080_308063

theorem quadratic_maximum : 
  (∃ (p : ℝ), -3 * p^2 + 18 * p + 24 = 51) ∧ 
  (∀ (p : ℝ), -3 * p^2 + 18 * p + 24 ≤ 51) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l3080_308063


namespace NUMINAMATH_CALUDE_bernoulli_max_value_l3080_308095

/-- Represents a Bernoulli random variable with parameter p -/
def BernoulliRV (p : ℝ) : Type :=
  {ξ : ℝ // ξ = 0 ∨ ξ = 1}

/-- The expectation of a Bernoulli random variable -/
def expectation (p : ℝ) (ξ : BernoulliRV p) : ℝ := p

/-- The variance of a Bernoulli random variable -/
def variance (p : ℝ) (ξ : BernoulliRV p) : ℝ := p * (1 - p)

/-- The main theorem: maximum value of (4Var(ξ) - 1) / E[ξ] is 0 -/
theorem bernoulli_max_value (p : ℝ) (hp : 0 < p ∧ p < 1) (ξ : BernoulliRV p) :
  (∀ q, 0 < q ∧ q < 1 → (4 * variance q ξ - 1) / expectation q ξ ≤ 0) ∧
  (∃ q, 0 < q ∧ q < 1 ∧ (4 * variance q ξ - 1) / expectation q ξ = 0) :=
sorry

end NUMINAMATH_CALUDE_bernoulli_max_value_l3080_308095


namespace NUMINAMATH_CALUDE_sams_remaining_pennies_l3080_308045

/-- Given an initial number of pennies and a number of spent pennies,
    calculate the remaining number of pennies. -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that Sam's remaining pennies are 5 given the initial and spent amounts. -/
theorem sams_remaining_pennies :
  remaining_pennies 98 93 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_pennies_l3080_308045


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3080_308070

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/27
  let a₃ : ℚ := 56/243
  let r : ℚ := -16/27
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3080_308070


namespace NUMINAMATH_CALUDE_segment_transformation_midpoint_l3080_308097

/-- Given a segment with endpoints (3, -2) and (9, 6), when translated 4 units left and 2 units down,
    then rotated 90° counterclockwise about its midpoint, the resulting segment has a midpoint at (2, 0) -/
theorem segment_transformation_midpoint : 
  let s₁_start : ℝ × ℝ := (3, -2)
  let s₁_end : ℝ × ℝ := (9, 6)
  let translate : ℝ × ℝ := (-4, -2)
  let s₁_midpoint := ((s₁_start.1 + s₁_end.1) / 2, (s₁_start.2 + s₁_end.2) / 2)
  let s₂_midpoint := (s₁_midpoint.1 + translate.1, s₁_midpoint.2 + translate.2)
  s₂_midpoint = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_segment_transformation_midpoint_l3080_308097


namespace NUMINAMATH_CALUDE_marble_distribution_l3080_308043

theorem marble_distribution (a : ℕ) : 
  let angela := a
  let brian := 2 * a
  let caden := angela + brian
  let daryl := 2 * caden
  angela + brian + caden + daryl = 144 → a = 12 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l3080_308043


namespace NUMINAMATH_CALUDE_equation_solution_l3080_308003

theorem equation_solution : 
  ∀ x y z : ℕ, 2^x + 3^y + 7 = z! ↔ (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3080_308003


namespace NUMINAMATH_CALUDE_cake_problem_l3080_308086

/-- Proves that the initial number of cakes is 12, given the conditions of the problem. -/
theorem cake_problem (total : ℕ) (fallen : ℕ) (undamaged : ℕ) (destroyed : ℕ) 
  (h1 : fallen = total / 2)
  (h2 : undamaged = fallen / 2)
  (h3 : destroyed = 3)
  (h4 : fallen = undamaged + destroyed) :
  total = 12 := by
  sorry

end NUMINAMATH_CALUDE_cake_problem_l3080_308086


namespace NUMINAMATH_CALUDE_middle_term_coefficient_l3080_308023

/-- Given a binomial expansion (x^2 - 2/x)^n where the 5th term is constant,
    prove that the coefficient of the middle term is -160 -/
theorem middle_term_coefficient
  (x : ℝ) (n : ℕ)
  (h_constant : ∃ k : ℝ, (n.choose 4) * (x^2)^(n-4) * (-2/x)^4 = k) :
  ∃ m : ℕ, m = (n+1)/2 ∧ (n.choose (m-1)) * (x^2)^(m-1) * (-2/x)^(n-m+1) = -160 * x^(2*m-n-1) :=
sorry

end NUMINAMATH_CALUDE_middle_term_coefficient_l3080_308023


namespace NUMINAMATH_CALUDE_klinker_age_proof_l3080_308008

/-- Mr. Klinker's current age -/
def klinker_age : ℕ := 35

/-- Mr. Klinker's daughter's current age -/
def daughter_age : ℕ := 10

/-- Years into the future when the age relation holds -/
def years_future : ℕ := 15

theorem klinker_age_proof :
  klinker_age = 35 ∧
  daughter_age = 10 ∧
  klinker_age + years_future = 2 * (daughter_age + years_future) := by
  sorry

#check klinker_age_proof

end NUMINAMATH_CALUDE_klinker_age_proof_l3080_308008


namespace NUMINAMATH_CALUDE_melodys_dogs_eating_frequency_l3080_308066

/-- Proves that each dog eats twice a day given the conditions of Melody's dog food problem -/
theorem melodys_dogs_eating_frequency :
  let num_dogs : ℕ := 3
  let food_per_meal : ℚ := 1/2
  let initial_food : ℕ := 30
  let remaining_food : ℕ := 9
  let days_in_week : ℕ := 7
  
  let total_food_eaten : ℕ := initial_food - remaining_food
  let food_per_day : ℚ := (total_food_eaten : ℚ) / days_in_week
  let meals_per_day : ℚ := food_per_day / (num_dogs * food_per_meal)
  
  meals_per_day = 2 := by sorry

end NUMINAMATH_CALUDE_melodys_dogs_eating_frequency_l3080_308066


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l3080_308071

/-- 
Given a parabola with equation y^2 = 2px and its latus rectum with equation x = -2,
prove that p = 4.
-/
theorem parabola_latus_rectum (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (∀ y : ℝ, y^2 = 2*p*(-2)) → -- Equation of the latus rectum
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l3080_308071


namespace NUMINAMATH_CALUDE_expression_simplification_l3080_308037

theorem expression_simplification (b c x : ℝ) (hb : b ≠ 1) (hc : c ≠ 1) (hbc : b ≠ c) :
  (x + 1)^2 / ((1 - b) * (1 - c)) + (x + b)^2 / ((b - 1) * (b - c)) + (x + c)^2 / ((c - 1) * (c - b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3080_308037


namespace NUMINAMATH_CALUDE_power_sum_l3080_308009

theorem power_sum (a m n : ℝ) (hm : a^m = 3) (hn : a^n = 2) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l3080_308009


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l3080_308088

/-- Given a quadratic equation x^2 - 4x + 3 = 0 with roots x₁ and x₂, 
    the product of the roots x₁ * x₂ equals 3. -/
theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 3 = 0 → x₂^2 - 4*x₂ + 3 = 0 → x₁ * x₂ = 3 := by
  sorry


end NUMINAMATH_CALUDE_product_of_roots_quadratic_l3080_308088


namespace NUMINAMATH_CALUDE_intersection_M_N_l3080_308002

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

theorem intersection_M_N : M ∩ N = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3080_308002


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3080_308038

/-- A line passing through point (-2, 0) and parallel to 3x - y + 1 = 0 has equation y = 3x + 6 -/
theorem parallel_line_equation :
  let point : ℝ × ℝ := (-2, 0)
  let parallel_line (x y : ℝ) := 3 * x - y + 1 = 0
  let proposed_line (x y : ℝ) := y = 3 * x + 6
  (∀ x y, parallel_line x y ↔ y = 3 * x - 1) →
  (proposed_line point.1 point.2) ∧
  (∀ x₁ y₁ x₂ y₂, parallel_line x₁ y₁ → proposed_line x₂ y₂ →
    y₂ - y₁ = 3 * (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3080_308038


namespace NUMINAMATH_CALUDE_parabola_vertex_l3080_308050

/-- The vertex of the parabola y = -2(x-3)^2 - 4 is at (3, -4) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => -2 * (x - 3)^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 3 ∧ p.2 = -4 ∧ ∀ x : ℝ, f x ≤ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3080_308050


namespace NUMINAMATH_CALUDE_triangle_problem_l3080_308042

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > t.c ∧
  t.a * t.c * (1/3) = 2 ∧  -- Vector BA · Vector BC = 2 and cos B = 1/3
  t.b = 3

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.a = 3 ∧ t.c = 2 ∧ Real.cos (t.B - t.C) = 23/27 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3080_308042


namespace NUMINAMATH_CALUDE_morgan_hula_hoop_time_l3080_308099

/-- Given information about hula hooping times for Nancy, Casey, and Morgan,
    prove that Morgan can hula hoop for 21 minutes. -/
theorem morgan_hula_hoop_time :
  ∀ (nancy casey morgan : ℕ),
    nancy = 10 →
    casey = nancy - 3 →
    morgan = 3 * casey →
    morgan = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_morgan_hula_hoop_time_l3080_308099


namespace NUMINAMATH_CALUDE_landscape_length_is_120_l3080_308064

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  playgroundArea : ℝ
  playgroundRatio : ℝ

/-- The length of the landscape is 4 times its breadth -/
def Landscape.length (l : Landscape) : ℝ := 4 * l.breadth

/-- The total area of the landscape -/
def Landscape.totalArea (l : Landscape) : ℝ := l.length * l.breadth

/-- Theorem: Given a landscape with specific properties, its length is 120 meters -/
theorem landscape_length_is_120 (l : Landscape) 
    (h1 : l.playgroundArea = 1200)
    (h2 : l.playgroundRatio = 1/3)
    (h3 : l.playgroundArea = l.playgroundRatio * l.totalArea) : 
  l.length = 120 := by
  sorry

#check landscape_length_is_120

end NUMINAMATH_CALUDE_landscape_length_is_120_l3080_308064


namespace NUMINAMATH_CALUDE_machine_a_production_rate_l3080_308036

/-- The number of sprockets produced by both machines -/
def total_sprockets : ℕ := 660

/-- The difference in production time between Machine A and Machine G -/
def time_difference : ℕ := 10

/-- The production rate of Machine G relative to Machine A -/
def g_to_a_ratio : ℚ := 11/10

/-- The production rate of Machine A in sprockets per hour -/
def machine_a_rate : ℚ := 6

theorem machine_a_production_rate :
  ∃ (machine_g_rate : ℚ) (time_g : ℚ),
    machine_g_rate = g_to_a_ratio * machine_a_rate ∧
    time_g * machine_g_rate = total_sprockets ∧
    (time_g + time_difference) * machine_a_rate = total_sprockets :=
by sorry

end NUMINAMATH_CALUDE_machine_a_production_rate_l3080_308036


namespace NUMINAMATH_CALUDE_textbook_cost_l3080_308076

/-- Given a textbook sold by a bookstore, prove that the cost to the bookstore
    is $44 when the selling price is $55 and the profit is $11. -/
theorem textbook_cost (selling_price profit : ℕ) (h1 : selling_price = 55) (h2 : profit = 11) :
  selling_price - profit = 44 := by
  sorry

end NUMINAMATH_CALUDE_textbook_cost_l3080_308076


namespace NUMINAMATH_CALUDE_mark_soup_donation_l3080_308081

/-- The number of homeless shelters -/
def num_shelters : ℕ := 6

/-- The number of people served by each shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup bought per person -/
def cans_per_person : ℕ := 10

/-- The total number of cans of soup Mark donates -/
def total_cans : ℕ := num_shelters * people_per_shelter * cans_per_person

theorem mark_soup_donation : total_cans = 1800 := by
  sorry

end NUMINAMATH_CALUDE_mark_soup_donation_l3080_308081


namespace NUMINAMATH_CALUDE_composite_ratio_l3080_308024

def first_seven_composites : List Nat := [4, 6, 8, 9, 10, 12, 14]
def next_seven_composites : List Nat := [15, 16, 18, 20, 21, 22, 24]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_of_list first_seven_composites) / 
  (product_of_list next_seven_composites) = 1 / 176 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_l3080_308024


namespace NUMINAMATH_CALUDE_bicycles_in_garage_l3080_308047

/-- The number of bicycles in Connor's garage --/
def num_bicycles : ℕ := 20

/-- The number of cars in Connor's garage --/
def num_cars : ℕ := 10

/-- The number of motorcycles in Connor's garage --/
def num_motorcycles : ℕ := 5

/-- The total number of wheels in Connor's garage --/
def total_wheels : ℕ := 90

/-- The number of wheels on a bicycle --/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a car --/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a motorcycle --/
def wheels_per_motorcycle : ℕ := 2

theorem bicycles_in_garage :
  num_bicycles * wheels_per_bicycle +
  num_cars * wheels_per_car +
  num_motorcycles * wheels_per_motorcycle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_bicycles_in_garage_l3080_308047


namespace NUMINAMATH_CALUDE_widescreen_tv_horizontal_length_l3080_308089

theorem widescreen_tv_horizontal_length :
  ∀ (h w d : ℝ),
  h > 0 ∧ w > 0 ∧ d > 0 →
  w / h = 16 / 9 →
  h^2 + w^2 = d^2 →
  d = 40 →
  w = (640 * Real.sqrt 337) / 337 := by
sorry

end NUMINAMATH_CALUDE_widescreen_tv_horizontal_length_l3080_308089


namespace NUMINAMATH_CALUDE_percentage_of_sum_l3080_308010

theorem percentage_of_sum (x y : ℝ) (P : ℝ) :
  (0.6 * (x - y) = (P / 100) * (x + y)) →
  (y = (1 / 3) * x) →
  P = 45 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l3080_308010


namespace NUMINAMATH_CALUDE_blisters_on_rest_eq_80_l3080_308083

/-- Represents the number of blisters on one arm -/
def blisters_per_arm : ℕ := 60

/-- Represents the total number of blisters -/
def total_blisters : ℕ := 200

/-- Calculates the number of blisters on the rest of the body -/
def blisters_on_rest : ℕ := total_blisters - 2 * blisters_per_arm

theorem blisters_on_rest_eq_80 : blisters_on_rest = 80 := by
  sorry

end NUMINAMATH_CALUDE_blisters_on_rest_eq_80_l3080_308083


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_unique_a_for_nonnegative_f_l3080_308013

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (1 + x)

theorem tangent_line_at_x_1 (h : ℝ) :
  ∃ (m b : ℝ), ∀ x, (f 2 x - (f 2 1)) = m * (x - 1) + b ∧ 
  m * x + b = Real.log 2 - 1 := by sorry

theorem unique_a_for_nonnegative_f :
  ∃! a : ℝ, ∀ x : ℝ, x > -1 → f a x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_unique_a_for_nonnegative_f_l3080_308013


namespace NUMINAMATH_CALUDE_distribute_8_balls_4_boxes_l3080_308093

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 139 ways to distribute 8 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_8_balls_4_boxes : distribute_balls 8 4 = 139 := by sorry

end NUMINAMATH_CALUDE_distribute_8_balls_4_boxes_l3080_308093


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3080_308054

/-- Proves that in a rhombus with an area of 432 sq m and one diagonal of 36 m, 
    the length of the other diagonal is 24 m. -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 432)
  (h_d1 : d1 = 36)
  (h_rhombus : area = (d1 * d2) / 2) : 
  d2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3080_308054


namespace NUMINAMATH_CALUDE_five_from_eight_l3080_308022

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem five_from_eight : choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_five_from_eight_l3080_308022


namespace NUMINAMATH_CALUDE_hockey_league_season_games_l3080_308011

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 25 teams, where each team plays every other team 12 times,
    the total number of games played in the season is 3600. -/
theorem hockey_league_season_games :
  hockey_league_games 25 12 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_season_games_l3080_308011


namespace NUMINAMATH_CALUDE_elective_course_schemes_l3080_308029

theorem elective_course_schemes (n : ℕ) (k : ℕ) : n = 4 ∧ k = 2 → Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_elective_course_schemes_l3080_308029


namespace NUMINAMATH_CALUDE_rotate_vector_2_3_l3080_308073

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Rotates a 2D vector 90 degrees clockwise -/
def rotate90Clockwise (v : Vector2D) : Vector2D :=
  { x := v.y, y := -v.x }

/-- The theorem stating that rotating (2, 3) by 90 degrees clockwise results in (3, -2) -/
theorem rotate_vector_2_3 :
  rotate90Clockwise { x := 2, y := 3 } = { x := 3, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_rotate_vector_2_3_l3080_308073


namespace NUMINAMATH_CALUDE_expression_simplification_l3080_308044

theorem expression_simplification (x : ℝ) : 
  ((7 * x - 3) + 3 * x * 2) * 2 + (5 + 2 * 2) * (4 * x + 6) = 62 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3080_308044


namespace NUMINAMATH_CALUDE_number_equation_solution_l3080_308025

theorem number_equation_solution :
  ∀ B : ℝ, (4 * B + 4 = 33) → B = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3080_308025


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3080_308049

/-- Given that x is inversely proportional to y, this function represents their relationship -/
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_ratio
  (x₁ x₂ y₁ y₂ : ℝ)
  (hx₁ : x₁ ≠ 0)
  (hx₂ : x₂ ≠ 0)
  (hy₁ : y₁ ≠ 0)
  (hy₂ : y₂ ≠ 0)
  (hxy₁ : inverse_proportion x₁ y₁)
  (hxy₂ : inverse_proportion x₂ y₂)
  (hx_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3080_308049


namespace NUMINAMATH_CALUDE_factor_expression_l3080_308028

theorem factor_expression : ∀ x : ℝ, 75 * x + 45 = 15 * (5 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3080_308028


namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l3080_308079

-- Define a function to represent the concept of simplicity for square roots
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (x = Real.sqrt y) → ¬∃ z : ℝ, z ≠ y ∧ Real.sqrt z = Real.sqrt y

-- State the theorem
theorem sqrt_3_simplest :
  is_simplest_sqrt (Real.sqrt 3) ∧
  ¬is_simplest_sqrt (Real.sqrt (a^2)) ∧
  ¬is_simplest_sqrt (Real.sqrt 0.3) ∧
  ¬is_simplest_sqrt (Real.sqrt 27) :=
sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l3080_308079


namespace NUMINAMATH_CALUDE_apples_per_crate_value_l3080_308091

/-- The number of apples in each crate -/
def apples_per_crate : ℕ := sorry

/-- The total number of crates -/
def total_crates : ℕ := 12

/-- The number of rotten apples -/
def rotten_apples : ℕ := 160

/-- The number of boxes filled with good apples -/
def filled_boxes : ℕ := 100

/-- The number of apples in each box -/
def apples_per_box : ℕ := 20

theorem apples_per_crate_value : apples_per_crate = 180 := by sorry

end NUMINAMATH_CALUDE_apples_per_crate_value_l3080_308091


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3080_308005

theorem diophantine_equation_solutions :
  ∀ a b : ℕ, 3 * 2^a + 1 = b^2 ↔ (a = 0 ∧ b = 2) ∨ (a = 3 ∧ b = 5) ∨ (a = 4 ∧ b = 7) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3080_308005


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l3080_308055

/-- 
Given three angles that meet at a point in a plane, 
if two of the angles are 145° and 95°, 
then the third angle is 120°.
-/
theorem angle_sum_at_point (a b c : ℝ) : 
  a + b + c = 360 → a = 145 → b = 95 → c = 120 := by sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l3080_308055


namespace NUMINAMATH_CALUDE_expression_factorization_l3080_308094

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x^2 - 15) - (-3 * x^3 + 6 * x^2 - 3) = 3 * (5 * x^3 + 13 * x^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3080_308094
