import Mathlib

namespace NUMINAMATH_GPT_tank_fraction_full_l320_32011

theorem tank_fraction_full 
  (initial_fraction : ℚ)
  (full_capacity : ℚ)
  (added_water : ℚ)
  (initial_fraction_eq : initial_fraction = 3/4)
  (full_capacity_eq : full_capacity = 40)
  (added_water_eq : added_water = 5) :
  ((initial_fraction * full_capacity + added_water) / full_capacity) = 7/8 :=
by 
  sorry

end NUMINAMATH_GPT_tank_fraction_full_l320_32011


namespace NUMINAMATH_GPT_min_number_of_bags_l320_32069

theorem min_number_of_bags (a b : ℕ) : 
  ∃ K : ℕ, K = a + b - Nat.gcd a b :=
by
  sorry

end NUMINAMATH_GPT_min_number_of_bags_l320_32069


namespace NUMINAMATH_GPT_problem_l320_32076

def f (a : ℕ) : ℕ := a + 3
def F (a b : ℕ) : ℕ := b^2 + a

theorem problem : F 4 (f 5) = 68 := by sorry

end NUMINAMATH_GPT_problem_l320_32076


namespace NUMINAMATH_GPT_product_calculation_l320_32093

theorem product_calculation :
  1500 * 2023 * 0.5023 * 50 = 306903675 :=
sorry

end NUMINAMATH_GPT_product_calculation_l320_32093


namespace NUMINAMATH_GPT_compute_expression_l320_32020

theorem compute_expression (x : ℝ) (h : x = 3) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 :=
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l320_32020


namespace NUMINAMATH_GPT_pear_juice_processed_l320_32094

theorem pear_juice_processed
  (total_pears : ℝ)
  (export_percentage : ℝ)
  (juice_percentage_of_remainder : ℝ) :
  total_pears = 8.5 →
  export_percentage = 0.30 →
  juice_percentage_of_remainder = 0.60 →
  ((total_pears * (1 - export_percentage)) * juice_percentage_of_remainder) = 3.6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pear_juice_processed_l320_32094


namespace NUMINAMATH_GPT_slope_of_parallel_lines_l320_32049

theorem slope_of_parallel_lines (m : ℝ)
  (y1 y2 y3 : ℝ)
  (h1 : y1 = 2) 
  (h2 : y2 = 3) 
  (h3 : y3 = 4)
  (sum_of_x_intercepts : (-2 / m) + (-3 / m) + (-4 / m) = 36) :
  m = -1 / 4 := by
  sorry

end NUMINAMATH_GPT_slope_of_parallel_lines_l320_32049


namespace NUMINAMATH_GPT_percentage_liked_B_l320_32042

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end NUMINAMATH_GPT_percentage_liked_B_l320_32042


namespace NUMINAMATH_GPT_base3_to_base10_l320_32017

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end NUMINAMATH_GPT_base3_to_base10_l320_32017


namespace NUMINAMATH_GPT_range_of_x_for_positive_y_l320_32038

theorem range_of_x_for_positive_y (x : ℝ) : 
  (-1 < x ∧ x < 3) ↔ (-x^2 + 2*x + 3 > 0) :=
sorry

end NUMINAMATH_GPT_range_of_x_for_positive_y_l320_32038


namespace NUMINAMATH_GPT_axis_of_symmetry_compare_m_n_range_t_max_t_l320_32030

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_compare_m_n_range_t_max_t_l320_32030


namespace NUMINAMATH_GPT_total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l320_32051

-- Definitions based on conditions
def standard_weight : ℝ := 25
def weight_diffs : List ℝ := [-3, -2, -2, -2, -2, -1.5, -1.5, 0, 0, 0, 1, 1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
def price_per_kg : ℝ := 10.6

-- Problem 1
theorem total_over_or_underweight_is_8kg :
  (weight_diffs.sum = 8) := 
  sorry

-- Problem 2
theorem total_selling_price_is_5384_point_8_yuan :
  (20 * standard_weight + 8) * price_per_kg = 5384.8 :=
  sorry

end NUMINAMATH_GPT_total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l320_32051


namespace NUMINAMATH_GPT_angle_sum_eq_180_l320_32036

theorem angle_sum_eq_180 (A B C D E F G : ℝ) 
  (h1 : A + B + C + D + E + F = 360) : 
  A + B + C + D + E + F + G = 180 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_eq_180_l320_32036


namespace NUMINAMATH_GPT_total_widgets_sold_15_days_l320_32095

def widgets_sold (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * n

theorem total_widgets_sold_15_days :
  (Finset.range 15).sum widgets_sold = 359 :=
by
  sorry

end NUMINAMATH_GPT_total_widgets_sold_15_days_l320_32095


namespace NUMINAMATH_GPT_correct_system_of_equations_l320_32070

theorem correct_system_of_equations (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  (∃ x y, (x / 3 = y - 2) ∧ ((x - 9) / 2 = y)) :=
by
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l320_32070


namespace NUMINAMATH_GPT_circle_radius_eq_l320_32031

theorem circle_radius_eq (r : ℝ) (AB : ℝ) (BC : ℝ) (hAB : AB = 10) (hBC : BC = 12) : r = 25 / 4 := by
  sorry

end NUMINAMATH_GPT_circle_radius_eq_l320_32031


namespace NUMINAMATH_GPT_inequality_sum_l320_32091

variables {a b c : ℝ}

theorem inequality_sum (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end NUMINAMATH_GPT_inequality_sum_l320_32091


namespace NUMINAMATH_GPT_ratio_diff_l320_32056

theorem ratio_diff (x : ℕ) (h1 : 7 * x = 56) : 56 - 3 * x = 32 :=
by
  sorry

end NUMINAMATH_GPT_ratio_diff_l320_32056


namespace NUMINAMATH_GPT_rectangle_area_l320_32009

theorem rectangle_area (L B r s : ℝ) (h1 : L = 5 * r)
                       (h2 : r = s)
                       (h3 : s^2 = 16)
                       (h4 : B = 11) :
  (L * B = 220) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l320_32009


namespace NUMINAMATH_GPT_beverage_price_function_l320_32010

theorem beverage_price_function (box_price : ℕ) (bottles_per_box : ℕ) (bottles_purchased : ℕ) (y : ℕ) :
  box_price = 55 →
  bottles_per_box = 6 →
  y = (55 * bottles_purchased) / 6 := 
sorry

end NUMINAMATH_GPT_beverage_price_function_l320_32010


namespace NUMINAMATH_GPT_island_length_l320_32082

/-- Proof problem: Given an island in the Indian Ocean with a width of 4 miles and a perimeter of 22 miles. 
    Assume the island is rectangular in shape. Prove that the length of the island is 7 miles. -/
theorem island_length
  (width length : ℝ) 
  (h_width : width = 4)
  (h_perimeter : 2 * (length + width) = 22) : 
  length = 7 :=
sorry

end NUMINAMATH_GPT_island_length_l320_32082


namespace NUMINAMATH_GPT_sum_of_fractions_l320_32005

theorem sum_of_fractions :
  (3 / 9) + (6 / 12) = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l320_32005


namespace NUMINAMATH_GPT_pinocchio_cannot_pay_exactly_l320_32092

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end NUMINAMATH_GPT_pinocchio_cannot_pay_exactly_l320_32092


namespace NUMINAMATH_GPT_fixed_numbers_in_diagram_has_six_solutions_l320_32096

-- Define the problem setup and constraints
def is_divisor (m n : ℕ) : Prop := ∃ k, n = k * m

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Formulating the main proof statement
theorem fixed_numbers_in_diagram_has_six_solutions : 
  ∃ (a b c k : ℕ),
    (14 * 4 * a = 14 * 6 * c) ∧
    (4 * a = 6 * c) ∧
    (2 * a = 3 * c) ∧
    (∃ k, c = 2 * k ∧ a = 3 * k) ∧
    (14 * 4 * 3 * k = 3 * k * b * 2 * k) ∧
    (∃ k, 56 * k = 6 * k^2 * b) ∧
    (b = 28 / k) ∧
    ((is_divisor k 28) ∧
     (k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 7 ∨ k = 14 ∨ k = 28)) ∧
    (6 = 6) := sorry

end NUMINAMATH_GPT_fixed_numbers_in_diagram_has_six_solutions_l320_32096


namespace NUMINAMATH_GPT_smallest_integer_condition_l320_32067

theorem smallest_integer_condition {A : ℕ} (h1 : A > 1) 
  (h2 : ∃ k : ℕ, A = 5 * k / 3 + 2 / 3)
  (h3 : ∃ m : ℕ, A = 7 * m / 5 + 2 / 5)
  (h4 : ∃ n : ℕ, A = 9 * n / 7 + 2 / 7)
  (h5 : ∃ p : ℕ, A = 11 * p / 9 + 2 / 9) : 
  A = 316 := 
sorry

end NUMINAMATH_GPT_smallest_integer_condition_l320_32067


namespace NUMINAMATH_GPT_brick_wall_problem_l320_32012

theorem brick_wall_problem : 
  ∀ (B1 B2 B3 B4 B5 : ℕ) (d : ℕ),
  B1 = 38 →
  B1 + B2 + B3 + B4 + B5 = 200 →
  B2 = B1 - d →
  B3 = B1 - 2 * d →
  B4 = B1 - 3 * d →
  B5 = B1 - 4 * d →
  d = 1 :=
by
  intros B1 B2 B3 B4 B5 d h1 h2 h3 h4 h5 h6
  rw [h1] at h2
  sorry

end NUMINAMATH_GPT_brick_wall_problem_l320_32012


namespace NUMINAMATH_GPT_set_intersection_example_l320_32083

theorem set_intersection_example :
  let M := {x : ℝ | -1 < x ∧ x < 1}
  let N := {x : ℝ | 0 ≤ x}
  {x : ℝ | -1 < x ∧ x < 1} ∩ {x : ℝ | 0 ≤ x} = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_example_l320_32083


namespace NUMINAMATH_GPT_quotient_remainder_base5_l320_32075

theorem quotient_remainder_base5 (n m : ℕ) 
    (hn : n = 3 * 5^3 + 2 * 5^2 + 3 * 5^1 + 2)
    (hm : m = 2 * 5^1 + 1) :
    n / m = 40 ∧ n % m = 2 :=
by
  sorry

end NUMINAMATH_GPT_quotient_remainder_base5_l320_32075


namespace NUMINAMATH_GPT_lines_intersect_l320_32024

-- Define the parameterizations of the two lines
def line1 (t : ℚ) : ℚ × ℚ := ⟨2 + 3 * t, 3 - 4 * t⟩
def line2 (u : ℚ) : ℚ × ℚ := ⟨4 + 5 * u, 1 + 3 * u⟩

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = line2 u ∧ line1 t = ⟨26 / 11, 19 / 11⟩ :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_l320_32024


namespace NUMINAMATH_GPT_cabin_price_correct_l320_32016

noncomputable def cabin_price 
  (cash : ℤ)
  (cypress_trees : ℤ) (pine_trees : ℤ) (maple_trees : ℤ)
  (price_cypress : ℤ) (price_pine : ℤ) (price_maple : ℤ)
  (remaining_cash : ℤ)
  (expected_price : ℤ) : Prop :=
   cash + (cypress_trees * price_cypress + pine_trees * price_pine + maple_trees * price_maple) - remaining_cash = expected_price

theorem cabin_price_correct :
  cabin_price 150 20 600 24 100 200 300 350 130000 :=
by
  sorry

end NUMINAMATH_GPT_cabin_price_correct_l320_32016


namespace NUMINAMATH_GPT_treaty_signed_on_thursday_l320_32043

def initial_day : ℕ := 0  -- 0 representing Monday, assuming a week cycle from 0 (Monday) to 6 (Sunday)
def days_in_week : ℕ := 7

def treaty_day (n : ℕ) : ℕ :=
(n + initial_day) % days_in_week

theorem treaty_signed_on_thursday :
  treaty_day 1000 = 4 :=  -- 4 representing Thursday
by
  sorry

end NUMINAMATH_GPT_treaty_signed_on_thursday_l320_32043


namespace NUMINAMATH_GPT_perpendicular_lines_have_given_slope_l320_32003

theorem perpendicular_lines_have_given_slope (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_have_given_slope_l320_32003


namespace NUMINAMATH_GPT_number_of_balls_l320_32057

noncomputable def frequency_of_yellow (n : ℕ) : ℚ := 9 / n

theorem number_of_balls (n : ℕ) (h1 : frequency_of_yellow n = 0.30) : n = 30 :=
by sorry

end NUMINAMATH_GPT_number_of_balls_l320_32057


namespace NUMINAMATH_GPT_non_isosceles_count_l320_32072

def n : ℕ := 20

def total_triangles : ℕ := Nat.choose n 3

def isosceles_triangles_per_vertex : ℕ := 9

def total_isosceles_triangles : ℕ := n * isosceles_triangles_per_vertex

def non_isosceles_triangles : ℕ := total_triangles - total_isosceles_triangles

theorem non_isosceles_count :
  non_isosceles_triangles = 960 := 
  by 
    -- proof details would go here
    sorry

end NUMINAMATH_GPT_non_isosceles_count_l320_32072


namespace NUMINAMATH_GPT_initial_percentage_of_managers_l320_32055

theorem initial_percentage_of_managers (P : ℕ) (h : 0 ≤ P ∧ P ≤ 100)
  (total_employees initial_managers : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : initial_managers = P * total_employees / 100) 
  (remaining_employees remaining_managers : ℕ)
  (h3 : remaining_employees = total_employees - 250)
  (h4 : remaining_managers = initial_managers - 250)
  (h5 : remaining_managers * 100 = 98 * remaining_employees) :
  P = 99 := 
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_managers_l320_32055


namespace NUMINAMATH_GPT_area_within_fence_is_328_l320_32014

-- Define the dimensions of the fenced area
def main_rectangle_length : ℝ := 20
def main_rectangle_width : ℝ := 18

-- Define the dimensions of the square cutouts
def cutout_length : ℝ := 4
def cutout_width : ℝ := 4

-- Calculate the areas
def main_rectangle_area : ℝ := main_rectangle_length * main_rectangle_width
def cutout_area : ℝ := cutout_length * cutout_width

-- Define the number of cutouts
def number_of_cutouts : ℝ := 2

-- Calculate the final area within the fence
def area_within_fence : ℝ := main_rectangle_area - number_of_cutouts * cutout_area

theorem area_within_fence_is_328 : area_within_fence = 328 := by
  -- This is a place holder for the proof, replace it with the actual proof
  sorry

end NUMINAMATH_GPT_area_within_fence_is_328_l320_32014


namespace NUMINAMATH_GPT_functions_are_computable_l320_32000

def f1 : ℕ → ℕ := λ n => 0
def f2 : ℕ → ℕ := λ n => n + 1
def f3 : ℕ → ℕ := λ n => max 0 (n - 1)
def f4 : ℕ → ℕ := λ n => n % 2
def f5 : ℕ → ℕ := λ n => n * 2
def f6 : ℕ × ℕ → ℕ := λ (m, n) => if m ≤ n then 1 else 0

theorem functions_are_computable :
  (Computable f1) ∧
  (Computable f2) ∧
  (Computable f3) ∧
  (Computable f4) ∧
  (Computable f5) ∧
  (Computable f6) := by
  sorry

end NUMINAMATH_GPT_functions_are_computable_l320_32000


namespace NUMINAMATH_GPT_jugs_needed_to_provide_water_for_students_l320_32054

def jug_capacity : ℕ := 40
def students : ℕ := 200
def cups_per_student : ℕ := 10

def total_cups_needed := students * cups_per_student

theorem jugs_needed_to_provide_water_for_students :
  total_cups_needed / jug_capacity = 50 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jugs_needed_to_provide_water_for_students_l320_32054


namespace NUMINAMATH_GPT_greatest_product_l320_32050

theorem greatest_product (x : ℤ) (h : x + (1998 - x) = 1998) : 
  x * (1998 - x) ≤ 998001 :=
  sorry

end NUMINAMATH_GPT_greatest_product_l320_32050


namespace NUMINAMATH_GPT_k3_to_fourth_equals_81_l320_32034

theorem k3_to_fourth_equals_81
  (h k : ℝ → ℝ)
  (h_cond : ∀ x, x ≥ 1 → h (k x) = x^3)
  (k_cond : ∀ x, x ≥ 1 → k (h x) = x^4)
  (k_81 : k 81 = 81) :
  k 3 ^ 4 = 81 :=
sorry

end NUMINAMATH_GPT_k3_to_fourth_equals_81_l320_32034


namespace NUMINAMATH_GPT_exists_student_not_wet_l320_32006

theorem exists_student_not_wet (n : ℕ) (students : Fin (2 * n + 1) → ℝ) (distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → students i ≠ students j) : 
  ∃ i : Fin (2 * n + 1), ∀ j : Fin (2 * n + 1), (j ≠ i → students j ≠ students i) :=
  sorry

end NUMINAMATH_GPT_exists_student_not_wet_l320_32006


namespace NUMINAMATH_GPT_problem1_problem2_l320_32060

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Statement 1: If a = 1 and p ∧ q is true, then the range of x is 2 < x < 3
theorem problem1 (x : ℝ) (h : 1 = 1) (hpq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
sorry

-- Statement 2: If ¬p is a sufficient but not necessary condition for ¬q, then the range of a is 1 < a ≤ 2
theorem problem2 (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 2) (h3 : ¬ (∃ x, p x a) → ¬ (∃ x, q x)) : 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l320_32060


namespace NUMINAMATH_GPT_largest_inscribed_triangle_area_l320_32047

theorem largest_inscribed_triangle_area 
  (radius : ℝ) 
  (diameter : ℝ)
  (base : ℝ)
  (height : ℝ) 
  (area : ℝ)
  (h1 : radius = 10)
  (h2 : diameter = 2 * radius)
  (h3 : base = diameter)
  (h4 : height = radius) 
  (h5 : area = (1/2) * base * height) : 
  area  = 100 :=
by 
  have h_area := (1/2) * 20 * 10
  sorry

end NUMINAMATH_GPT_largest_inscribed_triangle_area_l320_32047


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l320_32025

theorem problem1_solution (x : ℝ) : x^2 - x - 6 > 0 ↔ x < -2 ∨ x > 3 := sorry

theorem problem2_solution (x : ℝ) : -2*x^2 + x + 1 < 0 ↔ x < -1/2 ∨ x > 1 := sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l320_32025


namespace NUMINAMATH_GPT_inequality_ab_equals_bc_l320_32029

-- Define the given conditions and state the theorem as per the proof problem
theorem inequality_ab_equals_bc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^b * b^c * c^a ≤ a^a * b^b * c^c :=
by
  sorry

end NUMINAMATH_GPT_inequality_ab_equals_bc_l320_32029


namespace NUMINAMATH_GPT_area_shaded_region_l320_32018

-- Define the conditions in Lean

def semicircle_radius_ADB : ℝ := 2
def semicircle_radius_BEC : ℝ := 2
def midpoint_arc_ADB (D : ℝ) : Prop := D = semicircle_radius_ADB
def midpoint_arc_BEC (E : ℝ) : Prop := E = semicircle_radius_BEC
def semicircle_radius_DFE : ℝ := 1
def midpoint_arc_DFE (F : ℝ) : Prop := F = semicircle_radius_DFE

-- Given the mentioned conditions, we want to show the area of the shaded region is 8 square units
theorem area_shaded_region 
  (D E F : ℝ) 
  (hD : midpoint_arc_ADB D)
  (hE : midpoint_arc_BEC E)
  (hF : midpoint_arc_DFE F) : 
  ∃ (area : ℝ), area = 8 := 
sorry

end NUMINAMATH_GPT_area_shaded_region_l320_32018


namespace NUMINAMATH_GPT_cost_of_jam_l320_32053

theorem cost_of_jam (N B J : ℕ) (hN : N > 1) (h_total_cost : N * (5 * B + 6 * J) = 348) :
    6 * N * J = 348 := by
  sorry

end NUMINAMATH_GPT_cost_of_jam_l320_32053


namespace NUMINAMATH_GPT_even_two_digit_numbers_count_l320_32063

/-- Even positive integers less than 1000 with at most two different digits -/
def count_even_two_digit_numbers : ℕ :=
  let one_digit := [2, 4, 6, 8].length
  let two_d_same := [22, 44, 66, 88].length
  let two_d_diff := [24, 42, 26, 62, 28, 82, 46, 64, 48, 84, 68, 86].length
  let three_d_same := [222, 444, 666, 888].length
  let three_d_diff := 16 + 12
  one_digit + two_d_same + two_d_diff + three_d_same + three_d_diff

theorem even_two_digit_numbers_count :
  count_even_two_digit_numbers = 52 :=
by sorry

end NUMINAMATH_GPT_even_two_digit_numbers_count_l320_32063


namespace NUMINAMATH_GPT_parallel_lines_a_l320_32084

theorem parallel_lines_a (a : ℝ) (x y : ℝ)
  (h1 : x + 2 * a * y - 1 = 0)
  (h2 : (a + 1) * x - a * y = 0)
  (h_parallel : ∀ (l1 l2 : ℝ → ℝ → Prop), l1 x y ∧ l2 x y → l1 = l2) :
  a = -3 / 2 ∨ a = 0 :=
sorry

end NUMINAMATH_GPT_parallel_lines_a_l320_32084


namespace NUMINAMATH_GPT_fraction_ratios_l320_32046

theorem fraction_ratios (m n p q : ℕ) (h1 : (m : ℚ) / n = 18) (h2 : (p : ℚ) / n = 6) (h3 : (p : ℚ) / q = 1 / 15) :
  (m : ℚ) / q = 1 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_ratios_l320_32046


namespace NUMINAMATH_GPT_kia_vehicle_count_l320_32015

theorem kia_vehicle_count (total_vehicles dodge_vehicles hyundai_vehicles kia_vehicles : ℕ)
  (h_total: total_vehicles = 400)
  (h_dodge: dodge_vehicles = total_vehicles / 2)
  (h_hyundai: hyundai_vehicles = dodge_vehicles / 2)
  (h_kia: kia_vehicles = total_vehicles - dodge_vehicles - hyundai_vehicles) : kia_vehicles = 100 := 
by
  sorry

end NUMINAMATH_GPT_kia_vehicle_count_l320_32015


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l320_32021

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l320_32021


namespace NUMINAMATH_GPT_triangle_is_3_l320_32089

def base6_addition_valid (delta : ℕ) : Prop :=
  delta < 6 ∧ 
  2 + delta + delta + 4 < 6 ∧ -- No carry effect in the middle digits
  ((delta + 3) % 6 = 4) ∧
  ((5 + delta + (2 + delta + delta + 4) / 6) % 6 = 3) ∧
  ((4 + (5 + delta + (2 + delta + delta + 4) / 6) / 6) % 6 = 5)

theorem triangle_is_3 : ∃ (δ : ℕ), base6_addition_valid δ ∧ δ = 3 :=
by
  use 3
  sorry

end NUMINAMATH_GPT_triangle_is_3_l320_32089


namespace NUMINAMATH_GPT_product_of_consecutive_even_numbers_l320_32098

theorem product_of_consecutive_even_numbers
  (a b c : ℤ)
  (h : a + b + c = 18 ∧ 2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c ∧ a < b ∧ b < c ∧ b - a = 2 ∧ c - b = 2) :
  a * b * c = 192 :=
sorry

end NUMINAMATH_GPT_product_of_consecutive_even_numbers_l320_32098


namespace NUMINAMATH_GPT_sum_of_squares_of_coeffs_l320_32028

def poly_coeffs_squared_sum (p : Polynomial ℤ) : ℤ :=
  p.coeff 5 ^ 2 + p.coeff 3 ^ 2 + p.coeff 0 ^ 2

theorem sum_of_squares_of_coeffs (p : Polynomial ℤ) (h : p = 5 * (Polynomial.C 1 * Polynomial.X ^ 5 + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.C 3)) :
  poly_coeffs_squared_sum p = 350 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coeffs_l320_32028


namespace NUMINAMATH_GPT_total_people_surveyed_l320_32080

-- Define the conditions
variable (total_surveyed : ℕ) (disease_believers : ℕ)
variable (rabies_believers : ℕ)

-- Condition 1: 75% of the people surveyed thought rats carried diseases
def condition1 (total_surveyed disease_believers : ℕ) : Prop :=
  disease_believers = (total_surveyed * 75) / 100

-- Condition 2: 50% of the people who thought rats carried diseases said rats frequently carried rabies
def condition2 (disease_believers rabies_believers : ℕ) : Prop :=
  rabies_believers = (disease_believers * 50) / 100

-- Condition 3: 18 people were mistaken in thinking rats frequently carry rabies
def condition3 (rabies_believers : ℕ) : Prop := rabies_believers = 18

-- The theorem to prove the total number of people surveyed given the conditions
theorem total_people_surveyed (total_surveyed disease_believers rabies_believers : ℕ) :
  condition1 total_surveyed disease_believers →
  condition2 disease_believers rabies_believers →
  condition3 rabies_believers →
  total_surveyed = 48 :=
by sorry

end NUMINAMATH_GPT_total_people_surveyed_l320_32080


namespace NUMINAMATH_GPT_tourists_originally_in_group_l320_32041

theorem tourists_originally_in_group (x : ℕ) (h₁ : 220 / x - 220 / (x + 1) = 2) : x = 10 := 
by
  sorry

end NUMINAMATH_GPT_tourists_originally_in_group_l320_32041


namespace NUMINAMATH_GPT_initial_shares_bought_l320_32002

variable (x : ℕ) -- x is the number of shares Tom initially bought

-- Conditions:
def initial_cost_per_share : ℕ := 3
def num_shares_sold : ℕ := 10
def selling_price_per_share : ℕ := 4
def doubled_value_per_remaining_share : ℕ := 2 * initial_cost_per_share
def total_profit : ℤ := 40

-- Proving the number of shares initially bought
theorem initial_shares_bought (h : num_shares_sold * selling_price_per_share - x * initial_cost_per_share = total_profit) :
  x = 10 := by sorry

end NUMINAMATH_GPT_initial_shares_bought_l320_32002


namespace NUMINAMATH_GPT_least_number_added_to_divisible_l320_32086

theorem least_number_added_to_divisible (n : ℕ) (k : ℕ) : n = 1789 → k = 11 → (n + k) % Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 4 3)) = 0 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_least_number_added_to_divisible_l320_32086


namespace NUMINAMATH_GPT_greatest_length_of_equal_pieces_l320_32058

theorem greatest_length_of_equal_pieces (a b c : ℕ) (h₁ : a = 42) (h₂ : b = 63) (h₃ : c = 84) :
  Nat.gcd (Nat.gcd a b) c = 21 :=
by
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_greatest_length_of_equal_pieces_l320_32058


namespace NUMINAMATH_GPT_prime_1002_n_count_l320_32033

theorem prime_1002_n_count :
  ∃! n : ℕ, n ≥ 2 ∧ Prime (n^3 + 2) :=
by
  sorry

end NUMINAMATH_GPT_prime_1002_n_count_l320_32033


namespace NUMINAMATH_GPT_framed_painting_ratio_l320_32039

theorem framed_painting_ratio
  (width_painting : ℕ)
  (height_painting : ℕ)
  (frame_side : ℕ)
  (frame_top_bottom : ℕ)
  (h1 : width_painting = 20)
  (h2 : height_painting = 30)
  (h3 : frame_top_bottom = 3 * frame_side)
  (h4 : (width_painting + 2 * frame_side) * (height_painting + 2 * frame_top_bottom) = 2 * width_painting * height_painting):
  (width_painting + 2 * frame_side) = 1/2 * (height_painting + 2 * frame_top_bottom) := 
by
  sorry

end NUMINAMATH_GPT_framed_painting_ratio_l320_32039


namespace NUMINAMATH_GPT_division_remainder_l320_32027

theorem division_remainder 
  (R D Q : ℕ) 
  (h1 : D = 3 * Q)
  (h2 : D = 3 * R + 3)
  (h3 : 113 = D * Q + R) : R = 5 :=
sorry

end NUMINAMATH_GPT_division_remainder_l320_32027


namespace NUMINAMATH_GPT_quadratic_real_roots_l320_32085

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l320_32085


namespace NUMINAMATH_GPT_perimeter_of_grid_l320_32071

theorem perimeter_of_grid (area: ℕ) (side_length: ℕ) (perimeter: ℕ) 
  (h1: area = 144) 
  (h2: 4 * side_length * side_length = area) 
  (h3: perimeter = 4 * 2 * side_length) : 
  perimeter = 48 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_grid_l320_32071


namespace NUMINAMATH_GPT_binomial_coefficients_sum_l320_32035

theorem binomial_coefficients_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficients_sum_l320_32035


namespace NUMINAMATH_GPT_pages_needed_l320_32088

def total_new_cards : ℕ := 8
def total_old_cards : ℕ := 10
def cards_per_page : ℕ := 3

theorem pages_needed (h : total_new_cards = 8) (h2 : total_old_cards = 10) (h3 : cards_per_page = 3) : 
  (total_new_cards + total_old_cards) / cards_per_page = 6 := by 
  sorry

end NUMINAMATH_GPT_pages_needed_l320_32088


namespace NUMINAMATH_GPT_find_a_l320_32001

theorem find_a (a : ℝ) (t : ℝ) :
  (4 = 1 + 3 * t) ∧ (3 = a * t^2 + 2) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l320_32001


namespace NUMINAMATH_GPT_cubics_sum_l320_32040

theorem cubics_sum (a b c : ℝ) (h₁ : a + b + c = 4) (h₂ : ab + ac + bc = 6) (h₃ : abc = -8) :
  a^3 + b^3 + c^3 = 8 :=
by {
  -- proof steps would go here
  sorry
}

end NUMINAMATH_GPT_cubics_sum_l320_32040


namespace NUMINAMATH_GPT_prism_faces_same_color_l320_32037

structure PrismColoring :=
  (A : Fin 5 → Fin 5 → Bool)
  (B : Fin 5 → Fin 5 → Bool)
  (A_to_B : Fin 5 → Fin 5 → Bool)

def all_triangles_diff_colors (pc : PrismColoring) : Prop :=
  ∀ i j k : Fin 5, i ≠ j → j ≠ k → k ≠ i →
    (pc.A i j = !pc.A i k ∨ pc.A i j = !pc.A j k) ∧
    (pc.B i j = !pc.B i k ∨ pc.B i j = !pc.B j k) ∧
    (pc.A_to_B i j = !pc.A_to_B i k ∨ pc.A_to_B i j = !pc.A_to_B j k)

theorem prism_faces_same_color (pc : PrismColoring) (h : all_triangles_diff_colors pc) :
  (∀ i j : Fin 5, pc.A i j = pc.A 0 1) ∧ (∀ i j : Fin 5, pc.B i j = pc.B 0 1) :=
sorry

end NUMINAMATH_GPT_prism_faces_same_color_l320_32037


namespace NUMINAMATH_GPT_Mike_changed_64_tires_l320_32022

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end NUMINAMATH_GPT_Mike_changed_64_tires_l320_32022


namespace NUMINAMATH_GPT_part1_prove_BD_eq_b_part2_prove_cos_ABC_l320_32073

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_part1_prove_BD_eq_b_part2_prove_cos_ABC_l320_32073


namespace NUMINAMATH_GPT_sin_cos_eq_one_l320_32007

open Real

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h2 : x < 2 * π) (h : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_eq_one_l320_32007


namespace NUMINAMATH_GPT_infinite_series_fraction_l320_32097

theorem infinite_series_fraction:
  (∑' n : ℕ, (if n = 0 then 0 else ((2 : ℚ) / (3 * n) - (1 : ℚ) / (3 * (n + 1)) - (7 : ℚ) / (6 * (n + 3))))) =
  (1 : ℚ) / 3 := 
sorry

end NUMINAMATH_GPT_infinite_series_fraction_l320_32097


namespace NUMINAMATH_GPT_avg_integer_N_between_fractions_l320_32044

theorem avg_integer_N_between_fractions (N : ℕ) (h1 : (2 : ℚ) / 5 < N / 42) (h2 : N / 42 < 1 / 3) : 
  N = 15 := 
by
  sorry

end NUMINAMATH_GPT_avg_integer_N_between_fractions_l320_32044


namespace NUMINAMATH_GPT_initial_loss_percentage_l320_32013

theorem initial_loss_percentage 
  (C : ℝ) 
  (h1 : selling_price_one_pencil_20 = 1 / 20)
  (h2 : selling_price_one_pencil_10 = 1 / 10)
  (h3 : C = 1 / (10 * 1.30)) :
  (C - selling_price_one_pencil_20) / C * 100 = 35 :=
by
  sorry

end NUMINAMATH_GPT_initial_loss_percentage_l320_32013


namespace NUMINAMATH_GPT_simplify_expression_l320_32059

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 
  3 * (4 - 2 * i) + 2 * i * (3 + i) + 5 * (-1 + i) = 5 + 5 * i :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l320_32059


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l320_32061

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := {x | x^2 - 2 * x < 0}
def setB : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem to prove the intersection A ∩ B
theorem intersection_of_A_and_B : ((setA ∩ setB) = {x : ℝ | 0 < x ∧ x < 1}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l320_32061


namespace NUMINAMATH_GPT_triangle_side_relation_l320_32099

theorem triangle_side_relation (a b c : ℝ) (h1 : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) (h2 : a + b > c) :
  a + c = 2 * b := 
sorry

end NUMINAMATH_GPT_triangle_side_relation_l320_32099


namespace NUMINAMATH_GPT_probability_one_instrument_l320_32077

theorem probability_one_instrument (total_people : ℕ) (at_least_one_instrument_ratio : ℚ) (two_or_more_instruments : ℕ)
  (h1 : total_people = 800) (h2 : at_least_one_instrument_ratio = 1 / 5) (h3 : two_or_more_instruments = 128) :
  (160 - 128) / 800 = 1 / 25 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_instrument_l320_32077


namespace NUMINAMATH_GPT_power_identity_l320_32045

theorem power_identity :
  (3 ^ 12) * (3 ^ 8) = 243 ^ 4 :=
sorry

end NUMINAMATH_GPT_power_identity_l320_32045


namespace NUMINAMATH_GPT_evaluate_expression_l320_32004

-- Define the greatest power of 2 and 3 that are factors of 360
def a : ℕ := 3 -- 2^3 is the greatest power of 2 that is a factor of 360
def b : ℕ := 2 -- 3^2 is the greatest power of 3 that is a factor of 360

theorem evaluate_expression : (1 / 4)^(b - a) = 4 := 
by 
  have h1 : a = 3 := rfl
  have h2 : b = 2 := rfl
  rw [h1, h2]
  simp
  sorry

end NUMINAMATH_GPT_evaluate_expression_l320_32004


namespace NUMINAMATH_GPT_card_picking_l320_32081

/-
Statement of the problem:
- A modified deck of cards has 65 cards.
- The deck is divided into 5 suits, each of which has 13 cards.
- The cards are placed in random order.
- Prove that the number of ways to pick two different cards from this deck with the order of picking being significant is 4160.
-/
theorem card_picking : (65 * 64) = 4160 := by
  sorry

end NUMINAMATH_GPT_card_picking_l320_32081


namespace NUMINAMATH_GPT_find_m_eq_l320_32066

theorem find_m_eq : 
  (∀ (m : ℝ),
    ((m + 2)^2 + (m + 3)^2 = m^2 + 16 + 4 + (m - 1)^2) →
    m = 2 / 3 ) :=
by
  intros m h
  sorry

end NUMINAMATH_GPT_find_m_eq_l320_32066


namespace NUMINAMATH_GPT_work_rate_l320_32019

theorem work_rate (A_rate : ℝ) (combined_rate : ℝ) (B_days : ℝ) :
  A_rate = 1 / 12 ∧ combined_rate = 1 / 6.461538461538462 → 1 / B_days = combined_rate - A_rate → B_days = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_work_rate_l320_32019


namespace NUMINAMATH_GPT_license_plate_combinations_l320_32032

-- Definitions representing the conditions
def valid_license_plates_count : ℕ :=
  let letter_combinations := Nat.choose 26 2 -- Choose 2 unique letters
  let letter_arrangements := Nat.choose 4 2 * 2 -- Arrange the repeated letters
  let digit_combinations := 10 * 9 * 8 -- Choose different digits
  letter_combinations * letter_arrangements * digit_combinations

-- The theorem representing the problem statement
theorem license_plate_combinations :
  valid_license_plates_count = 2808000 := 
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l320_32032


namespace NUMINAMATH_GPT_total_notes_proof_l320_32008

variable (x : Nat)

def total_money := 10350
def fifty_notes_count := 17
def fifty_notes_value := 850  -- 17 * 50
def five_hundred_notes_value := 500 * x
def total_value_proposition := fifty_notes_value + five_hundred_notes_value = total_money

theorem total_notes_proof :
  total_value_proposition -> (fifty_notes_count + x) = 36 :=
by
  intros h
  -- The proof steps would go here, but we use sorry for now.
  sorry

end NUMINAMATH_GPT_total_notes_proof_l320_32008


namespace NUMINAMATH_GPT_incorrect_statement_about_absolute_value_l320_32090

theorem incorrect_statement_about_absolute_value (x : ℝ) : abs x = 0 → x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_statement_about_absolute_value_l320_32090


namespace NUMINAMATH_GPT_shaded_area_in_design_l320_32023

theorem shaded_area_in_design (side_length : ℝ) (radius : ℝ)
  (h1 : side_length = 30) (h2 : radius = side_length / 6)
  (h3 : 6 * (π * radius^2) = 150 * π) :
  (side_length^2) - 6 * (π * radius^2) = 900 - 150 * π := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_in_design_l320_32023


namespace NUMINAMATH_GPT_base6_sum_l320_32074

theorem base6_sum (D C : ℕ) (h₁ : D + 2 = C) (h₂ : C + 3 = 7) : C + D = 6 :=
by
  sorry

end NUMINAMATH_GPT_base6_sum_l320_32074


namespace NUMINAMATH_GPT_nathan_and_parents_total_cost_l320_32087

/-- Define the total number of people -/
def num_people := 3

/-- Define the cost per object -/
def cost_per_object := 11

/-- Define the number of objects per person -/
def objects_per_person := 2 + 2 + 1

/-- Define the total number of objects -/
def total_objects := num_people * objects_per_person

/-- Define the total cost -/
def total_cost := total_objects * cost_per_object

/-- The main theorem to prove the total cost -/
theorem nathan_and_parents_total_cost : total_cost = 165 := by
  sorry

end NUMINAMATH_GPT_nathan_and_parents_total_cost_l320_32087


namespace NUMINAMATH_GPT_emma_uniform_number_correct_l320_32062

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

noncomputable def dan : ℕ := 11  -- Example value, but needs to satisfy all conditions
noncomputable def emma : ℕ := 19  -- This is what we need to prove
noncomputable def fiona : ℕ := 13  -- Example value, but needs to satisfy all conditions
noncomputable def george : ℕ := 11  -- Example value, but needs to satisfy all conditions

theorem emma_uniform_number_correct :
  is_two_digit_prime dan ∧
  is_two_digit_prime emma ∧
  is_two_digit_prime fiona ∧
  is_two_digit_prime george ∧
  dan ≠ emma ∧ dan ≠ fiona ∧ dan ≠ george ∧
  emma ≠ fiona ∧ emma ≠ george ∧
  fiona ≠ george ∧
  dan + fiona = 23 ∧
  george + emma = 9 ∧
  dan + fiona + george + emma = 32
  → emma = 19 :=
sorry

end NUMINAMATH_GPT_emma_uniform_number_correct_l320_32062


namespace NUMINAMATH_GPT_find_a_l320_32078

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x + 1)

theorem find_a {a : ℝ} (h : (deriv (f a) 0 = 1)) : a = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_a_l320_32078


namespace NUMINAMATH_GPT_shirts_not_all_on_sale_implications_l320_32068

variable (Shirts : Type) (store_contains : Shirts → Prop) (on_sale : Shirts → Prop)

theorem shirts_not_all_on_sale_implications :
  ¬ ∀ s, store_contains s → on_sale s → 
  (∃ s, store_contains s ∧ ¬ on_sale s) ∧ (∃ s, store_contains s ∧ ¬ on_sale s) :=
by
  sorry

end NUMINAMATH_GPT_shirts_not_all_on_sale_implications_l320_32068


namespace NUMINAMATH_GPT_global_maximum_condition_l320_32052

noncomputable def f (x m : ℝ) : ℝ :=
if x ≤ m then -x^2 - 2 * x else -x + 2

theorem global_maximum_condition (m : ℝ) (h : ∃ (x0 : ℝ), ∀ (x : ℝ), f x m ≤ f x0 m) : m ≥ 1 :=
sorry

end NUMINAMATH_GPT_global_maximum_condition_l320_32052


namespace NUMINAMATH_GPT_second_set_length_is_20_l320_32026

-- Define the lengths
def length_first_set : ℕ := 4
def length_second_set : ℕ := 5 * length_first_set

-- Formal proof statement
theorem second_set_length_is_20 : length_second_set = 20 :=
by
  sorry

end NUMINAMATH_GPT_second_set_length_is_20_l320_32026


namespace NUMINAMATH_GPT_tea_or_coffee_indifference_l320_32064

open Classical

theorem tea_or_coffee_indifference : 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) → 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) :=
by
  sorry

end NUMINAMATH_GPT_tea_or_coffee_indifference_l320_32064


namespace NUMINAMATH_GPT_g_five_eq_thirteen_sevenths_l320_32065

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_five_eq_thirteen_sevenths : g 5 = 13 / 7 := by
  sorry

end NUMINAMATH_GPT_g_five_eq_thirteen_sevenths_l320_32065


namespace NUMINAMATH_GPT_cricket_initial_overs_l320_32079

theorem cricket_initial_overs
  (target_runs : ℚ) (initial_run_rate : ℚ) (remaining_run_rate : ℚ) (remaining_overs : ℕ)
  (total_runs_needed : target_runs = 282)
  (run_rate_initial : initial_run_rate = 3.4)
  (run_rate_remaining : remaining_run_rate = 6.2)
  (overs_remaining : remaining_overs = 40) :
  ∃ (initial_overs : ℕ), initial_overs = 10 :=
by
  sorry

end NUMINAMATH_GPT_cricket_initial_overs_l320_32079


namespace NUMINAMATH_GPT_Fred_earned_4_dollars_l320_32048

-- Conditions are translated to definitions
def initial_amount_Fred : ℕ := 111
def current_amount_Fred : ℕ := 115

-- Proof problem in Lean 4 statement
theorem Fred_earned_4_dollars : current_amount_Fred - initial_amount_Fred = 4 := by
  sorry

end NUMINAMATH_GPT_Fred_earned_4_dollars_l320_32048
