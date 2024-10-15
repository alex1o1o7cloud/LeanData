import Mathlib

namespace NUMINAMATH_GPT_α_in_quadrants_l2220_222021

def α (k : ℤ) : ℝ := k * 180 + 45

theorem α_in_quadrants (k : ℤ) : 
  (0 ≤ α k ∧ α k < 90) ∨ (180 < α k ∧ α k ≤ 270) :=
sorry

end NUMINAMATH_GPT_α_in_quadrants_l2220_222021


namespace NUMINAMATH_GPT_initial_price_of_phone_l2220_222057

theorem initial_price_of_phone
  (initial_price_TV : ℕ)
  (increase_TV_fraction : ℚ)
  (initial_price_phone : ℚ)
  (increase_phone_percentage : ℚ)
  (total_amount : ℚ)
  (h1 : initial_price_TV = 500)
  (h2 : increase_TV_fraction = 2/5)
  (h3 : increase_phone_percentage = 0.40)
  (h4 : total_amount = 1260) :
  initial_price_phone = 400 := by
  sorry

end NUMINAMATH_GPT_initial_price_of_phone_l2220_222057


namespace NUMINAMATH_GPT_range_of_a_l2220_222036

def f (x a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ a < -3 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2220_222036


namespace NUMINAMATH_GPT_not_possible_acquaintance_arrangement_l2220_222074

-- Definitions and conditions for the problem
def num_people : ℕ := 40
def even_people_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 0 → A ≠ B → true -- A and B have a mutual acquaintance if an even number of people sit between them

def odd_people_not_acquainted (A B : ℕ) (num_between : ℕ) : Prop :=
  num_between % 2 = 1 → A ≠ B → true -- A and B do not have a mutual acquaintance if an odd number of people sit between them

theorem not_possible_acquaintance_arrangement : ¬ (∀ A B : ℕ, A ≠ B →
  (∀ num_between : ℕ, (num_between % 2 = 0 → even_people_acquainted A B num_between) ∧
  (num_between % 2 = 1 → odd_people_not_acquainted A B num_between))) :=
sorry

end NUMINAMATH_GPT_not_possible_acquaintance_arrangement_l2220_222074


namespace NUMINAMATH_GPT_solve_trig_eq_l2220_222096

theorem solve_trig_eq (k : ℤ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * (Real.sin t)^2 - Real.sin (2 * t) + 3 * Real.cos t^2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l2220_222096


namespace NUMINAMATH_GPT_simplify_expression_l2220_222091

theorem simplify_expression (n : ℕ) : 
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2220_222091


namespace NUMINAMATH_GPT_digital_earth_correct_purposes_l2220_222025

def Purpose : Type := String

def P1 : Purpose := "To deal with natural and social issues of the entire Earth using digital means."
def P2 : Purpose := "To maximize the utilization of natural resources."
def P3 : Purpose := "To conveniently obtain information about the Earth."
def P4 : Purpose := "To provide precise locations, directions of movement, and speeds of moving objects."

def correct_purposes : Set Purpose := {P1, P2, P3}

theorem digital_earth_correct_purposes :
  {P1, P2, P3} = correct_purposes :=
by 
  sorry

end NUMINAMATH_GPT_digital_earth_correct_purposes_l2220_222025


namespace NUMINAMATH_GPT_emily_strawberry_harvest_l2220_222042

-- Define the dimensions of the garden
def garden_length : ℕ := 10
def garden_width : ℕ := 7

-- Define the planting density
def plants_per_sqft : ℕ := 3

-- Define the yield per plant
def strawberries_per_plant : ℕ := 12

-- Define the expected number of strawberries
def expected_strawberries : ℕ := 2520

-- Theorem statement to prove the total number of strawberries
theorem emily_strawberry_harvest :
  garden_length * garden_width * plants_per_sqft * strawberries_per_plant = expected_strawberries :=
by
  -- Proof goes here (for now, we use sorry to indicate the proof is omitted)
  sorry

end NUMINAMATH_GPT_emily_strawberry_harvest_l2220_222042


namespace NUMINAMATH_GPT_number_b_smaller_than_number_a_l2220_222092

theorem number_b_smaller_than_number_a (A B : ℝ)
  (h : A = B + 1/4) : (B + 1/4 = A) ∧ (B < A) → B = (4 * A - A) / 5 := by
  sorry

end NUMINAMATH_GPT_number_b_smaller_than_number_a_l2220_222092


namespace NUMINAMATH_GPT_remainder_x2023_l2220_222058

theorem remainder_x2023 (x : ℤ) : 
  let dividend := x^2023 + 1
  let divisor := x^6 - x^4 + x^2 - 1
  let remainder := -x^7 + 1
  dividend % divisor = remainder :=
by
  sorry

end NUMINAMATH_GPT_remainder_x2023_l2220_222058


namespace NUMINAMATH_GPT_students_in_classroom_l2220_222006

theorem students_in_classroom :
  ∃ n : ℕ, (n < 50) ∧ (n % 6 = 5) ∧ (n % 3 = 2) ∧ 
  (n = 5 ∨ n = 11 ∨ n = 17 ∨ n = 23 ∨ n = 29 ∨ n = 35 ∨ n = 41 ∨ n = 47) :=
by
  sorry

end NUMINAMATH_GPT_students_in_classroom_l2220_222006


namespace NUMINAMATH_GPT_derivative_of_curve_tangent_line_at_one_l2220_222051

-- Definition of the curve
def curve (x : ℝ) : ℝ := x^3 + 5 * x^2 + 3 * x

-- Part 1: Prove the derivative of the curve
theorem derivative_of_curve (x : ℝ) :
  deriv curve x = 3 * x^2 + 10 * x + 3 :=
sorry

-- Part 2: Prove the equation of the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a = 16 ∧ b = -1 ∧ c = -7 ∧
  ∀ (x y : ℝ), curve 1 = 9 → y - 9 = 16 * (x - 1) → a * x + b * y + c = 0 :=
sorry

end NUMINAMATH_GPT_derivative_of_curve_tangent_line_at_one_l2220_222051


namespace NUMINAMATH_GPT_triangle_right_triangle_l2220_222049

theorem triangle_right_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 3) (h2 : b = 4) (h3 : c^2 - 10 * c + 25 = 0) : 
  a^2 + b^2 = c^2 :=
by
  -- We know the values of a, b, and c by the conditions
  sorry

end NUMINAMATH_GPT_triangle_right_triangle_l2220_222049


namespace NUMINAMATH_GPT_maximum_number_of_intersections_of_150_lines_is_7171_l2220_222076

def lines_are_distinct (L : ℕ → Type) : Prop := 
  ∀ n m : ℕ, n ≠ m → L n ≠ L m

def lines_parallel_to_each_other (L : ℕ → Type) (k : ℕ) : Prop :=
  ∀ n m : ℕ, n ≠ m → L (k * n) = L (k * m)

def lines_pass_through_point_B (L : ℕ → Type) (B : Type) (k : ℕ) : Prop :=
  ∀ n : ℕ, L (k * n - 4) = B

def lines_not_parallel (L : ℕ → Type) (k1 k2 : ℕ) : Prop :=
  ∀ n m : ℕ, L (k1 * n) ≠ L (k2 * m)

noncomputable def max_points_of_intersection
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : ℕ :=
  7171

theorem maximum_number_of_intersections_of_150_lines_is_7171
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : max_points_of_intersection L B k1 k2 h_distinct h_parallel1 h_parallel2 h_pass_through_B h_not_parallel = 7171 := 
  by 
  sorry

end NUMINAMATH_GPT_maximum_number_of_intersections_of_150_lines_is_7171_l2220_222076


namespace NUMINAMATH_GPT_rationalize_sqrt_fraction_l2220_222095

theorem rationalize_sqrt_fraction {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) : 
  (Real.sqrt ((a : ℝ) / b)) = (Real.sqrt (a * (b / (b * b)))) → 
  (Real.sqrt (5 / 12)) = (Real.sqrt 15 / 6) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_sqrt_fraction_l2220_222095


namespace NUMINAMATH_GPT_evaluate_complex_ratio_l2220_222034

noncomputable def complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) : ℂ :=
(a^12 + b^12) / (a + b)^12

theorem evaluate_complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) :
  complex_ratio a b h1 h2 h3 = 1 / 32 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_complex_ratio_l2220_222034


namespace NUMINAMATH_GPT_oldest_child_age_l2220_222030

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_oldest_child_age_l2220_222030


namespace NUMINAMATH_GPT_binary_representation_of_28_l2220_222029

-- Define a function to convert a number to binary representation.
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem binary_representation_of_28 : decimalToBinary 28 = [1, 1, 1, 0, 0] := 
  sorry

end NUMINAMATH_GPT_binary_representation_of_28_l2220_222029


namespace NUMINAMATH_GPT_area_of_rhombus_l2220_222005

theorem area_of_rhombus (P D : ℕ) (area : ℝ) (hP : P = 48) (hD : D = 26) :
  area = 25 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l2220_222005


namespace NUMINAMATH_GPT_ratio_minutes_l2220_222078

theorem ratio_minutes (x : ℝ) : 
  (12 / 8) = (6 / (x * 60)) → x = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_minutes_l2220_222078


namespace NUMINAMATH_GPT_find_r_s_l2220_222022

def N : Matrix (Fin 2) (Fin 2) Int := ![![3, 4], ![-2, 0]]
def I : Matrix (Fin 2) (Fin 2) Int := ![![1, 0], ![0, 1]]

theorem find_r_s :
  ∃ (r s : Int), (N * N = r • N + s • I) ∧ (r = 3) ∧ (s = 16) :=
by
  sorry

end NUMINAMATH_GPT_find_r_s_l2220_222022


namespace NUMINAMATH_GPT_committee_member_count_l2220_222039

theorem committee_member_count (n : ℕ) (M : ℕ) (Q : ℚ) 
  (h₁ : M = 6) 
  (h₂ : 2 * n = M) 
  (h₃ : Q = 0.4) 
  (h₄ : Q = (n - 1) / (M - 1)) : 
  n = 3 :=
by
  sorry

end NUMINAMATH_GPT_committee_member_count_l2220_222039


namespace NUMINAMATH_GPT_smallest_y_value_l2220_222075

theorem smallest_y_value (y : ℝ) : (12 * y^2 - 56 * y + 48 = 0) → y = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_value_l2220_222075


namespace NUMINAMATH_GPT_probability_of_selecting_product_not_less_than_4_l2220_222007

theorem probability_of_selecting_product_not_less_than_4 :
  let total_products := 5 
  let favorable_outcomes := 2 
  (favorable_outcomes : ℚ) / total_products = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_selecting_product_not_less_than_4_l2220_222007


namespace NUMINAMATH_GPT_geometric_series_m_value_l2220_222082

theorem geometric_series_m_value (m : ℝ) : 
    let a : ℝ := 20
    let r₁ : ℝ := 1 / 2  -- Common ratio for the first series
    let S₁ : ℝ := a / (1 - r₁)  -- Sum of the first series
    let b : ℝ := 1 / 2 + m / 20  -- Common ratio for the second series
    let S₂ : ℝ := a / (1 - b)  -- Sum of the second series
    S₁ = 40 ∧ S₂ = 120 → m = 20 / 3 :=
sorry

end NUMINAMATH_GPT_geometric_series_m_value_l2220_222082


namespace NUMINAMATH_GPT_equal_perimeter_triangle_side_length_l2220_222023

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end NUMINAMATH_GPT_equal_perimeter_triangle_side_length_l2220_222023


namespace NUMINAMATH_GPT_find_angle_C_l2220_222037

theorem find_angle_C (a b c A B C : ℝ) (h₀ : 0 < C) (h₁ : C < Real.pi)
  (h₂ : 2 * c * Real.sin A = a * Real.tan C) :
  C = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_C_l2220_222037


namespace NUMINAMATH_GPT_basketball_game_l2220_222089

/-- Given the conditions of the basketball game:
  * a, ar, ar^2, ar^3 form the Dragons' scores
  * b, b + d, b + 2d, b + 3d form the Lions' scores
  * The game was tied at halftime: a + ar = b + (b + d)
  * The Dragons won by three points at the end: a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3
  * Neither team scored more than 100 points
Prove that the total number of points scored by the two teams in the first half is 30.
-/
theorem basketball_game (a r b d : ℕ) (h1 : a + a * r = b + (b + d))
  (h2 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (h3 : a * (1 + r + r^2 + r^3) < 100)
  (h4 : 4 * b + 6 * d < 100) :
  a + a * r + b + (b + d) = 30 :=
by
  sorry

end NUMINAMATH_GPT_basketball_game_l2220_222089


namespace NUMINAMATH_GPT_upper_bound_y_l2220_222059

theorem upper_bound_y 
  (U : ℤ) 
  (x y : ℤ)
  (h1 : 3 < x ∧ x < 6) 
  (h2 : 6 < y ∧ y < U) 
  (h3 : y - x = 4) : 
  U = 10 := 
sorry

end NUMINAMATH_GPT_upper_bound_y_l2220_222059


namespace NUMINAMATH_GPT_intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l2220_222070

noncomputable def setA : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | 1 < x }

theorem intersection_AB : setA ∩ setB = { x : ℝ | 1 < x ∧ x < 2 } := by
  sorry

theorem union_AB : setA ∪ setB = { x : ℝ | -1 < x } := by
  sorry

theorem difference_A_minus_B : setA \ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } := by
  sorry

theorem difference_B_minus_A : setB \ setA = { x : ℝ | 2 ≤ x } := by
  sorry

end NUMINAMATH_GPT_intersection_AB_union_AB_difference_A_minus_B_difference_B_minus_A_l2220_222070


namespace NUMINAMATH_GPT_Joan_spent_68_353_on_clothing_l2220_222098

theorem Joan_spent_68_353_on_clothing :
  let shorts := 15.00
  let jacket := 14.82 * 0.9
  let shirt := 12.51 * 0.5
  let shoes := 21.67 - 3
  let hat := 8.75
  let belt := 6.34
  shorts + jacket + shirt + shoes + hat + belt = 68.353 :=
sorry

end NUMINAMATH_GPT_Joan_spent_68_353_on_clothing_l2220_222098


namespace NUMINAMATH_GPT_candy_distribution_l2220_222043

-- Definition of the problem
def emily_candies : ℕ := 30
def friends : ℕ := 4

-- Lean statement to prove
theorem candy_distribution : emily_candies % friends = 2 :=
by sorry

end NUMINAMATH_GPT_candy_distribution_l2220_222043


namespace NUMINAMATH_GPT_water_added_l2220_222083

theorem water_added (W x : ℕ) (h₁ : 2 * W = 5 * 10)
                    (h₂ : 2 * (W + x) = 7 * 10) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_water_added_l2220_222083


namespace NUMINAMATH_GPT_range_of_m_l2220_222052

theorem range_of_m (x y m : ℝ) 
  (h1 : 3 * x + y = m - 1)
  (h2 : x - 3 * y = 2 * m)
  (h3 : x + 2 * y ≥ 0) : 
  m ≤ -1 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2220_222052


namespace NUMINAMATH_GPT_number_of_daisies_is_two_l2220_222090

theorem number_of_daisies_is_two :
  ∀ (total_flowers daisies tulips sunflowers remaining_flowers : ℕ), 
    total_flowers = 12 →
    sunflowers = 4 →
    (3 / 5) * remaining_flowers = tulips →
    (2 / 5) * remaining_flowers = sunflowers →
    remaining_flowers = total_flowers - daisies - sunflowers →
    daisies = 2 :=
by
  intros total_flowers daisies tulips sunflowers remaining_flowers 
  sorry

end NUMINAMATH_GPT_number_of_daisies_is_two_l2220_222090


namespace NUMINAMATH_GPT_green_shirt_pairs_l2220_222068

theorem green_shirt_pairs (blue_shirts green_shirts total_pairs blue_blue_pairs : ℕ) 
(h1 : blue_shirts = 68) 
(h2 : green_shirts = 82) 
(h3 : total_pairs = 75) 
(h4 : blue_blue_pairs = 30) 
: (green_shirts - (blue_shirts - 2 * blue_blue_pairs)) / 2 = 37 := 
by 
  -- This is where the proof would be written, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_green_shirt_pairs_l2220_222068


namespace NUMINAMATH_GPT_find_y_l2220_222053

theorem find_y (y : ℚ) (h : Real.sqrt (1 + Real.sqrt (3 * y - 4)) = Real.sqrt 9) : y = 68 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l2220_222053


namespace NUMINAMATH_GPT_inequality_proof_l2220_222060

theorem inequality_proof (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
    (a * b + b * c + c * a) * (1 / (a + b)^2 + 1 / (b + c)^2 + 1 / (c + a)^2) ≥ 9 / 4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2220_222060


namespace NUMINAMATH_GPT_sum_of_real_roots_of_even_function_l2220_222013

noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem sum_of_real_roots_of_even_function (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_intersects : ∃ a b c d, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d) :
  a + b + c + d = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_real_roots_of_even_function_l2220_222013


namespace NUMINAMATH_GPT_cars_served_from_4pm_to_6pm_l2220_222088

theorem cars_served_from_4pm_to_6pm : 
  let cars_per_15_min_peak := 12
  let cars_per_15_min_offpeak := 8 
  let blocks_in_an_hour := 4 
  let total_peak_hour := cars_per_15_min_peak * blocks_in_an_hour 
  let total_offpeak_hour := cars_per_15_min_offpeak * blocks_in_an_hour 
  total_peak_hour + total_offpeak_hour = 80 := 
by 
  sorry 

end NUMINAMATH_GPT_cars_served_from_4pm_to_6pm_l2220_222088


namespace NUMINAMATH_GPT_curve_symmetry_l2220_222027

-- Define the curve as a predicate
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0

-- Define the point symmetry condition for a line
def is_symmetric_about_line (curve : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, curve x y → line x y

-- Define the line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop := x + y = 0

-- Main theorem stating the curve is symmetrical about the line x + y = 0
theorem curve_symmetry : is_symmetric_about_line curve line_x_plus_y_eq_0 := 
sorry

end NUMINAMATH_GPT_curve_symmetry_l2220_222027


namespace NUMINAMATH_GPT_minimum_value_of_f_l2220_222031

open Real

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sin x

theorem minimum_value_of_f (x : ℝ) (h : abs x ≤ π / 4) : 
  ∃ m : ℝ, (∀ y : ℝ, f y ≥ m) ∧ m = 1 / 2 - sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2220_222031


namespace NUMINAMATH_GPT_initial_number_of_persons_l2220_222011

/-- The average weight of some persons increases by 3 kg when a new person comes in place of one of them weighing 65 kg. 
    The weight of the new person might be 89 kg.
    Prove that the number of persons initially was 8.
-/
theorem initial_number_of_persons (n : ℕ) (h1 : (89 - 65 = 3 * n)) : n = 8 := by
  sorry

end NUMINAMATH_GPT_initial_number_of_persons_l2220_222011


namespace NUMINAMATH_GPT_value_of_y_minus_x_l2220_222048

theorem value_of_y_minus_x (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : x + y = 8) 
  (h3 : y - 3 * x + z = 9) : 
  y - x = 6.5 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_value_of_y_minus_x_l2220_222048


namespace NUMINAMATH_GPT_number_of_members_l2220_222015

theorem number_of_members (n : ℕ) (H : n * n = 5776) : n = 76 :=
by
  sorry

end NUMINAMATH_GPT_number_of_members_l2220_222015


namespace NUMINAMATH_GPT_homer_second_try_points_l2220_222065

theorem homer_second_try_points (x : ℕ) :
  400 + x + 2 * x = 1390 → x = 330 :=
by
  sorry

end NUMINAMATH_GPT_homer_second_try_points_l2220_222065


namespace NUMINAMATH_GPT_variance_of_scores_l2220_222044

def scores : List ℝ := [8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (List.sum (List.map (λ x => (x - m) ^ 2) xs)) / (xs.length)

theorem variance_of_scores : variance scores = 40 / 9 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_scores_l2220_222044


namespace NUMINAMATH_GPT_domain_of_h_l2220_222066

theorem domain_of_h (x : ℝ) : |x - 5| + |x + 3| ≠ 0 := by
  sorry

end NUMINAMATH_GPT_domain_of_h_l2220_222066


namespace NUMINAMATH_GPT_inequality_proof_l2220_222035

theorem inequality_proof (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 → x > 5 / 7 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2220_222035


namespace NUMINAMATH_GPT_sum_of_consecutive_even_negative_integers_l2220_222073

theorem sum_of_consecutive_even_negative_integers (n m : ℤ) 
  (h1 : n % 2 = 0)
  (h2 : m % 2 = 0)
  (h3 : n < 0)
  (h4 : m < 0)
  (h5 : m = n + 2)
  (h6 : n * m = 2496) : n + m = -102 := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_negative_integers_l2220_222073


namespace NUMINAMATH_GPT_f_increasing_interval_l2220_222024

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end NUMINAMATH_GPT_f_increasing_interval_l2220_222024


namespace NUMINAMATH_GPT_evaluate_rr2_l2220_222041

def q (x : ℝ) : ℝ := x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := (x - 3) * (x - 2)

theorem evaluate_rr2 : r (r 2) = 6 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_rr2_l2220_222041


namespace NUMINAMATH_GPT_birthday_candles_l2220_222012

def number_of_red_candles : ℕ := 18
def number_of_green_candles : ℕ := 37
def number_of_yellow_candles := number_of_red_candles / 2
def total_age : ℕ := 85
def total_candles_so_far := number_of_red_candles + number_of_yellow_candles + number_of_green_candles
def number_of_blue_candles := total_age - total_candles_so_far

theorem birthday_candles :
  number_of_yellow_candles = 9 ∧
  number_of_blue_candles = 21 ∧
  (number_of_red_candles + number_of_yellow_candles + number_of_green_candles + number_of_blue_candles) = total_age :=
by
  sorry

end NUMINAMATH_GPT_birthday_candles_l2220_222012


namespace NUMINAMATH_GPT_nancy_first_counted_l2220_222085

theorem nancy_first_counted (x : ℤ) (h : (x + 12 + 1 + 12 + 7 + 3 + 8) / 6 = 7) : x = -1 := 
by 
  sorry

end NUMINAMATH_GPT_nancy_first_counted_l2220_222085


namespace NUMINAMATH_GPT_find_a2_l2220_222072

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_a2_l2220_222072


namespace NUMINAMATH_GPT_max_term_of_sequence_l2220_222019

noncomputable def a_n (n : ℕ) : ℚ := (n^2 : ℚ) / (2^n : ℚ)

theorem max_term_of_sequence :
  ∃ n : ℕ, (∀ m : ℕ, a_n n ≥ a_n m) ∧ a_n n = 9 / 8 :=
sorry

end NUMINAMATH_GPT_max_term_of_sequence_l2220_222019


namespace NUMINAMATH_GPT_prime_arithmetic_progression_difference_divisible_by_6_l2220_222002

theorem prime_arithmetic_progression_difference_divisible_by_6
    (p d : ℕ) (h₀ : Prime p) (h₁ : Prime (p - d)) (h₂ : Prime (p + d))
    (p_neq_3 : p ≠ 3) :
    ∃ (k : ℕ), d = 6 * k := by
  sorry

end NUMINAMATH_GPT_prime_arithmetic_progression_difference_divisible_by_6_l2220_222002


namespace NUMINAMATH_GPT_joe_money_left_l2220_222064

theorem joe_money_left
  (joe_savings : ℕ := 6000)
  (flight_cost : ℕ := 1200)
  (hotel_cost : ℕ := 800)
  (food_cost : ℕ := 3000) :
  joe_savings - (flight_cost + hotel_cost + food_cost) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_joe_money_left_l2220_222064


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2220_222061

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 3) → (x ≥ 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2220_222061


namespace NUMINAMATH_GPT_certain_number_value_l2220_222032

theorem certain_number_value :
  let D := 20
  let S := 55
  3 * D - 5 + (D - S) = 15 :=
by
  -- Definitions for D and S
  let D := 20
  let S := 55
  -- The main assertion
  show 3 * D - 5 + (D - S) = 15
  sorry

end NUMINAMATH_GPT_certain_number_value_l2220_222032


namespace NUMINAMATH_GPT_JackBuckets_l2220_222001

theorem JackBuckets (tank_capacity buckets_per_trip_jill trips_jill time_ratio trip_buckets_jack : ℕ) :
  tank_capacity = 600 → buckets_per_trip_jill = 5 → trips_jill = 30 →
  time_ratio = 3 / 2 → trip_buckets_jack = 2 :=
  sorry

end NUMINAMATH_GPT_JackBuckets_l2220_222001


namespace NUMINAMATH_GPT_distinct_integers_sum_l2220_222047

theorem distinct_integers_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 357) : a + b + c + d = 28 :=
by
  sorry

end NUMINAMATH_GPT_distinct_integers_sum_l2220_222047


namespace NUMINAMATH_GPT_factory_produces_correct_number_of_candies_l2220_222062

-- Definitions of the given conditions
def candies_per_hour : ℕ := 50
def hours_per_day : ℕ := 10
def days_to_complete_order : ℕ := 8

-- The theorem we want to prove
theorem factory_produces_correct_number_of_candies :
  days_to_complete_order * hours_per_day * candies_per_hour = 4000 :=
by 
  sorry

end NUMINAMATH_GPT_factory_produces_correct_number_of_candies_l2220_222062


namespace NUMINAMATH_GPT_solve_system_l2220_222081

theorem solve_system (x y : ℝ) :
  (x + 3*y + 3*x*y = -1) ∧ (x^2*y + 3*x*y^2 = -4) →
  (x = -3 ∧ y = -1/3) ∨ (x = -1 ∧ y = -1) ∨ (x = -1 ∧ y = 4/3) ∨ (x = 4 ∧ y = -1/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2220_222081


namespace NUMINAMATH_GPT_find_weight_of_A_l2220_222014

theorem find_weight_of_A 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 5) 
  (h4 : (B + C + D + E) / 4 = 79) 
  : A = 77 := 
sorry

end NUMINAMATH_GPT_find_weight_of_A_l2220_222014


namespace NUMINAMATH_GPT_jony_speed_l2220_222010

theorem jony_speed :
  let start_block := 10
  let end_block := 90
  let turn_around_block := 70
  let block_length := 40 -- meters
  let start_time := 0 -- 07:00 in minutes from the start of his walk
  let end_time := 40 -- 07:40 in minutes from the start of his walk
  let total_blocks_walked := (end_block - start_block) + (end_block - turn_around_block)
  let total_distance := total_blocks_walked * block_length
  let total_time := end_time - start_time
  total_distance / total_time = 100 :=
by
  sorry

end NUMINAMATH_GPT_jony_speed_l2220_222010


namespace NUMINAMATH_GPT_fourth_term_of_gp_is_negative_10_point_42_l2220_222069

theorem fourth_term_of_gp_is_negative_10_point_42 (x : ℝ) 
  (h : ∃ r : ℝ, r * (5 * x + 5) = (3 * x + 3) * ((3 * x + 3) / x)) :
  r * (5 * x + 5) * ((3 * x + 3) / x) * ((3 * x + 3) / x) = -10.42 :=
by
  sorry

end NUMINAMATH_GPT_fourth_term_of_gp_is_negative_10_point_42_l2220_222069


namespace NUMINAMATH_GPT_socks_impossible_l2220_222018

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end NUMINAMATH_GPT_socks_impossible_l2220_222018


namespace NUMINAMATH_GPT_geometric_progression_terms_l2220_222009

theorem geometric_progression_terms (b1 b2 bn : ℕ) (q n : ℕ)
  (h1 : b1 = 3) 
  (h2 : b2 = 12)
  (h3 : bn = 3072)
  (h4 : b2 = b1 * q)
  (h5 : bn = b1 * q^(n-1)) : 
  n = 6 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_progression_terms_l2220_222009


namespace NUMINAMATH_GPT_brad_zip_code_l2220_222040

theorem brad_zip_code (a b c d e : ℕ) 
  (h1 : a = b) 
  (h2 : c = 0) 
  (h3 : d = 2 * a) 
  (h4 : d + e = 8) 
  (h5 : a + b + c + d + e = 10) : 
  (a, b, c, d, e) = (1, 1, 0, 2, 6) :=
by 
  -- Proof omitted on purpose
  sorry

end NUMINAMATH_GPT_brad_zip_code_l2220_222040


namespace NUMINAMATH_GPT_num_undefined_values_l2220_222016

-- Condition: Denominator is given as (x^2 + 2x - 3)(x - 3)(x + 1)
def denominator (x : ℝ) : ℝ := (x^2 + 2 * x - 3) * (x - 3) * (x + 1)

-- The Lean statement to prove the number of values of x for which the expression is undefined
theorem num_undefined_values : 
  ∃ (n : ℕ), (∀ x : ℝ, denominator x = 0 → (x = 1 ∨ x = -3 ∨ x = 3 ∨ x = -1)) ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_num_undefined_values_l2220_222016


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2220_222054

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2) :
  ( ( (2 * x - 1) / (x + 1) - x + 1 ) / (x - 2) / (x^2 + 2 * x + 1) ) = -2 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2220_222054


namespace NUMINAMATH_GPT_diane_honey_harvest_l2220_222086

theorem diane_honey_harvest (last_year : ℕ) (increase : ℕ) (this_year : ℕ) :
  last_year = 2479 → increase = 6085 → this_year = last_year + increase → this_year = 8564 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_diane_honey_harvest_l2220_222086


namespace NUMINAMATH_GPT_ratio_M_N_l2220_222033

theorem ratio_M_N (M Q P N : ℝ) (hM : M = 0.40 * Q) (hQ : Q = 0.25 * P) (hN : N = 0.60 * P) (hP : P ≠ 0) : 
  (M / N) = (1 / 6) := 
by 
  sorry

end NUMINAMATH_GPT_ratio_M_N_l2220_222033


namespace NUMINAMATH_GPT_largest_common_term_lt_300_l2220_222087

theorem largest_common_term_lt_300 :
  ∃ a : ℕ, a < 300 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 3 + 7 * m) ∧ ∀ b : ℕ, b < 300 → (∃ n : ℤ, b = 4 + 5 * n) → (∃ m : ℤ, b = 3 + 7 * m) → b ≤ a :=
sorry

end NUMINAMATH_GPT_largest_common_term_lt_300_l2220_222087


namespace NUMINAMATH_GPT_greatest_consecutive_integers_sum_36_l2220_222080

-- Definition of the sum of N consecutive integers starting from a
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Problem statement in Lean 4
theorem greatest_consecutive_integers_sum_36 (N : ℤ) (h : sum_consecutive_integers (-35) 72 = 36) : N = 72 := by
  sorry

end NUMINAMATH_GPT_greatest_consecutive_integers_sum_36_l2220_222080


namespace NUMINAMATH_GPT_part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l2220_222003

-- Part (Ⅰ)
theorem part1_coordinates_of_P_if_AB_perp_PB :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (7, 0)) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_coordinates_of_P_area_ABP_10 :
  ∃ P : ℝ × ℝ, P.2 = 0 ∧ (P = (9, 0) ∨ P = (-11, 0)) :=
by
  sorry

end NUMINAMATH_GPT_part1_coordinates_of_P_if_AB_perp_PB_part2_coordinates_of_P_area_ABP_10_l2220_222003


namespace NUMINAMATH_GPT_find_distinct_natural_numbers_l2220_222028

theorem find_distinct_natural_numbers :
  ∃ (x y : ℕ), x ≥ 10 ∧ y ≠ 1 ∧
  (x * y + x) + (x * y - x) + (x * y * x) + (x * y / x) = 576 :=
by
  sorry

end NUMINAMATH_GPT_find_distinct_natural_numbers_l2220_222028


namespace NUMINAMATH_GPT_thrown_away_oranges_l2220_222020

theorem thrown_away_oranges (x : ℕ) (h : 40 - x + 7 = 10) : x = 37 :=
by sorry

end NUMINAMATH_GPT_thrown_away_oranges_l2220_222020


namespace NUMINAMATH_GPT_perfect_square_solution_l2220_222077

theorem perfect_square_solution (x : ℤ) : 
  ∃ k : ℤ, x^2 - 14 * x - 256 = k^2 ↔ x = 15 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_solution_l2220_222077


namespace NUMINAMATH_GPT_find_divisor_l2220_222004

theorem find_divisor (x : ℕ) (h : 172 = 10 * x + 2) : x = 17 :=
sorry

end NUMINAMATH_GPT_find_divisor_l2220_222004


namespace NUMINAMATH_GPT_arithmetic_sequence_y_value_l2220_222099

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_y_value_l2220_222099


namespace NUMINAMATH_GPT_simplify_radical_1_simplify_radical_2_find_value_of_a_l2220_222093

-- Problem 1
theorem simplify_radical_1 : 7 + 2 * (Real.sqrt 10) = (Real.sqrt 2 + Real.sqrt 5) ^ 2 := 
by sorry

-- Problem 2
theorem simplify_radical_2 : (Real.sqrt (11 - 6 * (Real.sqrt 2))) = 3 - Real.sqrt 2 := 
by sorry

-- Problem 3
theorem find_value_of_a (a m n : ℕ) (h : a + 2 * Real.sqrt 21 = (Real.sqrt m + Real.sqrt n) ^ 2) : 
  a = 10 ∨ a = 22 := 
by sorry

end NUMINAMATH_GPT_simplify_radical_1_simplify_radical_2_find_value_of_a_l2220_222093


namespace NUMINAMATH_GPT_square_angle_l2220_222084

theorem square_angle (PQ QR : ℝ) (x : ℝ) (PQR_is_square : true)
  (angle_sum_of_triangle : ∀ a b c : ℝ, a + b + c = 180)
  (right_angle : ∀ a, a = 90) :
  x = 45 :=
by
  -- We start with the properties of the square (implicitly given by the conditions)
  -- Now use the conditions and provided values to conclude the proof
  sorry

end NUMINAMATH_GPT_square_angle_l2220_222084


namespace NUMINAMATH_GPT_least_positive_integer_for_multiple_of_five_l2220_222071

theorem least_positive_integer_for_multiple_of_five (x : ℕ) (h_pos : 0 < x) (h_multiple : (625 + x) % 5 = 0) : x = 5 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_for_multiple_of_five_l2220_222071


namespace NUMINAMATH_GPT_canonical_form_lines_l2220_222055

theorem canonical_form_lines (x y z : ℝ) :
  (2 * x - y + 3 * z - 1 = 0) →
  (5 * x + 4 * y - z - 7 = 0) →
  (∃ (k : ℝ), x = -11 * k ∧ y = 17 * k + 2 ∧ z = 13 * k + 1) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_canonical_form_lines_l2220_222055


namespace NUMINAMATH_GPT_range_of_a_l2220_222079

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2220_222079


namespace NUMINAMATH_GPT_exponentiation_condition_l2220_222017

theorem exponentiation_condition (a b : ℝ) (h0 : a > 0) (h1 : a ≠ 1) : 
  (a ^ b > 1 ↔ (a - 1) * b > 0) :=
sorry

end NUMINAMATH_GPT_exponentiation_condition_l2220_222017


namespace NUMINAMATH_GPT_book_pairs_count_l2220_222067

theorem book_pairs_count :
  let mystery_books := 4
  let science_fiction_books := 4
  let historical_books := 4
  (mystery_books + science_fiction_books + historical_books) = 12 ∧ 
  (mystery_books = 4 ∧ science_fiction_books = 4 ∧ historical_books = 4) →
  let genres := 3
  ∃ pairs, pairs = 48 :=
by
  sorry

end NUMINAMATH_GPT_book_pairs_count_l2220_222067


namespace NUMINAMATH_GPT_min_a_plus_b_l2220_222050

-- Given conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Equation of line L passing through point (4,1) with intercepts a and b
def line_eq (a b : ℝ) : Prop := (4 / a) + (1 / b) = 1

-- Proof statement
theorem min_a_plus_b (h : line_eq a b) : a + b ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_a_plus_b_l2220_222050


namespace NUMINAMATH_GPT_man_savings_l2220_222045

theorem man_savings (I : ℝ) (S : ℝ) (h1 : S = 0.35) (h2 : 2 * (0.65 * I) = 0.65 * I + 0.70 * I) :
  S = 0.35 :=
by
  -- Introduce necessary assumptions
  let savings_first_year := S * I
  let expenditure_first_year := I - savings_first_year
  let savings_second_year := 2 * savings_first_year

  have h3 : expenditure_first_year = 0.65 * I := by sorry
  have h4 : savings_first_year = 0.35 * I := by sorry

  -- Using given condition to resolve S
  exact h1

end NUMINAMATH_GPT_man_savings_l2220_222045


namespace NUMINAMATH_GPT_percentage_of_green_ducks_l2220_222063

def total_ducks := 100
def green_ducks_smaller_pond := 9
def green_ducks_larger_pond := 22
def total_green_ducks := green_ducks_smaller_pond + green_ducks_larger_pond

theorem percentage_of_green_ducks :
  (total_green_ducks / total_ducks) * 100 = 31 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_green_ducks_l2220_222063


namespace NUMINAMATH_GPT_find_m_condition_l2220_222094

theorem find_m_condition (m : ℕ) (h : 9^4 = 3^(2*m)) : m = 4 := by
  sorry

end NUMINAMATH_GPT_find_m_condition_l2220_222094


namespace NUMINAMATH_GPT_complement_is_correct_l2220_222046

-- Define the universal set U and set M
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

-- Define the complement of M with respect to U
def complement_U (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

-- State the theorem to be proved
theorem complement_is_correct : complement_U U M = {3, 5, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_is_correct_l2220_222046


namespace NUMINAMATH_GPT_count_integers_with_zero_l2220_222097

/-- There are 740 positive integers less than or equal to 3017 that contain the digit 0. -/
theorem count_integers_with_zero (n : ℕ) (h : n ≤ 3017) : 
  (∃ k : ℕ, k ≤ 3017 ∧ ∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ k / 10 ^ d % 10 = 0) ↔ n = 740 :=
by sorry

end NUMINAMATH_GPT_count_integers_with_zero_l2220_222097


namespace NUMINAMATH_GPT_hot_water_bottles_sold_l2220_222008

theorem hot_water_bottles_sold (T H : ℕ) (h1 : 2 * T + 6 * H = 1200) (h2 : T = 7 * H) : H = 60 := 
by 
  sorry

end NUMINAMATH_GPT_hot_water_bottles_sold_l2220_222008


namespace NUMINAMATH_GPT_remainder_sum_of_six_primes_div_seventh_prime_l2220_222038

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_of_six_primes_div_seventh_prime_l2220_222038


namespace NUMINAMATH_GPT_mass_percentage_Na_in_NaClO_l2220_222056

theorem mass_percentage_Na_in_NaClO :
  let mass_Na : ℝ := 22.99
  let mass_Cl : ℝ := 35.45
  let mass_O : ℝ := 16.00
  let mass_NaClO : ℝ := mass_Na + mass_Cl + mass_O
  (mass_Na / mass_NaClO) * 100 = 30.89 := by
sorry

end NUMINAMATH_GPT_mass_percentage_Na_in_NaClO_l2220_222056


namespace NUMINAMATH_GPT_area_of_field_l2220_222000

-- Define the conditions: length, width, and total fencing
def length : ℕ := 40
def fencing : ℕ := 74

-- Define the property being proved: the area of the field
theorem area_of_field : ∃ (width : ℕ), 2 * width + length = fencing ∧ length * width = 680 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_area_of_field_l2220_222000


namespace NUMINAMATH_GPT_base_conversion_l2220_222026

theorem base_conversion (k : ℕ) : (5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k + 4) → k = 7 :=
by 
  let x := 5 * 8^2 + 2 * 8^1 + 4 * 8^0
  have h : x = 340 := by sorry
  have hk : 6 * k^2 + 6 * k + 4 = 340 := by sorry
  sorry

end NUMINAMATH_GPT_base_conversion_l2220_222026
