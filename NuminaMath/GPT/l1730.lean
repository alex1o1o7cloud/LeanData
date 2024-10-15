import Mathlib

namespace NUMINAMATH_GPT_minimum_triangle_perimeter_l1730_173025

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem minimum_triangle_perimeter (l m n : ℕ) (h1 : l > m) (h2 : m > n)
  (h3 : fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4)) 
  (h4 : fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) :
   l + m + n = 3003 := 
sorry

end NUMINAMATH_GPT_minimum_triangle_perimeter_l1730_173025


namespace NUMINAMATH_GPT_sum_seq_equals_2_pow_n_minus_1_l1730_173062

-- Define the sequences a_n and b_n with given conditions
def a (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry
def b (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry

-- Relation for a_n: 2a_{n+1} = a_n + a_{n+2}
axiom a_relation (n : ℕ) : 2 * a (n + 1) = a n + a (n + 2)

-- Inequalities for b_n
axiom b_inequality_1 (n : ℕ) : b (n + 1) - b n < 2^n + 1 / 2
axiom b_inequality_2 (n : ℕ) : b (n + 2) - b n > 3 * 2^n - 1

-- Note that b_n ∈ ℤ is implied by the definition being in ℕ

-- Prove that the sum of the first n terms of the sequence { n * b_n / a_n }
theorem sum_seq_equals_2_pow_n_minus_1 (n : ℕ) : 
  (Finset.range n).sum (λ k => k * b k / a k) = 2^n - 1 := 
sorry

end NUMINAMATH_GPT_sum_seq_equals_2_pow_n_minus_1_l1730_173062


namespace NUMINAMATH_GPT_find_divisor_value_l1730_173022

theorem find_divisor_value (x : ℝ) (h : 63 / x = 63 - 42) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_value_l1730_173022


namespace NUMINAMATH_GPT_permutations_of_six_digit_number_l1730_173096

/-- 
Theorem: The number of distinct permutations of the digits 1, 1, 3, 3, 3, 8 
to form six-digit positive integers is 60. 
-/
theorem permutations_of_six_digit_number : 
  (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 3)) = 60 := 
by 
  sorry

end NUMINAMATH_GPT_permutations_of_six_digit_number_l1730_173096


namespace NUMINAMATH_GPT_set_complement_intersection_l1730_173085

theorem set_complement_intersection
  (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)
  (hU : U = {0, 1, 2, 3, 4})
  (hM : M = {0, 1, 2})
  (hN : N = {2, 3}) :
  ((U \ M) ∩ N) = {3} :=
  by sorry

end NUMINAMATH_GPT_set_complement_intersection_l1730_173085


namespace NUMINAMATH_GPT_range_of_omega_l1730_173018

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f ω x = 0 → 
      (∃ x₁ x₂, x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 
        0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f ω x₁ = 0 ∧ f ω x₂ = 0)) ↔ 2 ≤ ω ∧ ω < 4 :=
sorry

end NUMINAMATH_GPT_range_of_omega_l1730_173018


namespace NUMINAMATH_GPT_total_strawberry_weight_l1730_173090

def MarcosStrawberries : ℕ := 3
def DadsStrawberries : ℕ := 17

theorem total_strawberry_weight : MarcosStrawberries + DadsStrawberries = 20 := by
  sorry

end NUMINAMATH_GPT_total_strawberry_weight_l1730_173090


namespace NUMINAMATH_GPT_g_at_3_l1730_173048

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_at_3 : g 3 = 79 := 
by 
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_g_at_3_l1730_173048


namespace NUMINAMATH_GPT_sara_staircase_steps_l1730_173060

-- Define the problem statement and conditions
theorem sara_staircase_steps (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → n = 12 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_sara_staircase_steps_l1730_173060


namespace NUMINAMATH_GPT_pyramid_surface_area_l1730_173024

noncomputable def total_surface_area (a : ℝ) : ℝ :=
  a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7) / 2

theorem pyramid_surface_area (a : ℝ) :
  let hexagon_base_area := 3 * a^2 * Real.sqrt 3 / 2
  let triangle_area_1 := a^2 / 2
  let triangle_area_2 := a^2
  let triangle_area_3 := a^2 * Real.sqrt 7 / 4
  let lateral_area := 2 * (triangle_area_1 + triangle_area_2 + triangle_area_3)
  total_surface_area a = hexagon_base_area + lateral_area := 
sorry

end NUMINAMATH_GPT_pyramid_surface_area_l1730_173024


namespace NUMINAMATH_GPT_animal_shelter_l1730_173066

theorem animal_shelter : ∃ D C : ℕ, (D = 75) ∧ (D / C = 15 / 7) ∧ (D / (C + 20) = 15 / 11) :=
by
  sorry

end NUMINAMATH_GPT_animal_shelter_l1730_173066


namespace NUMINAMATH_GPT_parallel_lines_condition_l1730_173052

theorem parallel_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * m * x + y + 6 = 0 → (m - 3) * x - y + 7 = 0) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l1730_173052


namespace NUMINAMATH_GPT_parade_team_people_count_min_l1730_173092

theorem parade_team_people_count_min (n : ℕ) :
  n ≥ 1000 ∧ n % 5 = 0 ∧ n % 4 = 3 ∧ n % 3 = 2 ∧ n % 2 = 1 → n = 1045 :=
by
  sorry

end NUMINAMATH_GPT_parade_team_people_count_min_l1730_173092


namespace NUMINAMATH_GPT_sum_of_coordinates_D_l1730_173029

structure Point where
  x : ℝ
  y : ℝ

def is_midpoint (M C D : Point) : Prop :=
  M = ⟨(C.x + D.x) / 2, (C.y + D.y) / 2⟩

def sum_of_coordinates (P : Point) : ℝ :=
  P.x + P.y

theorem sum_of_coordinates_D :
  ∀ (C M : Point), C = ⟨1/2, 3/2⟩ → M = ⟨2, 5⟩ →
  ∃ D : Point, is_midpoint M C D ∧ sum_of_coordinates D = 12 :=
by
  intros C M hC hM
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_D_l1730_173029


namespace NUMINAMATH_GPT_max_tulips_l1730_173000

theorem max_tulips (y r : ℕ) (h1 : (y + r) % 2 = 1) (h2 : r = y + 1 ∨ y = r + 1) (h3 : 50 * y + 31 * r ≤ 600) : y + r = 15 :=
by
  sorry

end NUMINAMATH_GPT_max_tulips_l1730_173000


namespace NUMINAMATH_GPT_rectangle_length_l1730_173001

theorem rectangle_length (side_of_square : ℕ) (width_of_rectangle : ℕ) (same_wire_length : ℕ) 
(side_eq : side_of_square = 12) (width_eq : width_of_rectangle = 6) 
(square_perimeter : same_wire_length = 4 * side_of_square) :
  ∃ (length_of_rectangle : ℕ), 2 * (length_of_rectangle + width_of_rectangle) = same_wire_length ∧ length_of_rectangle = 18 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1730_173001


namespace NUMINAMATH_GPT_apples_sold_fresh_l1730_173078

-- Definitions per problem conditions
def total_production : Float := 8.0
def initial_percentage_mixed : Float := 0.30
def percentage_increase_per_million : Float := 0.05
def percentage_for_apple_juice : Float := 0.60
def percentage_sold_fresh : Float := 0.40

-- We need to prove that given the conditions, the amount of apples sold fresh is 2.24 million tons
theorem apples_sold_fresh :
  ( (total_production - (initial_percentage_mixed * total_production)) * percentage_sold_fresh = 2.24 ) :=
by
  sorry

end NUMINAMATH_GPT_apples_sold_fresh_l1730_173078


namespace NUMINAMATH_GPT_square_area_from_hexagon_l1730_173047

theorem square_area_from_hexagon (hex_side length square_side : ℝ) (h1 : hex_side = 4) (h2 : length = 6 * hex_side)
  (h3 : square_side = length / 4) : square_side ^ 2 = 36 :=
by 
  sorry

end NUMINAMATH_GPT_square_area_from_hexagon_l1730_173047


namespace NUMINAMATH_GPT_add_to_fraction_l1730_173038

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_add_to_fraction_l1730_173038


namespace NUMINAMATH_GPT_complex_number_in_second_quadrant_l1730_173004

theorem complex_number_in_second_quadrant 
  (a b : ℝ) 
  (h : ¬ (a ≥ 0 ∨ b ≤ 0)) : 
  (a < 0 ∧ b > 0) :=
sorry

end NUMINAMATH_GPT_complex_number_in_second_quadrant_l1730_173004


namespace NUMINAMATH_GPT_smaller_balloon_radius_is_correct_l1730_173091

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end NUMINAMATH_GPT_smaller_balloon_radius_is_correct_l1730_173091


namespace NUMINAMATH_GPT_additional_sugar_is_correct_l1730_173071

def sugar_needed : ℝ := 450
def sugar_in_house : ℝ := 287
def sugar_in_basement_kg : ℝ := 50
def kg_to_lbs : ℝ := 2.20462

def sugar_in_basement : ℝ := sugar_in_basement_kg * kg_to_lbs
def total_sugar : ℝ := sugar_in_house + sugar_in_basement
def additional_sugar_needed : ℝ := sugar_needed - total_sugar

theorem additional_sugar_is_correct : additional_sugar_needed = 52.769 := by
  sorry

end NUMINAMATH_GPT_additional_sugar_is_correct_l1730_173071


namespace NUMINAMATH_GPT_rectangle_width_l1730_173036

-- Conditions
def length (w : Real) : Real := 4 * w
def area (w : Real) : Real := w * length w

-- Theorem stating that the width of the rectangle is 5 inches if the area is 100 square inches
theorem rectangle_width (h : area w = 100) : w = 5 :=
sorry

end NUMINAMATH_GPT_rectangle_width_l1730_173036


namespace NUMINAMATH_GPT_unique_common_tangent_l1730_173032

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp (x + 1)

theorem unique_common_tangent (a : ℝ) (h : a > 0) : 
  (∃ k x₁ x₂, k = 2 * x₁ ∧ k = a * Real.exp (x₂ + 1) ∧ k = (g a x₂ - f x₁) / (x₂ - x₁)) →
  a = 4 / Real.exp 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_common_tangent_l1730_173032


namespace NUMINAMATH_GPT_distinct_nonzero_digits_sum_l1730_173083

theorem distinct_nonzero_digits_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) 
  (h7 : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*a + c + 100*b + 10*c + a + 100*c + 10*a + b + 100*c + 10*b + a = 1776) : 
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 1 ∧ b = 3 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 3) ∨ (a = 1 ∧ b = 5 ∧ c = 2) ∨ (a = 2 ∧ b = 1 ∧ c = 5) ∨
  (a = 2 ∧ b = 5 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 4) ∨ (a = 3 ∧ b = 4 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 3 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 2) ∨ (a = 5 ∧ b = 2 ∧ c = 1) :=
sorry

end NUMINAMATH_GPT_distinct_nonzero_digits_sum_l1730_173083


namespace NUMINAMATH_GPT_cone_volume_calc_l1730_173087

noncomputable def cone_volume (diameter slant_height: ℝ) : ℝ :=
  let r := diameter / 2
  let h := Real.sqrt (slant_height^2 - r^2)
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_calc :
  cone_volume 12 10 = 96 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_calc_l1730_173087


namespace NUMINAMATH_GPT_forming_n_and_m_l1730_173094

def is_created_by_inserting_digit (n: ℕ) (base: ℕ): Prop :=
  ∃ d1 d2 d3 d: ℕ, n = d1 * 1000 + d * 100 + d2 * 10 + d3 ∧ base = d1 * 100 + d2 * 10 + d3

theorem forming_n_and_m (a b: ℕ) (base: ℕ) (sum: ℕ) 
  (h1: is_created_by_inserting_digit a base)
  (h2: is_created_by_inserting_digit b base) 
  (h3: a + b = sum):
  (a = 2195 ∧ b = 2165) 
  ∨ (a = 2185 ∧ b = 2175) 
  ∨ (a = 2215 ∧ b = 2145) 
  ∨ (a = 2165 ∧ b = 2195) 
  ∨ (a = 2175 ∧ b = 2185) 
  ∨ (a = 2145 ∧ b = 2215) := 
sorry

end NUMINAMATH_GPT_forming_n_and_m_l1730_173094


namespace NUMINAMATH_GPT_price_of_peas_l1730_173046

theorem price_of_peas
  (P : ℝ) -- price of peas per kg in rupees
  (price_soybeans : ℝ) (price_mixture : ℝ)
  (ratio_peas_soybeans : ℝ) :
  price_soybeans = 25 →
  price_mixture = 19 →
  ratio_peas_soybeans = 2 →
  P = 16 :=
by
  intros h_price_soybeans h_price_mixture h_ratio
  sorry

end NUMINAMATH_GPT_price_of_peas_l1730_173046


namespace NUMINAMATH_GPT_num_triangles_in_circle_l1730_173003

noncomputable def num_triangles (n : ℕ) : ℕ :=
  n.choose 3

theorem num_triangles_in_circle (n : ℕ) :
  num_triangles n = n.choose 3 :=
by
  sorry

end NUMINAMATH_GPT_num_triangles_in_circle_l1730_173003


namespace NUMINAMATH_GPT_company_percentage_increase_l1730_173072

theorem company_percentage_increase (employees_jan employees_dec : ℝ) (P_increase : ℝ) 
  (h_jan : employees_jan = 391.304347826087)
  (h_dec : employees_dec = 450)
  (h_P : P_increase = 15) : 
  (employees_dec - employees_jan) / employees_jan * 100 = P_increase :=
by 
  sorry

end NUMINAMATH_GPT_company_percentage_increase_l1730_173072


namespace NUMINAMATH_GPT_square_field_area_l1730_173074

theorem square_field_area (speed time perimeter : ℕ) (h1 : speed = 20) (h2 : time = 4) (h3 : perimeter = speed * time) :
  ∃ s : ℕ, perimeter = 4 * s ∧ s * s = 400 :=
by
  -- All conditions and definitions are stated, proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_square_field_area_l1730_173074


namespace NUMINAMATH_GPT_solve_quadratic_l1730_173027

theorem solve_quadratic (y : ℝ) :
  3 * y * (y - 1) = 2 * (y - 1) → y = 2 / 3 ∨ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1730_173027


namespace NUMINAMATH_GPT_students_preferring_windows_is_correct_l1730_173079

-- Define the total number of students surveyed
def total_students : ℕ := 210

-- Define the number of students preferring Mac
def students_preferring_mac : ℕ := 60

-- Define the number of students preferring both Mac and Windows equally
def students_preferring_both : ℕ := students_preferring_mac / 3

-- Define the number of students with no preference
def students_no_preference : ℕ := 90

-- Calculate the total number of students with a preference
def students_with_preference : ℕ := total_students - students_no_preference

-- Calculate the number of students preferring Windows
def students_preferring_windows : ℕ := students_with_preference - (students_preferring_mac + students_preferring_both)

-- State the theorem to prove that the number of students preferring Windows is 40
theorem students_preferring_windows_is_correct : students_preferring_windows = 40 :=
by
  -- calculations based on definitions
  unfold students_preferring_windows students_with_preference students_preferring_mac students_preferring_both students_no_preference total_students
  sorry

end NUMINAMATH_GPT_students_preferring_windows_is_correct_l1730_173079


namespace NUMINAMATH_GPT_cloud_height_l1730_173069

/--
Given:
- α : ℝ (elevation angle from the top of a tower)
- β : ℝ (depression angle seen in the lake)
- m : ℝ (height of the tower)
Prove:
- The height of the cloud hovering above the observer (h - m) is given by
 2 * m * cos β * sin α / sin (β - α)
-/
theorem cloud_height (α β m : ℝ) :
  (∃ h : ℝ, h - m = 2 * m * Real.cos β * Real.sin α / Real.sin (β - α)) :=
by
  sorry

end NUMINAMATH_GPT_cloud_height_l1730_173069


namespace NUMINAMATH_GPT_simplify_expression_l1730_173067

theorem simplify_expression (w x : ℝ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w - 2 * x - 4 * x - 6 * x - 8 * x - 10 * x + 24 = 
  45 * w - 30 * x + 24 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1730_173067


namespace NUMINAMATH_GPT_complex_number_identity_l1730_173019

theorem complex_number_identity (a b : ℝ) (i : ℂ) (h : (a + i) * (1 + i) = b * i) : a + b * i = 1 + 2 * i := 
by
  sorry

end NUMINAMATH_GPT_complex_number_identity_l1730_173019


namespace NUMINAMATH_GPT_number_of_welders_left_l1730_173061

-- Define the constants and variables
def welders_total : ℕ := 36
def days_to_complete : ℕ := 5
def rate : ℝ := 1  -- Assume the rate per welder is 1 for simplicity
def total_work : ℝ := welders_total * days_to_complete * rate

def days_after_first : ℕ := 6
def work_done_in_first_day : ℝ := welders_total * 1 * rate
def remaining_work : ℝ := total_work - work_done_in_first_day

-- Define the theorem to solve for the number of welders x that started to work on another project
theorem number_of_welders_left (x : ℕ) : (welders_total - x) * days_after_first * rate = remaining_work → x = 12 := by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_welders_left_l1730_173061


namespace NUMINAMATH_GPT_amplitude_five_phase_shift_minus_pi_over_4_l1730_173028

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos (x + (Real.pi / 4))

theorem amplitude_five : ∀ x : ℝ, 5 * Real.cos (x + (Real.pi / 4)) = f x :=
by
  sorry

theorem phase_shift_minus_pi_over_4 : ∀ x : ℝ, f x = 5 * Real.cos (x + (Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_GPT_amplitude_five_phase_shift_minus_pi_over_4_l1730_173028


namespace NUMINAMATH_GPT_pounds_added_l1730_173020

-- Definitions based on conditions
def initial_weight : ℝ := 5
def weight_increase_percent : ℝ := 1.5  -- 150% increase
def final_weight : ℝ := 28

-- Statement to prove
theorem pounds_added (w_initial w_final w_percent_added : ℝ) (h_initial: w_initial = 5) (h_final: w_final = 28)
(h_percent: w_percent_added = 1.5) :
  w_final - w_initial = 23 := 
by
  sorry

end NUMINAMATH_GPT_pounds_added_l1730_173020


namespace NUMINAMATH_GPT_smallest_number_is_21_5_l1730_173035

-- Definitions of the numbers in their respective bases
def num1 := 3 * 4^0 + 3 * 4^1
def num2 := 0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3
def num3 := 2 * 3^0 + 2 * 3^1 + 1 * 3^2
def num4 := 1 * 5^0 + 2 * 5^1

-- Statement asserting that num4 is the smallest number
theorem smallest_number_is_21_5 : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end NUMINAMATH_GPT_smallest_number_is_21_5_l1730_173035


namespace NUMINAMATH_GPT_no_such_integers_exist_l1730_173044

theorem no_such_integers_exist (x y z : ℤ) (hx : x ≠ 0) :
  ¬ (2 * x ^ 4 + 2 * x ^ 2 * y ^ 2 + y ^ 4 = z ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_exist_l1730_173044


namespace NUMINAMATH_GPT_quadratic_real_roots_condition_l1730_173042

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_condition_l1730_173042


namespace NUMINAMATH_GPT_number_of_posts_needed_l1730_173005

-- Define the conditions
def length_of_field : ℕ := 80
def width_of_field : ℕ := 60
def distance_between_posts : ℕ := 10

-- Statement to prove the number of posts needed to completely fence the field
theorem number_of_posts_needed : 
  (2 * (length_of_field / distance_between_posts + 1) + 
   2 * (width_of_field / distance_between_posts + 1) - 
   4) = 28 := 
by
  -- Skipping the proof for this theorem
  sorry

end NUMINAMATH_GPT_number_of_posts_needed_l1730_173005


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_problem3_solution_l1730_173080

noncomputable def problem1 : Real :=
  3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27

theorem problem1_solution : problem1 = 6 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

noncomputable def problem2 : Real :=
  (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12)

theorem problem2_solution : problem2 = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by
  sorry

noncomputable def problem3 : Real :=
  (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6) ^ 2

theorem problem3_solution : problem3 = 3 + 2 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_problem3_solution_l1730_173080


namespace NUMINAMATH_GPT_num_brownies_correct_l1730_173006

-- Define the conditions (pan dimensions and brownie piece dimensions)
def pan_width : ℕ := 24
def pan_length : ℕ := 15
def piece_width : ℕ := 3
def piece_length : ℕ := 2

-- Define the area calculations for the pan and each piece
def pan_area : ℕ := pan_width * pan_length
def piece_area : ℕ := piece_width * piece_length

-- Define the problem statement to prove the number of brownies
def number_of_brownies : ℕ := pan_area / piece_area

-- The statement we need to prove
theorem num_brownies_correct : number_of_brownies = 60 :=
by
  sorry

end NUMINAMATH_GPT_num_brownies_correct_l1730_173006


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l1730_173012

theorem revenue_from_full_price_tickets (f h p : ℕ) (h1 : f + h = 160) (h2 : f * p + h * (p / 2) = 2400) : f * p = 1600 :=
by
  sorry

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l1730_173012


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1730_173063

variable (x : ℝ) -- Speed of the boat in still water
variable (r : ℝ) -- Rate of the stream
variable (d : ℝ) -- Distance covered downstream
variable (t : ℝ) -- Time taken downstream

theorem boat_speed_in_still_water (h_rate : r = 5) (h_distance : d = 168) (h_time : t = 8) :
  x = 16 :=
by
  -- Substitute conditions into the equation.
  -- Calculate the effective speed downstream.
  -- Solve x from the resulting equation.
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1730_173063


namespace NUMINAMATH_GPT_ellipse_parameters_l1730_173039

theorem ellipse_parameters 
  (x y : ℝ)
  (h : 2 * x^2 + y^2 + 42 = 8 * x + 36 * y) :
  ∃ (h k : ℝ) (a b : ℝ), 
    (h = 2) ∧ (k = 18) ∧ (a = Real.sqrt 290) ∧ (b = Real.sqrt 145) ∧ 
    ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
sorry

end NUMINAMATH_GPT_ellipse_parameters_l1730_173039


namespace NUMINAMATH_GPT_rectangle_perimeter_l1730_173016

variable (x : ℝ) (y : ℝ)

-- Definitions based on conditions
def area_of_rectangle : Prop := x * (x + 5) = 500
def side_length_relation : Prop := y = x + 5

-- The theorem we want to prove
theorem rectangle_perimeter (h_area : area_of_rectangle x) (h_side_length : side_length_relation x y) : 2 * (x + y) = 90 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1730_173016


namespace NUMINAMATH_GPT_page_shoes_count_l1730_173013

theorem page_shoes_count (p_i : ℕ) (d : ℝ) (b : ℕ) (h1 : p_i = 120) (h2 : d = 0.45) (h3 : b = 15) : 
  (p_i - (d * p_i)) + b = 81 :=
by
  sorry

end NUMINAMATH_GPT_page_shoes_count_l1730_173013


namespace NUMINAMATH_GPT_algebraic_expression_value_l1730_173054

/-- Given \( x^2 - 5x - 2006 = 0 \), prove that the expression \(\frac{(x-2)^3 - (x-1)^2 + 1}{x-2}\) is equal to 2010. -/
theorem algebraic_expression_value (x : ℝ) (h: x^2 - 5 * x - 2006 = 0) :
  ( (x - 2)^3 - (x - 1)^2 + 1 ) / (x - 2) = 2010 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1730_173054


namespace NUMINAMATH_GPT_negation_of_exists_x_squared_gt_one_l1730_173075

-- Negation of the proposition
theorem negation_of_exists_x_squared_gt_one :
  ¬ (∃ x : ℝ, x^2 > 1) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_x_squared_gt_one_l1730_173075


namespace NUMINAMATH_GPT_remainder_5n_div_3_l1730_173011

theorem remainder_5n_div_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_5n_div_3_l1730_173011


namespace NUMINAMATH_GPT_polynomial_roots_l1730_173084

theorem polynomial_roots : ∀ x : ℝ, (x^3 - 4*x^2 - x + 4) * (x - 3) * (x + 2) = 0 ↔ 
  (x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 3 ∨ x = 4) :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_roots_l1730_173084


namespace NUMINAMATH_GPT_anna_interest_l1730_173026

noncomputable def interest_earned (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n - P

theorem anna_interest : interest_earned 2000 0.08 5 = 938.66 := by
  sorry

end NUMINAMATH_GPT_anna_interest_l1730_173026


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1730_173058

theorem other_root_of_quadratic (m t : ℝ) : (∀ (x : ℝ),
    (3 * x^2 - m * x - 3 = 0) → 
    (x = 1)) → 
    (1 * t = -1) := 
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1730_173058


namespace NUMINAMATH_GPT_transfer_people_eq_l1730_173068

theorem transfer_people_eq : ∃ x : ℕ, 22 + x = 2 * (26 - x) := 
by 
  -- hypothesis and equation statement
  sorry

end NUMINAMATH_GPT_transfer_people_eq_l1730_173068


namespace NUMINAMATH_GPT_lines_perpendicular_l1730_173099

-- Define the lines l1 and l2
def line1 (m x y : ℝ) := m * x + y - 1 = 0
def line2 (m x y : ℝ) := x + (m - 1) * y + 2 = 0

-- State the problem: Find the value of m such that the lines l1 and l2 are perpendicular.
theorem lines_perpendicular (m : ℝ) (h₁ : line1 m x y) (h₂ : line2 m x y) : m = 1/2 := 
sorry

end NUMINAMATH_GPT_lines_perpendicular_l1730_173099


namespace NUMINAMATH_GPT_original_cost_of_car_l1730_173055

theorem original_cost_of_car (C : ℝ)
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (h1 : repairs_cost = 14000)
  (h2 : selling_price = 72900)
  (h3 : profit_percent = 17.580645161290324)
  (h4 : profit_percent = ((selling_price - (C + repairs_cost)) / C) * 100) :
  C = 50075 := 
sorry

end NUMINAMATH_GPT_original_cost_of_car_l1730_173055


namespace NUMINAMATH_GPT_michael_brought_5000_rubber_bands_l1730_173053

noncomputable def totalRubberBands
  (small_band_count : ℕ) (large_band_count : ℕ)
  (small_ball_count : ℕ := 22) (large_ball_count : ℕ := 13)
  (rubber_bands_per_small : ℕ := 50) (rubber_bands_per_large : ℕ := 300) 
: ℕ :=
small_ball_count * rubber_bands_per_small + large_ball_count * rubber_bands_per_large

theorem michael_brought_5000_rubber_bands :
  totalRubberBands 22 13 = 5000 := by
  sorry

end NUMINAMATH_GPT_michael_brought_5000_rubber_bands_l1730_173053


namespace NUMINAMATH_GPT_tangent_line_eq_extreme_values_interval_l1730_173023

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_extreme_values_interval_l1730_173023


namespace NUMINAMATH_GPT_evaluate_exponential_operations_l1730_173031

theorem evaluate_exponential_operations (a : ℝ) :
  (2 * a^2 - a^2 ≠ 2) ∧
  (a^2 * a^4 = a^6) ∧
  ((a^2)^3 ≠ a^5) ∧
  (a^6 / a^2 ≠ a^3) := by
  sorry

end NUMINAMATH_GPT_evaluate_exponential_operations_l1730_173031


namespace NUMINAMATH_GPT_family_can_purchase_furniture_in_april_l1730_173014

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end NUMINAMATH_GPT_family_can_purchase_furniture_in_april_l1730_173014


namespace NUMINAMATH_GPT_number_of_valid_numbers_l1730_173007

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def four_digit_number_conditions : Prop :=
  (∀ N : ℕ, 7000 ≤ N ∧ N < 9000 → 
    (N % 5 = 0) →
    (∃ a b c d : ℕ, 
      N = 1000 * a + 100 * b + 10 * c + d ∧
      (a = 7 ∨ a = 8) ∧
      (d = 0 ∨ d = 5) ∧
      3 ≤ b ∧ is_prime b ∧ b < c ∧ c ≤ 7))

theorem number_of_valid_numbers : four_digit_number_conditions → 
  (∃ n : ℕ, n = 24) :=
  sorry

end NUMINAMATH_GPT_number_of_valid_numbers_l1730_173007


namespace NUMINAMATH_GPT_initial_fliers_l1730_173077

theorem initial_fliers (F : ℕ) (morning_sent afternoon_sent remaining : ℕ) :
  morning_sent = F / 5 → 
  afternoon_sent = (F - morning_sent) / 4 → 
  remaining = F - morning_sent - afternoon_sent → 
  remaining = 1800 → 
  F = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_initial_fliers_l1730_173077


namespace NUMINAMATH_GPT_lionel_initial_boxes_crackers_l1730_173017

/--
Lionel went to the grocery store and bought some boxes of Graham crackers and 15 packets of Oreos. 
To make an Oreo cheesecake, Lionel needs 2 boxes of Graham crackers and 3 packets of Oreos. 
After making the maximum number of Oreo cheesecakes he can with the ingredients he bought, 
he had 4 boxes of Graham crackers left over. 

The number of boxes of Graham crackers Lionel initially bought is 14.
-/
theorem lionel_initial_boxes_crackers (G : ℕ) (h1 : G - 4 = 10) : G = 14 := 
by sorry

end NUMINAMATH_GPT_lionel_initial_boxes_crackers_l1730_173017


namespace NUMINAMATH_GPT_determine_p_and_q_l1730_173034

noncomputable def find_p_and_q (a : ℝ) (p q : ℝ) : Prop :=
  (∀ x : ℝ, x = 1 ∨ x = -1 → (x^4 + p * x^2 + q * x + a^2 = 0))

theorem determine_p_and_q (a p q : ℝ) (h : find_p_and_q a p q) : p = -(a^2 + 1) ∧ q = 0 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_determine_p_and_q_l1730_173034


namespace NUMINAMATH_GPT_cubic_difference_l1730_173089

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : a^3 - b^3 = 353.5 := by
  sorry

end NUMINAMATH_GPT_cubic_difference_l1730_173089


namespace NUMINAMATH_GPT_attendance_calculation_l1730_173015

theorem attendance_calculation (total_students : ℕ) (attendance_rate : ℚ)
  (h1 : total_students = 120)
  (h2 : attendance_rate = 0.95) :
  total_students * attendance_rate = 114 := 
  sorry

end NUMINAMATH_GPT_attendance_calculation_l1730_173015


namespace NUMINAMATH_GPT_expression_equals_66069_l1730_173045

-- Definitions based on the conditions
def numerator : Nat := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10
def denominator : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
def expression : Rat := numerator / denominator

-- The main theorem to be proven
theorem expression_equals_66069 : expression = 66069 := by
  sorry

end NUMINAMATH_GPT_expression_equals_66069_l1730_173045


namespace NUMINAMATH_GPT_roots_cube_reciprocal_eqn_l1730_173041

variable (a b c r s : ℝ)

def quadratic_eqn (r s : ℝ) : Prop :=
  3 * a * r ^ 2 + 5 * b * r + 7 * c = 0 ∧ 
  3 * a * s ^ 2 + 5 * b * s + 7 * c = 0

theorem roots_cube_reciprocal_eqn (h : quadratic_eqn a b c r s) :
  (1 / r^3 + 1 / s^3) = (-5 * b * (25 * b ^ 2 - 63 * c) / (343 * c^3)) :=
sorry

end NUMINAMATH_GPT_roots_cube_reciprocal_eqn_l1730_173041


namespace NUMINAMATH_GPT_buffy_whiskers_l1730_173040

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end NUMINAMATH_GPT_buffy_whiskers_l1730_173040


namespace NUMINAMATH_GPT_quadrilateral_diagonals_inequality_l1730_173050

theorem quadrilateral_diagonals_inequality (a b c d e f : ℝ) :
  e^2 + f^2 ≤ b^2 + d^2 + 2 * a * c :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_diagonals_inequality_l1730_173050


namespace NUMINAMATH_GPT_intersection_A_B_l1730_173076

def set_A (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def set_B (x : ℝ) : Prop := 2 < x ∧ x < 4

theorem intersection_A_B (x : ℝ) :
  (set_A x ∧ set_B x) ↔ 2 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1730_173076


namespace NUMINAMATH_GPT_interval_monotonically_decreasing_l1730_173059

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 3)

theorem interval_monotonically_decreasing :
  ∀ x y : ℝ, 1 < x → x < 3 → 1 < y → y < 3 → x < y → f y < f x := 
by sorry

end NUMINAMATH_GPT_interval_monotonically_decreasing_l1730_173059


namespace NUMINAMATH_GPT_proportion_not_necessarily_correct_l1730_173037

theorem proportion_not_necessarily_correct
  (a b c d : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : c ≠ 0)
  (h₄ : d ≠ 0)
  (h₅ : a * d = b * c) :
  ¬ ((a + 1) / b = (c + 1) / d) :=
by 
  sorry

end NUMINAMATH_GPT_proportion_not_necessarily_correct_l1730_173037


namespace NUMINAMATH_GPT_parallel_vectors_sum_l1730_173009

variable (x y : ℝ)
variable (k : ℝ)

theorem parallel_vectors_sum :
  (k * 3 = 2) ∧ (k * x = 4) ∧ (k * y = 5) → x + y = 27 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_sum_l1730_173009


namespace NUMINAMATH_GPT_ratio_of_group_average_l1730_173064

theorem ratio_of_group_average
  (d l e : ℕ)
  (avg_group_age : ℕ := 45) 
  (avg_doctors_age : ℕ := 40) 
  (avg_lawyers_age : ℕ := 55) 
  (avg_engineers_age : ℕ := 35)
  (h : (40 * d + 55 * l + 35 * e) / (d + l + e) = avg_group_age)
  : d = 2 * l - e ∧ l = 2 * e :=
sorry

end NUMINAMATH_GPT_ratio_of_group_average_l1730_173064


namespace NUMINAMATH_GPT_profit_percentage_B_l1730_173095

theorem profit_percentage_B (cost_price_A : ℝ) (sell_price_C : ℝ) 
  (profit_A_percent : ℝ) (profit_B_percent : ℝ) 
  (cost_price_A_eq : cost_price_A = 148) 
  (sell_price_C_eq : sell_price_C = 222) 
  (profit_A_percent_eq : profit_A_percent = 0.2) :
  profit_B_percent = 0.25 := 
by
  have cost_price_B := cost_price_A * (1 + profit_A_percent)
  have profit_B := sell_price_C - cost_price_B
  have profit_B_percent := (profit_B / cost_price_B) * 100 
  sorry

end NUMINAMATH_GPT_profit_percentage_B_l1730_173095


namespace NUMINAMATH_GPT_cos_tan_quadrant_l1730_173097

theorem cos_tan_quadrant (α : ℝ) 
  (hcos : Real.cos α < 0) 
  (htan : Real.tan α > 0) : 
  (2 * π / 2 < α ∧ α < π) :=
by
  sorry

end NUMINAMATH_GPT_cos_tan_quadrant_l1730_173097


namespace NUMINAMATH_GPT_isaac_ribbon_length_l1730_173002

variable (part_length : ℝ) (total_length : ℝ := part_length * 6) (unused_length : ℝ := part_length * 2)

theorem isaac_ribbon_length
  (total_parts : ℕ := 6)
  (used_parts : ℕ := 4)
  (not_used_parts : ℕ := total_parts - used_parts)
  (not_used_length : Real := 10)
  (equal_parts : total_length / total_parts = part_length) :
  total_length = 30 := by
  sorry

end NUMINAMATH_GPT_isaac_ribbon_length_l1730_173002


namespace NUMINAMATH_GPT_crescents_area_eq_rectangle_area_l1730_173010

noncomputable def rectangle_area (a b : ℝ) : ℝ := 4 * a * b

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

noncomputable def circumscribed_circle_area (a b : ℝ) : ℝ :=
  Real.pi * (a^2 + b^2)

noncomputable def combined_area (a b : ℝ) : ℝ :=
  rectangle_area a b + 2 * (semicircle_area a) + 2 * (semicircle_area b)

theorem crescents_area_eq_rectangle_area (a b : ℝ) : 
  combined_area a b - circumscribed_circle_area a b = rectangle_area a b :=
by
  unfold combined_area
  unfold circumscribed_circle_area
  unfold rectangle_area
  unfold semicircle_area
  sorry

end NUMINAMATH_GPT_crescents_area_eq_rectangle_area_l1730_173010


namespace NUMINAMATH_GPT_distinct_cyclic_quadrilaterals_perimeter_36_l1730_173088

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end NUMINAMATH_GPT_distinct_cyclic_quadrilaterals_perimeter_36_l1730_173088


namespace NUMINAMATH_GPT_product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l1730_173086

theorem product_div_sum_eq_5 (x : ℤ) (h : (x^3 - x) / (3 * x) = 5) : x = 4 := by
  sorry

theorem quotient_integer_condition (x : ℤ) : ((∃ k : ℤ, x = 3 * k + 1) ∨ (∃ k : ℤ, x = 3 * k - 1)) ↔ ∃ q : ℤ, (x^3 - x) / (3 * x) = q := by
  sorry

theorem next_consecutive_set (x : ℤ) (h : x = 4) : x - 1 = 3 ∧ x = 4 ∧ x + 1 = 5 := by
  sorry

end NUMINAMATH_GPT_product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l1730_173086


namespace NUMINAMATH_GPT_time_to_school_building_l1730_173030

theorem time_to_school_building 
  (total_time : ℕ := 30) 
  (time_to_gate : ℕ := 15) 
  (time_to_room : ℕ := 9)
  (remaining_time := total_time - time_to_gate - time_to_room) : 
  remaining_time = 6 :=
by
  sorry

end NUMINAMATH_GPT_time_to_school_building_l1730_173030


namespace NUMINAMATH_GPT_rival_awards_l1730_173049

theorem rival_awards (S J R : ℕ) (h1 : J = 3 * S) (h2 : S = 4) (h3 : R = 2 * J) : R = 24 := 
by sorry

end NUMINAMATH_GPT_rival_awards_l1730_173049


namespace NUMINAMATH_GPT_correct_statement_d_l1730_173008

theorem correct_statement_d (x : ℝ) : 2 * (x + 1) = x + 7 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_d_l1730_173008


namespace NUMINAMATH_GPT_three_tenths_of_number_l1730_173093

theorem three_tenths_of_number (N : ℝ) (h : (1/3) * (1/4) * N = 15) : (3/10) * N = 54 :=
sorry

end NUMINAMATH_GPT_three_tenths_of_number_l1730_173093


namespace NUMINAMATH_GPT_richmond_tickets_l1730_173021

theorem richmond_tickets (total_tickets : ℕ) (second_half_tickets : ℕ) (first_half_tickets : ℕ) :
  total_tickets = 9570 →
  second_half_tickets = 5703 →
  first_half_tickets = total_tickets - second_half_tickets →
  first_half_tickets = 3867 := by
  sorry

end NUMINAMATH_GPT_richmond_tickets_l1730_173021


namespace NUMINAMATH_GPT_max_n_for_factorization_l1730_173098

theorem max_n_for_factorization (A B n : ℤ) (AB_cond : A * B = 48) (n_cond : n = 5 * B + A) :
  n ≤ 241 :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_factorization_l1730_173098


namespace NUMINAMATH_GPT_cookies_per_person_l1730_173082

/-- Brenda's mother made cookies for 5 people. She prepared 35 cookies, 
    and each of them had the same number of cookies. 
    We aim to prove that each person had 7 cookies. --/
theorem cookies_per_person (total_cookies : ℕ) (number_of_people : ℕ) 
  (h1 : total_cookies = 35) (h2 : number_of_people = 5) : total_cookies / number_of_people = 7 := 
by
  sorry

end NUMINAMATH_GPT_cookies_per_person_l1730_173082


namespace NUMINAMATH_GPT_number_of_female_students_school_l1730_173051

theorem number_of_female_students_school (T S G_s B_s B G : ℕ) (h1 : T = 1600)
    (h2 : S = 200) (h3 : G_s = B_s - 10) (h4 : G_s + B_s = 200) (h5 : B_s = 105) (h6 : G_s = 95) (h7 : B + G = 1600) : 
    G = 760 :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_students_school_l1730_173051


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l1730_173056

theorem solve_system_of_inequalities (x y : ℤ) :
  (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := 
by { sorry }

end NUMINAMATH_GPT_solve_system_of_inequalities_l1730_173056


namespace NUMINAMATH_GPT_inverse_of_3_mod_199_l1730_173033

theorem inverse_of_3_mod_199 : (3 * 133) % 199 = 1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_3_mod_199_l1730_173033


namespace NUMINAMATH_GPT_value_makes_expression_undefined_l1730_173043

theorem value_makes_expression_undefined (a : ℝ) : 
    (a^2 - 9 * a + 20 = 0) ↔ (a = 4 ∨ a = 5) :=
by
  sorry

end NUMINAMATH_GPT_value_makes_expression_undefined_l1730_173043


namespace NUMINAMATH_GPT_piggy_bank_total_l1730_173073

def amount_added_in_january: ℕ := 19
def amount_added_in_february: ℕ := 19
def amount_added_in_march: ℕ := 8

theorem piggy_bank_total:
  amount_added_in_january + amount_added_in_february + amount_added_in_march = 46 := by
  sorry

end NUMINAMATH_GPT_piggy_bank_total_l1730_173073


namespace NUMINAMATH_GPT_arrangement_count_example_l1730_173065

theorem arrangement_count_example 
  (teachers : Finset String) 
  (students : Finset String) 
  (locations : Finset String) 
  (h_teachers : teachers.card = 2) 
  (h_students : students.card = 4) 
  (h_locations : locations.card = 2)
  : ∃ n : ℕ, n = 12 := 
sorry

end NUMINAMATH_GPT_arrangement_count_example_l1730_173065


namespace NUMINAMATH_GPT_expression_value_eq_3084_l1730_173070

theorem expression_value_eq_3084 (x : ℤ) (hx : x = -3007) :
  (abs (abs (Real.sqrt (abs x - x) - x) - x) - Real.sqrt (abs (x - x^2)) = 3084) :=
by
  sorry

end NUMINAMATH_GPT_expression_value_eq_3084_l1730_173070


namespace NUMINAMATH_GPT_crayons_erasers_difference_l1730_173057

theorem crayons_erasers_difference
  (initial_erasers : ℕ) (initial_crayons : ℕ) (final_crayons : ℕ)
  (no_eraser_lost : initial_erasers = 457)
  (initial_crayons_condition : initial_crayons = 617)
  (final_crayons_condition : final_crayons = 523) :
  final_crayons - initial_erasers = 66 :=
by
  -- These would be assumptions in the proof; be aware that 'sorry' is used to skip the proof details.
  sorry

end NUMINAMATH_GPT_crayons_erasers_difference_l1730_173057


namespace NUMINAMATH_GPT_simplify_trig_expression_l1730_173081

open Real

theorem simplify_trig_expression (theta : ℝ) (h : 0 < theta ∧ theta < π / 4) :
  sqrt (1 - 2 * sin (π + theta) * sin (3 * π / 2 - theta)) = cos theta - sin theta :=
sorry

end NUMINAMATH_GPT_simplify_trig_expression_l1730_173081
