import Mathlib

namespace NUMINAMATH_GPT_negation_of_existential_l672_67296

theorem negation_of_existential:
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 = 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l672_67296


namespace NUMINAMATH_GPT_sports_popularity_order_l672_67217

theorem sports_popularity_order :
  let soccer := (13 : ℚ) / 40
  let baseball := (9 : ℚ) / 30
  let basketball := (7 : ℚ) / 20
  let volleyball := (3 : ℚ) / 10
  basketball > soccer ∧ soccer > baseball ∧ baseball = volleyball :=
by
  sorry

end NUMINAMATH_GPT_sports_popularity_order_l672_67217


namespace NUMINAMATH_GPT_sand_loss_l672_67299

variable (initial_sand : ℝ) (final_sand : ℝ)

theorem sand_loss (h1 : initial_sand = 4.1) (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
  -- With the given conditions we'll prove this theorem
  sorry

end NUMINAMATH_GPT_sand_loss_l672_67299


namespace NUMINAMATH_GPT_find_amount_l672_67221

theorem find_amount (x : ℝ) (h1 : 0.25 * x = 0.15 * 1500 - 30) (h2 : x = 780) : 30 = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_amount_l672_67221


namespace NUMINAMATH_GPT_ball_hit_ground_in_time_l672_67278

theorem ball_hit_ground_in_time :
  ∃ t : ℝ, t ≥ 0 ∧ -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 :=
by sorry

end NUMINAMATH_GPT_ball_hit_ground_in_time_l672_67278


namespace NUMINAMATH_GPT_add_decimals_l672_67235

theorem add_decimals : 5.763 + 2.489 = 8.252 := 
by
  sorry

end NUMINAMATH_GPT_add_decimals_l672_67235


namespace NUMINAMATH_GPT_eggs_remainder_and_full_cartons_l672_67245

def abigail_eggs := 48
def beatrice_eggs := 63
def carson_eggs := 27
def carton_size := 15

theorem eggs_remainder_and_full_cartons :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  ∃ (full_cartons left_over : ℕ),
    total_eggs = full_cartons * carton_size + left_over ∧
    left_over = 3 ∧
    full_cartons = 9 :=
by
  sorry

end NUMINAMATH_GPT_eggs_remainder_and_full_cartons_l672_67245


namespace NUMINAMATH_GPT_age_ratio_in_9_years_l672_67279

-- Initial age definitions for Mike and Sam
def ages (m s : ℕ) : Prop :=
  (m - 5 = 2 * (s - 5)) ∧ (m - 12 = 3 * (s - 12))

-- Proof that in 9 years the ratio of their ages will be 3:2
theorem age_ratio_in_9_years (m s x : ℕ) (h_ages : ages m s) :
  (m + x) * 2 = 3 * (s + x) ↔ x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_ratio_in_9_years_l672_67279


namespace NUMINAMATH_GPT_range_of_a_l672_67252

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l672_67252


namespace NUMINAMATH_GPT_project_hours_l672_67281

variable (K : ℕ)

theorem project_hours 
    (h_total : K + 2 * K + 3 * K + K / 2 = 180)
    (h_k_nearest : K = 28) :
    3 * K - K = 56 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_project_hours_l672_67281


namespace NUMINAMATH_GPT_employees_six_years_or_more_percentage_l672_67260

theorem employees_six_years_or_more_percentage 
  (Y : ℕ)
  (Total : ℝ := (3 * Y:ℝ) + (4 * Y:ℝ) + (7 * Y:ℝ) - (2 * Y:ℝ) + (6 * Y:ℝ) + (1 * Y:ℝ))
  (Employees_Six_Years : ℝ := (6 * Y:ℝ) + (1 * Y:ℝ))
  : Employees_Six_Years / Total * 100 = 36.84 :=
by
  sorry

end NUMINAMATH_GPT_employees_six_years_or_more_percentage_l672_67260


namespace NUMINAMATH_GPT_intersecting_lines_product_l672_67276

theorem intersecting_lines_product 
  (a b : ℝ)
  (T : Set (ℝ × ℝ)) (S : Set (ℝ × ℝ))
  (hT : T = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ a * x + y - 3 = 0})
  (hS : S = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x - y - b = 0})
  (h_intersect : (2, 1) ∈ T) (h_intersect_S : (2, 1) ∈ S) :
  a * b = 1 := 
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_product_l672_67276


namespace NUMINAMATH_GPT_sum_of_digits_of_x_l672_67263

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_x (x : ℕ) (h1 : 100 ≤ x) (h2 : x ≤ 949)
  (h3 : is_palindrome x) (h4 : is_palindrome (x + 50)) :
  sum_of_digits x = 19 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_x_l672_67263


namespace NUMINAMATH_GPT_find_speed_grocery_to_gym_l672_67253

variables (v : ℝ) (speed_grocery_to_gym : ℝ)
variables (d_home_to_grocery : ℝ) (d_grocery_to_gym : ℝ)
variables (time_diff : ℝ)

def problem_conditions : Prop :=
  d_home_to_grocery = 840 ∧
  d_grocery_to_gym = 480 ∧
  time_diff = 40 ∧
  speed_grocery_to_gym = 2 * v

def correct_answer : Prop :=
  speed_grocery_to_gym = 30

theorem find_speed_grocery_to_gym :
  problem_conditions v speed_grocery_to_gym d_home_to_grocery d_grocery_to_gym time_diff →
  correct_answer speed_grocery_to_gym :=
by
  sorry

end NUMINAMATH_GPT_find_speed_grocery_to_gym_l672_67253


namespace NUMINAMATH_GPT_find_a_plus_b_l672_67284

def star (a b : ℕ) : ℕ := a^b - a*b + 5

theorem find_a_plus_b (a b : ℕ) (ha : 2 ≤ a) (hb : 3 ≤ b) (h : star a b = 13) : a + b = 6 :=
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l672_67284


namespace NUMINAMATH_GPT_inequality_solution_l672_67225

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l672_67225


namespace NUMINAMATH_GPT_smallest_N_for_abs_x_squared_minus_4_condition_l672_67218

theorem smallest_N_for_abs_x_squared_minus_4_condition (x : ℝ) 
  (h : abs (x - 2) < 0.01) : abs (x^2 - 4) < 0.0401 := 
sorry

end NUMINAMATH_GPT_smallest_N_for_abs_x_squared_minus_4_condition_l672_67218


namespace NUMINAMATH_GPT_number_of_neutrons_eq_l672_67210

variable (A n x : ℕ)

/-- The number of neutrons N in the nucleus of an atom R, given that:
  1. A is the atomic mass number of R.
  2. The ion RO3^(n-) contains x outer electrons. -/
theorem number_of_neutrons_eq (N : ℕ) (h : A - N + 24 + n = x) : N = A + n + 24 - x :=
by sorry

end NUMINAMATH_GPT_number_of_neutrons_eq_l672_67210


namespace NUMINAMATH_GPT_trigonometric_identity_l672_67222

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) : 
  (1 / Real.cos (2 * α)) + Real.tan (2 * α) = 2012 := 
by
  -- This will be the proof body which we omit with sorry
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l672_67222


namespace NUMINAMATH_GPT_sale_in_first_month_l672_67230

theorem sale_in_first_month (sale1 sale2 sale3 sale4 sale5 : ℕ) 
  (h1 : sale1 = 5660) (h2 : sale2 = 6200) (h3 : sale3 = 6350) (h4 : sale4 = 6500) 
  (h_avg : (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 6000) : 
  sale5 = 5290 := 
by
  sorry

end NUMINAMATH_GPT_sale_in_first_month_l672_67230


namespace NUMINAMATH_GPT_manager_salary_l672_67295

theorem manager_salary (n : ℕ) (avg_salary : ℕ) (increment : ℕ) (new_avg_salary : ℕ) (new_total_salary : ℕ) (old_total_salary : ℕ) :
  n = 20 →
  avg_salary = 1500 →
  increment = 1000 →
  new_avg_salary = avg_salary + increment →
  old_total_salary = n * avg_salary →
  new_total_salary = (n + 1) * new_avg_salary →
  (new_total_salary - old_total_salary) = 22500 :=
by
  intros h_n h_avg_salary h_increment h_new_avg_salary h_old_total_salary h_new_total_salary
  sorry

end NUMINAMATH_GPT_manager_salary_l672_67295


namespace NUMINAMATH_GPT_geom_prog_common_ratio_l672_67256

variable {α : Type*} [Field α]

theorem geom_prog_common_ratio (x y z r : α) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (h1 : x * (y + z) = a) (h2 : y * (z + x) = a * r) (h3 : z * (x + y) = a * r^2) :
  r^2 + r + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_geom_prog_common_ratio_l672_67256


namespace NUMINAMATH_GPT_find_f_l672_67268

theorem find_f (f : ℤ → ℤ) (h : ∀ n : ℤ, n^2 + 4 * (f n) = (f (f n))^2) :
  (∀ x : ℤ, f x = 1 + x) ∨
  (∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)) ∨
  (f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)) :=
sorry

end NUMINAMATH_GPT_find_f_l672_67268


namespace NUMINAMATH_GPT_rectangle_perimeter_l672_67200

theorem rectangle_perimeter (A W : ℝ) (hA : A = 300) (hW : W = 15) : 
  (2 * ((A / W) + W)) = 70 := 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l672_67200


namespace NUMINAMATH_GPT_solve_percentage_chromium_first_alloy_l672_67272

noncomputable def percentage_chromium_first_alloy (x : ℝ) : Prop :=
  let w1 := 15 -- weight of the first alloy
  let c2 := 10 -- percentage of chromium in the second alloy
  let w2 := 35 -- weight of the second alloy
  let w_total := 50 -- total weight of the new alloy formed by mixing
  let c_new := 10.6 -- percentage of chromium in the new alloy
  -- chromium percentage equation
  ((x / 100) * w1 + (c2 / 100) * w2) = (c_new / 100) * w_total

theorem solve_percentage_chromium_first_alloy : percentage_chromium_first_alloy 12 :=
  sorry -- proof goes here

end NUMINAMATH_GPT_solve_percentage_chromium_first_alloy_l672_67272


namespace NUMINAMATH_GPT_solution_to_equation_l672_67285

theorem solution_to_equation (x y : ℤ) (h : x^6 - y^2 = 648) : 
  (x = 3 ∧ y = 9) ∨ 
  (x = -3 ∧ y = 9) ∨ 
  (x = 3 ∧ y = -9) ∨ 
  (x = -3 ∧ y = -9) :=
sorry

end NUMINAMATH_GPT_solution_to_equation_l672_67285


namespace NUMINAMATH_GPT_find_quotient_l672_67275

-- Define the given conditions
def dividend : ℤ := 144
def divisor : ℤ := 11
def remainder : ℤ := 1

-- Define the quotient logically derived from the given conditions
def quotient : ℤ := dividend / divisor

-- The theorem we need to prove
theorem find_quotient : quotient = 13 := by
  sorry

end NUMINAMATH_GPT_find_quotient_l672_67275


namespace NUMINAMATH_GPT_function_value_at_2018_l672_67219

theorem function_value_at_2018 (f : ℝ → ℝ)
  (h1 : f 4 = 2 - Real.sqrt 3)
  (h2 : ∀ x, f (x + 2) = 1 / (- f x)) :
  f 2018 = -2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_function_value_at_2018_l672_67219


namespace NUMINAMATH_GPT_no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l672_67266

-- Problem 1: Square of an even number followed by three times a square number
theorem no_consecutive_even_square_and_three_times_square :
  ∀ (k n : ℕ), ¬(3 * n ^ 2 = 4 * k ^ 2 + 1) :=
by sorry

-- Problem 2: Square number followed by seven times another square number
theorem no_consecutive_square_and_seven_times_square :
  ∀ (r s : ℕ), ¬(7 * s ^ 2 = r ^ 2 + 1) :=
by sorry

end NUMINAMATH_GPT_no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l672_67266


namespace NUMINAMATH_GPT_point_coordinates_l672_67250

-- Definitions based on conditions
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0
def dist_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := abs P.1 = d

-- Lean 4 statement
theorem point_coordinates {P : ℝ × ℝ} (h1 : on_x_axis P) (h2 : dist_to_y_axis P 3) :
  P = (3, 0) ∨ P = (-3, 0) :=
by sorry

end NUMINAMATH_GPT_point_coordinates_l672_67250


namespace NUMINAMATH_GPT_find_ray_solutions_l672_67251

noncomputable def polynomial (a x : ℝ) : ℝ :=
  x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3

theorem find_ray_solutions (a : ℝ) :
  (∀ x : ℝ, polynomial a x ≥ 0 → ∃ b : ℝ, ∀ y ≥ b, polynomial a y ≥ 0) ↔ a = 1 ∨ a = -1 :=
sorry

end NUMINAMATH_GPT_find_ray_solutions_l672_67251


namespace NUMINAMATH_GPT_minimum_rotation_angle_of_square_l672_67287

theorem minimum_rotation_angle_of_square : 
  ∀ (angle : ℝ), (∃ n : ℕ, angle = 360 / n) ∧ (n ≥ 1) ∧ (n ≤ 4) → angle = 90 :=
by
  sorry

end NUMINAMATH_GPT_minimum_rotation_angle_of_square_l672_67287


namespace NUMINAMATH_GPT_gcd_n_cube_plus_25_n_plus_3_l672_67280

theorem gcd_n_cube_plus_25_n_plus_3 (n : ℕ) (h : n > 3^2) : 
  Int.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_n_cube_plus_25_n_plus_3_l672_67280


namespace NUMINAMATH_GPT_equation_of_parallel_line_through_point_l672_67227

theorem equation_of_parallel_line_through_point :
  ∃ m b, (∀ x y, y = m * x + b → (∃ k, k = 3 ^ 2 - 9 * 2 + 1)) ∧ 
         (∀ x y, y = 3 * x + b → y - 0 = 3 * (x - (-2))) :=
sorry

end NUMINAMATH_GPT_equation_of_parallel_line_through_point_l672_67227


namespace NUMINAMATH_GPT_no_solution_xyz_l672_67249

theorem no_solution_xyz : ∀ (x y z : Nat), (1 ≤ x) → (x ≤ 9) → (0 ≤ y) → (y ≤ 9) → (0 ≤ z) → (z ≤ 9) →
    100 * x + 10 * y + z ≠ 10 * x * y + x * z :=
by
  intros x y z hx1 hx9 hy1 hy9 hz1 hz9
  sorry

end NUMINAMATH_GPT_no_solution_xyz_l672_67249


namespace NUMINAMATH_GPT_complex_number_proof_l672_67258

open Complex

noncomputable def problem_complex (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) : ℂ :=
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1)

theorem complex_number_proof (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) :
  problem_complex z h1 h2 = 8 :=
  sorry

end NUMINAMATH_GPT_complex_number_proof_l672_67258


namespace NUMINAMATH_GPT_sampling_methods_correct_l672_67241

def condition1 : Prop :=
  ∃ yogurt_boxes : ℕ, yogurt_boxes = 10 ∧ ∃ sample_boxes : ℕ, sample_boxes = 3

def condition2 : Prop :=
  ∃ rows seats_per_row attendees sample_size : ℕ,
    rows = 32 ∧ seats_per_row = 40 ∧ attendees = rows * seats_per_row ∧ sample_size = 32

def condition3 : Prop :=
  ∃ liberal_arts_classes science_classes total_classes sample_size : ℕ,
    liberal_arts_classes = 4 ∧ science_classes = 8 ∧ total_classes = liberal_arts_classes + science_classes ∧ sample_size = 50

def simple_random_sampling (s : Prop) : Prop := sorry -- definition for simple random sampling
def systematic_sampling (s : Prop) : Prop := sorry -- definition for systematic sampling
def stratified_sampling (s : Prop) : Prop := sorry -- definition for stratified sampling

theorem sampling_methods_correct :
  (condition1 → simple_random_sampling condition1) ∧
  (condition2 → systematic_sampling condition2) ∧
  (condition3 → stratified_sampling condition3) :=
by {
  sorry
}

end NUMINAMATH_GPT_sampling_methods_correct_l672_67241


namespace NUMINAMATH_GPT_matrix_addition_correct_l672_67238

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 4, -2], ![5, -3, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![ -3,  2, -4], ![ 1, -6,  3], ![-2,  4,  0]]

def expectedSum : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![-1,  1, -1], ![ 1, -2,  1], ![ 3,  1,  1]]

theorem matrix_addition_correct :
  A + B = expectedSum := by
  sorry

end NUMINAMATH_GPT_matrix_addition_correct_l672_67238


namespace NUMINAMATH_GPT_product_is_two_l672_67292

theorem product_is_two : 
  ((10 : ℚ) * (1/5) * 4 * (1/16) * (1/2) * 8 = 2) :=
sorry

end NUMINAMATH_GPT_product_is_two_l672_67292


namespace NUMINAMATH_GPT_sum_f_inv_l672_67297

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 2 * x - 1 else x ^ 2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 9 then (y + 1) / 2 else Real.sqrt y

theorem sum_f_inv :
  (f_inv (-3) + f_inv (-2) + 
   f_inv (-1) + f_inv 0 + 
   f_inv 1 + f_inv 2 + 
   f_inv 3 + f_inv 4 + 
   f_inv 9) = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_inv_l672_67297


namespace NUMINAMATH_GPT_k_value_if_perfect_square_l672_67237

theorem k_value_if_perfect_square (k : ℤ) (x : ℝ) (h : ∃ (a : ℝ), x^2 + k * x + 25 = a^2) : k = 10 ∨ k = -10 := by
  sorry

end NUMINAMATH_GPT_k_value_if_perfect_square_l672_67237


namespace NUMINAMATH_GPT_set_intersection_union_eq_complement_l672_67223

def A : Set ℝ := {x | 2 * x^2 + x - 3 = 0}
def B : Set ℝ := {i | i^2 ≥ 4}
def complement_C : Set ℝ := {-1, 1, 3/2}

theorem set_intersection_union_eq_complement :
  A ∩ B ∪ complement_C = complement_C :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_union_eq_complement_l672_67223


namespace NUMINAMATH_GPT_prove_expression_l672_67283

theorem prove_expression (a b : ℕ) 
  (h1 : 180 % 2^a = 0 ∧ 180 % 2^(a+1) ≠ 0)
  (h2 : 180 % 3^b = 0 ∧ 180 % 3^(b+1) ≠ 0) :
  (1 / 4 : ℚ)^(b - a) = 1 := 
sorry

end NUMINAMATH_GPT_prove_expression_l672_67283


namespace NUMINAMATH_GPT_frac_sum_eq_l672_67242

theorem frac_sum_eq (a b : ℝ) (h1 : a^2 + a - 1 = 0) (h2 : b^2 + b - 1 = 0) : 
  (a / b + b / a = 2) ∨ (a / b + b / a = -3) := 
sorry

end NUMINAMATH_GPT_frac_sum_eq_l672_67242


namespace NUMINAMATH_GPT_new_average_of_remaining_numbers_l672_67203

theorem new_average_of_remaining_numbers (sum_12 avg_12 n1 n2 : ℝ) 
  (h1 : avg_12 = 90)
  (h2 : sum_12 = 1080)
  (h3 : n1 = 80)
  (h4 : n2 = 85)
  : (sum_12 - n1 - n2) / 10 = 91.5 := 
by
  sorry

end NUMINAMATH_GPT_new_average_of_remaining_numbers_l672_67203


namespace NUMINAMATH_GPT_length_of_train_l672_67246

-- declare constants
variables (L S : ℝ)

-- state conditions
def condition1 : Prop := L = S * 50
def condition2 : Prop := L + 500 = S * 100

-- state the theorem to prove
theorem length_of_train (h1 : condition1 L S) (h2 : condition2 L S) : L = 500 :=
by sorry

end NUMINAMATH_GPT_length_of_train_l672_67246


namespace NUMINAMATH_GPT_brothers_complete_task_in_3_days_l672_67269

theorem brothers_complete_task_in_3_days :
  (1 / 4 + 1 / 12) * 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_brothers_complete_task_in_3_days_l672_67269


namespace NUMINAMATH_GPT_integer_solutions_equation_l672_67255

theorem integer_solutions_equation : 
  (∃ x y : ℤ, (1 / (2022 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ))) → 
  ∃! (n : ℕ), n = 53 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_equation_l672_67255


namespace NUMINAMATH_GPT_chocolates_initial_count_l672_67271

theorem chocolates_initial_count : 
  ∀ (chocolates_first_day chocolates_second_day chocolates_third_day chocolates_fourth_day chocolates_fifth_day initial_chocolates : ℕ),
  chocolates_first_day = 4 →
  chocolates_second_day = 2 * chocolates_first_day - 3 →
  chocolates_third_day = chocolates_first_day - 2 →
  chocolates_fourth_day = chocolates_third_day - 1 →
  chocolates_fifth_day = 12 →
  initial_chocolates = chocolates_first_day + chocolates_second_day + chocolates_third_day + chocolates_fourth_day + chocolates_fifth_day →
  initial_chocolates = 24 :=
by {
  -- the proof will go here,
  sorry
}

end NUMINAMATH_GPT_chocolates_initial_count_l672_67271


namespace NUMINAMATH_GPT_three_lines_pass_through_point_and_intersect_parabola_l672_67233

-- Define the point (0,1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x as a set of points
def parabola (p : ℝ × ℝ) : Prop :=
  (p.snd)^2 = 4 * (p.fst)

-- Define the condition for the line passing through (0,1)
def line_through_point (line_eq : ℝ → ℝ) : Prop :=
  line_eq 0 = 1

-- Define the condition for the line intersecting the parabola at only one point
def intersects_once (line_eq : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, parabola (x, line_eq x)

-- The main theorem statement
theorem three_lines_pass_through_point_and_intersect_parabola :
  ∃ (f1 f2 f3 : ℝ → ℝ), 
    line_through_point f1 ∧ line_through_point f2 ∧ line_through_point f3 ∧
    intersects_once f1 ∧ intersects_once f2 ∧ intersects_once f3 ∧
    (∀ (f : ℝ → ℝ), (line_through_point f ∧ intersects_once f) ->
      (f = f1 ∨ f = f2 ∨ f = f3)) :=
sorry

end NUMINAMATH_GPT_three_lines_pass_through_point_and_intersect_parabola_l672_67233


namespace NUMINAMATH_GPT_new_car_travel_distance_l672_67229

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end NUMINAMATH_GPT_new_car_travel_distance_l672_67229


namespace NUMINAMATH_GPT_car_first_hour_speed_l672_67247

theorem car_first_hour_speed
  (x speed2 : ℝ)
  (avgSpeed : ℝ)
  (h_speed2 : speed2 = 60)
  (h_avgSpeed : avgSpeed = 35) :
  (avgSpeed = (x + speed2) / 2) → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_car_first_hour_speed_l672_67247


namespace NUMINAMATH_GPT_geometric_sequence_a4_l672_67259

theorem geometric_sequence_a4 {a : ℕ → ℝ} (q : ℝ) (h₁ : q > 0)
  (h₂ : ∀ n, a (n + 1) = a 1 * q ^ (n)) (h₃ : a 1 = 2) 
  (h₄ : a 2 + 4 = (a 1 + a 3) / 2) : a 4 = 54 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l672_67259


namespace NUMINAMATH_GPT_remaining_credit_l672_67202

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end NUMINAMATH_GPT_remaining_credit_l672_67202


namespace NUMINAMATH_GPT_cecile_apples_l672_67224

theorem cecile_apples (C D : ℕ) (h1 : D = C + 20) (h2 : C + D = 50) : C = 15 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_cecile_apples_l672_67224


namespace NUMINAMATH_GPT_percentage_exceeds_self_l672_67204

theorem percentage_exceeds_self (N : ℕ) (P : ℝ) (h1 : N = 150) (h2 : N = (P / 100) * N + 126) : P = 16 := by
  sorry

end NUMINAMATH_GPT_percentage_exceeds_self_l672_67204


namespace NUMINAMATH_GPT_max_value_of_sum_l672_67226

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + 4 * y^2 + 9 * z^2 = 3) : x + 2 * y + 3 * z ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_sum_l672_67226


namespace NUMINAMATH_GPT_increasing_interval_of_f_l672_67206

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem increasing_interval_of_f :
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) → MonotoneOn f (Set.Icc (-3 * Real.pi / 4) (Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l672_67206


namespace NUMINAMATH_GPT_value_in_parentheses_l672_67213

theorem value_in_parentheses (x : ℝ) (h : x / Real.sqrt 18 = Real.sqrt 2) : x = 6 :=
sorry

end NUMINAMATH_GPT_value_in_parentheses_l672_67213


namespace NUMINAMATH_GPT_factorization_2109_two_digit_l672_67231

theorem factorization_2109_two_digit (a b: ℕ) : 
  2109 = a * b ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 → false :=
by
  sorry

end NUMINAMATH_GPT_factorization_2109_two_digit_l672_67231


namespace NUMINAMATH_GPT_tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l672_67243

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end NUMINAMATH_GPT_tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l672_67243


namespace NUMINAMATH_GPT_percentage_increase_is_50_l672_67267

def papaya_growth (P : ℝ) : Prop :=
  let growth1 := 2
  let growth2 := 2 * (1 + P / 100)
  let growth3 := 1.5 * growth2
  let growth4 := 2 * growth3
  let growth5 := 0.5 * growth4
  growth1 + growth2 + growth3 + growth4 + growth5 = 23

theorem percentage_increase_is_50 :
  ∃ (P : ℝ), papaya_growth P ∧ P = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_50_l672_67267


namespace NUMINAMATH_GPT_relationship_y1_y2_l672_67240

theorem relationship_y1_y2 :
  ∀ (b y1 y2 : ℝ), 
  (∃ b y1 y2, y1 = -2023 * (-2) + b ∧ y2 = -2023 * (-1) + b) → y1 > y2 :=
by
  intro b y1 y2 h
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_l672_67240


namespace NUMINAMATH_GPT_find_initial_order_l672_67293

variables (x : ℕ)

def initial_order (x : ℕ) :=
  x + 60 = 72 * (x / 90 + 1)

theorem find_initial_order (h1 : initial_order x) : x = 60 :=
  sorry

end NUMINAMATH_GPT_find_initial_order_l672_67293


namespace NUMINAMATH_GPT_units_digit_seven_pow_ten_l672_67236

theorem units_digit_seven_pow_ten : ∃ u : ℕ, (7^10) % 10 = u ∧ u = 9 :=
by
  use 9
  sorry

end NUMINAMATH_GPT_units_digit_seven_pow_ten_l672_67236


namespace NUMINAMATH_GPT_problem_solution_l672_67274

theorem problem_solution (a b d : ℤ) (ha : a = 2500) (hb : b = 2409) (hd : d = 81) :
  (a - b) ^ 2 / d = 102 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l672_67274


namespace NUMINAMATH_GPT_trains_at_initial_stations_l672_67286

-- Define the durations of round trips for each line.
def red_round_trip : ℕ := 14
def blue_round_trip : ℕ := 16
def green_round_trip : ℕ := 18

-- Define the total time we are analyzing.
def total_time : ℕ := 2016

-- Define the statement that needs to be proved.
theorem trains_at_initial_stations : 
  (total_time % red_round_trip = 0) ∧ 
  (total_time % blue_round_trip = 0) ∧ 
  (total_time % green_round_trip = 0) := 
by
  -- The proof can be added here.
  sorry

end NUMINAMATH_GPT_trains_at_initial_stations_l672_67286


namespace NUMINAMATH_GPT_find_added_number_l672_67291

theorem find_added_number (R X : ℕ) (hR : R = 45) (h : 2 * (2 * R + X) = 188) : X = 4 :=
by 
  -- We would normally provide the proof here
  sorry  -- We skip the proof as per the instructions

end NUMINAMATH_GPT_find_added_number_l672_67291


namespace NUMINAMATH_GPT_range_of_y_l672_67270

theorem range_of_y (x y : ℝ) (h1 : |y - 2 * x| = x^2) (h2 : -1 < x) (h3 : x < 0) : -3 < y ∧ y < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l672_67270


namespace NUMINAMATH_GPT_height_table_l672_67298

variable (l w h : ℝ)

theorem height_table (h_eq1 : l + h - w = 32) (h_eq2 : w + h - l = 28) : h = 30 := by
  sorry

end NUMINAMATH_GPT_height_table_l672_67298


namespace NUMINAMATH_GPT_max_triangles_in_graph_l672_67262

def points : Finset Point := sorry
def no_coplanar (points : Finset Point) : Prop := sorry
def no_tetrahedron (points : Finset Point) : Prop := sorry
def triangles (points : Finset Point) : ℕ := sorry

theorem max_triangles_in_graph (points : Finset Point) 
  (H1 : points.card = 9) 
  (H2 : no_coplanar points) 
  (H3 : no_tetrahedron points) : 
  triangles points ≤ 27 := 
sorry

end NUMINAMATH_GPT_max_triangles_in_graph_l672_67262


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l672_67211

noncomputable def x := (2 : ℚ) / (3 : ℚ)
noncomputable def y := (5 : ℚ) / (11 : ℚ)

theorem sum_of_repeating_decimals : x + y = (37 : ℚ) / (33 : ℚ) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_repeating_decimals_l672_67211


namespace NUMINAMATH_GPT_quadrilateral_area_l672_67207

structure Point :=
  (x : ℝ)
  (y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_quadrilateral (A B C D : Point) : ℝ :=
  area_of_triangle A B C + area_of_triangle A C D

def A : Point := ⟨2, 2⟩
def B : Point := ⟨2, -1⟩
def C : Point := ⟨3, -1⟩
def D : Point := ⟨2007, 2008⟩

theorem quadrilateral_area :
  area_of_quadrilateral A B C D = 2008006.5 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l672_67207


namespace NUMINAMATH_GPT_geometric_sequence_problem_l672_67264

variable {a : ℕ → ℝ}

-- Given conditions
def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q r, (∀ n, a (n + 1) = q * a n ∧ a 0 = r)

-- Define the conditions from the problem
def condition1 (a : ℕ → ℝ) :=
  a 3 + a 6 = 6

def condition2 (a : ℕ → ℝ) :=
  a 5 + a 8 = 9

-- Theorem to be proved
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (hgeom : geometric_sequence a)
  (h1 : condition1 a)
  (h2 : condition2 a) :
  a 7 + a 10 = 27 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l672_67264


namespace NUMINAMATH_GPT_smallest_three_digit_number_with_property_l672_67214

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a < 1000 ∧ ∃ n : ℕ, 1000 * a + (a + 1) = n^2 ∧ a = 183 :=
sorry

end NUMINAMATH_GPT_smallest_three_digit_number_with_property_l672_67214


namespace NUMINAMATH_GPT_shortest_part_length_l672_67257

theorem shortest_part_length (total_length : ℝ) (r1 r2 r3 : ℝ) (shortest_length : ℝ) :
  total_length = 196.85 → r1 = 3.6 → r2 = 8.4 → r3 = 12 → shortest_length = 29.5275 :=
by
  sorry

end NUMINAMATH_GPT_shortest_part_length_l672_67257


namespace NUMINAMATH_GPT_min_value_seq_div_n_l672_67277

-- Definitions of the conditions
def a_seq (n : ℕ) : ℕ := 
  if n = 0 then 0 else if n = 1 then 98 else 102 + (n - 2) * (2 * n + 2)

-- The property we need to prove
theorem min_value_seq_div_n :
  (∀ n : ℕ, (n ≥ 1) → (a_seq n / n) ≥ 26) ∧ (∃ n : ℕ, (n ≥ 1) ∧ (a_seq n / n) = 26) :=
sorry

end NUMINAMATH_GPT_min_value_seq_div_n_l672_67277


namespace NUMINAMATH_GPT_infinite_double_perfect_squares_l672_67228

def is_double_number (n : ℕ) : Prop :=
  ∃ k m : ℕ, m > 0 ∧ n = m * 10^k + m

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem infinite_double_perfect_squares : ∀ n : ℕ, ∃ m, n < m ∧ is_double_number m ∧ is_perfect_square m :=
  sorry

end NUMINAMATH_GPT_infinite_double_perfect_squares_l672_67228


namespace NUMINAMATH_GPT_exists_sequence_a_l672_67212

-- Define the sequence and properties
def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 18 = 2019 ∧
  ∀ k, 3 ≤ k → k ≤ 18 → ∃ i j, 1 ≤ i → i < j → j < k → a k = a i + a j

-- The main theorem statement
theorem exists_sequence_a : ∃ (a : ℕ → ℤ), sequence_a a := 
sorry

end NUMINAMATH_GPT_exists_sequence_a_l672_67212


namespace NUMINAMATH_GPT_total_students_in_high_school_l672_67205

theorem total_students_in_high_school (sample_size first_year third_year second_year : ℕ) (total_students : ℕ) 
  (h1 : sample_size = 45) 
  (h2 : first_year = 20) 
  (h3 : third_year = 10) 
  (h4 : second_year = 300)
  (h5 : sample_size = first_year + third_year + (sample_size - first_year - third_year)) :
  total_students = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_high_school_l672_67205


namespace NUMINAMATH_GPT_solution_set_inequality_l672_67239

theorem solution_set_inequality {a b c x : ℝ} (h1 : a < 0)
  (h2 : -b / a = 1 + 2) (h3 : c / a = 1 * 2) :
  a - c * (x^2 - x - 1) - b * x ≥ 0 ↔ x ≤ -3 / 2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l672_67239


namespace NUMINAMATH_GPT_sum_of_decimals_l672_67220

theorem sum_of_decimals :
  let a := 0.35
  let b := 0.048
  let c := 0.0072
  a + b + c = 0.4052 := by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l672_67220


namespace NUMINAMATH_GPT_students_sampled_from_schoolB_l672_67273

-- Definitions from the conditions in a)
def schoolA_students := 800
def schoolB_students := 500
def total_students := schoolA_students + schoolB_students
def schoolA_sampled_students := 48

-- Mathematically equivalent proof problem
theorem students_sampled_from_schoolB : 
  let proportionA := (schoolA_students : ℝ) / total_students
  let proportionB := (schoolB_students : ℝ) / total_students
  let total_sampled_students := schoolA_sampled_students / proportionA
  let b_sampled_students := proportionB * total_sampled_students
  b_sampled_students = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_students_sampled_from_schoolB_l672_67273


namespace NUMINAMATH_GPT_evaluate_expression_l672_67265

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = -3) :
  (2 * x)^2 * (y^2)^3 * z^2 = 1 / 81 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_evaluate_expression_l672_67265


namespace NUMINAMATH_GPT_problem_statement_l672_67282

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def x : ℝ := alpha ^ 1000
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l672_67282


namespace NUMINAMATH_GPT_point_M_coordinates_l672_67294

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 4 * x

-- Define the condition given in the problem: instantaneous rate of change
def rate_of_change (a : ℝ) : Prop := f' a = -4

-- Define the point on the curve
def point_M (a b : ℝ) : Prop := f a = b

-- Proof statement
theorem point_M_coordinates : 
  ∃ (a b : ℝ), rate_of_change a ∧ point_M a b ∧ a = -1 ∧ b = 3 :=  
by
  sorry

end NUMINAMATH_GPT_point_M_coordinates_l672_67294


namespace NUMINAMATH_GPT_unique_solution_abs_eq_l672_67289

theorem unique_solution_abs_eq : 
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 :=
by
  use -5
  sorry

end NUMINAMATH_GPT_unique_solution_abs_eq_l672_67289


namespace NUMINAMATH_GPT_pillows_from_feathers_l672_67261

def feathers_per_pound : ℕ := 300
def feathers_total : ℕ := 3600
def pounds_per_pillow : ℕ := 2

theorem pillows_from_feathers :
  (feathers_total / feathers_per_pound / pounds_per_pillow) = 6 :=
by
  sorry

end NUMINAMATH_GPT_pillows_from_feathers_l672_67261


namespace NUMINAMATH_GPT_problem1_problem2_l672_67232

-- Problem statement 1: Prove (a-2)(a-6) < (a-3)(a-5)
theorem problem1 (a : ℝ) : (a - 2) * (a - 6) < (a - 3) * (a - 5) :=
by
  sorry

-- Problem statement 2: Prove the range of values for 2x - y given -2 < x < 1 and 1 < y < 2 is (-6, 1)
theorem problem2 (x y : ℝ) (hx : -2 < x) (hx1 : x < 1) (hy : 1 < y) (hy1 : y < 2) : -6 < 2 * x - y ∧ 2 * x - y < 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l672_67232


namespace NUMINAMATH_GPT_terminating_decimals_nat_l672_67209

theorem terminating_decimals_nat (n : ℕ) (h1 : ∃ a b : ℕ, n = 2^a * 5^b)
  (h2 : ∃ c d : ℕ, n + 1 = 2^c * 5^d) : n = 1 ∨ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimals_nat_l672_67209


namespace NUMINAMATH_GPT_find_x_l672_67216

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def parallel (v w : Point) : Prop :=
  v.x * w.y = v.y * w.x

theorem find_x (A B C : Point) (hA : A = ⟨0, -3⟩) (hB : B = ⟨3, 3⟩) (hC : C = ⟨x, -1⟩) (h_parallel : parallel (vector A B) (vector A C)) : x = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l672_67216


namespace NUMINAMATH_GPT_decreasing_on_transformed_interval_l672_67201

theorem decreasing_on_transformed_interval
  (f : ℝ → ℝ)
  (h : ∀ ⦃x₁ x₂ : ℝ⦄, 1 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 2 → f x₁ ≤ f x₂) :
  ∀ ⦃x₁ x₂ : ℝ⦄, -1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 0 → f (1 - x₂) ≤ f (1 - x₁) :=
sorry

end NUMINAMATH_GPT_decreasing_on_transformed_interval_l672_67201


namespace NUMINAMATH_GPT_additional_chair_frequency_l672_67234

theorem additional_chair_frequency 
  (workers : ℕ)
  (chairs_per_worker_per_hour : ℕ)
  (hours : ℕ)
  (total_chairs : ℕ) 
  (additional_chairs_rate : ℕ)
  (h_workers : workers = 3) 
  (h_chairs_per_worker : chairs_per_worker_per_hour = 4) 
  (h_hours : hours = 6 ) 
  (h_total_chairs : total_chairs = 73) :
  additional_chairs_rate = 6 :=
by
  sorry

end NUMINAMATH_GPT_additional_chair_frequency_l672_67234


namespace NUMINAMATH_GPT_find_function_l672_67290

theorem find_function (f : ℚ → ℚ) (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_function_l672_67290


namespace NUMINAMATH_GPT_trees_planted_l672_67288

theorem trees_planted (current_short_trees planted_short_trees total_short_trees : ℕ)
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees = 217) :
  planted_short_trees = 105 :=
by
  sorry

end NUMINAMATH_GPT_trees_planted_l672_67288


namespace NUMINAMATH_GPT_min_value_a2_b2_l672_67215

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_a2_b2_l672_67215


namespace NUMINAMATH_GPT_num_integers_D_l672_67244

theorem num_integers_D :
  ∃ (D : ℝ) (n : ℕ), 
    (∀ (a b : ℝ), -1/4 < a → a < 1/4 → -1/4 < b → b < 1/4 → abs (a^2 - D * b^2) < 1) → n = 32 :=
sorry

end NUMINAMATH_GPT_num_integers_D_l672_67244


namespace NUMINAMATH_GPT_factor_polynomial_l672_67254

variable {R : Type*} [CommRing R]

theorem factor_polynomial (a b c : R) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + ab + bc + ac)) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l672_67254


namespace NUMINAMATH_GPT_chosen_number_is_30_l672_67208

theorem chosen_number_is_30 (x : ℤ) 
  (h1 : 8 * x - 138 = 102) : x = 30 := 
sorry

end NUMINAMATH_GPT_chosen_number_is_30_l672_67208


namespace NUMINAMATH_GPT_find_n_l672_67248

theorem find_n {x n : ℕ} (h1 : 3 * x - 4 = 8) (h2 : 7 * x - 15 = 13) (h3 : 4 * x + 2 = 18) 
  (h4 : n = 803) : 8 + (n - 1) * 5 = 4018 := by
  sorry

end NUMINAMATH_GPT_find_n_l672_67248
