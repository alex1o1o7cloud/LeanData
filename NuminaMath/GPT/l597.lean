import Mathlib

namespace NUMINAMATH_GPT_range_of_sum_of_two_l597_59767

theorem range_of_sum_of_two (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4 / 3 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_range_of_sum_of_two_l597_59767


namespace NUMINAMATH_GPT_solve_inequality_l597_59726

theorem solve_inequality (x : ℝ) (hx : x ≥ 0) :
  2021 * (x ^ (2020/202)) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l597_59726


namespace NUMINAMATH_GPT_sarah_total_weeds_l597_59745

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end NUMINAMATH_GPT_sarah_total_weeds_l597_59745


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l597_59787

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) (h1 : (3 * x + 3)^2 = x * (6 * x + 6)) 
(h2 : r = (3 * x + 3) / x) :
  (6 * x + 6) * r = -24 :=
by {
  -- Definitions of x, r and condition h1, h2 are given.
  -- Conclusion must follow that the fourth term is -24.
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l597_59787


namespace NUMINAMATH_GPT_probability_palindrome_divisible_by_11_l597_59776

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_all_palindromes : ℕ :=
  9 * 10 * 10

def count_palindromes_div_by_11 : ℕ :=
  9 * 10

theorem probability_palindrome_divisible_by_11 :
  (count_palindromes_div_by_11 : ℚ) / count_all_palindromes = 1 / 10 :=
by sorry

end NUMINAMATH_GPT_probability_palindrome_divisible_by_11_l597_59776


namespace NUMINAMATH_GPT_two_pow_n_add_two_gt_n_sq_l597_59721

open Nat

theorem two_pow_n_add_two_gt_n_sq (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 :=
by
  sorry

end NUMINAMATH_GPT_two_pow_n_add_two_gt_n_sq_l597_59721


namespace NUMINAMATH_GPT_total_cost_l597_59795

variables (p e n : ℕ) -- represent the costs of pencil, eraser, and notebook in cents

-- Given conditions
def conditions : Prop :=
  9 * p + 7 * e + 4 * n = 220 ∧
  p > n ∧ n > e ∧ e > 0

-- Prove the total cost
theorem total_cost (h : conditions p e n) : p + n + e = 26 :=
sorry

end NUMINAMATH_GPT_total_cost_l597_59795


namespace NUMINAMATH_GPT_a_and_b_together_complete_work_in_12_days_l597_59796

-- Define the rate of work for b
def R_b : ℚ := 1 / 60

-- Define the rate of work for a based on the given condition that a is four times as fast as b
def R_a : ℚ := 4 * R_b

-- Define the combined rate of work for a and b working together
def R_a_plus_b : ℚ := R_a + R_b

-- Define the target time
def target_time : ℚ := 12

-- Proof statement
theorem a_and_b_together_complete_work_in_12_days :
  (R_a_plus_b * target_time) = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_a_and_b_together_complete_work_in_12_days_l597_59796


namespace NUMINAMATH_GPT_find_a_plus_c_l597_59771

theorem find_a_plus_c (a b c d : ℝ)
  (h₁ : -(3 - a) ^ 2 + b = 6) (h₂ : (3 - c) ^ 2 + d = 6)
  (h₃ : -(7 - a) ^ 2 + b = 2) (h₄ : (7 - c) ^ 2 + d = 2) :
  a + c = 10 := sorry

end NUMINAMATH_GPT_find_a_plus_c_l597_59771


namespace NUMINAMATH_GPT_initial_students_l597_59717

variable (n : ℝ) (W : ℝ)

theorem initial_students 
  (h1 : W = n * 15)
  (h2 : W + 11 = (n + 1) * 14.8)
  (h3 : 15 * n + 11 = 14.8 * n + 14.8)
  (h4 : 0.2 * n = 3.8) :
  n = 19 :=
sorry

end NUMINAMATH_GPT_initial_students_l597_59717


namespace NUMINAMATH_GPT_fraction_problem_l597_59723

theorem fraction_problem :
  ((3 / 4 - 5 / 8) / 2) = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l597_59723


namespace NUMINAMATH_GPT_sphere_surface_area_increase_l597_59708

theorem sphere_surface_area_increase (V A : ℝ) (r : ℝ)
  (hV : V = (4/3) * π * r^3)
  (hA : A = 4 * π * r^2)
  : (∃ r', (V = 8 * ((4/3) * π * r'^3)) ∧ (∃ A', A' = 4 * A)) :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_increase_l597_59708


namespace NUMINAMATH_GPT_cups_filled_l597_59754

def total_tea : ℕ := 1050
def tea_per_cup : ℕ := 65

theorem cups_filled : Nat.floor (total_tea / (tea_per_cup : ℚ)) = 16 :=
by
  sorry

end NUMINAMATH_GPT_cups_filled_l597_59754


namespace NUMINAMATH_GPT_nat_number_solution_odd_l597_59755

theorem nat_number_solution_odd (x y z : ℕ) (h : x + y + z = 100) : 
  ∃ P : ℕ, P = 49 ∧ P % 2 = 1 := 
sorry

end NUMINAMATH_GPT_nat_number_solution_odd_l597_59755


namespace NUMINAMATH_GPT_a_is_multiple_of_2_l597_59739

theorem a_is_multiple_of_2 (a : ℕ) (h1 : 0 < a) (h2 : (4 ^ a) % 10 = 6) : a % 2 = 0 :=
sorry

end NUMINAMATH_GPT_a_is_multiple_of_2_l597_59739


namespace NUMINAMATH_GPT_product_of_numbers_is_86_l597_59709

-- Definitions of the two conditions
def sum_eq_24 (x y : ℝ) : Prop := x + y = 24
def sum_of_squares_eq_404 (x y : ℝ) : Prop := x^2 + y^2 = 404

-- The theorem to prove the product of the two numbers
theorem product_of_numbers_is_86 (x y : ℝ) (h1 : sum_eq_24 x y) (h2 : sum_of_squares_eq_404 x y) : x * y = 86 :=
  sorry

end NUMINAMATH_GPT_product_of_numbers_is_86_l597_59709


namespace NUMINAMATH_GPT_probability_of_interval_l597_59799

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_interval_l597_59799


namespace NUMINAMATH_GPT_range_of_a_l597_59720

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then 2^x + 1 else -x^2 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a < 3) ↔ (2 ≤ a ∧ a < 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l597_59720


namespace NUMINAMATH_GPT_product_of_five_numbers_is_256_l597_59783

def possible_numbers : Set ℕ := {1, 2, 4}

theorem product_of_five_numbers_is_256 
  (x1 x2 x3 x4 x5 : ℕ) 
  (h1 : x1 ∈ possible_numbers) 
  (h2 : x2 ∈ possible_numbers) 
  (h3 : x3 ∈ possible_numbers) 
  (h4 : x4 ∈ possible_numbers) 
  (h5 : x5 ∈ possible_numbers) : 
  x1 * x2 * x3 * x4 * x5 = 256 :=
sorry

end NUMINAMATH_GPT_product_of_five_numbers_is_256_l597_59783


namespace NUMINAMATH_GPT_problems_finished_equals_45_l597_59757

/-- Mathematical constants and conditions -/
def ratio_finished_left (F L : ℕ) : Prop := F = 9 * (L / 4)
def total_problems (F L : ℕ) : Prop := F + L = 65

/-- Lean theorem to prove the problem statement -/
theorem problems_finished_equals_45 :
  ∃ F L : ℕ, ratio_finished_left F L ∧ total_problems F L ∧ F = 45 :=
by
  sorry

end NUMINAMATH_GPT_problems_finished_equals_45_l597_59757


namespace NUMINAMATH_GPT_exists_real_a_l597_59740

noncomputable def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem exists_real_a : ∃ a : ℝ, a = -2 ∧ A a ∩ C = ∅ ∧ ∅ ⊂ A a ∩ B := 
by {
  sorry
}

end NUMINAMATH_GPT_exists_real_a_l597_59740


namespace NUMINAMATH_GPT_ball_hits_ground_at_time_l597_59766

-- Given definitions from the conditions
def y (t : ℝ) : ℝ := -4.9 * t^2 + 5 * t + 8

-- Statement of the problem: proving the time t when the ball hits the ground
theorem ball_hits_ground_at_time :
  ∃ t : ℝ, y t = 0 ∧ t = 1.887 := 
sorry

end NUMINAMATH_GPT_ball_hits_ground_at_time_l597_59766


namespace NUMINAMATH_GPT_range_of_a_l597_59775

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x ^ 2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l597_59775


namespace NUMINAMATH_GPT_value_of_a_l597_59752

noncomputable def a : ℕ := 4

def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a*a}
def C : Set ℕ := {0, 1, 2, 4, 16}

theorem value_of_a : A ∪ B = C → a = 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_a_l597_59752


namespace NUMINAMATH_GPT_required_volume_proof_l597_59701

-- Defining the conditions
def initial_volume : ℝ := 60
def initial_concentration : ℝ := 0.10
def final_concentration : ℝ := 0.15

-- Defining the equation
def required_volume (V : ℝ) : Prop :=
  (initial_concentration * initial_volume + V = final_concentration * (initial_volume + V))

-- Stating the proof problem
theorem required_volume_proof :
  ∃ V : ℝ, required_volume V ∧ V = 3 / 0.85 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_required_volume_proof_l597_59701


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l597_59772

theorem quadratic_no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2 * x + c ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l597_59772


namespace NUMINAMATH_GPT_david_money_l597_59784

theorem david_money (S : ℝ) (h_initial : 1500 - S = S - 500) : 1500 - S = 500 :=
by
  sorry

end NUMINAMATH_GPT_david_money_l597_59784


namespace NUMINAMATH_GPT_store_discount_problem_l597_59718

theorem store_discount_problem (original_price : ℝ) :
  let price_after_first_discount := original_price * 0.75
  let price_after_second_discount := price_after_first_discount * 0.90
  let true_discount := 1 - price_after_second_discount / original_price
  let claimed_discount := 0.40
  let difference := claimed_discount - true_discount
  true_discount = 0.325 ∧ difference = 0.075 :=
by
  sorry

end NUMINAMATH_GPT_store_discount_problem_l597_59718


namespace NUMINAMATH_GPT_income_max_takehome_pay_l597_59751

theorem income_max_takehome_pay :
  ∃ x : ℝ, (∀ y : ℝ, 1000 * y - 5 * y^2 ≤ 1000 * x - 5 * x^2) ∧ x = 100 :=
by
  sorry

end NUMINAMATH_GPT_income_max_takehome_pay_l597_59751


namespace NUMINAMATH_GPT_equal_chessboard_numbers_l597_59730

theorem equal_chessboard_numbers (n : ℕ) (board : ℕ → ℕ → ℕ) 
  (mean_property : ∀ (x y : ℕ), board x y = (board (x-1) y + board (x+1) y + board x (y-1) + board x (y+1)) / 4) : 
  ∀ (x y : ℕ), board x y = board 0 0 :=
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_equal_chessboard_numbers_l597_59730


namespace NUMINAMATH_GPT_triangle_obtuse_l597_59759

theorem triangle_obtuse 
  (A B : ℝ)
  (hA : 0 < A ∧ A < π/2)
  (hB : 0 < B ∧ B < π/2)
  (h_cosA_gt_sinB : Real.cos A > Real.sin B) :
  π - (A + B) > π/2 ∧ π - (A + B) < π :=
by
  sorry

end NUMINAMATH_GPT_triangle_obtuse_l597_59759


namespace NUMINAMATH_GPT_largest_five_digit_integer_with_conditions_l597_59763

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * ((n / 10000) % 10)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + ((n / 10000) % 10)

theorem largest_five_digit_integer_with_conditions :
  ∃ n : ℕ, is_five_digit n ∧ digits_product n = 40320 ∧ digits_sum n < 35 ∧
  ∀ m : ℕ, is_five_digit m ∧ digits_product m = 40320 ∧ digits_sum m < 35 → n ≥ m :=
sorry

end NUMINAMATH_GPT_largest_five_digit_integer_with_conditions_l597_59763


namespace NUMINAMATH_GPT_minimum_value_of_3x_plus_4y_l597_59761

theorem minimum_value_of_3x_plus_4y :
  ∀ (x y : ℝ), 0 < x → 0 < y → x + 3 * y = 5 * x * y → (3 * x + 4 * y) ≥ 24 / 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_3x_plus_4y_l597_59761


namespace NUMINAMATH_GPT_gum_pieces_bought_correct_l597_59794

-- Define initial number of gum pieces
def initial_gum_pieces : ℕ := 10

-- Define number of friends Adrianna gave gum to
def friends_given_gum : ℕ := 11

-- Define the number of pieces Adrianna has left
def remaining_gum_pieces : ℕ := 2

-- Define a function to calculate the number of gum pieces Adrianna bought at the store
def gum_pieces_bought (initial_gum : ℕ) (given_gum : ℕ) (remaining_gum : ℕ) : ℕ :=
  (given_gum + remaining_gum) - initial_gum

-- Now state the theorem to prove the number of pieces bought is 3
theorem gum_pieces_bought_correct : 
  gum_pieces_bought initial_gum_pieces friends_given_gum remaining_gum_pieces = 3 :=
by
  sorry

end NUMINAMATH_GPT_gum_pieces_bought_correct_l597_59794


namespace NUMINAMATH_GPT_distance_between_intersections_l597_59716

open Function

def cube_vertices : List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0), (5, 0, 5), (5, 5, 0), (5, 5, 5)]

def intersecting_points : List (ℝ × ℝ × ℝ) :=
  [(0, 3, 0), (2, 0, 0), (2, 5, 5)]

noncomputable def plane_distance_between_points : ℝ :=
  let S := (11 / 3, 0, 5)
  let T := (0, 5, 4)
  Real.sqrt ((11 / 3 - 0)^2 + (0 - 5)^2 + (5 - 4)^2)

theorem distance_between_intersections : plane_distance_between_points = Real.sqrt (355 / 9) :=
  sorry

end NUMINAMATH_GPT_distance_between_intersections_l597_59716


namespace NUMINAMATH_GPT_sugar_percentage_of_second_solution_l597_59727

theorem sugar_percentage_of_second_solution :
  ∀ (W : ℝ) (P : ℝ),
  (0.10 * W * (3 / 4) + P / 100 * (1 / 4) * W = 0.18 * W) → 
  (P = 42) :=
by
  intros W P h
  sorry

end NUMINAMATH_GPT_sugar_percentage_of_second_solution_l597_59727


namespace NUMINAMATH_GPT_time_morning_is_one_l597_59725

variable (D : ℝ)  -- Define D as the distance between the two points.

def morning_speed := 20 -- Morning speed (km/h)
def afternoon_speed := 10 -- Afternoon speed (km/h)
def time_difference := 1 -- Time difference (hour)

-- Proving that the morning time t_m is equal to 1 hour
theorem time_morning_is_one (t_m t_a : ℝ) 
  (h1 : t_m - t_a = time_difference) 
  (h2 : D = morning_speed * t_m) 
  (h3 : D = afternoon_speed * t_a) : 
  t_m = 1 := 
by
  sorry

end NUMINAMATH_GPT_time_morning_is_one_l597_59725


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l597_59779

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n+1) = a n * q) → 
  (a 1 + a 5 = 17) → 
  (a 2 * a 4 = 16) → 
  (∀ i j, i < j → a i < a j) → 
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l597_59779


namespace NUMINAMATH_GPT_fitted_ball_volume_l597_59703

noncomputable def volume_of_fitted_ball (d_ball d_h1 r_h1 d_h2 r_h2 : ℝ) : ℝ :=
  let r_ball := d_ball / 2
  let v_ball := (4 / 3) * Real.pi * r_ball^3
  let r_hole1 := r_h1
  let r_hole2 := r_h2
  let v_hole1 := Real.pi * r_hole1^2 * d_h1
  let v_hole2 := Real.pi * r_hole2^2 * d_h2
  v_ball - 2 * v_hole1 - v_hole2

theorem fitted_ball_volume :
  volume_of_fitted_ball 24 10 (3 / 2) 10 2 = 2219 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_fitted_ball_volume_l597_59703


namespace NUMINAMATH_GPT_tony_rope_length_l597_59793

-- Define the lengths of the individual ropes.
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]

-- Define the total number of ropes Tony has.
def num_ropes : ℕ := rope_lengths.length

-- Calculate the total length of ropes before tying them together.
def total_length_before_tying : ℝ := rope_lengths.sum

-- Define the length lost per knot.
def length_lost_per_knot : ℝ := 1.2

-- Calculate the total number of knots needed.
def num_knots : ℕ := num_ropes - 1

-- Calculate the total length lost due to knots.
def total_length_lost : ℝ := num_knots * length_lost_per_knot

-- Calculate the total length of the rope after tying them all together.
def total_length_after_tying : ℝ := total_length_before_tying - total_length_lost

-- The theorem we want to prove.
theorem tony_rope_length : total_length_after_tying = 35 :=
by sorry

end NUMINAMATH_GPT_tony_rope_length_l597_59793


namespace NUMINAMATH_GPT_no_cubic_solution_l597_59765

theorem no_cubic_solution (t : ℤ) : ¬ ∃ k : ℤ, (7 * t + 3 = k ^ 3) := by
  sorry

end NUMINAMATH_GPT_no_cubic_solution_l597_59765


namespace NUMINAMATH_GPT_sin_cos_identity_second_quadrant_l597_59713

open Real

theorem sin_cos_identity_second_quadrant (α : ℝ) (hcos : cos α < 0) (hsin : sin α > 0) :
  (sin α / cos α) * sqrt ((1 / (sin α)^2) - 1) = -1 :=
sorry

end NUMINAMATH_GPT_sin_cos_identity_second_quadrant_l597_59713


namespace NUMINAMATH_GPT_perimeter_T2_l597_59738

def Triangle (a b c : ℝ) :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem perimeter_T2 (a b c : ℝ) (h : Triangle a b c) (ha : a = 10) (hb : b = 15) (hc : c = 20) : 
  let AM := a / 2
  let BN := b / 2
  let CP := c / 2
  0 < AM ∧ 0 < BN ∧ 0 < CP →
  AM + BN + CP = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_T2_l597_59738


namespace NUMINAMATH_GPT_arithmetic_sequence_second_term_l597_59749

theorem arithmetic_sequence_second_term (a d : ℤ)
  (h1 : a + 11 * d = 11)
  (h2 : a + 12 * d = 14) :
  a + d = -19 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_second_term_l597_59749


namespace NUMINAMATH_GPT_g_eq_one_l597_59748

theorem g_eq_one (g : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), g (x - y) = g x * g y) 
  (h2 : ∀ (x : ℝ), g x ≠ 0) : 
  g 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_eq_one_l597_59748


namespace NUMINAMATH_GPT_subtract_30_divisible_l597_59702

theorem subtract_30_divisible (n : ℕ) (d : ℕ) (r : ℕ) 
  (h1 : n = 13602) (h2 : d = 87) (h3 : r = 30) 
  (h4 : n % d = r) : (n - r) % d = 0 :=
by
  -- Skipping the proof as it's not required
  sorry

end NUMINAMATH_GPT_subtract_30_divisible_l597_59702


namespace NUMINAMATH_GPT_line_does_not_pass_third_quadrant_l597_59786

theorem line_does_not_pass_third_quadrant (a b c x y : ℝ) (h_ac : a * c < 0) (h_bc : b * c < 0) :
  ¬(x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_third_quadrant_l597_59786


namespace NUMINAMATH_GPT_distance_between_Q_and_R_l597_59711

noncomputable def distance_QR : ℝ :=
  let DE : ℝ := 9
  let EF : ℝ := 12
  let DF : ℝ := 15
  let N : ℝ := 7.5
  let QF : ℝ := (N * DF) / EF
  let QD : ℝ := DF - QF
  let QR : ℝ := (QD * DF) / EF
  QR

theorem distance_between_Q_and_R 
  (DE EF DF N QF QD QR : ℝ )
  (h1 : DE = 9)
  (h2 : EF = 12)
  (h3 : DF = 15)
  (h4 : N = DF / 2)
  (h5 : QF = N * DF / EF)
  (h6 : QD = DF - QF)
  (h7 : QR = QD * DF / EF) :
  QR = 7.03125 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_Q_and_R_l597_59711


namespace NUMINAMATH_GPT_find_k_plus_m_l597_59710

def initial_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
def initial_count := 9

def new_list_sum (m k : ℕ) := initial_sum + 8 * m + 9 * k
def new_list_count (m k : ℕ) := initial_count + m + k

def average_eq_73 (m k : ℕ) := (new_list_sum m k : ℝ) / (new_list_count m k : ℝ) = 7.3

theorem find_k_plus_m : ∃ (m k : ℕ), average_eq_73 m k ∧ (k + m = 21) :=
by
  sorry

end NUMINAMATH_GPT_find_k_plus_m_l597_59710


namespace NUMINAMATH_GPT_notebook_price_l597_59729

theorem notebook_price (students_buying_notebooks n c : ℕ) (total_students : ℕ := 36) (total_cost : ℕ := 990) :
  students_buying_notebooks > 18 ∧ c > n ∧ students_buying_notebooks * n * c = total_cost → c = 15 :=
by
  sorry

end NUMINAMATH_GPT_notebook_price_l597_59729


namespace NUMINAMATH_GPT_problems_per_page_l597_59778

-- Define the initial conditions
def total_problems : ℕ := 101
def finished_problems : ℕ := 47
def remaining_pages : ℕ := 6

-- State the theorem
theorem problems_per_page : 54 / remaining_pages = 9 :=
by
  -- Sorry is used to ignore the proof step
  sorry

end NUMINAMATH_GPT_problems_per_page_l597_59778


namespace NUMINAMATH_GPT_ant_food_cost_l597_59762

-- Definitions for the conditions
def number_of_ants : ℕ := 400
def food_per_ant : ℕ := 2
def job_charge : ℕ := 5
def leaf_charge : ℕ := 1 / 100 -- 1 penny is 1 cent which is 0.01 dollars
def leaves_raked : ℕ := 6000
def jobs_completed : ℕ := 4

-- Compute the total money earned from jobs
def money_from_jobs : ℕ := jobs_completed * job_charge

-- Compute the total money earned from raking leaves
def money_from_leaves : ℕ := leaves_raked * leaf_charge

-- Compute the total money earned
def total_money_earned : ℕ := money_from_jobs + money_from_leaves

-- Compute the total ounces of food needed
def total_food_needed : ℕ := number_of_ants * food_per_ant

-- Calculate the cost per ounce of food
def cost_per_ounce : ℕ := total_money_earned / total_food_needed

theorem ant_food_cost :
  cost_per_ounce = 1 / 10 := sorry

end NUMINAMATH_GPT_ant_food_cost_l597_59762


namespace NUMINAMATH_GPT_range_of_a_l597_59747

variable {a : ℝ}

def A (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a^2 + 1)) < 0 }

theorem range_of_a (a : ℝ) : B a ⊆ A a ↔ (a = -1 / 2) ∨ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l597_59747


namespace NUMINAMATH_GPT_triangle_inequality_l597_59741

variable {a b c : ℝ}

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (habc1 : a + b > c) (habc2 : a + c > b) (habc3 : b + c > a) :
  (a / (b + c) + b / (c + a) + c / (a + b) < 2) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l597_59741


namespace NUMINAMATH_GPT_prob_yellow_and_straight_l597_59714

-- Definitions of probabilities given in the problem
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2

-- Derived probability of picking a yellow flower
def prob_yellow : ℚ := 1 - prob_green

-- Statement to prove
theorem prob_yellow_and_straight : prob_yellow * prob_straight = 1 / 6 :=
by
  -- sorry is used here to skip the proof.
  sorry

end NUMINAMATH_GPT_prob_yellow_and_straight_l597_59714


namespace NUMINAMATH_GPT_tomatoes_left_l597_59737

theorem tomatoes_left (initial_tomatoes picked_yesterday picked_today : ℕ)
    (h_initial : initial_tomatoes = 171)
    (h_picked_yesterday : picked_yesterday = 134)
    (h_picked_today : picked_today = 30) :
    initial_tomatoes - picked_yesterday - picked_today = 7 :=
by
    sorry

end NUMINAMATH_GPT_tomatoes_left_l597_59737


namespace NUMINAMATH_GPT_simplify_expression_l597_59734

theorem simplify_expression (x : ℤ) : 
  (2 * x ^ 13 + 3 * x ^ 12 - 4 * x ^ 9 + 5 * x ^ 7) + 
  (8 * x ^ 11 - 2 * x ^ 9 + 3 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9) + 
  (x ^ 13 + 4 * x ^ 12 + x ^ 11 + 9 * x ^ 9) = 
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 3 * x ^ 9 + 8 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l597_59734


namespace NUMINAMATH_GPT_consistent_values_for_a_l597_59782

def eq1 (x a : ℚ) : Prop := 10 * x^2 + x - a - 11 = 0
def eq2 (x a : ℚ) : Prop := 4 * x^2 + (a + 4) * x - 3 * a - 8 = 0

theorem consistent_values_for_a : ∃ x, (eq1 x 0 ∧ eq2 x 0) ∨ (eq1 x (-2) ∧ eq2 x (-2)) ∨ (eq1 x (54) ∧ eq2 x (54)) :=
by
  sorry

end NUMINAMATH_GPT_consistent_values_for_a_l597_59782


namespace NUMINAMATH_GPT_price_per_liter_l597_59792

theorem price_per_liter (cost : ℕ) (bottles : ℕ) (liters_per_bottle : ℕ) (total_cost : ℕ) (total_liters : ℕ) :
  bottles = 6 → liters_per_bottle = 2 → total_cost = 12 → total_liters = 12 → cost = total_cost / total_liters → cost = 1 :=
by
  intros h_bottles h_liters_per_bottle h_total_cost h_total_liters h_cost_div
  sorry

end NUMINAMATH_GPT_price_per_liter_l597_59792


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l597_59712

def geometric_seq_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem sum_of_first_six_terms (a : ℕ) (r : ℕ) (h1 : r = 2) (h2 : a * (1 + r + r^2) = 3) :
  geometric_seq_sum a r 6 = 27 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_l597_59712


namespace NUMINAMATH_GPT_find_a_plus_b_l597_59758

theorem find_a_plus_b (a b : ℝ)
  (h1 : ab^2 = 0)
  (h2 : 2 * a^2 * b = 0)
  (h3 : a^3 + b^2 = 0)
  (h4 : ab = 1) : a + b = -2 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l597_59758


namespace NUMINAMATH_GPT_correct_calculation_l597_59744

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end NUMINAMATH_GPT_correct_calculation_l597_59744


namespace NUMINAMATH_GPT_ellipse_eccentricity_l597_59773

noncomputable def eccentricity_of_ellipse (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_eccentricity (F1 A : ℝ) (v : ℝ) (a c : ℝ)
  (h1 : 4 * a = 10 * (a - c))
  (h2 : F1 = 0 ∧ A = 0 ∧ v ≠ 0) :
  eccentricity_of_ellipse a c = 3 / 5 := by
sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l597_59773


namespace NUMINAMATH_GPT_farm_horses_cows_ratio_l597_59715

variable (x y : ℕ)  -- x is the base variable related to the initial counts, y is the number of horses sold (and cows bought)

theorem farm_horses_cows_ratio (h1 : 4 * x / x = 4)
    (h2 : 13 * (x + y) = 7 * (4 * x - y))
    (h3 : 4 * x - y = (x + y) + 30) :
    y = 15 := sorry

end NUMINAMATH_GPT_farm_horses_cows_ratio_l597_59715


namespace NUMINAMATH_GPT_nehas_mother_age_l597_59724

variables (N M : ℕ)

axiom age_condition1 : M - 12 = 4 * (N - 12)
axiom age_condition2 : M + 12 = 2 * (N + 12)

theorem nehas_mother_age : M = 60 :=
by
  -- Sorry added to skip the proof
  sorry

end NUMINAMATH_GPT_nehas_mother_age_l597_59724


namespace NUMINAMATH_GPT_inequality_proof_l597_59742

theorem inequality_proof
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (6841 * x - 1) / 9973 + (9973 * y - 1) / 6841 = z) :
  x / 9973 + y / 6841 > 1 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l597_59742


namespace NUMINAMATH_GPT_percentage_shaded_l597_59785

def area_rect (width height : ℝ) : ℝ := width * height

def overlap_area (side_length : ℝ) (width_rect : ℝ) (length_rect: ℝ) (length_total: ℝ) : ℝ :=
  (side_length - (length_total - length_rect)) * width_rect

theorem percentage_shaded (sqr_side length_rect width_rect total_length total_width : ℝ) (h1 : sqr_side = 12) (h2 : length_rect = 9) (h3 : width_rect = 12)
  (h4 : total_length = 18) (h5 : total_width = 12) :
  (overlap_area sqr_side width_rect length_rect total_length) / (area_rect total_width total_length) * 100 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_shaded_l597_59785


namespace NUMINAMATH_GPT_average_coins_collected_per_day_l597_59798

noncomputable def average_coins (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (a + (a + (n - 1) * d)) / 2

theorem average_coins_collected_per_day :
  average_coins 10 5 7 = 25 := by
  sorry

end NUMINAMATH_GPT_average_coins_collected_per_day_l597_59798


namespace NUMINAMATH_GPT_domain_of_f_l597_59756

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (x - 2)

theorem domain_of_f : {x : ℝ | x > -1 ∧ x ≠ 2} = {x : ℝ | x ∈ Set.Ioo (-1) 2 ∪ Set.Ioi 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_f_l597_59756


namespace NUMINAMATH_GPT_proof_not_sufficient_nor_necessary_l597_59704

noncomputable def not_sufficient_nor_necessary (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : Prop :=
  ¬ ((a > b) → (Real.log b / Real.log a < 1)) ∧ ¬ ((Real.log b / Real.log a < 1) → (a > b))

theorem proof_not_sufficient_nor_necessary (a b: ℝ) (h₁: 0 < a) (h₂: 0 < b) :
  not_sufficient_nor_necessary a b h₁ h₂ :=
  sorry

end NUMINAMATH_GPT_proof_not_sufficient_nor_necessary_l597_59704


namespace NUMINAMATH_GPT_find_n_divisors_l597_59781

theorem find_n_divisors (n : ℕ) (h1 : 2287 % n = 2028 % n)
                        (h2 : 2028 % n = 1806 % n) : n = 37 := 
by
  sorry

end NUMINAMATH_GPT_find_n_divisors_l597_59781


namespace NUMINAMATH_GPT_cube_volume_surface_area_l597_59732

theorem cube_volume_surface_area (x : ℝ) (s : ℝ)
  (h1 : s^3 = 3 * x)
  (h2 : 6 * s^2 = 6 * x) :
  x = 3 :=
by sorry

end NUMINAMATH_GPT_cube_volume_surface_area_l597_59732


namespace NUMINAMATH_GPT_probability_of_green_l597_59774

-- Define the conditions
def P_R : ℝ := 0.15
def P_O : ℝ := 0.35
def P_B : ℝ := 0.2
def total_probability (P_Y P_G : ℝ) : Prop := P_R + P_O + P_B + P_Y + P_G = 1

-- State the theorem to be proven
theorem probability_of_green (P_Y : ℝ) (P_G : ℝ) (h : total_probability P_Y P_G) (P_Y_assumption : P_Y = 0.15) : P_G = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_green_l597_59774


namespace NUMINAMATH_GPT_fivefold_composition_l597_59768

def f (x : ℚ) : ℚ := -2 / x

theorem fivefold_composition :
  f (f (f (f (f (3))))) = -2 / 3 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fivefold_composition_l597_59768


namespace NUMINAMATH_GPT_percentage_of_number_l597_59722

theorem percentage_of_number (P : ℝ) (h : 0.10 * 3200 - 190 = P * 650) :
  P = 0.2 :=
sorry

end NUMINAMATH_GPT_percentage_of_number_l597_59722


namespace NUMINAMATH_GPT_full_price_tickets_revenue_l597_59753

theorem full_price_tickets_revenue (f h p : ℕ) (h1 : f + h + 12 = 160) (h2 : f * p + h * (p / 2) + 12 * (2 * p) = 2514) :  f * p = 770 := 
sorry

end NUMINAMATH_GPT_full_price_tickets_revenue_l597_59753


namespace NUMINAMATH_GPT_exists_two_digit_number_l597_59760

theorem exists_two_digit_number :
  ∃ x y : ℕ, (1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ (10 * x + y = (x + y) * (x - y)) ∧ (10 * x + y = 48) :=
by
  sorry

end NUMINAMATH_GPT_exists_two_digit_number_l597_59760


namespace NUMINAMATH_GPT_unique_function_satisfying_conditions_l597_59705

open Nat

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (f 1 = 1) ∧ (∀ n, f n * f (n + 2) = (f (n + 1))^2 + 1997)

theorem unique_function_satisfying_conditions :
  (∃! f : ℕ → ℕ, satisfies_conditions f) :=
sorry

end NUMINAMATH_GPT_unique_function_satisfying_conditions_l597_59705


namespace NUMINAMATH_GPT_length_AE_l597_59700

theorem length_AE (A B C D E : Type) 
  (AB AC AD AE : ℝ) 
  (angle_BAC : ℝ)
  (h1 : AB = 4.5) 
  (h2 : AC = 5) 
  (h3 : angle_BAC = 30) 
  (h4 : AD = 1.5) 
  (h5 : AD / AB = AE / AC) : 
  AE = 1.6667 := 
sorry

end NUMINAMATH_GPT_length_AE_l597_59700


namespace NUMINAMATH_GPT_group_B_equal_l597_59746

noncomputable def neg_two_pow_three := (-2)^3
noncomputable def minus_two_pow_three := -(2^3)

theorem group_B_equal : neg_two_pow_three = minus_two_pow_three :=
by sorry

end NUMINAMATH_GPT_group_B_equal_l597_59746


namespace NUMINAMATH_GPT_at_least_six_stones_empty_l597_59731

def frogs_on_stones (a : Fin 23 → Fin 23) (k : Nat) : Fin 22 → Fin 23 :=
  fun i => (a i + i.1 * k) % 23

theorem at_least_six_stones_empty 
  (a : Fin 22 → Fin 23) :
  ∃ k : Nat, ∀ (s : Fin 23), ∃ (j : Fin 22), frogs_on_stones (fun i => a i) k j ≠ s ↔ ∃! t : Fin 23, ∃! j, (frogs_on_stones (fun i => a i) k j) = t := 
  sorry

end NUMINAMATH_GPT_at_least_six_stones_empty_l597_59731


namespace NUMINAMATH_GPT_hall_ratio_l597_59764

theorem hall_ratio (w l : ℕ) (h1 : w * l = 450) (h2 : l - w = 15) : w / l = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_hall_ratio_l597_59764


namespace NUMINAMATH_GPT_problem_l597_59788

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x|

theorem problem :
  (∀ x, f x ≤ 1) ∧
  (∃ x, f x = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 
    ∃ x, (x = (a^2 / (b + 1) + b^2 / (a + 1)) ∧ x = 1 / 3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l597_59788


namespace NUMINAMATH_GPT_each_member_score_l597_59777

def total_members : ℝ := 5.0
def members_didnt_show_up : ℝ := 2.0
def total_points_by_showed_up_members : ℝ := 6.0

theorem each_member_score
  (h1 : total_members - members_didnt_show_up = 3.0)
  (h2 : total_points_by_showed_up_members = 6.0) :
  total_points_by_showed_up_members / (total_members - members_didnt_show_up) = 2.0 :=
sorry

end NUMINAMATH_GPT_each_member_score_l597_59777


namespace NUMINAMATH_GPT_range_of_k_l597_59750

noncomputable def f (k : ℝ) (x : ℝ) := 1 - k * x^2
noncomputable def g (x : ℝ) := Real.cos x

theorem range_of_k (k : ℝ) : (∀ x : ℝ, f k x < g x) ↔ k ≥ (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l597_59750


namespace NUMINAMATH_GPT_quadratic_sum_roots_l597_59797

-- We define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- The function f passes through points (r, k) and (s, k)
variables (a b c r s k : ℝ)
variable (ha : a ≠ 0)
variable (hr : f a b c r = k)
variable (hs : f a b c s = k)

-- What we want to prove
theorem quadratic_sum_roots :
  f a b c (r + s) = c :=
sorry

end NUMINAMATH_GPT_quadratic_sum_roots_l597_59797


namespace NUMINAMATH_GPT_range_of_xy_l597_59719

-- Given conditions
variables {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1)

-- To Prove
theorem range_of_xy (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1) : 64 ≤ x * y :=
sorry

end NUMINAMATH_GPT_range_of_xy_l597_59719


namespace NUMINAMATH_GPT_value_of_x_l597_59736

theorem value_of_x (x y z w : ℕ) (h1 : x = y + 7) (h2 : y = z + 12) (h3 : z = w + 25) (h4 : w = 90) : x = 134 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l597_59736


namespace NUMINAMATH_GPT_gcd_of_A_and_B_l597_59770

theorem gcd_of_A_and_B (A B : ℕ) (h_lcm : Nat.lcm A B = 120) (h_ratio : A * 4 = B * 3) : Nat.gcd A B = 10 :=
sorry

end NUMINAMATH_GPT_gcd_of_A_and_B_l597_59770


namespace NUMINAMATH_GPT_enrollment_difference_l597_59707

theorem enrollment_difference :
  let M := 1500
  let S := 2100
  let L := 2700
  let R := 1800
  let B := 900
  max M (max S (max L (max R B))) - min M (min S (min L (min R B))) = 1800 := 
by 
  sorry

end NUMINAMATH_GPT_enrollment_difference_l597_59707


namespace NUMINAMATH_GPT_coordinates_on_y_axis_l597_59706

theorem coordinates_on_y_axis (a : ℝ) 
  (h : (a - 3) = 0) : 
  P = (0, -1) :=
by 
  have ha : a = 3 := by sorry
  subst ha
  sorry

end NUMINAMATH_GPT_coordinates_on_y_axis_l597_59706


namespace NUMINAMATH_GPT_analysis_method_proves_sufficient_condition_l597_59733

-- Definitions and conditions from part (a)
def analysis_method_traces_cause_from_effect : Prop := true
def analysis_method_seeks_sufficient_conditions : Prop := true
def analysis_method_finds_conditions_for_inequality : Prop := true

-- The statement to be proven
theorem analysis_method_proves_sufficient_condition :
  analysis_method_finds_conditions_for_inequality →
  analysis_method_traces_cause_from_effect →
  analysis_method_seeks_sufficient_conditions →
  (B = "Sufficient condition") :=
by 
  sorry

end NUMINAMATH_GPT_analysis_method_proves_sufficient_condition_l597_59733


namespace NUMINAMATH_GPT_Eric_rent_days_l597_59790

-- Define the conditions given in the problem
def daily_rate := 50.00
def rate_14_days := 500.00
def total_cost := 800.00

-- State the problem as a theorem in Lean
theorem Eric_rent_days : ∀ (d : ℕ), (d : ℕ) = 20 :=
by
  sorry

end NUMINAMATH_GPT_Eric_rent_days_l597_59790


namespace NUMINAMATH_GPT_total_amount_is_2500_l597_59789

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_is_2500_l597_59789


namespace NUMINAMATH_GPT_vacation_cost_division_l597_59780

theorem vacation_cost_division (n : ℕ) (total_cost : ℕ) 
  (cost_difference : ℕ)
  (cost_per_person_5 : ℕ) :
  total_cost = 1000 → 
  cost_difference = 50 → 
  cost_per_person_5 = total_cost / 5 →
  (total_cost / n) = cost_per_person_5 + cost_difference → 
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_vacation_cost_division_l597_59780


namespace NUMINAMATH_GPT_find_b_l597_59769

theorem find_b (a b : ℝ) (h : ∀ x, 2 * x^2 - a * x + 4 < 0 ↔ 1 < x ∧ x < b) : b = 2 :=
sorry

end NUMINAMATH_GPT_find_b_l597_59769


namespace NUMINAMATH_GPT_cost_of_sneakers_l597_59728

theorem cost_of_sneakers (saved money per_action_figure final_money cost : ℤ) 
  (h1 : saved = 15) 
  (h2 : money = 10) 
  (h3 : per_action_figure = 10) 
  (h4 : final_money = 25) 
  (h5 : money * per_action_figure + saved - cost = final_money) 
  : cost = 90 := 
sorry

end NUMINAMATH_GPT_cost_of_sneakers_l597_59728


namespace NUMINAMATH_GPT_quadratic_roots_l597_59791

theorem quadratic_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + bx + c = 0 ↔ x^2 - 5 * x + 2 = 0):
  c / b = -4 / 21 :=
  sorry

end NUMINAMATH_GPT_quadratic_roots_l597_59791


namespace NUMINAMATH_GPT_solution_set_of_inequality_l597_59735

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l597_59735


namespace NUMINAMATH_GPT_boy_completes_work_in_nine_days_l597_59743

theorem boy_completes_work_in_nine_days :
  let M := (1 : ℝ) / 6
  let W := (1 : ℝ) / 18
  let B := (1 / 3 : ℝ) - M - W
  B = (1 : ℝ) / 9 := by
    sorry

end NUMINAMATH_GPT_boy_completes_work_in_nine_days_l597_59743
