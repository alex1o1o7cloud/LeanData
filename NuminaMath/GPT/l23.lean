import Mathlib

namespace park_is_square_l23_2322

-- Defining the concept of a square field
def square_field : ℕ := 4

-- Given condition: The sum of the right angles from the park and the square field
axiom angles_sum (park_angles : ℕ) : park_angles + square_field = 8

-- The theorem to be proven
theorem park_is_square (park_angles : ℕ) (h : park_angles + square_field = 8) : park_angles = 4 :=
by sorry

end park_is_square_l23_2322


namespace no_integer_solutions_for_sum_of_squares_l23_2333

theorem no_integer_solutions_for_sum_of_squares :
  ∀ a b c : ℤ, a^2 + b^2 + c^2 ≠ 20122012 := 
by sorry

end no_integer_solutions_for_sum_of_squares_l23_2333


namespace simplify_expression_l23_2377

theorem simplify_expression :
  18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 :=
by
  sorry

end simplify_expression_l23_2377


namespace number_of_kiwis_l23_2302

/-
There are 500 pieces of fruit in a crate. One fourth of the fruits are apples,
20% are oranges, one fifth are strawberries, and the rest are kiwis.
Prove that the number of kiwis is 175.
-/

theorem number_of_kiwis (total_fruits apples oranges strawberries kiwis : ℕ)
  (h1 : total_fruits = 500)
  (h2 : apples = total_fruits / 4)
  (h3 : oranges = 20 * total_fruits / 100)
  (h4 : strawberries = total_fruits / 5)
  (h5 : kiwis = total_fruits - (apples + oranges + strawberries)) :
  kiwis = 175 :=
sorry

end number_of_kiwis_l23_2302


namespace toothpaste_runs_out_in_two_days_l23_2303

noncomputable def toothpaste_capacity := 90
noncomputable def dad_usage_per_brushing := 4
noncomputable def mom_usage_per_brushing := 3
noncomputable def anne_usage_per_brushing := 2
noncomputable def brother_usage_per_brushing := 1
noncomputable def sister_usage_per_brushing := 1

noncomputable def dad_brushes_per_day := 4
noncomputable def mom_brushes_per_day := 4
noncomputable def anne_brushes_per_day := 4
noncomputable def brother_brushes_per_day := 4
noncomputable def sister_brushes_per_day := 2

noncomputable def total_daily_usage :=
  dad_usage_per_brushing * dad_brushes_per_day + 
  mom_usage_per_brushing * mom_brushes_per_day + 
  anne_usage_per_brushing * anne_brushes_per_day + 
  brother_usage_per_brushing * brother_brushes_per_day + 
  sister_usage_per_brushing * sister_brushes_per_day

theorem toothpaste_runs_out_in_two_days :
  toothpaste_capacity / total_daily_usage = 2 := by
  -- Proof omitted
  sorry

end toothpaste_runs_out_in_two_days_l23_2303


namespace part1_part2_l23_2332

-- Statement for part (1)
theorem part1 (m : ℝ) : 
  (∀ x1 x2 : ℝ, (m - 1) * x1^2 + 3 * x1 - 2 = 0 ∧ 
               (m - 1) * x2^2 + 3 * x2 - 2 = 0 ∧ x1 ≠ x2) ↔ m > -1/8 :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 2 = 0 ∧ ∀ y : ℝ, (m - 1) * y^2 + 3 * y - 2 = 0 → y = x) ↔ 
  (m = 1 ∨ m = -1/8) :=
sorry

end part1_part2_l23_2332


namespace race_time_A_l23_2300

noncomputable def time_for_A_to_cover_distance (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) : ℝ :=
  let speed_of_B := distance / time_of_B
  let time_for_B_to_cover_remaining := remaining_distance_for_B / speed_of_B
  time_for_B_to_cover_remaining

theorem race_time_A (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) :
  distance = 100 ∧ time_of_B = 25 ∧ remaining_distance_for_B = distance - 20 →
  time_for_A_to_cover_distance distance time_of_B remaining_distance_for_B = 20 :=
by
  intros h
  rcases h with ⟨h_distance, h_time_of_B, h_remaining_distance_for_B⟩
  rw [h_distance, h_time_of_B, h_remaining_distance_for_B]
  sorry

end race_time_A_l23_2300


namespace problem1_problem2_l23_2324

-- Define the first problem: For positive real numbers a and b,
-- with the condition a + b = 2, show that the minimum value of 
-- (1 / (1 + a) + 4 / (1 + b)) is 9/4.
theorem problem1 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  1 / (1 + a) + 4 / (1 + b) ≥ 9 / 4 :=
sorry

-- Define the second problem: For any positive real numbers a and b,
-- prove that a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1).
theorem problem2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end problem1_problem2_l23_2324


namespace min_sum_of_intercepts_l23_2301

-- Definitions based on conditions
def line (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = a * b
def point_on_line (a b : ℝ) : Prop := line a b 1 1

-- Main theorem statement
theorem min_sum_of_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_point : point_on_line a b) : 
  a + b >= 4 :=
sorry

end min_sum_of_intercepts_l23_2301


namespace range_of_root_difference_l23_2358

variable (a b c d : ℝ)
variable (x1 x2 : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_of_root_difference
  (h1 : a ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hroot1 : f a b c x1 = 0)
  (hroot2 : f a b c x2 = 0)
  : |x1 - x2| ∈ Set.Ico (Real.sqrt 3 / 3) (2 / 3) := sorry

end range_of_root_difference_l23_2358


namespace more_roses_than_orchids_l23_2383

theorem more_roses_than_orchids (roses orchids : ℕ) (h1 : roses = 12) (h2 : orchids = 2) : roses - orchids = 10 := by
  sorry

end more_roses_than_orchids_l23_2383


namespace walking_rate_on_escalator_l23_2368

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 160)
  (time_taken : ℝ := 8)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) :
  v = 8 :=
by
  sorry

end walking_rate_on_escalator_l23_2368


namespace range_of_a_l23_2359

theorem range_of_a (m a : ℝ) (h1 : m < a) (h2 : m ≤ -1) : a > -1 :=
by sorry

end range_of_a_l23_2359


namespace angle_in_third_quadrant_half_l23_2342

theorem angle_in_third_quadrant_half {
  k : ℤ 
} (h1: (k * 360 + 180) < α) (h2 : α < k * 360 + 270) :
  (k * 180 + 90) < (α / 2) ∧ (α / 2) < (k * 180 + 135) :=
sorry

end angle_in_third_quadrant_half_l23_2342


namespace HorseKeepsPower_l23_2392

/-- If the Little Humpbacked Horse does not eat for seven days or does not sleep for seven days,
    he will lose his magic power. Suppose he did not eat or sleep for a whole week. 
    Prove that by the end of the seventh day, he must do the activity he did not do right before 
    the start of the first period of seven days in order to keep his power. -/
theorem HorseKeepsPower (eat sleep : ℕ → Prop) :
  (∀ (n : ℕ), (n ≥ 7 → ¬eat n) ∨ (n ≥ 7 → ¬sleep n)) →
  (∀ (n : ℕ), n < 7 → (¬eat n ∧ ¬sleep n)) →
  ∃ (t : ℕ), t > 7 → (eat t ∨ sleep t) :=
sorry

end HorseKeepsPower_l23_2392


namespace b6_b8_equals_16_l23_2349

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_arithmetic : ∃ d, ∀ n, a_seq (n + 1) = a_seq n + d
axiom b_geometric : ∃ r, ∀ n, b_seq (n + 1) = b_seq n * r
axiom a_nonzero : ∀ n, a_seq n ≠ 0
axiom a_eq : 2 * a_seq 3 - (a_seq 7)^2 + 2 * a_seq 11 = 0
axiom b7_eq_a7 : b_seq 7 = a_seq 7

theorem b6_b8_equals_16 : b_seq 6 * b_seq 8 = 16 := by
  sorry

end b6_b8_equals_16_l23_2349


namespace star_test_one_star_test_two_l23_2384

def star (x y : ℤ) : ℤ :=
  if x = 0 then Int.natAbs y
  else if y = 0 then Int.natAbs x
  else if (x < 0) = (y < 0) then Int.natAbs x + Int.natAbs y
  else -(Int.natAbs x + Int.natAbs y)

theorem star_test_one :
  star 11 (star 0 (-12)) = 23 :=
by
  sorry

theorem star_test_two (a : ℤ) :
  2 * (2 * star 1 a) - 1 = 3 * a ↔ a = 3 ∨ a = -5 :=
by
  sorry

end star_test_one_star_test_two_l23_2384


namespace find_a1_l23_2371

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

theorem find_a1 (h_arith : is_arithmetic_sequence a 3) (ha2 : a 2 = -5) : a 1 = -8 :=
sorry

end find_a1_l23_2371


namespace distance_covered_l23_2360

noncomputable def boat_speed_still_water : ℝ := 6.5
noncomputable def current_speed : ℝ := 2.5
noncomputable def time_taken : ℝ := 35.99712023038157

noncomputable def effective_speed_downstream (boat_speed_still_water current_speed : ℝ) : ℝ :=
  boat_speed_still_water + current_speed

noncomputable def convert_kmph_to_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

noncomputable def calculate_distance (speed_in_mps time_in_seconds : ℝ) : ℝ :=
  speed_in_mps * time_in_seconds

theorem distance_covered :
  calculate_distance (convert_kmph_to_mps (effective_speed_downstream boat_speed_still_water current_speed)) time_taken = 89.99280057595392 :=
by
  sorry

end distance_covered_l23_2360


namespace total_marbles_l23_2357

variables (y : ℝ) 

def first_friend_marbles : ℝ := 2 * y + 2
def second_friend_marbles : ℝ := y
def third_friend_marbles : ℝ := 3 * y - 1

theorem total_marbles :
  (first_friend_marbles y) + (second_friend_marbles y) + (third_friend_marbles y) = 6 * y + 1 :=
by
  sorry

end total_marbles_l23_2357


namespace sequence_formula_general_formula_l23_2379

open BigOperators

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * n + 2

def S_n (n : ℕ) : ℕ :=
  n^2 + 3 * n + 1

theorem sequence_formula :
  ∀ n, a_n n =
    if n = 1 then 5 else 2 * n + 2 := by
  sorry

theorem general_formula (n : ℕ) :
  a_n n =
    if n = 1 then S_n 1 else S_n n - S_n (n - 1) := by
  sorry

end sequence_formula_general_formula_l23_2379


namespace max_self_intersections_polyline_7_l23_2336

def max_self_intersections (n : ℕ) : ℕ :=
  if h : n > 2 then (n * (n - 3)) / 2 else 0

theorem max_self_intersections_polyline_7 :
  max_self_intersections 7 = 14 := 
sorry

end max_self_intersections_polyline_7_l23_2336


namespace problem1_problem2_l23_2321

-- Definition for the first problem: determine the number of arrangements when no box is empty and ball 3 is in box B
def arrangements_with_ball3_in_B_and_no_empty_box : ℕ :=
  12

theorem problem1 : arrangements_with_ball3_in_B_and_no_empty_box = 12 :=
  by
    sorry

-- Definition for the second problem: determine the number of arrangements when ball 1 is not in box A and ball 2 is not in box B
def arrangements_with_ball1_not_in_A_and_ball2_not_in_B : ℕ :=
  36

theorem problem2 : arrangements_with_ball1_not_in_A_and_ball2_not_in_B = 36 :=
  by
    sorry

end problem1_problem2_l23_2321


namespace inverse_prop_relation_l23_2378

theorem inverse_prop_relation (y₁ y₂ y₃ : ℝ) :
  (y₁ = (1 : ℝ) / (-1)) →
  (y₂ = (1 : ℝ) / (-2)) →
  (y₃ = (1 : ℝ) / (3)) →
  y₃ > y₂ ∧ y₂ > y₁ :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  constructor
  · norm_num
  · norm_num

end inverse_prop_relation_l23_2378


namespace expression_eq_l23_2389

theorem expression_eq (x : ℝ) : 
    (x + 1)^4 + 4 * (x + 1)^3 + 6 * (x + 1)^2 + 4 * (x + 1) + 1 = (x + 2)^4 := 
  sorry

end expression_eq_l23_2389


namespace find_x_l23_2339

theorem find_x :
  ∃ x : ℝ, (2020 + x)^2 = x^2 ∧ x = -1010 :=
sorry

end find_x_l23_2339


namespace equal_roots_of_quadratic_eq_l23_2311

theorem equal_roots_of_quadratic_eq (n : ℝ) : (∃ x : ℝ, (x^2 - x + n = 0) ∧ (Δ = 0)) ↔ n = 1 / 4 :=
by
  have h₁ : Δ = 0 := by sorry  -- The discriminant condition
  sorry  -- Placeholder for completing the theorem proof

end equal_roots_of_quadratic_eq_l23_2311


namespace abs_add_gt_abs_sub_l23_2337

variables {a b : ℝ}

theorem abs_add_gt_abs_sub (h : a * b > 0) : |a + b| > |a - b| :=
sorry

end abs_add_gt_abs_sub_l23_2337


namespace correct_proposition_D_l23_2373

theorem correct_proposition_D (a b c : ℝ) (h : a > b) : a - c > b - c :=
by
  sorry

end correct_proposition_D_l23_2373


namespace regular_polygon_interior_angle_l23_2352

theorem regular_polygon_interior_angle (S : ℝ) (n : ℕ) (h1 : S = 720) (h2 : (n - 2) * 180 = S) : 
  (S / n) = 120 := 
by
  sorry

end regular_polygon_interior_angle_l23_2352


namespace triangle_perimeter_correct_l23_2307

def side_a : ℕ := 15
def side_b : ℕ := 8
def side_c : ℕ := 10
def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter_correct :
  perimeter side_a side_b side_c = 33 := by
sorry

end triangle_perimeter_correct_l23_2307


namespace arithmetic_sequence_n_value_l23_2375

theorem arithmetic_sequence_n_value (a_1 d a_nm1 n : ℤ) (h1 : a_1 = -1) (h2 : d = 2) (h3 : a_nm1 = 15) :
    a_nm1 = a_1 + (n - 2) * d → n = 10 :=
by
  intros h
  sorry

end arithmetic_sequence_n_value_l23_2375


namespace number_to_multiply_l23_2365

theorem number_to_multiply (a b x : ℝ) (h1 : x * a = 4 * b) (h2 : a * b ≠ 0) (h3 : a / 4 = b / 3) : x = 3 :=
sorry

end number_to_multiply_l23_2365


namespace factors_multiple_of_120_l23_2312

theorem factors_multiple_of_120 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9 * 7^5) :
  ∃ k : ℕ, k = 8100 ∧ ∀ d : ℕ, d ∣ n ∧ 120 ∣ d ↔ ∃ a b c d : ℕ, 3 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 15 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 5 ∧ d = 2^a * 3^b * 5^c * 7^d :=
by
  sorry

end factors_multiple_of_120_l23_2312


namespace ralph_squares_count_l23_2350

def total_matchsticks := 50
def elvis_square_sticks := 4
def ralph_square_sticks := 8
def elvis_squares := 5
def leftover_sticks := 6

theorem ralph_squares_count : 
  ∃ R : ℕ, 
  (elvis_squares * elvis_square_sticks) + (R * ralph_square_sticks) + leftover_sticks = total_matchsticks ∧ R = 3 :=
by 
  sorry

end ralph_squares_count_l23_2350


namespace math_problem_l23_2388

theorem math_problem 
  (X : ℝ)
  (num1 : ℝ := 1 + 28/63)
  (num2 : ℝ := 8 + 7/16)
  (frac_sub1 : ℝ := 19/24 - 21/40)
  (frac_sub2 : ℝ := 1 + 28/63 - 17/21)
  (denom_calc : ℝ := 0.675 * 2.4 - 0.02) :
  0.125 * X / (frac_sub1 * num2) = (frac_sub2 * 0.7) / denom_calc → X = 5 := 
sorry

end math_problem_l23_2388


namespace distance_from_P_to_focus_l23_2398

-- Define the parabola equation and the definition of the point P
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the given condition that P's distance to the x-axis is 12
def point_P (x y : ℝ) : Prop := parabola x y ∧ |y| = 12

-- The Lean proof problem statement
theorem distance_from_P_to_focus :
  ∃ (x y : ℝ), point_P x y → dist (x, y) (4, 0) = 13 :=
by {
  sorry   -- proof to be completed
}

end distance_from_P_to_focus_l23_2398


namespace find_a_l23_2315

noncomputable def f (x a : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem find_a : (∃ a : ℝ, ((∀ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a ≤ -3) ∧ (∃ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a = -3)) ↔ a = Real.sqrt 6 + 2) :=
by
  sorry

end find_a_l23_2315


namespace opposite_of_2023_l23_2380

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l23_2380


namespace find_a_l23_2320

theorem find_a (a : ℝ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := sorry

end find_a_l23_2320


namespace Question_D_condition_l23_2382

theorem Question_D_condition (P Q : Prop) (h : P → Q) : ¬ Q → ¬ P :=
by sorry

end Question_D_condition_l23_2382


namespace vasya_drives_fraction_l23_2381

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l23_2381


namespace evaluate_expression_l23_2367

theorem evaluate_expression :
  ( ( ( 5 / 2 : ℚ ) / ( 7 / 12 : ℚ ) ) - ( 4 / 9 : ℚ ) ) = ( 242 / 63 : ℚ ) :=
by
  sorry

end evaluate_expression_l23_2367


namespace find_vector_l23_2354

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 3 + 2 * t)

noncomputable def line_m (s : ℝ) : ℝ × ℝ :=
  (-4 + 3 * s, 5 + 2 * s)

def vector_condition (v1 v2 : ℝ) : Prop :=
  v1 - v2 = 1

theorem find_vector :
  ∃ (v1 v2 : ℝ), vector_condition v1 v2 ∧ (v1, v2) = (3, 2) :=
sorry

end find_vector_l23_2354


namespace area_OMVK_l23_2328

theorem area_OMVK :
  ∀ (S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK : ℝ),
    S_OKCL = 6 →
    S_ONAM = 12 →
    S_ONBM = 24 →
    S_ABCD = 4 * (S_OKCL + S_ONAM) →
    S_OMVK = S_ABCD - S_OKCL - S_ONAM - S_ONBM →
    S_OMVK = 30 :=
by
  intros S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK h_OKCL h_ONAM h_ONBM h_ABCD h_OMVK
  rw [h_OKCL, h_ONAM, h_ONBM] at *
  sorry

end area_OMVK_l23_2328


namespace number_of_bricks_needed_l23_2309

theorem number_of_bricks_needed :
  ∀ (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ),
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_length = 750 → 
  wall_height = 600 → 
  wall_width = 22.5 → 
  (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) = 6000 :=
by
  intros brick_length brick_width brick_height wall_length wall_height wall_width
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end number_of_bricks_needed_l23_2309


namespace real_solutions_in_interval_l23_2369

noncomputable def problem_statement (x : ℝ) : Prop :=
  (x + 1 > 0) ∧ 
  (x ≠ -1) ∧
  (x^2 / (x + 1 - Real.sqrt (x + 1))^2 < (x^2 + 3 * x + 18) / (x + 1)^2)
  
theorem real_solutions_in_interval (x : ℝ) (h : problem_statement x) : -1 < x ∧ x < 3 :=
sorry

end real_solutions_in_interval_l23_2369


namespace largest_divisor_of_product_l23_2370

theorem largest_divisor_of_product (n : ℕ) (h : n % 3 = 0) : ∃ d, d = 288 ∧ ∀ n (h : n % 3 = 0), d ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) := 
sorry

end largest_divisor_of_product_l23_2370


namespace cone_dimensions_l23_2344

noncomputable def cone_height (r_sector : ℝ) (r_cone_base : ℝ) : ℝ :=
  Real.sqrt (r_sector^2 - r_cone_base^2)

noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * radius^2 * height

theorem cone_dimensions 
  (r_circle : ℝ) (num_sectors : ℕ) (r_cone_base : ℝ) :
  r_circle = 12 → num_sectors = 4 → r_cone_base = 3 → 
  cone_height r_circle r_cone_base = 3 * Real.sqrt 15 ∧ 
  cone_volume r_cone_base (cone_height r_circle r_cone_base) = 9 * Real.pi * Real.sqrt 15 :=
by
  intros
  sorry

end cone_dimensions_l23_2344


namespace golden_section_AC_correct_l23_2390

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def segment_length := 20
noncomputable def golden_section_point (AB AC BC : ℝ) (h1 : AB = AC + BC) (h2 : AC > BC) (h3 : AB = segment_length) : Prop :=
  AC = (Real.sqrt 5 - 1) / 2 * AB

theorem golden_section_AC_correct :
  ∃ (AC BC : ℝ), (AC + BC = segment_length) ∧ (AC > BC) ∧ (AC = 10 * (Real.sqrt 5 - 1)) :=
by
  sorry

end golden_section_AC_correct_l23_2390


namespace min_value_ineq_least_3_l23_2376

noncomputable def min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : ℝ :=
  1 / (x + y) + (x + y) / z

theorem min_value_ineq_least_3 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  min_value_ineq x y z h1 h2 h3 h4 ≥ 3 :=
sorry

end min_value_ineq_least_3_l23_2376


namespace new_train_travel_distance_l23_2323

-- Definitions of the trains' travel distances
def older_train_distance : ℝ := 180
def new_train_additional_distance_ratio : ℝ := 0.50

-- Proof that the new train can travel 270 miles
theorem new_train_travel_distance
: new_train_additional_distance_ratio * older_train_distance + older_train_distance = 270 := 
by
  sorry

end new_train_travel_distance_l23_2323


namespace bob_ears_left_l23_2353

namespace CornProblem

-- Definitions of the given conditions
def initial_bob_bushels : ℕ := 120
def ears_per_bushel : ℕ := 15

def given_away_bushels_terry : ℕ := 15
def given_away_bushels_jerry : ℕ := 8
def given_away_bushels_linda : ℕ := 25
def given_away_ears_stacy : ℕ := 42
def given_away_bushels_susan : ℕ := 9
def given_away_bushels_tim : ℕ := 4
def given_away_ears_tim : ℕ := 18

-- Calculate initial ears of corn
noncomputable def initial_ears_of_corn : ℕ := initial_bob_bushels * ears_per_bushel

-- Calculate total ears given away in bushels
def total_ears_given_away_bushels : ℕ :=
  (given_away_bushels_terry + given_away_bushels_jerry + given_away_bushels_linda +
   given_away_bushels_susan + given_away_bushels_tim) * ears_per_bushel

-- Calculate total ears directly given away
def total_ears_given_away_direct : ℕ :=
  given_away_ears_stacy + given_away_ears_tim

-- Calculate total ears given away
def total_ears_given_away : ℕ :=
  total_ears_given_away_bushels + total_ears_given_away_direct

-- Calculate ears of corn Bob has left
noncomputable def ears_left : ℕ :=
  initial_ears_of_corn - total_ears_given_away

-- The proof statement
theorem bob_ears_left : ears_left = 825 := by
  sorry

end CornProblem

end bob_ears_left_l23_2353


namespace sum_of_first_8_terms_l23_2345

noncomputable def sum_of_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
a * (1 - r^n) / (1 - r)

theorem sum_of_first_8_terms 
  (a r : ℝ)
  (h₁ : sum_of_geometric_sequence a r 4 = 5)
  (h₂ : sum_of_geometric_sequence a r 12 = 35) :
  sum_of_geometric_sequence a r 8 = 15 := 
sorry

end sum_of_first_8_terms_l23_2345


namespace frequency_of_hits_l23_2319

theorem frequency_of_hits (n m : ℕ) (h_n : n = 20) (h_m : m = 15) : (m / n : ℚ) = 0.75 := by
  sorry

end frequency_of_hits_l23_2319


namespace simplified_value_of_sum_l23_2399

theorem simplified_value_of_sum :
  (-1)^(2004) + (-1)^(2005) + 1^(2006) - 1^(2007) = -2 := by
  sorry

end simplified_value_of_sum_l23_2399


namespace Jason_attended_36_games_l23_2325

noncomputable def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (percentage_missed : ℕ) : ℕ :=
  let total_planned := planned_this_month + planned_last_month
  let missed_games := (percentage_missed * total_planned) / 100
  total_planned - missed_games

theorem Jason_attended_36_games :
  games_attended 24 36 40 = 36 :=
by
  sorry

end Jason_attended_36_games_l23_2325


namespace sum_digits_probability_l23_2374

noncomputable def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def numInRange : ℕ := 1000000

noncomputable def coefficient : ℕ :=
  Nat.choose 24 5 - 6 * Nat.choose 14 5

noncomputable def probability : ℚ :=
  coefficient / numInRange

theorem sum_digits_probability :
  probability = 7623 / 250000 :=
by
  sorry

end sum_digits_probability_l23_2374


namespace danny_bottle_cap_count_l23_2343

theorem danny_bottle_cap_count 
  (initial_caps : Int) 
  (found_caps : Int) 
  (final_caps : Int) 
  (h1 : initial_caps = 6) 
  (h2 : found_caps = 22) 
  (h3 : final_caps = initial_caps + found_caps) : 
  final_caps = 28 :=
by
  sorry

end danny_bottle_cap_count_l23_2343


namespace episodes_per_season_l23_2316

theorem episodes_per_season
  (days_to_watch : ℕ)
  (episodes_per_day : ℕ)
  (seasons : ℕ) :
  days_to_watch = 10 →
  episodes_per_day = 6 →
  seasons = 4 →
  (episodes_per_day * days_to_watch) / seasons = 15 :=
by
  intros
  sorry

end episodes_per_season_l23_2316


namespace product_of_distances_l23_2305

-- Definitions based on the conditions
def curve (x y : ℝ) : Prop := x * y = 2

-- The theorem to prove
theorem product_of_distances (x y : ℝ) (h : curve x y) : abs x * abs y = 2 := by
  -- This is where the proof would go
  sorry

end product_of_distances_l23_2305


namespace fred_carrots_l23_2364

-- Define the conditions
def sally_carrots : Nat := 6
def total_carrots : Nat := 10

-- Define the problem question and the proof statement
theorem fred_carrots : ∃ fred_carrots : Nat, fred_carrots = total_carrots - sally_carrots := 
by
  sorry

end fred_carrots_l23_2364


namespace min_value_l23_2314

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) := 
sorry

end min_value_l23_2314


namespace misha_needs_total_l23_2362

theorem misha_needs_total (
  current_amount : ℤ := 34
) (additional_amount : ℤ := 13) : 
  current_amount + additional_amount = 47 :=
by
  sorry

end misha_needs_total_l23_2362


namespace right_triangle_condition_l23_2387

theorem right_triangle_condition (a d : ℝ) (h : d > 0) : 
  (a = d * (1 + Real.sqrt 7)) ↔ (a^2 + (a + 2 * d)^2 = (a + 4 * d)^2) := 
sorry

end right_triangle_condition_l23_2387


namespace predict_height_at_age_10_l23_2391

def regression_line := fun (x : ℝ) => 7.19 * x + 73.93

theorem predict_height_at_age_10 :
  regression_line 10 = 145.83 :=
by
  sorry

end predict_height_at_age_10_l23_2391


namespace rachel_total_problems_l23_2329

theorem rachel_total_problems
    (problems_per_minute : ℕ)
    (minutes_before_bed : ℕ)
    (problems_next_day : ℕ) 
    (h1 : problems_per_minute = 5) 
    (h2 : minutes_before_bed = 12) 
    (h3 : problems_next_day = 16) : 
    problems_per_minute * minutes_before_bed + problems_next_day = 76 :=
by
  sorry

end rachel_total_problems_l23_2329


namespace initial_speed_solution_l23_2385

def initial_speed_problem : Prop :=
  ∃ V : ℝ, 
    (∀ t t_new : ℝ, 
      t = 300 / V ∧ 
      t_new = t - 4 / 5 ∧ 
      (∀ d d_remaining : ℝ, 
        d = V * (5 / 4) ∧ 
        d_remaining = 300 - d ∧ 
        t_new = (5 / 4) + d_remaining / (V + 16)) 
    ) → 
    V = 60

theorem initial_speed_solution : initial_speed_problem :=
by
  unfold initial_speed_problem
  sorry

end initial_speed_solution_l23_2385


namespace complement_A_eq_l23_2334

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_A_eq :
  U \ A = {0, 2} :=
by
  sorry

end complement_A_eq_l23_2334


namespace find_f1_find_f8_inequality_l23_2356

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f x
axiom f_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y
axiom f_multiplicative : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x * f y
axiom f_of_2 : f 2 = 4

-- Statements to prove
theorem find_f1 : f 1 = 1 := sorry
theorem find_f8 : f 8 = 64 := sorry
theorem inequality : ∀ x : ℝ, 3 < x → x ≤ 7 / 2 → 16 * f (1 / (x - 3)) ≥ f (2 * x + 1) := sorry

end find_f1_find_f8_inequality_l23_2356


namespace number_of_correct_statements_l23_2317

def line : Type := sorry
def plane : Type := sorry
def parallel (x y : line) : Prop := sorry
def perpendicular (x : line) (y : plane) : Prop := sorry
def subset (x : line) (y : plane) : Prop := sorry
def skew (x y : line) : Prop := sorry

variable (m n : line) -- two different lines
variable (alpha beta : plane) -- two different planes

theorem number_of_correct_statements :
  (¬parallel m alpha ∨ subset n alpha ∧ parallel m n) ∧
  (parallel m alpha ∧ perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular n beta) ∧
  (subset m alpha ∧ subset n beta ∧ perpendicular m n) ∧
  (skew m n ∧ subset m alpha ∧ subset n beta ∧ parallel m beta ∧ parallel n alpha) :=
sorry

end number_of_correct_statements_l23_2317


namespace number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l23_2351

-- Define the conditions
def total_matches := 16
def played_matches := 9
def lost_matches := 2
def current_points := 19
def max_points_per_win := 3
def draw_points := 1
def remaining_matches := total_matches - played_matches
def required_points := 34

-- Statements to prove
theorem number_of_wins_in_first_9_matches :
  ∃ wins_in_first_9, 3 * wins_in_first_9 + draw_points * (played_matches - lost_matches - wins_in_first_9) = current_points :=
sorry

theorem highest_possible_points :
  current_points + remaining_matches * max_points_per_win = 40 :=
sorry

theorem minimum_wins_in_remaining_matches :
  ∃ min_wins_in_remaining_7, (min_wins_in_remaining_7 = 4 ∧ 3 * min_wins_in_remaining_7 + current_points + (remaining_matches - min_wins_in_remaining_7) * draw_points ≥ required_points) :=
sorry

end number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l23_2351


namespace max_value_of_A_l23_2335

theorem max_value_of_A (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end max_value_of_A_l23_2335


namespace water_formed_from_reaction_l23_2340

-- Definitions
def mol_mass_water : ℝ := 18.015
def water_formed_grams (moles_water : ℝ) : ℝ := moles_water * mol_mass_water

-- Statement
theorem water_formed_from_reaction (moles_water : ℝ) :
  18 = water_formed_grams moles_water :=
by sorry

end water_formed_from_reaction_l23_2340


namespace probability_not_greater_than_two_l23_2396

theorem probability_not_greater_than_two : 
  let cards := [1, 2, 3, 4]
  let favorable_cards := [1, 2]
  let total_scenarios := cards.length
  let favorable_scenarios := favorable_cards.length
  let prob := favorable_scenarios / total_scenarios
  prob = 1 / 2 :=
by
  sorry

end probability_not_greater_than_two_l23_2396


namespace C_recurrence_S_recurrence_l23_2330

noncomputable def C (x : ℝ) : ℝ := 2 * Real.cos x
noncomputable def C_n (n : ℕ) (x : ℝ) : ℝ := 2 * Real.cos (n * x)
noncomputable def S_n (n : ℕ) (x : ℝ) : ℝ := Real.sin (n * x) / Real.sin x

theorem C_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  C_n n x = C x * C_n (n - 1) x - C_n (n - 2) x := sorry

theorem S_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  S_n n x = C x * S_n (n - 1) x - S_n (n - 2) x := sorry

end C_recurrence_S_recurrence_l23_2330


namespace collinear_probability_l23_2347

-- Define the rectangular array
def rows : ℕ := 4
def cols : ℕ := 5
def total_dots : ℕ := rows * cols
def chosen_dots : ℕ := 4

-- Define the collinear sets
def horizontal_lines : ℕ := rows
def vertical_lines : ℕ := cols
def collinear_sets : ℕ := horizontal_lines + vertical_lines

-- Define the total combinations of choosing 4 dots out of 20
def total_combinations : ℕ := Nat.choose total_dots chosen_dots

-- Define the probability
def probability : ℚ := collinear_sets / total_combinations

theorem collinear_probability : probability = 9 / 4845 := by
  sorry

end collinear_probability_l23_2347


namespace appliance_costs_l23_2331

theorem appliance_costs (a b : ℕ) 
  (h1 : a + 2 * b = 2300) 
  (h2 : 2 * a + b = 2050) : 
  a = 600 ∧ b = 850 := 
by 
  sorry

end appliance_costs_l23_2331


namespace geometric_sequence_a9_l23_2308

theorem geometric_sequence_a9 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 2) 
  (h2 : a 4 = 8 * a 7) 
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (hq : q > 0) 
  : a 9 = 1 / 32 := 
by sorry

end geometric_sequence_a9_l23_2308


namespace height_relationship_l23_2372

theorem height_relationship 
  (r₁ h₁ r₂ h₂ : ℝ)
  (h_volume : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius : r₂ = (6/5) * r₁) :
  h₁ = 1.44 * h₂ :=
by
  sorry

end height_relationship_l23_2372


namespace cargo_total_ship_l23_2394

-- Define the initial cargo and the additional cargo loaded
def initial_cargo := 5973
def additional_cargo := 8723

-- Define the total cargo the ship holds after loading additional cargo
def total_cargo := initial_cargo + additional_cargo

-- Statement of the problem
theorem cargo_total_ship (h1 : initial_cargo = 5973) (h2 : additional_cargo = 8723) : 
  total_cargo = 14696 := 
by
  sorry

end cargo_total_ship_l23_2394


namespace problem_solution_l23_2326

theorem problem_solution (x y : ℕ) (hxy : x + y + x * y = 104) (hx : 0 < x) (hy : 0 < y) (hx30 : x < 30) (hy30 : y < 30) : 
  x + y = 20 := 
sorry

end problem_solution_l23_2326


namespace k_greater_than_inv_e_l23_2346

theorem k_greater_than_inv_e (k : ℝ) (x : ℝ) (hx_pos : 0 < x) (hcond : k * (Real.exp (k * x) + 1) - (1 + (1 / x)) * Real.log x > 0) : 
  k > 1 / Real.exp 1 :=
sorry

end k_greater_than_inv_e_l23_2346


namespace sum_of_twos_and_threes_3024_l23_2366

theorem sum_of_twos_and_threes_3024 : ∃ n : ℕ, n = 337 ∧ (∃ (a b : ℕ), 3024 = 2 * a + 3 * b) :=
sorry

end sum_of_twos_and_threes_3024_l23_2366


namespace election_total_votes_l23_2386

theorem election_total_votes (V_A V_B V : ℕ) (H1 : V_A = V_B + 15/100 * V) (H2 : V_A + V_B = 80/100 * V) (H3 : V_B = 2184) : V = 6720 :=
sorry

end election_total_votes_l23_2386


namespace circle_center_l23_2397

theorem circle_center :
    ∃ (h k : ℝ), (x^2 - 10 * x + y^2 - 4 * y = -4) →
                 (x - h)^2 + (y - k)^2 = 25 ∧ h = 5 ∧ k = 2 :=
sorry

end circle_center_l23_2397


namespace min_value_of_expression_l23_2341

noncomputable def min_value_expression (a b c d : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2

theorem min_value_of_expression (a b c d : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  min_value_expression a b c d = 1 / 4 :=
sorry

end min_value_of_expression_l23_2341


namespace sum_terms_a1_a17_l23_2338

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 := by
  sorry

end sum_terms_a1_a17_l23_2338


namespace sri_lanka_population_problem_l23_2348

theorem sri_lanka_population_problem
  (P : ℝ)
  (h1 : 0.85 * (0.9 * P) = 3213) :
  P = 4200 :=
sorry

end sri_lanka_population_problem_l23_2348


namespace min_balls_to_draw_l23_2363

theorem min_balls_to_draw (black white red : ℕ) (h_black : black = 10) (h_white : white = 9) (h_red : red = 8) :
  ∃ n, n = 20 ∧
  ∀ k, (k < 20) → ¬ (∃ b w r, b + w + r = k ∧ b ≤ black ∧ w ≤ white ∧ r ≤ red ∧ r > 0 ∧ w > 0) :=
by {
  sorry
}

end min_balls_to_draw_l23_2363


namespace third_smallest_number_l23_2361

/-- 
  The third smallest two-decimal-digit number that can be made
  using the digits 3, 8, 2, and 7 each exactly once is 27.38.
-/
theorem third_smallest_number (digits : List ℕ) (h : digits = [3, 8, 2, 7]) : 
  ∃ x y, 
  x < y ∧
  x = 23.78 ∧
  y = 23.87 ∧
  ∀ z, z > x ∧ z < y → z = 27.38 :=
by 
  sorry

end third_smallest_number_l23_2361


namespace second_increase_is_40_l23_2313

variable (P : ℝ) (x : ℝ)

def second_increase (P : ℝ) (x : ℝ) : Prop :=
  1.30 * P * (1 + x / 100) = 1.82 * P

theorem second_increase_is_40 (P : ℝ) : ∃ x, second_increase P x ∧ x = 40 := by
  use 40
  sorry

end second_increase_is_40_l23_2313


namespace correct_answer_l23_2310

theorem correct_answer (a b c : ℤ) 
  (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by 
  sorry

end correct_answer_l23_2310


namespace average_age_l23_2393

theorem average_age (women men : ℕ) (avg_age_women avg_age_men : ℝ) 
  (h_women : women = 12) 
  (h_men : men = 18) 
  (h_avg_women : avg_age_women = 28) 
  (h_avg_men : avg_age_men = 40) : 
  (12 * 28 + 18 * 40) / (12 + 18) = 35.2 :=
by {
  sorry
}

end average_age_l23_2393


namespace seashells_unbroken_l23_2395

theorem seashells_unbroken (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) 
  (h3 : unbroken_seashells = total_seashells - broken_seashells) :
  unbroken_seashells = 2 :=
by
  sorry

end seashells_unbroken_l23_2395


namespace max_distance_AB_l23_2355

-- Define curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define curve C2 in Cartesian coordinates
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the problem to prove the maximum value of distance AB is 8
theorem max_distance_AB :
  ∀ (Ax Ay Bx By : ℝ),
    C1 Ax Ay →
    C2 Bx By →
    dist (Ax, Ay) (Bx, By) ≤ 8 :=
sorry

end max_distance_AB_l23_2355


namespace number_of_valid_4_digit_integers_l23_2318

/-- 
Prove that the number of 4-digit positive integers that satisfy the following conditions:
1. Each of the first two digits must be 2, 3, or 5.
2. The last two digits cannot be the same.
3. Each of the last two digits must be 4, 6, or 9.
is equal to 54.
-/
theorem number_of_valid_4_digit_integers : 
  ∃ n : ℕ, n = 54 ∧ 
  ∀ d1 d2 d3 d4 : ℕ, 
    (d1 = 2 ∨ d1 = 3 ∨ d1 = 5) ∧ 
    (d2 = 2 ∨ d2 = 3 ∨ d2 = 5) ∧ 
    (d3 = 4 ∨ d3 = 6 ∨ d3 = 9) ∧ 
    (d4 = 4 ∨ d4 = 6 ∨ d4 = 9) ∧ 
    (d3 ≠ d4) → 
    n = 54 := 
sorry

end number_of_valid_4_digit_integers_l23_2318


namespace quadratic_root_solution_l23_2306

theorem quadratic_root_solution (k : ℤ) (a : ℤ) :
  (∀ x, x^2 + k * x - 10 = 0 → x = 2 ∨ x = a) →
  2 + a = -k →
  2 * a = -10 →
  k = 3 ∧ a = -5 :=
by
  sorry

end quadratic_root_solution_l23_2306


namespace part1_part2_part3_l23_2304

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end part1_part2_part3_l23_2304


namespace pencil_price_l23_2327

variable (P N : ℕ) -- This assumes the price of a pencil (P) and the price of a notebook (N) are natural numbers (non-negative integers).

-- Define the conditions
def conditions : Prop :=
  (P + N = 950) ∧ (N = P + 150)

-- The theorem to prove
theorem pencil_price (h : conditions P N) : P = 400 :=
by
  sorry

end pencil_price_l23_2327
