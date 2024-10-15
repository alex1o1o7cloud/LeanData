import Mathlib

namespace NUMINAMATH_GPT_abc_sum_l1275_127594

theorem abc_sum
  (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 70 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 - 19 * x + 84 = (x - b) * (x - c)) :
  a + b + c = 29 := by
  sorry

end NUMINAMATH_GPT_abc_sum_l1275_127594


namespace NUMINAMATH_GPT_sum_of_inverses_A_B_C_eq_300_l1275_127591

theorem sum_of_inverses_A_B_C_eq_300 
  (p q r : ℝ)
  (hroots : ∀ x, (x^3 - 30*x^2 + 105*x - 114 = 0) → (x = p ∨ x = q ∨ x = r))
  (A B C : ℝ)
  (hdecomp : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    (1 / (s^3 - 30*s^2 + 105*s - 114) = A/(s - p) + B/(s - q) + C/(s - r))) :
  (1 / A) + (1 / B) + (1 / C) = 300 :=
sorry

end NUMINAMATH_GPT_sum_of_inverses_A_B_C_eq_300_l1275_127591


namespace NUMINAMATH_GPT_smallest_integer_x_l1275_127568

theorem smallest_integer_x (x : ℤ) : 
  ( ∀ x : ℤ, ( 2 * (x : ℚ) / 5 + 3 / 4 > 7 / 5 → 2 ≤ x )) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_smallest_integer_x_l1275_127568


namespace NUMINAMATH_GPT_range_of_g_l1275_127579

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + (Real.pi / 4) * (Real.arcsin (x / 3)) 
    - (Real.arcsin (x / 3))^2 + (Real.pi^2 / 16) * (x^2 + 2 * x + 3)

theorem range_of_g : 
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → 
  ∃ y, y = g x ∧ y ∈ (Set.Icc (Real.pi^2 / 4) (15 * Real.pi^2 / 16 + Real.pi / 4 * Real.arcsin 1)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_g_l1275_127579


namespace NUMINAMATH_GPT_minimum_function_value_l1275_127543

theorem minimum_function_value :
  ∃ (x y : ℕ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 3 ∧
  (∀ x' y', 0 ≤ x' ∧ x' ≤ 2 → 0 ≤ y' ∧ y' ≤ 3 →
  (x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) ≤ (x'^2 * y'^2 : ℝ) / ((x'^2 + y'^2)^2 : ℝ)) ∧
  (x = 0 ∨ y = 0) ∧ ((x^2 * y^2 : ℝ) / ((x^2 + y^2)^2 : ℝ) = 0) :=
by
  --; Implementation of the theorem would follow
  sorry

end NUMINAMATH_GPT_minimum_function_value_l1275_127543


namespace NUMINAMATH_GPT_bear_meat_needs_l1275_127569

theorem bear_meat_needs (B_total : ℕ) (cubs : ℕ) (w_cub : ℚ) 
  (h1 : B_total = 210)
  (h2 : cubs = 4)
  (h3 : w_cub = B_total / cubs) : 
  w_cub = 52.5 :=
by 
  sorry

end NUMINAMATH_GPT_bear_meat_needs_l1275_127569


namespace NUMINAMATH_GPT_inequality_bound_l1275_127508

theorem inequality_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ( (2 * a + b + c)^2 / (2 * a ^ 2 + (b + c) ^2) + 
    (2 * b + c + a)^2 / (2 * b ^ 2 + (c + a) ^2) + 
    (2 * c + a + b)^2 / (2 * c ^ 2 + (a + b) ^2) ) ≤ 8 := 
sorry

end NUMINAMATH_GPT_inequality_bound_l1275_127508


namespace NUMINAMATH_GPT_am_gm_inequality_l1275_127518

theorem am_gm_inequality (x y z : ℝ) (n : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0):
  x^n + y^n + z^n ≥ 1 / 3^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1275_127518


namespace NUMINAMATH_GPT_real_roots_for_all_a_b_l1275_127523

theorem real_roots_for_all_a_b (a b : ℝ) : ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) :=
sorry

end NUMINAMATH_GPT_real_roots_for_all_a_b_l1275_127523


namespace NUMINAMATH_GPT_tangerine_boxes_l1275_127545

theorem tangerine_boxes
  (num_boxes_apples : ℕ)
  (apples_per_box : ℕ)
  (num_boxes_tangerines : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : num_boxes_apples = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : num_boxes_tangerines = 6 := 
  sorry

end NUMINAMATH_GPT_tangerine_boxes_l1275_127545


namespace NUMINAMATH_GPT_fraction_evaluation_l1275_127549

theorem fraction_evaluation (x z : ℚ) (hx : x = 4/7) (hz : z = 8/11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l1275_127549


namespace NUMINAMATH_GPT_completion_time_workshop_3_l1275_127592

-- Define the times for workshops
def time_in_workshop_3 : ℝ := 8
def time_in_workshop_1 : ℝ := time_in_workshop_3 + 10
def time_in_workshop_2 : ℝ := (time_in_workshop_3 + 10) - 3.6

-- Define the combined work equation
def combined_work_eq := (1 / time_in_workshop_1) + (1 / time_in_workshop_2) = (1 / time_in_workshop_3)

-- Final theorem statement
theorem completion_time_workshop_3 (h : combined_work_eq) : time_in_workshop_3 - 7 = 1 :=
by
  sorry

end NUMINAMATH_GPT_completion_time_workshop_3_l1275_127592


namespace NUMINAMATH_GPT_regular_polygon_of_45_deg_l1275_127538

def is_regular_polygon (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 2 ∧ 360 % k = 0 ∧ n = 360 / k

def regular_polygon_is_octagon (angle : ℕ) : Prop :=
  is_regular_polygon 8 ∧ angle = 45

theorem regular_polygon_of_45_deg : regular_polygon_is_octagon 45 :=
  sorry

end NUMINAMATH_GPT_regular_polygon_of_45_deg_l1275_127538


namespace NUMINAMATH_GPT_arithmetic_seq_a2_l1275_127581

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n m : ℕ, a m = a (n + 1) + d * (m - (n + 1))

theorem arithmetic_seq_a2 
  (a : ℕ → ℤ) (d a1 : ℤ)
  (h_arith: ∀ n : ℕ, a n = a1 + n * d)
  (h_sum: a 3 + a 11 = 50)
  (h_a4: a 4 = 13) :
  a 2 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_a2_l1275_127581


namespace NUMINAMATH_GPT_average_value_correct_l1275_127582

noncomputable def average_value (k z : ℝ) : ℝ :=
  (k + 2 * k * z + 4 * k * z + 8 * k * z + 16 * k * z) / 5

theorem average_value_correct (k z : ℝ) :
  average_value k z = (k * (1 + 30 * z)) / 5 := by
  sorry

end NUMINAMATH_GPT_average_value_correct_l1275_127582


namespace NUMINAMATH_GPT_problem_statement_l1275_127502

noncomputable def U : Set Int := {-2, -1, 0, 1, 2}
noncomputable def A : Set Int := {x : Int | -2 ≤ x ∧ x < 0}
noncomputable def B : Set Int := {x : Int | (x = 0 ∨ x = 1)} -- since natural numbers typically include positive integers, adapting B contextually

theorem problem_statement : ((U \ A) ∩ B) = {0, 1} := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1275_127502


namespace NUMINAMATH_GPT_adapted_bowling_ball_volume_l1275_127585

noncomputable def volume_adapted_bowling_ball : ℝ :=
  let volume_sphere := (4/3) * Real.pi * (20 ^ 3)
  let volume_hole1 := Real.pi * (1 ^ 2) * 10
  let volume_hole2 := Real.pi * (1.5 ^ 2) * 10
  let volume_hole3 := Real.pi * (2 ^ 2) * 10
  volume_sphere - (volume_hole1 + volume_hole2 + volume_hole3)

theorem adapted_bowling_ball_volume :
  volume_adapted_bowling_ball = 10594.17 * Real.pi :=
sorry

end NUMINAMATH_GPT_adapted_bowling_ball_volume_l1275_127585


namespace NUMINAMATH_GPT_arithmetic_sequence_max_min_b_l1275_127503

-- Define the sequence a_n
def S (n : ℕ) : ℚ := (1/2) * n^2 - 2 * n
def a (n : ℕ) : ℚ := S n - S (n - 1)

-- Question 1: Prove that {a_n} is an arithmetic sequence with a common difference of 1
theorem arithmetic_sequence (n : ℕ) (hn : n ≥ 2) : 
  a n - a (n - 1) = 1 :=
sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (a n + 1) / a n

-- Question 2: Prove that b_3 is the maximum value and b_2 is the minimum value in {b_n}
theorem max_min_b (hn2 : 2 ≥ 1) (hn3 : 3 ≥ 1) : 
  b 3 = 3 ∧ b 2 = -1 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_min_b_l1275_127503


namespace NUMINAMATH_GPT_solution_set_x_squared_minus_3x_lt_0_l1275_127525

theorem solution_set_x_squared_minus_3x_lt_0 : { x : ℝ | x^2 - 3 * x < 0 } = { x : ℝ | 0 < x ∧ x < 3 } :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_x_squared_minus_3x_lt_0_l1275_127525


namespace NUMINAMATH_GPT_middle_term_arithmetic_sequence_l1275_127577

theorem middle_term_arithmetic_sequence (m : ℝ) (h : 2 * m = 1 + 5) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_middle_term_arithmetic_sequence_l1275_127577


namespace NUMINAMATH_GPT_johnsonville_max_band_members_l1275_127551

def max_band_members :=
  ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
  ∀ n : ℤ, (30 * n % 34 = 2 ∧ 30 * n < 1500) → 30 * n ≤ 30 * m

theorem johnsonville_max_band_members : ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
                                           30 * m = 1260 :=
by 
  sorry

end NUMINAMATH_GPT_johnsonville_max_band_members_l1275_127551


namespace NUMINAMATH_GPT_total_order_cost_l1275_127565

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end NUMINAMATH_GPT_total_order_cost_l1275_127565


namespace NUMINAMATH_GPT_solve_abs_inequality_l1275_127526

theorem solve_abs_inequality (x : ℝ) :
  3 ≤ abs ((x - 3)^2 - 4) ∧ abs ((x - 3)^2 - 4) ≤ 7 ↔ 3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_solve_abs_inequality_l1275_127526


namespace NUMINAMATH_GPT_angle_of_inclination_of_line_l1275_127557

theorem angle_of_inclination_of_line (θ : ℝ) (m : ℝ) (h : |m| = 1) :
  θ = 45 ∨ θ = 135 :=
sorry

end NUMINAMATH_GPT_angle_of_inclination_of_line_l1275_127557


namespace NUMINAMATH_GPT_decreasing_function_range_l1275_127559

theorem decreasing_function_range {f : ℝ → ℝ} (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) :
  {x : ℝ | f (x^2 - 3 * x - 3) < f 1} = {x : ℝ | x < -1 ∨ x > 4} :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l1275_127559


namespace NUMINAMATH_GPT_series_sum_l1275_127589

variable {c d : ℝ}

theorem series_sum (h : ∑' n : ℕ, c / d ^ ((3 : ℝ) ^ n) = 9) :
  ∑' n : ℕ, c / (c + 2 * d) ^ (n + 1) = 9 / 11 :=
by
  -- The code that follows will include the steps and proof to reach the conclusion
  sorry

end NUMINAMATH_GPT_series_sum_l1275_127589


namespace NUMINAMATH_GPT_ratio_sheep_horses_l1275_127597

theorem ratio_sheep_horses (amount_food_per_horse : ℕ) (total_food_per_day : ℕ) (num_sheep : ℕ) (num_horses : ℕ) :
  amount_food_per_horse = 230 ∧ total_food_per_day = 12880 ∧ num_sheep = 24 ∧ num_horses = total_food_per_day / amount_food_per_horse →
  num_sheep / num_horses = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_sheep_horses_l1275_127597


namespace NUMINAMATH_GPT_yellow_white_flowers_count_l1275_127513

theorem yellow_white_flowers_count
    (RY RW : Nat)
    (hRY : RY = 17)
    (hRW : RW = 14)
    (hRedMoreThanWhite : (RY + RW) - (RW + YW) = 4) :
    ∃ YW, YW = 13 := 
by
  sorry

end NUMINAMATH_GPT_yellow_white_flowers_count_l1275_127513


namespace NUMINAMATH_GPT_find_y_z_l1275_127500

theorem find_y_z (y z : ℝ) : 
  (∃ k : ℝ, (1:ℝ) = -k ∧ (2:ℝ) = k * y ∧ (3:ℝ) = k * z) → y = -2 ∧ z = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_y_z_l1275_127500


namespace NUMINAMATH_GPT_incorrect_observation_l1275_127562

theorem incorrect_observation (n : ℕ) (mean_original mean_corrected correct_obs incorrect_obs : ℝ)
  (h1 : n = 40) 
  (h2 : mean_original = 36) 
  (h3 : mean_corrected = 36.45) 
  (h4 : correct_obs = 34) 
  (h5 : n * mean_original = 1440) 
  (h6 : n * mean_corrected = 1458) 
  (h_diff : 1458 - 1440 = 18) :
  incorrect_obs = 52 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_observation_l1275_127562


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1275_127564

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

noncomputable def sum_geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum 
  (a₁ : ℝ) (q : ℝ) 
  (h_q : q = 1 / 2) 
  (h_a₂ : geometric_sequence a₁ q 2 = 2) : 
  sum_geometric_sequence a₁ q 6 = 63 / 8 :=
by
  -- The proof is skipped here
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1275_127564


namespace NUMINAMATH_GPT_polynomial_remainder_l1275_127507

theorem polynomial_remainder (x : ℂ) (hx : x^5 = 1) :
  (x^25 + x^20 + x^15 + x^10 + x^5 + 1) % (x^5 - 1) = 6 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1275_127507


namespace NUMINAMATH_GPT_total_fish_at_wedding_l1275_127505

def num_tables : ℕ := 32
def fish_per_table_except_one : ℕ := 2
def fish_on_special_table : ℕ := 3
def number_of_special_tables : ℕ := 1
def number_of_regular_tables : ℕ := num_tables - number_of_special_tables

theorem total_fish_at_wedding : 
  (number_of_regular_tables * fish_per_table_except_one) + (number_of_special_tables * fish_on_special_table) = 65 :=
by
  sorry

end NUMINAMATH_GPT_total_fish_at_wedding_l1275_127505


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l1275_127548

theorem largest_angle_of_triangle (x : ℝ) (h : x + 3 * x + 5 * x = 180) : 5 * x = 100 :=
sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l1275_127548


namespace NUMINAMATH_GPT_ron_tickets_sold_l1275_127555

theorem ron_tickets_sold 
  (R K : ℕ) 
  (h1 : R + K = 20) 
  (h2 : 2 * R + 9 / 2 * K = 60) : 
  R = 12 := 
by 
  sorry

end NUMINAMATH_GPT_ron_tickets_sold_l1275_127555


namespace NUMINAMATH_GPT_bicycle_route_total_length_l1275_127536

theorem bicycle_route_total_length :
  let horizontal_length := 13 
  let vertical_length := 13 
  2 * horizontal_length + 2 * vertical_length = 52 :=
by
  let horizontal_length := 13
  let vertical_length := 13
  sorry

end NUMINAMATH_GPT_bicycle_route_total_length_l1275_127536


namespace NUMINAMATH_GPT_digit_positions_in_8008_l1275_127541

theorem digit_positions_in_8008 :
  (8008 % 10 = 8) ∧ (8008 / 1000 % 10 = 8) :=
by
  sorry

end NUMINAMATH_GPT_digit_positions_in_8008_l1275_127541


namespace NUMINAMATH_GPT_remainder_of_p_div_10_is_6_l1275_127584

-- Define the problem
def a : ℕ := sorry -- a is a positive integer and a multiple of 2

-- Define p based on a
def p : ℕ := 4^a

-- The main goal is to prove the remainder when p is divided by 10 is 6
theorem remainder_of_p_div_10_is_6 (ha : a > 0 ∧ a % 2 = 0) : p % 10 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_of_p_div_10_is_6_l1275_127584


namespace NUMINAMATH_GPT_not_linear_eq_l1275_127512

-- Representing the given equations
def eq1 (x : ℝ) : Prop := 5 * x + 3 = 3 * x - 7
def eq2 (x : ℝ) : Prop := 1 + 2 * x = 3
def eq4 (x : ℝ) : Prop := x - 7 = 0

-- The equation to verify if it's not linear
def eq3 (x : ℝ) : Prop := abs (2 * x) / 3 + 5 / x = 3

-- Stating the Lean statement to be proved
theorem not_linear_eq : ¬ (eq3 x) := by
  sorry

end NUMINAMATH_GPT_not_linear_eq_l1275_127512


namespace NUMINAMATH_GPT_science_book_multiple_l1275_127588

theorem science_book_multiple (history_pages novel_pages science_pages : ℕ)
  (H1 : history_pages = 300)
  (H2 : novel_pages = history_pages / 2)
  (H3 : science_pages = 600) :
  science_pages / novel_pages = 4 := 
by
  -- Proof will be filled out here
  sorry

end NUMINAMATH_GPT_science_book_multiple_l1275_127588


namespace NUMINAMATH_GPT_exp_gt_pow_l1275_127529

theorem exp_gt_pow (x : ℝ) (h_pos : 0 < x) (h_ne : x ≠ Real.exp 1) : Real.exp x > x ^ Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_exp_gt_pow_l1275_127529


namespace NUMINAMATH_GPT_necessarily_negative_sum_l1275_127572

theorem necessarily_negative_sum 
  (u v w : ℝ)
  (hu : -1 < u ∧ u < 0)
  (hv : 0 < v ∧ v < 1)
  (hw : -2 < w ∧ w < -1) :
  v + w < 0 :=
sorry

end NUMINAMATH_GPT_necessarily_negative_sum_l1275_127572


namespace NUMINAMATH_GPT_relay_race_total_time_l1275_127566

theorem relay_race_total_time :
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  athlete1 + athlete2 + athlete3 + athlete4 = 200 :=
by
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  show athlete1 + athlete2 + athlete3 + athlete4 = 200
  sorry

end NUMINAMATH_GPT_relay_race_total_time_l1275_127566


namespace NUMINAMATH_GPT_simplify_fraction_l1275_127521

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1275_127521


namespace NUMINAMATH_GPT_minimum_distance_PQ_l1275_127587

open Real

noncomputable def minimum_distance (t : ℝ) : ℝ := 
  (|t - 1|) / (sqrt (1 + t ^ 2))

theorem minimum_distance_PQ :
  let t := sqrt 2 / 2
  let x_P := 2
  let y_P := 0
  let x_Q := -1 + t
  let y_Q := 2 + t
  let d := minimum_distance (x_Q - y_Q + 3)
  (d - 2) = (5 * sqrt 2) / 2 - 2 :=
sorry

end NUMINAMATH_GPT_minimum_distance_PQ_l1275_127587


namespace NUMINAMATH_GPT_hotel_flat_fee_l1275_127516

theorem hotel_flat_fee
  (f n : ℝ)
  (h1 : f + 3 * n = 195)
  (h2 : f + 7 * n = 380) :
  f = 56.25 :=
by sorry

end NUMINAMATH_GPT_hotel_flat_fee_l1275_127516


namespace NUMINAMATH_GPT_total_seats_in_stadium_l1275_127514

theorem total_seats_in_stadium (people_at_game : ℕ) (empty_seats : ℕ) (total_seats : ℕ)
  (h1 : people_at_game = 47) (h2 : empty_seats = 45) :
  total_seats = people_at_game + empty_seats :=
by
  rw [h1, h2]
  show total_seats = 47 + 45
  sorry

end NUMINAMATH_GPT_total_seats_in_stadium_l1275_127514


namespace NUMINAMATH_GPT_x_square_minus_5x_is_necessary_not_sufficient_l1275_127573

theorem x_square_minus_5x_is_necessary_not_sufficient (x : ℝ) :
  (x^2 - 5 * x < 0) → (|x - 1| < 1) → (x^2 - 5 * x < 0 ∧ ∃ y : ℝ, (0 < y ∧ y < 2) → x = y) :=
by
  sorry

end NUMINAMATH_GPT_x_square_minus_5x_is_necessary_not_sufficient_l1275_127573


namespace NUMINAMATH_GPT_find_remainder_l1275_127522

theorem find_remainder (x : ℤ) (h : 0 < x ∧ 7 * x % 26 = 1) : (13 + 3 * x) % 26 = 6 :=
sorry

end NUMINAMATH_GPT_find_remainder_l1275_127522


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1275_127596

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x : ℝ | 3 * a < x ∧ x < -a} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1275_127596


namespace NUMINAMATH_GPT_tim_campaign_total_l1275_127553

theorem tim_campaign_total (amount_max : ℕ) (num_max : ℕ) (num_half : ℕ) (total_donations : ℕ) (total_raised : ℕ)
  (H1 : amount_max = 1200)
  (H2 : num_max = 500)
  (H3 : num_half = 3 * num_max)
  (H4 : total_donations = num_max * amount_max + num_half * (amount_max / 2))
  (H5 : total_donations = 40 * total_raised / 100) :
  total_raised = 3750000 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_tim_campaign_total_l1275_127553


namespace NUMINAMATH_GPT_usual_time_is_75_l1275_127530

variable (T : ℕ) -- let T be the usual time in minutes

theorem usual_time_is_75 (h1 : (6 * T) / 5 = T + 15) : T = 75 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_is_75_l1275_127530


namespace NUMINAMATH_GPT_domain_of_function_l1275_127576

theorem domain_of_function : 
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by 
  sorry

end NUMINAMATH_GPT_domain_of_function_l1275_127576


namespace NUMINAMATH_GPT_cos_double_angle_l1275_127535
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l1275_127535


namespace NUMINAMATH_GPT_certain_number_d_sq_l1275_127533

theorem certain_number_d_sq (d n m : ℕ) (hd : d = 14) (h : n * d = m^2) : n = 14 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_d_sq_l1275_127533


namespace NUMINAMATH_GPT_greatest_number_is_2040_l1275_127599

theorem greatest_number_is_2040 (certain_number : ℕ) : 
  (∀ d : ℕ, d ∣ certain_number ∧ d ∣ 2037 → d ≤ 1) ∧ 
  (certain_number % 1 = 10) ∧ 
  (2037 % 1 = 7) → 
  certain_number = 2040 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_is_2040_l1275_127599


namespace NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1275_127511

theorem geometric_sequence_sum_ratio (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_nonzero_q : q ≠ 0) 
  (a2 : a_n 2 = a_n 1 * q) (a5 : a_n 5 = a_n 1 * q^4) 
  (h_condition : 8 * a_n 2 + a_n 5 = 0)
  (h_sum : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) : 
  S 5 / S 2 = -11 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_ratio_l1275_127511


namespace NUMINAMATH_GPT_cloth_sales_value_l1275_127540

theorem cloth_sales_value (commission_rate : ℝ) (commission : ℝ) (total_sales : ℝ) 
  (h1: commission_rate = 2.5)
  (h2: commission = 18)
  (h3: total_sales = commission / (commission_rate / 100)):
  total_sales = 720 := by
  sorry

end NUMINAMATH_GPT_cloth_sales_value_l1275_127540


namespace NUMINAMATH_GPT_part1_part2_l1275_127598

theorem part1 (a : ℝ) (h1 : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ x1 * x2 + (a * x1 + 1) * (a * x2 + 1) = 0) : a = 1 ∨ a = -1 := sorry

theorem part2 (h : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (a : ℝ) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ (y1 + y2) / 2 = (1 / 2) * (x1 + x2) / 2 ∧ (y1 - y2) / (x1 - x2) = -2) : false := sorry

end NUMINAMATH_GPT_part1_part2_l1275_127598


namespace NUMINAMATH_GPT_problem1_problem2_l1275_127563

-- Problem (1)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 2) : 
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 := 
by 
  sorry

-- Problem (2)
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = -4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1275_127563


namespace NUMINAMATH_GPT_sqrt_x_div_sqrt_y_as_fraction_l1275_127509

theorem sqrt_x_div_sqrt_y_as_fraction 
  (x y : ℝ)
  (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = 54 * x / 115 * y * ((1/5)^2 + (1/7)^2 + (1/8)^2)) : 
  (Real.sqrt x) / (Real.sqrt y) = 49 / 29 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_x_div_sqrt_y_as_fraction_l1275_127509


namespace NUMINAMATH_GPT_complement_of_M_in_U_l1275_127532

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}
def C_U (M : Set ℕ) (U : Set ℕ) : Set ℕ := U \ M

theorem complement_of_M_in_U : C_U M U = {1, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l1275_127532


namespace NUMINAMATH_GPT_exist_coprime_integers_l1275_127586

theorem exist_coprime_integers:
  ∀ (a b p : ℤ), ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end NUMINAMATH_GPT_exist_coprime_integers_l1275_127586


namespace NUMINAMATH_GPT_bajazet_winning_strategy_l1275_127534

-- Define the polynomial P with place holder coefficients a, b, c (assuming they are real numbers)
def P (a b c : ℝ) (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + 1

-- The statement that regardless of how Alcina plays, Bajazet can ensure that P has a real root.
theorem bajazet_winning_strategy :
  ∃ (a b c : ℝ), ∃ (x : ℝ), P a b c x = 0 :=
sorry

end NUMINAMATH_GPT_bajazet_winning_strategy_l1275_127534


namespace NUMINAMATH_GPT_new_ratio_is_one_half_l1275_127506

theorem new_ratio_is_one_half (x : ℕ) (y : ℕ) (h1 : y = 4 * x) (h2 : y = 48) :
  (x + 12) / y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_new_ratio_is_one_half_l1275_127506


namespace NUMINAMATH_GPT_trapezoid_midsegment_l1275_127544

-- Define the problem conditions and question
theorem trapezoid_midsegment (b h x : ℝ) (h_nonzero : h ≠ 0) (hx : x = b + 75)
  (equal_areas : (1 / 2) * (h / 2) * (b + (b + 75)) = (1 / 2) * (h / 2) * ((b + 75) + (b + 150))) :
  ∃ n : ℤ, n = ⌊x^2 / 120⌋ ∧ n = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_trapezoid_midsegment_l1275_127544


namespace NUMINAMATH_GPT_candy_bar_cost_l1275_127546

variable (C : ℕ)

theorem candy_bar_cost
  (soft_drink_cost : ℕ)
  (num_candy_bars : ℕ)
  (total_spent : ℕ)
  (h1 : soft_drink_cost = 2)
  (h2 : num_candy_bars = 5)
  (h3 : total_spent = 27) :
  num_candy_bars * C + soft_drink_cost = total_spent → C = 5 := by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l1275_127546


namespace NUMINAMATH_GPT_graphs_intersect_once_l1275_127520

theorem graphs_intersect_once : 
  ∃! (x : ℝ), |3 * x + 6| = -|4 * x - 3| :=
sorry

end NUMINAMATH_GPT_graphs_intersect_once_l1275_127520


namespace NUMINAMATH_GPT_athlete_running_minutes_l1275_127528

theorem athlete_running_minutes (r w : ℕ) 
  (h1 : r + w = 60)
  (h2 : 10 * r + 4 * w = 450) : 
  r = 35 := 
sorry

end NUMINAMATH_GPT_athlete_running_minutes_l1275_127528


namespace NUMINAMATH_GPT_product_of_N1_N2_l1275_127574

theorem product_of_N1_N2 :
  (∃ (N1 N2 : ℤ),
    (∀ (x : ℚ),
      (47 * x - 35) * (x - 1) * (x - 2) = N1 * (x - 2) * (x - 1) + N2 * (x - 1) * (x - 2)) ∧
    N1 * N2 = -708) :=
sorry

end NUMINAMATH_GPT_product_of_N1_N2_l1275_127574


namespace NUMINAMATH_GPT_cyclists_meeting_l1275_127580

-- Define the velocities of the cyclists and the time variable
variables (v₁ v₂ t : ℝ)

-- Define the conditions for the problem
def condition1 : Prop := v₁ * t = v₂ * (2/3)
def condition2 : Prop := v₂ * t = v₁ * 1.5

-- Define the main theorem to be proven
theorem cyclists_meeting (h1 : condition1 v₁ v₂ t) (h2 : condition2 v₁ v₂ t) :
  t = 1 ∧ (v₁ / v₂ = 3 / 2) :=
by sorry

end NUMINAMATH_GPT_cyclists_meeting_l1275_127580


namespace NUMINAMATH_GPT_ages_total_l1275_127519

-- Define the variables and conditions
variables (A B C : ℕ)

-- State the conditions
def condition1 (B : ℕ) : Prop := B = 14
def condition2 (A B : ℕ) : Prop := A = B + 2
def condition3 (B C : ℕ) : Prop := B = 2 * C

-- The main theorem to prove
theorem ages_total (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 B C) : A + B + C = 37 :=
by
  sorry

end NUMINAMATH_GPT_ages_total_l1275_127519


namespace NUMINAMATH_GPT_solve_inequality_l1275_127501

noncomputable def P (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem solve_inequality (x : ℝ) : (P x > 0) ↔ (x < 1 ∨ x > 2) := 
  sorry

end NUMINAMATH_GPT_solve_inequality_l1275_127501


namespace NUMINAMATH_GPT_eggs_per_box_l1275_127524

-- Conditions
def num_eggs : ℝ := 3.0
def num_boxes : ℝ := 2.0

-- Theorem statement
theorem eggs_per_box (h1 : num_eggs = 3.0) (h2 : num_boxes = 2.0) : (num_eggs / num_boxes = 1.5) :=
sorry

end NUMINAMATH_GPT_eggs_per_box_l1275_127524


namespace NUMINAMATH_GPT_simple_interest_rate_l1275_127570

theorem simple_interest_rate (P R: ℝ) (T : ℝ) (hT : T = 8) (h : 2 * P = P + (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1275_127570


namespace NUMINAMATH_GPT_igor_min_score_needed_l1275_127590

theorem igor_min_score_needed
  (scores : List ℕ)
  (goal : ℚ)
  (next_test_score : ℕ)
  (h_scores : scores = [88, 92, 75, 83, 90])
  (h_goal : goal = 87)
  (h_solution : next_test_score = 94)
  : 
  let current_sum := scores.sum
  let current_tests := scores.length
  let required_total := (goal * (current_tests + 1))
  let next_test_needed := required_total - current_sum
  next_test_needed ≤ next_test_score := 
by 
  sorry

end NUMINAMATH_GPT_igor_min_score_needed_l1275_127590


namespace NUMINAMATH_GPT_catch_up_time_l1275_127547

noncomputable def speed_ratios (v : ℝ) : Prop :=
  let a_speed := (4 / 5) * v
  let b_speed := (2 / 5) * v
  a_speed = 2 * b_speed

theorem catch_up_time (v t : ℝ) (a_speed b_speed : ℝ)
  (h1 : a_speed = (4 / 5) * v)
  (h2 : b_speed = (2 / 5) * v)
  (h3 : a_speed = 2 * b_speed) :
  (t = 11) := by
  sorry

end NUMINAMATH_GPT_catch_up_time_l1275_127547


namespace NUMINAMATH_GPT_units_digit_35_pow_35_mul_17_pow_17_l1275_127542

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_35_pow_35_mul_17_pow_17:
  units_digit (35 ^ (35 * 17 ^ 17)) = 5 := 
by {
  -- Here we're skipping the proof.
  sorry
}

end NUMINAMATH_GPT_units_digit_35_pow_35_mul_17_pow_17_l1275_127542


namespace NUMINAMATH_GPT_range_of_a_l1275_127504

noncomputable def f (x a : ℝ) := Real.log x + a / x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 0, x * (2 * Real.log a - Real.log x) ≤ a) : 
  0 < a ∧ a ≤ 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1275_127504


namespace NUMINAMATH_GPT_multimedia_sets_max_profit_l1275_127560

-- Definitions of conditions:
def cost_A : ℝ := 3
def cost_B : ℝ := 2.4
def price_A : ℝ := 3.3
def price_B : ℝ := 2.8
def total_sets : ℕ := 50
def total_cost : ℝ := 132
def min_m : ℕ := 11

-- Problem 1: Prove the number of sets based on equations
theorem multimedia_sets (x y : ℕ) (h1 : x + y = total_sets) (h2 : cost_A * x + cost_B * y = total_cost) :
  x = 20 ∧ y = 30 :=
by sorry

-- Problem 2: Prove the maximum profit within a given range
theorem max_profit (m : ℕ) (h_m : 10 < m ∧ m < 20) :
  (-(0.1 : ℝ) * m + 20 = 18.9) ↔ m = min_m :=
by sorry

end NUMINAMATH_GPT_multimedia_sets_max_profit_l1275_127560


namespace NUMINAMATH_GPT_ratio_of_allergic_to_peanut_to_total_l1275_127515

def total_children : ℕ := 34
def children_not_allergic_to_cashew : ℕ := 10
def children_allergic_to_both : ℕ := 10
def children_allergic_to_cashew : ℕ := 18
def children_not_allergic_to_any : ℕ := 6
def children_allergic_to_peanut : ℕ := 20

theorem ratio_of_allergic_to_peanut_to_total :
  (children_allergic_to_peanut : ℚ) / (total_children : ℚ) = 10 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_allergic_to_peanut_to_total_l1275_127515


namespace NUMINAMATH_GPT_required_C6H6_for_C6H5CH3_and_H2_l1275_127575

-- Define the necessary molecular structures and stoichiometry
def C6H6 : Type := ℕ -- Benzene
def CH4 : Type := ℕ -- Methane
def C6H5CH3 : Type := ℕ -- Toluene
def H2 : Type := ℕ -- Hydrogen

-- Balanced equation condition
def balanced_reaction (x : C6H6) (y : CH4) (z : C6H5CH3) (w : H2) : Prop :=
  x = y ∧ x = z ∧ x = w

-- Given conditions
def condition (m : ℕ) : Prop :=
  balanced_reaction m m m m

theorem required_C6H6_for_C6H5CH3_and_H2 :
  ∀ (n : ℕ), condition n → n = 3 → n = 3 :=
by
  intros n h hn
  exact hn

end NUMINAMATH_GPT_required_C6H6_for_C6H5CH3_and_H2_l1275_127575


namespace NUMINAMATH_GPT_sum_h_k_a_b_l1275_127510

-- Defining h, k, a, and b with their respective given values
def h : Int := -4
def k : Int := 2
def a : Int := 5
def b : Int := 3

-- Stating the theorem to prove \( h + k + a + b = 6 \)
theorem sum_h_k_a_b : h + k + a + b = 6 := by
  /- Proof omitted as per instructions -/
  sorry

end NUMINAMATH_GPT_sum_h_k_a_b_l1275_127510


namespace NUMINAMATH_GPT_year_2022_form_l1275_127583

theorem year_2022_form :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    2001 ≤ (a + b * c * d * e) / (f + g * h * i * j) ∧ (a + b * c * d * e) / (f + g * h * i * j) ≤ 2100 ∧
    (a + b * c * d * e) / (f + g * h * i * j) = 2022 :=
sorry

end NUMINAMATH_GPT_year_2022_form_l1275_127583


namespace NUMINAMATH_GPT_find_possible_m_values_l1275_127517

theorem find_possible_m_values (m : ℕ) (a : ℕ) (h₀ : m > 1) (h₁ : m * a + (m * (m - 1) / 2) = 33) :
  m = 2 ∨ m = 3 ∨ m = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_possible_m_values_l1275_127517


namespace NUMINAMATH_GPT_greatest_percentage_l1275_127550

theorem greatest_percentage (pA : ℝ) (pB : ℝ) (wA : ℝ) (wB : ℝ) (sA : ℝ) (sB : ℝ) :
  pA = 0.4 → pB = 0.6 → wA = 0.8 → wB = 0.1 → sA = 0.9 → sB = 0.5 →
  pA * min wA sA + pB * min wB sB = 0.38 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Here you would continue with the proof by leveraging the conditions
  sorry

end NUMINAMATH_GPT_greatest_percentage_l1275_127550


namespace NUMINAMATH_GPT_coefficient_of_x_in_expansion_l1275_127567

theorem coefficient_of_x_in_expansion : 
  (Polynomial.coeff (((X ^ 2 + 3 * X + 2) ^ 6) : Polynomial ℤ) 1) = 576 := 
by 
  sorry

end NUMINAMATH_GPT_coefficient_of_x_in_expansion_l1275_127567


namespace NUMINAMATH_GPT_bob_salary_is_14400_l1275_127593

variables (mario_salary_current : ℝ) (mario_salary_last_year : ℝ) (bob_salary_last_year : ℝ) (bob_salary_current : ℝ)

-- Given Conditions
axiom mario_salary_increase : mario_salary_current = 4000
axiom mario_salary_equation : 1.40 * mario_salary_last_year = mario_salary_current
axiom bob_salary_last_year_equation : bob_salary_last_year = 3 * mario_salary_current
axiom bob_salary_increase : bob_salary_current = bob_salary_last_year + 0.20 * bob_salary_last_year

-- Theorem to prove
theorem bob_salary_is_14400 
    (mario_salary_last_year_eq : mario_salary_last_year = 4000 / 1.40)
    (bob_salary_last_year_eq : bob_salary_last_year = 3 * 4000)
    (bob_salary_current_eq : bob_salary_current = 12000 + 0.20 * 12000) :
    bob_salary_current = 14400 := 
by
  sorry

end NUMINAMATH_GPT_bob_salary_is_14400_l1275_127593


namespace NUMINAMATH_GPT_real_solutions_iff_a_geq_3_4_l1275_127554

theorem real_solutions_iff_a_geq_3_4:
  (∃ (x y : ℝ), x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3 / 4 := sorry

end NUMINAMATH_GPT_real_solutions_iff_a_geq_3_4_l1275_127554


namespace NUMINAMATH_GPT_coin_problem_l1275_127561

theorem coin_problem :
  ∃ (p n d q : ℕ), p + n + d + q = 11 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 132 ∧
                   p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ 
                   q = 3 :=
by
  sorry

end NUMINAMATH_GPT_coin_problem_l1275_127561


namespace NUMINAMATH_GPT_solve_recursive_fn_eq_l1275_127571

-- Define the recursive function
def recursive_fn (x : ℝ) : ℝ :=
  2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

-- State the theorem we need to prove
theorem solve_recursive_fn_eq (x : ℝ) : recursive_fn x = x → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_recursive_fn_eq_l1275_127571


namespace NUMINAMATH_GPT_volume_ratio_l1275_127537

noncomputable def V_D (s : ℝ) := (15 + 7 * Real.sqrt 5) * s^3 / 4
noncomputable def a (s : ℝ) := s / 2 * (1 + Real.sqrt 5)
noncomputable def V_I (a : ℝ) := 5 * (3 + Real.sqrt 5) * a^3 / 12

theorem volume_ratio (s : ℝ) (h₁ : 0 < s) :
  V_I (a s) / V_D s = (5 * (3 + Real.sqrt 5) * (1 + Real.sqrt 5)^3) / (12 * 2 * (15 + 7 * Real.sqrt 5)) :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l1275_127537


namespace NUMINAMATH_GPT_emily_speed_l1275_127595

theorem emily_speed (distance time : ℝ) (h1 : distance = 10) (h2 : time = 2) : (distance / time) = 5 := 
by sorry

end NUMINAMATH_GPT_emily_speed_l1275_127595


namespace NUMINAMATH_GPT_length_DE_l1275_127531

theorem length_DE (AB : ℝ) (h_base : AB = 15) (DE_parallel : ∀ x y z : Triangle, True) (area_ratio : ℝ) (h_area_ratio : area_ratio = 0.25) : 
  ∃ DE : ℝ, DE = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_length_DE_l1275_127531


namespace NUMINAMATH_GPT_monochromatic_rectangle_l1275_127558

theorem monochromatic_rectangle (n : ℕ) (coloring : ℕ × ℕ → Fin n) :
  ∃ (a b c d : ℕ × ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end NUMINAMATH_GPT_monochromatic_rectangle_l1275_127558


namespace NUMINAMATH_GPT_intersection_distance_l1275_127552

open Real

-- Definition of the curve C in standard coordinates
def curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Definition of the line l in parametric form
def line_l (x y t : ℝ) : Prop :=
  x = 1 + t ∧ y = -1 + t

-- The length of the intersection points A and B of curve C and line l
theorem intersection_distance : ∃ t1 t2 : ℝ, (curve_C (1 + t1) (-1 + t1) ∧ curve_C (1 + t2) (-1 + t2)) ∧ (abs (t1 - t2) = 4 * sqrt 6) :=
sorry

end NUMINAMATH_GPT_intersection_distance_l1275_127552


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1275_127556

theorem sum_of_two_numbers :
  (∃ x y : ℕ, y = 2 * x - 43 ∧ y = 31 ∧ x + y = 68) :=
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1275_127556


namespace NUMINAMATH_GPT_shaded_area_is_correct_l1275_127539

-- Defining the conditions
def grid_width : ℝ := 15 -- in units
def grid_height : ℝ := 5 -- in units
def total_grid_area : ℝ := grid_width * grid_height -- in square units

def larger_triangle_base : ℝ := grid_width -- in units
def larger_triangle_height : ℝ := grid_height -- in units
def larger_triangle_area : ℝ := 0.5 * larger_triangle_base * larger_triangle_height -- in square units

def smaller_triangle_base : ℝ := 3 -- in units
def smaller_triangle_height : ℝ := 2 -- in units
def smaller_triangle_area : ℝ := 0.5 * smaller_triangle_base * smaller_triangle_height -- in square units

-- The total area of the triangles that are not shaded
def unshaded_areas : ℝ := larger_triangle_area + smaller_triangle_area

-- The area of the shaded region
def shaded_area : ℝ := total_grid_area - unshaded_areas

-- The statement to be proven
theorem shaded_area_is_correct : shaded_area = 34.5 := 
by 
  -- This is a placeholder for the actual proof, which would normally go here
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l1275_127539


namespace NUMINAMATH_GPT_single_discount_percentage_l1275_127527

noncomputable def original_price : ℝ := 9795.3216374269
noncomputable def sale_price : ℝ := 6700
noncomputable def discount_percentage (p₀ p₁ : ℝ) : ℝ := ((p₀ - p₁) / p₀) * 100

theorem single_discount_percentage :
  discount_percentage original_price sale_price = 31.59 := 
by
  sorry

end NUMINAMATH_GPT_single_discount_percentage_l1275_127527


namespace NUMINAMATH_GPT_set_intersection_l1275_127578

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2, 5}
noncomputable def B : Set ℕ := {x ∈ U | (3 / (2 - x) + 1 ≤ 0)}
noncomputable def C_U_B : Set ℕ := U \ B

theorem set_intersection : A ∩ C_U_B = {1, 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_set_intersection_l1275_127578
