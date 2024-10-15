import Mathlib

namespace NUMINAMATH_GPT_neha_amount_removed_l898_89873

theorem neha_amount_removed (N S M : ℝ) (x : ℝ) (total_amnt : ℝ) (M_val : ℝ) (ratio2 : ℝ) (ratio8 : ℝ) (ratio6 : ℝ) :
  total_amnt = 1100 →
  M_val = 102 →
  ratio2 = 2 →
  ratio8 = 8 →
  ratio6 = 6 →
  (M - 4 = ratio6 * x) →
  (S - 8 = ratio8 * x) →
  (N - (N - (ratio2 * x)) = ratio2 * x) →
  (N + S + M = total_amnt) →
  (N - 32.66 = N - (ratio2 * (total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6))) →
  N - (N - (ratio2 * ((total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6)))) = 826.70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_neha_amount_removed_l898_89873


namespace NUMINAMATH_GPT_time_after_increment_l898_89813

-- Define the current time in minutes
def current_time_minutes : ℕ := 15 * 60  -- 3:00 p.m. in minutes

-- Define the time increment in minutes
def time_increment : ℕ := 1567

-- Calculate the total time in minutes after the increment
def total_time_minutes : ℕ := current_time_minutes + time_increment

-- Convert total time back to hours and minutes
def calculated_hours : ℕ := total_time_minutes / 60
def calculated_minutes : ℕ := total_time_minutes % 60

-- The expected hours and minutes after the increment
def expected_hours : ℕ := 17 -- 17:00 hours which is 5:00 p.m.
def expected_minutes : ℕ := 7 -- 7 minutes

theorem time_after_increment :
  (calculated_hours - 24 * (calculated_hours / 24) = expected_hours) ∧ (calculated_minutes = expected_minutes) :=
by
  sorry

end NUMINAMATH_GPT_time_after_increment_l898_89813


namespace NUMINAMATH_GPT_lines_intersect_and_find_point_l898_89840

theorem lines_intersect_and_find_point (n : ℝ)
  (h₁ : ∀ t : ℝ, ∃ (x y z : ℝ), x / 2 = t ∧ y / -3 = t ∧ z / n = t)
  (h₂ : ∀ t : ℝ, ∃ (x y z : ℝ), (x + 1) / 3 = t ∧ (y + 5) / 2 = t ∧ z / 1 = t) :
  n = 1 ∧ (∃ (x y z : ℝ), x = 2 ∧ y = -3 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_lines_intersect_and_find_point_l898_89840


namespace NUMINAMATH_GPT_larger_square_area_l898_89801

theorem larger_square_area 
    (s₁ s₂ s₃ s₄ : ℕ) 
    (H1 : s₁ = 20) 
    (H2 : s₂ = 10) 
    (H3 : s₃ = 18) 
    (H4 : s₄ = 12) :
    (s₃ + s₄) > (s₁ + s₂) :=
by
  sorry

end NUMINAMATH_GPT_larger_square_area_l898_89801


namespace NUMINAMATH_GPT_x0_equals_pm1_l898_89852

-- Define the function f and its second derivative
def f (x : ℝ) : ℝ := x^3
def f'' (x : ℝ) : ℝ := 6 * x

-- Prove that if f''(x₀) = 6 then x₀ = ±1
theorem x0_equals_pm1 (x0 : ℝ) (h : f'' x0 = 6) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end NUMINAMATH_GPT_x0_equals_pm1_l898_89852


namespace NUMINAMATH_GPT_g_at_minus_six_l898_89865

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end NUMINAMATH_GPT_g_at_minus_six_l898_89865


namespace NUMINAMATH_GPT_one_of_sum_of_others_l898_89888

theorem one_of_sum_of_others (a b c : ℝ) 
  (cond1 : |a - b| ≥ |c|)
  (cond2 : |b - c| ≥ |a|)
  (cond3 : |c - a| ≥ |b|) :
  (a = b + c) ∨ (b = c + a) ∨ (c = a + b) :=
by
  sorry

end NUMINAMATH_GPT_one_of_sum_of_others_l898_89888


namespace NUMINAMATH_GPT_total_bike_count_l898_89863

def total_bikes (bikes_jungkook bikes_yoongi : Nat) : Nat :=
  bikes_jungkook + bikes_yoongi

theorem total_bike_count : total_bikes 3 4 = 7 := 
  by 
  sorry

end NUMINAMATH_GPT_total_bike_count_l898_89863


namespace NUMINAMATH_GPT_find_C_coordinates_l898_89851

noncomputable def pointC_coordinates : Prop :=
  let A : (ℝ × ℝ) := (-2, 1)
  let B : (ℝ × ℝ) := (4, 9)
  ∃ C : (ℝ × ℝ), 
    (dist (A.1, A.2) (C.1, C.2) = 2 * dist (B.1, B.2) (C.1, C.2)) ∧ 
    C = (2, 19 / 3)

theorem find_C_coordinates : pointC_coordinates :=
  sorry

end NUMINAMATH_GPT_find_C_coordinates_l898_89851


namespace NUMINAMATH_GPT_vasya_wins_l898_89895

/-
  Petya and Vasya are playing a game where initially there are 2022 boxes, 
  each containing exactly one matchstick. In one move, a player can transfer 
  all matchsticks from one non-empty box to another non-empty box. They take turns, 
  with Petya starting first. The winner is the one who, after their move, has 
  at least half of all the matchsticks in one box for the first time. 

  We want to prove that Vasya will win the game with the optimal strategy.
-/

theorem vasya_wins : true :=
  sorry -- placeholder for the actual proof

end NUMINAMATH_GPT_vasya_wins_l898_89895


namespace NUMINAMATH_GPT_find_number_l898_89830

theorem find_number (N : ℝ) (h : 0.015 * N = 90) : N = 6000 :=
  sorry

end NUMINAMATH_GPT_find_number_l898_89830


namespace NUMINAMATH_GPT_unique_k_value_l898_89872
noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m ∣ n → m = n

theorem unique_k_value :
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 74 ∧ p * q = 213) ∧
  ∀ (p₁ q₁ k₁ p₂ q₂ k₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ p₁ + q₁ = 74 ∧ p₁ * q₁ = k₁ ∧
    is_prime p₂ ∧ is_prime q₂ ∧ p₂ + q₂ = 74 ∧ p₂ * q₂ = k₂ →
    k₁ = k₂ :=
by
  sorry

end NUMINAMATH_GPT_unique_k_value_l898_89872


namespace NUMINAMATH_GPT_general_term_formula_l898_89818

def seq (n : ℕ) : ℤ :=
  match n with
  | 1     => 2
  | 2     => -6
  | 3     => 12
  | 4     => -20
  | 5     => 30
  | 6     => -42
  | _     => 0 -- We match only the first few elements as given

theorem general_term_formula (n : ℕ) :
  seq n = (-1)^(n+1) * n * (n + 1) := by
  sorry

end NUMINAMATH_GPT_general_term_formula_l898_89818


namespace NUMINAMATH_GPT_fraction_identity_l898_89878

noncomputable def calc_fractions (x y : ℝ) : ℝ :=
  (x + y) / (x - y)

theorem fraction_identity (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) : calc_fractions x y = -1001 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l898_89878


namespace NUMINAMATH_GPT_total_bales_in_barn_l898_89809

-- Definitions based on the conditions 
def initial_bales : ℕ := 47
def added_bales : ℕ := 35

-- Statement to prove the final number of bales in the barn
theorem total_bales_in_barn : initial_bales + added_bales = 82 :=
by
  sorry

end NUMINAMATH_GPT_total_bales_in_barn_l898_89809


namespace NUMINAMATH_GPT_ratio_of_sums_l898_89876

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def square_of_sum (n : ℕ) : ℚ :=
  ((n * (n + 1)) / 2) ^ 2

theorem ratio_of_sums (n : ℕ) (h : n = 25) :
  sum_of_squares n / square_of_sum n = 1 / 19 :=
by
  have hn : n = 25 := h
  rw [hn]
  dsimp [sum_of_squares, square_of_sum]
  have : (25 * (25 + 1) * (2 * 25 + 1)) / 6 = 5525 := by norm_num
  have : ((25 * (25 + 1)) / 2) ^ 2 = 105625 := by norm_num
  norm_num
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l898_89876


namespace NUMINAMATH_GPT_expression_equals_20_over_9_l898_89892

noncomputable def complex_fraction_expression := 
  let a := 11 + 1 / 9
  let b := 3 + 2 / 5
  let c := 1 + 2 / 17
  let d := 8 + 2 / 5
  let e := 3.6
  let f := 2 + 6 / 25
  ((a - b * c) - d / e) / f

theorem expression_equals_20_over_9 : complex_fraction_expression = 20 / 9 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_20_over_9_l898_89892


namespace NUMINAMATH_GPT_p_finishes_job_after_q_in_24_minutes_l898_89847

theorem p_finishes_job_after_q_in_24_minutes :
  let P_rate := 1 / 4
  let Q_rate := 1 / 20
  let together_rate := P_rate + Q_rate
  let work_done_in_3_hours := together_rate * 3
  let remaining_work := 1 - work_done_in_3_hours
  let time_for_p_to_finish := remaining_work / P_rate
  let time_in_minutes := time_for_p_to_finish * 60
  time_in_minutes = 24 :=
by
  sorry

end NUMINAMATH_GPT_p_finishes_job_after_q_in_24_minutes_l898_89847


namespace NUMINAMATH_GPT_greater_than_neg2_by_1_l898_89835

theorem greater_than_neg2_by_1 : -2 + 1 = -1 := by
  sorry

end NUMINAMATH_GPT_greater_than_neg2_by_1_l898_89835


namespace NUMINAMATH_GPT_angle_PQC_in_triangle_l898_89884

theorem angle_PQC_in_triangle 
  (A B C P Q: ℝ)
  (h_in_triangle: A + B + C = 180)
  (angle_B_exterior_bisector: ∀ B_ext, B_ext = 180 - B →  angle_B = 90 - B / 2)
  (angle_C_exterior_bisector: ∀ C_ext, C_ext = 180 - C →  angle_C = 90 - C / 2)
  (h_PQ_BC_angle: ∀ PQ_angle BC_angle, PQ_angle = 30 → BC_angle = 30) :
  ∃ PQC_angle, PQC_angle = (180 - A) / 2 :=
by
  sorry

end NUMINAMATH_GPT_angle_PQC_in_triangle_l898_89884


namespace NUMINAMATH_GPT_new_profit_percentage_l898_89853

def original_cost (c : ℝ) : ℝ := c
def original_selling_price (c : ℝ) : ℝ := 1.2 * c
def new_cost (c : ℝ) : ℝ := 0.9 * c
def new_selling_price (c : ℝ) : ℝ := 1.05 * 1.2 * c

theorem new_profit_percentage (c : ℝ) (hc : c > 0) :
  ((new_selling_price c - new_cost c) / new_cost c) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_new_profit_percentage_l898_89853


namespace NUMINAMATH_GPT_count_arrangements_california_l898_89845

-- Defining the counts of letters in "CALIFORNIA"
def word_length : ℕ := 10
def count_A : ℕ := 3
def count_I : ℕ := 2
def count_C : ℕ := 1
def count_L : ℕ := 1
def count_F : ℕ := 1
def count_O : ℕ := 1
def count_R : ℕ := 1
def count_N : ℕ := 1

-- The final proof statement to show the number of unique arrangements
theorem count_arrangements_california : 
  (Nat.factorial word_length) / 
  ((Nat.factorial count_A) * (Nat.factorial count_I)) = 302400 := by
  -- Placeholder for the proof, can be filled in later by providing the actual steps
  sorry

end NUMINAMATH_GPT_count_arrangements_california_l898_89845


namespace NUMINAMATH_GPT_point_G_six_l898_89834

theorem point_G_six : 
  ∃ (A B C D E F G : ℕ), 
    1 ≤ A ∧ A ≤ 10 ∧
    1 ≤ B ∧ B ≤ 10 ∧
    1 ≤ C ∧ C ≤ 10 ∧
    1 ≤ D ∧ D ≤ 10 ∧
    1 ≤ E ∧ E ≤ 10 ∧
    1 ≤ F ∧ F ≤ 10 ∧
    1 ≤ G ∧ G ≤ 10 ∧
    (A + B = A + C + D) ∧ 
    (A + B = B + E + F) ∧
    (A + B = C + F + G) ∧
    (A + B = D + E + G) ∧ 
    (A + B = 12) →
    G = 6 := 
by
  sorry

end NUMINAMATH_GPT_point_G_six_l898_89834


namespace NUMINAMATH_GPT_ratio_of_height_to_width_l898_89825

-- Define variables
variable (W H L V : ℕ)
variable (x : ℝ)

-- Given conditions
def condition_1 := W = 3
def condition_2 := H = x * W
def condition_3 := L = 7 * H
def condition_4 := V = 6804

-- Prove that the ratio of height to width is 6√3
theorem ratio_of_height_to_width : (W = 3 ∧ H = x * W ∧ L = 7 * H ∧ V = 6804 ∧ V = W * H * L) → x = 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_height_to_width_l898_89825


namespace NUMINAMATH_GPT_min_chord_length_l898_89828

-- Definitions of the problem conditions
def circle_center : ℝ × ℝ := (2, 3)
def circle_radius : ℝ := 3
def point_P : ℝ × ℝ := (1, 1)

-- The mathematical statement to prove
theorem min_chord_length : 
  ∀ (A B : ℝ × ℝ), 
  (A ≠ B) ∧ ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ ((B.1 - 2)^2 + (B.2 - 3)^2 = 9) ∧ 
  ((A.1 - 1) / (B.1 - 1) = (A.2 - 1) / (B.2 - 1)) → 
  dist A B ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_chord_length_l898_89828


namespace NUMINAMATH_GPT_fractions_zero_condition_l898_89836

variable {a b c : ℝ}

theorem fractions_zero_condition 
  (h : (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0) :
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := 
sorry

end NUMINAMATH_GPT_fractions_zero_condition_l898_89836


namespace NUMINAMATH_GPT_fraction_of_mothers_with_full_time_jobs_l898_89827

theorem fraction_of_mothers_with_full_time_jobs :
  (0.4 : ℝ) * M = 0.3 →
  (9 / 10 : ℝ) * 0.6 = 0.54 →
  1 - 0.16 = 0.84 →
  0.84 - 0.54 = 0.3 →
  M = 3 / 4 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_fraction_of_mothers_with_full_time_jobs_l898_89827


namespace NUMINAMATH_GPT_road_construction_problem_l898_89822

theorem road_construction_problem (x : ℝ) (h₁ : x > 0) :
    1200 / x - 1200 / (1.20 * x) = 2 :=
by
  sorry

end NUMINAMATH_GPT_road_construction_problem_l898_89822


namespace NUMINAMATH_GPT_bob_gave_terry_24_bushels_l898_89886

def bushels_given_to_terry (total_bushels : ℕ) (ears_per_bushel : ℕ) (ears_left : ℕ) : ℕ :=
    (total_bushels * ears_per_bushel - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels : bushels_given_to_terry 50 14 357 = 24 := by
    sorry

end NUMINAMATH_GPT_bob_gave_terry_24_bushels_l898_89886


namespace NUMINAMATH_GPT_terminating_decimal_contains_digit_3_l898_89824

theorem terminating_decimal_contains_digit_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ a b : ℕ, n = 2 ^ a * 5 ^ b) ∧ (∃ d, n = d * 10 ^ 0 + 3) ∧ n = 32 :=
by sorry

end NUMINAMATH_GPT_terminating_decimal_contains_digit_3_l898_89824


namespace NUMINAMATH_GPT_sum_of_digits_2_2010_mul_5_2012_mul_7_l898_89890

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_2_2010_mul_5_2012_mul_7 : 
  sum_of_digits (2^2010 * 5^2012 * 7) = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_digits_2_2010_mul_5_2012_mul_7_l898_89890


namespace NUMINAMATH_GPT_quadratic_has_real_root_l898_89808

theorem quadratic_has_real_root (a b : ℝ) : (∃ x : ℝ, x^2 + a * x + b = 0) :=
by
  -- To use contradiction, we assume the negation
  have h : ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry
  -- By contradiction, this assumption should lead to a contradiction
  sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l898_89808


namespace NUMINAMATH_GPT_largest_number_of_cakes_l898_89839

theorem largest_number_of_cakes : ∃ (c : ℕ), c = 65 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_of_cakes_l898_89839


namespace NUMINAMATH_GPT_books_sale_correct_l898_89875

variable (books_original books_left : ℕ)

def books_sold (books_original books_left : ℕ) : ℕ :=
  books_original - books_left

theorem books_sale_correct : books_sold 108 66 = 42 := by
  -- Since there is no need for the solution steps, we can assert the proof
  sorry

end NUMINAMATH_GPT_books_sale_correct_l898_89875


namespace NUMINAMATH_GPT_work_together_days_l898_89891

theorem work_together_days (a_days : ℕ) (b_days : ℕ) :
  a_days = 10 → b_days = 9 → (1 / ((1 / (a_days : ℝ)) + (1 / (b_days : ℝ)))) = 90 / 19 :=
by
  intros ha hb
  sorry

end NUMINAMATH_GPT_work_together_days_l898_89891


namespace NUMINAMATH_GPT_melanie_trout_catch_l898_89800

theorem melanie_trout_catch (T M : ℕ) 
  (h1 : T = 2 * M) 
  (h2 : T = 16) : 
  M = 8 :=
by
  sorry

end NUMINAMATH_GPT_melanie_trout_catch_l898_89800


namespace NUMINAMATH_GPT_problem_solution_set_l898_89860

theorem problem_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ ax^2 + x + b > 0) : a + b = -1 :=
sorry

end NUMINAMATH_GPT_problem_solution_set_l898_89860


namespace NUMINAMATH_GPT_eval_f_function_l898_89810

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem eval_f_function : f (f (f (-1))) = Real.pi + 1 :=
  sorry

end NUMINAMATH_GPT_eval_f_function_l898_89810


namespace NUMINAMATH_GPT_bill_left_with_22_l898_89864

def bill_earnings (ounces : ℕ) (rate_per_ounce : ℕ) : ℕ :=
  ounces * rate_per_ounce

def bill_remaining_money (total_earnings : ℕ) (fine : ℕ) : ℕ :=
  total_earnings - fine

theorem bill_left_with_22 (ounces sold_rate fine total_remaining : ℕ)
  (h1 : ounces = 8)
  (h2 : sold_rate = 9)
  (h3 : fine = 50)
  (h4 : total_remaining = 22)
  : bill_remaining_money (bill_earnings ounces sold_rate) fine = total_remaining :=
by
  sorry

end NUMINAMATH_GPT_bill_left_with_22_l898_89864


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l898_89826

theorem largest_angle_in_triangle (x : ℝ) (h1 : 40 + 60 + x = 180) (h2 : max 40 60 ≤ x) : x = 80 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l898_89826


namespace NUMINAMATH_GPT_hourly_rate_for_carriage_l898_89866

theorem hourly_rate_for_carriage
  (d : ℕ) (s : ℕ) (f : ℕ) (c : ℕ)
  (h_d : d = 20)
  (h_s : s = 10)
  (h_f : f = 20)
  (h_c : c = 80) :
  (c - f) / (d / s) = 30 := by
  sorry

end NUMINAMATH_GPT_hourly_rate_for_carriage_l898_89866


namespace NUMINAMATH_GPT_geometric_power_inequality_l898_89816

theorem geometric_power_inequality {a : ℝ} {n k : ℕ} (h₀ : 1 < a) (h₁ : 0 < n) (h₂ : n < k) :
  (a^n - 1) / n < (a^k - 1) / k :=
sorry

end NUMINAMATH_GPT_geometric_power_inequality_l898_89816


namespace NUMINAMATH_GPT_remainder_of_power_of_five_modulo_500_l898_89885

theorem remainder_of_power_of_five_modulo_500 :
  (5 ^ (5 ^ (5 ^ 2))) % 500 = 25 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_power_of_five_modulo_500_l898_89885


namespace NUMINAMATH_GPT_cube_root_of_neg8_l898_89837

-- Define the condition
def is_cube_root (x : ℝ) : Prop := x^3 = -8

-- State the problem to be proved.
theorem cube_root_of_neg8 : is_cube_root (-2) :=
by 
  sorry

end NUMINAMATH_GPT_cube_root_of_neg8_l898_89837


namespace NUMINAMATH_GPT_inequality_solution_l898_89823

theorem inequality_solution (x : ℝ) : x + 1 < (4 + 3 * x) / 2 → x > -2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_inequality_solution_l898_89823


namespace NUMINAMATH_GPT_distance_between_A_and_B_l898_89850

-- Given conditions as definitions

def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the possible solutions for the distance between A and B
def distance_AB (x : ℝ) := 
  (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time) 
  ∨ 
  (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time)

-- Problem statement
theorem distance_between_A_and_B :
  ∃ x : ℝ, (distance_AB x) ∧ (x = 20 ∨ x = 20 / 3) :=
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l898_89850


namespace NUMINAMATH_GPT_no_such_real_numbers_l898_89880

noncomputable def have_integer_roots (a b c : ℝ) : Prop :=
  ∃ r s : ℤ, a * (r:ℝ)^2 + b * r + c = 0 ∧ a * (s:ℝ)^2 + b * s + c = 0

theorem no_such_real_numbers (a b c : ℝ) :
  have_integer_roots a b c → have_integer_roots (a + 1) (b + 1) (c + 1) → False :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_no_such_real_numbers_l898_89880


namespace NUMINAMATH_GPT_no_common_elements_in_sequences_l898_89833

theorem no_common_elements_in_sequences :
  ∀ (k : ℕ), (∃ n : ℕ, k = n^2 - 1) ∧ (∃ m : ℕ, k = m^2 + 1) → False :=
by sorry

end NUMINAMATH_GPT_no_common_elements_in_sequences_l898_89833


namespace NUMINAMATH_GPT_problem1_problem2_l898_89812

namespace MathProblem

-- Problem 1
theorem problem1 : (π - 2)^0 + (-1)^3 = 0 := by
  sorry

-- Problem 2
variable (m n : ℤ)

theorem problem2 : (3 * m + n) * (m - 2 * n) = 3 * m ^ 2 - 5 * m * n - 2 * n ^ 2 := by
  sorry

end MathProblem

end NUMINAMATH_GPT_problem1_problem2_l898_89812


namespace NUMINAMATH_GPT_lowest_score_to_average_90_l898_89842

theorem lowest_score_to_average_90 {s1 s2 s3 max_score avg_score : ℕ} 
    (h1: s1 = 88) 
    (h2: s2 = 96) 
    (h3: s3 = 105) 
    (hmax: max_score = 120) 
    (havg: avg_score = 90) 
    : ∃ s4 s5, s4 ≤ max_score ∧ s5 ≤ max_score ∧ (s1 + s2 + s3 + s4 + s5) / 5 = avg_score ∧ (min s4 s5 = 41) :=
by {
    sorry
}

end NUMINAMATH_GPT_lowest_score_to_average_90_l898_89842


namespace NUMINAMATH_GPT_track_length_l898_89841

theorem track_length (x : ℕ) 
  (diametrically_opposite : ∃ a b : ℕ, a + b = x)
  (first_meeting : ∃ b : ℕ, b = 100)
  (second_meeting : ∃ s s' : ℕ, s = 150 ∧ s' = (x / 2 - 100 + s))
  (constant_speed : ∀ t₁ t₂ : ℕ, t₁ / t₂ = 100 / (x / 2 - 100)) :
  x = 400 := 
by sorry

end NUMINAMATH_GPT_track_length_l898_89841


namespace NUMINAMATH_GPT_January_to_November_ratio_l898_89846

variable (N D J : ℝ)

-- Condition 1: November revenue is 3/5 of December revenue
axiom revenue_Nov : N = (3 / 5) * D

-- Condition 2: December revenue is 2.5 times the average of November and January revenues
axiom revenue_Dec : D = 2.5 * (N + J) / 2

-- Goal: Prove the ratio of January revenue to November revenue is 1/3
theorem January_to_November_ratio : J / N = 1 / 3 :=
by
  -- We will use the given axioms to derive the proof
  sorry

end NUMINAMATH_GPT_January_to_November_ratio_l898_89846


namespace NUMINAMATH_GPT_equality_of_x_and_y_l898_89806

theorem equality_of_x_and_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x^(y^x) = y^(x^y)) : x = y :=
sorry

end NUMINAMATH_GPT_equality_of_x_and_y_l898_89806


namespace NUMINAMATH_GPT_range_of_f_at_most_7_l898_89894

theorem range_of_f_at_most_7 (f : ℤ × ℤ → ℝ)
  (H : ∀ (x y m n : ℤ), f (x + 3 * m - 2 * n, y - 4 * m + 5 * n) = f (x, y)) :
  ∃ (s : Finset ℝ), s.card ≤ 7 ∧ ∀ (a : ℤ × ℤ), f a ∈ s :=
sorry

end NUMINAMATH_GPT_range_of_f_at_most_7_l898_89894


namespace NUMINAMATH_GPT_root_condition_l898_89881

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m * x + m

theorem root_condition (m l : ℝ) (h : m < l) : 
  (∀ x : ℝ, f x m = 0 → x ≠ x) ∨ (∃ x : ℝ, f x m = 0) :=
sorry

end NUMINAMATH_GPT_root_condition_l898_89881


namespace NUMINAMATH_GPT_no_valid_coloring_l898_89883

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end NUMINAMATH_GPT_no_valid_coloring_l898_89883


namespace NUMINAMATH_GPT_miles_driven_each_day_l898_89844

-- Definition of the given conditions
def total_miles : ℝ := 1250
def number_of_days : ℝ := 5.0

-- The statement to be proved
theorem miles_driven_each_day :
  total_miles / number_of_days = 250 :=
by
  sorry

end NUMINAMATH_GPT_miles_driven_each_day_l898_89844


namespace NUMINAMATH_GPT_nh4i_required_l898_89832

theorem nh4i_required (KOH NH4I NH3 KI H2O : ℕ) (h_eq : 1 * NH4I + 1 * KOH = 1 * NH3 + 1 * KI + 1 * H2O)
  (h_KOH : KOH = 3) : NH4I = 3 := 
by
  sorry

end NUMINAMATH_GPT_nh4i_required_l898_89832


namespace NUMINAMATH_GPT_uncovered_side_length_l898_89877

theorem uncovered_side_length (L W : ℝ) (h1 : L * W = 120) (h2 : L + 2 * W = 32) : L = 20 :=
sorry

end NUMINAMATH_GPT_uncovered_side_length_l898_89877


namespace NUMINAMATH_GPT_num_foxes_l898_89870

structure Creature :=
  (is_squirrel : Bool)
  (is_fox : Bool)
  (is_salamander : Bool)

def Anna : Creature := sorry
def Bob : Creature := sorry
def Cara : Creature := sorry
def Daniel : Creature := sorry

def tells_truth (c : Creature) : Bool :=
  c.is_squirrel || (c.is_salamander && ¬c.is_fox)

def Anna_statement : Prop := Anna.is_fox ≠ Daniel.is_fox
def Bob_statement : Prop := tells_truth Bob ↔ Cara.is_salamander
def Cara_statement : Prop := tells_truth Cara ↔ Bob.is_fox
def Daniel_statement : Prop := tells_truth Daniel ↔ (Anna.is_squirrel ∧ Bob.is_squirrel ∧ Cara.is_squirrel ∨ Daniel.is_squirrel)

theorem num_foxes :
  (Anna.is_fox + Bob.is_fox + Cara.is_fox + Daniel.is_fox = 2) :=
  sorry

end NUMINAMATH_GPT_num_foxes_l898_89870


namespace NUMINAMATH_GPT_triangle_right_l898_89882

theorem triangle_right (a b c : ℝ) (h₀ : a ≠ c) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2 * a * x₀ + b^2 = 0 ∧ x₀^2 + 2 * c * x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := 
sorry

end NUMINAMATH_GPT_triangle_right_l898_89882


namespace NUMINAMATH_GPT_max_value_of_expression_l898_89831

theorem max_value_of_expression 
  (x y : ℝ)
  (h : x^2 + y^2 = 20 * x + 9 * y + 9) :
  ∃ x y : ℝ, 4 * x + 3 * y = 83 := sorry

end NUMINAMATH_GPT_max_value_of_expression_l898_89831


namespace NUMINAMATH_GPT_largest_factor_of_form_l898_89869

theorem largest_factor_of_form (n : ℕ) (h : n % 10 = 4) : 120 ∣ n * (n + 1) * (n + 2) :=
sorry

end NUMINAMATH_GPT_largest_factor_of_form_l898_89869


namespace NUMINAMATH_GPT_most_frequent_third_number_l898_89805

def is_lottery_condition (e1 e2 e3 e4 e5 : ℕ) : Prop :=
  1 ≤ e1 ∧ e1 < e2 ∧ e2 < e3 ∧ e3 < e4 ∧ e4 < e5 ∧ e5 ≤ 90 ∧ (e1 + e2 = e3)

theorem most_frequent_third_number :
  ∃ h : ℕ, 3 ≤ h ∧ h ≤ 88 ∧ (∀ h', (h' = 31 → ¬ (31 < h')) ∧ 
        ∀ e1 e2 e3 e4 e5, is_lottery_condition e1 e2 e3 e4 e5 → e3 = h) :=
sorry

end NUMINAMATH_GPT_most_frequent_third_number_l898_89805


namespace NUMINAMATH_GPT_second_divisor_203_l898_89854

theorem second_divisor_203 (x : ℕ) (h1 : 210 % 13 = 3) (h2 : 210 % x = 7) : x = 203 :=
by sorry

end NUMINAMATH_GPT_second_divisor_203_l898_89854


namespace NUMINAMATH_GPT_operation_result_l898_89879

def operation (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem operation_result : operation 3 (-1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_operation_result_l898_89879


namespace NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l898_89898

theorem solve_first_equation (x : ℝ) : 3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 :=
by {
  sorry
}

theorem solve_second_equation (x : ℝ) : 2 * (x + 1)^3 + 54 = 0 ↔ x = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l898_89898


namespace NUMINAMATH_GPT_solve_for_a_b_c_l898_89896

-- Conditions and necessary context
def m_angle_A : ℝ := 60  -- In degrees
def BC_length : ℝ := 12  -- Length of BC in units
def angle_DBC_eq_three_times_angle_ECB (DBC ECB : ℝ) : Prop := DBC = 3 * ECB

-- Definitions for perpendicularity could be checked by defining angles
-- between lines, but we can assert these as properties.
axiom BD_perpendicular_AC : Prop
axiom CE_perpendicular_AB : Prop

-- The proof problem
theorem solve_for_a_b_c :
  ∃ (EC a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  b ≠ c ∧ 
  (∀ d, b ∣ d → d = b ∨ d = 1) ∧ 
  (∀ d, c ∣ d → d = c ∨ d = 1) ∧
  EC = a * (Real.sqrt b + Real.sqrt c) ∧ 
  a + b + c = 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_b_c_l898_89896


namespace NUMINAMATH_GPT_pumpkin_patch_pie_filling_l898_89803

def pumpkin_cans (small_pumpkins : ℕ) (large_pumpkins : ℕ) (sales : ℕ) (small_price : ℕ) (large_price : ℕ) : ℕ :=
  let remaining_small_pumpkins := small_pumpkins
  let remaining_large_pumpkins := large_pumpkins
  let small_cans := remaining_small_pumpkins / 2
  let large_cans := remaining_large_pumpkins
  small_cans + large_cans

#eval pumpkin_cans 50 33 120 3 5 -- This evaluates the function with the given data to ensure the logic matches the question

theorem pumpkin_patch_pie_filling : pumpkin_cans 50 33 120 3 5 = 58 := by sorry

end NUMINAMATH_GPT_pumpkin_patch_pie_filling_l898_89803


namespace NUMINAMATH_GPT_largest_integer_of_five_with_product_12_l898_89821

theorem largest_integer_of_five_with_product_12 (a b c d e : ℤ) (h : a * b * c * d * e = 12) (h_diff : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ d ∧ b ≠ e ∧ c ≠ e) : 
  max a (max b (max c (max d e))) = 3 :=
sorry

end NUMINAMATH_GPT_largest_integer_of_five_with_product_12_l898_89821


namespace NUMINAMATH_GPT_count_indistinguishable_distributions_l898_89829

theorem count_indistinguishable_distributions (balls : ℕ) (boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) : 
  ∃ n : ℕ, n = 6 := by
  sorry

end NUMINAMATH_GPT_count_indistinguishable_distributions_l898_89829


namespace NUMINAMATH_GPT_problem_solution_l898_89868

def f (x : ℕ) : ℝ := sorry

axiom f_add_eq_mul (p q : ℕ) : f (p + q) = f p * f q
axiom f_one_eq_three : f 1 = 3

theorem problem_solution :
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 + 
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 = 24 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l898_89868


namespace NUMINAMATH_GPT_tan_alpha_fraction_eq_five_sevenths_l898_89802

theorem tan_alpha_fraction_eq_five_sevenths (α : ℝ) (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 :=
sorry

end NUMINAMATH_GPT_tan_alpha_fraction_eq_five_sevenths_l898_89802


namespace NUMINAMATH_GPT_total_cost_is_correct_l898_89807

def cost_shirt (S : ℝ) : Prop := S = 12
def cost_shoes (Sh S : ℝ) : Prop := Sh = S + 5
def cost_dress (D : ℝ) : Prop := D = 25
def discount_shoes (Sh Sh' : ℝ) : Prop := Sh' = Sh - 0.10 * Sh
def discount_dress (D D' : ℝ) : Prop := D' = D - 0.05 * D
def cost_bag (B twoS Sh' D' : ℝ) : Prop := B = (twoS + Sh' + D') / 2
def total_cost_before_tax (T_before twoS Sh' D' B : ℝ) : Prop := T_before = twoS + Sh' + D' + B
def sales_tax (tax T_before : ℝ) : Prop := tax = 0.07 * T_before
def total_cost_including_tax (T_total T_before tax : ℝ) : Prop := T_total = T_before + tax
def convert_to_usd (T_usd T_total : ℝ) : Prop := T_usd = T_total * 1.18

theorem total_cost_is_correct (S Sh D Sh' D' twoS B T_before tax T_total T_usd : ℝ) :
  cost_shirt S →
  cost_shoes Sh S →
  cost_dress D →
  discount_shoes Sh Sh' →
  discount_dress D D' →
  twoS = 2 * S →
  cost_bag B twoS Sh' D' →
  total_cost_before_tax T_before twoS Sh' D' B →
  sales_tax tax T_before →
  total_cost_including_tax T_total T_before tax →
  convert_to_usd T_usd T_total →
  T_usd = 119.42 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l898_89807


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l898_89871

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 2) (h2 : b > 1) : 
  (a + b > 3 ∧ a * b > 2) ∧ ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ (¬ (x > 2 ∧ y > 1)) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l898_89871


namespace NUMINAMATH_GPT_perpendicular_vectors_l898_89811

-- Define the vectors a and b.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (-2, x)

-- Define the dot product function.
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition that a is perpendicular to b.
def perp_condition (x : ℝ) : Prop :=
  dot_product vector_a (vector_b x) = 0

-- Main theorem stating that if a is perpendicular to b, then x = -1.
theorem perpendicular_vectors (x : ℝ) (h : perp_condition x) : x = -1 :=
by sorry

end NUMINAMATH_GPT_perpendicular_vectors_l898_89811


namespace NUMINAMATH_GPT_lynne_total_spent_l898_89817

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end NUMINAMATH_GPT_lynne_total_spent_l898_89817


namespace NUMINAMATH_GPT_ce_over_de_l898_89862

theorem ce_over_de {A B C D E T : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ (A →ₗ[ℝ] B)]
  {AT DT BT ET CE DE : ℝ}
  (h1 : AT / DT = 2)
  (h2 : BT / ET = 3) :
  CE / DE = 1 / 2 := 
sorry

end NUMINAMATH_GPT_ce_over_de_l898_89862


namespace NUMINAMATH_GPT_sugar_inventory_l898_89820

theorem sugar_inventory :
  ∀ (initial : ℕ) (day2_use : ℕ) (day2_borrow : ℕ) (day3_buy : ℕ) (day4_buy : ℕ) (day5_use : ℕ) (day5_return : ℕ),
  initial = 65 →
  day2_use = 18 →
  day2_borrow = 5 →
  day3_buy = 30 →
  day4_buy = 20 →
  day5_use = 10 →
  day5_return = 3 →
  initial - day2_use - day2_borrow + day3_buy + day4_buy - day5_use + day5_return = 85 :=
by
  intros initial day2_use day2_borrow day3_buy day4_buy day5_use day5_return
  intro h_initial
  intro h_day2_use
  intro h_day2_borrow
  intro h_day3_buy
  intro h_day4_buy
  intro h_day5_use
  intro h_day5_return
  subst h_initial
  subst h_day2_use
  subst h_day2_borrow
  subst h_day3_buy
  subst h_day4_buy
  subst h_day5_use
  subst h_day5_return
  sorry

end NUMINAMATH_GPT_sugar_inventory_l898_89820


namespace NUMINAMATH_GPT_car_sales_total_l898_89857

theorem car_sales_total (a b c : ℕ) (h1 : a = 14) (h2 : b = 16) (h3 : c = 27):
  a + b + c = 57 :=
by
  repeat {rwa [h1, h2, h3]}
  sorry

end NUMINAMATH_GPT_car_sales_total_l898_89857


namespace NUMINAMATH_GPT_rem_l898_89887

def rem' (x y : ℚ) : ℚ := x - y * (⌊ x / (2 * y) ⌋)

theorem rem'_value : rem' (5 / 9 : ℚ) (-3 / 7) = 62 / 63 := by
  sorry

end NUMINAMATH_GPT_rem_l898_89887


namespace NUMINAMATH_GPT_abs_inequality_range_l898_89859

theorem abs_inequality_range (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 := 
sorry

end NUMINAMATH_GPT_abs_inequality_range_l898_89859


namespace NUMINAMATH_GPT_lost_revenue_is_correct_l898_89874

-- Define the ticket prices
def general_admission_price : ℤ := 10
def children_price : ℤ := 6
def senior_price : ℤ := 8
def veteran_discount : ℤ := 2

-- Define the number of tickets sold
def general_tickets_sold : ℤ := 20
def children_tickets_sold : ℤ := 3
def senior_tickets_sold : ℤ := 4
def veteran_tickets_sold : ℤ := 2

-- Calculate the actual revenue from sold tickets
def actual_revenue := (general_tickets_sold * general_admission_price) + 
                      (children_tickets_sold * children_price) + 
                      (senior_tickets_sold * senior_price) + 
                      (veteran_tickets_sold * (general_admission_price - veteran_discount))

-- Define the maximum potential revenue assuming all tickets are sold at general admission price
def max_potential_revenue : ℤ := 50 * general_admission_price

-- Define the potential revenue lost
def potential_revenue_lost := max_potential_revenue - actual_revenue

-- The theorem to prove
theorem lost_revenue_is_correct : potential_revenue_lost = 234 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_lost_revenue_is_correct_l898_89874


namespace NUMINAMATH_GPT_minimum_turns_to_exceed_1000000_l898_89856

theorem minimum_turns_to_exceed_1000000 :
  let a : Fin 5 → ℕ := fun n => if n = 0 then 1 else 0
  (∀ n : ℕ, ∃ (b_2 b_3 b_4 b_5 : ℕ),
    a 4 + b_2 ≥ 0 ∧
    a 3 + b_3 ≥ 0 ∧
    a 2 + b_4 ≥ 0 ∧
    a 1 + b_5 ≥ 0 ∧
    b_2 * b_3 * b_4 * b_5 > 1000000 →
    b_2 + b_3 + b_4 + b_5 = n) → 
    ∃ n, n = 127 :=
by
  sorry

end NUMINAMATH_GPT_minimum_turns_to_exceed_1000000_l898_89856


namespace NUMINAMATH_GPT_fraction_numerator_l898_89804

theorem fraction_numerator (x : ℚ) :
  (∃ n : ℚ, 4 * n - 4 = x ∧ x / (4 * n - 4) = 3 / 7) → x = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_numerator_l898_89804


namespace NUMINAMATH_GPT_part1_simplified_part2_value_part3_independent_l898_89819

-- Definitions of A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Proof statement for part 1
theorem part1_simplified (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
by sorry

-- Proof statement for part 2
theorem part2_value (x y : ℝ) (hxy : x + y = 6/7) (hprod : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Proof statement for part 3
theorem part3_independent (y : ℝ) :
  2 * A (7/11) y - 3 * B (7/11) y = 49/11 :=
by sorry

end NUMINAMATH_GPT_part1_simplified_part2_value_part3_independent_l898_89819


namespace NUMINAMATH_GPT_ratio_blue_to_gold_l898_89855

-- Define the number of brown stripes
def brown_stripes : Nat := 4

-- Given condition: There are three times as many gold stripes as brown stripes
def gold_stripes : Nat := 3 * brown_stripes

-- Given condition: There are 60 blue stripes
def blue_stripes : Nat := 60

-- The actual statement to prove
theorem ratio_blue_to_gold : blue_stripes / gold_stripes = 5 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_ratio_blue_to_gold_l898_89855


namespace NUMINAMATH_GPT_max_popsicles_is_13_l898_89889

/-- Pablo's budgets and prices for buying popsicles. -/
structure PopsicleStore where
  single_popsicle_cost : ℕ
  three_popsicle_box_cost : ℕ
  five_popsicle_box_cost : ℕ
  starting_budget : ℕ

/-- The maximum number of popsicles Pablo can buy given the store's prices and his budget. -/
def maxPopsicles (store : PopsicleStore) : ℕ :=
  let num_five_popsicle_boxes := store.starting_budget / store.five_popsicle_box_cost
  let remaining_after_five_boxes := store.starting_budget % store.five_popsicle_box_cost
  let num_three_popsicle_boxes := remaining_after_five_boxes / store.three_popsicle_box_cost
  let remaining_after_three_boxes := remaining_after_five_boxes % store.three_popsicle_box_cost
  let num_single_popsicles := remaining_after_three_boxes / store.single_popsicle_cost
  num_five_popsicle_boxes * 5 + num_three_popsicle_boxes * 3 + num_single_popsicles

theorem max_popsicles_is_13 :
  maxPopsicles { single_popsicle_cost := 1, 
                 three_popsicle_box_cost := 2, 
                 five_popsicle_box_cost := 3, 
                 starting_budget := 8 } = 13 := by
  sorry

end NUMINAMATH_GPT_max_popsicles_is_13_l898_89889


namespace NUMINAMATH_GPT_square_of_radius_l898_89858

theorem square_of_radius 
  (AP PB CQ QD : ℝ) 
  (hAP : AP = 25)
  (hPB : PB = 35)
  (hCQ : CQ = 30)
  (hQD : QD = 40) 
  : ∃ r : ℝ, r^2 = 13325 := 
sorry

end NUMINAMATH_GPT_square_of_radius_l898_89858


namespace NUMINAMATH_GPT_binary_arithmetic_l898_89849

theorem binary_arithmetic :
    let a := 0b1011101
    let b := 0b1101
    let c := 0b101010
    let d := 0b110
    ((a + b) * c) / d = 0b1110111100 :=
by
  sorry

end NUMINAMATH_GPT_binary_arithmetic_l898_89849


namespace NUMINAMATH_GPT_distinct_license_plates_l898_89843

theorem distinct_license_plates :
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  total = 122504000 :=
by
  -- Definitions from the conditions
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  -- Calculation
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  -- Assertion
  have h : total = 122504000 := sorry
  exact h

end NUMINAMATH_GPT_distinct_license_plates_l898_89843


namespace NUMINAMATH_GPT_range_of_b2_plus_c2_l898_89814

theorem range_of_b2_plus_c2 (A B C : ℝ) (a b c : ℝ) 
  (h1 : (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C)
  (ha : a = Real.sqrt 3)
  (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) :
  (∃ x, 5 < x ∧ x ≤ 6 ∧ x = b^2 + c^2) :=
sorry

end NUMINAMATH_GPT_range_of_b2_plus_c2_l898_89814


namespace NUMINAMATH_GPT_days_A_worked_l898_89861

theorem days_A_worked (W : ℝ) (x : ℝ) (hA : W / 15 * x = W - 6 * (W / 9))
  (hB : W = 6 * (W / 9)) : x = 5 :=
sorry

end NUMINAMATH_GPT_days_A_worked_l898_89861


namespace NUMINAMATH_GPT_total_eggs_sold_l898_89899

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end NUMINAMATH_GPT_total_eggs_sold_l898_89899


namespace NUMINAMATH_GPT_second_oldest_brother_age_l898_89838

theorem second_oldest_brother_age
  (y s o : ℕ)
  (h1 : y + s + o = 34)
  (h2 : o = 3 * y)
  (h3 : s = 2 * y - 2) :
  s = 10 := by
  sorry

end NUMINAMATH_GPT_second_oldest_brother_age_l898_89838


namespace NUMINAMATH_GPT_length_of_rectangular_plot_l898_89893

variable (L : ℕ)

-- Given conditions
def width := 50
def poles := 14
def distance_between_poles := 20
def intervals := poles - 1
def perimeter := intervals * distance_between_poles

-- The perimeter of the rectangle in terms of length and width
def rectangle_perimeter := 2 * (L + width)

-- The main statement to be proven
theorem length_of_rectangular_plot :
  rectangle_perimeter L = perimeter → L = 80 :=
by
  sorry

end NUMINAMATH_GPT_length_of_rectangular_plot_l898_89893


namespace NUMINAMATH_GPT_multiply_by_nine_l898_89867

theorem multiply_by_nine (x : ℝ) (h : 9 * x = 36) : x = 4 :=
sorry

end NUMINAMATH_GPT_multiply_by_nine_l898_89867


namespace NUMINAMATH_GPT_unique_involution_l898_89815

noncomputable def f (x : ℤ) : ℤ := sorry

theorem unique_involution (f : ℤ → ℤ) :
  (∀ x : ℤ, f (f x) = x) →
  (∀ x y : ℤ, (x + y) % 2 = 1 → f x + f y ≥ x + y) →
  (∀ x : ℤ, f x = x) :=
sorry

end NUMINAMATH_GPT_unique_involution_l898_89815


namespace NUMINAMATH_GPT_machine_performance_l898_89848

noncomputable def machine_A_data : List ℕ :=
  [4, 1, 0, 2, 2, 1, 3, 1, 2, 4]

noncomputable def machine_B_data : List ℕ :=
  [2, 3, 1, 1, 3, 2, 2, 1, 2, 3]

noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

noncomputable def variance (data : List ℕ) (mean : ℝ) : ℝ :=
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

theorem machine_performance :
  let mean_A := mean machine_A_data
  let mean_B := mean machine_B_data
  let variance_A := variance machine_A_data mean_A
  let variance_B := variance machine_B_data mean_B
  mean_A = 2 ∧ mean_B = 2 ∧ variance_A = 1.6 ∧ variance_B = 0.6 ∧ variance_B < variance_A := 
sorry

end NUMINAMATH_GPT_machine_performance_l898_89848


namespace NUMINAMATH_GPT_louisa_average_speed_l898_89897

-- Problem statement
theorem louisa_average_speed :
  ∃ v : ℝ, (250 / v * v = 250 ∧ 350 / v * v = 350) ∧ ((350 / v) = (250 / v) + 3) ∧ v = 100 / 3 := by
  sorry

end NUMINAMATH_GPT_louisa_average_speed_l898_89897
