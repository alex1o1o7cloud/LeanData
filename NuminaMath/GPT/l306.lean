import Mathlib

namespace NUMINAMATH_GPT_birgit_numbers_sum_l306_30606

theorem birgit_numbers_sum (a b c d : ℕ) 
  (h1 : a + b + c = 415) 
  (h2 : a + b + d = 442) 
  (h3 : a + c + d = 396) 
  (h4 : b + c + d = 325) : 
  a + b + c + d = 526 :=
by
  sorry

end NUMINAMATH_GPT_birgit_numbers_sum_l306_30606


namespace NUMINAMATH_GPT_polynomial_roots_power_sum_l306_30698

theorem polynomial_roots_power_sum {a b c : ℝ}
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 6)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_power_sum_l306_30698


namespace NUMINAMATH_GPT_find_g_of_3_l306_30667

noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for the function g

theorem find_g_of_3 (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 3 = 26 / 7 :=
by sorry

end NUMINAMATH_GPT_find_g_of_3_l306_30667


namespace NUMINAMATH_GPT_intersection_A_B_l306_30639

def A : Set ℝ := { x | abs x ≤ 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l306_30639


namespace NUMINAMATH_GPT_find_y_l306_30616

theorem find_y (x y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end NUMINAMATH_GPT_find_y_l306_30616


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l306_30676

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 4) 
  (h2 : a 4 + a 7 = 15) : 
  ∃ d : ℝ, ∀ n : ℕ, a n = n + 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l306_30676


namespace NUMINAMATH_GPT_jordyn_total_payment_l306_30600

theorem jordyn_total_payment :
  let price_cherries := 5
  let price_olives := 7
  let price_grapes := 11
  let num_cherries := 50
  let num_olives := 75
  let num_grapes := 25
  let discount_cherries := 0.12
  let discount_olives := 0.08
  let discount_grapes := 0.15
  let sales_tax := 0.05
  let service_charge := 0.02
  let total_cherries := num_cherries * price_cherries
  let total_olives := num_olives * price_olives
  let total_grapes := num_grapes * price_grapes
  let discounted_cherries := total_cherries * (1 - discount_cherries)
  let discounted_olives := total_olives * (1 - discount_olives)
  let discounted_grapes := total_grapes * (1 - discount_grapes)
  let subtotal := discounted_cherries + discounted_olives + discounted_grapes
  let taxed_amount := subtotal * (1 + sales_tax)
  let final_amount := taxed_amount * (1 + service_charge)
  final_amount = 1002.32 :=
by
  sorry

end NUMINAMATH_GPT_jordyn_total_payment_l306_30600


namespace NUMINAMATH_GPT_linear_function_increasing_l306_30665

variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)
variable (hx : x1 < x2)
variable (P1_eq : y1 = 2 * x1 + 1)
variable (P2_eq : y2 = 2 * x2 + 1)

theorem linear_function_increasing (hx : x1 < x2) (P1_eq : y1 = 2 * x1 + 1) (P2_eq : y2 = 2 * x2 + 1) 
    : y1 < y2 := sorry

end NUMINAMATH_GPT_linear_function_increasing_l306_30665


namespace NUMINAMATH_GPT_exist_equal_success_rate_l306_30694

noncomputable def S : ℕ → ℝ := sorry -- Definition of S(N), the number of successful free throws

theorem exist_equal_success_rate (N1 N2 : ℕ) 
  (h1 : S N1 < 0.8 * N1) 
  (h2 : S N2 > 0.8 * N2) : 
  ∃ (N : ℕ), N1 ≤ N ∧ N ≤ N2 ∧ S N = 0.8 * N :=
sorry

end NUMINAMATH_GPT_exist_equal_success_rate_l306_30694


namespace NUMINAMATH_GPT_equivalent_systems_solution_and_value_l306_30644

-- Definitions for the conditions
def system1 (x y a b : ℝ) : Prop := 
  (2 * (x + 1) - y = 7) ∧ (x + b * y = a)

def system2 (x y a b : ℝ) : Prop := 
  (a * x + y = b) ∧ (3 * x + 2 * (y - 1) = 9)

-- The proof problem as a Lean 4 statement
theorem equivalent_systems_solution_and_value (a b : ℝ) :
  (∃ x y : ℝ, system1 x y a b ∧ system2 x y a b) →
  ((∃ x y : ℝ, x = 3 ∧ y = 1) ∧ (3 * a - b) ^ 2023 = -1) :=
  by sorry

end NUMINAMATH_GPT_equivalent_systems_solution_and_value_l306_30644


namespace NUMINAMATH_GPT_moles_of_KOH_used_l306_30679

variable {n_KOH : ℝ}

theorem moles_of_KOH_used :
  ∃ n_KOH, (NH4I + KOH = KI_produced) → (KI_produced = 1) → n_KOH = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_KOH_used_l306_30679


namespace NUMINAMATH_GPT_speed_ratio_l306_30601

theorem speed_ratio (L tA tB : ℝ) (R : ℝ) (h1: A_speed = R * B_speed) 
  (h2: head_start = 0.35 * L) (h3: finish_margin = 0.25 * L)
  (h4: A_distance = L + head_start) (h5: B_distance = L)
  (h6: A_finish = A_distance / A_speed)
  (h7: B_finish = B_distance / B_speed)
  (h8: B_finish_time = A_finish + finish_margin / B_speed)
  : R = 1.08 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l306_30601


namespace NUMINAMATH_GPT_candidate_percentage_valid_votes_l306_30642

theorem candidate_percentage_valid_votes (total_votes invalid_percentage valid_votes_received : ℕ) 
    (h_total_votes : total_votes = 560000)
    (h_invalid_percentage : invalid_percentage = 15)
    (h_valid_votes_received : valid_votes_received = 333200) :
    (valid_votes_received : ℚ) / (total_votes * (1 - invalid_percentage / 100) : ℚ) * 100 = 70 :=
by
  sorry

end NUMINAMATH_GPT_candidate_percentage_valid_votes_l306_30642


namespace NUMINAMATH_GPT_john_moves_correct_total_weight_l306_30658

noncomputable def johns_total_weight_moved : ℝ := 5626.398

theorem john_moves_correct_total_weight :
  let initial_back_squat : ℝ := 200
  let back_squat_increase : ℝ := 50
  let front_squat_ratio : ℝ := 0.8
  let back_squat_set_increase : ℝ := 0.05
  let front_squat_ratio_increase : ℝ := 0.04
  let front_squat_effort : ℝ := 0.9
  let deadlift_ratio : ℝ := 1.2
  let deadlift_effort : ℝ := 0.85
  let deadlift_set_increase : ℝ := 0.03
  let updated_back_squat := (initial_back_squat + back_squat_increase)
  let back_squat_set_1 := updated_back_squat
  let back_squat_set_2 := back_squat_set_1 * (1 + back_squat_set_increase)
  let back_squat_set_3 := back_squat_set_2 * (1 + back_squat_set_increase)
  let back_squat_total := 3 * (back_squat_set_1 + back_squat_set_2 + back_squat_set_3)
  let updated_front_squat := updated_back_squat * front_squat_ratio
  let front_squat_set_1 := updated_front_squat * front_squat_effort
  let front_squat_set_2 := (updated_front_squat * (1 + front_squat_ratio_increase)) * front_squat_effort
  let front_squat_set_3 := (updated_front_squat * (1 + 2 * front_squat_ratio_increase)) * front_squat_effort
  let front_squat_total := 3 * (front_squat_set_1 + front_squat_set_2 + front_squat_set_3)
  let updated_deadlift := updated_back_squat * deadlift_ratio
  let deadlift_set_1 := updated_deadlift * deadlift_effort
  let deadlift_set_2 := (updated_deadlift * (1 + deadlift_set_increase)) * deadlift_effort
  let deadlift_set_3 := (updated_deadlift * (1 + 2 * deadlift_set_increase)) * deadlift_effort
  let deadlift_total := 2 * (deadlift_set_1 + deadlift_set_2 + deadlift_set_3)
  (back_squat_total + front_squat_total + deadlift_total) = johns_total_weight_moved :=
by sorry

end NUMINAMATH_GPT_john_moves_correct_total_weight_l306_30658


namespace NUMINAMATH_GPT_stock_percentage_calculation_l306_30640

noncomputable def stock_percentage (investment_amount stock_price annual_income : ℝ) : ℝ :=
  (annual_income / (investment_amount / stock_price) / stock_price) * 100

theorem stock_percentage_calculation :
  stock_percentage 6800 136 1000 = 14.71 :=
by
  sorry

end NUMINAMATH_GPT_stock_percentage_calculation_l306_30640


namespace NUMINAMATH_GPT_printers_ratio_l306_30629

theorem printers_ratio (Rate_X : ℝ := 1 / 16) (Rate_Y : ℝ := 1 / 10) (Rate_Z : ℝ := 1 / 20) :
  let Time_X := 1 / Rate_X
  let Time_YZ := 1 / (Rate_Y + Rate_Z)
  (Time_X / Time_YZ) = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_printers_ratio_l306_30629


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l306_30692

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l306_30692


namespace NUMINAMATH_GPT_total_students_l306_30699

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l306_30699


namespace NUMINAMATH_GPT_symmetric_point_in_third_quadrant_l306_30615

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to find the symmetric point about the y-axis
def symmetric_about_y (P : Point) : Point :=
  Point.mk (-P.x) P.y

-- Define the original point P
def P : Point := { x := 3, y := -2 }

-- Define the symmetric point P' about the y-axis
def P' : Point := symmetric_about_y P

-- Define a condition to determine if a point is in the third quadrant
def is_in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- The theorem stating that the symmetric point of P about the y-axis is in the third quadrant
theorem symmetric_point_in_third_quadrant : is_in_third_quadrant P' :=
  by
  sorry

end NUMINAMATH_GPT_symmetric_point_in_third_quadrant_l306_30615


namespace NUMINAMATH_GPT_immigration_per_year_l306_30654

-- Definitions based on the initial conditions
def initial_population : ℕ := 100000
def birth_rate : ℕ := 60 -- this represents 60%
def duration_years : ℕ := 10
def emigration_per_year : ℕ := 2000
def final_population : ℕ := 165000

-- Theorem statement: The number of people that immigrated per year
theorem immigration_per_year (immigration_per_year : ℕ) :
  immigration_per_year = 2500 :=
  sorry

end NUMINAMATH_GPT_immigration_per_year_l306_30654


namespace NUMINAMATH_GPT_max_students_distribution_l306_30681

-- Define the four quantities
def pens : ℕ := 4261
def pencils : ℕ := 2677
def erasers : ℕ := 1759
def notebooks : ℕ := 1423

-- Prove that the greatest common divisor (GCD) of these four quantities is 1
theorem max_students_distribution : Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_students_distribution_l306_30681


namespace NUMINAMATH_GPT_total_students_multiple_of_8_l306_30686

theorem total_students_multiple_of_8 (B G T : ℕ) (h : G = 7 * B) (ht : T = B + G) : T % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_total_students_multiple_of_8_l306_30686


namespace NUMINAMATH_GPT_rope_folded_three_times_parts_l306_30632

theorem rope_folded_three_times_parts (total_length : ℕ) :
  ∀ parts : ℕ, parts = (total_length / 8) →
  ∀ n : ℕ, n = 3 →
  (∀ length_each_part : ℚ, length_each_part = 1 / (2 ^ n) →
  length_each_part = 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_rope_folded_three_times_parts_l306_30632


namespace NUMINAMATH_GPT_greatest_integer_gcd_3_l306_30657

theorem greatest_integer_gcd_3 : ∃ n, n < 100 ∧ gcd n 18 = 3 ∧ ∀ m, m < 100 ∧ gcd m 18 = 3 → m ≤ n := by
  sorry

end NUMINAMATH_GPT_greatest_integer_gcd_3_l306_30657


namespace NUMINAMATH_GPT_parabola_point_coordinates_l306_30660

theorem parabola_point_coordinates (x y : ℝ) (h_parabola : y^2 = 8 * x) 
    (h_distance_focus : (x + 2)^2 + y^2 = 81) : 
    (x = 7 ∧ y = 2 * Real.sqrt 14) ∨ (x = 7 ∧ y = -2 * Real.sqrt 14) :=
by {
  -- Proof will be inserted here
  sorry
}

end NUMINAMATH_GPT_parabola_point_coordinates_l306_30660


namespace NUMINAMATH_GPT_problem1_problem2a_problem2b_problem3_l306_30664

noncomputable def f (a x : ℝ) := -x^2 + a * x - 2
noncomputable def g (x : ℝ) := x * Real.log x

-- Problem 1
theorem problem1 {a : ℝ} : (∀ x : ℝ, 0 < x → g x ≥ f a x) → a ≤ 3 :=
sorry

-- Problem 2 
theorem problem2a (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1 / Real.exp 1) :
  ∃ xmin : ℝ, g (1 / Real.exp 1) = -1 / Real.exp 1 ∧ 
  ∃ xmax : ℝ, g (m + 1) = (m + 1) * Real.log (m + 1) :=
sorry

theorem problem2b (m : ℝ) (h₀ : 1 / Real.exp 1 ≤ m) :
  ∃ xmin ymax : ℝ, xmin = g m ∧ ymax = g (m + 1) :=
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : 0 < x) : 
  Real.log x + (2 / (Real.exp 1 * x)) ≥ 1 / Real.exp x :=
sorry

end NUMINAMATH_GPT_problem1_problem2a_problem2b_problem3_l306_30664


namespace NUMINAMATH_GPT_initial_number_of_macaroons_l306_30645

theorem initial_number_of_macaroons 
  (w : ℕ) (bag_count : ℕ) (eaten_bag_count : ℕ) (remaining_weight : ℕ) 
  (macaroon_weight : ℕ) (remaining_bags : ℕ) (initial_macaroons : ℕ) :
  w = 5 → bag_count = 4 → eaten_bag_count = 1 → remaining_weight = 45 → 
  macaroon_weight = w → remaining_bags = (bag_count - eaten_bag_count) → 
  initial_macaroons = (remaining_bags * remaining_weight / macaroon_weight) * bag_count / remaining_bags →
  initial_macaroons = 12 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_initial_number_of_macaroons_l306_30645


namespace NUMINAMATH_GPT_sum_of_three_rel_prime_pos_integers_l306_30649

theorem sum_of_three_rel_prime_pos_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_rel_prime_ab : Nat.gcd a b = 1) (h_rel_prime_ac : Nat.gcd a c = 1) (h_rel_prime_bc : Nat.gcd b c = 1)
  (h_product : a * b * c = 2700) :
  a + b + c = 56 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_rel_prime_pos_integers_l306_30649


namespace NUMINAMATH_GPT_solution_l306_30611

noncomputable def problem (x : ℕ) : Prop :=
  2 ^ 28 = 4 ^ x  -- Simplified form of the condition given

theorem solution : problem 14 :=
by
  sorry

end NUMINAMATH_GPT_solution_l306_30611


namespace NUMINAMATH_GPT_find_number_of_piles_l306_30655

theorem find_number_of_piles 
  (Q : ℕ) 
  (h1 : Q = Q) 
  (h2 : ∀ (piles : ℕ), piles = 3) 
  (total_coins : ℕ) 
  (h3 : total_coins = 30) 
  (e : 6 * Q = total_coins) :
  Q = 5 := 
by sorry

end NUMINAMATH_GPT_find_number_of_piles_l306_30655


namespace NUMINAMATH_GPT_excluded_avg_mark_l306_30670

theorem excluded_avg_mark (N A A_remaining excluded_count : ℕ)
  (hN : N = 15)
  (hA : A = 80)
  (hA_remaining : A_remaining = 90) 
  (h_excluded : excluded_count = 5) :
  (A * N - A_remaining * (N - excluded_count)) / excluded_count = 60 := sorry

end NUMINAMATH_GPT_excluded_avg_mark_l306_30670


namespace NUMINAMATH_GPT_reeyas_first_subject_score_l306_30653

theorem reeyas_first_subject_score
  (second_subject_score third_subject_score fourth_subject_score : ℕ)
  (num_subjects : ℕ)
  (average_score : ℕ)
  (total_subjects_score : ℕ)
  (condition1 : second_subject_score = 76)
  (condition2 : third_subject_score = 82)
  (condition3 : fourth_subject_score = 85)
  (condition4 : num_subjects = 4)
  (condition5 : average_score = 75)
  (condition6 : total_subjects_score = num_subjects * average_score) :
  67 = total_subjects_score - (second_subject_score + third_subject_score + fourth_subject_score) := 
  sorry

end NUMINAMATH_GPT_reeyas_first_subject_score_l306_30653


namespace NUMINAMATH_GPT_find_x_l306_30684

theorem find_x 
  (x y : ℤ) 
  (h1 : 2 * x - y = 5) 
  (h2 : x + 2 * y = 5) : 
  x = 3 := 
sorry

end NUMINAMATH_GPT_find_x_l306_30684


namespace NUMINAMATH_GPT_tin_silver_ratio_l306_30689

theorem tin_silver_ratio (T S : ℝ) 
  (h1 : T + S = 50) 
  (h2 : 0.1375 * T + 0.075 * S = 5) : 
  T / S = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tin_silver_ratio_l306_30689


namespace NUMINAMATH_GPT_find_x_from_equation_l306_30688

/-- If (1 / 8) * 2^36 = 4^x, then x = 16.5 -/
theorem find_x_from_equation (x : ℝ) (h : (1/8) * (2:ℝ)^36 = (4:ℝ)^x) : x = 16.5 :=
by sorry

end NUMINAMATH_GPT_find_x_from_equation_l306_30688


namespace NUMINAMATH_GPT_fish_too_small_l306_30633

theorem fish_too_small
    (ben_fish : ℕ) (judy_fish : ℕ) (billy_fish : ℕ) (jim_fish : ℕ) (susie_fish : ℕ)
    (total_filets : ℕ) (filets_per_fish : ℕ) :
    ben_fish = 4 →
    judy_fish = 1 →
    billy_fish = 3 →
    jim_fish = 2 →
    susie_fish = 5 →
    total_filets = 24 →
    filets_per_fish = 2 →
    (ben_fish + judy_fish + billy_fish + jim_fish + susie_fish) - (total_filets / filets_per_fish) = 3 := 
by 
  intros
  sorry

end NUMINAMATH_GPT_fish_too_small_l306_30633


namespace NUMINAMATH_GPT_remainder_sum_div_6_l306_30677

theorem remainder_sum_div_6 (n : ℤ) : ((5 - n) + (n + 4)) % 6 = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_remainder_sum_div_6_l306_30677


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l306_30628

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y ^ 3 - 8 * y ^ 2 + 9 * y + 2 = 0 → y ≥ 0) →
  let s : ℝ := 8
  let p : ℝ := 9
  let q : ℝ := -2
  (s ^ 2 - 2 * p = 46) :=
by
  -- Placeholders for definitions extracted from the conditions
  -- and additional necessary let-bindings from Vieta's formulas
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l306_30628


namespace NUMINAMATH_GPT_heartsuit_ratio_l306_30668

-- Define the operation \heartsuit
def heartsuit (n m : ℕ) : ℕ := n^3 * m^2

-- The proposition we want to prove
theorem heartsuit_ratio :
  heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_heartsuit_ratio_l306_30668


namespace NUMINAMATH_GPT_equal_play_time_for_students_l306_30625

theorem equal_play_time_for_students 
  (total_students : ℕ) 
  (start_time end_time : ℕ) 
  (tables : ℕ) 
  (playing_students refereeing_students : ℕ) 
  (time_played : ℕ) :
  total_students = 6 →
  start_time = 8 * 60 →
  end_time = 11 * 60 + 30 →
  tables = 2 →
  playing_students = 4 →
  refereeing_students = 2 →
  time_played = (end_time - start_time) * tables / (total_students / refereeing_students) →
  time_played = 140 :=
by
  sorry

end NUMINAMATH_GPT_equal_play_time_for_students_l306_30625


namespace NUMINAMATH_GPT_initial_money_l306_30612

theorem initial_money {M : ℝ} (h : (M - 10) - (M - 10) / 4 = 15) : M = 30 :=
sorry

end NUMINAMATH_GPT_initial_money_l306_30612


namespace NUMINAMATH_GPT_triangle_problem_l306_30641

/-- In triangle ABC, the sides opposite to angles A, B, C are a, b, c respectively.
Given that b = sqrt 2, c = 3, B + C = 3A, prove:
1. The length of side a equals sqrt 5.
2. sin (B + 3π/4) equals sqrt(10) / 10.
-/
theorem triangle_problem 
  (a b c A B C : ℝ)
  (hb : b = Real.sqrt 2)
  (hc : c = 3)
  (hBC : B + C = 3 * A)
  (hA : A = π / 4)
  : (a = Real.sqrt 5)
  ∧ (Real.sin (B + 3 * π / 4) = Real.sqrt 10 / 10) :=
sorry

end NUMINAMATH_GPT_triangle_problem_l306_30641


namespace NUMINAMATH_GPT_negation_of_p_l306_30661
open Classical

variable (n : ℕ)

def p : Prop := ∀ n : ℕ, n^2 < 2^n

theorem negation_of_p : ¬ p ↔ ∃ n₀ : ℕ, n₀^2 ≥ 2^n₀ := 
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l306_30661


namespace NUMINAMATH_GPT_solution_inequality_part1_solution_inequality_part2_l306_30650

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem solution_inequality_part1 (x : ℝ) :
  (f x + x^2 - 4 > 0) ↔ (x > 2 ∨ x < -1) :=
sorry

theorem solution_inequality_part2 (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → (m > 3) :=
sorry

end NUMINAMATH_GPT_solution_inequality_part1_solution_inequality_part2_l306_30650


namespace NUMINAMATH_GPT_goose_eggs_count_l306_30631

theorem goose_eggs_count (E : ℕ)
  (hatch_ratio : ℚ := 2 / 3)
  (survive_first_month_ratio : ℚ := 3 / 4)
  (survive_first_year_ratio : ℚ := 2 / 5)
  (survived_first_year : ℕ := 130) :
  (survive_first_year_ratio * survive_first_month_ratio * hatch_ratio * (E : ℚ) = survived_first_year) →
  E = 1300 := by
  sorry

end NUMINAMATH_GPT_goose_eggs_count_l306_30631


namespace NUMINAMATH_GPT_parallel_lines_l306_30693

theorem parallel_lines (k1 k2 l1 l2 : ℝ) :
  (∀ x, (k1 ≠ k2) -> (k1 * x + l1 ≠ k2 * x + l2)) ↔ 
  (k1 = k2 ∧ l1 ≠ l2) := 
by sorry

end NUMINAMATH_GPT_parallel_lines_l306_30693


namespace NUMINAMATH_GPT_triangle_inscribed_and_arcs_l306_30602

theorem triangle_inscribed_and_arcs
  (PQ QR PR : ℝ) (X Y Z : ℝ)
  (QY XZ QX YZ PX RY : ℝ)
  (H1 : PQ = 26)
  (H2 : QR = 28) 
  (H3 : PR = 27)
  (H4 : QY = XZ)
  (H5 : QX = YZ)
  (H6 : PX = RY)
  (H7 : RY = PX + 1)
  (H8 : XZ = QX + 1)
  (H9 : QY = YZ + 2) :
  QX = 29 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inscribed_and_arcs_l306_30602


namespace NUMINAMATH_GPT_sequence_a_is_perfect_square_l306_30678

theorem sequence_a_is_perfect_square :
  ∃ (a b : ℕ → ℤ),
    a 0 = 1 ∧ 
    b 0 = 0 ∧ 
    (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    ∀ n, ∃ m : ℕ, a n = m * m := sorry

end NUMINAMATH_GPT_sequence_a_is_perfect_square_l306_30678


namespace NUMINAMATH_GPT_roots_are_prime_then_a_is_five_l306_30697

theorem roots_are_prime_then_a_is_five (x1 x2 a : ℕ) (h_prime_x1 : Prime x1) (h_prime_x2 : Prime x2)
  (h_eq : x1 + x2 = a) (h_eq_mul : x1 * x2 = a + 1) : a = 5 :=
sorry

end NUMINAMATH_GPT_roots_are_prime_then_a_is_five_l306_30697


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_value_l306_30636

theorem arithmetic_sequence_a2_value :
  ∃ (a : ℕ) (d : ℕ), (a = 3) ∧ (a + d + (a + 2 * d) = 12) ∧ (a + d = 5) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_value_l306_30636


namespace NUMINAMATH_GPT_sum_a_m_eq_2_pow_n_b_n_l306_30669

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ k => x ^ k)

noncomputable def b_n (x : ℝ) (n : ℕ) : ℝ := 
  (Finset.range (n + 1)).sum (λ k => ((x + 1) / 2) ^ k)

theorem sum_a_m_eq_2_pow_n_b_n 
  (x : ℝ) (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ m => a_n x m * Nat.choose (n + 1) (m + 1)) = 2 ^ n * b_n x n :=
by
  sorry

end NUMINAMATH_GPT_sum_a_m_eq_2_pow_n_b_n_l306_30669


namespace NUMINAMATH_GPT_smallest_multiple_3_4_5_l306_30675

theorem smallest_multiple_3_4_5 : ∃ (n : ℕ), (∀ (m : ℕ), (m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0) → n ≤ m) ∧ n = 60 := 
sorry

end NUMINAMATH_GPT_smallest_multiple_3_4_5_l306_30675


namespace NUMINAMATH_GPT_melons_count_l306_30605

theorem melons_count (w_apples_total w_apple w_2apples w_watermelons w_total w_melons : ℕ) :
  w_apples_total = 4500 →
  9 * w_apple = w_apples_total →
  2 * w_apple = w_2apples →
  5 * 1050 = w_watermelons →
  w_total = w_2apples + w_melons →
  w_total = w_watermelons →
  w_melons / 850 = 5 :=
by
  sorry

end NUMINAMATH_GPT_melons_count_l306_30605


namespace NUMINAMATH_GPT_find_large_number_l306_30637

theorem find_large_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 :=
sorry

end NUMINAMATH_GPT_find_large_number_l306_30637


namespace NUMINAMATH_GPT_distance_from_LV_to_LA_is_273_l306_30682

-- Define the conditions
def distance_SLC_to_LV : ℝ := 420
def total_time : ℝ := 11
def avg_speed : ℝ := 63

-- Define the total distance covered given the average speed and time
def total_distance : ℝ := avg_speed * total_time

-- Define the distance from Las Vegas to Los Angeles
def distance_LV_to_LA : ℝ := total_distance - distance_SLC_to_LV

-- Now state the theorem we want to prove
theorem distance_from_LV_to_LA_is_273 :
  distance_LV_to_LA = 273 :=
sorry

end NUMINAMATH_GPT_distance_from_LV_to_LA_is_273_l306_30682


namespace NUMINAMATH_GPT_find_dividend_l306_30638

theorem find_dividend (dividend divisor quotient : ℕ) 
  (h_sum : dividend + divisor + quotient = 103)
  (h_quotient : quotient = 3)
  (h_divisor : divisor = dividend / quotient) : 
  dividend = 75 :=
by
  rw [h_quotient, h_divisor] at h_sum
  sorry

end NUMINAMATH_GPT_find_dividend_l306_30638


namespace NUMINAMATH_GPT_distinct_triangle_not_isosceles_l306_30696

theorem distinct_triangle_not_isosceles (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  ¬(a = b ∨ b = c ∨ c = a) :=
by {
  sorry
}

end NUMINAMATH_GPT_distinct_triangle_not_isosceles_l306_30696


namespace NUMINAMATH_GPT_points_on_opposite_sides_of_line_l306_30607

theorem points_on_opposite_sides_of_line (a : ℝ) :
  let A := (3, 1)
  let B := (-4, 6)
  (3 * A.1 - 2 * A.2 + a) * (3 * B.1 - 2 * B.2 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  let A := (3, 1)
  let B := (-4, 6)
  have hA : 3 * A.1 - 2 * A.2 + a = 7 + a := by sorry
  have hB : 3 * B.1 - 2 * B.2 + a = -24 + a := by sorry
  exact sorry

end NUMINAMATH_GPT_points_on_opposite_sides_of_line_l306_30607


namespace NUMINAMATH_GPT_find_a_l306_30609

theorem find_a (a : ℝ) (h : ∀ x y : ℝ, ax + y - 4 = 0 → x + (a + 3/2) * y + 2 = 0 → True) : a = 1/2 :=
sorry

end NUMINAMATH_GPT_find_a_l306_30609


namespace NUMINAMATH_GPT_min_inequality_l306_30648

theorem min_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 2) :
  ∃ L, L = 9 / 4 ∧ (1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ L) :=
sorry

end NUMINAMATH_GPT_min_inequality_l306_30648


namespace NUMINAMATH_GPT_company_employee_percentage_l306_30622

theorem company_employee_percentage (M : ℝ)
  (h1 : 0.20 * M + 0.40 * (1 - M) = 0.31000000000000007) :
  M = 0.45 :=
sorry

end NUMINAMATH_GPT_company_employee_percentage_l306_30622


namespace NUMINAMATH_GPT_total_share_amount_l306_30652

theorem total_share_amount (x y z : ℝ) (hx : y = 0.45 * x) (hz : z = 0.30 * x) (hy_share : y = 63) : x + y + z = 245 := by
  sorry

end NUMINAMATH_GPT_total_share_amount_l306_30652


namespace NUMINAMATH_GPT_rows_in_initial_patios_l306_30624

theorem rows_in_initial_patios (r c : ℕ) (h1 : r * c = 60) (h2 : (2 * c : ℚ) / r = 3 / 2) (h3 : (r + 5) * (c - 3) = 60) : r = 10 :=
sorry

end NUMINAMATH_GPT_rows_in_initial_patios_l306_30624


namespace NUMINAMATH_GPT_radius_circle_D_eq_five_l306_30634

-- Definitions for circles with given radii and tangency conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def circle_C : Circle := ⟨(0, 0), 5⟩
noncomputable def circle_D (rD : ℝ) : Circle := ⟨(4 * rD, 0), 4 * rD⟩
noncomputable def circle_E (rE : ℝ) : Circle := ⟨(5 - rE, rE * 5), rE⟩

-- Prove that the radius of circle D is 5
theorem radius_circle_D_eq_five (rE : ℝ) (rD : ℝ) : circle_D rE = circle_C → rD = 5 := by
  sorry

end NUMINAMATH_GPT_radius_circle_D_eq_five_l306_30634


namespace NUMINAMATH_GPT_pollution_control_l306_30621

theorem pollution_control (x y : ℕ) (h1 : x - y = 5) (h2 : 2 * x + 3 * y = 45) : x = 12 ∧ y = 7 :=
by
  sorry

end NUMINAMATH_GPT_pollution_control_l306_30621


namespace NUMINAMATH_GPT_remaining_garden_space_l306_30666

theorem remaining_garden_space : 
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  Area_rectangle - Area_square_cutout + Area_triangle = 347 :=
by
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  show Area_rectangle - Area_square_cutout + Area_triangle = 347
  sorry

end NUMINAMATH_GPT_remaining_garden_space_l306_30666


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l306_30663

theorem isosceles_triangle_perimeter (m : ℝ) (a b : ℝ) 
  (h1 : 3 = a ∨ 3 = b)
  (h2 : a ≠ b)
  (h3 : a^2 - (m+1)*a + 2*m = 0)
  (h4 : b^2 - (m+1)*b + 2*m = 0) :
  (a + b + a = 11) ∨ (a + a + b = 10) := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l306_30663


namespace NUMINAMATH_GPT_three_times_sum_first_35_odd_l306_30623

/-- 
The sum of the first n odd numbers --/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

/-- Given that 69 is the 35th odd number --/
theorem three_times_sum_first_35_odd : 3 * sum_first_n_odd 35 = 3675 := by
  sorry

end NUMINAMATH_GPT_three_times_sum_first_35_odd_l306_30623


namespace NUMINAMATH_GPT_find_t_l306_30672

-- Defining variables and assumptions
variables (V V0 g S t : Real)
variable (h1 : V = g * t + V0)
variable (h2 : S = (1 / 2) * g * t^2 + V0 * t)

-- The goal: to prove t equals 2S / (V + V0)
theorem find_t (V V0 g S t : Real) (h1 : V = g * t + V0) (h2 : S = (1 / 2) * g * t^2 + V0 * t):
  t = 2 * S / (V + V0) := by
  sorry

end NUMINAMATH_GPT_find_t_l306_30672


namespace NUMINAMATH_GPT_total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l306_30643

-- Define the conditions
def number_of_bags : ℕ := 9
def vitamins_per_bag : ℚ := 0.2

-- Define the total vitamins in the box
def total_vitamins_in_box : ℚ := number_of_bags * vitamins_per_bag

-- Define the vitamins intake by drinking half a bag
def vitamins_per_half_bag : ℚ := vitamins_per_bag / 2

-- Prove that the total grams of vitamins in the box is 1.8 grams
theorem total_vitamins_in_box_correct : total_vitamins_in_box = 1.8 := by
  sorry

-- Prove that the vitamins intake by drinking half a bag is 0.1 grams
theorem vitamins_per_half_bag_correct : vitamins_per_half_bag = 0.1 := by
  sorry

end NUMINAMATH_GPT_total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l306_30643


namespace NUMINAMATH_GPT_fraction_not_equal_l306_30614

theorem fraction_not_equal : ¬ (7 / 5 = 1 + 4 / 20) :=
by
  -- We'll use simplification to demonstrate the inequality
  sorry

end NUMINAMATH_GPT_fraction_not_equal_l306_30614


namespace NUMINAMATH_GPT_supplementary_angles_ratio_l306_30685

theorem supplementary_angles_ratio (A B : ℝ) (h1 : A + B = 180) (h2 : A / B = 5 / 4) : B = 80 :=
by
   sorry

end NUMINAMATH_GPT_supplementary_angles_ratio_l306_30685


namespace NUMINAMATH_GPT_sequence_general_term_l306_30674

theorem sequence_general_term
  (a : ℕ → ℝ)
  (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n, a (n + 1) = 3 * a n + 7) :
  ∀ n, a n = 4 * 3^(n - 1) - 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l306_30674


namespace NUMINAMATH_GPT_problem1_problem2_l306_30687

section
variable {x a : ℝ}

-- Definitions of the functions
def f (x : ℝ) : ℝ := |x + 1|
def g (x : ℝ) (a : ℝ) : ℝ := 2 * |x| + a

-- Problem 1
theorem problem1 (a : ℝ) (H : a = -1) : 
  ∀ x : ℝ, f x ≤ g x a ↔ (x ≤ -2/3 ∨ 2 ≤ x) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∃ x₀ : ℝ, f x₀ ≥ 1/2 * g x₀ a) → a ≤ 2 :=
sorry

end

end NUMINAMATH_GPT_problem1_problem2_l306_30687


namespace NUMINAMATH_GPT_striped_turtles_adult_percentage_l306_30608

noncomputable def percentage_of_adult_striped_turtles (total_turtles : ℕ) (female_percentage : ℝ) (stripes_per_male : ℕ) (baby_stripes : ℕ) : ℝ :=
  let total_male := total_turtles * (1 - female_percentage)
  let total_striped_male := total_male / stripes_per_male
  let adult_striped_males := total_striped_male - baby_stripes
  (adult_striped_males / total_striped_male) * 100

theorem striped_turtles_adult_percentage :
  percentage_of_adult_striped_turtles 100 0.60 4 4 = 60 := 
  by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_striped_turtles_adult_percentage_l306_30608


namespace NUMINAMATH_GPT_aunt_may_milk_left_l306_30673

def morningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def eveningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def spoiledMilk (milkProduction : ℝ) (spoilageRate : ℝ) : ℝ :=
  milkProduction * spoilageRate

def freshMilk (totalMilk spoiledMilk : ℝ) : ℝ :=
  totalMilk - spoiledMilk

def soldMilk (freshMilk : ℝ) (saleRate : ℝ) : ℝ :=
  freshMilk * saleRate

def milkLeft (freshMilk soldMilk : ℝ) : ℝ :=
  freshMilk - soldMilk

noncomputable def totalMilkLeft (previousLeftover : ℝ) (morningLeft eveningLeft : ℝ) : ℝ :=
  previousLeftover + morningLeft + eveningLeft

theorem aunt_may_milk_left :
  let numCows := 5
  let numGoats := 4
  let numSheep := 10
  let cowMilkMorning := 13
  let goatMilkMorning := 0.5
  let sheepMilkMorning := 0.25
  let cowMilkEvening := 14
  let goatMilkEvening := 0.6
  let sheepMilkEvening := 0.2
  let morningSpoilageRate := 0.10
  let eveningSpoilageRate := 0.05
  let iceCreamSaleRate := 0.70
  let cheeseShopSaleRate := 0.80
  let previousLeftover := 15
  let morningMilk := morningMilkProduction numCows numGoats numSheep cowMilkMorning goatMilkMorning sheepMilkMorning
  let eveningMilk := eveningMilkProduction numCows numGoats numSheep cowMilkEvening goatMilkEvening sheepMilkEvening
  let morningSpoiled := spoiledMilk morningMilk morningSpoilageRate
  let eveningSpoiled := spoiledMilk eveningMilk eveningSpoilageRate
  let freshMorningMilk := freshMilk morningMilk morningSpoiled
  let freshEveningMilk := freshMilk eveningMilk eveningSpoiled
  let morningSold := soldMilk freshMorningMilk iceCreamSaleRate
  let eveningSold := soldMilk freshEveningMilk cheeseShopSaleRate
  let morningLeft := milkLeft freshMorningMilk morningSold
  let eveningLeft := milkLeft freshEveningMilk eveningSold
  totalMilkLeft previousLeftover morningLeft eveningLeft = 47.901 :=
by
  sorry

end NUMINAMATH_GPT_aunt_may_milk_left_l306_30673


namespace NUMINAMATH_GPT_more_blue_blocks_than_red_l306_30610

theorem more_blue_blocks_than_red 
  (red_blocks : ℕ) 
  (yellow_blocks : ℕ) 
  (blue_blocks : ℕ) 
  (total_blocks : ℕ) 
  (h_red : red_blocks = 18) 
  (h_yellow : yellow_blocks = red_blocks + 7) 
  (h_total : total_blocks = red_blocks + yellow_blocks + blue_blocks) 
  (h_total_given : total_blocks = 75) :
  blue_blocks - red_blocks = 14 :=
by sorry

end NUMINAMATH_GPT_more_blue_blocks_than_red_l306_30610


namespace NUMINAMATH_GPT_average_runs_l306_30651

/-- The average runs scored by the batsman in the first 20 matches is 40,
and in the next 10 matches is 30. We want to prove the average runs scored
by the batsman in all 30 matches is 36.67. --/
theorem average_runs (avg20 avg10 : ℕ) (num_matches_20 num_matches_10 : ℕ)
  (h1 : avg20 = 40) (h2 : avg10 = 30) (h3 : num_matches_20 = 20) (h4 : num_matches_10 = 10) :
  ((num_matches_20 * avg20 + num_matches_10 * avg10 : ℕ) : ℚ) / (num_matches_20 + num_matches_10 : ℕ) = 36.67 := by
  sorry

end NUMINAMATH_GPT_average_runs_l306_30651


namespace NUMINAMATH_GPT_profit_equation_correct_l306_30690

theorem profit_equation_correct (x : ℝ) : 
  let original_selling_price := 36
  let purchase_price := 20
  let original_sales_volume := 200
  let price_increase_effect := 5
  let desired_profit := 1200
  let original_profit_per_unit := original_selling_price - purchase_price
  let new_selling_price := original_selling_price + x
  let new_sales_volume := original_sales_volume - price_increase_effect * x
  (original_profit_per_unit + x) * new_sales_volume = desired_profit :=
sorry

end NUMINAMATH_GPT_profit_equation_correct_l306_30690


namespace NUMINAMATH_GPT_sum_division_l306_30619

theorem sum_division (x y z : ℝ) (total_share_y : ℝ) 
  (Hx : x = 1) 
  (Hy : y = 0.45) 
  (Hz : z = 0.30) 
  (share_y : total_share_y = 36) 
  : (x + y + z) * (total_share_y / y) = 140 := by
  sorry

end NUMINAMATH_GPT_sum_division_l306_30619


namespace NUMINAMATH_GPT_diff_of_squares_l306_30603

theorem diff_of_squares (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_diff_of_squares_l306_30603


namespace NUMINAMATH_GPT_additional_weight_difference_l306_30659

theorem additional_weight_difference (raw_squat sleeves_add wraps_percentage : ℝ) 
  (raw_squat_val : raw_squat = 600) 
  (sleeves_add_val : sleeves_add = 30) 
  (wraps_percentage_val : wraps_percentage = 0.25) : 
  (wraps_percentage * raw_squat) - sleeves_add = 120 :=
by
  rw [ raw_squat_val, sleeves_add_val, wraps_percentage_val ]
  norm_num

end NUMINAMATH_GPT_additional_weight_difference_l306_30659


namespace NUMINAMATH_GPT_evaluate_expression_l306_30626

theorem evaluate_expression : 
  (3^2 - 3 * 2) - (4^2 - 4 * 2) + (5^2 - 5 * 2) - (6^2 - 6 * 2) = -14 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l306_30626


namespace NUMINAMATH_GPT_Zelda_probability_success_l306_30680

variable (P : ℝ → ℝ)
variable (X Y Z : ℝ)

theorem Zelda_probability_success :
  P X = 1/3 ∧ P Y = 1/2 ∧ (P X) * (P Y) * (1 - P Z) = 0.0625 → P Z = 0.625 :=
by
  sorry

end NUMINAMATH_GPT_Zelda_probability_success_l306_30680


namespace NUMINAMATH_GPT_smallest_nonprime_with_large_primes_l306_30656

theorem smallest_nonprime_with_large_primes
  (n : ℕ)
  (h1 : n > 1)
  (h2 : ¬ Prime n)
  (h3 : ∀ p : ℕ, Prime p → p ∣ n → p ≥ 20) :
  660 < n ∧ n ≤ 670 :=
sorry

end NUMINAMATH_GPT_smallest_nonprime_with_large_primes_l306_30656


namespace NUMINAMATH_GPT_expand_product_l306_30646

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := 
by
  sorry

end NUMINAMATH_GPT_expand_product_l306_30646


namespace NUMINAMATH_GPT_remainder_when_x_plus_3uy_divided_by_y_eq_v_l306_30695

theorem remainder_when_x_plus_3uy_divided_by_y_eq_v
  (x y u v : ℕ) (h_pos_y : 0 < y) (h_division_algo : x = u * y + v) (h_remainder : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_x_plus_3uy_divided_by_y_eq_v_l306_30695


namespace NUMINAMATH_GPT_power_addition_proof_l306_30620

theorem power_addition_proof :
  (-2) ^ 48 + 3 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2) = 2 ^ 48 + 3 ^ 40 := 
by
  sorry

end NUMINAMATH_GPT_power_addition_proof_l306_30620


namespace NUMINAMATH_GPT_problem_arithmetic_sequence_l306_30617

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arithmetic_sequence (a : ℕ → ℝ) (d a2 a8 : ℝ) :
  arithmetic_sequence a d →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 450) →
  (a 1 + a 7 = 2 * a 4) →
  (a 2 + a 6 = 2 * a 4) →
  (a 2 + a 8 = 180) :=
by
  sorry

end NUMINAMATH_GPT_problem_arithmetic_sequence_l306_30617


namespace NUMINAMATH_GPT_base9_problem_l306_30671

def base9_add (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual addition for base 9
def base9_mul (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual multiplication for base 9

theorem base9_problem : base9_mul (base9_add 35 273) 2 = 620 := sorry

end NUMINAMATH_GPT_base9_problem_l306_30671


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l306_30613

theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) :
  r = 1 / 4 → S = 20 → S = a / (1 - r) → a = 15 :=
by
  intro hr hS hsum
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l306_30613


namespace NUMINAMATH_GPT_find_k_intersects_parabola_at_one_point_l306_30635

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end NUMINAMATH_GPT_find_k_intersects_parabola_at_one_point_l306_30635


namespace NUMINAMATH_GPT_smallest_k_divides_l306_30630

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end NUMINAMATH_GPT_smallest_k_divides_l306_30630


namespace NUMINAMATH_GPT_train_speed_proof_l306_30618

noncomputable def speedOfTrain (lengthOfTrain : ℝ) (timeToCross : ℝ) (speedOfMan : ℝ) : ℝ :=
  let man_speed_m_per_s := speedOfMan * 1000 / 3600
  let relative_speed := lengthOfTrain / timeToCross
  let train_speed_m_per_s := relative_speed + man_speed_m_per_s
  train_speed_m_per_s * 3600 / 1000

theorem train_speed_proof :
  speedOfTrain 100 5.999520038396929 3 = 63 := by
  sorry

end NUMINAMATH_GPT_train_speed_proof_l306_30618


namespace NUMINAMATH_GPT_Paul_correct_probability_l306_30691

theorem Paul_correct_probability :
  let P_Ghana := 1/2
  let P_Bolivia := 1/6
  let P_Argentina := 1/6
  let P_France := 1/6
  (P_Ghana^2 + P_Bolivia^2 + P_Argentina^2 + P_France^2) = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_Paul_correct_probability_l306_30691


namespace NUMINAMATH_GPT_billy_sleep_total_l306_30604

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end NUMINAMATH_GPT_billy_sleep_total_l306_30604


namespace NUMINAMATH_GPT_compute_expr_l306_30647

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end NUMINAMATH_GPT_compute_expr_l306_30647


namespace NUMINAMATH_GPT_problem_solution_l306_30662

theorem problem_solution :
  { x : ℝ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) } 
  = { x : ℝ | x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l306_30662


namespace NUMINAMATH_GPT_divisor_of_1025_l306_30683

theorem divisor_of_1025 (d : ℕ) (h1: 1015 + 10 = 1025) (h2 : d ∣ 1025) : d = 5 := 
sorry

end NUMINAMATH_GPT_divisor_of_1025_l306_30683


namespace NUMINAMATH_GPT_cookies_per_child_is_22_l306_30627

def total_cookies (num_packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  num_packages * cookies_per_package

def total_children (num_friends : ℕ) : ℕ :=
  num_friends + 1

def cookies_per_child (total_cookies : ℕ) (total_children : ℕ) : ℕ :=
  total_cookies / total_children

theorem cookies_per_child_is_22 :
  total_cookies 5 36 / total_children 7 = 22 := 
by
  sorry

end NUMINAMATH_GPT_cookies_per_child_is_22_l306_30627
