import Mathlib

namespace NUMINAMATH_GPT_quadratic_sum_solutions_l567_56768

noncomputable def sum_of_solutions (a b c : ℝ) : ℝ := 
  (-b/a)

theorem quadratic_sum_solutions : 
  ∀ x : ℝ, sum_of_solutions 1 (-9) (-45) = 9 := 
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_sum_solutions_l567_56768


namespace NUMINAMATH_GPT_stmt_A_stmt_B_stmt_C_stmt_D_l567_56705
open Real

def x_and_y_conditions := ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 3

theorem stmt_A : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (2 * (x * x + y * y) = 4) :=
by sorry

theorem stmt_B : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x * y = 9 / 8) :=
by sorry

theorem stmt_C : x_and_y_conditions → ¬ (∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (sqrt (x) + sqrt (2 * y) = sqrt 6)) :=
by sorry

theorem stmt_D : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x^2 + 4 * y^2 = 9 / 2) :=
by sorry

end NUMINAMATH_GPT_stmt_A_stmt_B_stmt_C_stmt_D_l567_56705


namespace NUMINAMATH_GPT_no_positive_integer_solution_l567_56759

def is_solution (x y z t : ℕ) : Prop :=
  x^2 + 5 * y^2 = z^2 ∧ 5 * x^2 + y^2 = t^2

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ is_solution x y z t :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solution_l567_56759


namespace NUMINAMATH_GPT_correct_points_per_answer_l567_56762

noncomputable def points_per_correct_answer (total_questions : ℕ) 
  (answered_correctly : ℕ) (final_score : ℝ) (penalty_per_incorrect : ℝ)
  (total_incorrect : ℕ := total_questions - answered_correctly) 
  (points_subtracted : ℝ := total_incorrect * penalty_per_incorrect) 
  (earned_points : ℝ := final_score + points_subtracted) : ℝ := 
    earned_points / answered_correctly

theorem correct_points_per_answer :
  points_per_correct_answer 120 104 100 0.25 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_correct_points_per_answer_l567_56762


namespace NUMINAMATH_GPT_cos_double_angle_identity_l567_56703

variable (α : Real)

theorem cos_double_angle_identity (h : Real.sin (Real.pi / 6 + α) = 1/3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7/9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_identity_l567_56703


namespace NUMINAMATH_GPT_price_per_vanilla_cookie_l567_56796

theorem price_per_vanilla_cookie (P : ℝ) (h1 : 220 + 70 * P = 360) : P = 2 := 
by 
  sorry

end NUMINAMATH_GPT_price_per_vanilla_cookie_l567_56796


namespace NUMINAMATH_GPT_markdown_calculation_l567_56749

noncomputable def markdown_percentage (P S : ℝ) (h_inc : P = S * 1.1494) : ℝ :=
  1 - (1 / 1.1494)

theorem markdown_calculation (P S : ℝ) (h_sale : S = P * (1 - markdown_percentage P S sorry / 100)) (h_inc : P = S * 1.1494) :
  markdown_percentage P S h_inc = 12.99 := 
sorry

end NUMINAMATH_GPT_markdown_calculation_l567_56749


namespace NUMINAMATH_GPT_dave_final_tickets_l567_56765

-- Define the initial number of tickets and operations
def initial_tickets : ℕ := 25
def tickets_spent_on_beanie : ℕ := 22
def tickets_won_after : ℕ := 15

-- Define the final number of tickets function
def final_tickets (initial : ℕ) (spent : ℕ) (won : ℕ) : ℕ :=
  initial - spent + won

-- Theorem stating that Dave would end up with 18 tickets given the conditions
theorem dave_final_tickets : final_tickets initial_tickets tickets_spent_on_beanie tickets_won_after = 18 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_dave_final_tickets_l567_56765


namespace NUMINAMATH_GPT_time_b_is_54_l567_56719

-- Define the time A takes to complete the work
def time_a := 27

-- Define the time B takes to complete the work as twice the time A takes
def time_b := 2 * time_a

-- Prove that B takes 54 days to complete the work
theorem time_b_is_54 : time_b = 54 :=
by
  sorry

end NUMINAMATH_GPT_time_b_is_54_l567_56719


namespace NUMINAMATH_GPT_inverse_modulo_1000000_l567_56735

def A : ℕ := 123456
def B : ℕ := 769230
def N : ℕ := 1053

theorem inverse_modulo_1000000 : (A * B * N) % 1000000 = 1 := 
  by 
  sorry

end NUMINAMATH_GPT_inverse_modulo_1000000_l567_56735


namespace NUMINAMATH_GPT_expand_binomial_trinomial_l567_56737

theorem expand_binomial_trinomial (x y z : ℝ) :
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 :=
by sorry

end NUMINAMATH_GPT_expand_binomial_trinomial_l567_56737


namespace NUMINAMATH_GPT_flashlight_price_percentage_l567_56797

theorem flashlight_price_percentage 
  (hoodie_price boots_price total_spent flashlight_price : ℝ)
  (discount_rate : ℝ)
  (h1 : hoodie_price = 80)
  (h2 : boots_price = 110)
  (h3 : discount_rate = 0.10)
  (h4 : total_spent = 195) 
  (h5 : total_spent = hoodie_price + ((1 - discount_rate) * boots_price) + flashlight_price) : 
  (flashlight_price / hoodie_price) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_flashlight_price_percentage_l567_56797


namespace NUMINAMATH_GPT_Melanie_gumballs_sale_l567_56745

theorem Melanie_gumballs_sale (gumballs : ℕ) (price_per_gumball : ℕ) (total_price : ℕ) :
  gumballs = 4 →
  price_per_gumball = 8 →
  total_price = gumballs * price_per_gumball →
  total_price = 32 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end NUMINAMATH_GPT_Melanie_gumballs_sale_l567_56745


namespace NUMINAMATH_GPT_smallest_y_divisible_l567_56779

theorem smallest_y_divisible (y : ℕ) : 
  (y % 3 = 2) ∧ (y % 5 = 4) ∧ (y % 7 = 6) → y = 104 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_divisible_l567_56779


namespace NUMINAMATH_GPT_price_of_pants_l567_56752

theorem price_of_pants (P : ℝ) 
  (h1 : (3 / 4) * P + P + (P + 10) = 340)
  (h2 : ∃ P, (3 / 4) * P + P + (P + 10) = 340) : 
  P = 120 :=
sorry

end NUMINAMATH_GPT_price_of_pants_l567_56752


namespace NUMINAMATH_GPT_solve_for_x_l567_56701

theorem solve_for_x (i x : ℂ) (h : i^2 = -1) (eq : 3 - 2 * i * x = 5 + 4 * i * x) : x = i / 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l567_56701


namespace NUMINAMATH_GPT_sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l567_56740

-- 1. Prove that 33 * 207 = 6831
theorem sum_of_207_instances_of_33 : 33 * 207 = 6831 := by
    sorry

-- 2. Prove that 3000 - 112 * 25 = 200
theorem difference_when_25_instances_of_112_are_subtracted_from_3000 : 3000 - 112 * 25 = 200 := by
    sorry

-- 3. Prove that 12 * 13 - (12 + 13) = 131
theorem difference_between_product_and_sum_of_12_and_13 : 12 * 13 - (12 + 13) = 131 := by
    sorry

end NUMINAMATH_GPT_sum_of_207_instances_of_33_difference_when_25_instances_of_112_are_subtracted_from_3000_difference_between_product_and_sum_of_12_and_13_l567_56740


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l567_56761

theorem arithmetic_expression_evaluation : 
  2000 - 80 + 200 - 120 = 2000 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l567_56761


namespace NUMINAMATH_GPT_pool_length_l567_56727

def volume_of_pool (width length depth : ℕ) : ℕ :=
  width * length * depth

def volume_of_water (volume : ℕ) (capacity : ℝ) : ℝ :=
  volume * capacity

theorem pool_length (L : ℕ) (width depth : ℕ) (capacity : ℝ) (drain_rate drain_time : ℕ) (h_capacity : capacity = 0.80)
  (h_width : width = 50) (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60) (h_drain_time : drain_time = 1000)
  (h_drain_volume : volume_of_water (volume_of_pool width L depth) capacity = drain_rate * drain_time) :
  L = 150 :=
by
  sorry

end NUMINAMATH_GPT_pool_length_l567_56727


namespace NUMINAMATH_GPT_total_first_year_students_l567_56758

theorem total_first_year_students (males : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) (N : ℕ)
  (h1 : males = 570)
  (h2 : sample_size = 110)
  (h3 : female_in_sample = 53)
  (h4 : N = ((sample_size - female_in_sample) * males) / (sample_size - (sample_size - female_in_sample)))
  : N = 1100 := 
by
  sorry

end NUMINAMATH_GPT_total_first_year_students_l567_56758


namespace NUMINAMATH_GPT_half_angle_quadrants_l567_56789

theorem half_angle_quadrants (α : ℝ) (k : ℤ) (hα : ∃ k : ℤ, (π/2 + k * 2 * π < α ∧ α < π + k * 2 * π)) : 
  ∃ k : ℤ, (π/4 + k * π < α/2 ∧ α/2 < π/2 + k * π) := 
sorry

end NUMINAMATH_GPT_half_angle_quadrants_l567_56789


namespace NUMINAMATH_GPT_probability_event_B_l567_56747

-- Define the type of trial outcomes, we're considering binary outcomes for simplicity
inductive Outcome
| win : Outcome
| lose : Outcome

open Outcome

def all_possible_outcomes := [
  [win, win, win],
  [win, win, win, lose],
  [win],
  [win],
  [lose],
  [win, win, lose, lose],
  [win, lose],
  [win, lose, win, lose, win],
  [win],
  [lose],
  [lose],
  [lose],
  [lose, win, win],
  [win, lose, lose, win],
  [lose, win, lose, lose],
  [win],
  [win],
  [lose],
  [lose],
  [lose, lose],
  [lose],
  [lose],
  [],
  [lose, lose, lose, lose]
]

-- Event A is winning a prize
def event_A := [
  [win, win, win],
  [win, win, win, lose],
  [win, win, lose, lose],
  [win, lose, win, lose, win],
  [win, lose, lose, win]
]

-- Event B is satisfying the condition \(a + b + c + d \leq 2\)
def event_B := [
  [lose],
  [win, lose],
  [lose, win],
  [win],
  [lose, lose],
  [lose, win, lose],
  [lose, lose, win],
  [lose, win, win],
  [win, lose, lose],
  [lose, lose, lose],
  []
]

-- Proof that the probability of event B equals 11/16
theorem probability_event_B : (event_B.length / all_possible_outcomes.length) = 11 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_event_B_l567_56747


namespace NUMINAMATH_GPT_find_k_l567_56769

theorem find_k 
  (h : ∀ x k : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0):
  ∃ (k : ℝ), k = -2 :=
sorry

end NUMINAMATH_GPT_find_k_l567_56769


namespace NUMINAMATH_GPT_man_older_than_son_l567_56780

theorem man_older_than_son (S M : ℕ) (h1 : S = 18) (h2 : M + 2 = 2 * (S + 2)) : M - S = 20 :=
by
  sorry

end NUMINAMATH_GPT_man_older_than_son_l567_56780


namespace NUMINAMATH_GPT_profit_correct_A_B_l567_56782

noncomputable def profit_per_tire_A (batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A : ℕ) : ℚ :=
  let cost_first_5000 := batch_cost_A1 + (cost_per_tire_A1 * 5000)
  let revenue_first_5000 := sell_price_tire_A1 * 5000
  let profit_first_5000 := revenue_first_5000 - cost_first_5000
  let cost_remaining := batch_cost_A2 + (cost_per_tire_A2 * (produced_A - 5000))
  let revenue_remaining := sell_price_tire_A2 * (produced_A - 5000)
  let profit_remaining := revenue_remaining - cost_remaining
  let total_profit := profit_first_5000 + profit_remaining
  total_profit / produced_A

noncomputable def profit_per_tire_B (batch_cost_B cost_per_tire_B sell_price_tire_B produced_B : ℕ) : ℚ :=
  let cost := batch_cost_B + (cost_per_tire_B * produced_B)
  let revenue := sell_price_tire_B * produced_B
  let profit := revenue - cost
  profit / produced_B

theorem profit_correct_A_B
  (batch_cost_A1 : ℕ := 22500) 
  (batch_cost_A2 : ℕ := 20000) 
  (cost_per_tire_A1 : ℕ := 8) 
  (cost_per_tire_A2 : ℕ := 6) 
  (sell_price_tire_A1 : ℕ := 20) 
  (sell_price_tire_A2 : ℕ := 18) 
  (produced_A : ℕ := 15000)
  (batch_cost_B : ℕ := 24000) 
  (cost_per_tire_B : ℕ := 7) 
  (sell_price_tire_B : ℕ := 19) 
  (produced_B : ℕ := 10000) :
  profit_per_tire_A batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A = 9.17 ∧
  profit_per_tire_B batch_cost_B cost_per_tire_B sell_price_tire_B produced_B = 9.60 :=
by
  sorry

end NUMINAMATH_GPT_profit_correct_A_B_l567_56782


namespace NUMINAMATH_GPT_count_four_digit_numbers_ending_25_l567_56702

theorem count_four_digit_numbers_ending_25 : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 100]) → ∃ n : ℕ, n = 100 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_ending_25_l567_56702


namespace NUMINAMATH_GPT_evaluate_expression_l567_56792

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l567_56792


namespace NUMINAMATH_GPT_fourth_vertex_of_parallelogram_l567_56706

structure Point where
  x : ℝ
  y : ℝ

def Q := Point.mk 1 (-1)
def R := Point.mk (-1) 0
def S := Point.mk 0 1
def V := Point.mk (-2) 2

theorem fourth_vertex_of_parallelogram (Q R S V : Point) :
  Q = ⟨1, -1⟩ ∧ R = ⟨-1, 0⟩ ∧ S = ⟨0, 1⟩ → V = ⟨-2, 2⟩ := by 
  sorry

end NUMINAMATH_GPT_fourth_vertex_of_parallelogram_l567_56706


namespace NUMINAMATH_GPT_find_second_number_l567_56716

theorem find_second_number (X : ℝ) : 
  (0.6 * 50 - 0.3 * X = 27) → X = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l567_56716


namespace NUMINAMATH_GPT_symmetric_point_l567_56743

theorem symmetric_point (x y : ℝ) (hx : x = -2) (hy : y = 3) (a b : ℝ) (hne : y = x + 1)
  (halfway : (a = (x + (-2)) / 2) ∧ (b = (y + 3) / 2) ∧ (2 * b = 2 * a + 2) ∧ (2 * b = 1)):
  (a, b) = (0, 1) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l567_56743


namespace NUMINAMATH_GPT_some_number_is_ten_l567_56739

theorem some_number_is_ten (x : ℕ) (h : 5 ^ 29 * 4 ^ 15 = 2 * x ^ 29) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_some_number_is_ten_l567_56739


namespace NUMINAMATH_GPT_arithmetic_mean_frac_l567_56750

theorem arithmetic_mean_frac (y b : ℝ) (h : y ≠ 0) : 
  (1 / 2 : ℝ) * ((y + b) / y + (2 * y - b) / y) = 1.5 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_frac_l567_56750


namespace NUMINAMATH_GPT_sequence_integral_terms_l567_56753

theorem sequence_integral_terms (x : ℕ → ℝ) (h1 : ∀ n, x n ≠ 0)
  (h2 : ∀ n > 2, x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))) :
  (∀ n, ∃ k : ℤ, x n = k) → x 1 = x 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_integral_terms_l567_56753


namespace NUMINAMATH_GPT_evaluate_expression_l567_56767

theorem evaluate_expression (x : ℝ) (h : |7 - 8 * (x - 12)| - |5 - 11| = 73) : x = 3 :=
  sorry

end NUMINAMATH_GPT_evaluate_expression_l567_56767


namespace NUMINAMATH_GPT_light_match_first_l567_56738

-- Define the conditions
def dark_room : Prop := true
def has_candle : Prop := true
def has_kerosene_lamp : Prop := true
def has_ready_to_use_stove : Prop := true
def has_single_match : Prop := true

-- Define the main question as a theorem
theorem light_match_first (h1 : dark_room) (h2 : has_candle) (h3 : has_kerosene_lamp) (h4 : has_ready_to_use_stove) (h5 : has_single_match) : true :=
by
  sorry

end NUMINAMATH_GPT_light_match_first_l567_56738


namespace NUMINAMATH_GPT_base_number_exponent_l567_56741

theorem base_number_exponent (x : ℝ) (h : ((x^4) * 3.456789) ^ 12 = y) (has_24_digits : true) : x = 10^12 :=
  sorry

end NUMINAMATH_GPT_base_number_exponent_l567_56741


namespace NUMINAMATH_GPT_tomato_seed_cost_l567_56731

theorem tomato_seed_cost (T : ℝ) 
  (h1 : 3 * 2.50 + 4 * T + 5 * 0.90 = 18) : 
  T = 1.50 := 
by
  sorry

end NUMINAMATH_GPT_tomato_seed_cost_l567_56731


namespace NUMINAMATH_GPT_no_solution_15x_29y_43z_t2_l567_56734

theorem no_solution_15x_29y_43z_t2 (x y z t : ℕ) : ¬ (15 ^ x + 29 ^ y + 43 ^ z = t ^ 2) :=
by {
  -- We'll insert the necessary conditions for the proof here
  sorry -- proof goes here
}

end NUMINAMATH_GPT_no_solution_15x_29y_43z_t2_l567_56734


namespace NUMINAMATH_GPT_parabola_equation_l567_56773

def is_parabola (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

def has_vertex (h k a b c : ℝ) : Prop :=
  b = -2 * a * h ∧ c = k + a * h^2 

def contains_point (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

theorem parabola_equation (a b c : ℝ) :
  has_vertex 3 (-2) a b c ∧ contains_point a b c 5 6 → 
  a = 2 ∧ b = -12 ∧ c = 16 := by
  sorry

end NUMINAMATH_GPT_parabola_equation_l567_56773


namespace NUMINAMATH_GPT_solve_linear_system_l567_56722

theorem solve_linear_system :
  ∃ x y z : ℝ, 
    (2 * x + y + z = -1) ∧ 
    (3 * y - z = -1) ∧ 
    (3 * x + 2 * y + 3 * z = -5) ∧ 
    (x = 1) ∧ 
    (y = -1) ∧ 
    (z = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l567_56722


namespace NUMINAMATH_GPT_power_inequality_l567_56744

variable (a b c : ℝ)

theorem power_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a * b^2 + a^2 * b + b * c^2 + b^2 * c + a * c^2 + a^2 * c :=
by sorry

end NUMINAMATH_GPT_power_inequality_l567_56744


namespace NUMINAMATH_GPT_time_to_fill_box_correct_l567_56786

def total_toys := 50
def mom_rate := 5
def mia_rate := 3

def time_to_fill_box (total_toys mom_rate mia_rate : ℕ) : ℚ :=
  let net_rate_per_cycle := mom_rate - mia_rate
  let cycles := ((total_toys - 1) / net_rate_per_cycle) + 1
  let total_seconds := cycles * 30
  total_seconds / 60

theorem time_to_fill_box_correct : time_to_fill_box total_toys mom_rate mia_rate = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_time_to_fill_box_correct_l567_56786


namespace NUMINAMATH_GPT_symmetric_points_sum_l567_56772

theorem symmetric_points_sum
  (a b : ℝ)
  (h1 : a = -3)
  (h2 : b = 2) :
  a + b = -1 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l567_56772


namespace NUMINAMATH_GPT_problem_180_l567_56755

variables (P Q : Prop)

theorem problem_180 (h : P → Q) : ¬ (P ∨ ¬Q) :=
sorry

end NUMINAMATH_GPT_problem_180_l567_56755


namespace NUMINAMATH_GPT_inequality_proof_l567_56788

theorem inequality_proof (x y z : ℝ) (hx : -1 < x) (hy : -1 < y) (hz : -1 < z) :
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l567_56788


namespace NUMINAMATH_GPT_min_value_expression_l567_56774

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + a) / b + 3

theorem min_value_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  ∃ x, min_expression a b c = x ∧ x ≥ 9 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l567_56774


namespace NUMINAMATH_GPT_train_trip_length_l567_56777

theorem train_trip_length (x D : ℝ) (h1 : D > 0) (h2 : x > 0) 
(h3 : 2 + 3 * (D - 2 * x) / (2 * x) + 1 = (x + 240) / x + 1 + 3 * (D - 2 * x - 120) / (2 * x) - 0.5) 
(h4 : 3 + 3 * (D - 2 * x) / (2 * x) = 7) :
  D = 640 :=
by
  sorry

end NUMINAMATH_GPT_train_trip_length_l567_56777


namespace NUMINAMATH_GPT_find_q_l567_56715

theorem find_q (q : ℕ) (h1 : 32 = 2^5) (h2 : 32^5 = 2^q) : q = 25 := by
  sorry

end NUMINAMATH_GPT_find_q_l567_56715


namespace NUMINAMATH_GPT_numWaysToPaintDoors_l567_56707

-- Define the number of doors and choices per door
def numDoors : ℕ := 3
def numChoicesPerDoor : ℕ := 2

-- Theorem statement that we want to prove
theorem numWaysToPaintDoors : numChoicesPerDoor ^ numDoors = 8 := by
  sorry

end NUMINAMATH_GPT_numWaysToPaintDoors_l567_56707


namespace NUMINAMATH_GPT_pq_sum_l567_56798

def single_digit (n : ℕ) : Prop := n < 10

theorem pq_sum (P Q : ℕ) (hP : single_digit P) (hQ : single_digit Q)
  (hSum : P * 100 + Q * 10 + Q + P * 110 + Q + Q * 111 = 876) : P + Q = 5 :=
by 
  -- Here we assume the expected outcome based on the problem solution
  sorry

end NUMINAMATH_GPT_pq_sum_l567_56798


namespace NUMINAMATH_GPT_problem_l567_56725

theorem problem (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : (14 * y - 2)^2 = 258 := by
  sorry

end NUMINAMATH_GPT_problem_l567_56725


namespace NUMINAMATH_GPT_pos_sol_eq_one_l567_56781

theorem pos_sol_eq_one (n : ℕ) (hn : 1 < n) :
  ∀ x : ℝ, 0 < x → (x ^ n - n * x + n - 1 = 0) → x = 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_pos_sol_eq_one_l567_56781


namespace NUMINAMATH_GPT_second_polygon_sides_l567_56799

theorem second_polygon_sides 
  (s : ℝ) -- side length of the second polygon
  (n1 n2 : ℕ) -- n1 = number of sides of the first polygon, n2 = number of sides of the second polygon
  (h1 : n1 = 40) -- first polygon has 40 sides
  (h2 : ∀ s1 s2 : ℝ, s1 = 3 * s2 → n1 * s1 = n2 * s2 → n2 = 120)
  : n2 = 120 := 
by
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l567_56799


namespace NUMINAMATH_GPT_angleC_equals_40_of_angleA_40_l567_56732

-- Define an arbitrary quadrilateral type and its angle A and angle C
structure Quadrilateral :=
  (angleA : ℝ)  -- angleA is in degrees
  (angleC : ℝ)  -- angleC is in degrees

-- Given condition in the problem
def quadrilateral_with_A_40 : Quadrilateral :=
  { angleA := 40, angleC := 0 } -- Initialize angleC as a placeholder

-- Theorem stating the problem's claim
theorem angleC_equals_40_of_angleA_40 :
  quadrilateral_with_A_40.angleA = 40 → quadrilateral_with_A_40.angleC = 40 :=
by
  sorry  -- Proof is omitted for brevity

end NUMINAMATH_GPT_angleC_equals_40_of_angleA_40_l567_56732


namespace NUMINAMATH_GPT_math_problem_l567_56783

-- Definitions of conditions
def cond1 (x a y b z c : ℝ) : Prop := x / a + y / b + z / c = 1
def cond2 (x a y b z c : ℝ) : Prop := a / x + b / y + c / z = 0

-- Theorem statement
theorem math_problem (x a y b z c : ℝ)
  (h1 : cond1 x a y b z c) (h2 : cond2 x a y b z c) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l567_56783


namespace NUMINAMATH_GPT_expression_value_l567_56714

theorem expression_value : (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l567_56714


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l567_56766

theorem geometric_sequence_fifth_term (α : ℕ → ℝ) (h : α 4 * α 5 * α 6 = 27) : α 5 = 3 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l567_56766


namespace NUMINAMATH_GPT_age_multiple_l567_56710

variables {R J K : ℕ}

theorem age_multiple (h1 : R = J + 6) (h2 : R = K + 3) (h3 : (R + 4) * (K + 4) = 108) :
  ∃ M : ℕ, R + 4 = M * (J + 4) ∧ M = 2 :=
sorry

end NUMINAMATH_GPT_age_multiple_l567_56710


namespace NUMINAMATH_GPT_joe_initial_paint_l567_56751
-- Use necessary imports

-- Define the hypothesis
def initial_paint_gallons (g : ℝ) :=
  (1 / 4) * g + (1 / 7) * (3 / 4) * g = 128.57

-- Define the theorem
theorem joe_initial_paint (P : ℝ) (h : initial_paint_gallons P) : P = 360 :=
  sorry

end NUMINAMATH_GPT_joe_initial_paint_l567_56751


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l567_56713

theorem sufficient_but_not_necessary_condition (a b : ℝ) (hb : b < -1) : |a| + |b| > 1 := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l567_56713


namespace NUMINAMATH_GPT_solve_for_a_l567_56785

theorem solve_for_a (x a : ℝ) (h : x = -2) (hx : 2 * x + 3 * a = 0) : a = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l567_56785


namespace NUMINAMATH_GPT_abs_neg_2023_l567_56778

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l567_56778


namespace NUMINAMATH_GPT_smallest_denominator_is_168_l567_56787

theorem smallest_denominator_is_168 (a b : ℕ) (h1: Nat.gcd a 600 = 1) (h2: Nat.gcd b 700 = 1) :
  ∃ k, Nat.gcd (7 * a + 6 * b) 4200 = k ∧ k = 25 ∧ (4200 / k) = 168 :=
sorry

end NUMINAMATH_GPT_smallest_denominator_is_168_l567_56787


namespace NUMINAMATH_GPT_polynomial_factorization_l567_56721

theorem polynomial_factorization : ∃ q : Polynomial ℝ, (Polynomial.X ^ 4 - 6 * Polynomial.X ^ 2 + 25) = (Polynomial.X ^ 2 + 5) * q :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l567_56721


namespace NUMINAMATH_GPT_eq_solution_set_l567_56711

theorem eq_solution_set (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^(a^a)) :
  (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) :=
by
  sorry

end NUMINAMATH_GPT_eq_solution_set_l567_56711


namespace NUMINAMATH_GPT_solve_for_y_l567_56760

-- Define the main theorem to be proven
theorem solve_for_y (y : ℤ) (h : 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y) : y = 22 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l567_56760


namespace NUMINAMATH_GPT_find_number_of_spiders_l567_56771

theorem find_number_of_spiders (S : ℕ) (h1 : (1 / 2) * S = 5) : S = 10 := sorry

end NUMINAMATH_GPT_find_number_of_spiders_l567_56771


namespace NUMINAMATH_GPT_minimum_value_expression_l567_56724

theorem minimum_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l567_56724


namespace NUMINAMATH_GPT_graph_transformation_matches_B_l567_56746

noncomputable def f (x : ℝ) : ℝ :=
  if (-3 : ℝ) ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- Define this part to handle cases outside the given range.

noncomputable def g (x : ℝ) : ℝ :=
  f ((1 - x) / 2)

theorem graph_transformation_matches_B :
  g = some_graph_function_B := 
sorry

end NUMINAMATH_GPT_graph_transformation_matches_B_l567_56746


namespace NUMINAMATH_GPT_volume_of_prism_l567_56757

-- Define the variables a, b, c and the conditions
variables (a b c : ℝ)

-- Given conditions
theorem volume_of_prism (h1 : a * b = 48) (h2 : b * c = 49) (h3 : a * c = 50) :
  a * b * c = 343 :=
by {
  sorry
}

end NUMINAMATH_GPT_volume_of_prism_l567_56757


namespace NUMINAMATH_GPT_complex_div_imag_unit_l567_56728

theorem complex_div_imag_unit (i : ℂ) (h : i^2 = -1) : (1 + i) / (1 - i) = i :=
sorry

end NUMINAMATH_GPT_complex_div_imag_unit_l567_56728


namespace NUMINAMATH_GPT_lines_parallel_a_eq_sqrt2_l567_56748

theorem lines_parallel_a_eq_sqrt2 (a : ℝ) (h1 : 1 ≠ 0) :
  (∀ a ≠ 0, ((- (1 / (2 * a))) = (- a / 2)) → a = Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_a_eq_sqrt2_l567_56748


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l567_56742

open Real

theorem tan_alpha_plus_pi (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  tan (α + π) = -3 / 4 :=
sorry

theorem cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two (α : ℝ)
  (h1 : sin α = 3 / 5)
  (h2 : π / 2 < α ∧ α < π) :
  cos (α - π / 2) * sin (α + 3 * π / 2) = 12 / 25 :=
sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_cos_alpha_minus_pi_div_two_sin_alpha_plus_3pi_div_two_l567_56742


namespace NUMINAMATH_GPT_value_of_a_l567_56733

theorem value_of_a (a : ℤ) (x y : ℝ) :
  (a - 2) ≠ 0 →
  (2 + |a| + 1 = 5) →
  a = -2 :=
by
  intro ha hdeg
  sorry

end NUMINAMATH_GPT_value_of_a_l567_56733


namespace NUMINAMATH_GPT_remainder_1234_mul_5678_mod_1000_l567_56723

theorem remainder_1234_mul_5678_mod_1000 :
  (1234 * 5678) % 1000 = 652 := by
  sorry

end NUMINAMATH_GPT_remainder_1234_mul_5678_mod_1000_l567_56723


namespace NUMINAMATH_GPT_solve_abs_eq_l567_56712

theorem solve_abs_eq (x : ℝ) : (|x - 3| = 5 - x) ↔ (x = 4) := 
by
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l567_56712


namespace NUMINAMATH_GPT_baby_grasshoppers_l567_56790

-- Definition for the number of grasshoppers on the plant
def grasshoppers_on_plant : ℕ := 7

-- Definition for the total number of grasshoppers found
def total_grasshoppers : ℕ := 31

-- The theorem to prove the number of baby grasshoppers under the plant
theorem baby_grasshoppers : 
  (total_grasshoppers - grasshoppers_on_plant) = 24 := 
by
  sorry

end NUMINAMATH_GPT_baby_grasshoppers_l567_56790


namespace NUMINAMATH_GPT_father_age_three_times_xiaojun_after_years_l567_56754

theorem father_age_three_times_xiaojun_after_years (years_passed : ℕ) (xiaojun_current_age : ℕ) (father_current_age : ℕ) 
  (h1 : xiaojun_current_age = 5) (h2 : father_current_age = 31) (h3 : years_passed = 8) :
  father_current_age + years_passed = 3 * (xiaojun_current_age + years_passed) := by
  sorry

end NUMINAMATH_GPT_father_age_three_times_xiaojun_after_years_l567_56754


namespace NUMINAMATH_GPT_computation_result_l567_56763

def a : ℕ := 3
def b : ℕ := 5
def c : ℕ := 7

theorem computation_result :
  (a + b + c) ^ 2 + (a ^ 2 + b ^ 2 + c ^ 2) = 308 := by
  sorry

end NUMINAMATH_GPT_computation_result_l567_56763


namespace NUMINAMATH_GPT_total_students_l567_56795

theorem total_students
  (T : ℝ) 
  (h1 : 0.20 * T = 168)
  (h2 : 0.30 * T = 252) : T = 840 :=
sorry

end NUMINAMATH_GPT_total_students_l567_56795


namespace NUMINAMATH_GPT_max_a_value_l567_56726

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : b + d = 200) : a ≤ 449 :=
by sorry

end NUMINAMATH_GPT_max_a_value_l567_56726


namespace NUMINAMATH_GPT_percent_increase_l567_56791

theorem percent_increase (N : ℝ) (h : (1 / 7) * N = 1) : 
  N = 7 ∧ (N - (4 / 7)) / (4 / 7) * 100 = 1125.0000000000002 := 
by 
  sorry

end NUMINAMATH_GPT_percent_increase_l567_56791


namespace NUMINAMATH_GPT_probability_of_two_digit_number_l567_56775

def total_elements_in_set : ℕ := 961
def two_digit_elements_in_set : ℕ := 60

theorem probability_of_two_digit_number :
  (two_digit_elements_in_set : ℚ) / total_elements_in_set = 60 / 961 := by
  sorry

end NUMINAMATH_GPT_probability_of_two_digit_number_l567_56775


namespace NUMINAMATH_GPT_trapezoid_perimeter_l567_56717

noncomputable def perimeter_of_trapezoid (AB CD BC AD AP DQ : ℕ) : ℕ :=
  AB + BC + CD + AD

theorem trapezoid_perimeter (AB CD BC AP DQ : ℕ) (hBC : BC = 50) (hAP : AP = 18) (hDQ : DQ = 7) :
  perimeter_of_trapezoid AB CD BC (AP + BC + DQ) AP DQ = 180 :=
by 
  unfold perimeter_of_trapezoid
  rw [hBC, hAP, hDQ]
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l567_56717


namespace NUMINAMATH_GPT_selected_six_numbers_have_two_correct_statements_l567_56700

def selection := {n : ℕ // 1 ≤ n ∧ n ≤ 11}

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_multiple (a b : ℕ) : Prop := a ≠ b ∧ (b % a = 0 ∨ a % b = 0)

def is_double_multiple (a b : ℕ) : Prop := a ≠ b ∧ (2 * a = b ∨ 2 * b = a)

theorem selected_six_numbers_have_two_correct_statements (s : Finset selection) (h : s.card = 6) :
  ∃ n1 n2 : selection, is_coprime n1.1 n2.1 ∧ ∃ n1 n2 : selection, is_double_multiple n1.1 n2.1 :=
by
  -- The detailed proof is omitted.
  sorry

end NUMINAMATH_GPT_selected_six_numbers_have_two_correct_statements_l567_56700


namespace NUMINAMATH_GPT_paco_cookies_l567_56764

theorem paco_cookies (initial_cookies: ℕ) (eaten_cookies: ℕ) (final_cookies: ℕ) (bought_cookies: ℕ) 
  (h1 : initial_cookies = 40)
  (h2 : eaten_cookies = 2)
  (h3 : final_cookies = 75)
  (h4 : initial_cookies - eaten_cookies + bought_cookies = final_cookies) :
  bought_cookies = 37 :=
by
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_paco_cookies_l567_56764


namespace NUMINAMATH_GPT_correct_option_is_C_l567_56709

def option_A (x : ℝ) : Prop := (-x^2)^3 = -x^5
def option_B (x : ℝ) : Prop := x^2 + x^3 = x^5
def option_C (x : ℝ) : Prop := x^3 * x^4 = x^7
def option_D (x : ℝ) : Prop := 2 * x^3 - x^3 = 1

theorem correct_option_is_C (x : ℝ) : ¬ option_A x ∧ ¬ option_B x ∧ option_C x ∧ ¬ option_D x :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_C_l567_56709


namespace NUMINAMATH_GPT_valid_tree_arrangements_l567_56718

-- Define the types of trees
inductive TreeType
| Birch
| Oak

-- Define the condition that each tree must be adjacent to a tree of the other type
def isValidArrangement (trees : List TreeType) : Prop :=
  ∀ (i : ℕ), i < trees.length - 1 → trees.nthLe i sorry ≠ trees.nthLe (i + 1) sorry

-- Define the main problem
theorem valid_tree_arrangements : ∃ (ways : Nat), ways = 16 ∧
  ∃ (arrangements : List (List TreeType)), arrangements.length = ways ∧
    ∀ arrangement ∈ arrangements, arrangement.length = 7 ∧ isValidArrangement arrangement :=
sorry

end NUMINAMATH_GPT_valid_tree_arrangements_l567_56718


namespace NUMINAMATH_GPT_expression_I_evaluation_expression_II_evaluation_l567_56704

theorem expression_I_evaluation :
  ( (3 / 2) ^ (-2: ℤ) - (49 / 81) ^ (0.5: ℝ) + (0.008: ℝ) ^ (-2 / 3: ℝ) * (2 / 25) ) = (5 / 3) := 
by
  sorry

theorem expression_II_evaluation :
  ( (Real.logb 2 2) ^ 2 + (Real.logb 10 20) * (Real.logb 10 5) ) = (17 / 9) := 
by
  sorry

end NUMINAMATH_GPT_expression_I_evaluation_expression_II_evaluation_l567_56704


namespace NUMINAMATH_GPT_agent_commission_calculation_l567_56793

-- Define the conditions
def total_sales : ℝ := 250
def commission_rate : ℝ := 0.05

-- Define the commission calculation function
def calculate_commission (sales : ℝ) (rate : ℝ) : ℝ :=
  sales * rate

-- Proposition stating the desired commission
def agent_commission_is_correct : Prop :=
  calculate_commission total_sales commission_rate = 12.5

-- State the proof problem
theorem agent_commission_calculation : agent_commission_is_correct :=
by sorry

end NUMINAMATH_GPT_agent_commission_calculation_l567_56793


namespace NUMINAMATH_GPT_propP_necessary_but_not_sufficient_l567_56784

open Function Real

variable (f : ℝ → ℝ)

-- Conditions: differentiable function f and the proposition Q
def diff_and_propQ (h_deriv : Differentiable ℝ f) : Prop :=
∀ x : ℝ, abs (deriv f x) < 2018

-- Proposition P
def propP : Prop :=
∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018

-- Final statement
theorem propP_necessary_but_not_sufficient (h_deriv : Differentiable ℝ f) (hQ : diff_and_propQ f h_deriv) : 
  (∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) ∧ 
  ¬(∀ x : ℝ, abs (deriv f x) < 2018 ↔ ∀ x1 x2 : ℝ, x1 ≠ x2 → abs ((f x1 - f x2) / (x1 - x2)) < 2018) :=
by
  sorry

end NUMINAMATH_GPT_propP_necessary_but_not_sufficient_l567_56784


namespace NUMINAMATH_GPT_roots_quartic_ab_plus_a_plus_b_l567_56730

theorem roots_quartic_ab_plus_a_plus_b (a b : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0) :
  a * b + a + b = -1 := 
sorry

end NUMINAMATH_GPT_roots_quartic_ab_plus_a_plus_b_l567_56730


namespace NUMINAMATH_GPT_chastity_leftover_money_l567_56756

theorem chastity_leftover_money (n_lollipops : ℕ) (price_lollipop : ℝ) (n_gummies : ℕ) (price_gummy : ℝ) (initial_money : ℝ) :
  n_lollipops = 4 →
  price_lollipop = 1.50 →
  n_gummies = 2 →
  price_gummy = 2 →
  initial_money = 15 →
  initial_money - ((n_lollipops * price_lollipop) + (n_gummies * price_gummy)) = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end NUMINAMATH_GPT_chastity_leftover_money_l567_56756


namespace NUMINAMATH_GPT_range_my_function_l567_56708

noncomputable def my_function (x : ℝ) := (x^2 + 4 * x + 3) / (x + 2)

theorem range_my_function : 
  Set.range my_function = Set.univ := 
sorry

end NUMINAMATH_GPT_range_my_function_l567_56708


namespace NUMINAMATH_GPT_shaded_area_l567_56770

theorem shaded_area 
  (side_of_square : ℝ)
  (arc_radius : ℝ)
  (side_length_eq_sqrt_two : side_of_square = Real.sqrt 2)
  (radius_eq_one : arc_radius = 1) :
  let square_area := 4
  let sector_area := 3 * Real.pi
  let shaded_area := square_area + sector_area
  shaded_area = 4 + 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l567_56770


namespace NUMINAMATH_GPT_find_fixed_point_on_ellipse_l567_56776

theorem find_fixed_point_on_ellipse (a b c : ℝ) (h_gt_zero : a > b ∧ b > 0)
    (h_ellipse : ∀ (P : ℝ × ℝ), (P.1 ^ 2 / a ^ 2) + (P.2 ^ 2 / b ^ 2) = 1)
    (A1 A2 : ℝ × ℝ)
    (h_A1 : A1 = (-a, 0))
    (h_A2 : A2 = (a, 0))
    (MC : ℝ) (h_MC : MC = (a^2 + b^2) / c) :
  ∃ (M : ℝ × ℝ), M = (MC, 0) := 
sorry

end NUMINAMATH_GPT_find_fixed_point_on_ellipse_l567_56776


namespace NUMINAMATH_GPT_side_length_of_square_l567_56736

theorem side_length_of_square (s : ℝ) (h : s^2 = 6 * (4 * s)) : s = 24 :=
by sorry

end NUMINAMATH_GPT_side_length_of_square_l567_56736


namespace NUMINAMATH_GPT_train_platform_length_l567_56729

theorem train_platform_length (time_platform : ℝ) (time_man : ℝ) (speed_km_per_hr : ℝ) :
  time_platform = 34 ∧ time_man = 20 ∧ speed_km_per_hr = 54 →
  let speed_m_per_s := speed_km_per_hr * (5/18)
  let length_train := speed_m_per_s * time_man
  let time_to_cover_platform := time_platform - time_man
  let length_platform := speed_m_per_s * time_to_cover_platform
  length_platform = 210 := 
by {
  sorry
}

end NUMINAMATH_GPT_train_platform_length_l567_56729


namespace NUMINAMATH_GPT_spaceship_distance_traveled_l567_56720

theorem spaceship_distance_traveled (d_ex : ℝ) (d_xy : ℝ) (d_total : ℝ) :
  d_ex = 0.5 → d_xy = 0.1 → d_total = 0.7 → (d_total - (d_ex + d_xy)) = 0.1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_spaceship_distance_traveled_l567_56720


namespace NUMINAMATH_GPT_train_cross_pole_time_l567_56794

-- Defining the given conditions
def speed_km_hr : ℕ := 54
def length_m : ℕ := 135

-- Conversion of speed from km/hr to m/s
def speed_m_s : ℤ := (54 * 1000) / 3600

-- Statement to be proved
theorem train_cross_pole_time : (length_m : ℤ) / speed_m_s = 9 := by
  sorry

end NUMINAMATH_GPT_train_cross_pole_time_l567_56794
