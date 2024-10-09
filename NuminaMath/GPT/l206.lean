import Mathlib

namespace ellipse_focus_value_l206_20657

theorem ellipse_focus_value (k : ℝ) (hk : 5 * (0:ℝ)^2 - k * (2:ℝ)^2 = 5) : k = -1 :=
by
  sorry

end ellipse_focus_value_l206_20657


namespace largest_n_for_factorable_polynomial_l206_20662

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end largest_n_for_factorable_polynomial_l206_20662


namespace verify_quadratic_eq_l206_20683

def is_quadratic (eq : String) : Prop :=
  eq = "ax^2 + bx + c = 0"

theorem verify_quadratic_eq :
  is_quadratic "x^2 - 1 = 0" :=
by
  -- Auxiliary functions or steps can be introduced if necessary, but proof is omitted here.
  sorry

end verify_quadratic_eq_l206_20683


namespace exist_identical_2x2_squares_l206_20611

theorem exist_identical_2x2_squares : 
  ∃ sq1 sq2 : Finset (Fin 5 × Fin 5), 
    sq1.card = 4 ∧ sq2.card = 4 ∧ 
    (∀ (i : Fin 5) (j : Fin 5), 
      (i = 0 ∧ j = 0) ∨ (i = 4 ∧ j = 4) → 
      (i, j) ∈ sq1 ∧ (i, j) ∈ sq2 ∧ 
      (sq1 ≠ sq2 → ∃ p ∈ sq1, p ∉ sq2)) :=
sorry

end exist_identical_2x2_squares_l206_20611


namespace number_of_odd_positive_integer_triples_sum_25_l206_20646

theorem number_of_odd_positive_integer_triples_sum_25 :
  ∃ n : ℕ, (
    n = 78 ∧
    ∃ (a b c : ℕ), 
      (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 25
  ) := 
sorry

end number_of_odd_positive_integer_triples_sum_25_l206_20646


namespace larger_of_two_numbers_l206_20636

theorem larger_of_two_numbers
  (A B hcf : ℕ)
  (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 9)
  (h_factor2 : factor2 = 10)
  (h_lcm : (A * B) / (hcf) = (hcf * factor1 * factor2))
  (h_A : A = hcf * 9)
  (h_B : B = hcf * 10) :
  max A B = 230 := by
  sorry

end larger_of_two_numbers_l206_20636


namespace derivative_at_2_l206_20678

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_at_2 : deriv f 2 = (1 - Real.log 2) / 4 :=
by
  sorry

end derivative_at_2_l206_20678


namespace total_soccer_games_l206_20627

theorem total_soccer_games (months : ℕ) (games_per_month : ℕ) (h_months : months = 3) (h_games_per_month : games_per_month = 9) : months * games_per_month = 27 :=
by
  sorry

end total_soccer_games_l206_20627


namespace max_temp_range_l206_20661

theorem max_temp_range (avg_temp : ℝ) (lowest_temp : ℝ) (days : ℕ) (total_temp : ℝ) (range : ℝ) : 
  avg_temp = 45 → 
  lowest_temp = 42 → 
  days = 5 → 
  total_temp = avg_temp * days → 
  range = 6 := 
by 
  sorry

end max_temp_range_l206_20661


namespace number_of_pieces_of_bubble_gum_l206_20673

theorem number_of_pieces_of_bubble_gum (cost_per_piece total_cost : ℤ) (h1 : cost_per_piece = 18) (h2 : total_cost = 2448) :
  total_cost / cost_per_piece = 136 :=
by
  rw [h1, h2]
  norm_num

end number_of_pieces_of_bubble_gum_l206_20673


namespace fraction_meaningful_iff_l206_20621

theorem fraction_meaningful_iff (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) := 
by
  sorry

end fraction_meaningful_iff_l206_20621


namespace amanda_jogging_distance_l206_20669

/-- Amanda's jogging path and the distance calculation. -/
theorem amanda_jogging_distance:
  let east_leg := 1.5
  let northwest_leg := 2
  let southwest_leg := 1
  -- Convert runs to displacement components
  let nw_x := northwest_leg / Real.sqrt 2
  let nw_y := northwest_leg / Real.sqrt 2
  let sw_x := southwest_leg / Real.sqrt 2
  let sw_y := southwest_leg / Real.sqrt 2
  -- Calculate net displacements
  let net_east := east_leg - (nw_x + sw_x)
  let net_north := nw_y - sw_y
  -- Final distance back to starting point
  let distance := Real.sqrt (net_east^2 + net_north^2)
  distance = Real.sqrt ((1.5 - 3 * Real.sqrt 2 / 2)^2 + (Real.sqrt 2 / 2)^2) := sorry

end amanda_jogging_distance_l206_20669


namespace number_of_sets_of_popcorn_l206_20604

theorem number_of_sets_of_popcorn (t p s : ℝ) (k : ℕ) 
  (h1 : t = 5)
  (h2 : p = 0.80 * t)
  (h3 : s = 0.50 * p)
  (h4 : 4 * t + 4 * s + k * p = 36) :
  k = 2 :=
by sorry

end number_of_sets_of_popcorn_l206_20604


namespace sum_of_x_coordinates_l206_20623

def exists_common_point (x : ℕ) : Prop :=
  (3 * x + 5) % 9 = (7 * x + 3) % 9

theorem sum_of_x_coordinates :
  ∃ x : ℕ, exists_common_point x ∧ x % 9 = 5 := 
by
  sorry

end sum_of_x_coordinates_l206_20623


namespace find_abs_3h_minus_4k_l206_20689

theorem find_abs_3h_minus_4k
  (h k : ℤ)
  (factor1_eq_zero : 3 * (-3)^3 - h * (-3) - 3 * k = 0)
  (factor2_eq_zero : 3 * 2^3 - h * 2 - 3 * k = 0) :
  |3 * h - 4 * k| = 615 :=
by
  sorry

end find_abs_3h_minus_4k_l206_20689


namespace Finn_initial_goldfish_l206_20644

variable (x : ℕ)

-- Defining the conditions
def number_of_goldfish_initial (x : ℕ) : Prop :=
  ∃ y z : ℕ, y = 32 ∧ z = 57 ∧ x = y + z 

-- Theorem statement to prove Finn's initial number of goldfish
theorem Finn_initial_goldfish (x : ℕ) (h : number_of_goldfish_initial x) : x = 89 := by
  sorry

end Finn_initial_goldfish_l206_20644


namespace ratio_age_difference_to_pencils_l206_20628

-- Definitions of the given problem conditions
def AsafAge : ℕ := 50
def SumOfAges : ℕ := 140
def AlexanderAge : ℕ := SumOfAges - AsafAge

def PencilDifference : ℕ := 60
def TotalPencils : ℕ := 220
def AsafPencils : ℕ := (TotalPencils - PencilDifference) / 2
def AlexanderPencils : ℕ := AsafPencils + PencilDifference

-- Define the age difference and the ratio
def AgeDifference : ℕ := AlexanderAge - AsafAge
def Ratio : ℚ := AgeDifference / AsafPencils

theorem ratio_age_difference_to_pencils : Ratio = 1 / 2 := by
  sorry

end ratio_age_difference_to_pencils_l206_20628


namespace calculate_expression_l206_20633

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 :=
by
  sorry

end calculate_expression_l206_20633


namespace polynomial_relation_l206_20643

variables {a b c : ℝ}

theorem polynomial_relation
  (h1: a ≠ 0) (h2: b ≠ 0) (h3: c ≠ 0) (h4: a + b + c = 0) :
  ((a^7 + b^7 + c^7)^2) / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 :=
sorry

end polynomial_relation_l206_20643


namespace find_product_of_variables_l206_20617

variables (a b c d : ℚ)

def system_of_equations (a b c d : ℚ) :=
  3 * a + 4 * b + 6 * c + 9 * d = 45 ∧
  4 * (d + c) = b + 1 ∧
  4 * b + 2 * c = a ∧
  2 * c - 2 = d

theorem find_product_of_variables :
  system_of_equations a b c d → a * b * c * d = 162 / 185 :=
by sorry

end find_product_of_variables_l206_20617


namespace find_n_18_l206_20649

def valid_denominations (n : ℕ) : Prop :=
  ∀ k < 106, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c

def cannot_form_106 (n : ℕ) : Prop :=
  ¬ ∃ a b c : ℕ, 106 = 7 * a + n * b + (n + 1) * c

theorem find_n_18 : 
  ∃ n : ℕ, valid_denominations n ∧ cannot_form_106 n ∧ ∀ m < n, ¬ (valid_denominations m ∧ cannot_form_106 m) :=
sorry

end find_n_18_l206_20649


namespace first_interest_rate_is_correct_l206_20664

theorem first_interest_rate_is_correct :
  let A1 := 1500.0000000000007
  let A2 := 2500 - A1
  let yearly_income := 135
  (15.0 * (r / 100) + 6.0 * (A2 / 100) = yearly_income) -> r = 5.000000000000003 :=
sorry

end first_interest_rate_is_correct_l206_20664


namespace edward_rides_l206_20655

theorem edward_rides (total_tickets tickets_spent tickets_per_ride rides : ℕ)
    (h1 : total_tickets = 79)
    (h2 : tickets_spent = 23)
    (h3 : tickets_per_ride = 7)
    (h4 : rides = (total_tickets - tickets_spent) / tickets_per_ride) :
    rides = 8 := by sorry

end edward_rides_l206_20655


namespace find_n_l206_20652

open Nat

theorem find_n (n : ℕ) (d : ℕ → ℕ) (h1 : d 1 = 1) (hk : d 6^2 + d 7^2 - 1 = n) :
  n = 1984 ∨ n = 144 :=
by
  sorry

end find_n_l206_20652


namespace rowing_time_ratio_l206_20603

theorem rowing_time_ratio
  (V_b : ℝ) (V_s : ℝ) (V_upstream : ℝ) (V_downstream : ℝ) (T_upstream T_downstream : ℝ)
  (h1 : V_b = 39) (h2 : V_s = 13)
  (h3 : V_upstream = V_b - V_s) (h4 : V_downstream = V_b + V_s)
  (h5 : T_upstream * V_upstream = T_downstream * V_downstream) :
  T_upstream / T_downstream = 2 := by
  sorry

end rowing_time_ratio_l206_20603


namespace PlayStation_cost_l206_20602

def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def price_per_game : ℝ := 7.5
def games_to_sell : ℕ := 20
def total_gift_money : ℝ := birthday_money + christmas_money
def total_games_money : ℝ := games_to_sell * price_per_game
def total_money : ℝ := total_gift_money + total_games_money

theorem PlayStation_cost : total_money = 500 := by
  sorry

end PlayStation_cost_l206_20602


namespace sufficient_and_necessary_l206_20693

theorem sufficient_and_necessary (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end sufficient_and_necessary_l206_20693


namespace tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l206_20619

theorem tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth (a : ℝ) (h : Real.tan a = 2) :
  Real.cos (2 * a) + Real.sin (2 * a) = 1 / 5 :=
by
  sorry

end tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l206_20619


namespace four_digit_number_l206_20635

theorem four_digit_number (a b c d : ℕ)
    (h1 : 0 ≤ a) (h2 : a ≤ 9)
    (h3 : 0 ≤ b) (h4 : b ≤ 9)
    (h5 : 0 ≤ c) (h6 : c ≤ 9)
    (h7 : 0 ≤ d) (h8 : d ≤ 9)
    (h9 : 2 * (1000 * a + 100 * b + 10 * c + d) + 1000 = 1000 * d + 100 * c + 10 * b + a)
    : (1000 * a + 100 * b + 10 * c + d) = 2996 :=
by
  sorry

end four_digit_number_l206_20635


namespace greatest_expression_value_l206_20618

noncomputable def greatest_expression : ℝ := 0.9986095661846496

theorem greatest_expression_value : greatest_expression = 0.9986095661846496 :=
by
  -- proof goes here
  sorry

end greatest_expression_value_l206_20618


namespace rational_coefficients_count_l206_20685

theorem rational_coefficients_count : 
  ∃ n, n = 84 ∧ ∀ k, (0 ≤ k ∧ k ≤ 500) → 
            (k % 3 = 0 ∧ (500 - k) % 2 = 0) → 
            n = 84 :=
by
  sorry

end rational_coefficients_count_l206_20685


namespace find_f_at_one_l206_20668

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

theorem find_f_at_one (h_cond : f a b (-1) = 10) : f a b (1) = 14 := by
  sorry

end find_f_at_one_l206_20668


namespace rectangle_same_color_l206_20616

/-- In a 3 × 7 grid where each square is either black or white, 
  there exists a rectangle whose four corners are of the same color. -/
theorem rectangle_same_color (grid : Fin 3 × Fin 7 → Bool) :
  ∃ (r1 r2 : Fin 3) (c1 c2 : Fin 7), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid (r1, c1) = grid (r1, c2) ∧ grid (r2, c1) = grid (r2, c2) :=
by
  sorry

end rectangle_same_color_l206_20616


namespace triangle_side_lengths_relationship_l206_20641

variable {a b c : ℝ}

def is_quadratic_mean (a b c : ℝ) : Prop :=
  (2 * b^2 = a^2 + c^2)

def is_geometric_mean (a b c : ℝ) : Prop :=
  (b * a = c^2)

theorem triangle_side_lengths_relationship (a b c : ℝ) :
  (is_quadratic_mean a b c ∧ is_geometric_mean a b c) → 
  ∃ a b c, (2 * b^2 = a^2 + c^2) ∧ (b * a = c^2) :=
sorry

end triangle_side_lengths_relationship_l206_20641


namespace benjamin_distance_l206_20653

def speed := 10  -- Speed in kilometers per hour
def time := 8    -- Time in hours

def distance (s t : ℕ) := s * t  -- Distance formula

theorem benjamin_distance : distance speed time = 80 :=
by
  -- proof omitted
  sorry

end benjamin_distance_l206_20653


namespace ceil_inequality_range_x_solve_eq_l206_20606

-- Definition of the mathematical ceiling function to comply with the condition a).
def ceil (a : ℚ) : ℤ := ⌈a⌉

-- Condition 1: Relationship between m and ⌈m⌉.
theorem ceil_inequality (m : ℚ) : m ≤ ceil m ∧ ceil m < m + 1 :=
sorry

-- Part 2.1: Range of x given {3x + 2} = 8.
theorem range_x (x : ℚ) (h : ceil (3 * x + 2) = 8) : 5 / 3 < x ∧ x ≤ 2 :=
sorry

-- Part 2.2: Solving {3x - 2} = 2x + 1/2
theorem solve_eq (x : ℚ) (h : ceil (3 * x - 2) = 2 * x + 1 / 2) : x = 7 / 4 ∨ x = 9 / 4 :=
sorry

end ceil_inequality_range_x_solve_eq_l206_20606


namespace larger_number_l206_20681

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l206_20681


namespace n_pow4_sub_n_pow2_divisible_by_12_l206_20630

theorem n_pow4_sub_n_pow2_divisible_by_12 (n : ℤ) (h : n > 1) : 12 ∣ (n^4 - n^2) :=
by sorry

end n_pow4_sub_n_pow2_divisible_by_12_l206_20630


namespace triple_hash_90_l206_20629

def hash (N : ℝ) : ℝ := 0.3 * N + 2

theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 :=
by
  sorry

end triple_hash_90_l206_20629


namespace carrots_picked_next_day_l206_20640

theorem carrots_picked_next_day :
  ∀ (initial_picked thrown_out additional_picked total : ℕ),
    initial_picked = 48 →
    thrown_out = 11 →
    total = 52 →
    additional_picked = total - (initial_picked - thrown_out) →
    additional_picked = 15 :=
by
  intros initial_picked thrown_out additional_picked total h_ip h_to h_total h_ap
  sorry

end carrots_picked_next_day_l206_20640


namespace trig_identity_simplification_l206_20696

theorem trig_identity_simplification (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end trig_identity_simplification_l206_20696


namespace determine_b_when_lines_parallel_l206_20682

theorem determine_b_when_lines_parallel (b : ℝ) : 
  (∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = (b + 9) * x) → b = -6 :=
by
  sorry

end determine_b_when_lines_parallel_l206_20682


namespace sequence_value_l206_20680

noncomputable def f : ℝ → ℝ := sorry

theorem sequence_value :
  ∃ a : ℕ → ℝ, 
    (a 1 = f 1) ∧ 
    (∀ n : ℕ, f (a (n + 1)) = f (2 * a n + 1)) ∧ 
    (a 2017 = 2 ^ 2016 - 1) := sorry

end sequence_value_l206_20680


namespace largest_consecutive_odd_number_sum_is_27_l206_20699

theorem largest_consecutive_odd_number_sum_is_27
  (a b c : ℤ)
  (h1 : a + b + c = 75)
  (h2 : c - a = 4)
  (h3 : a % 2 = 1)
  (h4 : b % 2 = 1)
  (h5 : c % 2 = 1) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_sum_is_27_l206_20699


namespace domain_of_g_cauchy_schwarz_inequality_l206_20648

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Question 1: Prove the domain of g(x) = log(f(x) - 2) is {x | 0.5 < x < 2.5}
theorem domain_of_g : {x : ℝ | 0.5 < x ∧ x < 2.5} = {x : ℝ | 0.5 < x ∧ x < 2.5} :=
by
  sorry

-- Minimum value of f(x)
def m : ℝ := 1

-- Question 2: Prove a^2 + b^2 + c^2 ≥ 1/3 given a + b + c = m
theorem cauchy_schwarz_inequality (a b c : ℝ) (h : a + b + c = m) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end domain_of_g_cauchy_schwarz_inequality_l206_20648


namespace no_solution_system_of_equations_l206_20656

theorem no_solution_system_of_equations :
  ¬ (∃ (x y : ℝ),
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0) :=
    by
      sorry

end no_solution_system_of_equations_l206_20656


namespace algebra_expression_opposite_l206_20676

theorem algebra_expression_opposite (a : ℚ) :
  3 * a + 1 = -(3 * (a - 1)) → a = 1 / 3 :=
by
  intro h
  sorry

end algebra_expression_opposite_l206_20676


namespace interest_time_period_l206_20612

-- Define the constants given in the problem
def principal : ℝ := 4000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def interest_difference : ℝ := 480

-- Define the time period T
def time_period : ℝ := 2

-- Define a proof statement
theorem interest_time_period :
  (principal * rate1 * time_period) - (principal * rate2 * time_period) = interest_difference :=
by {
  -- We skip the proof since it's not required by the problem statement
  sorry
}

end interest_time_period_l206_20612


namespace length_of_second_train_l206_20626

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (clear_time_seconds : ℝ)
  (relative_speed_kmph : ℝ) :
  speed_first_train_kmph + speed_second_train_kmph = relative_speed_kmph →
  relative_speed_kmph * (5 / 18) * clear_time_seconds = length_first_train + 280 :=
by
  let length_first_train := 120
  let speed_first_train_kmph := 42
  let speed_second_train_kmph := 30
  let clear_time_seconds := 20
  let relative_speed_kmph := 72
  sorry

end length_of_second_train_l206_20626


namespace value_of_k_l206_20684

theorem value_of_k {k : ℝ} :
  (∀ x : ℝ, (x^2 + k * x + 24 > 0) ↔ (x < -6 ∨ x > 4)) →
  k = 2 :=
by
  sorry

end value_of_k_l206_20684


namespace wechat_payment_meaning_l206_20625

theorem wechat_payment_meaning (initial_balance after_receive_balance : ℝ)
  (recv_amount sent_amount : ℝ)
  (h1 : recv_amount = 200)
  (h2 : initial_balance + recv_amount = after_receive_balance)
  (h3 : after_receive_balance - sent_amount = initial_balance)
  : sent_amount = 200 :=
by
  -- starting the proof becomes irrelevant
  sorry

end wechat_payment_meaning_l206_20625


namespace correct_vector_equation_l206_20624

variables {V : Type*} [AddCommGroup V]

variables (A B C: V)

theorem correct_vector_equation : 
  (A - B) - (B - C) = A - C :=
sorry

end correct_vector_equation_l206_20624


namespace walking_distance_l206_20672

theorem walking_distance (D : ℕ) (h : D / 15 = (D + 60) / 30) : D = 60 :=
by
  sorry

end walking_distance_l206_20672


namespace color_cartridge_cost_l206_20690

theorem color_cartridge_cost :
  ∃ C : ℝ, 
  (1 * 27) + (3 * C) = 123 ∧ C = 32 :=
by
  sorry

end color_cartridge_cost_l206_20690


namespace number_of_units_sold_l206_20654

theorem number_of_units_sold (p : ℕ) (c : ℕ) (k : ℕ) (h : p * c = k) (h₁ : c = 800) (h₂ : k = 8000) : p = 10 :=
by
  sorry

end number_of_units_sold_l206_20654


namespace algebra_expression_value_l206_20600

theorem algebra_expression_value (x y : ℝ)
  (h1 : x + y = 3)
  (h2 : x * y = 1) :
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 :=
by
  sorry

end algebra_expression_value_l206_20600


namespace find_number_of_violas_l206_20697

theorem find_number_of_violas (cellos : ℕ) (pairs : ℕ) (probability : ℚ) 
    (h1 : cellos = 800) 
    (h2 : pairs = 100) 
    (h3 : probability = 0.00020833333333333335) : 
    ∃ V : ℕ, V = 600 := 
by 
    sorry

end find_number_of_violas_l206_20697


namespace prob_A_and_B_truth_is_0_48_l206_20638

-- Conditions: Define the probabilities
def prob_A_truth : ℝ := 0.8
def prob_B_truth : ℝ := 0.6

-- Target: Define the probability that both A and B tell the truth at the same time.
def prob_A_and_B_truth : ℝ := prob_A_truth * prob_B_truth

-- Statement: Prove that the probability that both A and B tell the truth at the same time is 0.48.
theorem prob_A_and_B_truth_is_0_48 : prob_A_and_B_truth = 0.48 := by
  sorry

end prob_A_and_B_truth_is_0_48_l206_20638


namespace lisa_needs_4_weeks_to_eat_all_candies_l206_20660

-- Define the number of candies Lisa has initially.
def candies_initial : ℕ := 72

-- Define the number of candies Lisa eats per week based on the given conditions.
def candies_per_week : ℕ := (3 * 2) + (2 * 2) + (4 * 2) + 1

-- Define the number of weeks it takes for Lisa to eat all the candies.
def weeks_to_eat_all_candies (candies : ℕ) (weekly_candies : ℕ) : ℕ := 
  (candies + weekly_candies - 1) / weekly_candies

-- The theorem statement that proves Lisa needs 4 weeks to eat all 72 candies.
theorem lisa_needs_4_weeks_to_eat_all_candies :
  weeks_to_eat_all_candies candies_initial candies_per_week = 4 :=
by
  sorry

end lisa_needs_4_weeks_to_eat_all_candies_l206_20660


namespace fraction_comparison_l206_20659

theorem fraction_comparison :
  (2 : ℝ) * (4 : ℝ) > (7 : ℝ) → (4 / 7 : ℝ) > (1 / 2 : ℝ) :=
by
  sorry

end fraction_comparison_l206_20659


namespace sally_total_score_l206_20634

theorem sally_total_score :
  ∀ (correct incorrect unanswered : ℕ) (score_correct score_incorrect : ℝ),
    correct = 17 →
    incorrect = 8 →
    unanswered = 5 →
    score_correct = 1 →
    score_incorrect = -0.25 →
    (correct * score_correct +
     incorrect * score_incorrect +
     unanswered * 0) = 15 :=
by
  intros correct incorrect unanswered score_correct score_incorrect
  intros h_corr h_incorr h_unan h_sc h_si
  sorry

end sally_total_score_l206_20634


namespace simplify_expression_l206_20601

theorem simplify_expression (x : ℝ) :
  x - 3 * (1 + x) + 4 * (1 - x)^2 - 5 * (1 + 3 * x) = 4 * x^2 - 25 * x - 4 := by
  sorry

end simplify_expression_l206_20601


namespace plane_angle_divides_cube_l206_20667

noncomputable def angle_between_planes (m n : ℕ) (h : m ≤ n) : ℝ :=
  Real.arctan (2 * m / (m + n))

theorem plane_angle_divides_cube (m n : ℕ) (h : m ≤ n) :
  ∃ α, α = angle_between_planes m n h :=
sorry

end plane_angle_divides_cube_l206_20667


namespace profit_percentage_on_cost_price_l206_20615

theorem profit_percentage_on_cost_price (CP MP SP : ℝ)
    (h1 : CP = 100)
    (h2 : MP = 131.58)
    (h3 : SP = 0.95 * MP) :
    ((SP - CP) / CP) * 100 = 25 :=
by
  -- Sorry to skip the proof
  sorry

end profit_percentage_on_cost_price_l206_20615


namespace find_b_in_quadratic_eqn_l206_20677

theorem find_b_in_quadratic_eqn :
  ∃ (b : ℝ), ∃ (p : ℝ), 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → 
  b = 8 * Real.sqrt 3 :=
by 
  sorry

end find_b_in_quadratic_eqn_l206_20677


namespace replace_movies_cost_l206_20620

theorem replace_movies_cost
  (num_movies : ℕ)
  (trade_in_value_per_vhs : ℕ)
  (cost_per_dvd : ℕ)
  (h1 : num_movies = 100)
  (h2 : trade_in_value_per_vhs = 2)
  (h3 : cost_per_dvd = 10):
  (cost_per_dvd - trade_in_value_per_vhs) * num_movies = 800 :=
by sorry

end replace_movies_cost_l206_20620


namespace find_number_l206_20610

-- Define the main problem statement
theorem find_number (x : ℝ) (h : 0.50 * x = 0.80 * 150 + 80) : x = 400 := by
  sorry

end find_number_l206_20610


namespace problem_statement_l206_20647

theorem problem_statement : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end problem_statement_l206_20647


namespace oscar_leap_longer_than_elmer_stride_l206_20658

theorem oscar_leap_longer_than_elmer_stride :
  ∀ (elmer_strides_per_gap oscar_leaps_per_gap gaps_between_poles : ℕ)
    (total_distance : ℝ),
  elmer_strides_per_gap = 60 →
  oscar_leaps_per_gap = 16 →
  gaps_between_poles = 60 →
  total_distance = 7920 →
  let elmer_stride_length := total_distance / (elmer_strides_per_gap * gaps_between_poles)
  let oscar_leap_length := total_distance / (oscar_leaps_per_gap * gaps_between_poles)
  oscar_leap_length - elmer_stride_length = 6.05 :=
by
  intros
  sorry

end oscar_leap_longer_than_elmer_stride_l206_20658


namespace greatest_leftover_cookies_l206_20663

theorem greatest_leftover_cookies (n : ℕ) : ∃ k, k ≤ n ∧ k % 8 = 7 := sorry

end greatest_leftover_cookies_l206_20663


namespace number_divisible_by_k_cube_l206_20675

theorem number_divisible_by_k_cube (k : ℕ) (h : k = 42) : ∃ n, (k^3) % n = 0 ∧ n = 74088 := by
  sorry

end number_divisible_by_k_cube_l206_20675


namespace Patel_family_theme_park_expenses_l206_20609

def regular_ticket_price : ℝ := 12.5
def senior_discount : ℝ := 0.8
def child_discount : ℝ := 0.6
def senior_ticket_price := senior_discount * regular_ticket_price
def child_ticket_price := child_discount * regular_ticket_price

theorem Patel_family_theme_park_expenses :
  (2 * senior_ticket_price + 2 * child_ticket_price + 4 * regular_ticket_price) = 85 := by
  sorry

end Patel_family_theme_park_expenses_l206_20609


namespace base_is_16_l206_20605

noncomputable def base_y_eq : Prop := ∃ base : ℕ, base ^ 8 = 4 ^ 16

theorem base_is_16 (base : ℕ) (h₁ : base ^ 8 = 4 ^ 16) : base = 16 :=
by
  sorry  -- Proof goes here

end base_is_16_l206_20605


namespace rose_can_afford_l206_20688

noncomputable def total_cost_before_discount : ℝ :=
  2.40 + 9.20 + 6.50 + 12.25 + 4.75

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def budget : ℝ :=
  30.00

noncomputable def remaining_budget : ℝ :=
  budget - total_cost_after_discount

theorem rose_can_afford :
  remaining_budget = 0.165 :=
by
  -- proof goes here
  sorry

end rose_can_afford_l206_20688


namespace length_of_platform_l206_20698

theorem length_of_platform (length_of_train : ℕ) (speed_kmph : ℕ) (time_s : ℕ) (L : ℕ) :
  length_of_train = 160 → speed_kmph = 72 → time_s = 25 → (L = 340) :=
by
  sorry

end length_of_platform_l206_20698


namespace percent_of_day_is_hours_l206_20692

theorem percent_of_day_is_hours (h : ℝ) (day_hours : ℝ) (percent : ℝ) 
  (day_hours_def : day_hours = 24)
  (percent_def : percent = 29.166666666666668) :
  h = 7 :=
by
  sorry

end percent_of_day_is_hours_l206_20692


namespace problems_per_page_l206_20608

theorem problems_per_page (total_problems finished_problems remaining_pages problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : total_problems - finished_problems = 14)
  (h5 : 14 = remaining_pages * problems_per_page) :
  problems_per_page = 7 := 
by
  sorry

end problems_per_page_l206_20608


namespace weight_problem_l206_20645

variable (M T : ℕ)

theorem weight_problem
  (h1 : 220 = 3 * M + 10)
  (h2 : T = 2 * M)
  (h3 : 2 * T = 220) :
  M = 70 ∧ T = 140 :=
by
  sorry

end weight_problem_l206_20645


namespace product_of_primes_95_l206_20666

theorem product_of_primes_95 (p q : Nat) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p + q = 95) : p * q = 178 := sorry

end product_of_primes_95_l206_20666


namespace find_t_l206_20642

variable {a b c r s t : ℝ}

-- Conditions from part a)
def first_polynomial_has_roots (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c)) : Prop :=
  ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = 0 → x = a ∨ x = b ∨ x = c

def second_polynomial_has_roots (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) : Prop :=
  ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = 0 → x = (a + b) ∨ x = (b + c) ∨ x = (c + a)

-- Translate problem (find t) with conditions
theorem find_t (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c))
    (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a)))
    (sum_roots : a + b + c = -3) 
    (prod_roots : a * b * c = -11):
  t = 23 := 
sorry

end find_t_l206_20642


namespace problem1_problem2_l206_20632

-- Proof Problem 1: Prove that (x-y)^2 - (x+y)(x-y) = -2xy + 2y^2
theorem problem1 (x y : ℝ) : (x - y) ^ 2 - (x + y) * (x - y) = -2 * x * y + 2 * y ^ 2 := 
by
  sorry

-- Proof Problem 2: Prove that (12a^2b - 6ab^2) / (-3ab) = -4a + 2b
theorem problem2 (a b : ℝ) (h : -3 * a * b ≠ 0) : (12 * a^2 * b - 6 * a * b^2) / (-3 * a * b) = -4 * a + 2 * b := 
by
  sorry

end problem1_problem2_l206_20632


namespace susan_walked_9_miles_l206_20694

theorem susan_walked_9_miles (E S : ℕ) (h1 : E + S = 15) (h2 : E = S - 3) : S = 9 :=
by
  sorry

end susan_walked_9_miles_l206_20694


namespace pauls_total_cost_is_252_l206_20613

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l206_20613


namespace jacket_final_price_l206_20671

/-- 
The initial price of the jacket is $20, 
the first discount is 40%, and the second discount is 25%. 
We need to prove that the final price of the jacket is $9.
-/
theorem jacket_final_price :
  let initial_price := 20
  let first_discount := 0.40
  let second_discount := 0.25
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 9 :=
by
  sorry

end jacket_final_price_l206_20671


namespace unique_zero_iff_a_in_range_l206_20650

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem unique_zero_iff_a_in_range (a : ℝ) :
  (∃ x0 : ℝ, f a x0 = 0 ∧ (∀ x1 : ℝ, f a x1 = 0 → x1 = x0) ∧ x0 > 0) ↔ a < -2 :=
by sorry

end unique_zero_iff_a_in_range_l206_20650


namespace polynomial_sum_equals_one_l206_20665

theorem polynomial_sum_equals_one (a a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (2*x + 1)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_sum_equals_one_l206_20665


namespace tan_pi_div_4_add_alpha_l206_20687

theorem tan_pi_div_4_add_alpha (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : 
  Real.tan (π / 4 + α) = -3 :=
by
  sorry

end tan_pi_div_4_add_alpha_l206_20687


namespace question_1_part_1_question_1_part_2_question_2_l206_20622

universe u

variables (U : Type u) [PartialOrder U]
noncomputable def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}
noncomputable def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
noncomputable def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a }

theorem question_1_part_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} :=
sorry

theorem question_1_part_2 : B ∪ (Set.compl A) = {x | x ≤ 5 ∨ x ≥ 9} :=
sorry

theorem question_2 (a : ℝ) (h : C a ∪ (Set.compl B) = Set.univ) : a ≤ -3 :=
sorry

end question_1_part_1_question_1_part_2_question_2_l206_20622


namespace quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l206_20686

theorem quadratic_has_negative_root_sufficiency 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) → (a < 0) :=
sorry

theorem quadratic_has_negative_root_necessity 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0) :=
sorry

end quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l206_20686


namespace basic_astrophysics_degrees_l206_20691

open Real

theorem basic_astrophysics_degrees :
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  basic_astrophysics_percentage / 100 * circle_degrees = 43.2 :=
by
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  exact sorry

end basic_astrophysics_degrees_l206_20691


namespace ratio_of_terms_l206_20679

theorem ratio_of_terms (a_n b_n : ℕ → ℕ) (S_n T_n : ℕ → ℕ) :
  (∀ n, S_n n = (n * (2 * a_n n - (n - 1))) / 2) → 
  (∀ n, T_n n = (n * (2 * b_n n - (n - 1))) / 2) → 
  (∀ n, S_n n / T_n n = (n + 3) / (2 * n + 1)) → 
  S_n 6 / T_n 6 = 14 / 23 :=
by
  sorry

end ratio_of_terms_l206_20679


namespace derek_bought_more_cars_l206_20674

-- Define conditions
variables (d₆ c₆ d₁₆ c₁₆ : ℕ)

-- Given conditions
def initial_conditions :=
  (d₆ = 90) ∧
  (d₆ = 3 * c₆) ∧
  (d₁₆ = 120) ∧
  (c₁₆ = 2 * d₁₆)

-- Prove the number of cars Derek bought in ten years
theorem derek_bought_more_cars (h : initial_conditions d₆ c₆ d₁₆ c₁₆) : c₁₆ - c₆ = 210 :=
by sorry

end derek_bought_more_cars_l206_20674


namespace cost_of_cucumbers_l206_20631

theorem cost_of_cucumbers (C : ℝ) (h1 : ∀ (T : ℝ), T = 0.80 * C)
  (h2 : 2 * (0.80 * C) + 3 * C = 23) : C = 5 := by
  sorry

end cost_of_cucumbers_l206_20631


namespace initial_oranges_l206_20639

open Nat

theorem initial_oranges (initial_oranges: ℕ) (eaten_oranges: ℕ) (stolen_oranges: ℕ) (returned_oranges: ℕ) (current_oranges: ℕ):
  eaten_oranges = 10 → 
  stolen_oranges = (initial_oranges - eaten_oranges) / 2 →
  returned_oranges = 5 →
  current_oranges = 30 →
  initial_oranges - eaten_oranges - stolen_oranges + returned_oranges = current_oranges →
  initial_oranges = 60 :=
by
  sorry

end initial_oranges_l206_20639


namespace root_in_interval_l206_20651

def f (x : ℝ) : ℝ := x^3 + 5 * x^2 - 3 * x + 1

theorem root_in_interval : ∃ A B : ℤ, B = A + 1 ∧ (∃ ξ : ℝ, f ξ = 0 ∧ (A : ℝ) < ξ ∧ ξ < (B : ℝ)) ∧ A = -6 ∧ B = -5 :=
by
  sorry

end root_in_interval_l206_20651


namespace remainder_div_3005_95_l206_20695

theorem remainder_div_3005_95 : 3005 % 95 = 60 := 
by {
  sorry
}

end remainder_div_3005_95_l206_20695


namespace intersection_complement_l206_20637

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5 }
def C : Set ℝ := {x | x ≤ 2 ∨ x ≥ 5 }

theorem intersection_complement :
  (A ∩ C) = {1, 2, 5, 6} :=
by sorry

end intersection_complement_l206_20637


namespace problem_statement_l206_20607

theorem problem_statement :
  let a := (List.range (60 / 12)).card
  let b := (List.range (60 / Nat.lcm (Nat.lcm 2 3) 4)).card
  (a - b) ^ 3 = 0 :=
by
  sorry

end problem_statement_l206_20607


namespace find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l206_20670

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l206_20670


namespace complex_z24_condition_l206_20614

open Complex

theorem complex_z24_condition (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (5 * π / 180)) : 
  z^24 + z⁻¹^24 = -1 := sorry

end complex_z24_condition_l206_20614
