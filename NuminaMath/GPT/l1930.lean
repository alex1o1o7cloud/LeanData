import Mathlib

namespace math_proof_problem_l1930_193079

open Nat

noncomputable def number_of_pairs := 
  let N := 20^19
  let num_divisors := (38 + 1) * (19 + 1)
  let total_pairs := num_divisors * num_divisors
  let ab_dividing_pairs := 780 * 210
  total_pairs - ab_dividing_pairs

theorem math_proof_problem : number_of_pairs = 444600 := 
  by exact sorry

end math_proof_problem_l1930_193079


namespace cost_per_mile_l1930_193080

theorem cost_per_mile (x : ℝ) (daily_fee : ℝ) (daily_budget : ℝ) (max_miles : ℝ)
  (h1 : daily_fee = 50)
  (h2 : daily_budget = 88)
  (h3 : max_miles = 190)
  (h4 : daily_budget = daily_fee + x * max_miles) :
  x = 0.20 :=
by
  sorry

end cost_per_mile_l1930_193080


namespace jake_sold_tuesday_correct_l1930_193053

def jake_initial_pieces : ℕ := 80
def jake_sold_monday : ℕ := 15
def jake_remaining_wednesday : ℕ := 7

def pieces_sold_tuesday (initial : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) : ℕ :=
  initial - sold_monday - remaining_wednesday

theorem jake_sold_tuesday_correct :
  pieces_sold_tuesday jake_initial_pieces jake_sold_monday jake_remaining_wednesday = 58 :=
by
  unfold pieces_sold_tuesday
  norm_num
  sorry

end jake_sold_tuesday_correct_l1930_193053


namespace clea_ride_time_l1930_193039

noncomputable def walk_down_stopped (x y : ℝ) : Prop := 90 * x = y
noncomputable def walk_down_moving (x y k : ℝ) : Prop := 30 * (x + k) = y
noncomputable def ride_time (y k t : ℝ) : Prop := t = y / k

theorem clea_ride_time (x y k t : ℝ) (h1 : walk_down_stopped x y) (h2 : walk_down_moving x y k) :
  ride_time y k t → t = 45 :=
sorry

end clea_ride_time_l1930_193039


namespace time_between_rings_is_288_minutes_l1930_193015

def intervals_between_rings (total_rings : ℕ) (total_minutes : ℕ) : ℕ := 
  let intervals := total_rings - 1
  total_minutes / intervals

theorem time_between_rings_is_288_minutes (total_minutes_in_day total_rings : ℕ) 
  (h1 : total_minutes_in_day = 1440) (h2 : total_rings = 6) : 
  intervals_between_rings total_rings total_minutes_in_day = 288 := 
by 
  sorry

end time_between_rings_is_288_minutes_l1930_193015


namespace sequence_term_value_l1930_193070

theorem sequence_term_value :
  ∃ (a : ℕ → ℚ), a 1 = 2 ∧ (∀ n, a (n + 1) = a n + 1 / 2) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_value_l1930_193070


namespace inequality_holds_iff_m_eq_n_l1930_193026

theorem inequality_holds_iff_m_eq_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∀ (α β : ℝ), 
    ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ 
    ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) ↔ m = n :=
by
  sorry

end inequality_holds_iff_m_eq_n_l1930_193026


namespace find_x_solutions_l1930_193030

theorem find_x_solutions :
  ∀ {x : ℝ}, (x = (1/x) + (-x)^2 + 3) → (x = -1 ∨ x = 1) :=
by
  sorry

end find_x_solutions_l1930_193030


namespace edge_length_of_cube_l1930_193091

/--
Given:
1. A cuboid with base width of 70 cm, base length of 40 cm, and height of 150 cm.
2. A cube-shaped cabinet whose volume is 204,000 cm³ smaller than that of the cuboid.

Prove that one edge of the cube-shaped cabinet is 60 cm.
-/
theorem edge_length_of_cube (W L H V_diff : ℝ) (cuboid_vol : ℝ) (cube_vol : ℝ) (edge : ℝ) :
  W = 70 ∧ L = 40 ∧ H = 150 ∧ V_diff = 204000 ∧ 
  cuboid_vol = W * L * H ∧ cube_vol = cuboid_vol - V_diff ∧ edge ^ 3 = cube_vol -> 
  edge = 60 :=
by
  sorry

end edge_length_of_cube_l1930_193091


namespace Q_difference_l1930_193042

def Q (x n : ℕ) : ℕ :=
  (Finset.range (10^n)).sum (λ k => x / (k + 1))

theorem Q_difference (n : ℕ) : 
  Q (10^n) n - Q (10^n - 1) n = (n + 1)^2 :=
by
  sorry

end Q_difference_l1930_193042


namespace sum_of_legs_equal_l1930_193084

theorem sum_of_legs_equal
  (a b c d e f g h : ℝ)
  (x y : ℝ)
  (h_similar_shaded1 : a = a * x ∧ b = a * y)
  (h_similar_shaded2 : c = c * x ∧ d = c * y)
  (h_similar_shaded3 : e = e * x ∧ f = e * y)
  (h_similar_shaded4 : g = g * x ∧ h = g * y)
  (h_similar_unshaded1 : h = h * x ∧ a = h * y)
  (h_similar_unshaded2 : b = b * x ∧ c = b * y)
  (h_similar_unshaded3 : d = d * x ∧ e = d * y)
  (h_similar_unshaded4 : f = f * x ∧ g = f * y)
  (x_non_zero : x ≠ 0) (y_non_zero : y ≠ 0) : 
  (a * y + b + c * x) + (c * y + d + e * x) + (e * y + f + g * x) + (g * y + h + a * x) 
  = (h * x + a + b * y) + (b * x + c + d * y) + (d * x + e + f * y) + (f * x + g + h * y) :=
sorry

end sum_of_legs_equal_l1930_193084


namespace students_playing_both_l1930_193048

theorem students_playing_both
    (total_students baseball_team hockey_team : ℕ)
    (h1 : total_students = 36)
    (h2 : baseball_team = 25)
    (h3 : hockey_team = 19)
    (h4 : total_students = baseball_team + hockey_team - students_both) :
    students_both = 8 := by
  sorry

end students_playing_both_l1930_193048


namespace probability_different_suits_correct_l1930_193065

-- Definitions based on conditions
def cards_in_deck : ℕ := 52
def cards_picked : ℕ := 3
def first_card_suit_not_matter : Prop := True
def second_card_different_suit : Prop := True
def third_card_different_suit : Prop := True

-- Definition of the probability function
def probability_different_suits (cards_total : ℕ) (cards_picked : ℕ) : Rat :=
  let first_card_prob := 1
  let second_card_prob := 39 / 51
  let third_card_prob := 26 / 50
  first_card_prob * second_card_prob * third_card_prob

-- The theorem statement to prove the probability each card is of a different suit
theorem probability_different_suits_correct :
  probability_different_suits cards_in_deck cards_picked = 169 / 425 :=
by
  -- Proof should be written here
  sorry

end probability_different_suits_correct_l1930_193065


namespace melony_profit_l1930_193050

theorem melony_profit (profit_3_shirts : ℝ)
  (profit_2_sandals : ℝ)
  (h1 : profit_3_shirts = 21)
  (h2 : profit_2_sandals = 4 * 21) : profit_3_shirts / 3 * 7 + profit_2_sandals / 2 * 3 = 175 := 
by 
  sorry

end melony_profit_l1930_193050


namespace max_possible_number_under_operations_l1930_193069

theorem max_possible_number_under_operations :
  ∀ x : ℕ, x < 17 →
    ∀ n : ℕ, (∃ k : ℕ, k < n ∧ (x + 17 * k) % 19 = 0) →
    ∃ m : ℕ, m = (304 : ℕ) :=
sorry

end max_possible_number_under_operations_l1930_193069


namespace triangle_angle_sum_acute_l1930_193098

theorem triangle_angle_sum_acute (x : ℝ) (h1 : 60 + 70 + x = 180) (h2 : x ≠ 60 ∧ x ≠ 70) :
  x = 50 ∧ (60 < 90 ∧ 70 < 90 ∧ x < 90) := by
  sorry

end triangle_angle_sum_acute_l1930_193098


namespace soak_time_l1930_193016

/-- 
Bill needs to soak his clothes for 4 minutes to get rid of each grass stain.
His clothes have 3 grass stains and 1 marinara stain.
The total soaking time is 19 minutes.
Prove that the number of minutes needed to soak for each marinara stain is 7.
-/
theorem soak_time (m : ℕ) (grass_stain_time : ℕ) (num_grass_stains : ℕ) (num_marinara_stains : ℕ) (total_time : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_time = 19) :
  m = 7 :=
by sorry

end soak_time_l1930_193016


namespace max_length_sequence_l1930_193036

def seq_term (n : ℕ) (y : ℤ) : ℤ :=
  match n with
  | 0 => 2000
  | 1 => y
  | k + 2 => seq_term (k + 1) y - seq_term k y

theorem max_length_sequence (y : ℤ) :
  1200 < y ∧ y < 1334 ∧ (∀ n, seq_term n y ≥ 0 ∨ seq_term (n + 1) y < 0) ↔ y = 1333 :=
by
  sorry

end max_length_sequence_l1930_193036


namespace minimum_value_l1930_193052

def minimum_value_problem (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : Prop :=
  ∃ c : ℝ, c = (1 / (a + 1) + 4 / (b + 1)) ∧ c = 9 / 4

theorem minimum_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : 
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 :=
by 
  -- Proof goes here
  sorry

end minimum_value_l1930_193052


namespace eggs_remainder_l1930_193033

def daniel_eggs := 53
def eliza_eggs := 68
def fiona_eggs := 26
def george_eggs := 47
def total_eggs := daniel_eggs + eliza_eggs + fiona_eggs + george_eggs

theorem eggs_remainder :
  total_eggs % 15 = 14 :=
by
  sorry

end eggs_remainder_l1930_193033


namespace cylinder_volume_increase_factor_l1930_193014

theorem cylinder_volume_increase_factor
    (π : Real)
    (r h : Real)
    (V_original : Real := π * r^2 * h)
    (new_height : Real := 3 * h)
    (new_radius : Real := 4 * r)
    (V_new : Real := π * (new_radius)^2 * new_height) :
    V_new / V_original = 48 :=
by
  sorry

end cylinder_volume_increase_factor_l1930_193014


namespace find_value_l1930_193086

theorem find_value
  (y1 y2 y3 y4 y5 : ℝ)
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 = 20)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 = 150) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 = 336 :=
by
  sorry

end find_value_l1930_193086


namespace sum_of_squares_of_roots_l1930_193024

theorem sum_of_squares_of_roots :
  let a := 10
  let b := 16
  let c := -18
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots ^ 2 - 2 * product_of_roots = 244 / 25 := by
  sorry

end sum_of_squares_of_roots_l1930_193024


namespace Kaleb_candies_l1930_193028

theorem Kaleb_candies 
  (tickets_whack_a_mole : ℕ) 
  (tickets_skee_ball : ℕ) 
  (candy_cost : ℕ)
  (h1 : tickets_whack_a_mole = 8)
  (h2 : tickets_skee_ball = 7)
  (h3 : candy_cost = 5) : 
  (tickets_whack_a_mole + tickets_skee_ball) / candy_cost = 3 := 
by
  sorry

end Kaleb_candies_l1930_193028


namespace no_a_b_exist_no_a_b_c_exist_l1930_193064

-- Part (a):
theorem no_a_b_exist (a b : ℕ) (h0 : 0 < a) (h1 : 0 < b) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n = k^2) :=
sorry

-- Part (b):
theorem no_a_b_c_exist (a b c : ℕ) (h0 : 0 < a) (h1 : 0 < b) (h2 : 0 < c) :
  ¬ (∀ n : ℕ, 0 < n → ∃ k : ℕ, a * 2^n + b * 5^n + c = k^2) :=
sorry

end no_a_b_exist_no_a_b_c_exist_l1930_193064


namespace feathers_already_have_l1930_193008

-- Given conditions
def total_feathers : Nat := 900
def feathers_still_needed : Nat := 513

-- Prove that the number of feathers Charlie already has is 387
theorem feathers_already_have : (total_feathers - feathers_still_needed) = 387 := by
  sorry

end feathers_already_have_l1930_193008


namespace courtyard_width_is_14_l1930_193017

-- Given conditions
def length_courtyard := 24   -- 24 meters
def num_bricks := 8960       -- Total number of bricks

@[simp]
def brick_length_m : ℝ := 0.25  -- 25 cm in meters
@[simp]
def brick_width_m : ℝ := 0.15   -- 15 cm in meters

-- Correct answer
def width_courtyard : ℝ := 14

-- Prove that the width of the courtyard is 14 meters
theorem courtyard_width_is_14 : 
  (length_courtyard * width_courtyard) = (num_bricks * (brick_length_m * brick_width_m)) :=
by
  -- Lean proof will go here
  sorry

end courtyard_width_is_14_l1930_193017


namespace quadratic_inequality_l1930_193066

theorem quadratic_inequality (t x₁ x₂ : ℝ) (α β : ℝ)
  (ht : (2 * x₁^2 - t * x₁ - 2 = 0) ∧ (2 * x₂^2 - t * x₂ - 2 = 0))
  (hx : α ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ β)
  (hαβ : α < β)
  (roots : α + β = t / 2 ∧ α * β = -1) :
  4*x₁*x₂ - t*(x₁ + x₂) - 4 < 0 := 
sorry

end quadratic_inequality_l1930_193066


namespace number_of_repeating_decimals_l1930_193093

open Nat

theorem number_of_repeating_decimals :
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 15) → (¬ ∃ k : ℕ, k * 18 = n) :=
by
  intros n h
  sorry

end number_of_repeating_decimals_l1930_193093


namespace solve_equation_l1930_193090

theorem solve_equation :
  ∀ (x : ℝ), x * (3 * x + 6) = 7 * (3 * x + 6) → (x = 7 ∨ x = -2) :=
by
  intro x
  sorry

end solve_equation_l1930_193090


namespace original_number_is_500_l1930_193056

theorem original_number_is_500 (x : ℝ) (h1 : x * 1.3 = 650) : x = 500 :=
sorry

end original_number_is_500_l1930_193056


namespace max_a_no_lattice_points_l1930_193013

theorem max_a_no_lattice_points :
  ∀ (m : ℝ), (1 / 3) < m → m < (17 / 51) →
  ¬ ∃ (x : ℕ) (y : ℕ), 0 < x ∧ x ≤ 50 ∧ y = m * x + 3 := 
by
  sorry

end max_a_no_lattice_points_l1930_193013


namespace wellington_population_l1930_193071

theorem wellington_population 
  (W P L : ℕ)
  (h1 : P = 7 * W)
  (h2 : P = L + 800)
  (h3 : P + L = 11800) : 
  W = 900 :=
by
  sorry

end wellington_population_l1930_193071


namespace rate_of_interest_per_annum_l1930_193057

theorem rate_of_interest_per_annum (R : ℝ) : 
  (5000 * R * 2 / 100) + (3000 * R * 4 / 100) = 1540 → 
  R = 7 := 
by {
  sorry
}

end rate_of_interest_per_annum_l1930_193057


namespace find_x_from_triangle_area_l1930_193002

theorem find_x_from_triangle_area :
  ∀ (x : ℝ), x > 0 ∧ (1 / 2) * x * 3 * x = 96 → x = 8 :=
by
  intros x hx
  -- The proof goes here
  sorry

end find_x_from_triangle_area_l1930_193002


namespace pentagon_arithmetic_progression_angle_l1930_193029

theorem pentagon_arithmetic_progression_angle (a n : ℝ) 
  (h1 : a + (a + n) + (a + 2 * n) + (a + 3 * n) + (a + 4 * n) = 540) :
  a + 2 * n = 108 :=
by
  sorry

end pentagon_arithmetic_progression_angle_l1930_193029


namespace mod_product_l1930_193092

theorem mod_product (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 50) : 
  173 * 927 % 50 = n := 
  by
    sorry

end mod_product_l1930_193092


namespace investor_pieces_impossible_to_be_2002_l1930_193007

theorem investor_pieces_impossible_to_be_2002 : 
  ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := 
by
  sorry

end investor_pieces_impossible_to_be_2002_l1930_193007


namespace value_of_b_l1930_193035

-- Define the variables and conditions
variables (a b c : ℚ)
axiom h1 : a + b + c = 150
axiom h2 : a + 10 = b - 3
axiom h3 : b - 3 = 4 * c 

-- The statement we want to prove
theorem value_of_b : b = 655 / 9 := 
by 
  -- We start with assumptions h1, h2, and h3
  sorry

end value_of_b_l1930_193035


namespace second_man_speed_l1930_193000

/-- A formal statement of the problem -/
theorem second_man_speed (v : ℝ) 
  (start_same_place : ∀ t : ℝ, t ≥ 0 → 2 * t = (10 - v) * 1) : 
  v = 8 :=
by
  sorry

end second_man_speed_l1930_193000


namespace probability_not_losing_l1930_193061

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end probability_not_losing_l1930_193061


namespace height_of_the_carton_l1930_193062

noncomputable def carton_height : ℕ :=
  let carton_length := 25
  let carton_width := 42
  let soap_box_length := 7
  let soap_box_width := 6
  let soap_box_height := 10
  let max_soap_boxes := 150
  let boxes_per_row := carton_length / soap_box_length
  let boxes_per_column := carton_width / soap_box_width
  let boxes_per_layer := boxes_per_row * boxes_per_column
  let layers := max_soap_boxes / boxes_per_layer
  layers * soap_box_height

theorem height_of_the_carton :
  carton_height = 70 :=
by
  -- The computation and necessary assumptions for proving the height are encapsulated above.
  sorry

end height_of_the_carton_l1930_193062


namespace simone_finishes_task_at_1115_l1930_193031

noncomputable def simone_finish_time
  (start_time: Nat) -- Start time in minutes past midnight
  (task_1_duration: Nat) -- Duration of the first task in minutes
  (task_2_duration: Nat) -- Duration of the second task in minutes
  (break_duration: Nat) -- Duration of the break in minutes
  (task_3_duration: Nat) -- Duration of the third task in minutes
  (end_time: Nat) := -- End time to be proven
  start_time + task_1_duration + task_2_duration + break_duration + task_3_duration = end_time

theorem simone_finishes_task_at_1115 :
  simone_finish_time 480 45 45 15 90 675 := -- 480 minutes is 8:00 AM; 675 minutes is 11:15 AM
  by sorry

end simone_finishes_task_at_1115_l1930_193031


namespace systematic_sampling_first_group_l1930_193077

theorem systematic_sampling_first_group (x : ℕ) (n : ℕ) (k : ℕ) (total_students : ℕ) (sampled_students : ℕ) 
  (interval : ℕ) (group_num : ℕ) (group_val : ℕ) 
  (h1 : total_students = 1000) (h2 : sampled_students = 40) (h3 : interval = total_students / sampled_students)
  (h4 : interval = 25) (h5 : group_num = 18) 
  (h6 : group_val = 443) (h7 : group_val = x + (group_num - 1) * interval) : 
  x = 18 := 
by 
  sorry

end systematic_sampling_first_group_l1930_193077


namespace no_real_roots_x2_plus_4_l1930_193019

theorem no_real_roots_x2_plus_4 : ¬ ∃ x : ℝ, x^2 + 4 = 0 := by
  sorry

end no_real_roots_x2_plus_4_l1930_193019


namespace largest_even_number_l1930_193099

theorem largest_even_number (n : ℤ) 
    (h1 : (n-6) % 2 = 0) 
    (h2 : (n+6) = 3 * (n-6)) :
    (n + 6) = 18 :=
by
  sorry

end largest_even_number_l1930_193099


namespace ratio_male_to_female_l1930_193054

theorem ratio_male_to_female (total_members female_members : ℕ) (h_total : total_members = 18) (h_female : female_members = 6) :
  (total_members - female_members) / Nat.gcd (total_members - female_members) female_members = 2 ∧
  female_members / Nat.gcd (total_members - female_members) female_members = 1 :=
by
  sorry

end ratio_male_to_female_l1930_193054


namespace solve_system1_solve_system2_l1930_193068

-- Definition for System (1)
theorem solve_system1 (x y : ℤ) (h1 : x - 2 * y = 0) (h2 : 3 * x - y = 5) : x = 2 ∧ y = 1 := 
by
  sorry

-- Definition for System (2)
theorem solve_system2 (x y : ℤ) 
  (h1 : 3 * (x - 1) - 4 * (y + 1) = -1) 
  (h2 : (x / 2) + (y / 3) = -2) : x = -2 ∧ y = -3 := 
by
  sorry

end solve_system1_solve_system2_l1930_193068


namespace cubed_identity_l1930_193075

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l1930_193075


namespace abs_diff_inequality_l1930_193073

theorem abs_diff_inequality (m : ℝ) : (∃ x : ℝ, |x + 2| - |x + 3| > m) ↔ m < -1 :=
sorry

end abs_diff_inequality_l1930_193073


namespace max_value_set_x_graph_transformation_l1930_193078

noncomputable def function_y (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6)) + 2

theorem max_value_set_x :
  ∃ k : ℤ, ∀ x : ℝ, x = k * Real.pi + Real.pi / 6 → function_y x = 4 :=
by
  sorry

theorem graph_transformation :
  ∀ x : ℝ, ∃ y : ℝ, (y = Real.sin x → y = 2 * Real.sin (2 * x + (Real.pi / 6)) + 2) :=
by
  sorry

end max_value_set_x_graph_transformation_l1930_193078


namespace triangle_equilateral_if_condition_l1930_193003

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end triangle_equilateral_if_condition_l1930_193003


namespace farmer_field_m_value_l1930_193018

theorem farmer_field_m_value (m : ℝ) 
    (h_length : ∀ m, m > -4 → 2 * m + 9 > 0) 
    (h_breadth : ∀ m, m > -4 → m - 4 > 0)
    (h_area : (2 * m + 9) * (m - 4) = 88) : 
    m = 7.5 :=
by
  sorry

end farmer_field_m_value_l1930_193018


namespace number_value_proof_l1930_193006

theorem number_value_proof (x y : ℝ) (h1 : 0.5 * x = y + 20) (h2 : x - 2 * y = 40) : x = 40 := 
by
  sorry

end number_value_proof_l1930_193006


namespace area_of_ABCD_is_196_l1930_193022

-- Define the shorter side length of the smaller rectangles
def shorter_side : ℕ := 7

-- Define the longer side length of the smaller rectangles
def longer_side : ℕ := 2 * shorter_side

-- Define the width of rectangle ABCD
def width_ABCD : ℕ := 2 * shorter_side

-- Define the length of rectangle ABCD
def length_ABCD : ℕ := longer_side

-- Define the area of rectangle ABCD
def area_ABCD : ℕ := length_ABCD * width_ABCD

-- Statement of the problem
theorem area_of_ABCD_is_196 : area_ABCD = 196 :=
by
  -- insert proof here
  sorry

end area_of_ABCD_is_196_l1930_193022


namespace tina_assignment_time_l1930_193060

theorem tina_assignment_time (total_time clean_time_per_key remaining_keys assignment_time : ℕ) 
  (h1 : total_time = 52) 
  (h2 : clean_time_per_key = 3) 
  (h3 : remaining_keys = 14) 
  (h4 : assignment_time = total_time - remaining_keys * clean_time_per_key) :
  assignment_time = 10 :=
by
  rw [h1, h2, h3] at h4
  assumption

end tina_assignment_time_l1930_193060


namespace f_2023_l1930_193001

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_all : ∀ x : ℕ, f x ≠ 0 → (x ≥ 0)
axiom f_one : f 1 = 1
axiom f_functional_eq : ∀ a b : ℕ, f (a + b) = f a + f b - 3 * f (a * b)

theorem f_2023 : f 2023 = -(2^2022 - 1) := sorry

end f_2023_l1930_193001


namespace amount_distributed_l1930_193096

theorem amount_distributed (A : ℝ) (h : A / 20 = A / 25 + 120) : A = 12000 :=
by
  sorry

end amount_distributed_l1930_193096


namespace remainder_mod_41_l1930_193083

theorem remainder_mod_41 (M : ℤ) (hM1 : M = 1234567891011123940) : M % 41 = 0 :=
by
  sorry

end remainder_mod_41_l1930_193083


namespace FatherCandyCount_l1930_193043

variables (a b c d e : ℕ)

-- Conditions
def BillyInitial := 6
def CalebInitial := 11
def AndyInitial := 9
def BillyReceived := 8
def CalebReceived := 11
def AndyHasMore := 4

-- Define number of candies Andy has now based on Caleb's candies
def AndyTotal (b c : ℕ) : ℕ := c + AndyHasMore

-- Define number of candies received by Andy
def AndyReceived (a b c d e : ℕ) : ℕ := (AndyTotal b c) - AndyInitial

-- Define total candies bought by father
def FatherBoughtCandies (d e f : ℕ) : ℕ := d + e + f

theorem FatherCandyCount : FatherBoughtCandies BillyReceived CalebReceived (AndyReceived BillyInitial CalebInitial AndyInitial BillyReceived CalebReceived)  = 36 :=
by
  sorry

end FatherCandyCount_l1930_193043


namespace probability_X_l1930_193032

theorem probability_X (P : ℕ → ℚ) (h1 : P 1 = 1/10) (h2 : P 2 = 2/10) (h3 : P 3 = 3/10) (h4 : P 4 = 4/10) :
  P 2 + P 3 = 1/2 :=
by
  sorry

end probability_X_l1930_193032


namespace sum_of_squares_l1930_193088

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 :=
sorry

end sum_of_squares_l1930_193088


namespace inserted_number_sq_property_l1930_193045

noncomputable def inserted_number (n : ℕ) : ℕ :=
  (5 * 10^n - 1) * 10^(n+1) + 1

theorem inserted_number_sq_property (n : ℕ) : (inserted_number n)^2 = (10^(n+1) - 1)^2 :=
by sorry

end inserted_number_sq_property_l1930_193045


namespace percentage_problem_l1930_193055

theorem percentage_problem (x : ℝ) (h : (3 / 8) * x = 141) : (round (0.3208 * x) = 121) :=
by
  sorry

end percentage_problem_l1930_193055


namespace scientific_notation_of_investment_l1930_193011

theorem scientific_notation_of_investment : 41800000000 = 4.18 * 10^10 := 
by
  sorry

end scientific_notation_of_investment_l1930_193011


namespace coefficient_x18_is_zero_coefficient_x17_is_3420_l1930_193047

open Polynomial

noncomputable def P : Polynomial ℚ := (1 + X^5 + X^7)^20

theorem coefficient_x18_is_zero : coeff P 18 = 0 :=
sorry

theorem coefficient_x17_is_3420 : coeff P 17 = 3420 :=
sorry

end coefficient_x18_is_zero_coefficient_x17_is_3420_l1930_193047


namespace average_visitors_per_day_l1930_193040

/-- The average number of visitors per day in a month of 30 days that begins with a Sunday is 188, 
given that the library has 500 visitors on Sundays and 140 visitors on other days. -/
theorem average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) 
   (starts_on_sunday : Bool) (sundays : ℕ) 
   (visitors_sunday_eq_500 : visitors_sunday = 500)
   (visitors_other_eq_140 : visitors_other = 140)
   (days_in_month_eq_30 : days_in_month = 30)
   (starts_on_sunday_eq_true : starts_on_sunday = true)
   (sundays_eq_4 : sundays = 4) :
   (visitors_sunday * sundays + visitors_other * (days_in_month - sundays)) / days_in_month = 188 := 
by {
  sorry
}

end average_visitors_per_day_l1930_193040


namespace estimate_students_spending_more_than_60_l1930_193020

-- Definition of the problem
def students_surveyed : ℕ := 50
def students_inclined_to_subscribe : ℕ := 8
def total_students : ℕ := 1000
def estimated_students : ℕ := 600

-- Define the proof task
theorem estimate_students_spending_more_than_60 :
  (students_inclined_to_subscribe : ℝ) / (students_surveyed : ℝ) * (total_students : ℝ) = estimated_students :=
by
  sorry

end estimate_students_spending_more_than_60_l1930_193020


namespace pythagorean_triple_l1930_193009

theorem pythagorean_triple {c a b : ℕ} (h1 : a = 24) (h2 : b = 7) (h3 : c = 25) : a^2 + b^2 = c^2 :=
by
  rw [h1, h2, h3]
  norm_num

end pythagorean_triple_l1930_193009


namespace question_1_question_2_l1930_193085

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem question_1 :
  f 1 * f 2 * f 3 = 36 * 108 * 360 := by
  sorry

theorem question_2 :
  ∃ m ≥ 2, ∀ n : ℕ, n > 0 → f n % m = 0 ∧ m = 36 := by
  sorry

end question_1_question_2_l1930_193085


namespace integer_root_of_polynomial_l1930_193082

/-- Prove that -6 is a root of the polynomial equation x^3 + bx + c = 0,
    where b and c are rational numbers and 3 - sqrt(5) is a root
 -/
theorem integer_root_of_polynomial (b c : ℚ)
  (h : ∀ x : ℝ, (x^3 + (b : ℝ)*x + (c : ℝ) = 0) → x = (3 - Real.sqrt 5) ∨ x = (3 + Real.sqrt 5) ∨ x = -6) :
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -6 :=
by
  sorry

end integer_root_of_polynomial_l1930_193082


namespace correctness_of_propositions_l1930_193076

-- Definitions of the conditions
def residual_is_random_error (e : ℝ) : Prop := ∃ (y : ℝ) (y_hat : ℝ), e = y - y_hat
def data_constraints (a b c d : ℕ) : Prop := a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5
def histogram_judgement : Prop := ∀ (H : Type) (rel : H → H → Prop), ¬(H ≠ H) ∨ (∀ x y : H, rel x y ↔ true)

-- The mathematical equivalence proof problem
theorem correctness_of_propositions (e : ℝ) (a b c d : ℕ) : 
  (residual_is_random_error e → false) ∧
  (data_constraints a b c d → true) ∧
  (histogram_judgement → true) :=
by
  sorry

end correctness_of_propositions_l1930_193076


namespace geometric_series_sum_l1930_193038

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l1930_193038


namespace coordinates_of_C_prime_l1930_193037

-- Define the given vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def similarity_ratio : ℝ := 2

-- Define the function for the similarity transformation
def similarity_transform (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (ratio * x, ratio * y)

-- Prove the coordinates of C'
theorem coordinates_of_C_prime :
  similarity_transform (0, 0) similarity_ratio C = (6, 4) ∨ 
  similarity_transform (0, 0) similarity_ratio C = (-6, -4) :=
by
  sorry

end coordinates_of_C_prime_l1930_193037


namespace quadratic_root_l1930_193094

theorem quadratic_root (a b c : ℝ) (h : 9 * a - 3 * b + c = 0) : 
  a * (-3)^2 + b * (-3) + c = 0 :=
by
  sorry

end quadratic_root_l1930_193094


namespace arithmetic_sequence_sum_l1930_193044

theorem arithmetic_sequence_sum (a : ℕ → Int) (a1 a2017 : Int)
  (h1 : a 1 = a1) 
  (h2017 : a 2017 = a2017)
  (roots_eq : ∀ x, x^2 - 10 * x + 16 = 0 → (x = a1 ∨ x = a2017))
  (arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) :
  a 2 + a 1009 + a 2016 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l1930_193044


namespace Wendy_earned_45_points_l1930_193063

-- Definitions for the conditions
def points_per_bag : Nat := 5
def total_bags : Nat := 11
def unrecycled_bags : Nat := 2

-- The variable for recycled bags and total points earned
def recycled_bags := total_bags - unrecycled_bags
def total_points := recycled_bags * points_per_bag

theorem Wendy_earned_45_points : total_points = 45 :=
by
  -- Proof goes here
  sorry

end Wendy_earned_45_points_l1930_193063


namespace polynomial_divisible_l1930_193041

theorem polynomial_divisible (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, (x-1)^3 ∣ x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 :=
by
  sorry

end polynomial_divisible_l1930_193041


namespace right_triangle_hypotenuse_enlargement_l1930_193010

theorem right_triangle_hypotenuse_enlargement
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  ((5 * a)^2 + (5 * b)^2 = (5 * c)^2) :=
by sorry

end right_triangle_hypotenuse_enlargement_l1930_193010


namespace problem1_problem2_problem3_l1930_193059

theorem problem1 : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 :=
by
  sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 6) ^ 2 - (Real.sqrt 5 + Real.sqrt 6) ^ 2 = -4 * Real.sqrt 30 :=
by
  sorry

theorem problem3 : (2 * Real.sqrt (3 / 2) - Real.sqrt (1 / 2)) * (1 / 2 * Real.sqrt 8 + Real.sqrt (2 / 3)) = (5 / 3) * Real.sqrt 3 + 1 :=
by
  sorry

end problem1_problem2_problem3_l1930_193059


namespace find_x_for_prime_power_l1930_193067

theorem find_x_for_prime_power (x : ℤ) :
  (∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ (2 * x * x + x - 6 = p ^ k)) → (x = -3 ∨ x = 2 ∨ x = 5) := by
  sorry

end find_x_for_prime_power_l1930_193067


namespace a_c_sum_l1930_193012

theorem a_c_sum (a b c d : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : d = a * b * c) (h5 : 233 % d = 79) : a + c = 13 :=
sorry

end a_c_sum_l1930_193012


namespace correct_solution_to_equation_l1930_193089

theorem correct_solution_to_equation :
  ∃ x m : ℚ, (m = 3 ∧ x = 14 / 23 → 7 * (2 - 2 * x) = 3 * (3 * x - m) + 63) ∧ (∃ x : ℚ, (∃ m : ℚ, m = 3) ∧ (7 * (2 - 2 * x) - (3 * (3 * x - 3) + 63) = 0)) →
  x = 2 := 
sorry

end correct_solution_to_equation_l1930_193089


namespace construct_quadratic_l1930_193049

-- Definitions from the problem's conditions
def quadratic_has_zeros (f : ℝ → ℝ) (r1 r2 : ℝ) : Prop :=
  f r1 = 0 ∧ f r2 = 0

def quadratic_value_at (f : ℝ → ℝ) (x_val value : ℝ) : Prop :=
  f x_val = value

-- Construct the Lean theorem statement
theorem construct_quadratic :
  ∃ f : ℝ → ℝ, quadratic_has_zeros f 1 5 ∧ quadratic_value_at f 3 10 ∧
  ∀ x, f x = (-5/2 : ℝ) * x^2 + 15 * x - 25 / 2 :=
sorry

end construct_quadratic_l1930_193049


namespace neither_chemistry_nor_biology_l1930_193074

variable (club_size chemistry_students biology_students both_students neither_students : ℕ)

def students_in_club : Prop :=
  club_size = 75

def students_taking_chemistry : Prop :=
  chemistry_students = 40

def students_taking_biology : Prop :=
  biology_students = 35

def students_taking_both : Prop :=
  both_students = 25

theorem neither_chemistry_nor_biology :
  students_in_club club_size ∧ 
  students_taking_chemistry chemistry_students ∧
  students_taking_biology biology_students ∧
  students_taking_both both_students →
  neither_students = 75 - ((chemistry_students - both_students) + (biology_students - both_students) + both_students) :=
by
  intros
  sorry

end neither_chemistry_nor_biology_l1930_193074


namespace no_simultaneous_inequalities_l1930_193005

theorem no_simultaneous_inequalities (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end no_simultaneous_inequalities_l1930_193005


namespace youngest_brother_age_l1930_193023

theorem youngest_brother_age (x : ℕ) (h : x + (x + 1) + (x + 2) = 96) : x = 31 :=
sorry

end youngest_brother_age_l1930_193023


namespace average_of_scores_l1930_193027

theorem average_of_scores :
  let scores := [50, 60, 70, 80, 80]
  let total := 340
  let num_subjects := 5
  let average := total / num_subjects
  average = 68 :=
by
  sorry

end average_of_scores_l1930_193027


namespace percentage_increase_l1930_193051

theorem percentage_increase (N P : ℕ) (h1 : N = 40)
       (h2 : (N + (P / 100) * N) - (N - (30 / 100) * N) = 22) : P = 25 :=
by 
  have p1 := h1
  have p2 := h2
  sorry

end percentage_increase_l1930_193051


namespace cost_of_expensive_feed_l1930_193095

open Lean Real

theorem cost_of_expensive_feed (total_feed : Real)
                              (total_cost_per_pound : Real) 
                              (cheap_feed_weight : Real)
                              (cheap_cost_per_pound : Real)
                              (expensive_feed_weight : Real)
                              (expensive_cost_per_pound : Real):
  total_feed = 35 ∧ 
  total_cost_per_pound = 0.36 ∧ 
  cheap_feed_weight = 17 ∧ 
  cheap_cost_per_pound = 0.18 ∧ 
  expensive_feed_weight = total_feed - cheap_feed_weight →
  total_feed * total_cost_per_pound - cheap_feed_weight * cheap_cost_per_pound = expensive_feed_weight * expensive_cost_per_pound →
  expensive_cost_per_pound = 0.53 :=
by {
  sorry
}

end cost_of_expensive_feed_l1930_193095


namespace ac_bd_bound_l1930_193021

variables {a b c d : ℝ}

theorem ac_bd_bound (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 4) : |a * c + b * d| ≤ 2 := 
sorry

end ac_bd_bound_l1930_193021


namespace part_a_total_time_part_b_average_time_part_c_probability_l1930_193087

theorem part_a_total_time :
  ∃ (total_combinations: ℕ) (time_per_attempt: ℕ) (total_time: ℕ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_per_attempt = 2 ∧ 
    total_time = total_combinations * time_per_attempt / 60 ∧ 
    total_time = 4 := sorry

theorem part_b_average_time :
  ∃ (total_combinations: ℕ) (avg_attempts: ℚ) (time_per_attempt: ℕ) (avg_time: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    avg_attempts = (1 + total_combinations) / 2 ∧ 
    time_per_attempt = 2 ∧ 
    avg_time = (avg_attempts * time_per_attempt) / 60 ∧ 
    avg_time = 2 + 1 / 60 := sorry

theorem part_c_probability :
  ∃ (total_combinations: ℕ) (time_limit: ℕ) (attempt_in_time: ℕ) (probability: ℚ),
    total_combinations = Nat.choose 10 3 ∧ 
    time_limit = 60 ∧ 
    attempt_in_time = time_limit / 2 ∧ 
    probability = (attempt_in_time - 1) / total_combinations ∧ 
    probability = 29 / 120 := sorry

end part_a_total_time_part_b_average_time_part_c_probability_l1930_193087


namespace tangent_line_equation_l1930_193004

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l1930_193004


namespace find_number_of_elements_l1930_193072

theorem find_number_of_elements (n S : ℕ) (h1: (S + 26) / n = 15) (h2: (S + 36) / n = 16) : n = 10 := by
  sorry

end find_number_of_elements_l1930_193072


namespace xiao_ming_should_choose_store_A_l1930_193097

def storeB_cost (x : ℕ) : ℝ := 0.85 * x

def storeA_cost (x : ℕ) : ℝ :=
  if x ≤ 10 then x
  else 0.7 * x + 3

theorem xiao_ming_should_choose_store_A (x : ℕ) (h : x = 22) :
  storeA_cost x < storeB_cost x := by
  sorry

end xiao_ming_should_choose_store_A_l1930_193097


namespace garden_wall_additional_courses_l1930_193025

theorem garden_wall_additional_courses (initial_courses additional_courses : ℕ) (bricks_per_course total_bricks bricks_removed : ℕ) 
  (h1 : bricks_per_course = 400) 
  (h2 : initial_courses = 3) 
  (h3 : bricks_removed = bricks_per_course / 2) 
  (h4 : total_bricks = 1800) 
  (h5 : total_bricks = initial_courses * bricks_per_course + additional_courses * bricks_per_course - bricks_removed) : 
  additional_courses = 2 :=
by
  sorry

end garden_wall_additional_courses_l1930_193025


namespace sum_odd_divisors_90_eq_78_l1930_193081

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l1930_193081


namespace intersection_of_A_and_B_l1930_193034

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x | -Real.sqrt 3 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l1930_193034


namespace largest_lcm_value_l1930_193058

open Nat

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_value_l1930_193058


namespace min_value_2013_Quanzhou_simulation_l1930_193046

theorem min_value_2013_Quanzhou_simulation:
  ∃ (x y : ℝ), (x - y - 1 = 0) ∧ (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
by
  use 2
  use 3
  sorry

end min_value_2013_Quanzhou_simulation_l1930_193046
