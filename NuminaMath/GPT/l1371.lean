import Mathlib

namespace sum_of_cubes_of_three_consecutive_integers_l1371_137199

theorem sum_of_cubes_of_three_consecutive_integers (a : ℕ) (h : (a * a) + (a + 1) * (a + 1) + (a + 2) * (a + 2) = 2450) : a * a * a + (a + 1) * (a + 1) * (a + 1) + (a + 2) * (a + 2) * (a + 2) = 73341 :=
by
  sorry

end sum_of_cubes_of_three_consecutive_integers_l1371_137199


namespace range_of_a_l1371_137136

noncomputable def f (a x : ℝ) : ℝ := x + (a^2) / (4 * x)
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f a x1 ≥ g x2) → 
  2 * Real.sqrt (Real.exp 1 - 2) ≤ a := sorry

end range_of_a_l1371_137136


namespace find_f4_l1371_137185

noncomputable def f : ℝ → ℝ := sorry

theorem find_f4 (hf_odd : ∀ x : ℝ, f (-x) = -f x)
                (hf_property : ∀ x : ℝ, f (x + 2) = -f x) :
  f 4 = 0 :=
sorry

end find_f4_l1371_137185


namespace relationship_between_a_b_c_l1371_137101

variable (a b c : ℝ)
variable (h_a : a = 0.4 ^ 0.2)
variable (h_b : b = 0.4 ^ 0.6)
variable (h_c : c = 2.1 ^ 0.2)

-- Prove the relationship c > a > b
theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l1371_137101


namespace num_new_students_l1371_137115

-- Definitions based on the provided conditions
def original_class_strength : ℕ := 10
def original_average_age : ℕ := 40
def new_students_avg_age : ℕ := 32
def decrease_in_average_age : ℕ := 4
def new_average_age : ℕ := original_average_age - decrease_in_average_age
def new_class_strength (n : ℕ) : ℕ := original_class_strength + n

-- The proof statement
theorem num_new_students (n : ℕ) :
  (original_class_strength * original_average_age + n * new_students_avg_age) 
  = new_class_strength n * new_average_age → n = 10 :=
by
  sorry

end num_new_students_l1371_137115


namespace find_QS_l1371_137138

theorem find_QS (RS QR QS : ℕ) (h1 : RS = 13) (h2 : QR = 5) (h3 : QR * 13 = 5 * 13) :
  QS = 12 :=
by
  sorry

end find_QS_l1371_137138


namespace least_number_to_add_l1371_137113

theorem least_number_to_add (x : ℕ) : (1053 + x) % 23 = 0 ↔ x = 5 := by
  sorry

end least_number_to_add_l1371_137113


namespace age_hence_l1371_137146

theorem age_hence (A x : ℕ) (hA : A = 24) (hx : 4 * (A + x) - 4 * (A - 3) = A) : x = 3 :=
by {
  sorry
}

end age_hence_l1371_137146


namespace election_winner_votes_l1371_137179

-- Define the conditions and question in Lean 4
theorem election_winner_votes (V : ℝ) (h1 : V > 0) 
  (h2 : 0.54 * V - 0.46 * V = 288) : 0.54 * V = 1944 :=
by
  sorry

end election_winner_votes_l1371_137179


namespace steve_first_stack_plastic_cups_l1371_137180

theorem steve_first_stack_plastic_cups (cups_n : ℕ -> ℕ)
  (h_prop : ∀ n, cups_n (n + 1) = cups_n n + 4)
  (h_second : cups_n 2 = 21)
  (h_third : cups_n 3 = 25)
  (h_fourth : cups_n 4 = 29) :
  cups_n 1 = 17 :=
sorry

end steve_first_stack_plastic_cups_l1371_137180


namespace geometric_sequence_sum_a_l1371_137114

theorem geometric_sequence_sum_a (a : ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = 4^n + a) :
  a = -1 :=
sorry

end geometric_sequence_sum_a_l1371_137114


namespace width_of_field_l1371_137150

noncomputable def field_width : ℝ := 60

theorem width_of_field (L W : ℝ) (hL : L = (7/5) * W) (hP : 288 = 2 * L + 2 * W) : W = field_width :=
by
  sorry

end width_of_field_l1371_137150


namespace eval_floor_expr_l1371_137183

def frac_part1 : ℚ := (15 / 8)
def frac_part2 : ℚ := (11 / 3)
def square_frac1 : ℚ := frac_part1 ^ 2
def ceil_part : ℤ := ⌈square_frac1⌉
def add_frac2 : ℚ := ceil_part + frac_part2

theorem eval_floor_expr : (⌊add_frac2⌋ : ℤ) = 7 := 
sorry

end eval_floor_expr_l1371_137183


namespace arable_land_decrease_max_l1371_137181

theorem arable_land_decrease_max
  (A₀ : ℕ := 100000)
  (grain_yield_increase : ℝ := 1.22)
  (per_capita_increase : ℝ := 1.10)
  (pop_growth_rate : ℝ := 0.01)
  (years : ℕ := 10) :
  ∃ (max_decrease : ℕ), max_decrease = 4 := sorry

end arable_land_decrease_max_l1371_137181


namespace Jill_ball_difference_l1371_137176

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l1371_137176


namespace emily_first_round_points_l1371_137190

theorem emily_first_round_points (x : ℤ) 
  (second_round : ℤ := 33) 
  (last_round_loss : ℤ := 48) 
  (total_points_end : ℤ := 1) 
  (eqn : x + second_round - last_round_loss = total_points_end) : 
  x = 16 := 
by 
  sorry

end emily_first_round_points_l1371_137190


namespace A_and_B_worked_together_for_5_days_before_A_left_the_job_l1371_137166

noncomputable def workRate_A (W : ℝ) : ℝ := W / 20
noncomputable def workRate_B (W : ℝ) : ℝ := W / 12

noncomputable def combinedWorkRate (W : ℝ) : ℝ := workRate_A W + workRate_B W

noncomputable def workDoneTogether (x : ℝ) (W : ℝ) : ℝ := x * combinedWorkRate W
noncomputable def workDoneBy_B_Alone (W : ℝ) : ℝ := 3 * workRate_B W

theorem A_and_B_worked_together_for_5_days_before_A_left_the_job (W : ℝ) :
  ∃ x : ℝ, workDoneTogether x W + workDoneBy_B_Alone W = W ∧ x = 5 :=
by
  sorry

end A_and_B_worked_together_for_5_days_before_A_left_the_job_l1371_137166


namespace Eddy_travel_time_l1371_137130

theorem Eddy_travel_time (T V_e V_f : ℝ) 
  (dist_AB dist_AC : ℝ) 
  (time_Freddy : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : dist_AB = 600) 
  (h2 : dist_AC = 300) 
  (h3 : time_Freddy = 3) 
  (h4 : speed_ratio = 2)
  (h5 : V_f = dist_AC / time_Freddy)
  (h6 : V_e = speed_ratio * V_f)
  (h7 : T = dist_AB / V_e) :
  T = 3 :=
by
  sorry

end Eddy_travel_time_l1371_137130


namespace vasya_gift_choices_l1371_137177

theorem vasya_gift_choices :
  let cars := 7
  let construction_sets := 5
  (cars * construction_sets + Nat.choose cars 2 + Nat.choose construction_sets 2) = 66 :=
by
  sorry

end vasya_gift_choices_l1371_137177


namespace find_divisor_l1371_137140

theorem find_divisor
  (Dividend : ℕ)
  (Quotient : ℕ)
  (Remainder : ℕ)
  (h1 : Dividend = 686)
  (h2 : Quotient = 19)
  (h3 : Remainder = 2) :
  ∃ (Divisor : ℕ), (Dividend = (Divisor * Quotient) + Remainder) ∧ Divisor = 36 :=
by
  sorry

end find_divisor_l1371_137140


namespace lidia_money_left_l1371_137184

theorem lidia_money_left 
  (cost_per_app : ℕ := 4) 
  (num_apps : ℕ := 15) 
  (total_money : ℕ := 66) 
  (discount_rate : ℚ := 0.15) :
  total_money - (num_apps * cost_per_app - (num_apps * cost_per_app * discount_rate)) = 15 := by 
  sorry

end lidia_money_left_l1371_137184


namespace cube_as_difference_of_squares_l1371_137123

theorem cube_as_difference_of_squares (a : ℕ) : 
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := 
by 
  -- The proof portion would go here, but since we only need the statement:
  sorry

end cube_as_difference_of_squares_l1371_137123


namespace find_utilities_second_l1371_137160

def rent_first : ℝ := 800
def utilities_first : ℝ := 260
def distance_first : ℕ := 31
def rent_second : ℝ := 900
def distance_second : ℕ := 21
def cost_per_mile : ℝ := 0.58
def days_per_month : ℕ := 20
def cost_difference : ℝ := 76

-- Helper definitions
def driving_cost (distance : ℕ) : ℝ :=
  distance * days_per_month * cost_per_mile

def total_cost_first : ℝ :=
  rent_first + utilities_first + driving_cost distance_first

def total_cost_second_no_utilities : ℝ :=
  rent_second + driving_cost distance_second

theorem find_utilities_second :
  ∃ (utilities_second : ℝ),
  total_cost_first - total_cost_second_no_utilities = cost_difference →
  utilities_second = 200 :=
sorry

end find_utilities_second_l1371_137160


namespace solve_for_x_l1371_137178

theorem solve_for_x : ∀ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) → x = 0 := by
  intros x h
  sorry

end solve_for_x_l1371_137178


namespace arrangement_count_l1371_137198

def number_of_arrangements (slots total_geometry total_number_theory : ℕ) : ℕ :=
  Nat.choose slots total_geometry

theorem arrangement_count :
  number_of_arrangements 8 5 3 = 56 := 
by
  sorry

end arrangement_count_l1371_137198


namespace direct_proportion_function_l1371_137137

theorem direct_proportion_function (m : ℝ) (h1 : m^2 - 8 = 1) (h2 : m ≠ 3) : m = -3 :=
by
  sorry

end direct_proportion_function_l1371_137137


namespace committee_form_count_l1371_137142

def numWaysToFormCommittee (departments : Fin 4 → (ℕ × ℕ)) : ℕ :=
  let waysCase1 := 6 * 81 * 81
  let waysCase2 := 6 * 9 * 9 * 2 * 9 * 9
  waysCase1 + waysCase2

theorem committee_form_count (departments : Fin 4 → (ℕ × ℕ)) 
  (h : ∀ i, departments i = (3, 3)) :
  numWaysToFormCommittee departments = 48114 := 
by
  sorry

end committee_form_count_l1371_137142


namespace simplify_fraction_eq_l1371_137122

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end simplify_fraction_eq_l1371_137122


namespace al_told_the_truth_l1371_137148

-- Definitions of G, S, and B based on each pirate's claim
def tom_G := 10
def tom_S := 8
def tom_B := 11

def al_G := 9
def al_S := 11
def al_B := 10

def pit_G := 10
def pit_S := 10
def pit_B := 9

def jim_G := 8
def jim_S := 10
def jim_B := 11

-- Condition that the total number of coins is 30
def total_coins (G : ℕ) (S : ℕ) (B : ℕ) : Prop := G + S + B = 30

-- The assertion that only Al told the truth
theorem al_told_the_truth :
  (total_coins tom_G tom_S tom_B → false) →
  (total_coins al_G al_S al_B) →
  (total_coins pit_G pit_S pit_B → false) →
  (total_coins jim_G jim_S jim_B → false) →
  true :=
by
  intros
  sorry

end al_told_the_truth_l1371_137148


namespace polygon_diagonals_with_restriction_l1371_137194

def num_sides := 150

def total_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

def restricted_diagonals (n : ℕ) : ℕ :=
  n * 150 / 4

def valid_diagonals (n : ℕ) : ℕ :=
  total_diagonals n - restricted_diagonals n

theorem polygon_diagonals_with_restriction : valid_diagonals num_sides = 5400 :=
by
  sorry

end polygon_diagonals_with_restriction_l1371_137194


namespace smallest_n_satisfying_conditions_l1371_137195

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l1371_137195


namespace jimmy_fill_pool_time_l1371_137119

theorem jimmy_fill_pool_time (pool_gallons : ℕ) (bucket_gallons : ℕ) (time_per_trip_sec : ℕ) (sec_per_min : ℕ) :
  pool_gallons = 84 → 
  bucket_gallons = 2 → 
  time_per_trip_sec = 20 → 
  sec_per_min = 60 → 
  (pool_gallons / bucket_gallons) * time_per_trip_sec / sec_per_min = 14 :=
by
  sorry

end jimmy_fill_pool_time_l1371_137119


namespace effective_discount_l1371_137134

theorem effective_discount (original_price sale_price price_after_coupon : ℝ) :
  sale_price = 0.4 * original_price →
  price_after_coupon = 0.7 * sale_price →
  (original_price - price_after_coupon) / original_price * 100 = 72 :=
by
  intros h1 h2
  sorry

end effective_discount_l1371_137134


namespace max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l1371_137197

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l1371_137197


namespace smallest_integer_modulus_l1371_137125

theorem smallest_integer_modulus :
  ∃ n : ℕ, 0 < n ∧ (7 ^ n ≡ n ^ 4 [MOD 3]) ∧
  ∀ m : ℕ, 0 < m ∧ (7 ^ m ≡ m ^ 4 [MOD 3]) → n ≤ m :=
by
  sorry

end smallest_integer_modulus_l1371_137125


namespace greatest_b_solution_l1371_137182

def f (b : ℝ) : ℝ := b^2 - 10 * b + 24

theorem greatest_b_solution : ∃ (b : ℝ), (f b ≤ 0) ∧ (∀ (b' : ℝ), (f b' ≤ 0) → b' ≤ b) ∧ b = 6 :=
by
  sorry

end greatest_b_solution_l1371_137182


namespace present_age_of_son_l1371_137116

-- Define variables for the current ages of the son and the man (father).
variables (S M : ℕ)

-- Define the conditions:
-- The man is 35 years older than his son.
def condition1 : Prop := M = S + 35

-- In two years, the man's age will be twice the age of his son.
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The theorem that we need to prove:
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 33 :=
by
  -- Add sorry to skip the proof.
  sorry

end present_age_of_son_l1371_137116


namespace houses_built_during_boom_l1371_137144

-- Define initial and current number of houses
def initial_houses : ℕ := 1426
def current_houses : ℕ := 2000

-- Define the expected number of houses built during the boom
def expected_houses_built : ℕ := 574

-- The theorem to prove
theorem houses_built_during_boom : (current_houses - initial_houses) = expected_houses_built :=
by 
    sorry

end houses_built_during_boom_l1371_137144


namespace problem_II_l1371_137102

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3)^n

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 2) * (1 - (1 / 3)^n)

lemma problem_I_1 (n : ℕ) (hn : n > 0) : a_n n = (1 / 3)^n := by
  sorry

lemma problem_I_2 (n : ℕ) (hn : n > 0) : S_n n = (1 / 2) * (1 - (1 / 3)^n) := by
  sorry

theorem problem_II (t : ℝ) : S_n 1 = 1 / 3 ∧ S_n 2 = 4 / 9 ∧ S_n 3 = 13 / 27 ∧
  (S_n 1 + 3 * (S_n 2 + S_n 3) = 2 * (S_n 1 + S_n 2) * t) ↔ t = 2 := by
  sorry

end problem_II_l1371_137102


namespace days_matt_and_son_eat_only_l1371_137131

theorem days_matt_and_son_eat_only (x y : ℕ) 
  (h1 : x + y = 7)
  (h2 : 2 * x + 8 * y = 38) : 
  x = 3 :=
by
  sorry

end days_matt_and_son_eat_only_l1371_137131


namespace find_largest_number_l1371_137170

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
  sorry

end find_largest_number_l1371_137170


namespace james_profit_l1371_137143

/--
  Prove that James's profit from buying 200 lotto tickets at $2 each, given the 
  conditions about winning tickets, is $4,830.
-/
theorem james_profit 
  (total_tickets : ℕ := 200)
  (cost_per_ticket : ℕ := 2)
  (winner_percentage : ℝ := 0.2)
  (five_dollar_win_pct : ℝ := 0.8)
  (grand_prize : ℝ := 5000)
  (average_other_wins : ℝ := 10) :
  let total_cost := total_tickets * cost_per_ticket 
  let total_winners := winner_percentage * total_tickets
  let five_dollar_winners := five_dollar_win_pct * total_winners
  let total_five_dollar := five_dollar_winners * 5
  let remaining_winners := total_winners - 1 - five_dollar_winners
  let total_remaining_winners := remaining_winners * average_other_wins
  let total_winnings := total_five_dollar + grand_prize + total_remaining_winners
  let profit := total_winnings - total_cost
  profit = 4830 :=
by
  sorry

end james_profit_l1371_137143


namespace bus_ride_time_l1371_137153

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end bus_ride_time_l1371_137153


namespace percentage_decrease_in_larger_angle_l1371_137192

-- Define the angles and conditions
def angles_complementary (A B : ℝ) : Prop := A + B = 90
def angle_ratio (A B : ℝ) : Prop := A / B = 1 / 2
def small_angle_increase (A A' : ℝ) : Prop := A' = A * 1.2
def large_angle_new (A' B' : ℝ) : Prop := A' + B' = 90

-- Prove percentage decrease in larger angle
theorem percentage_decrease_in_larger_angle (A B A' B' : ℝ) 
    (h1 : angles_complementary A B)
    (h2 : angle_ratio A B)
    (h3 : small_angle_increase A A')
    (h4 : large_angle_new A' B')
    : (B - B') / B * 100 = 10 :=
sorry

end percentage_decrease_in_larger_angle_l1371_137192


namespace gambler_difference_eq_two_l1371_137129

theorem gambler_difference_eq_two (x y : ℕ) (x_lost y_lost : ℕ) :
  20 * x + 100 * y = 3000 ∧
  x + y = 14 ∧
  20 * (14 - y_lost) + 100 * y_lost = 760 →
  (x_lost - y_lost = 2) := sorry

end gambler_difference_eq_two_l1371_137129


namespace sequence_is_decreasing_l1371_137126

-- Define the sequence {a_n} using a recursive function
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1))

-- Define a condition ensuring the sequence a_n is decreasing
theorem sequence_is_decreasing (a : ℕ → ℝ) (h : seq a) : ∀ n, a (n + 1) < a n :=
by
  intro n
  sorry

end sequence_is_decreasing_l1371_137126


namespace mrs_hilt_total_spent_l1371_137107

def kids_ticket_usual_cost : ℕ := 1 -- $1 for 4 tickets
def adults_ticket_usual_cost : ℕ := 2 -- $2 for 3 tickets

def kids_ticket_deal_cost : ℕ := 4 -- $4 for 20 tickets
def adults_ticket_deal_cost : ℕ := 8 -- $8 for 15 tickets

def kids_tickets_purchased : ℕ := 24
def adults_tickets_purchased : ℕ := 18

def total_kids_ticket_cost : ℕ :=
  let kids_deal_tickets := kids_ticket_deal_cost
  let remaining_kids_tickets := kids_ticket_usual_cost
  kids_deal_tickets + remaining_kids_tickets

def total_adults_ticket_cost : ℕ :=
  let adults_deal_tickets := adults_ticket_deal_cost
  let remaining_adults_tickets := adults_ticket_usual_cost
  adults_deal_tickets + remaining_adults_tickets

def total_cost (kids_cost adults_cost : ℕ) : ℕ :=
  kids_cost + adults_cost

theorem mrs_hilt_total_spent : total_cost total_kids_ticket_cost total_adults_ticket_cost = 15 := by
  sorry

end mrs_hilt_total_spent_l1371_137107


namespace price_restoration_l1371_137145

theorem price_restoration (P : Real) (hP : P > 0) :
  let new_price := 0.85 * P
  let required_increase := ((1 / 0.85) - 1) * 100
  required_increase = 17.65 :=
by 
  sorry

end price_restoration_l1371_137145


namespace customers_left_proof_l1371_137133

def initial_customers : ℕ := 21
def tables : ℕ := 3
def people_per_table : ℕ := 3
def remaining_customers : ℕ := tables * people_per_table
def customers_left (initial remaining : ℕ) : ℕ := initial - remaining

theorem customers_left_proof : customers_left initial_customers remaining_customers = 12 := sorry

end customers_left_proof_l1371_137133


namespace total_population_correct_l1371_137135

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l1371_137135


namespace vertex_of_parabola_l1371_137186

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end vertex_of_parabola_l1371_137186


namespace age_difference_l1371_137191

theorem age_difference (A B C : ℕ) (hB : B = 14) (hBC : B = 2 * C) (hSum : A + B + C = 37) : A - B = 2 :=
by
  sorry

end age_difference_l1371_137191


namespace number_of_wins_and_losses_l1371_137163

theorem number_of_wins_and_losses (x y : ℕ) (h1 : x + y = 15) (h2 : 3 * x + y = 41) :
  x = 13 ∧ y = 2 :=
sorry

end number_of_wins_and_losses_l1371_137163


namespace find_d_l1371_137100

theorem find_d (A B C D : ℕ) (h1 : (A + B + C) / 3 = 130) (h2 : (A + B + C + D) / 4 = 126) : D = 114 :=
by
  sorry

end find_d_l1371_137100


namespace minimal_distance_l1371_137149

noncomputable def minimum_distance_travel (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) : ℝ :=
  2 * Real.sqrt 19

theorem minimal_distance (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) :
  minimum_distance_travel a b c ha hb hc = 2 * Real.sqrt 19 :=
by
  -- Proof is omitted
  sorry

end minimal_distance_l1371_137149


namespace pairs_condition_l1371_137169

theorem pairs_condition (a b : ℕ) (prime_p : ∃ p, p = a^2 + b + 1 ∧ Nat.Prime p)
    (divides : ∀ p, p = a^2 + b + 1 → p ∣ (b^2 - a^3 - 1))
    (not_divides : ∀ p, p = a^2 + b + 1 → ¬ p ∣ (a + b - 1)^2) :
  ∃ x, x ≥ 2 ∧ a = 2 ^ x ∧ b = 2 ^ (2 * x) - 1 := sorry

end pairs_condition_l1371_137169


namespace total_pennies_l1371_137196

theorem total_pennies (rachelle gretchen rocky max taylor : ℕ) (h_r : rachelle = 720) (h_g : gretchen = rachelle / 2)
  (h_ro : rocky = gretchen / 3) (h_m : max = rocky * 4) (h_t : taylor = max / 5) :
  rachelle + gretchen + rocky + max + taylor = 1776 := 
by
  sorry

end total_pennies_l1371_137196


namespace inequality_solution_l1371_137108

theorem inequality_solution (x : ℝ) :
  (7 : ℝ) / 30 + abs (x - 7 / 60) < 11 / 20 ↔ -1 / 5 < x ∧ x < 13 / 30 :=
by
  sorry

end inequality_solution_l1371_137108


namespace range_mn_squared_l1371_137156

-- Let's define the conditions in Lean

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is strictly increasing
axiom h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0

-- Condition 2: f(x-1) is centrally symmetric about (1,0)
axiom h2 : ∀ x : ℝ, f (x - 1) = - f (2 - (x - 1))

-- Condition 3: Given inequality
axiom h3 : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0

-- Prove the range for m^2 + n^2 is (9, 49)
theorem range_mn_squared : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0 →
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end range_mn_squared_l1371_137156


namespace max_savings_theorem_band_members_theorem_selection_plans_theorem_l1371_137118

/-- Given conditions for maximum savings calculation -/
def number_of_sets_purchased : ℕ := 75
def max_savings (cost_separate : ℕ) (cost_together : ℕ) : Prop :=
cost_separate - cost_together = 800

theorem max_savings_theorem : 
    ∃ cost_separate cost_together, 
    (cost_separate = 5600) ∧ (cost_together = 4800) → max_savings cost_separate cost_together := by
  sorry

/-- Given conditions for number of members in bands A and B -/
def conditions (x y : ℕ) : Prop :=
x + y = 75 ∧ 70 * x + 80 * y = 5600 ∧ x >= 40

theorem band_members_theorem :
    ∃ x y, conditions x y → (x = 40 ∧ y = 35) := by
  sorry

/-- Given conditions for possible selection plans for charity event -/
def heart_to_heart_activity (a b : ℕ) : Prop :=
3 * a + 5 * b = 65 ∧ a >= 5 ∧ b >= 5

theorem selection_plans_theorem :
    ∃ a b, heart_to_heart_activity a b → 
    ((a = 5 ∧ b = 10) ∨ (a = 10 ∧ b = 7)) := by
  sorry

end max_savings_theorem_band_members_theorem_selection_plans_theorem_l1371_137118


namespace oldest_sibling_multiple_l1371_137161

-- Definitions according to the conditions
def kay_age : Nat := 32
def youngest_sibling_age : Nat := kay_age / 2 - 5
def oldest_sibling_age : Nat := 44

-- The statement to prove
theorem oldest_sibling_multiple : oldest_sibling_age = 4 * youngest_sibling_age :=
by sorry

end oldest_sibling_multiple_l1371_137161


namespace circle_center_coordinates_l1371_137109

theorem circle_center_coordinates (x y : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → (x = 1 ∧ y = -2) :=
by
  sorry

end circle_center_coordinates_l1371_137109


namespace adjusted_ratio_l1371_137188

theorem adjusted_ratio :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 :=
by
  sorry

end adjusted_ratio_l1371_137188


namespace perpendicular_lines_and_slope_l1371_137159

theorem perpendicular_lines_and_slope (b : ℝ) : (x + 3 * y + 4 = 0) ∧ (b * x + 3 * y + 6 = 0) → b = -9 :=
by
  sorry

end perpendicular_lines_and_slope_l1371_137159


namespace annual_interest_rate_is_12_percent_l1371_137154

theorem annual_interest_rate_is_12_percent
  (P : ℕ := 750000)
  (I : ℕ := 37500)
  (t : ℕ := 5)
  (months_in_year : ℕ := 12)
  (annual_days : ℕ := 360)
  (days_per_month : ℕ := 30) :
  ∃ r : ℚ, (r * 100 * months_in_year = 12) ∧ I = P * r * t := 
sorry

end annual_interest_rate_is_12_percent_l1371_137154


namespace total_selling_price_l1371_137103

theorem total_selling_price
  (CP : ℕ) (Gain : ℕ) (TCP : ℕ)
  (h1 : CP = 1200)
  (h2 : Gain = 3 * CP)
  (h3 : TCP = 18 * CP) :
  ∃ TSP : ℕ, TSP = 25200 := 
by
  sorry

end total_selling_price_l1371_137103


namespace correct_equation_l1371_137157

theorem correct_equation :
  (2 * Real.sqrt 2) / (Real.sqrt 2) = 2 :=
by
  -- Proof goes here
  sorry

end correct_equation_l1371_137157


namespace Sammy_has_8_bottle_caps_l1371_137162

-- Definitions representing the conditions
def BilliesBottleCaps := 2
def JaninesBottleCaps := 3 * BilliesBottleCaps
def SammysBottleCaps := JaninesBottleCaps + 2

-- Goal: Prove that Sammy has 8 bottle caps
theorem Sammy_has_8_bottle_caps : 
  SammysBottleCaps = 8 := 
sorry

end Sammy_has_8_bottle_caps_l1371_137162


namespace parents_give_per_year_l1371_137120

def Mikail_age (x : ℕ) : Prop :=
  x = 3 * (x - 3)

noncomputable def money_per_year (total_money : ℕ) (age : ℕ) : ℕ :=
  total_money / age

theorem parents_give_per_year 
  (x : ℕ) (hx : Mikail_age x) : 
  money_per_year 45 x = 5 :=
sorry

end parents_give_per_year_l1371_137120


namespace binomial_product_l1371_137106

theorem binomial_product (x : ℝ) : 
  (2 - x^4) * (3 + x^5) = -x^9 - 3 * x^4 + 2 * x^5 + 6 :=
by 
  sorry

end binomial_product_l1371_137106


namespace complete_work_together_in_days_l1371_137158

-- Define the work rates for John, Rose, and Michael
def johnWorkRate : ℚ := 1 / 10
def roseWorkRate : ℚ := 1 / 40
def michaelWorkRate : ℚ := 1 / 20

-- Define the combined work rate when they work together
def combinedWorkRate : ℚ := johnWorkRate + roseWorkRate + michaelWorkRate

-- Define the total work to be done
def totalWork : ℚ := 1

-- Calculate the total number of days required to complete the work together
def totalDays : ℚ := totalWork / combinedWorkRate

-- Theorem to prove the total days is 40/7
theorem complete_work_together_in_days : totalDays = 40 / 7 :=
by
  -- Following steps would be the complete proofs if required
  rw [totalDays, totalWork, combinedWorkRate, johnWorkRate, roseWorkRate, michaelWorkRate]
  sorry

end complete_work_together_in_days_l1371_137158


namespace parabola_opens_downward_iff_l1371_137165

theorem parabola_opens_downward_iff (m : ℝ) : (m - 1 < 0) ↔ (m < 1) :=
by
  sorry

end parabola_opens_downward_iff_l1371_137165


namespace polynomial_divisibility_l1371_137128

theorem polynomial_divisibility (t : ℤ) : 
  (∀ x : ℤ, (5 * x^3 - 15 * x^2 + t * x - 20) ∣ (x - 2)) → (t = 20) → 
  ∀ x : ℤ, (5 * x^3 - 15 * x^2 + 20 * x - 20) ∣ (5 * x^2 + 5 * x + 5) :=
by
  intro h₁ h₂
  sorry

end polynomial_divisibility_l1371_137128


namespace problem_statement_l1371_137124

noncomputable def find_pq_sum (XZ YZ : ℕ) (XY_perimeter_ratio : ℕ × ℕ) : ℕ :=
  let XY := Real.sqrt (XZ^2 + YZ^2)
  let ZD := Real.sqrt (XZ * YZ)
  let O_radius := 0.5 * ZD
  let tangent_length := Real.sqrt ((XY / 2)^2 - O_radius^2)
  let perimeter := XY + 2 * tangent_length
  let (p, q) := XY_perimeter_ratio
  p + q

theorem problem_statement :
  find_pq_sum 8 15 (30, 17) = 47 :=
by sorry

end problem_statement_l1371_137124


namespace range_x_f_inequality_l1371_137111

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - |x|) + 1 / (x^2 + 1)

theorem range_x_f_inequality :
  (∀ x : ℝ, f (2 * x + 1) ≥ f x) ↔ x ∈ Set.Icc (-1 : ℝ) (-1 / 3) := sorry

end range_x_f_inequality_l1371_137111


namespace small_cuboid_length_is_five_l1371_137173

-- Define initial conditions
def large_cuboid_length : ℝ := 18
def large_cuboid_width : ℝ := 15
def large_cuboid_height : ℝ := 2
def num_small_cuboids : ℕ := 6
def small_cuboid_width : ℝ := 6
def small_cuboid_height : ℝ := 3

-- Theorem to prove the length of the smaller cuboid
theorem small_cuboid_length_is_five (small_cuboid_length : ℝ) 
  (h1 : large_cuboid_length * large_cuboid_width * large_cuboid_height 
          = num_small_cuboids * (small_cuboid_length * small_cuboid_width * small_cuboid_height)) :
  small_cuboid_length = 5 := by
  sorry

end small_cuboid_length_is_five_l1371_137173


namespace average_rate_of_change_l1371_137147

def f (x : ℝ) : ℝ := x^2 - 1

theorem average_rate_of_change : (f 1.1) - (f 1) / (1.1 - 1) = 2.1 :=
by
  sorry

end average_rate_of_change_l1371_137147


namespace sum_first_7_terms_eq_105_l1371_137174

variable {a : ℕ → ℤ}

-- Definitions from conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a)

def a_4_eq_15 : a 4 = 15 := sorry

-- Sum definition specific for 7 terms of an arithmetic sequence.
def sum_first_7_terms (a : ℕ → ℤ) : ℤ := (7 / 2 : ℤ) * (a 1 + a 7)

-- The theorem to prove.
theorem sum_first_7_terms_eq_105 
    (arith_seq : is_arithmetic_sequence a) 
    (a4 : a 4 = 15) : 
  sum_first_7_terms a = 105 := 
sorry

end sum_first_7_terms_eq_105_l1371_137174


namespace cost_per_bottle_l1371_137152

theorem cost_per_bottle (cost_3_bottles cost_4_bottles : ℝ) (n_bottles : ℕ) 
  (h1 : cost_3_bottles = 1.50) (h2 : cost_4_bottles = 2) : 
  (cost_3_bottles / 3) = (cost_4_bottles / 4) ∧ (cost_3_bottles / 3) * n_bottles = 0.50 * n_bottles :=
by
  sorry

end cost_per_bottle_l1371_137152


namespace find_b_l1371_137132

def p (x : ℝ) : ℝ := 2 * x - 3
def q (x : ℝ) (b : ℝ) : ℝ := 5 * x - b

theorem find_b (b : ℝ) (h : p (q 3 b) = 13) : b = 7 :=
by sorry

end find_b_l1371_137132


namespace egg_laying_hens_l1371_137155

theorem egg_laying_hens (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) :
  total_chickens = 325 →
  roosters = 28 →
  non_laying_hens = 20 →
  (total_chickens - roosters - non_laying_hens = 277) :=
by
  intros
  sorry

end egg_laying_hens_l1371_137155


namespace line_intersects_ellipse_max_chord_length_l1371_137110

theorem line_intersects_ellipse (m : ℝ) :
  (-2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), (9 * x^2 + 6 * m * x + 2 * m^2 - 8 = 0) ∧ (y = (3 / 2) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) :=
sorry

theorem max_chord_length (m : ℝ) :
  m = 0 → (∃ (A B : ℝ × ℝ),
  ((A.1^2 / 4 + A.2^2 / 9 = 1) ∧ (A.2 = (3 / 2) * A.1 + m)) ∧
  ((B.1^2 / 4 + B.2^2 / 9 = 1) ∧ (B.2 = (3 / 2) * B.1 + m)) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 26 / 3)) :=
sorry

end line_intersects_ellipse_max_chord_length_l1371_137110


namespace lassis_from_12_mangoes_l1371_137127

-- Conditions as definitions in Lean 4
def total_mangoes : ℕ := 12
def damaged_mango_ratio : ℕ := 1 / 6
def lassis_per_pair_mango : ℕ := 11

-- Equation to calculate the lassis
theorem lassis_from_12_mangoes : (total_mangoes - total_mangoes / 6) / 2 * lassis_per_pair_mango = 55 :=
by
  -- calculation steps should go here, but are omitted as per instructions
  sorry

end lassis_from_12_mangoes_l1371_137127


namespace parabola_points_relation_l1371_137164

theorem parabola_points_relation (c y1 y2 y3 : ℝ)
  (h1 : y1 = -(-2)^2 - 2*(-2) + c)
  (h2 : y2 = -(0)^2 - 2*(0) + c)
  (h3 : y3 = -(1)^2 - 2*(1) + c) :
  y1 = y2 ∧ y2 > y3 :=
by
  sorry

end parabola_points_relation_l1371_137164


namespace sandy_final_fish_l1371_137167

theorem sandy_final_fish :
  let Initial_fish := 26
  let Bought_fish := 6
  let Given_away_fish := 10
  let Babies_fish := 15
  let Final_fish := Initial_fish + Bought_fish - Given_away_fish + Babies_fish
  Final_fish = 37 :=
by
  sorry

end sandy_final_fish_l1371_137167


namespace inequality_relation_l1371_137187

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log Q / Real.log 2

theorem inequality_relation : R < Q ∧ Q < P := by
  sorry

end inequality_relation_l1371_137187


namespace sale_price_is_91_percent_of_original_price_l1371_137189

variable (x : ℝ)
variable (h_increase : ∀ p : ℝ, p * 1.4)
variable (h_sale : ∀ p : ℝ, p * 0.65)

/--The sale price of an item is 91% of the original price.-/
theorem sale_price_is_91_percent_of_original_price {x : ℝ} 
  (h_increase : ∀ p, p * 1.4 = 1.40 * p)
  (h_sale : ∀ p, p * 0.65 = 0.65 * p): 
  (0.65 * 1.40 * x = 0.91 * x) := 
by 
  sorry

end sale_price_is_91_percent_of_original_price_l1371_137189


namespace f_of_x_plus_1_f_of_2_f_of_x_l1371_137175

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x := sorry

theorem f_of_2 : f 2 = 3 := sorry

theorem f_of_x (x : ℝ) : f x = x^2 - 1 := sorry

end f_of_x_plus_1_f_of_2_f_of_x_l1371_137175


namespace total_supervisors_correct_l1371_137112

-- Define the number of supervisors on each bus
def bus_supervisors : List ℕ := [4, 5, 3, 6, 7]

-- Define the total number of supervisors
def total_supervisors := bus_supervisors.sum

-- State the theorem to prove that the total number of supervisors is 25
theorem total_supervisors_correct : total_supervisors = 25 :=
by
  sorry -- Proof is to be completed

end total_supervisors_correct_l1371_137112


namespace hyperbola_eccentricity_l1371_137121

theorem hyperbola_eccentricity (a c b : ℝ) (h₀ : b = 3)
  (h₁ : ∃ p, (p = 5) ∧ (a^2 + b^2 = (p : ℝ)^2))
  (h₂ : ∃ f, f = (p : ℝ)) :
  ∃ e, e = c / a ∧ e = 5 / 4 :=
by
  obtain ⟨p, hp, hap⟩ := h₁
  obtain ⟨f, hf⟩ := h₂
  sorry

end hyperbola_eccentricity_l1371_137121


namespace unique_tangent_circle_of_radius_2_l1371_137151

noncomputable def is_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  dist c₁ c₂ = r₁ + r₂

theorem unique_tangent_circle_of_radius_2
    (C1_center C2_center C3_center : ℝ × ℝ)
    (h_C1_C2 : is_tangent C1_center C2_center 1 1)
    (h_C2_C3 : is_tangent C2_center C3_center 1 1)
    (h_C3_C1 : is_tangent C3_center C1_center 1 1):
    ∃! center : ℝ × ℝ, is_tangent center C1_center 2 1 ∧
                        is_tangent center C2_center 2 1 ∧
                        is_tangent center C3_center 2 1 := sorry

end unique_tangent_circle_of_radius_2_l1371_137151


namespace solve_for_3x_plus_9_l1371_137172

theorem solve_for_3x_plus_9 :
  ∀ (x : ℝ), (5 * x - 8 = 15 * x + 18) → 3 * (x + 9) = 96 / 5 :=
by
  intros x h
  sorry

end solve_for_3x_plus_9_l1371_137172


namespace actual_average_height_calculation_l1371_137141

noncomputable def actual_average_height (incorrect_avg_height : ℚ) (number_of_boys : ℕ) (incorrect_recorded_height : ℚ) (actual_height : ℚ) : ℚ :=
  let incorrect_total_height := incorrect_avg_height * number_of_boys
  let overestimated_height := incorrect_recorded_height - actual_height
  let correct_total_height := incorrect_total_height - overestimated_height
  correct_total_height / number_of_boys

theorem actual_average_height_calculation :
  actual_average_height 182 35 166 106 = 180.29 :=
by
  -- The detailed proof is omitted here.
  sorry

end actual_average_height_calculation_l1371_137141


namespace bananas_oranges_equiv_l1371_137168

def bananas_apples_equiv (x y : ℕ) : Prop :=
  4 * x = 3 * y

def apples_oranges_equiv (w z : ℕ) : Prop :=
  9 * w = 5 * z

theorem bananas_oranges_equiv (x y w z : ℕ) (h1 : bananas_apples_equiv x y) (h2 : apples_oranges_equiv y z) :
  bananas_apples_equiv 24 18 ∧ apples_oranges_equiv 18 10 :=
by sorry

end bananas_oranges_equiv_l1371_137168


namespace find_p_l1371_137139

theorem find_p (p: ℝ) (x1 x2: ℝ) (h1: p > 0) (h2: x1^2 + p * x1 + 1 = 0) (h3: x2^2 + p * x2 + 1 = 0) (h4: |x1^2 - x2^2| = p) : p = 5 :=
sorry

end find_p_l1371_137139


namespace inequality_property_l1371_137117

variable {a b : ℝ} (h : a > b) (c : ℝ)

theorem inequality_property : a * |c| ≥ b * |c| :=
sorry

end inequality_property_l1371_137117


namespace quadratic_function_symmetry_l1371_137104

theorem quadratic_function_symmetry (a b x_1 x_2: ℝ) (h_roots: x_1^2 + a * x_1 + b = 0 ∧ x_2^2 + a * x_2 + b = 0)
(h_symmetry: ∀ x, (x - 2015)^2 + a * (x - 2015) + b = (x + 2015 - 2016)^2 + a * (x + 2015 - 2016) + b):
  (x_1 + x_2) / 2 = 2015 :=
sorry

end quadratic_function_symmetry_l1371_137104


namespace natural_numbers_not_divisible_by_5_or_7_l1371_137105

def num_not_divisible_by_5_or_7 (n : ℕ) : ℕ :=
  let num_div_5 := n / 5
  let num_div_7 := n / 7
  let num_div_35 := n / 35
  n - (num_div_5 + num_div_7 - num_div_35)

theorem natural_numbers_not_divisible_by_5_or_7 :
  num_not_divisible_by_5_or_7 999 = 686 :=
by sorry

end natural_numbers_not_divisible_by_5_or_7_l1371_137105


namespace roots_imply_value_l1371_137171

noncomputable def value_of_expression (a b c : ℝ) : ℝ :=
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b)

theorem roots_imply_value {a b c : ℝ} 
  (h1 : a + b + c = 15) 
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 10) 
  : value_of_expression a b c = 175 / 11 :=
sorry

end roots_imply_value_l1371_137171


namespace tiling_not_possible_l1371_137193

-- Definitions for the puzzle pieces
inductive Piece
| L | T | I | Z | O

-- Function to check if tiling a rectangle is possible
noncomputable def can_tile_rectangle (pieces : List Piece) : Prop :=
  ∀ (width height : ℕ), width * height % 4 = 0 → ∃ (tiling : List (Piece × ℕ × ℕ)), sorry

theorem tiling_not_possible : ¬ can_tile_rectangle [Piece.L, Piece.T, Piece.I, Piece.Z, Piece.O] :=
sorry

end tiling_not_possible_l1371_137193
