import Mathlib

namespace inequality_am_gm_l269_26987

variable {u v : ℝ}

theorem inequality_am_gm (hu : 0 < u) (hv : 0 < v) : u ^ 3 + v ^ 3 ≥ u ^ 2 * v + v ^ 2 * u := by
  sorry

end inequality_am_gm_l269_26987


namespace cistern_empty_time_without_tap_l269_26907

noncomputable def leak_rate (L : ℕ) : Prop :=
  let tap_rate := 4
  let cistern_volume := 480
  let empty_time_with_tap := 24
  let empty_rate_net := cistern_volume / empty_time_with_tap
  L - tap_rate = empty_rate_net

theorem cistern_empty_time_without_tap (L : ℕ) (h : leak_rate L) :
  480 / L = 20 := by
  -- placeholder for the proof
  sorry

end cistern_empty_time_without_tap_l269_26907


namespace equal_sets_l269_26954

def M : Set ℝ := {x | x^2 + 16 = 0}
def N : Set ℝ := {x | x^2 + 6 = 0}

theorem equal_sets : M = N := by
  sorry

end equal_sets_l269_26954


namespace range_of_a_l269_26953

noncomputable def f (x a : ℝ) : ℝ := 
  x * (a - 1 / Real.exp x)

noncomputable def gx (x : ℝ) : ℝ :=
  (1 + x) / Real.exp x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 a = 0 ∧ f x2 a = 0) →
  a < 2 / Real.exp 1 :=
by
  sorry

end range_of_a_l269_26953


namespace costume_processing_time_l269_26971

theorem costume_processing_time (x : ℕ) : 
  (300 - 60) / (2 * x) + 60 / x = 9 → (60 / x) + (240 / (2 * x)) = 9 :=
by
  sorry

end costume_processing_time_l269_26971


namespace car_mileage_city_l269_26958

theorem car_mileage_city {h c t : ℝ} (H1: 448 = h * t) (H2: 336 = c * t) (H3: c = h - 6) : c = 18 :=
sorry

end car_mileage_city_l269_26958


namespace last_digit_of_power_sum_l269_26990

theorem last_digit_of_power_sum (m : ℕ) (hm : 0 < m) : (2^(m + 2006) + 2^m) % 10 = 0 := 
sorry

end last_digit_of_power_sum_l269_26990


namespace cardProblem_l269_26910

structure InitialState where
  jimmy_cards : ℕ
  bob_cards : ℕ
  sarah_cards : ℕ

structure UpdatedState where
  jimmy_cards_final : ℕ
  sarah_cards_final : ℕ
  sarahs_friends_cards : ℕ

def cardProblemSolved (init : InitialState) (final : UpdatedState) : Prop :=
  let bob_initial := init.bob_cards + 6
  let bob_to_sarah := bob_initial / 3
  let bob_final := bob_initial - bob_to_sarah
  let sarah_initial := init.sarah_cards + bob_to_sarah
  let sarah_friends := sarah_initial / 3
  let sarah_final := sarah_initial - 3 * sarah_friends
  let mary_cards := 2 * 6
  let jimmy_final := init.jimmy_cards - 6 - mary_cards
  let sarah_to_tim := 0 -- since Sarah can't give fractional cards
  (final.jimmy_cards_final = jimmy_final) ∧ 
  (final.sarah_cards_final = sarah_final - sarah_to_tim) ∧ 
  (final.sarahs_friends_cards = sarah_friends)

theorem cardProblem : 
  cardProblemSolved 
    { jimmy_cards := 68, bob_cards := 5, sarah_cards := 7 }
    { jimmy_cards_final := 50, sarah_cards_final := 1, sarahs_friends_cards := 3 } :=
by 
  sorry

end cardProblem_l269_26910


namespace find_y_value_l269_26913

noncomputable def y_value (y : ℝ) : Prop :=
  let side1_sq_area := 9 * y^2
  let side2_sq_area := 36 * y^2
  let triangle_area := 9 * y^2
  (side1_sq_area + side2_sq_area + triangle_area = 1000)

theorem find_y_value (y : ℝ) : y_value y → y = 10 * Real.sqrt 3 / 3 :=
sorry

end find_y_value_l269_26913


namespace parents_present_l269_26904

theorem parents_present (pupils teachers total_people parents : ℕ)
  (h_pupils : pupils = 724)
  (h_teachers : teachers = 744)
  (h_total_people : total_people = 1541) :
  parents = total_people - (pupils + teachers) :=
sorry

end parents_present_l269_26904


namespace inequality_holds_for_all_real_numbers_l269_26916

theorem inequality_holds_for_all_real_numbers (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (k ∈ Set.Icc (-3 : ℝ) 0) := 
sorry

end inequality_holds_for_all_real_numbers_l269_26916


namespace cost_to_fill_pool_l269_26909

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l269_26909


namespace transformed_inequality_l269_26932

theorem transformed_inequality (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by
  sorry

end transformed_inequality_l269_26932


namespace solution_l269_26961

theorem solution (a b : ℝ) (h1 : a^2 + 2 * a - 2016 = 0) (h2 : b^2 + 2 * b - 2016 = 0) :
  a^2 + 3 * a + b = 2014 := 
sorry

end solution_l269_26961


namespace remainder_when_sum_divided_l269_26919

theorem remainder_when_sum_divided (p q : ℕ) (m n : ℕ) (hp : p = 80 * m + 75) (hq : q = 120 * n + 115) :
  (p + q) % 40 = 30 := 
by sorry

end remainder_when_sum_divided_l269_26919


namespace has_zero_when_a_gt_0_l269_26929

noncomputable def f (x a : ℝ) : ℝ :=
  x * Real.log (x - 1) - a

theorem has_zero_when_a_gt_0 (a : ℝ) (h : a > 0) :
  ∃ x0 : ℝ, f x0 a = 0 ∧ 2 < x0 :=
sorry

end has_zero_when_a_gt_0_l269_26929


namespace exists_xi_l269_26908

variable (f : ℝ → ℝ)
variable (hf_diff : ∀ x, DifferentiableAt ℝ f x)
variable (hf_twice_diff : ∀ x, DifferentiableAt ℝ (deriv f) x)
variable (hf₀ : f 0 = 2)
variable (hf_prime₀ : deriv f 0 = -2)
variable (hf₁ : f 1 = 1)

theorem exists_xi (h0 : f 0 = 2) (h1 : deriv f 0 = -2) (h2 : f 1 = 1) :
  ∃ ξ ∈ Set.Ioo 0 1, f ξ * deriv f ξ + deriv (deriv f) ξ = 0 :=
sorry

end exists_xi_l269_26908


namespace truth_of_compound_proposition_l269_26959

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, x^2 > 0

theorem truth_of_compound_proposition : p ∧ ¬ q :=
by
  sorry

end truth_of_compound_proposition_l269_26959


namespace third_side_correct_length_longest_side_feasibility_l269_26943

-- Definitions for part (a)
def adjacent_side_length : ℕ := 40
def total_fencing_length : ℕ := 140

-- Define third side given the conditions
def third_side_length : ℕ :=
  total_fencing_length - (2 * adjacent_side_length)

-- Problem (a)
theorem third_side_correct_length (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  third_side_length = 60 :=
sorry

-- Definitions for part (b)
def longest_side_possible1 : ℕ := 85
def longest_side_possible2 : ℕ := 65

-- Problem (b)
theorem longest_side_feasibility (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  ¬ (longest_side_possible1 = 85 ∧ longest_side_possible2 = 65) :=
sorry

end third_side_correct_length_longest_side_feasibility_l269_26943


namespace total_pieces_on_chessboard_l269_26977

-- Given conditions about initial chess pieces and lost pieces.
def initial_pieces_each : Nat := 16
def pieces_lost_arianna : Nat := 3
def pieces_lost_samantha : Nat := 9

-- The remaining pieces for each player.
def remaining_pieces_arianna : Nat := initial_pieces_each - pieces_lost_arianna
def remaining_pieces_samantha : Nat := initial_pieces_each - pieces_lost_samantha

-- The total remaining pieces on the chessboard.
def total_remaining_pieces : Nat := remaining_pieces_arianna + remaining_pieces_samantha

-- The theorem to prove
theorem total_pieces_on_chessboard : total_remaining_pieces = 20 :=
by
  sorry

end total_pieces_on_chessboard_l269_26977


namespace length_of_DC_l269_26925

theorem length_of_DC (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30) (h2 : angle_ADB = pi / 2) (h3 : sin_A = 3 / 5) (h4 : sin_C = 1 / 4) :
  ∃ DC : ℝ, DC = 18 * Real.sqrt 15 :=
by
  sorry

end length_of_DC_l269_26925


namespace simplify_expression_l269_26939

variable (x : ℝ) (hx : x ≠ 0)

theorem simplify_expression : 
  ( (x + 3)^2 + (x + 3) * (x - 3) ) / (2 * x) = x + 3 := by
  sorry

end simplify_expression_l269_26939


namespace find_all_good_sets_l269_26912

def is_good_set (A : Finset ℕ) : Prop :=
  (∀ (a b c : ℕ), a ∈ A → b ∈ A → c ∈ A → a ≠ b → b ≠ c → a ≠ c → Nat.gcd a (Nat.gcd b c) = 1) ∧
  (∀ (b c : ℕ), b ∈ A → c ∈ A → b ≠ c → ∃ (a : ℕ), a ∈ A ∧ a ≠ b ∧ a ≠ c ∧ (b * c) % a = 0)

theorem find_all_good_sets : ∀ (A : Finset ℕ), is_good_set A ↔ 
  (A = {a, b, a * b} ∧ Nat.gcd a b = 1) ∨ 
  ∃ (p q r : ℕ), Nat.gcd p q = 1 ∧ Nat.gcd q r = 1 ∧ Nat.gcd r p = 1 ∧ A = {p * q, q * r, r * p} :=
by
  sorry

end find_all_good_sets_l269_26912


namespace problem_solution_l269_26951

theorem problem_solution (k m : ℕ) (h1 : 30^k ∣ 929260) (h2 : 20^m ∣ 929260) : (3^k - k^3) + (2^m - m^3) = 2 := 
by sorry

end problem_solution_l269_26951


namespace four_person_apartments_l269_26985

theorem four_person_apartments : 
  ∃ x : ℕ, 
    (4 * (10 + 20 * 2 + 4 * x)) * 3 / 4 = 210 → x = 5 :=
by
  sorry

end four_person_apartments_l269_26985


namespace number_of_pairs_l269_26964

noncomputable def number_of_ordered_pairs (n : ℕ) : ℕ :=
  if n = 5 then 8 else 0

theorem number_of_pairs (f m: ℕ) : f ≥ 0 ∧ m ≥ 0 → number_of_ordered_pairs 5 = 8 :=
by
  intro h
  sorry

end number_of_pairs_l269_26964


namespace minimum_value_inequality_l269_26968

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end minimum_value_inequality_l269_26968


namespace largest_integer_same_cost_l269_26940

def cost_base_10 (n : ℕ) : ℕ :=
  (n.digits 10).sum

def cost_base_2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem largest_integer_same_cost : ∃ n < 1000, 
  cost_base_10 n = cost_base_2 n ∧
  ∀ m < 1000, cost_base_10 m = cost_base_2 m → n ≥ m :=
sorry

end largest_integer_same_cost_l269_26940


namespace probability_of_red_ball_l269_26996

-- Define the conditions
def num_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

-- Calculate the probability
def probability_drawing_red_ball : ℚ := red_balls / num_balls

-- The theorem statement to be proven
theorem probability_of_red_ball : probability_drawing_red_ball = 2 / 3 :=
by
  sorry

end probability_of_red_ball_l269_26996


namespace alpine_school_math_students_l269_26960

theorem alpine_school_math_students (total_players : ℕ) (physics_players : ℕ) (both_players : ℕ) :
  total_players = 15 → physics_players = 9 → both_players = 4 → 
  ∃ math_players : ℕ, math_players = total_players - (physics_players - both_players) + both_players := by
  sorry

end alpine_school_math_students_l269_26960


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l269_26973

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l269_26973


namespace fuel_calculation_l269_26942

def total_fuel_needed (empty_fuel_per_mile people_fuel_per_mile bag_fuel_per_mile num_passengers num_crew bags_per_person miles : ℕ) : ℕ :=
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let total_fuel_per_mile := empty_fuel_per_mile + people_fuel_per_mile * total_people + bag_fuel_per_mile * total_bags
  total_fuel_per_mile * miles

theorem fuel_calculation :
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 :=
by
  sorry

end fuel_calculation_l269_26942


namespace older_brother_catches_up_l269_26920

-- Define the initial conditions and required functions
def younger_brother_steps_before_chase : ℕ := 10
def time_per_3_steps_older := 1  -- in seconds
def time_per_4_steps_younger := 1  -- in seconds 
def dist_older_in_5_steps : ℕ := 7  -- 7d_younger / 5
def dist_younger_in_7_steps : ℕ := 5
def speed_older : ℕ := 3 * dist_older_in_5_steps / 5  -- steps/second 
def speed_younger : ℕ := 4 * dist_younger_in_7_steps / 7  -- steps/second

theorem older_brother_catches_up : ∃ n : ℕ, n = 150 :=
by sorry  -- final theorem statement with proof omitted

end older_brother_catches_up_l269_26920


namespace no_valid_N_for_case1_valid_N_values_for_case2_l269_26976

variable (P R N : ℕ)
variable (N_less_than_40 : N < 40)
variable (avg_all : ℕ)
variable (avg_promoted : ℕ)
variable (avg_repeaters : ℕ)
variable (new_avg_promoted : ℕ)
variable (new_avg_repeaters : ℕ)

variables
  (promoted_condition : (71 * P + 56 * R) / N = 66)
  (increase_condition : (76 * P) / (P + R) = 75 ∧ (61 * R) / (P + R) = 59)
  (equation1 : 71 * P = 2 * R)
  (equation2: P + R = N)

-- Proof for part (a)
theorem no_valid_N_for_case1 
  (new_avg_promoted' : ℕ := 75) 
  (new_avg_repeaters' : ℕ := 59)
  : ∀ N, ¬ N < 40 ∨ ¬ ((76 * P) / (P + R) = new_avg_promoted' ∧ (61 * R) / (P + R) = new_avg_repeaters') := 
  sorry

-- Proof for part (b)
theorem valid_N_values_for_case2
  (possible_N_values : Finset ℕ := {6, 12, 18, 24, 30, 36})
  (new_avg_promoted'' : ℕ := 79)
  (new_avg_repeaters'' : ℕ := 47)
  : ∀ N, N ∈ possible_N_values ↔ (((76 * P) / (P + R) = new_avg_promoted'') ∧ (61 * R) / (P + R) = new_avg_repeaters'') := 
  sorry

end no_valid_N_for_case1_valid_N_values_for_case2_l269_26976


namespace molecular_weight_is_62_024_l269_26974

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_atoms_H : ℕ := 2
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

def molecular_weight_compound : ℝ :=
  num_atoms_H * atomic_weight_H + num_atoms_C * atomic_weight_C + num_atoms_O * atomic_weight_O

theorem molecular_weight_is_62_024 :
  molecular_weight_compound = 62.024 :=
by
  have h_H := num_atoms_H * atomic_weight_H
  have h_C := num_atoms_C * atomic_weight_C
  have h_O := num_atoms_O * atomic_weight_O
  have h_sum := h_H + h_C + h_O
  show molecular_weight_compound = 62.024
  sorry

end molecular_weight_is_62_024_l269_26974


namespace minimize_quadratic_l269_26952

theorem minimize_quadratic : ∃ x : ℝ, x = -4 ∧ ∀ y : ℝ, x^2 + 8*x + 7 ≤ y^2 + 8*y + 7 :=
by 
  use -4
  sorry

end minimize_quadratic_l269_26952


namespace sum_a5_a6_a7_l269_26986

variable (a : ℕ → ℝ) (q : ℝ)

-- Assumptions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

axiom sum_a1_a2_a3 : a 1 + a 2 + a 3 = 1
axiom sum_a2_a3_a4 : a 2 + a 3 + a 4 = 2

-- The theorem we want to prove
theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 16 := sorry

end sum_a5_a6_a7_l269_26986


namespace function_increasing_l269_26911

variable {α : Type*} [LinearOrderedField α]

def is_increasing (f : α → α) : Prop :=
  ∀ x y : α, x < y → f x < f y

theorem function_increasing (f : α → α) (h : ∀ x1 x2 : α, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1) :
  is_increasing f :=
by
  sorry

end function_increasing_l269_26911


namespace remainder_of_3_pow_21_mod_11_l269_26982

theorem remainder_of_3_pow_21_mod_11 : (3^21 % 11) = 3 := 
by {
  sorry
}

end remainder_of_3_pow_21_mod_11_l269_26982


namespace minimum_value_l269_26931

open Real

theorem minimum_value (x : ℝ) (hx : x > 2) : 
  ∃ y ≥ 4 * Real.sqrt 2, ∀ z, (z = (x + 6) / (Real.sqrt (x - 2)) → y ≤ z) := 
sorry

end minimum_value_l269_26931


namespace x4_value_l269_26955

/-- Define x_n sequence based on given initial value and construction rules -/
def x_n (n : ℕ) : ℕ :=
  if n = 1 then 27
  else if n = 2 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1
  else if n = 3 then 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1 -- Need to generalize for actual sequence definition
  else if n = 4 then 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2 + 1
  else 0 -- placeholder for general case (not needed here)

/-- Prove that x_4 = 23 given x_1=27 and the sequence construction criteria --/
theorem x4_value : x_n 4 = 23 :=
by
  -- Proof not required, hence sorry is used
  sorry

end x4_value_l269_26955


namespace age_in_1988_equals_sum_of_digits_l269_26949

def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

def age_in_1988 (birth_year : ℕ) : ℕ := 1988 - birth_year

def sum_of_digits (x y : ℕ) : ℕ := 1 + 9 + x + y

theorem age_in_1988_equals_sum_of_digits (x y : ℕ) (h0 : 0 ≤ x) (h1 : x ≤ 9) (h2 : 0 ≤ y) (h3 : y ≤ 9) 
  (h4 : age_in_1988 (birth_year x y) = sum_of_digits x y) :
  age_in_1988 (birth_year x y) = 22 :=
by {
  sorry
}

end age_in_1988_equals_sum_of_digits_l269_26949


namespace compute_exponent_multiplication_l269_26969

theorem compute_exponent_multiplication : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_exponent_multiplication_l269_26969


namespace cos_half_pi_plus_alpha_l269_26956

theorem cos_half_pi_plus_alpha (α : ℝ) (h : Real.sin (π - α) = 1 / 3) : Real.cos (π / 2 + α) = - (1 / 3) :=
by
  sorry

end cos_half_pi_plus_alpha_l269_26956


namespace time_to_fill_pool_l269_26998

theorem time_to_fill_pool (V : ℕ) (n : ℕ) (r : ℕ) (fill_rate_per_hour : ℕ) :
  V = 24000 → 
  n = 4 →
  r = 25 → -- 2.5 gallons per minute expressed as 25/10 gallons
  fill_rate_per_hour = (n * r * 6) → -- since 6 * 10 = 60 (to convert per minute rate to per hour, we divide so r is 25 instead of 2.5)
  V / fill_rate_per_hour = 40 :=
by
  sorry

end time_to_fill_pool_l269_26998


namespace add_fractions_l269_26915

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = 5 / 8 :=
by
  sorry

end add_fractions_l269_26915


namespace max_area_triangle_PAB_l269_26965

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2 / 16) + (y^2 / 9) = 1

def point_A : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)

theorem max_area_triangle_PAB (P : ℝ × ℝ) (hP : ellipse_eq P.1 P.2) : 
  ∃ S, S = 6 * (sqrt 2 + 1) := 
sorry

end max_area_triangle_PAB_l269_26965


namespace coffee_maker_capacity_l269_26906

theorem coffee_maker_capacity (x : ℝ) (h : 0.36 * x = 45) : x = 125 :=
sorry

end coffee_maker_capacity_l269_26906


namespace unknown_number_is_10_l269_26938

def operation_e (x y : ℕ) : ℕ := 2 * x * y

theorem unknown_number_is_10 (n : ℕ) (h : operation_e 8 (operation_e n 5) = 640) : n = 10 :=
by
  sorry

end unknown_number_is_10_l269_26938


namespace range_of_k_l269_26979

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end range_of_k_l269_26979


namespace sam_morning_run_distance_l269_26905

variable (x : ℝ) -- The distance of Sam's morning run in miles

theorem sam_morning_run_distance (h1 : ∀ y, y = 2 * x) (h2 : 12 = 12) (h3 : x + 2 * x + 12 = 18) : x = 2 :=
by sorry

end sam_morning_run_distance_l269_26905


namespace square_area_l269_26967

theorem square_area (l w x : ℝ) (h1 : 2 * (l + w) = 20) (h2 : l = x / 2) (h3 : w = x / 4) :
  x^2 = 1600 / 9 :=
by
  sorry

end square_area_l269_26967


namespace Dave_earning_l269_26927

def action_games := 3
def adventure_games := 2
def role_playing_games := 3

def price_action := 6
def price_adventure := 5
def price_role_playing := 7

def earning_from_action_games := action_games * price_action
def earning_from_adventure_games := adventure_games * price_adventure
def earning_from_role_playing_games := role_playing_games * price_role_playing

def total_earning := earning_from_action_games + earning_from_adventure_games + earning_from_role_playing_games

theorem Dave_earning : total_earning = 49 := by
  show total_earning = 49
  sorry

end Dave_earning_l269_26927


namespace largest_common_term_l269_26994

theorem largest_common_term (b : ℕ) : 
  (b < 1000) ∧ (b % 5 = 4) ∧ (b % 11 = 7) → b = 964 :=
by
  intros h
  sorry

end largest_common_term_l269_26994


namespace original_number_of_men_l269_26944

/--A group of men decided to complete a work in 6 days. 
 However, 4 of them became absent, and the remaining men finished the work in 12 days. 
 Given these conditions, we need to prove that the original number of men was 8. --/
theorem original_number_of_men 
  (x : ℕ) -- original number of men
  (h1 : x * 6 = (x - 4) * 12) -- total work remains the same
  : x = 8 := 
sorry

end original_number_of_men_l269_26944


namespace abs_eq_imp_b_eq_2_l269_26902

theorem abs_eq_imp_b_eq_2 (b : ℝ) (h : |1 - b| = |3 - b|) : b = 2 := 
sorry

end abs_eq_imp_b_eq_2_l269_26902


namespace three_consecutive_arithmetic_l269_26935

def seq (n : ℕ) : ℝ := 
  if n % 2 = 1 then (n : ℝ)
  else 2 * 3^(n / 2 - 1)

theorem three_consecutive_arithmetic (m : ℕ) (h_m : seq m + seq (m+2) = 2 * seq (m+1)) : m = 1 :=
  sorry

end three_consecutive_arithmetic_l269_26935


namespace max_value_of_a1_l269_26992

theorem max_value_of_a1 (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h_distinct : ∀ i j, i ≠ j → (i ≠ a1 → i ≠ a2 → i ≠ a3 → i ≠ a4 → i ≠ a5 → i ≠ a6 → i ≠ a7)) 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 159) : a1 ≤ 19 :=
by
  sorry

end max_value_of_a1_l269_26992


namespace calc_total_push_ups_correct_l269_26933

-- Definitions based on conditions
def sets : ℕ := 9
def push_ups_per_set : ℕ := 12
def reduced_push_ups : ℕ := 8

-- Calculate total push-ups considering the reduction in the ninth set
def total_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (reduced_push_ups : ℕ) : ℕ :=
  (sets - 1) * push_ups_per_set + (push_ups_per_set - reduced_push_ups)

-- Theorem statement
theorem calc_total_push_ups_correct :
  total_push_ups sets push_ups_per_set reduced_push_ups = 100 :=
by
  sorry

end calc_total_push_ups_correct_l269_26933


namespace interest_rate_is_10_percent_l269_26930

theorem interest_rate_is_10_percent (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) 
  (hP : P = 9999.99999999988) 
  (ht : t = 1) 
  (hd : d = 25)
  : P * (1 + r / 2)^(2 * t) - P - (P * r * t) = d → r = 0.1 :=
by
  intros h
  rw [hP, ht, hd] at h
  sorry

end interest_rate_is_10_percent_l269_26930


namespace combined_teaching_years_l269_26978

def Adrienne_Yrs : ℕ := 22
def Virginia_Yrs : ℕ := Adrienne_Yrs + 9
def Dennis_Yrs : ℕ := 40

theorem combined_teaching_years :
  Adrienne_Yrs + Virginia_Yrs + Dennis_Yrs = 93 := by
  -- Proof omitted
  sorry

end combined_teaching_years_l269_26978


namespace zoo_recovery_time_l269_26991

theorem zoo_recovery_time (lions rhinos recover_time : ℕ) (total_animals : ℕ) (total_time : ℕ)
    (h_lions : lions = 3) (h_rhinos : rhinos = 2) (h_recover_time : recover_time = 2)
    (h_total_animals : total_animals = lions + rhinos) (h_total_time : total_time = total_animals * recover_time) :
    total_time = 10 :=
by
  rw [h_lions, h_rhinos] at h_total_animals
  rw [h_total_animals, h_recover_time] at h_total_time
  exact h_total_time

end zoo_recovery_time_l269_26991


namespace max_gross_profit_price_l269_26999

def purchase_price : ℝ := 20
def Q (P : ℝ) : ℝ := 8300 - 170 * P - P^2
def L (P : ℝ) : ℝ := (8300 - 170 * P - P^2) * (P - 20)

theorem max_gross_profit_price : ∃ P : ℝ, (∀ x : ℝ, L x ≤ L P) ∧ P = 30 :=
by
  sorry

end max_gross_profit_price_l269_26999


namespace jordon_machine_number_l269_26993

theorem jordon_machine_number : 
  ∃ x : ℝ, (2 * x + 3 = 27) ∧ x = 12 :=
by
  sorry

end jordon_machine_number_l269_26993


namespace determine_OP_l269_26950

theorem determine_OP 
  (a b c d k : ℝ)
  (h1 : k * b ≤ c) 
  (h2 : (A : ℝ) = a)
  (h3 : (B : ℝ) = k * b)
  (h4 : (C : ℝ) = c)
  (h5 : (D : ℝ) = k * d)
  (AP_PD : ∀ (P : ℝ), (a - P) / (P - k * d) = k * (k * b - P) / (P - c))
  :
  ∃ P : ℝ, P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
sorry

end determine_OP_l269_26950


namespace part1_part2_l269_26921

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2

theorem part1 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → f x a > 3*a*x) → a < 2*Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) :
  ∀ x : ℝ,
    ((a = 0) → x > 2) ∧
    ((a > 0) → (x < -1/a ∨ x > 2)) ∧
    ((-1/2 < a ∧ a < 0) → (2 < x ∧ x < -1/a)) ∧
    ((a = -1/2) → false) ∧
    ((a < -1/2) → (-1/a < x ∧ x < 2)) :=
sorry

end part1_part2_l269_26921


namespace certain_number_l269_26917

theorem certain_number (a b : ℕ) (n : ℕ) 
  (h1: a % n = 0) (h2: b % n = 0) 
  (h3: b = a + 9 * n)
  (h4: b = a + 126) : n = 14 :=
by
  sorry

end certain_number_l269_26917


namespace find_ten_x_l269_26901

theorem find_ten_x (x : ℝ) 
  (h : 4^(2*x) + 2^(-x) + 1 = (129 + 8 * Real.sqrt 2) * (4^x + 2^(- x) - 2^x)) : 
  10 * x = 35 := 
sorry

end find_ten_x_l269_26901


namespace tank_full_capacity_l269_26946

-- Define the conditions
def gas_tank_initially_full_fraction : ℚ := 4 / 5
def gas_tank_after_usage_fraction : ℚ := 1 / 3
def used_gallons : ℚ := 18

-- Define the statement that translates to "How many gallons does this tank hold when it is full?"
theorem tank_full_capacity (x : ℚ) : 
  gas_tank_initially_full_fraction * x - gas_tank_after_usage_fraction * x = used_gallons → 
  x = 270 / 7 :=
sorry

end tank_full_capacity_l269_26946


namespace hotel_towels_l269_26936

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l269_26936


namespace value_of_b_conditioned_l269_26972

theorem value_of_b_conditioned
  (b: ℝ) 
  (h0 : 0 < b ∧ b < 7)
  (h1 : (1 / 2) * (8 - b) * (b - 8) / ((1 / 2) * (b / 2) * b) = 4 / 9):
  b = 4 := 
sorry

end value_of_b_conditioned_l269_26972


namespace original_average_age_older_l269_26926

theorem original_average_age_older : 
  ∀ (n : ℕ) (T : ℕ), (T = n * 40) →
  (T + 408) / (n + 12) = 36 →
  40 - 36 = 4 :=
by
  intros n T hT hNewAvg
  sorry

end original_average_age_older_l269_26926


namespace mass_percentage_H_in_NH4I_is_correct_l269_26900

noncomputable def molar_mass_NH4I : ℝ := 1 * 14.01 + 4 * 1.01 + 1 * 126.90

noncomputable def mass_H_in_NH4I : ℝ := 4 * 1.01

noncomputable def mass_percentage_H_in_NH4I : ℝ := (mass_H_in_NH4I / molar_mass_NH4I) * 100

theorem mass_percentage_H_in_NH4I_is_correct :
  abs (mass_percentage_H_in_NH4I - 2.79) < 0.01 := by
  sorry

end mass_percentage_H_in_NH4I_is_correct_l269_26900


namespace greatest_whole_number_inequality_l269_26962

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end greatest_whole_number_inequality_l269_26962


namespace leigh_path_length_l269_26995

theorem leigh_path_length :
  let north := 10
  let south := 40
  let west := 60
  let east := 20
  let net_south := south - north
  let net_west := west - east
  let distance := Real.sqrt (net_south^2 + net_west^2)
  distance = 50 := 
by sorry

end leigh_path_length_l269_26995


namespace sum_of_fractions_l269_26914

theorem sum_of_fractions :
  (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 :=
by
  sorry

end sum_of_fractions_l269_26914


namespace largest_root_is_1011_l269_26903

theorem largest_root_is_1011 (a b c d x : ℝ) 
  (h1 : a + d = 2022) 
  (h2 : b + c = 2022) 
  (h3 : a ≠ c) 
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) : 
  x = 1011 := 
sorry

end largest_root_is_1011_l269_26903


namespace nature_of_roots_l269_26984

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 + 3 * x^2 - 8 * x + 16

theorem nature_of_roots : (∀ x : ℝ, x < 0 → P x > 0) ∧ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ P x = 0 := 
by
  sorry

end nature_of_roots_l269_26984


namespace lowest_point_graph_of_y_l269_26981

theorem lowest_point_graph_of_y (x : ℝ) (h : x > -1) :
  (x, (x^2 + 2 * x + 2) / (x + 1)) = (0, 2) ∧ ∀ y > -1, ( (y^2 + 2 * y + 2) / (y + 1) >= 2) := 
sorry

end lowest_point_graph_of_y_l269_26981


namespace complement_of_union_l269_26945

open Set

namespace Proof

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of the union of sets A and B with respect to U
theorem complement_of_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 3}) (hB : B = {3, 5}) : 
  U \ (A ∪ B) = {0, 2, 4} :=
by {
  sorry
}

end Proof

end complement_of_union_l269_26945


namespace opposite_of_three_l269_26983

theorem opposite_of_three : -3 = -3 := by
  -- The condition we have identified is the given number 3.
  -- We will directly state that the opposite of 3 is -3.
  -- This proof is trivial as we are directly replacing 3 with -3 to match the problem statement.
  rfl

end opposite_of_three_l269_26983


namespace rods_in_one_mile_l269_26928

theorem rods_in_one_mile (mile_to_furlong : ℕ) (furlong_to_rod : ℕ) (mile_eq : 1 = 8 * mile_to_furlong) (furlong_eq: 1 = 50 * furlong_to_rod) : 
  (1 * 8 * 50 = 400) :=
by
  sorry

end rods_in_one_mile_l269_26928


namespace class_size_l269_26948

theorem class_size (n : ℕ) (h1 : 85 - 33 + 90 - 40 = 102) (h2 : (102 : ℚ) / n = 1.5): n = 68 :=
by
  sorry

end class_size_l269_26948


namespace max_distance_circle_to_line_l269_26963

open Real

theorem max_distance_circle_to_line :
  let circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y + 1 = 0
  let line_eq (x y : ℝ) := x - y = 2
  ∃ (M : ℝ), (∀ x y, circle_eq x y → ∀ (d : ℝ), (line_eq x y → M ≤ d)) ∧ M = sqrt 2 + 1 :=
by
  sorry

end max_distance_circle_to_line_l269_26963


namespace sum_of_consecutive_perfect_squares_l269_26989

theorem sum_of_consecutive_perfect_squares (k : ℕ) (h_pos : 0 < k)
  (h_eq : 2 * k^2 + 2 * k + 1 = 181) : k = 9 ∧ (k + 1) = 10 := by
  sorry

end sum_of_consecutive_perfect_squares_l269_26989


namespace max_sum_distinct_factors_2029_l269_26966

theorem max_sum_distinct_factors_2029 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2029 ∧ A + B + C = 297 :=
by
  sorry

end max_sum_distinct_factors_2029_l269_26966


namespace coefficient_of_x_in_expansion_l269_26937

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end coefficient_of_x_in_expansion_l269_26937


namespace total_flour_l269_26997

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end total_flour_l269_26997


namespace average_weight_l269_26988

theorem average_weight (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 47) (h3 : B = 39) : (A + B + C) / 3 = 45 := 
  sorry

end average_weight_l269_26988


namespace triangle_side_length_l269_26922

theorem triangle_side_length (B C : Real) (b c : Real) 
  (h1 : c * Real.cos B = 12) 
  (h2 : b * Real.sin C = 5) 
  (h3 : b * Real.sin B = 5) : 
  c = 13 := 
sorry

end triangle_side_length_l269_26922


namespace find_reggie_long_shots_l269_26918

-- Define the constants used in the problem
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shooting results
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := sorry -- we need to find this

-- Define Reggie's brother's shooting results
def brother_long_shots : ℕ := 4

-- Given conditions
def reggie_total_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
def brother_total_points := brother_long_shots * long_shot_points

def reggie_lost_by_2_points := reggie_total_points + 2 = brother_total_points

-- The theorem we need to prove
theorem find_reggie_long_shots : reggie_long_shots = 1 :=
by
  sorry

end find_reggie_long_shots_l269_26918


namespace probability_blue_or_purple_l269_26941

def total_jelly_beans : ℕ := 35
def blue_jelly_beans : ℕ := 7
def purple_jelly_beans : ℕ := 10

theorem probability_blue_or_purple : (blue_jelly_beans + purple_jelly_beans: ℚ) / total_jelly_beans = 17 / 35 := 
by sorry

end probability_blue_or_purple_l269_26941


namespace hexagon_diagonals_l269_26980

theorem hexagon_diagonals : (6 * (6 - 3)) / 2 = 9 := 
by 
  sorry

end hexagon_diagonals_l269_26980


namespace find_a_value_l269_26957

namespace Proof

-- Define the context and variables
variables (a b c : ℝ)
variables (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variables (h2 : a * 15 * 2 = 4)

-- State the theorem we want to prove
theorem find_a_value: a = 6 :=
by
  sorry

end Proof

end find_a_value_l269_26957


namespace david_boxes_l269_26970

-- Conditions
def number_of_dogs_per_box : ℕ := 4
def total_number_of_dogs : ℕ := 28

-- Problem
theorem david_boxes : total_number_of_dogs / number_of_dogs_per_box = 7 :=
by
  sorry

end david_boxes_l269_26970


namespace nth_inequality_l269_26923

theorem nth_inequality (n : ℕ) (x : ℝ) (h : x > 0) : x + n^n / x^n ≥ n + 1 := 
sorry

end nth_inequality_l269_26923


namespace binary_to_decimal_l269_26934

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l269_26934


namespace range_of_t_l269_26947

noncomputable def f : ℝ → ℝ := sorry

axiom f_symmetric (x : ℝ) : f (x - 3) = f (-x - 3)
axiom f_ln_definition (x : ℝ) (h : x ≤ -3) : f x = Real.log (-x)

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f (Real.sin x - t) > f (3 * Real.sin x - 1)) ↔ (t < -1 ∨ t > 9) := sorry

end range_of_t_l269_26947


namespace find_x_for_fraction_equality_l269_26924

theorem find_x_for_fraction_equality (x : ℝ) : 
  (4 + 2 * x) / (7 + x) = (2 + x) / (3 + x) ↔ (x = -2 ∨ x = 1) := by
  sorry

end find_x_for_fraction_equality_l269_26924


namespace fedya_incorrect_l269_26975

theorem fedya_incorrect 
  (a b c d : ℕ) 
  (a_ends_in_9 : a % 10 = 9)
  (b_ends_in_7 : b % 10 = 7)
  (c_ends_in_3 : c % 10 = 3)
  (d_is_1 : d = 1) : 
  a ≠ b * c + d :=
by {
  sorry
}

end fedya_incorrect_l269_26975
