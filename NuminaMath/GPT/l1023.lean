import Mathlib

namespace field_ratio_l1023_102345

theorem field_ratio
  (l w : ℕ)
  (pond_length : ℕ)
  (pond_area_ratio : ℚ)
  (field_length : ℕ)
  (field_area : ℕ)
  (hl : l = 24)
  (hp : pond_length = 6)
  (hr : pond_area_ratio = 1 / 8)
  (hm : l % w = 0)
  (ha : field_area = 36 * 8)
  (hf : l * w = field_area) :
  l / w = 2 :=
by
  sorry

end field_ratio_l1023_102345


namespace sin_cos_sum_l1023_102361

/--
Given point P with coordinates (-3, 4) lies on the terminal side of angle α, prove that
sin α + cos α = 1/5.
-/
theorem sin_cos_sum (α : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4)) :
  Real.sin α + Real.cos α = 1/5 := sorry

end sin_cos_sum_l1023_102361


namespace calculate_integral_cos8_l1023_102384

noncomputable def integral_cos8 : ℝ :=
  ∫ x in (Real.pi / 2)..(2 * Real.pi), 2^8 * (Real.cos x)^8

theorem calculate_integral_cos8 :
  integral_cos8 = 219 * Real.pi :=
by
  sorry

end calculate_integral_cos8_l1023_102384


namespace equation_represents_3x_minus_7_equals_2x_plus_5_l1023_102309

theorem equation_represents_3x_minus_7_equals_2x_plus_5 (x : ℝ) :
  (3 * x - 7 = 2 * x + 5) :=
sorry

end equation_represents_3x_minus_7_equals_2x_plus_5_l1023_102309


namespace manufacturing_section_degrees_l1023_102366

def circle_total_degrees : ℕ := 360
def percentage_to_degree (percentage : ℕ) : ℕ := (circle_total_degrees / 100) * percentage
def manufacturing_percentage : ℕ := 60

theorem manufacturing_section_degrees : percentage_to_degree manufacturing_percentage = 216 :=
by
  -- Proof goes here
  sorry

end manufacturing_section_degrees_l1023_102366


namespace sally_picked_3_plums_l1023_102377

theorem sally_picked_3_plums (melanie_picked : ℕ) (dan_picked : ℕ) (total_picked : ℕ) 
    (h1 : melanie_picked = 4) (h2 : dan_picked = 9) (h3 : total_picked = 16) : 
    total_picked - (melanie_picked + dan_picked) = 3 := 
by 
  -- proof steps go here
  sorry

end sally_picked_3_plums_l1023_102377


namespace delaney_missed_bus_time_l1023_102324

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l1023_102324


namespace product_xyz_eq_one_l1023_102354

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end product_xyz_eq_one_l1023_102354


namespace smaller_number_l1023_102362

theorem smaller_number (L S : ℕ) (h₁ : L - S = 2395) (h₂ : L = 6 * S + 15) : S = 476 :=
by
sorry

end smaller_number_l1023_102362


namespace distinct_real_roots_m_range_root_zero_other_root_l1023_102317

open Real

-- Definitions of the quadratic equation and the conditions
def quadratic_eq (m x : ℝ) := x^2 + 2 * (m - 1) * x + m^2 - 1

-- Problem (1)
theorem distinct_real_roots_m_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0) → m < 1 :=
by
  sorry

-- Problem (2)
theorem root_zero_other_root (m x : ℝ) :
  (quadratic_eq m 0 = 0 ∧ quadratic_eq m x = 0) → (m = 1 ∧ x = 0) ∨ (m = -1 ∧ x = 4) :=
by
  sorry

end distinct_real_roots_m_range_root_zero_other_root_l1023_102317


namespace find_smaller_number_l1023_102323

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  -- Proof steps will be filled in here
  sorry

end find_smaller_number_l1023_102323


namespace prime_cubic_solution_l1023_102342

theorem prime_cubic_solution :
  ∃ p1 p2 : ℕ, (Nat.Prime p1 ∧ Nat.Prime p2) ∧ p1 ≠ p2 ∧
  (p1^3 + p1^2 - 18*p1 + 26 = 0) ∧ (p2^3 + p2^2 - 18*p2 + 26 = 0) :=
by
  sorry

end prime_cubic_solution_l1023_102342


namespace distinct_terms_count_l1023_102375

/-!
  Proving the number of distinct terms in the expansion of (x + 2y)^12
-/

theorem distinct_terms_count (x y : ℕ) : 
  (x + 2 * y) ^ 12 = 13 :=
by sorry

end distinct_terms_count_l1023_102375


namespace cindy_marbles_l1023_102321

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l1023_102321


namespace solve_for_x_l1023_102308

variable (a b c x y z : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem solve_for_x (h1 : (x * y) / (x + y) = a)
                   (h2 : (x * z) / (x + z) = b)
                   (h3 : (y * z) / (y + z) = c) :
                   x = (2 * a * b * c) / (a * c + b * c - a * b) :=
by 
  sorry

end solve_for_x_l1023_102308


namespace mother_daughter_age_l1023_102333

theorem mother_daughter_age (x : ℕ) :
  let mother_age := 42
  let daughter_age := 8
  (mother_age + x = 3 * (daughter_age + x)) → x = 9 :=
by
  let mother_age := 42
  let daughter_age := 8
  intro h
  sorry

end mother_daughter_age_l1023_102333


namespace endpoint_of_vector_a_l1023_102305

theorem endpoint_of_vector_a (x y : ℝ) (h : (x - 3) / -3 = (y + 1) / 4) : 
    x = 13 / 5 ∧ y = 2 / 5 :=
by sorry

end endpoint_of_vector_a_l1023_102305


namespace largest_b_l1023_102385

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end largest_b_l1023_102385


namespace bus_speed_express_mode_l1023_102326

theorem bus_speed_express_mode (L : ℝ) (t_red : ℝ) (speed_increase : ℝ) (x : ℝ) (normal_speed : ℝ) :
  L = 16 ∧ t_red = 1 / 15 ∧ speed_increase = 8 ∧ normal_speed = x - 8 ∧ 
  (16 / normal_speed - 16 / x = 1 / 15) → x = 48 :=
by
  sorry

end bus_speed_express_mode_l1023_102326


namespace girls_left_class_l1023_102394

variable (G B G₂ B₁ : Nat)

theorem girls_left_class (h₁ : 5 * B = 6 * G) 
                         (h₂ : B = 120)
                         (h₃ : 2 * B₁ = 3 * G₂)
                         (h₄ : B₁ = B) : 
                         G - G₂ = 20 :=
by
  sorry

end girls_left_class_l1023_102394


namespace totalTaxIsCorrect_l1023_102351

-- Define the different income sources
def dividends : ℝ := 50000
def couponIncomeOFZ : ℝ := 40000
def couponIncomeCorporate : ℝ := 30000
def capitalGain : ℝ := (100 * 200) - (100 * 150)

-- Define the tax rates
def taxRateDividends : ℝ := 0.13
def taxRateCorporateBond : ℝ := 0.13
def taxRateCapitalGain : ℝ := 0.13

-- Calculate the tax for each type of income
def taxOnDividends : ℝ := dividends * taxRateDividends
def taxOnCorporateCoupon : ℝ := couponIncomeCorporate * taxRateCorporateBond
def taxOnCapitalGain : ℝ := capitalGain * taxRateCapitalGain

-- Sum of all tax amounts
def totalTax : ℝ := taxOnDividends + taxOnCorporateCoupon + taxOnCapitalGain

-- Prove that total tax equals the calculated figure
theorem totalTaxIsCorrect : totalTax = 11050 := by
  sorry

end totalTaxIsCorrect_l1023_102351


namespace num_different_configurations_of_lights_l1023_102352

-- Definition of initial conditions
def num_rows : Nat := 6
def num_columns : Nat := 6
def possible_switch_states (n : Nat) : Nat := 2^n

-- Problem statement to be verified
theorem num_different_configurations_of_lights :
  let num_configurations := (possible_switch_states num_rows - 1) * (possible_switch_states num_columns - 1) + 1
  num_configurations = 3970 :=
by
  sorry

end num_different_configurations_of_lights_l1023_102352


namespace Laran_large_posters_daily_l1023_102316

/-
Problem statement:
Laran has started a poster business. She is selling 5 posters per day at school. Some posters per day are her large posters that sell for $10. The large posters cost her $5 to make. The remaining posters are small posters that sell for $6. They cost $3 to produce. Laran makes a profit of $95 per 5-day school week. How many large posters does Laran sell per day?
-/

/-
Mathematically equivalent proof problem:
Prove that the number of large posters Laran sells per day is 2, given the following conditions:
1) L + S = 5
2) 5L + 3S = 19
-/

variables (L S : ℕ)

-- Given conditions
def condition1 := L + S = 5
def condition2 := 5 * L + 3 * S = 19

-- Prove the desired statement
theorem Laran_large_posters_daily 
    (h1 : condition1 L S) 
    (h2 : condition2 L S) : 
    L = 2 := 
sorry

end Laran_large_posters_daily_l1023_102316


namespace no_play_students_count_l1023_102390

theorem no_play_students_count :
  let total_students := 420
  let football_players := 325
  let cricket_players := 175
  let both_players := 130
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end no_play_students_count_l1023_102390


namespace find_multiple_l1023_102322

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_ineq : (m * n - 15) > 2 * n) : m = 6 := 
by {
  sorry
}

end find_multiple_l1023_102322


namespace find_length_of_room_l1023_102335

theorem find_length_of_room (width area_existing area_needed : ℕ) (h_width : width = 15) (h_area_existing : area_existing = 16) (h_area_needed : area_needed = 149) :
  (area_existing + area_needed) / width = 11 :=
by
  sorry

end find_length_of_room_l1023_102335


namespace at_least_one_nonnegative_l1023_102391

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 :=
sorry

end at_least_one_nonnegative_l1023_102391


namespace smallest_number_bob_l1023_102364

-- Define the conditions given in the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factors (x : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ x }

-- The problem statement
theorem smallest_number_bob (b : ℕ) (h1 : prime_factors 30 = prime_factors b) : b = 30 :=
by
  sorry

end smallest_number_bob_l1023_102364


namespace additional_miles_proof_l1023_102346

-- Define the distances
def distance_to_bakery : ℕ := 9
def distance_bakery_to_grandma : ℕ := 24
def distance_grandma_to_apartment : ℕ := 27

-- Define the total distances
def total_distance_with_bakery : ℕ := distance_to_bakery + distance_bakery_to_grandma + distance_grandma_to_apartment
def total_distance_without_bakery : ℕ := 2 * distance_grandma_to_apartment

-- Define the additional miles
def additional_miles_with_bakery : ℕ := total_distance_with_bakery - total_distance_without_bakery

-- Theorem statement
theorem additional_miles_proof : additional_miles_with_bakery = 6 :=
by {
  -- Here should be the proof, but we insert sorry to indicate it's skipped
  sorry
}

end additional_miles_proof_l1023_102346


namespace simplify_expression_l1023_102311

variable (t : ℝ)

theorem simplify_expression (ht : t > 0) (ht_ne : t ≠ 1 / 2) :
  (1 - Real.sqrt (2 * t)) / ( (1 - Real.sqrt (4 * t ^ (3 / 4))) / (1 - Real.sqrt (2 * t ^ (1 / 4))) - Real.sqrt (2 * t)) *
  (Real.sqrt (1 / (1 / 2) + Real.sqrt (4 * t ^ 2)) / (1 + Real.sqrt (1 / (2 * t))) - Real.sqrt (2 * t))⁻¹ = 1 :=
by
  sorry

end simplify_expression_l1023_102311


namespace probability_of_sum_5_when_two_dice_rolled_l1023_102358

theorem probability_of_sum_5_when_two_dice_rolled :
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_possible_outcomes : ℝ) = (1 / 9 : ℝ) :=
by
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  have h : (favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ) = (1 / 9 : ℝ) := sorry
  exact h

end probability_of_sum_5_when_two_dice_rolled_l1023_102358


namespace problem1_problem2_problem3_l1023_102363

theorem problem1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := sorry
theorem problem2 (p q : ℝ) : (-p * q)^3 = -p^3 * q^3 := sorry
theorem problem3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2 * a^4)^2 = -2 * a^8 := sorry

end problem1_problem2_problem3_l1023_102363


namespace number_of_red_balls_l1023_102340

-- Definitions and conditions
def ratio_white_red (w : ℕ) (r : ℕ) : Prop := (w : ℤ) * 3 = 5 * (r : ℤ)
def white_balls : ℕ := 15

-- The theorem to prove
theorem number_of_red_balls (r : ℕ) (h : ratio_white_red white_balls r) : r = 9 :=
by
  sorry

end number_of_red_balls_l1023_102340


namespace speed_of_boat_in_still_water_l1023_102373

theorem speed_of_boat_in_still_water :
  ∀ (v : ℚ), (33 = (v + 3) * (44 / 60)) → v = 42 := 
by
  sorry

end speed_of_boat_in_still_water_l1023_102373


namespace stratified_sampling_elderly_l1023_102353

theorem stratified_sampling_elderly (total_elderly middle_aged young total_sample total_population elderly_to_sample : ℕ) 
  (h1: total_elderly = 30) 
  (h2: middle_aged = 90) 
  (h3: young = 60) 
  (h4: total_sample = 36) 
  (h5: total_population = total_elderly + middle_aged + young) 
  (h6: 1 / 5 * total_elderly = elderly_to_sample)
  : elderly_to_sample = 6 := 
  by 
    sorry

end stratified_sampling_elderly_l1023_102353


namespace find_rate_of_new_machine_l1023_102398

noncomputable def rate_of_new_machine (R : ℝ) : Prop :=
  let old_rate := 100
  let total_bolts := 350
  let time_in_hours := 84 / 60
  let bolts_by_old_machine := old_rate * time_in_hours
  let bolts_by_new_machine := total_bolts - bolts_by_old_machine
  R = bolts_by_new_machine / time_in_hours

theorem find_rate_of_new_machine : rate_of_new_machine 150 :=
by
  sorry

end find_rate_of_new_machine_l1023_102398


namespace sign_of_x_and_y_l1023_102304

theorem sign_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 :=
sorry

end sign_of_x_and_y_l1023_102304


namespace probability_interval_contains_p_l1023_102349

theorem probability_interval_contains_p (P_A P_B p : ℝ) 
  (hA : P_A = 5 / 6) 
  (hB : P_B = 3 / 4) 
  (hp : p = P_A + P_B - 1) : 
  (5 / 12 ≤ p ∧ p ≤ 3 / 4) :=
by
  -- The proof is skipped by sorry as per the instructions.
  sorry

end probability_interval_contains_p_l1023_102349


namespace log_eq_one_l1023_102336

theorem log_eq_one (log : ℝ → ℝ) (h1 : ∀ a b, log (a ^ b) = b * log a) (h2 : ∀ a b, log (a * b) = log a + log b) :
  (log 5) ^ 2 + log 2 * log 50 = 1 :=
sorry

end log_eq_one_l1023_102336


namespace find_nickels_l1023_102334

noncomputable def num_quarters1 := 25
noncomputable def num_dimes := 15
noncomputable def num_quarters2 := 15
noncomputable def value_quarter := 25
noncomputable def value_dime := 10
noncomputable def value_nickel := 5

theorem find_nickels (n : ℕ) :
  value_quarter * num_quarters1 + value_dime * num_dimes = value_quarter * num_quarters2 + value_nickel * n → 
  n = 80 :=
by
  sorry

end find_nickels_l1023_102334


namespace inequality_am_gm_l1023_102397

variable (a b x y : ℝ)

theorem inequality_am_gm (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  (a^2 / x) + (b^2 / y) ≥ (a + b)^2 / (x + y) :=
by {
  -- proof will be filled here
  sorry
}

end inequality_am_gm_l1023_102397


namespace cartesian_to_polar_coords_l1023_102310

theorem cartesian_to_polar_coords :
  ∃ ρ θ : ℝ, 
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) ∧ 
  (-1, Real.sqrt 3) = (ρ * Real.cos θ, ρ * Real.sin θ) :=
sorry

end cartesian_to_polar_coords_l1023_102310


namespace distance_A_to_focus_l1023_102374

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  ((b^2 - 4*a*c) / (4*a), 0)

theorem distance_A_to_focus 
  (P : ℝ × ℝ) (parabola : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hP : P = (-2, 0))
  (hPar : ∀ x y, parabola x y ↔ y^2 = 4 * x)
  (hLine : ∃ m b, ∀ x y, y = m * x + b ∧ y^2 = 4 * x → (x, y) = A ∨ (x, y) = B)
  (hDist : dist P A = (1 / 2) * dist A B)
  (hFocus : focus_of_parabola 1 0 (-1) = (1, 0)) :
  dist A (1, 0) = 5 / 3 :=
sorry

end distance_A_to_focus_l1023_102374


namespace distance_between_points_l1023_102399

theorem distance_between_points (x y : ℝ) (h : x + y = 10 / 3) : 
  4 * (x + y) = 40 / 3 :=
sorry

end distance_between_points_l1023_102399


namespace jelly_bean_problem_l1023_102387

variable (b c : ℕ)

theorem jelly_bean_problem (h1 : b = 3 * c) (h2 : b - 15 = 4 * (c - 15)) : b = 135 :=
sorry

end jelly_bean_problem_l1023_102387


namespace negation_of_existential_statement_l1023_102313

theorem negation_of_existential_statement : 
  (¬∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_of_existential_statement_l1023_102313


namespace find_m_l1023_102312

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3 * m - 1}
def B (m : ℝ) : Set ℝ := {2 * m, -3}
def C : Set ℝ := {-3}

theorem find_m (m : ℝ) : A m ∩ B m = C → m = 1 :=
by 
  intros h
  sorry

end find_m_l1023_102312


namespace inequality_non_empty_solution_l1023_102367

theorem inequality_non_empty_solution (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) → a ≤ 1 := sorry

end inequality_non_empty_solution_l1023_102367


namespace problem_graph_empty_l1023_102382

open Real

theorem problem_graph_empty : ∀ x y : ℝ, ¬ (x^2 + 3 * y^2 - 4 * x - 12 * y + 28 = 0) :=
by
  intro x y
  -- Apply the contradiction argument based on the conditions given
  sorry


end problem_graph_empty_l1023_102382


namespace train_speed_l1023_102368

theorem train_speed (length : ℕ) (time : ℝ)
  (h_length : length = 160)
  (h_time : time = 18) :
  (length / time * 3.6 : ℝ) = 32 :=
by
  sorry

end train_speed_l1023_102368


namespace factorization_identity_l1023_102356

theorem factorization_identity (x : ℝ) : 
  3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 :=
by
  sorry

end factorization_identity_l1023_102356


namespace machine_p_takes_longer_l1023_102344

variable (MachineP MachineQ MachineA : Type)
variable (s_prockets_per_hr : MachineA → ℝ)
variable (time_produce_s_prockets : MachineP → ℝ → ℝ)

noncomputable def machine_a_production : ℝ := 3
noncomputable def machine_q_production : ℝ := machine_a_production + 0.10 * machine_a_production

noncomputable def machine_q_time : ℝ := 330 / machine_q_production
noncomputable def additional_time : ℝ := sorry -- Since L is undefined

axiom machine_p_time : ℝ
axiom machine_p_time_eq_machine_q_time_plus_additional : machine_p_time = machine_q_time + additional_time

theorem machine_p_takes_longer : machine_p_time > machine_q_time := by
  rw [machine_p_time_eq_machine_q_time_plus_additional]
  exact lt_add_of_pos_right machine_q_time sorry  -- Need the exact L to conclude


end machine_p_takes_longer_l1023_102344


namespace no_pairs_satisfy_equation_l1023_102388

theorem no_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ¬ (2 / a + 2 / b = 1 / (a + b)) :=
by
  intros a b ha hb h
  -- the proof would go here
  sorry

end no_pairs_satisfy_equation_l1023_102388


namespace not_right_triangle_D_l1023_102369

theorem not_right_triangle_D : 
  ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) ∧
  (7^2 + 24^2 = 25^2) ∧
  (5^2 + 12^2 = 13^2) := 
by 
  have hA : 1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2 := by norm_num
  have hB : 7^2 + 24^2 = 25^2 := by norm_num
  have hC : 5^2 + 12^2 = 13^2 := by norm_num
  have hD : ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 := by norm_num
  exact ⟨hD, hA, hB, hC⟩

#print axioms not_right_triangle_D

end not_right_triangle_D_l1023_102369


namespace shadow_building_length_l1023_102395

-- Define the basic parameters
def height_flagpole : ℕ := 18
def shadow_flagpole : ℕ := 45
def height_building : ℕ := 20

-- Define the condition on similar conditions
def similar_conditions (h₁ s₁ h₂ s₂ : ℕ) : Prop :=
  h₁ * s₂ = h₂ * s₁

-- Theorem statement
theorem shadow_building_length :
  similar_conditions height_flagpole shadow_flagpole height_building 50 := 
sorry

end shadow_building_length_l1023_102395


namespace cos_sum_to_9_l1023_102301

open Real

theorem cos_sum_to_9 {x y z : ℝ} (h1 : cos x + cos y + cos z = 3) (h2 : sin x + sin y + sin z = 0) :
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 9 := 
sorry

end cos_sum_to_9_l1023_102301


namespace min_value_fraction_l1023_102357

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_fraction_l1023_102357


namespace augustus_makes_3_milkshakes_l1023_102347

def augMilkshakePerHour (A : ℕ) (Luna : ℕ) (hours : ℕ) (totalMilkshakes : ℕ) : Prop :=
  (A + Luna) * hours = totalMilkshakes

theorem augustus_makes_3_milkshakes :
  augMilkshakePerHour 3 7 8 80 :=
by
  -- We assume the proof here
  sorry

end augustus_makes_3_milkshakes_l1023_102347


namespace cubic_intersection_2_points_l1023_102360

theorem cubic_intersection_2_points (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^3 - 3*x₁ + c = 0) ∧ (x₂^3 - 3*x₂ + c = 0)) 
  → (c = -2 ∨ c = 2) :=
sorry

end cubic_intersection_2_points_l1023_102360


namespace blue_balls_taken_out_l1023_102365

theorem blue_balls_taken_out (x : ℕ) :
  ∀ (total_balls : ℕ) (initial_blue_balls : ℕ)
    (remaining_probability : ℚ),
    total_balls = 25 ∧ initial_blue_balls = 9 ∧ remaining_probability = 1/5 →
    (9 - x : ℚ) / (25 - x : ℚ) = 1/5 →
    x = 5 :=
by
  intros total_balls initial_blue_balls remaining_probability
  rintro ⟨h_total_balls, h_initial_blue_balls, h_remaining_probability⟩ h_eq
  -- Proof goes here
  sorry

end blue_balls_taken_out_l1023_102365


namespace find_principal_l1023_102396

noncomputable def compoundPrincipal (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem find_principal :
  let A := 3969
  let r := 0.05
  let n := 1
  let t := 2
  compoundPrincipal A r n t = 3600 :=
by
  sorry

end find_principal_l1023_102396


namespace sum_of_squares_of_roots_l1023_102381

theorem sum_of_squares_of_roots (x1 x2 : ℝ) (h1 : 2 * x1^2 + 5 * x1 - 12 = 0) (h2 : 2 * x2^2 + 5 * x2 - 12 = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 = 73 / 4 :=
sorry

end sum_of_squares_of_roots_l1023_102381


namespace mixture_milk_quantity_l1023_102332

variable (M W : ℕ)

theorem mixture_milk_quantity
  (h1 : M = 2 * W)
  (h2 : 6 * (W + 10) = 5 * M) :
  M = 30 := by
  sorry

end mixture_milk_quantity_l1023_102332


namespace scaling_transformation_l1023_102318

theorem scaling_transformation:
  ∀ (x y x' y': ℝ), 
  (x^2 + y^2 = 1) ∧ (x' = 5 * x) ∧ (y' = 3 * y) → 
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by intros x y x' y'
   sorry

end scaling_transformation_l1023_102318


namespace x_sq_plus_inv_sq_l1023_102337

theorem x_sq_plus_inv_sq (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
  sorry

end x_sq_plus_inv_sq_l1023_102337


namespace no_equilateral_triangle_OAB_exists_l1023_102392

theorem no_equilateral_triangle_OAB_exists :
  ∀ (A B : ℝ × ℝ), 
  ((∃ a : ℝ, A = (a, (3 / 2) ^ a)) ∧ B.1 > 0 ∧ B.2 = 0) → 
  ¬ (∃ k : ℝ, k = (A.2 / A.1) ∧ k > (3 ^ (1 / 2)) / 3) := 
by 
  intro A B h
  sorry

end no_equilateral_triangle_OAB_exists_l1023_102392


namespace right_triangle_candidate_l1023_102319

theorem right_triangle_candidate :
  (∃ a b c : ℕ, (a, b, c) = (1, 2, 3) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (2, 3, 4) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (4, 5, 6) ∧ a^2 + b^2 = c^2) ↔
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) :=
by
  sorry

end right_triangle_candidate_l1023_102319


namespace extremum_point_iff_nonnegative_condition_l1023_102339

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (x + 1)

theorem extremum_point_iff (a : ℝ) (h : 0 < a) :
  (∃ (x : ℝ), x = 1 ∧ ∀ (f' : ℝ), f' = (1 + x - a) / (x + 1)^2 ∧ f' = 0) ↔ a = 2 :=
by
  sorry

theorem nonnegative_condition (a : ℝ) (h0 : 0 < a) :
  (∀ (x : ℝ), x ∈ Set.Ici 0 → f a x ≥ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end extremum_point_iff_nonnegative_condition_l1023_102339


namespace orthocentric_tetrahedron_equivalence_l1023_102330

def isOrthocentricTetrahedron 
  (sums_of_squares_of_opposite_edges_equal : Prop) 
  (products_of_cosines_of_opposite_dihedral_angles_equal : Prop)
  (angles_between_opposite_edges_equal : Prop) : Prop :=
  sums_of_squares_of_opposite_edges_equal ∨
  products_of_cosines_of_opposite_dihedral_angles_equal ∨
  angles_between_opposite_edges_equal

theorem orthocentric_tetrahedron_equivalence
  (sums_of_squares_of_opposite_edges_equal 
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal : Prop) :
  isOrthocentricTetrahedron
    sums_of_squares_of_opposite_edges_equal
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal :=
sorry

end orthocentric_tetrahedron_equivalence_l1023_102330


namespace yellow_balls_in_bag_l1023_102370

theorem yellow_balls_in_bag (r y : ℕ) (P : ℚ) 
  (h1 : r = 10) 
  (h2 : P = 2 / 7) 
  (h3 : P = r / (r + y)) : 
  y = 25 := 
sorry

end yellow_balls_in_bag_l1023_102370


namespace marcus_percentage_of_team_points_l1023_102355

theorem marcus_percentage_of_team_points
  (three_point_goals : ℕ)
  (two_point_goals : ℕ)
  (team_points : ℕ)
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_points = 70) :
  ((three_point_goals * 3 + two_point_goals * 2) / team_points : ℚ) * 100 = 50 := by
sorry

end marcus_percentage_of_team_points_l1023_102355


namespace weightlifting_winner_l1023_102359

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end weightlifting_winner_l1023_102359


namespace ethan_pages_left_l1023_102350

-- Definitions based on the conditions
def total_pages := 360
def pages_read_morning := 40
def pages_read_night := 10
def pages_read_saturday := pages_read_morning + pages_read_night
def pages_read_sunday := 2 * pages_read_saturday
def total_pages_read := pages_read_saturday + pages_read_sunday

-- Lean 4 statement for the proof problem
theorem ethan_pages_left : total_pages - total_pages_read = 210 := by
  sorry

end ethan_pages_left_l1023_102350


namespace a_100_correct_l1023_102378

variable (a_n : ℕ → ℕ) (S₉ : ℕ) (a₁₀ : ℕ)

def is_arth_seq (a_n : ℕ → ℕ) := ∃ a d, ∀ n, a_n n = a + n * d

noncomputable def a_100 (a₅ d : ℕ) : ℕ := a₅ + 95 * d

theorem a_100_correct
  (h1 : ∃ S₉, 9 * a_n 4 = S₉)
  (h2 : a_n 9 = 8)
  (h3 : is_arth_seq a_n) :
  a_100 (a_n 4) 1 = 98 :=
by
  sorry

end a_100_correct_l1023_102378


namespace problem1_problem2_problem3_l1023_102328

-- First problem: Prove x = 4.2 given x + 2x = 12.6
theorem problem1 (x : ℝ) (h1 : x + 2 * x = 12.6) : x = 4.2 :=
  sorry

-- Second problem: Prove x = 2/5 given 1/4 * x + 1/2 = 3/5
theorem problem2 (x : ℚ) (h2 : (1 / 4) * x + 1 / 2 = 3 / 5) : x = 2 / 5 :=
  sorry

-- Third problem: Prove x = 20 given x + 130% * x = 46 (where 130% is 130/100)
theorem problem3 (x : ℝ) (h3 : x + (130 / 100) * x = 46) : x = 20 :=
  sorry

end problem1_problem2_problem3_l1023_102328


namespace remaining_statue_weight_l1023_102389

theorem remaining_statue_weight (w_initial w1 w2 w_discarded w_remaining : ℕ) 
    (h_initial : w_initial = 80)
    (h_w1 : w1 = 10)
    (h_w2 : w2 = 18)
    (h_discarded : w_discarded = 22) :
    2 * w_remaining = w_initial - w_discarded - w1 - w2 :=
by
  sorry

end remaining_statue_weight_l1023_102389


namespace rebecca_tent_stakes_l1023_102331

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end rebecca_tent_stakes_l1023_102331


namespace social_event_handshakes_l1023_102371

def handshake_count (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) : ℕ :=
  let introductions_handshakes := group_b * (group_b - 1) / 2
  let direct_handshakes := group_b * (group_a - 1)
  introductions_handshakes + direct_handshakes

theorem social_event_handshakes :
  handshake_count 40 25 15 = 465 := by
  sorry

end social_event_handshakes_l1023_102371


namespace ratio_of_chris_to_amy_l1023_102302

-- Definitions based on the conditions in the problem
def combined_age (Amy_age Jeremy_age Chris_age : ℕ) : Prop :=
  Amy_age + Jeremy_age + Chris_age = 132

def amy_is_one_third_jeremy (Amy_age Jeremy_age : ℕ) : Prop :=
  Amy_age = Jeremy_age / 3

def jeremy_age : ℕ := 66

-- The main theorem we need to prove
theorem ratio_of_chris_to_amy (Amy_age Chris_age : ℕ) (h1 : combined_age Amy_age jeremy_age Chris_age)
  (h2 : amy_is_one_third_jeremy Amy_age jeremy_age) : Chris_age / Amy_age = 2 :=
sorry

end ratio_of_chris_to_amy_l1023_102302


namespace pipe_fill_time_without_leak_l1023_102372

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : T > 0) 
  (h2 : 1/T - 1/8 = 1/8) :
  T = 4 := 
sorry

end pipe_fill_time_without_leak_l1023_102372


namespace correct_average_mark_l1023_102306

theorem correct_average_mark (
  num_students : ℕ := 50)
  (incorrect_avg : ℚ := 85.4)
  (wrong_mark_A : ℚ := 73.6) (correct_mark_A : ℚ := 63.5)
  (wrong_mark_B : ℚ := 92.4) (correct_mark_B : ℚ := 96.7)
  (wrong_mark_C : ℚ := 55.3) (correct_mark_C : ℚ := 51.8) :
  (incorrect_avg*num_students + 
   (correct_mark_A - wrong_mark_A) + 
   (correct_mark_B - wrong_mark_B) + 
   (correct_mark_C - wrong_mark_C)) / 
   num_students = 85.214 :=
sorry

end correct_average_mark_l1023_102306


namespace initial_spiders_correct_l1023_102386

-- Define the initial number of each type of animal
def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5

-- Conditions about the changes in the number of animals
def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

-- Number of animals left in the store
def total_animals_left : Nat := 25

-- Define the remaining animals after sales and adoptions
def remaining_birds : Nat := initial_birds - birds_sold
def remaining_puppies : Nat := initial_puppies - puppies_adopted
def remaining_cats : Nat := initial_cats

-- Define the remaining animals excluding spiders
def animals_without_spiders : Nat := remaining_birds + remaining_puppies + remaining_cats

-- Define the number of remaining spiders
def remaining_spiders : Nat := total_animals_left - animals_without_spiders

-- Prove the initial number of spiders
def initial_spiders : Nat := remaining_spiders + spiders_loose

theorem initial_spiders_correct :
  initial_spiders = 15 := by 
  sorry

end initial_spiders_correct_l1023_102386


namespace scientific_notation_of_42000_l1023_102327

theorem scientific_notation_of_42000 : 42000 = 4.2 * 10^4 := 
by 
  sorry

end scientific_notation_of_42000_l1023_102327


namespace only_one_student_remains_l1023_102338

theorem only_one_student_remains (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2002) :
  (∃! k, k = n ∧ n % 1331 = 0) ↔ n = 1331 :=
by
  sorry

end only_one_student_remains_l1023_102338


namespace total_trail_length_l1023_102320

-- Definitions based on conditions
variables (a b c d e : ℕ)

-- Conditions
def condition1 : Prop := a + b + c = 36
def condition2 : Prop := b + c + d = 48
def condition3 : Prop := c + d + e = 45
def condition4 : Prop := a + d = 31

-- Theorem statement
theorem total_trail_length (h1 : condition1 a b c) (h2 : condition2 b c d) (h3 : condition3 c d e) (h4 : condition4 a d) : 
  a + b + c + d + e = 81 :=
by 
  sorry

end total_trail_length_l1023_102320


namespace marbles_jack_gave_l1023_102300

-- Definitions based on conditions
def initial_marbles : ℕ := 22
def final_marbles : ℕ := 42

-- Theorem stating that the difference between final and initial marbles Josh collected is the marbles Jack gave
theorem marbles_jack_gave :
  final_marbles - initial_marbles = 20 :=
  sorry

end marbles_jack_gave_l1023_102300


namespace lines_intersect_sum_c_d_l1023_102380

theorem lines_intersect_sum_c_d (c d : ℝ) 
    (h1 : ∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) 
    (h2 : ∀ x y : ℝ, x = 3 ∧ y = 3) : 
    c + d = 4 :=
by sorry

end lines_intersect_sum_c_d_l1023_102380


namespace solve_system_a_l1023_102325

theorem solve_system_a (x y : ℝ) (h1 : x^2 - 3 * x * y - 4 * y^2 = 0) (h2 : x^3 + y^3 = 65) : 
    x = 4 ∧ y = 1 :=
sorry

end solve_system_a_l1023_102325


namespace sum_GCF_LCM_l1023_102383

-- Definitions of GCD and LCM for the numbers 18, 27, and 36
def GCF : ℕ := Nat.gcd (Nat.gcd 18 27) 36
def LCM : ℕ := Nat.lcm (Nat.lcm 18 27) 36

-- Theorem statement proof
theorem sum_GCF_LCM : GCF + LCM = 117 := by
  sorry

end sum_GCF_LCM_l1023_102383


namespace square_area_l1023_102379

theorem square_area (x : ℝ) (side1 side2 : ℝ) 
  (h_side1 : side1 = 6 * x - 27) 
  (h_side2 : side2 = 30 - 2 * x) 
  (h_equiv : side1 = side2) : 
  (side1 * side1 = 248.0625) := 
by
  sorry

end square_area_l1023_102379


namespace quadrilateral_side_squares_inequality_l1023_102393

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end quadrilateral_side_squares_inequality_l1023_102393


namespace total_tickets_sold_l1023_102303

theorem total_tickets_sold
  (advanced_ticket_cost : ℕ)
  (door_ticket_cost : ℕ)
  (total_collected : ℕ)
  (advanced_tickets_sold : ℕ)
  (door_tickets_sold : ℕ) :
  advanced_ticket_cost = 8 →
  door_ticket_cost = 14 →
  total_collected = 1720 →
  advanced_tickets_sold = 100 →
  total_collected = (advanced_tickets_sold * advanced_ticket_cost) + (door_tickets_sold * door_ticket_cost) →
  100 + door_tickets_sold = 165 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_tickets_sold_l1023_102303


namespace vector_decomposition_unique_l1023_102348

variable {m : ℝ}
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (m - 1, m + 3)

theorem vector_decomposition_unique (m : ℝ) : (m + 3 ≠ 2 * (m - 1)) ↔ (m ≠ 5) := 
sorry

end vector_decomposition_unique_l1023_102348


namespace sqrt_14400_eq_120_l1023_102307

theorem sqrt_14400_eq_120 : Real.sqrt 14400 = 120 :=
by
  sorry

end sqrt_14400_eq_120_l1023_102307


namespace cos_double_angle_sum_l1023_102343

theorem cos_double_angle_sum (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (h : Real.sin (α + π/6) = 3/5) : 
  Real.cos (2*α + π/12) = 31 / 50 * Real.sqrt 2 := sorry

end cos_double_angle_sum_l1023_102343


namespace product_of_coordinates_of_intersection_l1023_102341

-- Conditions: Defining the equations of the two circles
def circle1_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Translated problem to prove the question equals the correct answer
theorem product_of_coordinates_of_intersection :
  ∃ (x y : ℝ), circle1_eq x y ∧ circle2_eq x y ∧ x * y = 10 :=
sorry

end product_of_coordinates_of_intersection_l1023_102341


namespace train_speed_l1023_102329

theorem train_speed 
  (length_train : ℕ) 
  (time_crossing : ℕ) 
  (speed_kmph : ℕ)
  (h_length : length_train = 120)
  (h_time : time_crossing = 9)
  (h_speed : speed_kmph = 48) : 
  length_train / time_crossing * 3600 / 1000 = speed_kmph := 
by 
  sorry

end train_speed_l1023_102329


namespace smallest_n_for_simplest_form_l1023_102315

-- Definitions and conditions
def simplest_form_fractions (n : ℕ) :=
  ∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + 2) = 1

-- Problem statement
theorem smallest_n_for_simplest_form :
  ∃ n : ℕ, simplest_form_fractions (n) ∧ ∀ m : ℕ, m < n → ¬ simplest_form_fractions (m) := 
by 
  sorry

end smallest_n_for_simplest_form_l1023_102315


namespace cell_division_after_three_hours_l1023_102314

theorem cell_division_after_three_hours : (2 ^ 6) = 64 := by
  sorry

end cell_division_after_three_hours_l1023_102314


namespace salt_concentration_l1023_102376

theorem salt_concentration (volume_water volume_solution concentration_solution : ℝ)
  (h1 : volume_water = 1)
  (h2 : volume_solution = 0.5)
  (h3 : concentration_solution = 0.45) :
  (volume_solution * concentration_solution) / (volume_water + volume_solution) = 0.15 :=
by
  sorry

end salt_concentration_l1023_102376
