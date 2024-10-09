import Mathlib

namespace mario_savings_percentage_l1276_127675

-- Define the price of one ticket
def ticket_price : ℝ := sorry

-- Define the conditions
-- Condition 1: 5 tickets can be purchased for the usual price of 3 tickets
def price_for_5_tickets := 3 * ticket_price

-- Condition 2: Mario bought 5 tickets
def mario_tickets := 5 * ticket_price

-- Condition 3: Usual price for 5 tickets
def usual_price_5_tickets := 5 * ticket_price

-- Calculate the amount saved
def amount_saved := usual_price_5_tickets - price_for_5_tickets

theorem mario_savings_percentage
  (ticket_price: ℝ)
  (h1 : price_for_5_tickets = 3 * ticket_price)
  (h2 : mario_tickets = 5 * ticket_price)
  (h3 : usual_price_5_tickets = 5 * ticket_price)
  (h4 : amount_saved = usual_price_5_tickets - price_for_5_tickets):
  (amount_saved / usual_price_5_tickets) * 100 = 40 := 
by {
    -- Placeholder
    sorry
}

end mario_savings_percentage_l1276_127675


namespace smallest_x_with_18_factors_and_factors_18_21_exists_l1276_127653

def has_18_factors (x : ℕ) : Prop :=
(x.factors.length == 18)

def is_factor (a b : ℕ) : Prop :=
(b % a == 0)

theorem smallest_x_with_18_factors_and_factors_18_21_exists :
  ∃ x : ℕ, has_18_factors x ∧ is_factor 18 x ∧ is_factor 21 x ∧ ∀ y : ℕ, has_18_factors y ∧ is_factor 18 y ∧ is_factor 21 y → y ≥ x :=
sorry

end smallest_x_with_18_factors_and_factors_18_21_exists_l1276_127653


namespace fraction_of_l1276_127697

theorem fraction_of (a b : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) : (a / b) = 3/5 :=
by sorry

end fraction_of_l1276_127697


namespace prime_count_of_first_10_sums_is_2_l1276_127674

open Nat

def consecutivePrimes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def consecutivePrimeSums (n : Nat) : List Nat :=
  (List.range n).scanl (λ sum i => sum + consecutivePrimes.getD i 0) 0

theorem prime_count_of_first_10_sums_is_2 :
  let sums := consecutivePrimeSums 10;
  (sums.count isPrime) = 2 :=
by
  sorry

end prime_count_of_first_10_sums_is_2_l1276_127674


namespace eq1_solution_eq2_solution_l1276_127689

theorem eq1_solution (x : ℝ) : (x - 1)^2 - 1 = 15 ↔ x = 5 ∨ x = -3 := by sorry

theorem eq2_solution (x : ℝ) : (1 / 3) * (x + 3)^3 - 9 = 0 ↔ x = 0 := by sorry

end eq1_solution_eq2_solution_l1276_127689


namespace rectangular_solid_diagonal_l1276_127606

theorem rectangular_solid_diagonal (p q r : ℝ) (d : ℝ) :
  p^2 + q^2 + r^2 = d^2 :=
sorry

end rectangular_solid_diagonal_l1276_127606


namespace calculate_expression_l1276_127640

variable (x y : ℝ)

theorem calculate_expression : (-2 * x^2 * y) ^ 2 = 4 * x^4 * y^2 := by
  sorry

end calculate_expression_l1276_127640


namespace AB_vector_eq_l1276_127652

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (A B C D : V)
variables (a b : V)
variable (ABCD_parallelogram : is_parallelogram A B C D)

-- Definition of the diagonals
def AC_vector : V := C - A
def BD_vector : V := D - B

-- The given condition that diagonals AC and BD are equal to a and b respectively
axiom AC_eq_a : AC_vector A C = a
axiom BD_eq_b : BD_vector B D = b

-- Proof problem
theorem AB_vector_eq : (B - A) = (1/2) • (a - b) :=
sorry

end AB_vector_eq_l1276_127652


namespace paula_travel_fraction_l1276_127691

theorem paula_travel_fraction :
  ∀ (f : ℚ), 
    (∀ (L_time P_time travel_total : ℚ), 
      L_time = 70 →
      P_time = 70 * f →
      travel_total = 504 →
      (L_time + 5 * L_time + P_time + P_time = travel_total) →
      f = 3/5) :=
by
  sorry

end paula_travel_fraction_l1276_127691


namespace triangle_side_length_l1276_127627

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h₁ : a * Real.cos B = b * Real.sin A)
  (h₂ : C = Real.pi / 6) (h₃ : c = 2) : b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l1276_127627


namespace simplify_polynomial_l1276_127686

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end simplify_polynomial_l1276_127686


namespace sprockets_produced_by_machines_l1276_127664

noncomputable def machine_sprockets (t : ℝ) : Prop :=
  let machineA_hours := t + 10
  let machineA_rate := 4
  let machineA_sprockets := machineA_hours * machineA_rate
  let machineB_hours := t
  let machineB_rate := 4.4
  let machineB_sprockets := machineB_hours * machineB_rate
  machineA_sprockets = 440 ∧ machineB_sprockets = 440

theorem sprockets_produced_by_machines (t : ℝ) (h : machine_sprockets t) : t = 100 :=
  sorry

end sprockets_produced_by_machines_l1276_127664


namespace power_increased_by_four_l1276_127692

-- Definitions from the conditions
variables (F k v : ℝ) (initial_force_eq_resistive : F = k * v)

-- Define the new conditions with double the force
variables (new_force : ℝ) (new_velocity : ℝ) (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F)

-- The theorem statement
theorem power_increased_by_four (initial_force_eq_resistive : F = k * v) 
  (new_force_eq_resistive : new_force = k * new_velocity)
  (doubled_force : new_force = 2 * F) :
  new_velocity = 2 * v → 
  (new_force * new_velocity) = 4 * (F * v) :=
sorry

end power_increased_by_four_l1276_127692


namespace slower_train_speed_l1276_127681

theorem slower_train_speed
  (v : ℝ) -- the speed of the slower train (kmph)
  (faster_train_speed : ℝ := 72)        -- the speed of the faster train
  (time_to_cross_man : ℝ := 18)         -- time to cross a man in the slower train (seconds)
  (faster_train_length : ℝ := 180)      -- length of the faster train (meters))
  (conversion_factor : ℝ := 5 / 18)     -- conversion factor from kmph to m/s
  (relative_speed_m_s : ℝ := ((faster_train_speed - v) * conversion_factor)) :
  ((faster_train_length : ℝ) = (relative_speed_m_s * time_to_cross_man)) →
  v = 36 :=
by
  -- the actual proof needs to be filled here
  sorry

end slower_train_speed_l1276_127681


namespace coefficient_x_squared_l1276_127621

theorem coefficient_x_squared (a : ℝ) (x : ℝ) (h : x = 0.5) (eqn : a * x^2 + 9 * x - 5 = 0) : a = 2 :=
by
  sorry

end coefficient_x_squared_l1276_127621


namespace gain_percentage_l1276_127694

theorem gain_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 + C2 = 540) (h2 : C1 = 315)
    (h3 : SP1 = C1 - (0.15 * C1)) (h4 : SP1 = SP2) :
    ((SP2 - C2) / C2) * 100 = 19 :=
by
  sorry

end gain_percentage_l1276_127694


namespace boys_girls_dance_l1276_127608

theorem boys_girls_dance (b g : ℕ) 
  (h : ∀ n, (n <= b) → (n + 7) ≤ g) 
  (hb_lasts : b + 7 = g) :
  b = g - 7 := by
  sorry

end boys_girls_dance_l1276_127608


namespace number_of_elements_in_set_S_l1276_127601

-- Define the set S and its conditions
variable (S : Set ℝ) (n : ℝ) (sumS : ℝ)

-- Conditions given in the problem
axiom avg_S : sumS / n = 6.2
axiom new_avg_S : (sumS + 8) / n = 7

-- The statement to be proved
theorem number_of_elements_in_set_S : n = 10 := by 
  sorry

end number_of_elements_in_set_S_l1276_127601


namespace cole_drive_time_l1276_127617

theorem cole_drive_time (D T1 T2 : ℝ) (h1 : T1 = D / 75) 
  (h2 : T2 = D / 105) (h3 : T1 + T2 = 6) : 
  (T1 * 60 = 210) :=
by sorry

end cole_drive_time_l1276_127617


namespace parking_lot_wheels_l1276_127690

-- Define the total number of wheels for each type of vehicle
def car_wheels (n : ℕ) : ℕ := n * 4
def motorcycle_wheels (n : ℕ) : ℕ := n * 2
def truck_wheels (n : ℕ) : ℕ := n * 6
def van_wheels (n : ℕ) : ℕ := n * 4

-- Number of each type of guests' vehicles
def num_cars : ℕ := 5
def num_motorcycles : ℕ := 4
def num_trucks : ℕ := 3
def num_vans : ℕ := 2

-- Number of parents' vehicles and their wheels
def parents_car_wheels : ℕ := 4
def parents_jeep_wheels : ℕ := 4

-- Summing up all the wheels
def total_wheels : ℕ :=
  car_wheels num_cars +
  motorcycle_wheels num_motorcycles +
  truck_wheels num_trucks +
  van_wheels num_vans +
  parents_car_wheels +
  parents_jeep_wheels

theorem parking_lot_wheels : total_wheels = 62 := by
  sorry

end parking_lot_wheels_l1276_127690


namespace transform_cos_to_base_form_l1276_127615

theorem transform_cos_to_base_form :
  let f (x : ℝ) := Real.cos (2 * x + (Real.pi / 3))
  let g (x : ℝ) := Real.cos (2 * x)
  ∃ (shift : ℝ), shift = Real.pi / 6 ∧
    (∀ x : ℝ, f (x - shift) = g x) :=
by
  let f := λ x : ℝ => Real.cos (2 * x + (Real.pi / 3))
  let g := λ x : ℝ => Real.cos (2 * x)
  use Real.pi / 6
  sorry

end transform_cos_to_base_form_l1276_127615


namespace min_k_value_l1276_127683

noncomputable def f (k x : ℝ) : ℝ := k * (x^2 - x + 1) - x^4 * (1 - x)^4

theorem min_k_value : ∃ k : ℝ, (k = 1 / 192) ∧ ∀ x : ℝ, (0 ≤ x) → (x ≤ 1) → (f k x ≥ 0) :=
by
  existsi (1 / 192)
  sorry

end min_k_value_l1276_127683


namespace find_x_l1276_127698

-- Definitions based on the given conditions
variables {B C D : Type} (A : Type)

-- Angles in degrees
variables (angle_ACD : ℝ := 100)
variables (angle_ADB : ℝ)
variables (angle_ABD : ℝ := 2 * angle_ADB)
variables (angle_DAC : ℝ)
variables (angle_BAC : ℝ := angle_DAC)
variables (angle_ACB : ℝ := 180 - angle_ACD)
variables (y : ℝ := angle_DAC)
variables (x : ℝ := angle_ADB)

-- The proof statement
theorem find_x (h1 : B = C) (h2 : C = D) 
    (h3: angle_ACD = 100) 
    (h4: angle_ADB = x) 
    (h5: angle_ABD = 2 * x) 
    (h6: angle_DAC = angle_BAC) 
    (h7: angle_DAC = y)
    : x = 20 :=
sorry

end find_x_l1276_127698


namespace max_marks_l1276_127684

theorem max_marks (M : ℝ) (h1 : 0.42 * M = 80) : M = 190 :=
by
  sorry

end max_marks_l1276_127684


namespace ice_cream_flavors_l1276_127613

theorem ice_cream_flavors : (Nat.choose (4 + 4 - 1) (4 - 1) = 35) :=
by
  sorry

end ice_cream_flavors_l1276_127613


namespace geom_seq_sum_eqn_l1276_127622

theorem geom_seq_sum_eqn (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 2 + 2 * a 3 = a 1)
  (h2 : a 1 * a 4 = a 6)
  (h3 : ∀ n, a (n + 1) = a 1 * (1 / 2) ^ n)
  (h4 : ∀ n, S n = 2 * ((1 - (1 / 2) ^ n) / (1 - (1 / 2)))) :
  a n + S n = 4 :=
sorry

end geom_seq_sum_eqn_l1276_127622


namespace smallest_prime_divisor_and_cube_root_l1276_127695

theorem smallest_prime_divisor_and_cube_root (N : ℕ) (p : ℕ) (q : ℕ)
  (hN_composite : N > 1 ∧ ¬ (∃ p : ℕ, p > 1 ∧ p < N ∧ N = p))
  (h_divisor : N = p * q)
  (h_p_prime : Nat.Prime p)
  (h_min_prime : ∀ (d : ℕ), Nat.Prime d → d ∣ N → p ≤ d)
  (h_cube_root : p > Nat.sqrt (Nat.sqrt N)) :
  Nat.Prime q := 
sorry

end smallest_prime_divisor_and_cube_root_l1276_127695


namespace coordinates_of_F_double_prime_l1276_127685

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem coordinates_of_F_double_prime :
  let F : ℝ × ℝ := (3, 3)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_x_axis F'
  F'' = (-3, -3) :=
by
  sorry

end coordinates_of_F_double_prime_l1276_127685


namespace unique_k_n_m_solution_l1276_127651

-- Problem statement
theorem unique_k_n_m_solution :
  ∃ (k : ℕ) (n : ℕ) (m : ℕ), k = 1 ∧ n = 2 ∧ m = 3 ∧ 3^k + 5^k = n^m ∧
  ∀ (k₀ : ℕ) (n₀ : ℕ) (m₀ : ℕ), (3^k₀ + 5^k₀ = n₀^m₀ ∧ m₀ ≥ 2) → (k₀ = 1 ∧ n₀ = 2 ∧ m₀ = 3) :=
by
  sorry

end unique_k_n_m_solution_l1276_127651


namespace pure_imaginary_x_l1276_127659

theorem pure_imaginary_x (x : ℝ) (h: (x - 2008) = 0) : x = 2008 :=
by
  sorry

end pure_imaginary_x_l1276_127659


namespace possible_length_of_third_side_l1276_127603

theorem possible_length_of_third_side (a b c : ℤ) (h1 : a - b = 7) (h2 : (a + b + c) % 2 = 1) : c = 8 :=
sorry

end possible_length_of_third_side_l1276_127603


namespace evaluate_expression_l1276_127624

theorem evaluate_expression :
  ((Int.ceil ((21 : ℚ) / 5 - Int.ceil ((35 : ℚ) / 23))) : ℚ) /
  (Int.ceil ((35 : ℚ) / 5 + Int.ceil ((5 * 23 : ℚ) / 35))) = 3 / 11 := by
  sorry

end evaluate_expression_l1276_127624


namespace sum_inverses_of_roots_l1276_127655

open Polynomial

theorem sum_inverses_of_roots (a b c : ℝ) (h1 : a^3 - 2020 * a + 1010 = 0)
    (h2 : b^3 - 2020 * b + 1010 = 0) (h3 : c^3 - 2020 * c + 1010 = 0) :
    (1/a) + (1/b) + (1/c) = 2 := 
  sorry

end sum_inverses_of_roots_l1276_127655


namespace find_A_for_diamond_l1276_127619

def diamond (A B : ℕ) : ℕ := 4 * A + 3 * B + 7

theorem find_A_for_diamond (A : ℕ) (h : diamond A 7 = 76) : A = 12 :=
by
  sorry

end find_A_for_diamond_l1276_127619


namespace green_hats_count_l1276_127612

theorem green_hats_count 
  (B G : ℕ)
  (h1 : B + G = 85)
  (h2 : 6 * B + 7 * G = 530) : 
  G = 20 :=
by
  sorry

end green_hats_count_l1276_127612


namespace only_setA_forms_triangle_l1276_127647

-- Define the sets of line segments
def setA := [3, 5, 7]
def setB := [3, 6, 10]
def setC := [5, 5, 11]
def setD := [5, 6, 11]

-- Define a function to check the triangle inequality
def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Formalize the question
theorem only_setA_forms_triangle :
  satisfies_triangle_inequality 3 5 7 ∧
  ¬(satisfies_triangle_inequality 3 6 10) ∧
  ¬(satisfies_triangle_inequality 5 5 11) ∧
  ¬(satisfies_triangle_inequality 5 6 11) :=
by
  sorry

end only_setA_forms_triangle_l1276_127647


namespace doug_initial_marbles_l1276_127609

theorem doug_initial_marbles (ed_marbles : ℕ) (diff_ed_doug : ℕ) (final_ed_marbles : ed_marbles = 27) (diff : diff_ed_doug = 5) :
  ∃ doug_initial_marbles : ℕ, doug_initial_marbles = 22 :=
by
  sorry

end doug_initial_marbles_l1276_127609


namespace gcd_problem_l1276_127676

theorem gcd_problem (b : ℕ) (h : ∃ k : ℕ, b = 3150 * k) :
  gcd (b^2 + 9 * b + 54) (b + 4) = 2 := by
  sorry

end gcd_problem_l1276_127676


namespace second_train_length_l1276_127649

theorem second_train_length
  (train1_length : ℝ)
  (train1_speed_kmph : ℝ)
  (train2_speed_kmph : ℝ)
  (time_to_clear : ℝ)
  (h1 : train1_length = 135)
  (h2 : train1_speed_kmph = 80)
  (h3 : train2_speed_kmph = 65)
  (h4 : time_to_clear = 7.447680047665153) :
  ∃ l2 : ℝ, l2 = 165 :=
by
  let train1_speed_mps := train1_speed_kmph * 1000 / 3600
  let train2_speed_mps := train2_speed_kmph * 1000 / 3600
  let total_distance := (train1_speed_mps + train2_speed_mps) * time_to_clear
  have : total_distance = 300 := by sorry
  have l2 := total_distance - train1_length
  use l2
  have : l2 = 165 := by sorry
  assumption

end second_train_length_l1276_127649


namespace acme_vs_beta_l1276_127637

theorem acme_vs_beta (x : ℕ) :
  (80 + 10 * x < 20 + 15 * x) → (13 ≤ x) :=
by
  intro h
  sorry

end acme_vs_beta_l1276_127637


namespace problem_l1276_127610

theorem problem : (112^2 - 97^2) / 15 = 209 := by
  sorry

end problem_l1276_127610


namespace multiplication_simplify_l1276_127600

theorem multiplication_simplify :
  12 * (1 / 8) * 32 = 48 := 
sorry

end multiplication_simplify_l1276_127600


namespace combination_20_6_l1276_127633

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l1276_127633


namespace fresh_grapes_weight_l1276_127614

theorem fresh_grapes_weight :
  ∀ (F : ℝ), (∀ (water_content_fresh : ℝ) (water_content_dried : ℝ) (weight_dried : ℝ),
    water_content_fresh = 0.90 → water_content_dried = 0.20 → weight_dried = 3.125 →
    (F * 0.10 = 0.80 * weight_dried) → F = 78.125) := 
by
  intros F
  intros water_content_fresh water_content_dried weight_dried
  intros h1 h2 h3 h4
  sorry

end fresh_grapes_weight_l1276_127614


namespace part1_part2_part3_l1276_127654

universe u

def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12 * x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}
def CR_A : Set ℝ := {x | x < -3 ∨ x ≥ 7}

theorem part1 : A ∪ B = {x | -3 ≤ x ∧ x < 10} := by
  sorry

theorem part2 : CR_A ∩ B = {x | 7 ≤ x ∧ x < 10} := by
  sorry

theorem part3 (a : ℝ) (h : (A ∩ C a).Nonempty) : a > -3 := by
  sorry

end part1_part2_part3_l1276_127654


namespace cube_sum_equals_36_l1276_127693

variable {a b c k : ℝ}

theorem cube_sum_equals_36 (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (heq : (a^3 - 12) / a = (b^3 - 12) / b)
    (heq_another : (b^3 - 12) / b = (c^3 - 12) / c) :
    a^3 + b^3 + c^3 = 36 := by
  sorry

end cube_sum_equals_36_l1276_127693


namespace probability_zhang_watches_entire_news_l1276_127645

noncomputable def broadcast_time_start := 12 * 60 -- 12:00 in minutes
noncomputable def broadcast_time_end := 12 * 60 + 30 -- 12:30 in minutes
noncomputable def news_report_duration := 5 -- 5 minutes
noncomputable def zhang_on_tv_time := 12 * 60 + 20 -- 12:20 in minutes
noncomputable def favorable_time_start := zhang_on_tv_time
noncomputable def favorable_time_end := zhang_on_tv_time + news_report_duration -- 12:20 to 12:25

theorem probability_zhang_watches_entire_news : 
  let total_broadcast_time := broadcast_time_end - broadcast_time_start
  let favorable_time_span := favorable_time_end - favorable_time_start
  favorable_time_span / total_broadcast_time = 1 / 6 :=
by
  sorry

end probability_zhang_watches_entire_news_l1276_127645


namespace fifth_powers_sum_eq_l1276_127669

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l1276_127669


namespace carrot_broccoli_ratio_l1276_127680

variables (total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings : ℕ)

-- Define the conditions
def is_condition_satisfied :=
  total_earnings = 380 ∧
  broccoli_earnings = 57 ∧
  cauliflower_earnings = 136 ∧
  spinach_earnings = (carrot_earnings / 2 + 16)

-- Define the proof problem that checks the ratio
theorem carrot_broccoli_ratio (h : is_condition_satisfied total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings) :
  ((carrot_earnings + ((carrot_earnings / 2) + 16)) + broccoli_earnings + cauliflower_earnings = total_earnings) →
  (carrot_earnings / broccoli_earnings = 2) :=
sorry

end carrot_broccoli_ratio_l1276_127680


namespace simplify_polynomial_l1276_127670

theorem simplify_polynomial (x : ℝ) :
  3 + 5 * x - 7 * x^2 - 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 = 9 - x - x^2 := 
  by {
  -- placeholder for the proof
  sorry
}

end simplify_polynomial_l1276_127670


namespace choose_positions_from_8_people_l1276_127623

theorem choose_positions_from_8_people : 
  ∃ (ways : ℕ), ways = 8 * 7 * 6 := 
sorry

end choose_positions_from_8_people_l1276_127623


namespace problem_proof_l1276_127620

-- Formalizing the conditions of the problem
variable {a : ℕ → ℝ}  -- Define the arithmetic sequence
variable (d : ℝ)      -- Common difference of the arithmetic sequence
variable (a₅ a₆ a₇ : ℝ)  -- Specific terms in the sequence

-- The condition given in the problem
axiom cond1 : a 5 + a 6 + a 7 = 15

-- A definition for an arithmetic sequence
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Using the axiom to deduce that a₆ = 5
axiom prop_arithmetic : is_arithmetic_seq a d

-- We want to prove that sum of terms from a₃ to a₉ = 35
theorem problem_proof : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by sorry

end problem_proof_l1276_127620


namespace find_beta_l1276_127629

open Real

theorem find_beta (α β : ℝ) (h1 : cos α = 1 / 7) (h2 : cos (α - β) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : β = π / 3 :=
by
  sorry

end find_beta_l1276_127629


namespace ratio_of_logs_eq_golden_ratio_l1276_127625

theorem ratio_of_logs_eq_golden_ratio
  (r s : ℝ) (hr : 0 < r) (hs : 0 < s)
  (h : Real.log r / Real.log 4 = Real.log s / Real.log 18 ∧ Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) :
  s / r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_eq_golden_ratio_l1276_127625


namespace probability_three_red_balls_l1276_127665

open scoped BigOperators

noncomputable def hypergeometric_prob (r : ℕ) (b : ℕ) (k : ℕ) (d : ℕ) : ℝ :=
  (Nat.choose r d * Nat.choose b (k - d) : ℝ) / Nat.choose (r + b) k

theorem probability_three_red_balls :
  hypergeometric_prob 10 5 5 3 = 1200 / 3003 :=
by sorry

end probability_three_red_balls_l1276_127665


namespace village_population_l1276_127663

theorem village_population (x : ℝ) (h : 0.96 * x = 23040) : x = 24000 := sorry

end village_population_l1276_127663


namespace probability_of_5_odd_in_6_rolls_l1276_127661

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l1276_127661


namespace cubic_polynomial_sum_l1276_127634

-- Define the roots and their properties according to Vieta's formulas
variables {p q r : ℝ}
axiom root_poly : p * q * r = -1
axiom pq_sum : p * q + p * r + q * r = -3
axiom roots_sum : p + q + r = 0

-- Define the target equality to prove
theorem cubic_polynomial_sum :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 3 :=
by
  sorry

end cubic_polynomial_sum_l1276_127634


namespace solve_system_of_equations_l1276_127668

theorem solve_system_of_equations :
  {p : ℝ × ℝ | 
    (p.1^2 + p.2 + 1) * (p.2^2 + p.1 + 1) = 4 ∧
    (p.1^2 + p.2)^2 + (p.2^2 + p.1)^2 = 2} =
  {(0, 1), (1, 0), 
   ( (-1 + Real.sqrt 5) / 2, (-1 + Real.sqrt 5) / 2),
   ( (-1 - Real.sqrt 5) / 2, (-1 - Real.sqrt 5) / 2) } :=
by
  sorry

end solve_system_of_equations_l1276_127668


namespace inequality_correct_l1276_127677

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end inequality_correct_l1276_127677


namespace arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l1276_127650

-- Question 1
theorem arithmetic_sequence_n (a1 a4 a10 : ℤ) (d : ℤ) (n : ℤ) (Sn : ℤ) 
  (h1 : a1 + 3 * d = a4) 
  (h2 : a1 + 9 * d = a10)
  (h3 : Sn = n * (2 * a1 + (n - 1) * d) / 2)
  (h4 : a4 = 10)
  (h5 : a10 = -2)
  (h6 : Sn = 60)
  : n = 5 ∨ n = 6 := 
sorry

-- Question 2
theorem sum_arithmetic_sequence_S17 (a1 : ℤ) (d : ℤ) (a_n1 : ℤ → ℤ) (S17 : ℤ)
  (h1 : a1 = -7)
  (h2 : ∀ n, a_n1 (n + 1) = a_n1 n + d)
  (h3 : S17 = 17 * (2 * a1 + 16 * d) / 2)
  : S17 = 153 := 
sorry

-- Question 3
theorem arithmetic_sequence_S13 (a_2 a_7 a_12 : ℤ) (S13 : ℤ)
  (h1 : a_2 + a_7 + a_12 = 24)
  (h2 : S13 = a_7 * 13)
  : S13 = 104 := 
sorry

end arithmetic_sequence_n_sum_arithmetic_sequence_S17_arithmetic_sequence_S13_l1276_127650


namespace find_constants_l1276_127648

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_constants (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_min_value : ∃ x : ℝ, a * csc (b * x + c) = 3)
  (h_period : ∀ x, a * csc (b * (x + 4 * Real.pi) + c) = a * csc (b * x + c)) :
  a = 3 ∧ b = (1 / 2) :=
by
  sorry

end find_constants_l1276_127648


namespace total_is_correct_l1276_127679

-- Define the given conditions.
def dividend : ℕ := 55
def divisor : ℕ := 11
def quotient := dividend / divisor
def total := dividend + quotient + divisor

-- State the theorem to be proven.
theorem total_is_correct : total = 71 := by sorry

end total_is_correct_l1276_127679


namespace union_A_B_l1276_127656

def A : Set ℝ := {x | |x| < 3}
def B : Set ℝ := {x | 2 - x > 0}

theorem union_A_B (x : ℝ) : (x ∈ A ∨ x ∈ B) ↔ x < 3 := by
  sorry

end union_A_B_l1276_127656


namespace sandbag_weight_proof_l1276_127696

-- Define all given conditions
def bag_capacity : ℝ := 250
def fill_percentage : ℝ := 0.80
def material_weight_multiplier : ℝ := 1.40 -- since 40% heavier means 1 + 0.40
def empty_bag_weight : ℝ := 0

-- Using these definitions, form the goal to prove
theorem sandbag_weight_proof : 
  (fill_percentage * bag_capacity * material_weight_multiplier) + empty_bag_weight = 280 :=
by
  sorry

end sandbag_weight_proof_l1276_127696


namespace simplify_expr_l1276_127607

variable (a b : ℤ)

theorem simplify_expr :
  (22 * a + 60 * b) + (10 * a + 29 * b) - (9 * a + 50 * b) = 23 * a + 39 * b :=
by
  sorry

end simplify_expr_l1276_127607


namespace logarithm_base_l1276_127626

theorem logarithm_base (x : ℝ) (b : ℝ) : (9 : ℝ)^(x + 5) = (16 : ℝ)^x → b = 16 / 9 → x = Real.log 9^5 / Real.log b := by sorry

end logarithm_base_l1276_127626


namespace duration_of_period_l1276_127642

variable (t : ℝ)

theorem duration_of_period:
  (2800 * 0.185 * t - 2800 * 0.15 * t = 294) ↔ (t = 3) :=
by
  sorry

end duration_of_period_l1276_127642


namespace find_k_l1276_127616

theorem find_k (k : ℤ) (x : ℚ) (h1 : 5 * x + 3 * k = 24) (h2 : 5 * x + 3 = 0) : k = 9 := 
by
  sorry

end find_k_l1276_127616


namespace center_of_circle_l1276_127671

theorem center_of_circle (
  center : ℝ × ℝ
) :
  (∀ (p : ℝ × ℝ), (p.1 * 3 + p.2 * 4 = 24) ∨ (p.1 * 3 + p.2 * 4 = -6) → (dist center p = dist center p)) ∧
  (center.1 * 3 - center.2 = 0)
  → center = (3 / 5, 9 / 5) :=
by
  sorry

end center_of_circle_l1276_127671


namespace number_of_truthful_monkeys_l1276_127666

-- Define the conditions of the problem
def num_tigers : ℕ := 100
def num_foxes : ℕ := 100
def num_monkeys : ℕ := 100
def total_groups : ℕ := 100
def animals_per_group : ℕ := 3
def yes_tiger : ℕ := 138
def yes_fox : ℕ := 188

-- Problem statement to be proved
theorem number_of_truthful_monkeys :
  ∃ m : ℕ, m = 76 ∧
  ∃ (x y z m n : ℕ),
    -- The number of monkeys mixed with tigers
    x + 2 * (74 - y) = num_monkeys ∧

    -- Given constraints
    m ∈ {n : ℕ | n ≤ x} ∧
    n ∈ {n : ℕ | n ≤ (num_foxes - x)} ∧

    -- Equation setup and derived equations
    (x - m) + (num_foxes - y) + n = yes_tiger ∧
    m + (num_tigers - x - n) + (num_tigers - z) = yes_fox ∧
    y + z = 74 ∧
    
    -- ensuring the groups are valid
    2 * (74 - y) = z :=

sorry

end number_of_truthful_monkeys_l1276_127666


namespace man_l1276_127641

theorem man's_age_twice_son's_age_in_2_years
  (S : ℕ) (M : ℕ) (Y : ℕ)
  (h1 : M = S + 24)
  (h2 : S = 22)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 := by
  sorry

end man_l1276_127641


namespace taxi_fare_l1276_127657

theorem taxi_fare (x : ℝ) (h : 3.00 + 0.25 * ((x - 0.75) / 0.1) = 12) : x = 4.35 :=
  sorry

end taxi_fare_l1276_127657


namespace squareable_numbers_l1276_127635

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : ℕ → ℕ), (∀ i, 1 ≤ perm i ∧ perm i ≤ n) ∧ (∀ i, ∃ k, perm i + i = k * k)

theorem squareable_numbers : is_squareable 9 ∧ is_squareable 15 ∧ ¬ is_squareable 7 ∧ ¬ is_squareable 11 :=
by sorry

end squareable_numbers_l1276_127635


namespace subset_relation_l1276_127632

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x + 2}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (x - 4) / Real.log 2}

-- State the proof problem
theorem subset_relation : N ⊆ M := 
sorry

end subset_relation_l1276_127632


namespace cubic_root_form_addition_l1276_127636

theorem cubic_root_form_addition (p q r : ℕ) 
(h_root_form : ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧ x = (p^(1/3) + q^(1/3) + 2) / r) : 
  p + q + r = 10 :=
sorry

end cubic_root_form_addition_l1276_127636


namespace log_diff_eq_35_l1276_127604

theorem log_diff_eq_35 {a b : ℝ} (h₁ : a > b) (h₂ : b > 1)
  (h₃ : (1 / Real.log a / Real.log b) + (1 / (Real.log b / Real.log a)) = Real.sqrt 1229) :
  (1 / (Real.log b / Real.log (a * b))) - (1 / (Real.log a / Real.log (a * b))) = 35 :=
sorry

end log_diff_eq_35_l1276_127604


namespace emerie_dimes_count_l1276_127658

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end emerie_dimes_count_l1276_127658


namespace simplify_fraction_l1276_127688

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l1276_127688


namespace methane_tetrahedron_dot_product_l1276_127605

noncomputable def tetrahedron_vectors_dot_product_sum : ℝ :=
  let edge_length := 1
  let dot_product := -1 / 3 * edge_length^2
  let pair_count := 6 -- number of pairs in sum of dot products
  pair_count * dot_product

theorem methane_tetrahedron_dot_product :
  tetrahedron_vectors_dot_product_sum = - (3 / 4) := by
  sorry

end methane_tetrahedron_dot_product_l1276_127605


namespace binomial_square_l1276_127630

theorem binomial_square (p : ℝ) : (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 24 * x + p) → p = 16 := by
  sorry

end binomial_square_l1276_127630


namespace find_incorrect_statement_l1276_127643

def statement_A := ∀ (P Q : Prop), (P → Q) → (¬Q → ¬P)
def statement_B := ∀ (P : Prop), ((¬P) → false) → P
def statement_C := ∀ (shape : Type), (∃ s : shape, true) → false
def statement_D := ∀ (P : ℕ → Prop), P 0 → (∀ n, P n → P (n + 1)) → ∀ n, P n
def statement_E := ∀ {α : Type} (p : Prop), (¬p ∨ p)

theorem find_incorrect_statement : statement_C :=
sorry

end find_incorrect_statement_l1276_127643


namespace shop_width_correct_l1276_127631

-- Definition of the shop's monthly rent
def monthly_rent : ℝ := 2400

-- Definition of the shop's length in feet
def shop_length : ℝ := 10

-- Definition of the annual rent per square foot
def annual_rent_per_sq_ft : ℝ := 360

-- The mathematical assertion that the width of the shop is 8 feet
theorem shop_width_correct (width : ℝ) :
  (monthly_rent * 12) / annual_rent_per_sq_ft / shop_length = width :=
by
  sorry

end shop_width_correct_l1276_127631


namespace clock_hands_overlap_24_hours_l1276_127611

theorem clock_hands_overlap_24_hours : 
  (∀ t : ℕ, t < 12 →  ∃ n : ℕ, (n = 11 ∧ (∃ h m : ℕ, h * 60 + m = t * 60 + m))) →
  (∃ k : ℕ, k = 22) :=
by
  sorry

end clock_hands_overlap_24_hours_l1276_127611


namespace price_decrease_required_to_initial_l1276_127672

theorem price_decrease_required_to_initial :
  let P0 := 100.0
  let P1 := P0 * 1.15
  let P2 := P1 * 0.90
  let P3 := P2 * 1.20
  let P4 := P3 * 0.70
  let P5 := P4 * 1.10
  let P6 := P5 * (1.0 - d / 100.0)
  P6 = P0 -> d = 5.0 :=
by
  sorry

end price_decrease_required_to_initial_l1276_127672


namespace Taylor_needs_14_jars_l1276_127682

noncomputable def standard_jar_volume : ℕ := 60
noncomputable def big_container_volume : ℕ := 840

theorem Taylor_needs_14_jars : big_container_volume / standard_jar_volume = 14 :=
by sorry

end Taylor_needs_14_jars_l1276_127682


namespace age_difference_l1276_127673

variable (A B C : ℕ)

def age_relationship (B C : ℕ) : Prop :=
  B = 2 * C

def total_ages (A B C : ℕ) : Prop :=
  A + B + C = 72

theorem age_difference (B : ℕ) (hB : B = 28) (h1 : age_relationship B C) (h2 : total_ages A B C) :
  A - B = 2 :=
sorry

end age_difference_l1276_127673


namespace numberOfWaysToChooseLeadership_is_correct_l1276_127628

noncomputable def numberOfWaysToChooseLeadership (totalMembers : ℕ) : ℕ :=
  let choicesForGovernor := totalMembers
  let remainingAfterGovernor := totalMembers - 1

  let choicesForDeputies := Nat.choose remainingAfterGovernor 3
  let remainingAfterDeputies := remainingAfterGovernor - 3

  let choicesForLieutenants1 := Nat.choose remainingAfterDeputies 3
  let remainingAfterLieutenants1 := remainingAfterDeputies - 3

  let choicesForLieutenants2 := Nat.choose remainingAfterLieutenants1 3
  let remainingAfterLieutenants2 := remainingAfterLieutenants1 - 3

  let choicesForLieutenants3 := Nat.choose remainingAfterLieutenants2 3
  let remainingAfterLieutenants3 := remainingAfterLieutenants2 - 3

  let choicesForSubordinates : List ℕ := 
    (List.range 8).map (λ i => Nat.choose (remainingAfterLieutenants3 - 2*i) 2)

  choicesForGovernor 
  * choicesForDeputies 
  * choicesForLieutenants1 
  * choicesForLieutenants2 
  * choicesForLieutenants3 
  * List.prod choicesForSubordinates

theorem numberOfWaysToChooseLeadership_is_correct : 
  numberOfWaysToChooseLeadership 35 = 
    35 * Nat.choose 34 3 * Nat.choose 31 3 * Nat.choose 28 3 * Nat.choose 25 3 *
    Nat.choose 16 2 * Nat.choose 14 2 * Nat.choose 12 2 * Nat.choose 10 2 *
    Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2 :=
by
  sorry

end numberOfWaysToChooseLeadership_is_correct_l1276_127628


namespace jenna_average_speed_l1276_127678

theorem jenna_average_speed (total_distance : ℕ) (total_time : ℕ) 
(first_segment_speed : ℕ) (second_segment_speed : ℕ) (third_segment_speed : ℕ) : 
  total_distance = 150 ∧ total_time = 2 ∧ first_segment_speed = 50 ∧ 
  second_segment_speed = 70 → third_segment_speed = 105 := 
by 
  intros h
  sorry

end jenna_average_speed_l1276_127678


namespace original_wage_before_increase_l1276_127639

theorem original_wage_before_increase (new_wage : ℝ) (increase_rate : ℝ) (original_wage : ℝ) (h : new_wage = original_wage + increase_rate * original_wage) : 
  new_wage = 42 → increase_rate = 0.50 → original_wage = 28 :=
by
  intros h_new_wage h_increase_rate
  have h1 : new_wage = 42 := h_new_wage
  have h2 : increase_rate = 0.50 := h_increase_rate
  have h3 : new_wage = original_wage + increase_rate * original_wage := h
  sorry

end original_wage_before_increase_l1276_127639


namespace positive_integer_solutions_l1276_127699

theorem positive_integer_solutions :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x + y + x * y = 2008 ∧
  ((x = 6 ∧ y = 286) ∨ (x = 286 ∧ y = 6) ∨ (x = 40 ∧ y = 48) ∨ (x = 48 ∧ y = 40)) :=
by
  sorry

end positive_integer_solutions_l1276_127699


namespace part1_part2_l1276_127667

def proposition_p (m : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → 2 * x - 4 ≥ m^2 - 5 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ x^2 - 2 * x + m - 1 ≤ 0

theorem part1 (m : ℝ) : proposition_p m → 1 ≤ m ∧ m ≤ 4 := 
sorry

theorem part2 (m : ℝ) : (proposition_p m ∨ proposition_q m) → m ≤ 4 := 
sorry

end part1_part2_l1276_127667


namespace triangle_inequality_violation_l1276_127602

theorem triangle_inequality_violation (a b c : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 7) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  rw [ha, hb, hc]
  simp
  sorry

end triangle_inequality_violation_l1276_127602


namespace maximize_Sn_l1276_127660

theorem maximize_Sn (a1 : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : a1 > 0)
  (h2 : a1 + 9 * (a1 + 5 * d) = 0)
  (h_sn : ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)) :
  ∃ n_max, ∀ n, S n ≤ S n_max ∧ n_max = 5 :=
by
  sorry

end maximize_Sn_l1276_127660


namespace frequency_of_group_samples_l1276_127662

-- Conditions
def sample_capacity : ℕ := 32
def group_frequency : ℝ := 0.125

-- Theorem statement
theorem frequency_of_group_samples : group_frequency * sample_capacity = 4 :=
by sorry

end frequency_of_group_samples_l1276_127662


namespace sum_of_fractions_l1276_127644

theorem sum_of_fractions : (1 / 1) + (2 / 2) + (3 / 3) = 3 := 
by 
  norm_num

end sum_of_fractions_l1276_127644


namespace arithmetic_sequence_expression_l1276_127646

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (n < m) → (a (m + 1) - a m = a (n + 1) - a n)

theorem arithmetic_sequence_expression
  (h_arith_seq : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = -3) :
  ∀ n : ℕ, a n = -2 * n + 3 :=
  sorry

end arithmetic_sequence_expression_l1276_127646


namespace five_less_than_sixty_percent_of_cats_l1276_127638

theorem five_less_than_sixty_percent_of_cats (hogs cats : ℕ) 
  (hogs_eq : hogs = 3 * cats)
  (hogs_value : hogs = 75) : 
  5 < 60 * cats / 100 :=
by {
  sorry
}

end five_less_than_sixty_percent_of_cats_l1276_127638


namespace inequality_solution_l1276_127687

theorem inequality_solution :
  { x : ℝ // x < 2 ∨ (3 < x ∧ x < 6) ∨ (7 < x ∧ x < 8) } →
  ((x - 3) * (x - 5) * (x - 7)) / ((x - 2) * (x - 6) * (x - 8)) > 0 :=
by
  sorry

end inequality_solution_l1276_127687


namespace original_average_of_numbers_l1276_127618

theorem original_average_of_numbers 
  (A : ℝ) 
  (h : (A * 15) + (11 * 15) = 51 * 15) : 
  A = 40 :=
sorry

end original_average_of_numbers_l1276_127618
