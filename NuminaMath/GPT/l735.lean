import Mathlib

namespace problem_l735_73596

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem problem (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : 
  ((f b - f a) / (b - a) < 1 / (a * (a + 1))) :=
by
  sorry -- Proof steps go here

end problem_l735_73596


namespace larger_number_of_two_l735_73530

theorem larger_number_of_two (A B : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 28) (h_factors : A % hcf = 0 ∧ B % hcf = 0) 
  (h_f1 : factor1 = 12) (h_f2 : factor2 = 15)
  (h_lcm : Nat.lcm A B = hcf * factor1 * factor2)
  (h_coprime : Nat.gcd (A / hcf) (B / hcf) = 1)
  : max A B = 420 := 
sorry

end larger_number_of_two_l735_73530


namespace qy_length_l735_73523

theorem qy_length (Q : Type*) (C : Type*) (X Y Z : Q) (QX QZ QY : ℝ) 
  (h1 : 5 = QX)
  (h2 : QZ = 2 * (QY - QX))
  (PQ_theorem : QX * QY = QZ^2) :
  QY = 10 :=
by
  sorry

end qy_length_l735_73523


namespace least_faces_combined_l735_73569

noncomputable def least_number_of_faces (c d : ℕ) : ℕ :=
c + d

theorem least_faces_combined (c d : ℕ) (h_cge8 : c ≥ 8) (h_dge8 : d ≥ 8)
  (h_sum9_prob : 8 / (c * d) = 1 / 2 * 16 / (c * d))
  (h_sum15_prob : ∃ m : ℕ, m / (c * d) = 1 / 15) :
  least_number_of_faces c d = 28 := sorry

end least_faces_combined_l735_73569


namespace first_train_left_time_l735_73593

-- Definitions for conditions
def speed_first_train := 45
def speed_second_train := 90
def meeting_distance := 90

-- Prove the statement
theorem first_train_left_time (T : ℝ) (time_meeting : ℝ) :
  (time_meeting - T = 2) →
  (∀ t, 0 ≤ t → t ≤ 1 → speed_first_train * t ≤ meeting_distance) →
  (∀ t, 1 ≤ t → speed_first_train * (T + t) + speed_second_train * (t - 1) = meeting_distance) →
  (time_meeting = 2 + T) :=
by
  sorry

end first_train_left_time_l735_73593


namespace negation_of_proposition_l735_73594

theorem negation_of_proposition (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
sorry

end negation_of_proposition_l735_73594


namespace rounding_increases_value_l735_73577

theorem rounding_increases_value (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (rounded_a : ℕ := a + 1)
  (rounded_b : ℕ := b - 1)
  (rounded_c : ℕ := c + 1)
  (rounded_d : ℕ := d + 1) :
  (rounded_a * rounded_d) / rounded_b + rounded_c > (a * d) / b + c := 
sorry

end rounding_increases_value_l735_73577


namespace determine_a_for_line_l735_73536

theorem determine_a_for_line (a : ℝ) (h : a ≠ 0)
  (intercept_condition : ∃ (k : ℝ), 
    ∀ x y : ℝ, (a * x - 6 * y - 12 * a = 0) → (x = 12) ∧ (y = 2 * a * x / 6) ∧ (12 = 3 * (-2 * a))) : 
  a = -2 :=
by
  sorry

end determine_a_for_line_l735_73536


namespace first_group_men_l735_73574

theorem first_group_men (M : ℕ) (h : M * 15 = 25 * 24) : M = 40 := sorry

end first_group_men_l735_73574


namespace probability_factor_of_36_is_1_over_4_l735_73544

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l735_73544


namespace xy_value_l735_73590

theorem xy_value {x y : ℝ} (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 :=
by
  sorry

end xy_value_l735_73590


namespace original_six_digit_number_is_285714_l735_73598

theorem original_six_digit_number_is_285714 
  (N : ℕ) 
  (h1 : ∃ x, N = 200000 + x ∧ 10 * x + 2 = 3 * (200000 + x)) :
  N = 285714 := 
sorry

end original_six_digit_number_is_285714_l735_73598


namespace intersection_M_N_eq_set_l735_73511

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- The theorem to be proved
theorem intersection_M_N_eq_set : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_set_l735_73511


namespace maximum_a3_S10_l735_73554

-- Given definitions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def conditions (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (∀ n, a n > 0) ∧ (a 1 + a 3 + a 8 = a 4 ^ 2)

-- The problem statement
theorem maximum_a3_S10 (a : ℕ → ℝ) (h : conditions a) : 
  (∃ S : ℝ, S = a 3 * ((10 / 2) * (a 1 + a 10)) ∧ S ≤ 375 / 4) :=
sorry

end maximum_a3_S10_l735_73554


namespace sum_of_50th_terms_l735_73587

open Nat

-- Definition of arithmetic sequence
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Definition of geometric sequence
def geometric_sequence (g₁ r n : ℕ) : ℕ := g₁ * r^(n - 1)

-- Prove the sum of the 50th terms of the given sequences
theorem sum_of_50th_terms : 
  arithmetic_sequence 3 6 50 + geometric_sequence 2 3 50 = 297 + 2 * 3^49 :=
by
  sorry

end sum_of_50th_terms_l735_73587


namespace like_terms_product_l735_73567

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l735_73567


namespace hyperbola_s_squared_l735_73539

theorem hyperbola_s_squared 
  (s : ℝ) 
  (a b : ℝ) 
  (h1 : a = 3)
  (h2 : b^2 = 144 / 13) 
  (h3 : (2, s) ∈ {p : ℝ × ℝ | (p.2)^2 / a^2 - (p.1)^2 / b^2 = 1}) :
  s^2 = 441 / 36 :=
by sorry

end hyperbola_s_squared_l735_73539


namespace height_of_parallelogram_l735_73501

theorem height_of_parallelogram (Area Base : ℝ) (h1 : Area = 180) (h2 : Base = 18) : Area / Base = 10 :=
by
  sorry

end height_of_parallelogram_l735_73501


namespace find_2a6_minus_a4_l735_73524

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) = 2 * a (n + 1) - a n

theorem find_2a6_minus_a4 {a : ℕ → ℤ} 
  (h_seq : is_arithmetic_sequence a)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 6 - a 4 = 24 :=
by
  sorry

end find_2a6_minus_a4_l735_73524


namespace april_plant_arrangement_l735_73562

theorem april_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 4
  let total_units := (basil_plants - 2) + 1 + 1
  (Nat.factorial total_units) * (Nat.factorial tomato_plants) * (Nat.factorial 2) = 5760 :=
by
  sorry

end april_plant_arrangement_l735_73562


namespace number_in_2019th_field_l735_73563

theorem number_in_2019th_field (f : ℕ → ℕ) (h1 : ∀ n, 0 < f n) (h2 : ∀ n, f n * f (n+1) * f (n+2) = 2018) :
  f 2018 = 1009 := sorry

end number_in_2019th_field_l735_73563


namespace sequence_positive_from_26_l735_73504

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := 4 * n - 102

-- State the theorem that for all n ≥ 26, a_n > 0.
theorem sequence_positive_from_26 (n : ℕ) (h : n ≥ 26) : a_n n > 0 := by
  sorry

end sequence_positive_from_26_l735_73504


namespace no_solutions_exist_l735_73564

theorem no_solutions_exist (m n : ℤ) : ¬(m^2 = n^2 + 1954) :=
by sorry

end no_solutions_exist_l735_73564


namespace complement_union_l735_73556

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l735_73556


namespace consecutive_numbers_product_l735_73550

theorem consecutive_numbers_product : 
  ∃ n : ℕ, (n + n + 1 = 11) ∧ (n * (n + 1) * (n + 2) = 210) :=
sorry

end consecutive_numbers_product_l735_73550


namespace min_and_max_f_l735_73503

noncomputable def f (x : ℝ) : ℝ := -2 * x + 1

theorem min_and_max_f :
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≥ -9) ∧ (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ 1) :=
by
  sorry

end min_and_max_f_l735_73503


namespace transistors_in_2010_l735_73519

-- Define initial conditions
def initial_transistors : ℕ := 500000
def years_passed : ℕ := 15
def tripling_period : ℕ := 3
def tripling_factor : ℕ := 3

-- Define the function to compute the number of transistors after a number of years
noncomputable def final_transistors (initial : ℕ) (years : ℕ) (period : ℕ) (factor : ℕ) : ℕ :=
  initial * factor ^ (years / period)

-- State the proposition we aim to prove
theorem transistors_in_2010 : final_transistors initial_transistors years_passed tripling_period tripling_factor = 121500000 := 
by 
  sorry

end transistors_in_2010_l735_73519


namespace problem1_problem2_l735_73515

theorem problem1 :
  (2 / 3) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3) * Real.sqrt 27 = - (4 / 3) * Real.sqrt 6 :=
sorry

theorem problem2 :
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 :=
sorry

end problem1_problem2_l735_73515


namespace right_triangle_c_squared_value_l735_73560

theorem right_triangle_c_squared_value (a b c : ℕ) (h : a = 9) (k : b = 12) (right_triangle : True) :
  c^2 = a^2 + b^2 ∨ c^2 = b^2 - a^2 :=
by sorry

end right_triangle_c_squared_value_l735_73560


namespace evaluate_g_x_plus_2_l735_73512

theorem evaluate_g_x_plus_2 (x : ℝ) (h₁ : x ≠ -3/2) (h₂ : x ≠ 2) : 
  (2 * (x + 2) + 3) / ((x + 2) - 2) = (2 * x + 7) / x :=
by 
  sorry

end evaluate_g_x_plus_2_l735_73512


namespace value_of_each_walmart_gift_card_l735_73527

variable (best_buy_value : ℕ) (best_buy_count : ℕ) (walmart_count : ℕ) (points_sent_bb : ℕ) (points_sent_wm : ℕ) (total_returnable : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  best_buy_value = 500 ∧
  best_buy_count = 6 ∧
  walmart_count = 9 ∧
  points_sent_bb = 1 ∧
  points_sent_wm = 2 ∧
  total_returnable = 3900

-- Result to prove
theorem value_of_each_walmart_gift_card : conditions best_buy_value best_buy_count walmart_count points_sent_bb points_sent_wm total_returnable →
  (total_returnable - ((best_buy_count - points_sent_bb) * best_buy_value)) / (walmart_count - points_sent_wm) = 200 :=
by
  intros h
  rcases h with
    ⟨hbv, hbc, hwc, hsbb, hswm, htr⟩
  sorry

end value_of_each_walmart_gift_card_l735_73527


namespace total_potatoes_l735_73568

theorem total_potatoes (jane_potatoes mom_potatoes dad_potatoes : Nat) 
  (h1 : jane_potatoes = 8)
  (h2 : mom_potatoes = 8)
  (h3 : dad_potatoes = 8) :
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  sorry

end total_potatoes_l735_73568


namespace inequality_solution_range_l735_73566

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| + |x| ≤ a) ↔ a ≥ 2 :=
by
  sorry

end inequality_solution_range_l735_73566


namespace fraction_of_x_l735_73585

theorem fraction_of_x (w x y f : ℝ) (h1 : 2 / w + f * x = 2 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : f = 2 / x - 2 := 
sorry

end fraction_of_x_l735_73585


namespace geometric_sequence_condition_l735_73522

theorem geometric_sequence_condition {a : ℕ → ℝ} (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) : 
  (a 3 * a 5 = 16) ↔ a 4 = 4 :=
sorry

end geometric_sequence_condition_l735_73522


namespace cubic_polynomial_Q_l735_73532

noncomputable def Q (x : ℝ) : ℝ := 27 * x^3 - 162 * x^2 + 297 * x - 156

theorem cubic_polynomial_Q {a b c : ℝ} 
  (h_roots : ∀ x, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c)
  (h_vieta_sum : a + b + c = 6)
  (h_vieta_prod_sum : ab + bc + ca = 11)
  (h_vieta_prod : abc = 6)
  (hQ : Q a = b + c) 
  (hQb : Q b = a + c) 
  (hQc : Q c = a + b) 
  (hQ_sum : Q (a + b + c) = -27) :
  Q x = 27 * x^3 - 162 * x^2 + 297 * x - 156 :=
by { sorry }

end cubic_polynomial_Q_l735_73532


namespace find_m_l735_73534

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (h_intersection : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : 
  m = 6 :=
sorry

end find_m_l735_73534


namespace line_equation_of_point_and_slope_angle_l735_73584

theorem line_equation_of_point_and_slope_angle 
  (p : ℝ × ℝ) (θ : ℝ)
  (h₁ : p = (-1, 2))
  (h₂ : θ = 45) :
  ∃ (a b c : ℝ), a * (p.1) + b * (p.2) + c = 0 ∧ (a * 1 + b * 1 = c) :=
sorry

end line_equation_of_point_and_slope_angle_l735_73584


namespace painting_time_l735_73578

theorem painting_time (karl_time leo_time : ℝ) (t : ℝ) (break_time : ℝ) : 
  karl_time = 6 → leo_time = 8 → break_time = 0.5 → 
  (1 / karl_time + 1 / leo_time) * (t - break_time) = 1 :=
by
  intros h_karl h_leo h_break
  rw [h_karl, h_leo, h_break]
  -- sorry to skip the proof
  sorry

end painting_time_l735_73578


namespace diet_equivalence_l735_73555

variable (B E L D A : ℕ)

theorem diet_equivalence :
  (17 * B = 170 * L) →
  (100000 * A = 50 * L) →
  (10 * B = 4 * E) →
  12 * E = 600000 * A :=
sorry

end diet_equivalence_l735_73555


namespace problem1_problem2_l735_73570

def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6*x - 72 <= 0) ∧ (x^2 + x - 6 > 0)

theorem problem1 (x : ℝ) (a : ℝ) (h : a = -1): (p x a ∨ q x) → (-6 ≤ x ∧ x < -3) ∨ (1 < x ∧ x ≤ 12) :=
sorry

theorem problem2 (a : ℝ): (¬ ∃ x : ℝ, p x a) → (¬ ∃ x : ℝ, q x) → (-4 ≤ a ∧ a ≤ -2) :=
sorry

end problem1_problem2_l735_73570


namespace original_radius_of_cylinder_l735_73531

theorem original_radius_of_cylinder (r y : ℝ) 
  (h₁ : 3 * π * ((r + 5)^2 - r^2) = y) 
  (h₂ : 5 * π * r^2 = y)
  (h₃ : 3 > 0) :
  r = 7.5 :=
by
  sorry

end original_radius_of_cylinder_l735_73531


namespace probability_age_less_than_20_l735_73533

theorem probability_age_less_than_20 (total_people : ℕ) (over_30_years : ℕ) 
  (less_than_20_years : ℕ) (h1 : total_people = 120) (h2 : over_30_years = 90) 
  (h3 : less_than_20_years = total_people - over_30_years) : 
  (less_than_20_years : ℚ) / total_people = 1 / 4 :=
by {
  sorry
}

end probability_age_less_than_20_l735_73533


namespace total_bottles_in_market_l735_73575

theorem total_bottles_in_market (j w : ℕ) (hj : j = 34) (hw : w = 3 / 2 * j + 3) : j + w = 88 :=
by
  sorry

end total_bottles_in_market_l735_73575


namespace percentage_increase_in_second_year_l735_73595

def initial_deposit : ℝ := 5000
def first_year_balance : ℝ := 5500
def two_year_increase_percentage : ℝ := 21
def second_year_increase_percentage : ℝ := 10

theorem percentage_increase_in_second_year
  (initial_deposit first_year_balance : ℝ) 
  (two_year_increase_percentage : ℝ) 
  (h1 : first_year_balance = initial_deposit + 500) 
  (h2 : (initial_deposit * (1 + two_year_increase_percentage / 100)) = initial_deposit * 1.21) 
  : second_year_increase_percentage = 10 := 
sorry

end percentage_increase_in_second_year_l735_73595


namespace intersection_M_N_l735_73540

def M : Set ℝ := { x | x^2 + x - 2 < 0 }
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l735_73540


namespace simplify_expression_l735_73546

theorem simplify_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 :=
by
  sorry

end simplify_expression_l735_73546


namespace find_h_l735_73508

theorem find_h (h : ℝ) : (∀ x : ℝ, x^2 - 4 * h * x = 8) 
    ∧ (∀ r s : ℝ, r + s = 4 * h ∧ r * s = -8 → r^2 + s^2 = 18) 
    → h = (Real.sqrt 2) / 4 ∨ h = -(Real.sqrt 2) / 4 :=
by
  sorry

end find_h_l735_73508


namespace identical_lines_pairs_count_l735_73545

theorem identical_lines_pairs_count : 
  ∃ P : Finset (ℝ × ℝ), (∀ p ∈ P, 
    (∃ a b, p = (a, b) ∧ 
      (∀ x y, 2 * x + a * y + b = 0 ↔ b * x + 3 * y - 9 = 0))) ∧ P.card = 2 :=
sorry

end identical_lines_pairs_count_l735_73545


namespace part_one_part_two_l735_73520

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 5 then (16 / (9 - x) - 1) else (11 - (2 / 45) * x ^ 2)

theorem part_one (k : ℝ) (h : 1 ≤ k ∧ k ≤ 4) : k * (16 / (9 - 3) - 1) = 4 → k = 12 / 5 :=
by sorry

theorem part_two (y x : ℝ) (h_y : y = 4) :
  (1 ≤ x ∧ x ≤ 5 ∧ 4 * (16 / (9 - x) - 1) ≥ 4) ∨
  (5 < x ∧ x ≤ 15 ∧ 4 * (11 - (2/45) * x ^ 2) ≥ 4) :=
by sorry

end part_one_part_two_l735_73520


namespace overtime_pay_correct_l735_73589

theorem overtime_pay_correct
  (overlap_slow : ℝ := 69) -- Slow clock minute-hand overlap in minutes
  (overlap_normal : ℝ := 12 * 60 / 11) -- Normal clock minute-hand overlap in minutes
  (hours_worked : ℝ := 8) -- The normal working hours a worker believes working
  (hourly_wage : ℝ := 4) -- The normal hourly wage
  (overtime_rate : ℝ := 1.5) -- Overtime pay rate
  (expected_overtime_pay : ℝ := 2.60) -- The expected overtime pay
  
  : hours_worked * (overlap_slow / overlap_normal) * hourly_wage * (overtime_rate - 1) = expected_overtime_pay :=
by
  sorry

end overtime_pay_correct_l735_73589


namespace seminar_total_cost_l735_73538

theorem seminar_total_cost 
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ) 
  (food_allowance_per_teacher : ℝ)
  (total_cost : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10) 
  (h4 : food_allowance_per_teacher = 10)
  (h5 : total_cost = regular_fee * num_teachers * (1 - discount_rate) + food_allowance_per_teacher * num_teachers) :
  total_cost = 1525 := 
sorry

end seminar_total_cost_l735_73538


namespace necessary_and_sufficient_condition_for_tangency_l735_73542

-- Given conditions
variables (ρ θ D E : ℝ)

-- Definition of the circle in polar coordinates and the condition for tangency with the radial axis
def circle_eq : Prop := ρ = D * Real.cos θ + E * Real.sin θ

-- Statement of the proof problem
theorem necessary_and_sufficient_condition_for_tangency :
  (circle_eq ρ θ D E) → (D = 0 ∧ E ≠ 0) :=
sorry

end necessary_and_sufficient_condition_for_tangency_l735_73542


namespace equalize_foma_ierema_l735_73599

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l735_73599


namespace condition_s_for_q_condition_r_for_q_condition_p_for_s_l735_73582

variables {p q r s : Prop}

-- Given conditions from a)
axiom h₁ : r → p
axiom h₂ : q → r
axiom h₃ : s → r
axiom h₄ : q → s

-- The corresponding proof problems based on c)
theorem condition_s_for_q : (s ↔ q) :=
by sorry

theorem condition_r_for_q : (r ↔ q) :=
by sorry

theorem condition_p_for_s : (s → p) :=
by sorry

end condition_s_for_q_condition_r_for_q_condition_p_for_s_l735_73582


namespace circles_ACD_and_BCD_orthogonal_l735_73502

-- Define mathematical objects and conditions
variables (A B C D : Point) -- Points in general position on the plane
variables (circle : Point → Point → Point → Circle)

-- Circles intersect orthogonally property
def orthogonal_intersection (c1 c2 : Circle) : Prop :=
  -- Definition of orthogonal intersection of circles goes here (omitted for brevity)
  sorry

-- Given conditions
def circles_ABC_and_ABD_orthogonal : Prop :=
  orthogonal_intersection (circle A B C) (circle A B D)

-- Theorem statement
theorem circles_ACD_and_BCD_orthogonal (h : circles_ABC_and_ABD_orthogonal A B C D circle) :
  orthogonal_intersection (circle A C D) (circle B C D) :=
sorry

end circles_ACD_and_BCD_orthogonal_l735_73502


namespace sequence_eventually_constant_l735_73521

theorem sequence_eventually_constant (n : ℕ) (h : n ≥ 1) : 
  ∃ s, ∀ k ≥ s, (2 ^ (2 ^ k) % n) = (2 ^ (2 ^ (k + 1)) % n) :=
sorry

end sequence_eventually_constant_l735_73521


namespace tan_product_cos_conditions_l735_73526

variable {α β : ℝ}

theorem tan_product_cos_conditions
  (h1 : Real.cos (α + β) = 2 / 3)
  (h2 : Real.cos (α - β) = 1 / 3) :
  Real.tan α * Real.tan β = -1 / 3 :=
sorry

end tan_product_cos_conditions_l735_73526


namespace greatest_third_side_l735_73513

theorem greatest_third_side
  (a b : ℕ)
  (h₁ : a = 7)
  (h₂ : b = 10)
  (c : ℕ)
  (h₃ : a + b + c ≤ 30)
  (h₄ : 3 < c)
  (h₅ : c ≤ 13) :
  c = 13 := 
sorry

end greatest_third_side_l735_73513


namespace number_of_divisors_that_are_multiples_of_2_l735_73576

-- Define the prime factorization of 540
def prime_factorization_540 : ℕ × ℕ × ℕ := (2, 3, 5)

-- Define the constraints for a divisor to be a multiple of 2
def valid_divisor_form (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1

noncomputable def count_divisors (prime_info : ℕ × ℕ × ℕ) : ℕ :=
  let (p1, p2, p3) := prime_info
  2 * 4 * 2 -- Correspond to choices for \( a \), \( b \), and \( c \)

theorem number_of_divisors_that_are_multiples_of_2 (p1 p2 p3 : ℕ) (h : prime_factorization_540 = (p1, p2, p3)) :
  ∃ (count : ℕ), count = 16 :=
by
  use count_divisors (2, 3, 5)
  sorry

end number_of_divisors_that_are_multiples_of_2_l735_73576


namespace find_ks_l735_73514

theorem find_ks (k : ℕ) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end find_ks_l735_73514


namespace pocket_knife_value_l735_73541

noncomputable def value_of_pocket_knife (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    let total_rubles := n * n
    let tens (x : ℕ) := x / 10
    let units (x : ℕ) := x % 10
    let e := units n
    let d := tens n
    let remaining := total_rubles - ((total_rubles / 10) * 10)
    if remaining = 6 then 4 else sorry

theorem pocket_knife_value (n : ℕ) : value_of_pocket_knife n = 2 := by
  sorry

end pocket_knife_value_l735_73541


namespace technicians_count_l735_73507

-- Variables
variables (T R : ℕ)
-- Conditions from the problem
def avg_salary_all := 8000
def avg_salary_tech := 12000
def avg_salary_rest := 6000
def total_workers := 30
def total_salary := avg_salary_all * total_workers

-- Equations based on conditions
def eq1 : T + R = total_workers := sorry
def eq2 : avg_salary_tech * T + avg_salary_rest * R = total_salary := sorry

-- Proof statement (external conditions are reused for clarity)
theorem technicians_count : T = 10 :=
by sorry

end technicians_count_l735_73507


namespace perfect_square_trinomial_l735_73573

theorem perfect_square_trinomial (k : ℤ) : (∀ x : ℤ, x^2 + 2 * (k + 1) * x + 16 = (x + (k + 1))^2) → (k = 3 ∨ k = -5) :=
by
  sorry

end perfect_square_trinomial_l735_73573


namespace calculate_permutation_sum_l735_73529

noncomputable def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem calculate_permutation_sum (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 3) :
  A (2 * n) (n + 3) + A 4 (n + 1) = 744 := by
  sorry

end calculate_permutation_sum_l735_73529


namespace line_intersects_parabola_at_one_point_l735_73549

theorem line_intersects_parabola_at_one_point (k : ℝ) : (∃ y : ℝ, -y^2 - 4 * y + 2 = k) ↔ k = 6 :=
by 
  sorry

end line_intersects_parabola_at_one_point_l735_73549


namespace percentage_decrease_l735_73510

theorem percentage_decrease (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.20 * A) : 
  ∃ y : ℝ, A = C - (y/100) * C ∧ y = 50 / 3 :=
by {
  sorry
}

end percentage_decrease_l735_73510


namespace min_value_eq_ab_squared_l735_73525

noncomputable def min_value (x a b : ℝ) : ℝ := 1 / (x^a * (1 - x)^b)

theorem min_value_eq_ab_squared (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ min_value x a b = (a + b)^2 :=
by
  sorry

end min_value_eq_ab_squared_l735_73525


namespace at_least_one_triangle_l735_73537

theorem at_least_one_triangle {n : ℕ} (h1 : n ≥ 2) (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : 
(points.card = 2 * n) ∧ (segments.card = n^2 + 1) → 
∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ((a, b) ∈ segments ∨ (b, a) ∈ segments) ∧ ((b, c) ∈ segments ∨ (c, b) ∈ segments) ∧ ((c, a) ∈ segments ∨ (a, c) ∈ segments) := 
by 
  sorry

end at_least_one_triangle_l735_73537


namespace range_of_a_l735_73547

def A (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0
def B (x a : ℝ) : Prop := x < a + 1

theorem range_of_a (a : ℝ) : (∃ x : ℝ, A x ∧ B x a) ↔ a > 0 := by
  sorry

end range_of_a_l735_73547


namespace prob_none_given_not_A_l735_73518

-- Definitions based on the conditions
def prob_single (h : ℕ → Prop) : ℝ := 0.2
def prob_double (h1 h2 : ℕ → Prop) : ℝ := 0.1
def prob_triple_given_AB : ℝ := 0.5

-- Assume that h1, h2, and h3 represent the hazards A, B, and C respectively.
variables (A B C : ℕ → Prop)

-- The ultimate theorem we want to prove
theorem prob_none_given_not_A (P : ℕ → Prop) :
  ((1 - (0.2 * 3 + 0.1 * 3) + (prob_triple_given_AB * (prob_single A + prob_double A B))) / (1 - 0.2) = 11 / 9) :=
by
  sorry

end prob_none_given_not_A_l735_73518


namespace total_animal_eyes_l735_73580

def frogs_in_pond := 20
def crocodiles_in_pond := 6
def eyes_per_frog := 2
def eyes_per_crocodile := 2

theorem total_animal_eyes : (frogs_in_pond * eyes_per_frog + crocodiles_in_pond * eyes_per_crocodile) = 52 := by
  sorry

end total_animal_eyes_l735_73580


namespace madeline_rent_l735_73579

noncomputable def groceries : ℝ := 400
noncomputable def medical_expenses : ℝ := 200
noncomputable def utilities : ℝ := 60
noncomputable def emergency_savings : ℝ := 200
noncomputable def hourly_wage : ℝ := 15
noncomputable def hours_worked : ℕ := 138
noncomputable def total_expenses_and_savings : ℝ := groceries + medical_expenses + utilities + emergency_savings
noncomputable def total_earnings : ℝ := hourly_wage * hours_worked

theorem madeline_rent : total_earnings - total_expenses_and_savings = 1210 := by
  sorry

end madeline_rent_l735_73579


namespace proof_set_intersection_l735_73572

def set_M := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def set_N := {x : ℝ | Real.log x ≥ 0}
def set_answer := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem proof_set_intersection : 
  (set_M ∩ set_N) = set_answer := 
by 
  sorry

end proof_set_intersection_l735_73572


namespace Jill_arrives_30_minutes_before_Jack_l735_73591

theorem Jill_arrives_30_minutes_before_Jack
  (d : ℝ) (v_J : ℝ) (v_K : ℝ)
  (h₀ : d = 3) (h₁ : v_J = 12) (h₂ : v_K = 4) :
  (d / v_K - d / v_J) * 60 = 30 :=
by
  sorry

end Jill_arrives_30_minutes_before_Jack_l735_73591


namespace expression_equals_answer_l735_73561

noncomputable def verify_expression : ℚ :=
  15 * (1 / 17) * 34 - (1 / 2)

theorem expression_equals_answer :
  verify_expression = 59 / 2 :=
by
  sorry

end expression_equals_answer_l735_73561


namespace power_addition_identity_l735_73551

theorem power_addition_identity : 
  (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end power_addition_identity_l735_73551


namespace ceil_eq_intervals_l735_73500

theorem ceil_eq_intervals (x : ℝ) :
  (⌈⌈ 3 * x ⌉ + 1 / 2⌉ = ⌈ x - 2 ⌉) ↔ (-1 : ℝ) ≤ x ∧ x < -2 / 3 := 
by
  sorry

end ceil_eq_intervals_l735_73500


namespace fraction_left_handed_l735_73548

def total_participants (k : ℕ) := 15 * k

def red (k : ℕ) := 5 * k
def blue (k : ℕ) := 5 * k
def green (k : ℕ) := 3 * k
def yellow (k : ℕ) := 2 * k

def left_handed_red (k : ℕ) := (1 / 3) * red k
def left_handed_blue (k : ℕ) := (2 / 3) * blue k
def left_handed_green (k : ℕ) := (1 / 2) * green k
def left_handed_yellow (k : ℕ) := (1 / 4) * yellow k

def total_left_handed (k : ℕ) := left_handed_red k + left_handed_blue k + left_handed_green k + left_handed_yellow k

theorem fraction_left_handed (k : ℕ) : 
  (total_left_handed k) / (total_participants k) = 7 / 15 := 
sorry

end fraction_left_handed_l735_73548


namespace pie_slices_l735_73558

theorem pie_slices (total_pies : ℕ) (sold_pies : ℕ) (gifted_pies : ℕ) (left_pieces : ℕ) (eaten_fraction : ℚ) :
  total_pies = 4 →
  sold_pies = 1 →
  gifted_pies = 1 →
  eaten_fraction = 2/3 →
  left_pieces = 4 →
  (total_pies - sold_pies - gifted_pies) * (left_pieces * 3 / (1 - eaten_fraction)) / (total_pies - sold_pies - gifted_pies) = 6 :=
by
  sorry

end pie_slices_l735_73558


namespace not_all_roots_real_l735_73535

-- Define the quintic polynomial with coefficients a5, a4, a3, a2, a1, a0
def quintic_polynomial (a5 a4 a3 a2 a1 a0 : ℝ) (x : ℝ) : ℝ :=
  a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define a predicate for the existence of all real roots
def all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) : Prop :=
  ∀ r : ℝ, quintic_polynomial a5 a4 a3 a2 a1 a0 r = 0

-- Define the main theorem statement
theorem not_all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) :
  2 * a4^2 < 5 * a5 * a3 →
  ¬ all_roots_real a5 a4 a3 a2 a1 a0 :=
by
  sorry

end not_all_roots_real_l735_73535


namespace tangents_product_l735_73517

theorem tangents_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7) 
  (h2 : 2 * Real.sin (2 * (x - y)) = Real.sin (2 * x) * Real.sin (2 * y)) :
  Real.tan x * Real.tan y = -7/6 := 
sorry

end tangents_product_l735_73517


namespace triangle_min_diff_l735_73565

variable (XY YZ XZ : ℕ) -- Declaring the side lengths as natural numbers

theorem triangle_min_diff (h1 : XY < YZ ∧ YZ ≤ XZ) -- Condition for side length relations
  (h2 : XY + YZ + XZ = 2010) -- Condition for the perimeter
  (h3 : XY + YZ > XZ)
  (h4 : XY + XZ > YZ)
  (h5 : YZ + XZ > XY) :
  (YZ - XY) = 1 := -- Statement that the smallest possible value of YZ - XY is 1
sorry

end triangle_min_diff_l735_73565


namespace circular_garden_remaining_grass_area_l735_73553

noncomputable def remaining_grass_area (diameter : ℝ) (path_width: ℝ) : ℝ :=
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let path_area := path_width * diameter
  circle_area - path_area

theorem circular_garden_remaining_grass_area :
  remaining_grass_area 10 2 = 25 * Real.pi - 20 := sorry

end circular_garden_remaining_grass_area_l735_73553


namespace total_bird_families_l735_73597

-- Declare the number of bird families that flew to Africa
def a : Nat := 47

-- Declare the number of bird families that flew to Asia
def b : Nat := 94

-- Condition that Asia's number of bird families matches Africa + 47 more
axiom h : b = a + 47

-- Prove the total number of bird families is 141
theorem total_bird_families : a + b = 141 :=
by
  -- Insert proof here
  sorry

end total_bird_families_l735_73597


namespace social_media_usage_in_week_l735_73516

def days_in_week : ℕ := 7
def daily_phone_usage : ℕ := 16
def daily_social_media_usage : ℕ := daily_phone_usage / 2

theorem social_media_usage_in_week :
  daily_social_media_usage * days_in_week = 56 :=
by
  sorry

end social_media_usage_in_week_l735_73516


namespace perpendicular_lines_a_equals_one_l735_73583

theorem perpendicular_lines_a_equals_one
  (a : ℝ)
  (l1 : ∀ x y : ℝ, x - 2 * y + 1 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + a * y - 1 = 0)
  (perpendicular : ∀ x y : ℝ, (x - 2 * y + 1 = 0) ∧ (2 * x + a * y - 1 = 0) → 
    (-(1 / -2) * -(2 / a)) = -1) :
  a = 1 :=
by
  sorry

end perpendicular_lines_a_equals_one_l735_73583


namespace balloon_difference_l735_73592

theorem balloon_difference 
  (your_balloons : ℕ := 7) 
  (friend_balloons : ℕ := 5) : 
  your_balloons - friend_balloons = 2 := 
by 
  sorry

end balloon_difference_l735_73592


namespace only_rational_root_is_one_l735_73586

-- Define the polynomial
def polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 (x : ℚ) : ℚ :=
  3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

-- The main theorem stating that 1 is the only rational root
theorem only_rational_root_is_one : 
  ∀ x : ℚ, polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 x = 0 ↔ x = 1 :=
by
  sorry

end only_rational_root_is_one_l735_73586


namespace pauline_convertibles_l735_73588

theorem pauline_convertibles : 
  ∀ (total_cars regular_percentage truck_percentage sedan_percentage sports_percentage suv_percentage : ℕ),
  total_cars = 125 →
  regular_percentage = 38 →
  truck_percentage = 12 →
  sedan_percentage = 17 →
  sports_percentage = 22 →
  suv_percentage = 6 →
  (total_cars - (regular_percentage * total_cars / 100 + truck_percentage * total_cars / 100 + sedan_percentage * total_cars / 100 + sports_percentage * total_cars / 100 + suv_percentage * total_cars / 100)) = 8 :=
by
  intros
  sorry

end pauline_convertibles_l735_73588


namespace area_of_triangle_BFE_l735_73505

theorem area_of_triangle_BFE (A B C D E F : ℝ × ℝ) (u v : ℝ) 
  (h_rectangle : (0, 0) = A ∧ (3 * u, 0) = B ∧ (3 * u, 3 * v) = C ∧ (0, 3 * v) = D)
  (h_E : E = (0, 2 * v))
  (h_F : F = (2 * u, 0))
  (h_area_rectangle : 3 * u * 3 * v = 48) :
  ∃ (area : ℝ), area = |3 * u * 2 * v| / 2 ∧ area = 24 :=
by 
  sorry

end area_of_triangle_BFE_l735_73505


namespace distance_between_trees_l735_73528

theorem distance_between_trees (n : ℕ) (L : ℝ) (d : ℝ) (h1 : n = 26) (h2 : L = 700) (h3 : d = L / (n - 1)) : d = 28 :=
sorry

end distance_between_trees_l735_73528


namespace graph_passes_through_quadrants_l735_73543

theorem graph_passes_through_quadrants :
  (∃ x, x > 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant I condition: x > 0, y > 0
  (∃ x, x < 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant II condition: x < 0, y > 0
  (∃ x, x > 0 ∧ -1/2 * x + 2 < 0) := -- Quadrant IV condition: x > 0, y < 0
by
  sorry

end graph_passes_through_quadrants_l735_73543


namespace C_share_l735_73557

-- Conditions in Lean definition
def ratio_A_C (A C : ℕ) : Prop := 3 * C = 2 * A
def ratio_A_B (A B : ℕ) : Prop := 3 * B = A
def total_profit : ℕ := 60000

-- Lean statement
theorem C_share (A B C : ℕ) (h1 : ratio_A_C A C) (h2 : ratio_A_B A B) : (C * total_profit) / (A + B + C) = 20000 :=
  by
  sorry

end C_share_l735_73557


namespace carousel_ticket_cost_l735_73559

theorem carousel_ticket_cost :
  ∃ (x : ℕ), 
  (2 * 5) + (3 * x) = 19 ∧ x = 3 :=
by
  sorry

end carousel_ticket_cost_l735_73559


namespace seedling_probability_l735_73552

theorem seedling_probability (germination_rate survival_rate : ℝ)
    (h_germ : germination_rate = 0.9) (h_survival : survival_rate = 0.8) : 
    germination_rate * survival_rate = 0.72 :=
by
  rw [h_germ, h_survival]
  norm_num

end seedling_probability_l735_73552


namespace cubic_roots_reciprocal_sum_l735_73581

theorem cubic_roots_reciprocal_sum {α β γ : ℝ} 
  (h₁ : α + β + γ = 6)
  (h₂ : α * β + β * γ + γ * α = 11)
  (h₃ : α * β * γ = 6) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 49 / 36 := 
by 
  sorry

end cubic_roots_reciprocal_sum_l735_73581


namespace gcd_of_fraction_in_lowest_terms_l735_73509

theorem gcd_of_fraction_in_lowest_terms (n : ℤ) (h : n % 2 = 1) : Int.gcd (2 * n + 2) (3 * n + 2) = 1 := 
by 
  sorry

end gcd_of_fraction_in_lowest_terms_l735_73509


namespace article_usage_correct_l735_73506

def blank1 := "a"
def blank2 := ""  -- Representing "不填" (no article) as an empty string for simplicity

theorem article_usage_correct :
  (blank1 = "a" ∧ blank2 = "") :=
by
  sorry

end article_usage_correct_l735_73506


namespace base_n_system_digits_l735_73571

theorem base_n_system_digits (N : ℕ) (h : N ≥ 6) :
  ((N - 1) ^ 4).digits N = [N-4, 5, N-4, 1] :=
by
  sorry

end base_n_system_digits_l735_73571
