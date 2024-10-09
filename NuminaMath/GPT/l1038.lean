import Mathlib

namespace total_oysters_eaten_l1038_103852

/-- Squido eats 200 oysters -/
def Squido_eats := 200

/-- Crabby eats at least twice as many oysters as Squido -/
def Crabby_eats := 2 * Squido_eats

/-- Total oysters eaten by Squido and Crabby -/
theorem total_oysters_eaten : Squido_eats + Crabby_eats = 600 := 
by
  sorry

end total_oysters_eaten_l1038_103852


namespace max_positive_integer_value_of_n_l1038_103863

-- Define the arithmetic sequence with common difference d and first term a₁.
variable {d a₁ : ℝ}

-- The quadratic inequality condition which provides the solution set [0,9].
def inequality_condition (d a₁ : ℝ) : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 9) → d * x^2 + 2 * a₁ * x ≥ 0

-- Maximum integer n such that the sum of the first n terms of the sequence is maximum.
noncomputable def max_n (d a₁ : ℝ) : ℕ :=
  if d < 0 then 5 else 0

-- Statement to be proved.
theorem max_positive_integer_value_of_n (d a₁ : ℝ) 
  (h : inequality_condition d a₁) : max_n d a₁ = 5 :=
sorry

end max_positive_integer_value_of_n_l1038_103863


namespace highest_sum_vertex_l1038_103825

theorem highest_sum_vertex (a b c d e f : ℕ) (h₀ : a + d = 8) (h₁ : b + e = 8) (h₂ : c + f = 8) : 
  a + b + c ≤ 11 ∧ b + c + d ≤ 11 ∧ c + d + e ≤ 11 ∧ d + e + f ≤ 11 ∧ e + f + a ≤ 11 ∧ f + a + b ≤ 11 :=
sorry

end highest_sum_vertex_l1038_103825


namespace final_price_correct_l1038_103805

def original_cost : ℝ := 2.00
def discount : ℝ := 0.57
def final_price : ℝ := 1.43

theorem final_price_correct :
  original_cost - discount = final_price :=
by
  sorry

end final_price_correct_l1038_103805


namespace probability_of_prime_or_odd_is_half_l1038_103871

-- Define the list of sections on the spinner
def sections : List ℕ := [3, 6, 1, 4, 8, 10, 2, 7]

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Bool :=
  if n < 2 then false else List.foldr (λ p b => b && (n % p ≠ 0)) true (List.range (n - 2) |>.map (λ x => x + 2))

-- Define a function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Define the condition of being either prime or odd
def is_prime_or_odd (n : ℕ) : Bool := is_prime n || is_odd n

-- List of favorable outcomes where the number is either prime or odd
def favorable_outcomes : List ℕ := sections.filter is_prime_or_odd

-- Calculate the probability
def probability_prime_or_odd : ℚ := (favorable_outcomes.length : ℚ) / (sections.length : ℚ)

-- Statement to prove the probability is 1/2
theorem probability_of_prime_or_odd_is_half : probability_prime_or_odd = 1 / 2 := by
  sorry

end probability_of_prime_or_odd_is_half_l1038_103871


namespace conic_section_is_parabola_l1038_103877

def isParabola (equation : String) : Prop := 
  equation = "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)"

theorem conic_section_is_parabola : isParabola "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)" :=
  by
  sorry

end conic_section_is_parabola_l1038_103877


namespace m_value_for_power_function_l1038_103880

theorem m_value_for_power_function (m : ℝ) :
  (3 * m - 1 = 1) → (m = 2 / 3) :=
by
  sorry

end m_value_for_power_function_l1038_103880


namespace units_digit_of_5_pow_150_plus_7_l1038_103891

theorem units_digit_of_5_pow_150_plus_7 : (5^150 + 7) % 10 = 2 := by
  sorry

end units_digit_of_5_pow_150_plus_7_l1038_103891


namespace ducks_killed_is_20_l1038_103853

variable (x : ℕ)

def killed_ducks_per_year (x : ℕ) : Prop :=
  let initial_flock := 100
  let annual_births := 30
  let years := 5
  let additional_flock := 150
  let final_flock := 300
  initial_flock + years * (annual_births - x) + additional_flock = final_flock

theorem ducks_killed_is_20 : killed_ducks_per_year 20 :=
by
  sorry

end ducks_killed_is_20_l1038_103853


namespace finite_integer_solutions_l1038_103876

theorem finite_integer_solutions (n : ℕ) : 
  ∃ (S : Finset (ℤ × ℤ)), ∀ (x y : ℤ), (x^3 + y^3 = n) → (x, y) ∈ S := 
sorry

end finite_integer_solutions_l1038_103876


namespace simplify_fractions_l1038_103864

theorem simplify_fractions :
  (240 / 18) * (6 / 135) * (9 / 4) = 4 / 3 :=
by
  sorry

end simplify_fractions_l1038_103864


namespace dilute_lotion_l1038_103868

/-- Determine the number of ounces of water needed to dilute 12 ounces
    of a shaving lotion containing 60% alcohol to a lotion containing 45% alcohol. -/
theorem dilute_lotion (W : ℝ) : 
  ∃ W, 12 * (0.60 : ℝ) / (12 + W) = 0.45 ∧ W = 4 :=
by
  use 4
  sorry

end dilute_lotion_l1038_103868


namespace man_walking_speed_l1038_103824

-- This statement introduces the assumptions and goals of the proof problem.
theorem man_walking_speed
  (x : ℝ)
  (h1 : (25 * (1 / 12)) = (x * (1 / 3)))
  : x = 6.25 :=
sorry

end man_walking_speed_l1038_103824


namespace ratio_of_values_l1038_103888

-- Define the geometric sequence with first term and common ratio
def geom_seq_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n-1)

-- Define the sum of the first n terms of the geometric sequence
def geom_seq_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

-- Sum of the first n terms for given sequence
noncomputable def S_n (n : ℕ) : ℚ :=
  geom_seq_sum (3/2) (-1/2) n

-- Define the function f(t) = t - 1/t
def f (t : ℚ) : ℚ := t - 1 / t

-- Define the maximum and minimum values of f(S_n) and their ratio
noncomputable def ratio_max_min_values : ℚ :=
  let max_val := f (3/2)
  let min_val := f (3/4)
  max_val / min_val

-- The theorem to prove the ratio of the maximum and minimum values
theorem ratio_of_values :
  ratio_max_min_values = -10/7 := by
  sorry

end ratio_of_values_l1038_103888


namespace fibonacci_150_mod_9_l1038_103826

def fibonacci (n : ℕ) : ℕ :=
  if h : n < 2 then n else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 :=
  sorry

end fibonacci_150_mod_9_l1038_103826


namespace ratio_B_to_A_l1038_103859

theorem ratio_B_to_A (A B S : ℕ) 
  (h1 : A = 2 * S)
  (h2 : A = 80)
  (h3 : B - S = 200) :
  B / A = 3 :=
by sorry

end ratio_B_to_A_l1038_103859


namespace domino_perfect_play_winner_l1038_103822

theorem domino_perfect_play_winner :
  ∀ {PlayerI PlayerII : Type} 
    (legal_move : PlayerI → PlayerII → Prop)
    (initial_move : PlayerI → Prop)
    (next_moves : PlayerII → PlayerI → PlayerII → Prop),
    (∀ pI pII, legal_move pI pII) → 
    (∃ m, initial_move m) → 
    (∀ mI mII, next_moves mII mI mII) → 
    ∃ winner, winner = PlayerI :=
by
  sorry

end domino_perfect_play_winner_l1038_103822


namespace mean_of_remaining_two_l1038_103858

theorem mean_of_remaining_two (a b c d e : ℝ) (h : (a + b + c = 3 * 2010)) : 
  (a + b + c + d + e) / 5 = 2010 → (d + e) / 2 = 2011.5 :=
by
  sorry 

end mean_of_remaining_two_l1038_103858


namespace count_valid_three_digit_numbers_l1038_103886

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 36 ∧ 
    (∀ (a b c : ℕ), a ≠ 0 ∧ c ≠ 0 → 
    ((10 * b + c) % 4 = 0 ∧ (10 * b + a) % 4 = 0) → 
    n = 36) :=
sorry

end count_valid_three_digit_numbers_l1038_103886


namespace fruit_basket_count_l1038_103849

theorem fruit_basket_count :
  let apples := 6
  let oranges := 8
  let min_apples := 2
  let min_fruits := 1
  (0 <= oranges ∧ oranges <= 8) ∧ (min_apples <= apples ∧ apples <= 6) ∧ (min_fruits <= (apples + oranges)) →
  (5 * 9 = 45) :=
by
  intro h
  sorry

end fruit_basket_count_l1038_103849


namespace price_on_hot_day_l1038_103814

noncomputable def regular_price_P (P : ℝ) : Prop :=
  7 * 32 * (P - 0.75) + 3 * 32 * (1.25 * P - 0.75) = 450

theorem price_on_hot_day (P : ℝ) (h : regular_price_P P) : 1.25 * P = 2.50 :=
by sorry

end price_on_hot_day_l1038_103814


namespace equivalent_problem_l1038_103850

theorem equivalent_problem :
  let a : ℤ := (-6)
  let b : ℤ := 6
  let c : ℤ := 2
  let d : ℤ := 4
  (a^4 / b^2 - c^5 + d^2 = 20) :=
by
  sorry

end equivalent_problem_l1038_103850


namespace marbles_difference_l1038_103839

def lostMarbles : ℕ := 8
def foundMarbles : ℕ := 10

theorem marbles_difference (lostMarbles foundMarbles : ℕ) : foundMarbles - lostMarbles = 2 := 
by
  sorry

end marbles_difference_l1038_103839


namespace perimeter_of_one_of_the_rectangles_l1038_103817

noncomputable def perimeter_of_rectangle (z w : ℕ) : ℕ :=
  2 * z

theorem perimeter_of_one_of_the_rectangles (z w : ℕ) :
  ∃ P, P = perimeter_of_rectangle z w :=
by
  use 2 * z
  sorry

end perimeter_of_one_of_the_rectangles_l1038_103817


namespace bank_policy_advantageous_for_retirees_l1038_103870

theorem bank_policy_advantageous_for_retirees
  (special_programs : Prop)
  (higher_deposit_rates : Prop)
  (lower_credit_rates : Prop)
  (reliable_loan_payers : Prop)
  (stable_income : Prop)
  (family_interest : Prop)
  (savings_tendency : Prop)
  (regular_income : Prop)
  (long_term_deposits : Prop) :
  reliable_loan_payers ∧ stable_income ∧ family_interest ∧ savings_tendency ∧ regular_income ∧ long_term_deposits → 
  special_programs ∧ higher_deposit_rates ∧ lower_credit_rates :=
sorry

end bank_policy_advantageous_for_retirees_l1038_103870


namespace train_length_l1038_103866

noncomputable def length_of_train (time_in_seconds : ℝ) (speed_in_kmh : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmh * (5 / 18)
  speed_in_mps * time_in_seconds

theorem train_length :
  length_of_train 2.3998080153587713 210 = 140 :=
by
  sorry

end train_length_l1038_103866


namespace cubic_with_root_p_sq_l1038_103843

theorem cubic_with_root_p_sq (p : ℝ) (hp : p^3 + p - 3 = 0) : (p^2 : ℝ) ^ 3 + 2 * (p^2) ^ 2 + p^2 - 9 = 0 :=
sorry

end cubic_with_root_p_sq_l1038_103843


namespace antiderivative_correct_l1038_103873

def f (x : ℝ) : ℝ := 2 * x
def F (x : ℝ) : ℝ := x^2 + 2

theorem antiderivative_correct :
  (∀ x, f x = deriv (F) x) ∧ (F 1 = 3) :=
by
  sorry

end antiderivative_correct_l1038_103873


namespace max_value_b_exists_l1038_103816

theorem max_value_b_exists :
  ∃ a c : ℝ, ∃ b : ℝ, 
  (∀ x : ℤ, 
  ((x^4 - a * x^3 - b * x^2 - c * x - 2007) = 0) → 
  ∃ r s t : ℤ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  ((x = r) ∨ (x = s) ∨ (x = t))) ∧ 
  (∀ b' : ℝ, b' < b → 
  ¬ ( ∃ a' c' : ℝ, ( ∀ x : ℤ, 
  ((x^4 - a' * x^3 - b' * x^2 - c' * x - 2007) = 0) → 
  ∃ r' s' t' : ℤ, r' ≠ s' ∧ s' ≠ t' ∧ r' ≠ t' ∧ 
  ((x = r') ∨ (x = s') ∨ (x = t') )))) ∧ b = 3343 :=
sorry

end max_value_b_exists_l1038_103816


namespace total_toys_is_60_l1038_103820

def toy_cars : Nat := 20
def toy_soldiers : Nat := 2 * toy_cars
def total_toys : Nat := toy_cars + toy_soldiers

theorem total_toys_is_60 : total_toys = 60 := by
  sorry

end total_toys_is_60_l1038_103820


namespace shorter_piece_length_l1038_103818

theorem shorter_piece_length (x : ℝ) :
  (120 - (2 * x + 15) = x) → x = 35 := 
by
  intro h
  sorry

end shorter_piece_length_l1038_103818


namespace binomial_coeff_sum_abs_l1038_103812

theorem binomial_coeff_sum_abs (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ)
  (h : (2 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0):
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end binomial_coeff_sum_abs_l1038_103812


namespace number_of_younger_employees_correct_l1038_103846

noncomputable def total_employees : ℕ := 200
noncomputable def younger_employees : ℕ := 120
noncomputable def sample_size : ℕ := 25

def number_of_younger_employees_to_be_drawn (total younger sample : ℕ) : ℕ :=
  sample * younger / total

theorem number_of_younger_employees_correct :
  number_of_younger_employees_to_be_drawn total_employees younger_employees sample_size = 15 := by
  sorry

end number_of_younger_employees_correct_l1038_103846


namespace boat_speed_in_still_water_equals_6_l1038_103884

def river_flow_rate : ℝ := 2
def distance_upstream : ℝ := 40
def distance_downstream : ℝ := 40
def total_time : ℝ := 15

theorem boat_speed_in_still_water_equals_6 :
  ∃ b : ℝ, (40 / (b - river_flow_rate) + 40 / (b + river_flow_rate) = total_time) ∧ b = 6 :=
sorry

end boat_speed_in_still_water_equals_6_l1038_103884


namespace binomial_problem_l1038_103828

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The problem statement: prove that binomial(13, 11) * 2 = 156
theorem binomial_problem : binomial 13 11 * 2 = 156 := by
  sorry

end binomial_problem_l1038_103828


namespace calculate_area_of_square_field_l1038_103881

def area_of_square_field (t: ℕ) (v: ℕ) (d: ℕ) (s: ℕ) (a: ℕ) : Prop :=
  t = 10 ∧ v = 16 ∧ d = v * t ∧ 4 * s = d ∧ a = s^2

theorem calculate_area_of_square_field (t v d s a : ℕ) 
  (h1: t = 10) (h2: v = 16) (h3: d = v * t) (h4: 4 * s = d) 
  (h5: a = s^2) : a = 1600 := by
  sorry

end calculate_area_of_square_field_l1038_103881


namespace quadratic_has_real_roots_range_l1038_103855

-- Lean 4 statement

theorem quadratic_has_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) → m ≤ 7 ∧ m ≠ 3 :=
by
  sorry

end quadratic_has_real_roots_range_l1038_103855


namespace exact_two_solutions_l1038_103885

theorem exact_two_solutions (a : ℝ) : 
  (∃! x : ℝ, x^2 + 2*x + 2*|x+1| = a) ↔ a > -1 :=
sorry

end exact_two_solutions_l1038_103885


namespace exponent_value_l1038_103882

theorem exponent_value (exponent : ℕ) (y: ℕ) :
  (12 ^ exponent) * (6 ^ 4) / 432 = y → y = 36 → exponent = 1 :=
by
  intro h1 h2
  sorry

end exponent_value_l1038_103882


namespace sqrt_expression_eval_l1038_103889

theorem sqrt_expression_eval :
    (Real.sqrt 8 - 2 * Real.sqrt (1 / 2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3)) = Real.sqrt 2 + 1 := 
by
  sorry

end sqrt_expression_eval_l1038_103889


namespace geometric_sequence_second_term_l1038_103875

theorem geometric_sequence_second_term (a_1 q a_3 a_4 : ℝ) (h3 : a_1 * q^2 = 12) (h4 : a_1 * q^3 = 18) : a_1 * q = 8 :=
by
  sorry

end geometric_sequence_second_term_l1038_103875


namespace ratio_of_height_and_radius_l1038_103806

theorem ratio_of_height_and_radius 
  (h r : ℝ) 
  (V_X V_Y : ℝ)
  (hY rY : ℝ)
  (k : ℝ)
  (h_def : V_X = π * r^2 * h)
  (hY_def : hY = k * h)
  (rY_def : rY = k * r)
  (half_filled_VY : V_Y = 1/2 * π * rY^2 * hY)
  (V_X_value : V_X = 2)
  (V_Y_value : V_Y = 64):
  k = 4 :=
by
  sorry

end ratio_of_height_and_radius_l1038_103806


namespace part1_part2_part3_l1038_103890

open Set

-- Define the sets A and B and the universal set
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def U : Set ℝ := univ  -- Universal set R

theorem part1 : A ∩ B = { x | 3 ≤ x ∧ x < 7 } :=
by { sorry }

theorem part2 : U \ A = { x | x < 3 ∨ x ≥ 7 } :=
by { sorry }

theorem part3 : U \ (A ∪ B) = { x | x ≤ 2 ∨ x ≥ 10 } :=
by { sorry }

end part1_part2_part3_l1038_103890


namespace average_speed_of_bus_l1038_103837

theorem average_speed_of_bus (speed_bicycle : ℝ)
  (start_distance : ℝ) (catch_up_time : ℝ)
  (h1 : speed_bicycle = 15)
  (h2 : start_distance = 195)
  (h3 : catch_up_time = 3) : 
  (start_distance + speed_bicycle * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end average_speed_of_bus_l1038_103837


namespace pool_width_40_l1038_103854

theorem pool_width_40
  (hose_rate : ℕ)
  (pool_length : ℕ)
  (pool_depth : ℕ)
  (pool_capacity_percent : ℚ)
  (drain_time : ℕ)
  (water_drained : ℕ)
  (total_capacity : ℚ)
  (pool_width : ℚ) :
  hose_rate = 60 ∧
  pool_length = 150 ∧
  pool_depth = 10 ∧
  pool_capacity_percent = 0.8 ∧
  drain_time = 800 ∧
  water_drained = hose_rate * drain_time ∧
  total_capacity = water_drained / pool_capacity_percent ∧
  total_capacity = pool_length * pool_width * pool_depth →
  pool_width = 40 :=
by
  sorry

end pool_width_40_l1038_103854


namespace moles_of_NaCl_formed_l1038_103830

theorem moles_of_NaCl_formed (hcl moles : ℕ) (nahco3 moles : ℕ) (reaction : ℕ → ℕ → ℕ) :
  hcl = 3 → nahco3 = 3 → reaction 1 1 = 1 →
  reaction hcl nahco3 = 3 :=
by 
  intros h1 h2 h3
  -- Proof omitted
  sorry

end moles_of_NaCl_formed_l1038_103830


namespace arccos_one_eq_zero_l1038_103835

theorem arccos_one_eq_zero :
  Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l1038_103835


namespace solution_set_of_inequality_l1038_103897

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2 * a * x - 3 * a^2 < 0} = {x : ℝ | 3 * a < x ∧ x < -a} :=
by
  sorry

end solution_set_of_inequality_l1038_103897


namespace arithmetic_sequence_zero_l1038_103860

noncomputable def f (x : ℝ) : ℝ :=
  0.3 ^ x - Real.log x / Real.log 2

theorem arithmetic_sequence_zero (a b c x : ℝ) (h_seq : a < b ∧ b < c) (h_pos_diff : b - a = c - b)
    (h_f_product : f a * f b * f c > 0) (h_fx_zero : f x = 0) : ¬ (x < a) :=
by
  sorry

end arithmetic_sequence_zero_l1038_103860


namespace function_even_periodic_l1038_103851

theorem function_even_periodic (f : ℝ → ℝ) :
  (∀ x : ℝ, f (10 + x) = f (10 - x)) ∧ (∀ x : ℝ, f (5 - x) = f (5 + x)) →
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + 10) = f x) :=
by
  sorry

end function_even_periodic_l1038_103851


namespace greatest_percentage_l1038_103893

theorem greatest_percentage (pA : ℝ) (pB : ℝ) (wA : ℝ) (wB : ℝ) (sA : ℝ) (sB : ℝ) :
  pA = 0.4 → pB = 0.6 → wA = 0.8 → wB = 0.1 → sA = 0.9 → sB = 0.5 →
  pA * min wA sA + pB * min wB sB = 0.38 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Here you would continue with the proof by leveraging the conditions
  sorry

end greatest_percentage_l1038_103893


namespace sin_gt_cos_lt_nec_suff_l1038_103833

-- Define the triangle and the angles
variables {A B C : ℝ}
variables (t : triangle A B C)

-- Define conditions in the triangle: sum of angles is 180 degrees
axiom angle_sum : A + B + C = 180

-- Define sin and cos using the sides of the triangle
noncomputable def sin_A (A : ℝ) : ℝ := sorry -- placeholder for actual definition
noncomputable def sin_B (B : ℝ) : ℝ := sorry
noncomputable def cos_A (A : ℝ) : ℝ := sorry
noncomputable def cos_B (B : ℝ) : ℝ := sorry

-- The proposition to prove
theorem sin_gt_cos_lt_nec_suff {A B : ℝ} (h1 : sin_A A > sin_B B) :
  cos_A A < cos_B B ↔ sin_A A > sin_B B := sorry

end sin_gt_cos_lt_nec_suff_l1038_103833


namespace current_inventory_l1038_103801

noncomputable def initial_books : ℕ := 743
noncomputable def fiction_books : ℕ := 520
noncomputable def nonfiction_books : ℕ := 123
noncomputable def children_books : ℕ := 100

noncomputable def saturday_instore_sales : ℕ := 37
noncomputable def saturday_fiction_sales : ℕ := 15
noncomputable def saturday_nonfiction_sales : ℕ := 12
noncomputable def saturday_children_sales : ℕ := 10
noncomputable def saturday_online_sales : ℕ := 128

noncomputable def sunday_instore_multiplier : ℕ := 2
noncomputable def sunday_online_addition : ℕ := 34

noncomputable def new_shipment : ℕ := 160

noncomputable def current_books := 
  initial_books 
  - (saturday_instore_sales + saturday_online_sales)
  - (sunday_instore_multiplier * saturday_instore_sales + saturday_online_sales + sunday_online_addition)
  + new_shipment

theorem current_inventory : current_books = 502 := by
  sorry

end current_inventory_l1038_103801


namespace find_FC_l1038_103819

theorem find_FC
  (DC : ℝ) (CB : ℝ) (AD : ℝ)
  (hDC : DC = 9) (hCB : CB = 10)
  (hAB : ∃ (k1 : ℝ), k1 = 1/5 ∧ AB = k1 * AD)
  (hED : ∃ (k2 : ℝ), k2 = 3/4 ∧ ED = k2 * AD) :
  ∃ FC : ℝ, FC = 11.025 :=
by
  sorry

end find_FC_l1038_103819


namespace Tamara_height_l1038_103883

-- Define the conditions and goal as a theorem
theorem Tamara_height (K T : ℕ) (h1 : T = 3 * K - 4) (h2 : K + T = 92) : T = 68 :=
by
  sorry

end Tamara_height_l1038_103883


namespace ron_tickets_sold_l1038_103895

theorem ron_tickets_sold 
  (R K : ℕ) 
  (h1 : R + K = 20) 
  (h2 : 2 * R + 9 / 2 * K = 60) : 
  R = 12 := 
by 
  sorry

end ron_tickets_sold_l1038_103895


namespace probability_of_drawing_red_ball_l1038_103810

def totalBalls : Nat := 3 + 5 + 2
def redBalls : Nat := 3
def probabilityOfRedBall : ℚ := redBalls / totalBalls

theorem probability_of_drawing_red_ball :
  probabilityOfRedBall = 3 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l1038_103810


namespace gcd_poly_multiple_l1038_103857

theorem gcd_poly_multiple {x : ℤ} (h : ∃ k : ℤ, x = 54321 * k) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 14)) x = 1 :=
sorry

end gcd_poly_multiple_l1038_103857


namespace garage_sale_items_l1038_103800

-- Definition of conditions
def is_18th_highest (num_highest: ℕ) : Prop := num_highest = 17
def is_25th_lowest (num_lowest: ℕ) : Prop := num_lowest = 24

-- Theorem statement
theorem garage_sale_items (num_highest num_lowest total_items: ℕ) 
  (h1: is_18th_highest num_highest) (h2: is_25th_lowest num_lowest) :
  total_items = num_highest + num_lowest + 1 :=
by
  -- Proof omitted
  sorry

end garage_sale_items_l1038_103800


namespace simplify_expression_l1038_103848

theorem simplify_expression
  (h0 : (Real.pi / 2) < 2 ∧ 2 < Real.pi)  -- Given conditions on 2 related to π.
  (h1 : Real.sin 2 > 0)  -- Given condition that sin 2 is positive.
  (h2 : Real.cos 2 < 0)  -- Given condition that cos 2 is negative.
  : 2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 :=
sorry

end simplify_expression_l1038_103848


namespace min_value_of_g_inequality_f_l1038_103887

def f (x m : ℝ) : ℝ := abs (x - m)
def g (x m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem min_value_of_g (m : ℝ) (hm : m > 0) (h : ∀ x, g x m ≥ -1) : m = 1 :=
sorry

theorem inequality_f {m a b : ℝ} (hm : m > 0) (ha : abs a < m) (hb : abs b < m) (h0 : a ≠ 0) :
  f (a * b) m > abs a * f (b / a) m :=
sorry

end min_value_of_g_inequality_f_l1038_103887


namespace one_cow_one_bag_l1038_103838

theorem one_cow_one_bag (h : 50 * 1 * 50 = 50 * 50) : 50 = 50 :=
by
  sorry

end one_cow_one_bag_l1038_103838


namespace topsoil_cost_l1038_103827

theorem topsoil_cost (cost_per_cubic_foot : ℝ) (cubic_yards : ℝ) (conversion_factor : ℝ) : 
  cubic_yards = 8 →
  cost_per_cubic_foot = 7 →
  conversion_factor = 27 →
  ∃ total_cost : ℝ, total_cost = 1512 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l1038_103827


namespace remaining_surface_area_unchanged_l1038_103872

noncomputable def original_cube_surface_area : Nat := 6 * 4 * 4

def corner_cube_surface_area : Nat := 3 * 2 * 2

def remaining_surface_area (original_cube_surface_area : Nat) (corner_cube_surface_area : Nat) : Nat :=
  original_cube_surface_area

theorem remaining_surface_area_unchanged :
  remaining_surface_area original_cube_surface_area corner_cube_surface_area = 96 := 
by
  sorry

end remaining_surface_area_unchanged_l1038_103872


namespace fern_pays_228_11_usd_l1038_103836

open Real

noncomputable def high_heels_price : ℝ := 66
noncomputable def ballet_slippers_price : ℝ := (2 / 3) * high_heels_price
noncomputable def purse_price : ℝ := 49.5
noncomputable def scarf_price : ℝ := 27.5
noncomputable def high_heels_discount : ℝ := 0.10 * high_heels_price
noncomputable def discounted_high_heels_price : ℝ := high_heels_price - high_heels_discount
noncomputable def total_cost_before_tax : ℝ := discounted_high_heels_price + ballet_slippers_price + purse_price + scarf_price
noncomputable def sales_tax : ℝ := 0.075 * total_cost_before_tax
noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax
noncomputable def exchange_rate : ℝ := 1 / 0.85
noncomputable def total_cost_in_usd : ℝ := total_cost_after_tax * exchange_rate

theorem fern_pays_228_11_usd: total_cost_in_usd = 228.11 := by
  sorry

end fern_pays_228_11_usd_l1038_103836


namespace least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l1038_103829

-- The numbers involved and the requirements described
def num : ℕ := 427398

def least_to_subtract_10 : ℕ := 8
def least_to_subtract_100 : ℕ := 98
def least_to_subtract_1000 : ℕ := 398

-- Proving the conditions:
-- 1. (num - least_to_subtract_10) is divisible by 10
-- 2. (num - least_to_subtract_100) is divisible by 100
-- 3. (num - least_to_subtract_1000) is divisible by 1000

theorem least_subtract_divisible_by_10 : (num - least_to_subtract_10) % 10 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_100 : (num - least_to_subtract_100) % 100 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_1000 : (num - least_to_subtract_1000) % 1000 = 0 := 
by 
  sorry

end least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l1038_103829


namespace min_value_g_l1038_103815

noncomputable def g (x : ℝ) : ℝ := (6 * x^2 + 11 * x + 17) / (7 * (2 + x))

theorem min_value_g : ∃ x, x ≥ 0 ∧ g x = 127 / 24 :=
by
  sorry

end min_value_g_l1038_103815


namespace fanfan_home_distance_l1038_103823

theorem fanfan_home_distance (x y z : ℝ) 
  (h1 : x / 3 = 10) 
  (h2 : x / 3 + y / 2 = 25) 
  (h3 : x / 3 + y / 2 + z = 85) :
  x + y + z = 120 :=
sorry

end fanfan_home_distance_l1038_103823


namespace jonah_fishes_per_day_l1038_103861

theorem jonah_fishes_per_day (J G J_total : ℕ) (days : ℕ) (total : ℕ)
  (hJ : J = 6) (hG : G = 8) (hdays : days = 5) (htotal : total = 90) 
  (fish_total : days * J + days * G + days * J_total = total) : 
  J_total = 4 :=
by
  sorry

end jonah_fishes_per_day_l1038_103861


namespace triangle_count_l1038_103874

theorem triangle_count (a b c : ℕ) (h1 : a + b + c = 15) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a + b > c) :
  ∃ (n : ℕ), n = 7 :=
by
  -- Proceed with the proof steps, using a, b, c satisfying the given conditions
  sorry

end triangle_count_l1038_103874


namespace emily_speed_l1038_103896

theorem emily_speed (distance time : ℝ) (h1 : distance = 10) (h2 : time = 2) : (distance / time) = 5 := 
by sorry

end emily_speed_l1038_103896


namespace math_proof_problem_l1038_103862

-- Definitions
def PropA : Prop := ¬ (∀ n : ℤ, (3 ∣ n → ¬ (n % 2 = 1)))
def PropB : Prop := ¬ (¬ (∃ x : ℝ, x^2 + x + 1 ≥ 0))
def PropC : Prop := ∀ (α β : ℝ) (k : ℤ), α = k * Real.pi + β ↔ Real.tan α = Real.tan β
def PropD : Prop := ∀ (a b : ℝ), a ≠ 0 → a * b ≠ 0 → b ≠ 0

def correct_options : Prop := PropA ∧ PropC ∧ ¬PropB ∧ PropD

-- The theorem to be proven
theorem math_proof_problem : correct_options :=
by
  sorry

end math_proof_problem_l1038_103862


namespace value_of_p_h_3_l1038_103807

-- Define the functions h and p
def h (x : ℝ) : ℝ := 4 * x + 5
def p (x : ℝ) : ℝ := 6 * x - 11

-- Statement to prove
theorem value_of_p_h_3 : p (h 3) = 91 := sorry

end value_of_p_h_3_l1038_103807


namespace mark_new_phone_plan_cost_l1038_103856

noncomputable def total_new_plan_cost (old_plan_cost old_internet_cost old_intl_call_cost : ℝ) (percent_increase_plan percent_increase_internet percent_decrease_intl : ℝ) : ℝ :=
  let new_plan_cost := old_plan_cost * (1 + percent_increase_plan)
  let new_internet_cost := old_internet_cost * (1 + percent_increase_internet)
  let new_intl_call_cost := old_intl_call_cost * (1 - percent_decrease_intl)
  new_plan_cost + new_internet_cost + new_intl_call_cost

theorem mark_new_phone_plan_cost :
  let old_plan_cost := 150
  let old_internet_cost := 50
  let old_intl_call_cost := 30
  let percent_increase_plan := 0.30
  let percent_increase_internet := 0.20
  let percent_decrease_intl := 0.15
  total_new_plan_cost old_plan_cost old_internet_cost old_intl_call_cost percent_increase_plan percent_increase_internet percent_decrease_intl = 280.50 :=
by
  sorry

end mark_new_phone_plan_cost_l1038_103856


namespace constant_term_binomial_expansion_l1038_103892

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (8 - 2 * r = 0) ∧ Nat.choose 8 r = 70 := by
  sorry

end constant_term_binomial_expansion_l1038_103892


namespace f_at_one_is_zero_f_is_increasing_range_of_x_l1038_103869

open Function

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h1 : ∀ x > 1, f x > 0)
variable (h2 : ∀ x y, f (x * y) = f x + f y)

-- Problem Statements
theorem f_at_one_is_zero : f 1 = 0 := 
sorry

theorem f_is_increasing (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (h : x₁ > x₂) : 
  f x₁ > f x₂ := 
sorry

theorem range_of_x (f3_eq_1 : f 3 = 1) (x : ℝ) (h3 : x ≥ 1 + Real.sqrt 10) : 
  f x - f (1 / (x - 2)) ≥ 2 := 
sorry

end f_at_one_is_zero_f_is_increasing_range_of_x_l1038_103869


namespace shaded_L_area_l1038_103821

theorem shaded_L_area 
  (s₁ s₂ s₃ s₄ : ℕ)
  (hA : s₁ = 2)
  (hB : s₂ = 2)
  (hC : s₃ = 3)
  (hD : s₄ = 3)
  (side_ABC : ℕ := 6)
  (area_ABC : ℕ := side_ABC * side_ABC) : 
  area_ABC - (s₁ * s₁ + s₂ * s₂ + s₃ * s₃ + s₄ * s₄) = 10 :=
sorry

end shaded_L_area_l1038_103821


namespace find_C_l1038_103878

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 330) : C = 30 := 
sorry

end find_C_l1038_103878


namespace find_ice_cream_cost_l1038_103809

def cost_of_ice_cream (total_paid cost_chapati cost_rice cost_vegetable : ℕ) (n_chapatis n_rice n_vegetables n_ice_cream : ℕ) : ℕ :=
  (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetables * cost_vegetable)) / n_ice_cream

theorem find_ice_cream_cost :
  let total_paid := 1051
  let cost_chapati := 6
  let cost_rice := 45
  let cost_vegetable := 70
  let n_chapatis := 16
  let n_rice := 5
  let n_vegetables := 7
  let n_ice_cream := 6
  cost_of_ice_cream total_paid cost_chapati cost_rice cost_vegetable n_chapatis n_rice n_vegetables n_ice_cream = 40 :=
by
  sorry

end find_ice_cream_cost_l1038_103809


namespace bowls_remaining_l1038_103813

-- Definitions based on conditions.
def initial_collection : ℕ := 70
def reward_per_10_bowls : ℕ := 2
def total_customers : ℕ := 20
def customers_bought_20 : ℕ := total_customers / 2
def bowls_bought_per_customer : ℕ := 20
def total_bowls_bought : ℕ := customers_bought_20 * bowls_bought_per_customer
def reward_sets : ℕ := total_bowls_bought / 10
def total_reward_given : ℕ := reward_sets * reward_per_10_bowls

-- Theorem statement to be proved.
theorem bowls_remaining : initial_collection - total_reward_given = 30 :=
by
  sorry

end bowls_remaining_l1038_103813


namespace fraction_inequality_l1038_103832

theorem fraction_inequality 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : 1 / a > 1 / b)
  (h2 : x > y) : 
  x / (x + a) > y / (y + b) := 
  sorry

end fraction_inequality_l1038_103832


namespace problem_f_symmetry_problem_f_definition_problem_correct_answer_l1038_103811

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then Real.log x else Real.log (2 - x)

theorem problem_f_symmetry (x : ℝ) : f (2 - x) = f x := 
sorry

theorem problem_f_definition (x : ℝ) (hx : x ≥ 1) : f x = Real.log x :=
sorry

theorem problem_correct_answer: 
  f (1 / 2) < f 2 ∧ f 2 < f (1 / 3) :=
sorry

end problem_f_symmetry_problem_f_definition_problem_correct_answer_l1038_103811


namespace packages_bought_l1038_103803

theorem packages_bought (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) : 
  (total_tshirts / tshirts_per_package) = 71 :=
by 
  sorry

end packages_bought_l1038_103803


namespace proposition_negation_l1038_103840

theorem proposition_negation (p : Prop) : 
  (∃ x : ℝ, x < 1 ∧ x^2 < 1) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
sorry

end proposition_negation_l1038_103840


namespace fraction_evaluation_l1038_103894

theorem fraction_evaluation (x z : ℚ) (hx : x = 4/7) (hz : z = 8/11) :
  (7 * x + 10 * z) / (56 * x * z) = 31 / 176 := by
  sorry

end fraction_evaluation_l1038_103894


namespace relay_race_total_time_l1038_103899

theorem relay_race_total_time :
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  athlete1 + athlete2 + athlete3 + athlete4 = 200 :=
by
  let athlete1 := 55
  let athlete2 := athlete1 + 10
  let athlete3 := athlete2 - 15
  let athlete4 := athlete1 - 25
  show athlete1 + athlete2 + athlete3 + athlete4 = 200
  sorry

end relay_race_total_time_l1038_103899


namespace sqrt_fraction_fact_l1038_103841

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_fraction_fact :
  Real.sqrt (factorial 9 / 210 : ℝ) = 24 * Real.sqrt 3 := by
  sorry

end sqrt_fraction_fact_l1038_103841


namespace pool_capacity_is_80_percent_l1038_103804

noncomputable def current_capacity_percentage (width length depth rate time : ℝ) : ℝ :=
  let total_volume := width * length * depth
  let water_removed := rate * time
  (water_removed / total_volume) * 100

theorem pool_capacity_is_80_percent :
  current_capacity_percentage 50 150 10 60 1000 = 80 :=
by
  sorry

end pool_capacity_is_80_percent_l1038_103804


namespace total_order_cost_l1038_103898

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end total_order_cost_l1038_103898


namespace find_b_perpendicular_l1038_103802

theorem find_b_perpendicular (b : ℝ) : (∀ x y : ℝ, 4 * y - 2 * x = 6 → 5 * y + b * x - 2 = 0 → (1 / 2 : ℝ) * (-(b / 5) : ℝ) = -1) → b = 10 :=
by
  intro h
  sorry

end find_b_perpendicular_l1038_103802


namespace squares_end_with_76_l1038_103847

noncomputable def validNumbers : List ℕ := [24, 26, 74, 76]

theorem squares_end_with_76 (x : ℕ) (h₁ : x % 10 = 4 ∨ x % 10 = 6) 
    (h₂ : (x * x) % 100 = 76) : x ∈ validNumbers := by
  sorry

end squares_end_with_76_l1038_103847


namespace list_price_l1038_103865

theorem list_price (P : ℝ) (h₀ : 0.83817 * P = 56.16) : P = 67 :=
sorry

end list_price_l1038_103865


namespace arithmetic_progression_sum_l1038_103879

variable {α : Type*} [LinearOrderedField α]

def arithmetic_progression (S : ℕ → α) :=
  ∃ (a d : α), ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_progression_sum :
  ∀ (S : ℕ → α),
  arithmetic_progression S →
  (S 4) / (S 8) = 1 / 7 →
  (S 12) / (S 4) = 43 :=
by
  intros S h_arith_prog h_ratio
  sorry

end arithmetic_progression_sum_l1038_103879


namespace teresa_class_size_l1038_103834

theorem teresa_class_size :
  ∃ (a : ℤ), 50 < a ∧ a < 100 ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 2) ∧ 
  (a % 5 = 2) ∧ 
  a = 62 := 
by {
  sorry
}

end teresa_class_size_l1038_103834


namespace product_8_40_product_5_1_6_sum_6_instances_500_l1038_103842

-- The product of 8 and 40 is 320
theorem product_8_40 : 8 * 40 = 320 := sorry

-- 5 times 1/6 is 5/6
theorem product_5_1_6 : 5 * (1 / 6) = 5 / 6 := sorry

-- The sum of 6 instances of 500 ends with 3 zeros and the sum is 3000
theorem sum_6_instances_500 :
  (500 * 6 = 3000) ∧ ((3000 % 1000) = 0) := sorry

end product_8_40_product_5_1_6_sum_6_instances_500_l1038_103842


namespace cookies_on_ninth_plate_l1038_103867

-- Define the geometric sequence
def cookies_on_plate (n : ℕ) : ℕ :=
  2 * 2^(n - 1)

-- State the theorem
theorem cookies_on_ninth_plate : cookies_on_plate 9 = 512 :=
by
  sorry

end cookies_on_ninth_plate_l1038_103867


namespace julia_played_tag_l1038_103831

/-
Problem:
Let m be the number of kids Julia played with on Monday.
Let t be the number of kids Julia played with on Tuesday.
m = 24
m = t + 18
Show that t = 6
-/

theorem julia_played_tag (m t : ℕ) (h1 : m = 24) (h2 : m = t + 18) : t = 6 :=
by
  sorry

end julia_played_tag_l1038_103831


namespace sofie_total_distance_l1038_103845

-- Definitions for the conditions
def side1 : ℝ := 25
def side2 : ℝ := 35
def side3 : ℝ := 20
def side4 : ℝ := 40
def side5 : ℝ := 30
def laps_initial : ℕ := 2
def laps_additional : ℕ := 5
def perimeter : ℝ := side1 + side2 + side3 + side4 + side5

-- Theorem statement
theorem sofie_total_distance : laps_initial * perimeter + laps_additional * perimeter = 1050 := by
  sorry

end sofie_total_distance_l1038_103845


namespace find_n_minus_m_l1038_103844

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 25 - r^2 = 0

-- Given conditions
def circles_intersect (r : ℝ) : Prop :=
(r > 0) ∧ (∃ x y, circle1 x y ∧ circle2 x y r)

-- Prove the range of r for intersection
theorem find_n_minus_m : 
(∀ (r : ℝ), 2 ≤ r ∧ r ≤ 12 ↔ circles_intersect r) → 
12 - 2 = 10 :=
by
  sorry

end find_n_minus_m_l1038_103844


namespace largest_hexagon_angle_l1038_103808

theorem largest_hexagon_angle (x : ℝ) : 
  (2 * x + 2 * x + 2 * x + 3 * x + 4 * x + 5 * x = 720) → (5 * x = 200) := by
  sorry

end largest_hexagon_angle_l1038_103808
