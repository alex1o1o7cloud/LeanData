import Mathlib

namespace percentage_of_360_l978_97839

theorem percentage_of_360 (percentage : ℝ) : 
  (percentage / 100) * 360 = 93.6 → percentage = 26 := 
by
  intro h
  -- proof missing
  sorry

end percentage_of_360_l978_97839


namespace value_of_abs_div_sum_l978_97819

theorem value_of_abs_div_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (|a| / a + |b| / b = 2) ∨ (|a| / a + |b| / b = -2) ∨ (|a| / a + |b| / b = 0) := 
by
  sorry

end value_of_abs_div_sum_l978_97819


namespace cost_of_flowers_l978_97840

theorem cost_of_flowers 
  (interval : ℕ) (perimeter : ℕ) (cost_per_flower : ℕ)
  (h_interval : interval = 30)
  (h_perimeter : perimeter = 1500)
  (h_cost : cost_per_flower = 5000) :
  (perimeter / interval) * cost_per_flower = 250000 :=
by
  sorry

end cost_of_flowers_l978_97840


namespace part1_exists_n_part2_not_exists_n_l978_97801

open Nat

def is_prime (p : Nat) : Prop := p > 1 ∧ ∀ m : Nat, m ∣ p → m = 1 ∨ m = p

-- Part 1: Prove there exists an n such that n-96, n, n+96 are all primes
theorem part1_exists_n :
  ∃ (n : Nat), is_prime (n - 96) ∧ is_prime n ∧ is_prime (n + 96) :=
sorry

-- Part 2: Prove there does not exist an n such that n-1996, n, n+1996 are all primes
theorem part2_not_exists_n :
  ¬ (∃ (n : Nat), is_prime (n - 1996) ∧ is_prime n ∧ is_prime (n + 1996)) :=
sorry

end part1_exists_n_part2_not_exists_n_l978_97801


namespace find_f_and_q_l978_97818

theorem find_f_and_q (m : ℤ) (q : ℝ) :
  (∀ x > 0, (x : ℝ)^(-m^2 + 2*m + 3) = (x : ℝ)^4) ∧
  (∀ x ∈ [-1, 1], 2 * (x^2) - 8 * x + q - 1 > 0) →
  q > 7 :=
by
  sorry

end find_f_and_q_l978_97818


namespace find_x_l978_97846

theorem find_x (x : ℝ) (h : 0.40 * x = (1/3) * x + 110) : x = 1650 :=
sorry

end find_x_l978_97846


namespace hyperbola_center_l978_97873

theorem hyperbola_center (x y : ℝ) :
  (∃ h k, h = 2 ∧ k = -1 ∧ 
    (∀ x y, (3 * y + 3)^2 / 7^2 - (4 * x - 8)^2 / 6^2 = 1 ↔ 
      (y - (-1))^2 / ((7 / 3)^2) - (x - 2)^2 / ((3 / 2)^2) = 1)) :=
by sorry

end hyperbola_center_l978_97873


namespace weight_of_second_new_player_l978_97858

theorem weight_of_second_new_player 
  (total_weight_seven_players : ℕ)
  (average_weight_seven_players : ℕ)
  (total_players_with_new_players : ℕ)
  (average_weight_with_new_players : ℕ)
  (weight_first_new_player : ℕ)
  (W : ℕ) :
  total_weight_seven_players = 7 * average_weight_seven_players →
  total_players_with_new_players = 9 →
  average_weight_with_new_players = 106 →
  weight_first_new_player = 110 →
  (total_weight_seven_players + weight_first_new_player + W) / total_players_with_new_players = average_weight_with_new_players →
  W = 60 := 
by sorry

end weight_of_second_new_player_l978_97858


namespace intersection_points_of_graph_and_line_l978_97853

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end intersection_points_of_graph_and_line_l978_97853


namespace prove_scientific_notation_l978_97804

def scientific_notation_correct : Prop :=
  340000 = 3.4 * (10 ^ 5)

theorem prove_scientific_notation : scientific_notation_correct :=
  by
    sorry

end prove_scientific_notation_l978_97804


namespace odd_function_sum_zero_l978_97866

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

theorem odd_function_sum_zero (g : ℝ → ℝ) (a : ℝ) (h_odd : odd_function g) : 
  g a + g (-a) = 0 :=
by 
  sorry

end odd_function_sum_zero_l978_97866


namespace sum_of_squares_neq_fourth_powers_l978_97833

theorem sum_of_squares_neq_fourth_powers (m n : ℕ) : 
  m^2 + (m + 1)^2 ≠ n^4 + (n + 1)^4 :=
by 
  sorry

end sum_of_squares_neq_fourth_powers_l978_97833


namespace arithmetic_sequence_sum_l978_97871

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l978_97871


namespace wendy_third_day_miles_l978_97824

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end wendy_third_day_miles_l978_97824


namespace abs_diff_26th_term_l978_97815

def C (n : ℕ) : ℤ := 50 + 15 * (n - 1)
def D (n : ℕ) : ℤ := 85 - 20 * (n - 1)

theorem abs_diff_26th_term :
  |(C 26) - (D 26)| = 840 := by
  sorry

end abs_diff_26th_term_l978_97815


namespace value_of_k_l978_97886

theorem value_of_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2024)
: k = 2023 :=
by
  sorry

end value_of_k_l978_97886


namespace xyz_equivalence_l978_97898

theorem xyz_equivalence (x y z a b : ℝ) (h₁ : 4^x = a) (h₂: 2^y = b) (h₃ : 8^z = a * b) : 3 * z = 2 * x + y :=
by
  -- Here, we leave the proof as an exercise
  sorry

end xyz_equivalence_l978_97898


namespace number_of_decks_bought_l978_97894

theorem number_of_decks_bought :
  ∃ T : ℕ, (8 * T + 5 * 8 = 64) ∧ T = 3 :=
by
  sorry

end number_of_decks_bought_l978_97894


namespace sum_abc_eq_ten_l978_97857

theorem sum_abc_eq_ten (a b c : ℝ) (h : (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0) : a + b + c = 10 :=
by
  sorry

end sum_abc_eq_ten_l978_97857


namespace remainder_m_n_mod_1000_l978_97887

noncomputable def m : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2009 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

noncomputable def n : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2000 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

theorem remainder_m_n_mod_1000 : (m - n) % 1000 = 0 :=
by
  sorry

end remainder_m_n_mod_1000_l978_97887


namespace directrix_of_parabola_l978_97835

open Real

noncomputable def parabola_directrix (a : ℝ) : ℝ := -a / 4

theorem directrix_of_parabola (a : ℝ) (h : a = 4) : parabola_directrix a = -4 :=
by
  sorry

end directrix_of_parabola_l978_97835


namespace distance_to_grandma_l978_97860

-- Definitions based on the conditions
def miles_per_gallon : ℕ := 20
def gallons_needed : ℕ := 5

-- The theorem statement to prove the distance is 100 miles
theorem distance_to_grandma : miles_per_gallon * gallons_needed = 100 := by
  sorry

end distance_to_grandma_l978_97860


namespace product_of_solutions_l978_97817

theorem product_of_solutions : 
  (∃ x1 x2 : ℝ, |5 * x1 - 1| + 4 = 54 ∧ |5 * x2 - 1| + 4 = 54 ∧ x1 * x2 = -99.96) :=
  by sorry

end product_of_solutions_l978_97817


namespace min_value_of_n_l978_97814

def is_prime (p : ℕ) : Prop := p ≥ 2 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_not_prime (n : ℕ) : Prop := ¬ is_prime n

def decomposable_into_primes_leq_10 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≤ 10 ∧ q ≤ 10 ∧ n = p * q

theorem min_value_of_n : ∃ n : ℕ, is_not_prime n ∧ decomposable_into_primes_leq_10 n ∧ n = 6 :=
by
  -- The proof would go here.
  sorry

end min_value_of_n_l978_97814


namespace percent_of_part_is_20_l978_97841

theorem percent_of_part_is_20 {Part Whole : ℝ} (hPart : Part = 14) (hWhole : Whole = 70) : (Part / Whole) * 100 = 20 :=
by
  rw [hPart, hWhole]
  have h : (14 : ℝ) / 70 = 0.2 := by norm_num
  rw [h]
  norm_num

end percent_of_part_is_20_l978_97841


namespace cats_joined_l978_97861

theorem cats_joined (c : ℕ) (h : 1 + c + 2 * c + 6 * c = 37) : c = 4 :=
sorry

end cats_joined_l978_97861


namespace simplify_fraction_l978_97843

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) : (x + 1) / (x^2 + 2 * x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l978_97843


namespace correct_operation_l978_97823

-- Define the operations given in the conditions
def optionA (m : ℝ) := m^2 + m^2 = 2 * m^4
def optionB (a : ℝ) := a^2 * a^3 = a^5
def optionC (m n : ℝ) := (m * n^2) ^ 3 = m * n^6
def optionD (m : ℝ) := m^6 / m^2 = m^3

-- Theorem stating that option B is the correct operation
theorem correct_operation (a m n : ℝ) : optionB a :=
by sorry

end correct_operation_l978_97823


namespace new_supervisor_salary_correct_l978_97812

noncomputable def salary_new_supervisor
  (avg_salary_old : ℝ)
  (old_supervisor_salary : ℝ)
  (avg_salary_new : ℝ)
  (workers_count : ℝ)
  (total_salary_workers : ℝ := (avg_salary_old * (workers_count + 1)) - old_supervisor_salary)
  (new_supervisor_salary : ℝ := (avg_salary_new * (workers_count + 1)) - total_salary_workers)
  : ℝ :=
  new_supervisor_salary

theorem new_supervisor_salary_correct :
  salary_new_supervisor 430 870 420 8 = 780 :=
by
  simp [salary_new_supervisor]
  sorry

end new_supervisor_salary_correct_l978_97812


namespace number_of_poly_lines_l978_97889

def nonSelfIntersectingPolyLines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n - 3)
  else 0

theorem number_of_poly_lines (n : ℕ) (h : n > 1) :
  nonSelfIntersectingPolyLines n =
  if n = 2 then 1 else n * 2^(n - 3) :=
by sorry

end number_of_poly_lines_l978_97889


namespace repeating_decimal_division_l978_97816

theorem repeating_decimal_division:
  let x := (54 / 99 : ℚ)
  let y := (18 / 99 : ℚ)
  (x / y) * (1 / 2) = (3 / 2 : ℚ) := by
    sorry

end repeating_decimal_division_l978_97816


namespace koala_fiber_absorption_l978_97865

theorem koala_fiber_absorption (x : ℝ) (hx : 0.30 * x = 12) : x = 40 :=
by
  sorry

end koala_fiber_absorption_l978_97865


namespace calculate_fg_l978_97822

def f (x : ℝ) : ℝ := x - 4

def g (x : ℝ) : ℝ := x^2 + 5

theorem calculate_fg : f (g (-3)) = 10 := by
  sorry

end calculate_fg_l978_97822


namespace prob_first_red_light_third_intersection_l978_97850

noncomputable def red_light_at_third_intersection (p : ℝ) (h : p = 2/3) : ℝ :=
(1 - p) * (1 - (1/2)) * (1/2)

theorem prob_first_red_light_third_intersection (h : 2/3 = (2/3 : ℝ)) :
  red_light_at_third_intersection (2/3) h = 1/12 := sorry

end prob_first_red_light_third_intersection_l978_97850


namespace discount_offered_is_5_percent_l978_97877

noncomputable def cost_price : ℝ := 100

noncomputable def selling_price_with_discount : ℝ := cost_price * 1.216

noncomputable def selling_price_without_discount : ℝ := cost_price * 1.28

noncomputable def discount : ℝ := selling_price_without_discount - selling_price_with_discount

noncomputable def discount_percentage : ℝ := (discount / selling_price_without_discount) * 100

theorem discount_offered_is_5_percent : discount_percentage = 5 :=
by 
  sorry

end discount_offered_is_5_percent_l978_97877


namespace avg_price_two_returned_theorem_l978_97837

-- Defining the initial conditions given in the problem
def avg_price_of_five (price: ℕ) (packets: ℕ) : Prop :=
  packets = 5 ∧ price = 20

def avg_price_of_three_remaining (price: ℕ) (packets: ℕ) : Prop :=
  packets = 3 ∧ price = 12
  
def cost_of_packets (price packets: ℕ) := price * packets

noncomputable def avg_price_two_returned (total_initial_cost total_remaining_cost: ℕ) :=
  (total_initial_cost - total_remaining_cost) / 2

-- The Lean 4 proof statement
theorem avg_price_two_returned_theorem (p1 p2 p3 p4: ℕ):
  avg_price_of_five p1 5 →
  avg_price_of_three_remaining p2 3 →
  cost_of_packets p1 5 = 100 →
  cost_of_packets p2 3 = 36 →
  avg_price_two_returned 100 36 = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end avg_price_two_returned_theorem_l978_97837


namespace choir_membership_l978_97882

theorem choir_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 8 = 3) (h3 : n ≥ 100) (h4 : n ≤ 200) :
  n = 123 ∨ n = 179 :=
by
  sorry

end choir_membership_l978_97882


namespace calculate_m_l978_97884

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end calculate_m_l978_97884


namespace valid_combinations_l978_97899

def herbs : Nat := 4
def crystals : Nat := 6
def incompatible_pairs : Nat := 3

theorem valid_combinations : 
  (herbs * crystals) - incompatible_pairs = 21 := by
  sorry

end valid_combinations_l978_97899


namespace ShepherdProblem_l978_97825

theorem ShepherdProblem (x y : ℕ) :
  (x + 9 = 2 * (y - 9) ∧ y + 9 = x - 9) ↔
  ((x + 9 = 2 * (y - 9)) ∧ (y + 9 = x - 9)) :=
by
  sorry

end ShepherdProblem_l978_97825


namespace min_value_x1_squared_plus_x2_squared_plus_x3_squared_l978_97802

theorem min_value_x1_squared_plus_x2_squared_plus_x3_squared
    (x1 x2 x3 : ℝ) 
    (h1 : 3 * x1 + 2 * x2 + x3 = 30) 
    (h2 : x1 > 0) 
    (h3 : x2 > 0) 
    (h4 : x3 > 0) : 
    x1^2 + x2^2 + x3^2 ≥ 125 := 
  by sorry

end min_value_x1_squared_plus_x2_squared_plus_x3_squared_l978_97802


namespace arithmetic_geometric_sequence_fraction_l978_97806

theorem arithmetic_geometric_sequence_fraction 
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 + a2 = 10)
  (h2 : 1 * b3 = 9)
  (h3 : b2 ^ 2 = 9) : 
  b2 / (a1 + a2) = 3 / 10 := 
by 
  sorry

end arithmetic_geometric_sequence_fraction_l978_97806


namespace certain_time_in_seconds_l978_97845

theorem certain_time_in_seconds
  (ratio : ℕ) (minutes : ℕ) (time_in_minutes : ℕ) (seconds_in_a_minute : ℕ)
  (h_ratio : ratio = 8)
  (h_minutes : minutes = 4)
  (h_time : time_in_minutes = minutes)
  (h_conversion : seconds_in_a_minute = 60) :
  time_in_minutes * seconds_in_a_minute = 240 :=
by
  sorry

end certain_time_in_seconds_l978_97845


namespace al_sandwich_combinations_l978_97896

def types_of_bread : ℕ := 5
def types_of_meat : ℕ := 6
def types_of_cheese : ℕ := 5

def restricted_turkey_swiss_combinations : ℕ := 5
def restricted_white_chicken_combinations : ℕ := 5
def restricted_rye_turkey_combinations : ℕ := 5

def total_sandwich_combinations : ℕ := types_of_bread * types_of_meat * types_of_cheese

def valid_sandwich_combinations : ℕ :=
  total_sandwich_combinations - restricted_turkey_swiss_combinations
  - restricted_white_chicken_combinations - restricted_rye_turkey_combinations

theorem al_sandwich_combinations : valid_sandwich_combinations = 135 := 
  by
  sorry

end al_sandwich_combinations_l978_97896


namespace sequence_difference_l978_97874

theorem sequence_difference :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧ a 2 = 1 ∧
    (∀ n ≥ 1, (a (n + 2) : ℚ) / a (n + 1) - (a (n + 1) : ℚ) / a n = 1) ∧
    a 6 - a 5 = 96 :=
sorry

end sequence_difference_l978_97874


namespace max_sum_of_lengths_l978_97862

theorem max_sum_of_lengths (x y : ℕ) (hx : 1 < x) (hy : 1 < y) (hxy : x + 3 * y < 5000) :
  ∃ a b : ℕ, x = 2^a ∧ y = 2^b ∧ a + b = 20 := sorry

end max_sum_of_lengths_l978_97862


namespace trajectory_of_P_distance_EF_l978_97893

section Exercise

-- Define the curve C in polar coordinates
def curve_C (ρ' θ: ℝ) : Prop :=
  ρ' * Real.cos (θ + Real.pi / 4) = 1

-- Define the relationship between OP and OQ
def product_OP_OQ (ρ ρ' : ℝ) : Prop :=
  ρ * ρ' = Real.sqrt 2

-- Define the trajectory of point P (C1) as the goal
theorem trajectory_of_P (ρ θ: ℝ) (hC: curve_C ρ' θ) (hPQ: product_OP_OQ ρ ρ') :
  ρ = Real.cos θ - Real.sin θ :=
sorry

-- Define the coordinates and the curve C2
def curve_C2 (x y t: ℝ) : Prop :=
  x = 0.5 - Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

-- Define the line l in Cartesian coordinates that needs to be converted to polar
def line_l (x y: ℝ) : Prop :=
  y = -Real.sqrt 3 * x

-- Define the distance |EF| to be proved
theorem distance_EF (θ ρ_1 ρ_2: ℝ) (hx: curve_C2 (0.5 - Real.sqrt 2 / 2 * t) (Real.sqrt 2 / 2 * t) t)
  (hE: θ = 2 * Real.pi / 3 ∨ θ = -Real.pi / 3)
  (hρ1: ρ_1 = Real.cos (-Real.pi / 3) - Real.sin (-Real.pi / 3))
  (hρ2: ρ_2 = 0.5 * (Real.sqrt 3 + 1)) :
  |ρ_1 + ρ_2| = Real.sqrt 3 + 1 :=
sorry

end Exercise

end trajectory_of_P_distance_EF_l978_97893


namespace polygon_sides_l978_97883

theorem polygon_sides (n : ℕ) (h : (n-3) * 180 < 2008 ∧ 2008 < (n-1) * 180) : 
  n = 14 :=
sorry

end polygon_sides_l978_97883


namespace arithmetic_sequence_first_term_and_difference_l978_97805

theorem arithmetic_sequence_first_term_and_difference
  (a1 d : ℤ)
  (h1 : (a1 + 2 * d) * (a1 + 5 * d) = 406)
  (h2 : a1 + 8 * d = 2 * (a1 + 3 * d) + 6) : 
  a1 = 4 ∧ d = 5 :=
by 
  sorry

end arithmetic_sequence_first_term_and_difference_l978_97805


namespace domain_of_log_function_l978_97813

-- Define the problematic quadratic function
def quadratic_fn (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Define the domain condition for our function
def domain_condition (x : ℝ) : Prop := quadratic_fn x > 0

-- The actual statement to prove, stating that the domain is (1, 3)
theorem domain_of_log_function :
  {x : ℝ | domain_condition x} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end domain_of_log_function_l978_97813


namespace total_pens_l978_97854

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l978_97854


namespace lock_code_difference_l978_97897

theorem lock_code_difference :
  ∃ A B C D, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
             (A = 4 ∧ B = 2 * C ∧ C = D) ∨
             (A = 9 ∧ B = 3 * C ∧ C = D) ∧
             (A * 100 + B * 10 + C - (D * 100 + (2 * D) * 10 + D)) = 541 :=
sorry

end lock_code_difference_l978_97897


namespace max_principals_in_10_years_l978_97851

theorem max_principals_in_10_years (term_length : ℕ) (period_length : ℕ) (max_principals : ℕ)
  (term_length_eq : term_length = 4) (period_length_eq : period_length = 10) :
  max_principals = 4 :=
by
  sorry

end max_principals_in_10_years_l978_97851


namespace sets_are_equal_l978_97878

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem sets_are_equal : X = Y :=
by sorry

end sets_are_equal_l978_97878


namespace jenny_profit_l978_97848

-- Definitions for the conditions
def cost_per_pan : ℝ := 10.0
def pans_sold : ℕ := 20
def selling_price_per_pan : ℝ := 25.0

-- Definition for the profit calculation based on the given conditions
def total_revenue : ℝ := pans_sold * selling_price_per_pan
def total_cost : ℝ := pans_sold * cost_per_pan
def profit : ℝ := total_revenue - total_cost

-- The actual theorem statement
theorem jenny_profit : profit = 300.0 := by
  sorry

end jenny_profit_l978_97848


namespace sharks_win_percentage_at_least_ninety_percent_l978_97881

theorem sharks_win_percentage_at_least_ninety_percent (N : ℕ) :
  let initial_games := 3
  let initial_shark_wins := 2
  let total_games := initial_games + N
  let total_shark_wins := initial_shark_wins + N
  total_shark_wins * 10 ≥ total_games * 9 ↔ N ≥ 7 :=
by
  intros
  sorry

end sharks_win_percentage_at_least_ninety_percent_l978_97881


namespace smallest_integer_cubing_y_eq_350_l978_97868

def y : ℕ := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

theorem smallest_integer_cubing_y_eq_350 : ∃ z : ℕ, z * y = (2^23) * (3^9) * (5^6) * (7^6) → z = 350 :=
by
  sorry

end smallest_integer_cubing_y_eq_350_l978_97868


namespace triangle_circumscribed_circle_diameter_l978_97849

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem triangle_circumscribed_circle_diameter :
  let a := 16
  let A := Real.pi / 4   -- 45 degrees in radians
  circumscribed_circle_diameter a A = 16 * Real.sqrt 2 :=
by
  sorry

end triangle_circumscribed_circle_diameter_l978_97849


namespace suitable_for_experimental_method_is_meters_run_l978_97875

-- Define the options as a type
inductive ExperimentalOption
| recommending_class_monitor_candidates
| surveying_classmates_birthdays
| meters_run_in_10_seconds
| avian_influenza_occurrences_world

-- Define a function that checks if an option is suitable for the experimental method
def is_suitable_for_experimental_method (option: ExperimentalOption) : Prop :=
  option = ExperimentalOption.meters_run_in_10_seconds

-- The theorem stating which option is suitable for the experimental method
theorem suitable_for_experimental_method_is_meters_run :
  is_suitable_for_experimental_method ExperimentalOption.meters_run_in_10_seconds :=
by
  sorry

end suitable_for_experimental_method_is_meters_run_l978_97875


namespace jenny_questions_wrong_l978_97844

variable (j k l m : ℕ)

theorem jenny_questions_wrong
  (h1 : j + k = l + m)
  (h2 : j + m = k + l + 6)
  (h3 : l = 7) : j = 10 := by
  sorry

end jenny_questions_wrong_l978_97844


namespace people_in_room_eq_33_l978_97847

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end people_in_room_eq_33_l978_97847


namespace estimate_larger_than_difference_l978_97876

theorem estimate_larger_than_difference
  (u v δ γ : ℝ)
  (huv : u > v)
  (hδ : δ > 0)
  (hγ : γ > 0)
  (hδγ : δ > γ) : (u + δ) - (v - γ) > u - v := by
  sorry

end estimate_larger_than_difference_l978_97876


namespace new_trailers_added_l978_97895

theorem new_trailers_added :
  let initial_trailers := 25
  let initial_average_age := 15
  let years_passed := 3
  let current_average_age := 12
  let total_initial_age := initial_trailers * (initial_average_age + years_passed)
  ∀ n : Nat, 
    ((25 * 18) + (n * 3) = (25 + n) * 12) →
    n = 17 := 
by
  intros
  sorry

end new_trailers_added_l978_97895


namespace log_base2_probability_l978_97879

theorem log_base2_probability (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : ∃ k : ℕ, n = 2^k) : 
  ∃ p : ℚ, p = 1/300 :=
  sorry

end log_base2_probability_l978_97879


namespace distinct_after_removal_l978_97885

variable (n : ℕ)
variable (subsets : Fin n → Finset (Fin n))

theorem distinct_after_removal :
  ∃ k : Fin n, ∀ i j : Fin n, i ≠ j → (subsets i \ {k}) ≠ (subsets j \ {k}) := by
  sorry

end distinct_after_removal_l978_97885


namespace fraction_equality_l978_97810

theorem fraction_equality (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := 
by
  -- Use the hypthesis to derive that a = 2k, b = 3k, c = 4k and show the equality.
  sorry

end fraction_equality_l978_97810


namespace find_simple_annual_rate_l978_97821

-- Conditions from part a).
-- 1. Principal initial amount (P) is $5,000.
-- 2. Annual interest rate for compounded interest (r) is 0.06.
-- 3. Number of times it compounds per year (n) is 2 (semi-annually).
-- 4. Time period (t) is 1 year.
-- 5. The interest earned after one year for simple interest is $6 less than compound interest.

noncomputable def principal : ℝ := 5000
noncomputable def annual_rate_compound : ℝ := 0.06
noncomputable def times_compounded : ℕ := 2
noncomputable def time_years : ℝ := 1
noncomputable def compound_interest : ℝ := principal * (1 + annual_rate_compound / times_compounded) ^ (times_compounded * time_years) - principal
noncomputable def simple_interest : ℝ := compound_interest - 6

-- Question from part a) translated to Lean statement using the condition that simple interest satisfaction
theorem find_simple_annual_rate : 
    ∃ r : ℝ, principal * r * time_years = simple_interest :=
by
  exists (0.0597)
  sorry

end find_simple_annual_rate_l978_97821


namespace abcd_hife_value_l978_97890

theorem abcd_hife_value (a b c d e f g h i : ℝ) 
  (h1 : a / b = 1 / 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 1 / 2) 
  (h4 : d / e = 3) 
  (h5 : e / f = 1 / 10) 
  (h6 : f / g = 3 / 4) 
  (h7 : g / h = 1 / 5) 
  (h8 : h / i = 5) : 
  abcd / hife = 17.28 := sorry

end abcd_hife_value_l978_97890


namespace balls_in_box_l978_97864

def num_blue : Nat := 6
def num_red : Nat := 4
def num_green : Nat := 3 * num_blue
def num_yellow : Nat := 2 * num_red
def num_total : Nat := num_blue + num_red + num_green + num_yellow

theorem balls_in_box : num_total = 36 := by
  sorry

end balls_in_box_l978_97864


namespace sufficient_but_not_necessary_l978_97869

variable {a b : ℝ}

theorem sufficient_but_not_necessary (h : b < a ∧ a < 0) : 1 / a < 1 / b :=
by
  sorry

end sufficient_but_not_necessary_l978_97869


namespace common_chord_eq_l978_97807

theorem common_chord_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 2*x + 8*y - 8 = 0) → (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
    x + 2*y - 1 = 0 :=
by
  intros x y h1 h2
  sorry

end common_chord_eq_l978_97807


namespace parallel_lines_l978_97829

theorem parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, a * x + 2 * y - 1 = k * (2 * x + a * y + 2)) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end parallel_lines_l978_97829


namespace items_per_friend_l978_97838

theorem items_per_friend (pencils : ℕ) (erasers : ℕ) (friends : ℕ) 
    (pencils_eq : pencils = 35) 
    (erasers_eq : erasers = 5) 
    (friends_eq : friends = 5) : 
    (pencils + erasers) / friends = 8 := 
by
  sorry

end items_per_friend_l978_97838


namespace domain_of_k_l978_97872

noncomputable def domain_of_h := Set.Icc (-10 : ℝ) 6

def h (x : ℝ) : Prop := x ∈ domain_of_h
def k (x : ℝ) : Prop := h (-3 * x + 1)

theorem domain_of_k : ∀ x : ℝ, k x ↔ x ∈ Set.Icc (-5/3) (11/3) :=
by
  intro x
  change (-3 * x + 1 ∈ Set.Icc (-10 : ℝ) 6) ↔ (x ∈ Set.Icc (-5/3 : ℝ) (11/3))
  sorry

end domain_of_k_l978_97872


namespace integer_solutions_k_l978_97809

theorem integer_solutions_k (k n m : ℤ) (h1 : k + 1 = n^2) (h2 : 16 * k + 1 = m^2) :
  k = 0 ∨ k = 3 :=
by sorry

end integer_solutions_k_l978_97809


namespace adult_ticket_cost_l978_97856

def num_total_tickets : ℕ := 510
def cost_senior_ticket : ℕ := 15
def total_receipts : ℤ := 8748
def num_senior_tickets : ℕ := 327
def num_adult_tickets : ℕ := num_total_tickets - num_senior_tickets
def revenue_senior : ℤ := num_senior_tickets * cost_senior_ticket
def revenue_adult (cost_adult_ticket : ℤ) : ℤ := num_adult_tickets * cost_adult_ticket

theorem adult_ticket_cost : 
  ∃ (cost_adult_ticket : ℤ), 
    revenue_adult cost_adult_ticket + revenue_senior = total_receipts ∧ 
    cost_adult_ticket = 21 :=
by
  sorry

end adult_ticket_cost_l978_97856


namespace find_slope_of_line_l978_97831

theorem find_slope_of_line (k m x0 : ℝ) (P Q : ℝ × ℝ) 
  (hP : P.2^2 = 4 * P.1) 
  (hQ : Q.2^2 = 4 * Q.1) 
  (hMid : (P.1 + Q.1) / 2 = x0 ∧ (P.2 + Q.2) / 2 = 2) 
  (hLineP : P.2 = k * P.1 + m) 
  (hLineQ : Q.2 = k * Q.1 + m) : k = 1 :=
by sorry

end find_slope_of_line_l978_97831


namespace max_sum_of_abcd_l978_97863

noncomputable def abcd_product (a b c d : ℕ) : ℕ := a * b * c * d

theorem max_sum_of_abcd (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : abcd_product a b c d = 1995) : 
    a + b + c + d ≤ 142 :=
sorry

end max_sum_of_abcd_l978_97863


namespace arithmetic_mean_of_two_numbers_l978_97808

def is_arithmetic_mean (x y z : ℚ) : Prop :=
  (x + z) / 2 = y

theorem arithmetic_mean_of_two_numbers :
  is_arithmetic_mean (9 / 12) (5 / 6) (7 / 8) :=
by
  sorry

end arithmetic_mean_of_two_numbers_l978_97808


namespace multiple_of_C_share_l978_97870

noncomputable def find_multiple (A B C : ℕ) (total : ℕ) (mult : ℕ) (h1 : 4 * A = mult * C) (h2 : 5 * B = mult * C) (h3 : A + B + C = total) : ℕ :=
  mult

theorem multiple_of_C_share (A B : ℕ) (h1 : 4 * A = 10 * 160) (h2 : 5 * B = 10 * 160) (h3 : A + B + 160 = 880) : find_multiple A B 160 880 10 h1 h2 h3 = 10 :=
by
  sorry

end multiple_of_C_share_l978_97870


namespace Gerald_charge_per_chore_l978_97852

noncomputable def charge_per_chore (E SE SP C : ℕ) : ℕ :=
  let total_expenditure := E * SE
  let monthly_saving_goal := total_expenditure / SP
  monthly_saving_goal / C

theorem Gerald_charge_per_chore :
  charge_per_chore 100 4 8 5 = 10 :=
by
  sorry

end Gerald_charge_per_chore_l978_97852


namespace factor_x4_plus_64_l978_97803

theorem factor_x4_plus_64 (x : ℝ) : 
  (x^4 + 64) = (x^2 - 4 * x + 8) * (x^2 + 4 * x + 8) :=
sorry

end factor_x4_plus_64_l978_97803


namespace sea_lions_count_l978_97820

theorem sea_lions_count (S P : ℕ) (h1 : 11 * S = 4 * P) (h2 : P = S + 84) : S = 48 := 
by {
  sorry
}

end sea_lions_count_l978_97820


namespace banana_unique_permutations_l978_97826

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l978_97826


namespace arithmetic_expression_value_l978_97855

theorem arithmetic_expression_value : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end arithmetic_expression_value_l978_97855


namespace branches_and_ornaments_l978_97880

def numberOfBranchesAndOrnaments (b t : ℕ) : Prop :=
  (b = t - 1) ∧ (2 * b = t - 1)

theorem branches_and_ornaments : ∃ (b t : ℕ), numberOfBranchesAndOrnaments b t ∧ b = 3 ∧ t = 4 :=
by
  sorry

end branches_and_ornaments_l978_97880


namespace tesseract_hyper_volume_l978_97842

theorem tesseract_hyper_volume
  (a b c d : ℝ)
  (h1 : a * b * c = 72)
  (h2 : b * c * d = 75)
  (h3 : c * d * a = 48)
  (h4 : d * a * b = 50) :
  a * b * c * d = 3600 :=
sorry

end tesseract_hyper_volume_l978_97842


namespace cost_difference_l978_97867

def cost_per_copy_X : ℝ := 1.25
def cost_per_copy_Y : ℝ := 2.75
def num_copies : ℕ := 80

theorem cost_difference :
  num_copies * cost_per_copy_Y - num_copies * cost_per_copy_X = 120 := sorry

end cost_difference_l978_97867


namespace polynomial_min_k_eq_l978_97836

theorem polynomial_min_k_eq {k : ℝ} :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12 >= 0)
  ↔ k = (Real.sqrt 3) / 4 :=
sorry

end polynomial_min_k_eq_l978_97836


namespace revenue_from_full_price_tickets_l978_97828

-- Definitions of the conditions
def total_tickets (f h : ℕ) : Prop := f + h = 180
def total_revenue (f h p : ℕ) : Prop := f * p + h * (p / 2) = 2750

-- Theorem statement
theorem revenue_from_full_price_tickets (f h p : ℕ) 
  (h_total_tickets : total_tickets f h) 
  (h_total_revenue : total_revenue f h p) : 
  f * p = 1000 :=
  sorry

end revenue_from_full_price_tickets_l978_97828


namespace odd_function_iff_l978_97892

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := x * abs (x + a) + b

theorem odd_function_iff (a b : α) : 
  (∀ x : α, f a b (-x) = -f a b x) ↔ (a^2 + b^2 = 0) :=
by
  sorry

end odd_function_iff_l978_97892


namespace compare_y_values_l978_97859

theorem compare_y_values :
  let y₁ := 2 / (-2)
  let y₂ := 2 / (-1)
  y₁ > y₂ := by sorry

end compare_y_values_l978_97859


namespace price_decrease_is_50_percent_l978_97834

-- Original price is 50 yuan
def original_price : ℝ := 50

-- Price after 100% increase
def increased_price : ℝ := original_price * (1 + 1)

-- Required percentage decrease to return to original price
def required_percentage_decrease (x : ℝ) : ℝ := increased_price * (1 - x)

theorem price_decrease_is_50_percent : required_percentage_decrease 0.5 = 50 :=
  by 
    sorry

end price_decrease_is_50_percent_l978_97834


namespace madeline_water_intake_l978_97891

-- Declare necessary data and conditions
def bottle_A : ℕ := 8
def bottle_B : ℕ := 12
def bottle_C : ℕ := 16

def goal_yoga : ℕ := 15
def goal_work : ℕ := 35
def goal_jog : ℕ := 20
def goal_evening : ℕ := 30

def intake_yoga : ℕ := 2 * bottle_A
def intake_work : ℕ := 3 * bottle_B
def intake_jog : ℕ := 2 * bottle_C
def intake_evening : ℕ := 2 * bottle_A + 2 * bottle_C

def total_intake : ℕ := intake_yoga + intake_work + intake_jog + intake_evening
def goal_total : ℕ := 100

-- Statement of the proof problem
theorem madeline_water_intake : total_intake = 132 ∧ total_intake - goal_total = 32 :=
by
  -- Calculation parts go here (not needed per instruction)
  sorry

end madeline_water_intake_l978_97891


namespace log2_square_eq_37_l978_97830

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_square_eq_37
  {x y : ℝ}
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_log : log2 x = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (log2 (x / y))^2 = 37 := by
  sorry

end log2_square_eq_37_l978_97830


namespace no_2007_in_display_can_2008_appear_in_display_l978_97800

-- Definitions of the operations as functions on the display number.
def button1 (n : ℕ) : ℕ := 1
def button2 (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n
def button3 (n : ℕ) : ℕ := if n >= 3 then n - 3 else n
def button4 (n : ℕ) : ℕ := 4 * n

-- Initial condition
def initial_display : ℕ := 0

-- Define can_appear as a recursive function to determine if a number can appear on the display.
def can_appear (target : ℕ) : Prop :=
  ∃ n : ℕ, n = target ∧ (∃ f : (ℕ → ℕ) → ℕ, f initial_display = target)

-- Prove the statements:
theorem no_2007_in_display : ¬ can_appear 2007 :=
  sorry

theorem can_2008_appear_in_display : can_appear 2008 :=
  sorry

end no_2007_in_display_can_2008_appear_in_display_l978_97800


namespace tubs_from_usual_vendor_l978_97888

def total_tubs_needed : Nat := 100
def tubs_in_storage : Nat := 20
def fraction_from_new_vendor : Rat := 1 / 4

theorem tubs_from_usual_vendor :
  let remaining_tubs := total_tubs_needed - tubs_in_storage
  let tubs_from_new_vendor := remaining_tubs * fraction_from_new_vendor
  let tubs_from_usual_vendor := remaining_tubs - tubs_from_new_vendor
  tubs_from_usual_vendor = 60 :=
by
  intro remaining_tubs tubs_from_new_vendor
  exact sorry

end tubs_from_usual_vendor_l978_97888


namespace other_root_l978_97827

theorem other_root (x : ℚ) (h : 48 * x^2 + 29 = 35 * x + 12) : x = 3 / 4 ∨ x = 1 / 3 := 
by {
  -- Proof can be filled in here
  sorry
}

end other_root_l978_97827


namespace solve_inequality_l978_97832

theorem solve_inequality : 
  {x : ℝ | (1 / (x^2 + 1)) > (4 / x) + (21 / 10)} = {x : ℝ | -2 < x ∧ x < 0} :=
by
  sorry

end solve_inequality_l978_97832


namespace lion_turn_angles_l978_97811

-- Define the radius of the circle
def radius (r : ℝ) := r = 10

-- Define the path length the lion runs in meters
def path_length (d : ℝ) := d = 30000

-- Define the final goal: The sum of all the angles of its turns is at least 2998 radians
theorem lion_turn_angles (r d : ℝ) (α : ℝ) (hr : radius r) (hd : path_length d) (hα : d ≤ 10 * α) : α ≥ 2998 := 
sorry

end lion_turn_angles_l978_97811
