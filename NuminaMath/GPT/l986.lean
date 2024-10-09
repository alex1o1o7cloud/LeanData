import Mathlib

namespace intersection_M_N_l986_98611

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l986_98611


namespace seq_not_square_l986_98664

open Nat

theorem seq_not_square (n : ℕ) (r : ℕ) :
  (r = 11 ∨ r = 111 ∨ r = 1111 ∨ ∃ k : ℕ, r = k * 10^(n + 1) + 1) →
  (r % 4 = 3) →
  (¬ ∃ m : ℕ, r = m^2) :=
by
  intro h_seq h_mod
  intro h_square
  sorry

end seq_not_square_l986_98664


namespace not_perfect_square_4_2021_l986_98685

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * x

-- State the non-perfect square problem for the given choices
theorem not_perfect_square_4_2021 :
  ¬ is_perfect_square (4 ^ 2021) ∧
  is_perfect_square (1 ^ 2018) ∧
  is_perfect_square (6 ^ 2020) ∧
  is_perfect_square (5 ^ 2022) :=
by
  sorry

end not_perfect_square_4_2021_l986_98685


namespace total_workers_in_workshop_l986_98613

-- Definition of average salary calculation
def average_salary (total_salary : ℕ) (workers : ℕ) : ℕ := total_salary / workers

theorem total_workers_in_workshop :
  ∀ (W T R : ℕ),
  T = 5 →
  average_salary ((W - T) * 750) (W - T) = 700 →
  average_salary (T * 900) T = 900 →
  average_salary (W * 750) W = 750 →
  W = T + R →
  W = 20 :=
by
  sorry

end total_workers_in_workshop_l986_98613


namespace correct_computation_l986_98608

theorem correct_computation (x : ℕ) (h : x - 20 = 52) : x / 4 = 18 :=
  sorry

end correct_computation_l986_98608


namespace find_g_l986_98681

open Real

def even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

theorem find_g 
  (f g : ℝ → ℝ) 
  (hf : even f) 
  (hg : odd g)
  (h : ∀ x, f x + g x = exp x) :
  ∀ x, g x = exp x - exp (-x) :=
by
  sorry

end find_g_l986_98681


namespace distance_lines_eq_2_l986_98635

-- Define the first line in standard form
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the second line in standard form, established based on the parallel condition
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y - 14 = 0

-- Define the condition for parallel lines which gives m
axiom parallel_lines_condition : ∀ (x y : ℝ), (line1 x y) → (line2 x y)

-- Define the distance between two parallel lines formula
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / (Real.sqrt (a ^ 2 + b ^ 2))

-- Prove the distance between the given lines is 2
theorem distance_lines_eq_2 : distance_between_parallel_lines 3 4 (-3) 7 = 2 :=
by
  -- Details of proof are omitted, but would show how to manipulate and calculate distances
  sorry

end distance_lines_eq_2_l986_98635


namespace find_p_q_sum_l986_98638

-- Define the conditions
def p (q : ℤ) : ℤ := q + 20

theorem find_p_q_sum (p q : ℤ) (hp : p * q = 1764) (hq : p - q = 20) :
  p + q = 86 :=
  sorry

end find_p_q_sum_l986_98638


namespace cost_price_l986_98621

theorem cost_price (SP : ℝ) (profit_percent : ℝ) (C : ℝ) 
  (h1 : SP = 400) 
  (h2 : profit_percent = 25) 
  (h3 : SP = C + (profit_percent / 100) * C) : 
  C = 320 := 
by
  sorry

end cost_price_l986_98621


namespace acceleration_inverse_square_distance_l986_98615

noncomputable def s (t : ℝ) : ℝ := t^(2/3)

noncomputable def v (t : ℝ) : ℝ := (deriv s t : ℝ)

noncomputable def a (t : ℝ) : ℝ := (deriv v t : ℝ)

theorem acceleration_inverse_square_distance
  (t : ℝ) (h : t ≠ 0) :
  ∃ k : ℝ, k = -2/9 ∧ a t = k / (s t)^2 :=
sorry

end acceleration_inverse_square_distance_l986_98615


namespace colby_mangoes_harvested_60_l986_98670

variable (kg_left kg_each : ℕ)

def totalKgMangoes (x : ℕ) : Prop :=
  ∃ x : ℕ, 
  kg_left = (x - 20) / 2 ∧ 
  kg_each * kg_left = 160 ∧
  kg_each = 8

-- Problem Statement: Prove the total kilograms of mangoes harvested is 60 given the conditions.
theorem colby_mangoes_harvested_60 (x : ℕ) (h1 : x - 20 = 2 * kg_left)
(h2 : kg_each * kg_left = 160) (h3 : kg_each = 8) : x = 60 := by
  sorry

end colby_mangoes_harvested_60_l986_98670


namespace cupcakes_frosted_in_10_minutes_l986_98623

theorem cupcakes_frosted_in_10_minutes :
  let cagney_rate := 1 / 25 -- Cagney's rate in cupcakes per second
  let lacey_rate := 1 / 35 -- Lacey's rate in cupcakes per second
  let total_time := 600 -- Total time in seconds for 10 minutes
  let lacey_break := 60 -- Break duration in seconds
  let lacey_work_time := total_time - lacey_break
  let cupcakes_by_cagney := total_time / 25 
  let cupcakes_by_lacey := lacey_work_time / 35
  cupcakes_by_cagney + cupcakes_by_lacey = 39 := 
by {
  sorry
}

end cupcakes_frosted_in_10_minutes_l986_98623


namespace boat_stream_ratio_l986_98630

theorem boat_stream_ratio (B S : ℝ) (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l986_98630


namespace find_q_l986_98624

theorem find_q (p q : ℝ) (hp : 1 < p) (hq : 1 < q) (hcond1 : 1/p + 1/q = 1) (hcond2 : p * q = 9) :
    q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l986_98624


namespace simplify_cubicroot_1600_l986_98653

theorem simplify_cubicroot_1600 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c^3 * d = 1600) ∧ (c + d = 102) := 
by 
  sorry

end simplify_cubicroot_1600_l986_98653


namespace sum_powers_mod_5_l986_98645

theorem sum_powers_mod_5 (n : ℕ) (h : ¬ (n % 4 = 0)) : 
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 :=
by
  sorry

end sum_powers_mod_5_l986_98645


namespace connected_geometric_seq_a10_l986_98695

noncomputable def is_kth_order_geometric (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + k) = q * a n

theorem connected_geometric_seq_a10 (a : ℕ → ℝ) 
  (h : is_kth_order_geometric a 3) 
  (a1 : a 1 = 1) 
  (a4 : a 4 = 2) : 
  a 10 = 8 :=
sorry

end connected_geometric_seq_a10_l986_98695


namespace ellipse_semi_minor_axis_is_2_sqrt_3_l986_98687

/-- 
  Given an ellipse with the center at (2, -1), 
  one focus at (2, -3), and one endpoint of a semi-major axis at (2, 3), 
  we prove that the semi-minor axis is 2√3.
-/
theorem ellipse_semi_minor_axis_is_2_sqrt_3 :
  let center := (2, -1)
  let focus := (2, -3)
  let endpoint := (2, 3)
  let c := Real.sqrt ((2 - 2)^2 + (-3 + 1)^2)
  let a := Real.sqrt ((2 - 2)^2 + (3 + 1)^2)
  let b2 := a^2 - c^2
  let b := Real.sqrt b2
  c = 2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3 := 
by
  sorry

end ellipse_semi_minor_axis_is_2_sqrt_3_l986_98687


namespace smallest_x_solution_l986_98662

theorem smallest_x_solution :
  ∃ x : ℝ, x * |x| + 3 * x = 5 * x + 2 ∧ (∀ y : ℝ, y * |y| + 3 * y = 5 * y + 2 → x ≤ y)
:=
sorry

end smallest_x_solution_l986_98662


namespace train_pass_time_l986_98682

def speed_jogger := 9   -- in km/hr
def distance_ahead := 240   -- in meters
def length_train := 150   -- in meters
def speed_train := 45   -- in km/hr

noncomputable def time_to_pass_jogger : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := distance_ahead + length_train
  total_distance / relative_speed

theorem train_pass_time : time_to_pass_jogger = 39 :=
  by
    sorry

end train_pass_time_l986_98682


namespace quadratic_equation_correct_l986_98690

theorem quadratic_equation_correct :
    (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 = 5)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x y : ℝ, x + 2 * y = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 + 1/x = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^3 + x^2 = 0)) :=
by
  sorry

end quadratic_equation_correct_l986_98690


namespace days_for_Q_wages_l986_98628

variables (P Q S : ℝ) (D : ℝ)

theorem days_for_Q_wages (h1 : S = 24 * P) (h2 : S = 15 * (P + Q)) : S = D * Q → D = 40 :=
by
  sorry

end days_for_Q_wages_l986_98628


namespace geometric_sequence_sum_l986_98697

noncomputable def aₙ (n : ℕ) : ℝ := (2 / 3) ^ (n - 1)

noncomputable def Sₙ (n : ℕ) : ℝ := 3 * (1 - (2 / 3) ^ n)

theorem geometric_sequence_sum (n : ℕ) : Sₙ n = 3 - 2 * aₙ n := by
  sorry

end geometric_sequence_sum_l986_98697


namespace baseball_cap_problem_l986_98607

theorem baseball_cap_problem 
  (n_first_week n_second_week n_third_week n_fourth_week total_caps : ℕ) 
  (h2 : n_second_week = 400) 
  (h3 : n_third_week = 300) 
  (h4 : n_fourth_week = (n_first_week + n_second_week + n_third_week) / 3) 
  (h_total : n_first_week + n_second_week + n_third_week + n_fourth_week = 1360) : 
  n_first_week = 320 := 
by 
  sorry

end baseball_cap_problem_l986_98607


namespace math_problem_l986_98680

def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 2 * x + 5

theorem math_problem : f (g 4) - g (f 4) = 129 := by
  sorry

end math_problem_l986_98680


namespace solution_set_of_inequality_l986_98659

theorem solution_set_of_inequality 
  (f : ℝ → ℝ)
  (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_at_2 : f 2 = 0)
  (condition : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x > 0) :
  {x : ℝ | x * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_of_inequality_l986_98659


namespace find_value_of_A_l986_98641

theorem find_value_of_A (A B : ℕ) (h_ratio : A * 5 = 3 * B) (h_diff : B - A = 12) : A = 18 :=
by
  sorry

end find_value_of_A_l986_98641


namespace simplify_expression_l986_98663

theorem simplify_expression (q : ℤ) : 
  (((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6)) = 76 * q - 44 := by
  sorry

end simplify_expression_l986_98663


namespace general_term_a_sum_Tn_l986_98648

section sequence_problem

variables {n : ℕ} (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Problem 1: General term formula for {a_n}
axiom Sn_def : ∀ n, S n = 1/4 * (a n + 1)^2
axiom a1_def : a 1 = 1
axiom an_diff : ∀ n, a (n+1) - a n = 2

theorem general_term_a : a n = 2 * n - 1 := sorry

-- Problem 2: Sum of the first n terms of sequence {b_n}
axiom an_formula : ∀ n, a n = 2 * n - 1
axiom bn_def : ∀ n, b n = 1 / (a n * a (n+1))

theorem sum_Tn : T n = n / (2 * n + 1) := sorry

end sequence_problem

end general_term_a_sum_Tn_l986_98648


namespace number_of_rectangular_arrays_l986_98619

theorem number_of_rectangular_arrays (n : ℕ) (h : n = 48) : 
  ∃ k : ℕ, (k = 6 ∧ ∀ m p : ℕ, m * p = n → m ≥ 3 → p ≥ 3 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 8 ∨ m = 12 ∨ m = 16 ∨ m = 24) :=
by
  sorry

end number_of_rectangular_arrays_l986_98619


namespace domain_of_sqrt_cosine_sub_half_l986_98627

theorem domain_of_sqrt_cosine_sub_half :
  {x : ℝ | ∃ k : ℤ, (2 * k * π - π / 3) ≤ x ∧ x ≤ (2 * k * π + π / 3)} =
  {x : ℝ | ∃ k : ℤ, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + π / 3} :=
by sorry

end domain_of_sqrt_cosine_sub_half_l986_98627


namespace smallest_lcm_l986_98683

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l986_98683


namespace problem_statement_l986_98609

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
  (h7 : a^2 + b^2 + c^2 = 16) (h8 : x^2 + y^2 + z^2 = 49) (h9 : a * x + b * y + c * z = 28) : 
  (a + b + c) / (x + y + z) = 4 / 7 := 
by
  sorry

end problem_statement_l986_98609


namespace least_possible_value_of_smallest_integer_l986_98686

theorem least_possible_value_of_smallest_integer {A B C D : ℤ} 
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_mean: (A + B + C + D) / 4 = 68)
  (h_largest: D = 90) :
  A ≥ 5 := 
sorry

end least_possible_value_of_smallest_integer_l986_98686


namespace probability_of_Ace_then_King_l986_98610

def numAces : ℕ := 4
def numKings : ℕ := 4
def totalCards : ℕ := 52

theorem probability_of_Ace_then_King : 
  (numAces / totalCards) * (numKings / (totalCards - 1)) = 4 / 663 :=
by
  sorry

end probability_of_Ace_then_King_l986_98610


namespace sequoia_taller_than_maple_l986_98633

def height_maple_tree : ℚ := 13 + 3/4
def height_sequoia : ℚ := 20 + 1/2

theorem sequoia_taller_than_maple : (height_sequoia - height_maple_tree) = 6 + 3/4 :=
by
  sorry

end sequoia_taller_than_maple_l986_98633


namespace gas_consumption_100_l986_98665

noncomputable def gas_consumption (x : ℝ) : Prop :=
  60 * 1 + (x - 60) * 1.5 = 1.2 * x

theorem gas_consumption_100 (x : ℝ) (h : gas_consumption x) : x = 100 := 
by {
  sorry
}

end gas_consumption_100_l986_98665


namespace olivia_did_not_sell_4_bars_l986_98677

-- Define the constants and conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 7
def money_made : ℕ := 9

-- Calculate the number of bars sold
def bars_sold : ℕ := money_made / price_per_bar

-- Calculate the number of bars not sold
def bars_not_sold : ℕ := total_bars - bars_sold

-- Theorem to prove the answer
theorem olivia_did_not_sell_4_bars : bars_not_sold = 4 := 
by 
  sorry

end olivia_did_not_sell_4_bars_l986_98677


namespace calculate_expression_l986_98657

variables (a b : ℝ)

theorem calculate_expression : -a^2 * 2 * a^4 * b = -2 * (a^6) * b :=
by
  sorry

end calculate_expression_l986_98657


namespace total_resistance_l986_98660

theorem total_resistance (R₀ : ℝ) (h : R₀ = 10) : 
  let R₃ := R₀; let R₄ := R₀; let R₃₄ := R₃ + R₄;
  let R₂ := R₀; let R₅ := R₀; let R₂₃₄ := 1 / (1 / R₂ + 1 / R₃₄ + 1 / R₅);
  let R₁ := R₀; let R₆ := R₀; let R₁₂₃₄ := R₁ + R₂₃₄ + R₆;
  R₁₂₃₄ = 13.33 :=
by 
  sorry

end total_resistance_l986_98660


namespace find_n_l986_98698

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l986_98698


namespace Mitzi_leftover_money_l986_98676

def A := 75
def T := 30
def F := 13
def S := 23
def R := A - (T + F + S)

theorem Mitzi_leftover_money : R = 9 := by
  sorry

end Mitzi_leftover_money_l986_98676


namespace sum_of_integers_is_34_l986_98631

theorem sum_of_integers_is_34 (a b : ℕ) (h1 : a - b = 6) (h2 : a * b = 272) (h3a : a > 0) (h3b : b > 0) : a + b = 34 :=
  sorry

end sum_of_integers_is_34_l986_98631


namespace solve_for_x_l986_98622

theorem solve_for_x (x : ℚ) (h : 5 * (x - 6) = 3 * (3 - 3 * x) + 9) : x = 24 / 7 :=
sorry

end solve_for_x_l986_98622


namespace xiaoning_comprehensive_score_l986_98667

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end xiaoning_comprehensive_score_l986_98667


namespace riya_speed_l986_98602

theorem riya_speed 
  (R : ℝ)
  (priya_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ)
  (h_priya_speed : priya_speed = 22)
  (h_time : time = 1)
  (h_distance : distance = 43)
  : R + priya_speed * time = distance → R = 21 :=
by 
  sorry

end riya_speed_l986_98602


namespace solve_system_of_equations_l986_98617

theorem solve_system_of_equations :
  ∀ (x y : ℝ),
  (3 * x - 2 * y = 7) →
  (2 * x + 3 * y = 8) →
  x = 37 / 13 :=
by
  intros x y h1 h2
  -- to prove x = 37 / 13 from the given system of equations
  sorry

end solve_system_of_equations_l986_98617


namespace parallel_resistance_example_l986_98693

theorem parallel_resistance_example :
  ∀ (R1 R2 : ℕ), R1 = 3 → R2 = 6 → 1 / (R : ℚ) = 1 / (R1 : ℚ) + 1 / (R2 : ℚ) → R = 2 := by
  intros R1 R2 hR1 hR2 h_formula
  -- Formulation of the resistance equations and assumptions
  sorry

end parallel_resistance_example_l986_98693


namespace correct_pairings_l986_98637

-- Define the employees
inductive Employee
| Jia
| Yi
| Bing
deriving DecidableEq

-- Define the wives
inductive Wife
| A
| B
| C
deriving DecidableEq

-- Define the friendship and age relationships
def isGoodFriend (x y : Employee) : Prop :=
  -- A's husband is Yi's good friend.
  (x = Employee.Jia ∧ y = Employee.Yi) ∨
  (x = Employee.Yi ∧ y = Employee.Jia)

def isYoungest (x : Employee) : Prop :=
  -- Specify that Jia is the youngest
  x = Employee.Jia

def isOlder (x y : Employee) : Prop :=
  -- Bing is older than C's husband.
  x = Employee.Bing ∧ y ≠ Employee.Bing

-- The pairings of husbands and wives: Jia—A, Yi—C, Bing—B.
def pairings (x : Employee) : Wife :=
  match x with
  | Employee.Jia => Wife.A
  | Employee.Yi => Wife.C
  | Employee.Bing => Wife.B

-- Proving the given pairings fit the conditions.
theorem correct_pairings : 
  ∀ (x : Employee), 
  isGoodFriend (Employee.Jia) (Employee.Yi) ∧ 
  isYoungest Employee.Jia ∧ 
  (isOlder Employee.Bing Employee.Jia ∨ isOlder Employee.Bing Employee.Yi) → 
  pairings x = match x with
               | Employee.Jia => Wife.A
               | Employee.Yi => Wife.C
               | Employee.Bing => Wife.B :=
by
  sorry

end correct_pairings_l986_98637


namespace option_b_correct_l986_98649

theorem option_b_correct (a b c : ℝ) (hc : c ≠ 0) (h : a * c^2 > b * c^2) : a > b :=
sorry

end option_b_correct_l986_98649


namespace fraction_power_minus_one_l986_98612

theorem fraction_power_minus_one :
  (5 / 3) ^ 4 - 1 = 544 / 81 := 
by
  sorry

end fraction_power_minus_one_l986_98612


namespace min_fence_length_l986_98654

theorem min_fence_length (x : ℝ) (h : x > 0) (A : x * (64 / x) = 64) : 2 * (x + 64 / x) ≥ 32 :=
by
  have t := (2 * (x + 64 / x)) 
  sorry -- Proof omitted, only statement provided as per instructions

end min_fence_length_l986_98654


namespace pearls_problem_l986_98696

theorem pearls_problem :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n = 54) ∧ (n % 9 = 0) :=
by sorry

end pearls_problem_l986_98696


namespace last_three_digits_of_8_pow_1000_l986_98646

theorem last_three_digits_of_8_pow_1000 (h : 8 ^ 125 ≡ 2 [MOD 1250]) : (8 ^ 1000) % 1000 = 256 :=
by
  sorry

end last_three_digits_of_8_pow_1000_l986_98646


namespace baker_batches_chocolate_chip_l986_98643

noncomputable def number_of_batches (total_cookies : ℕ) (oatmeal_cookies : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  (total_cookies - oatmeal_cookies) / cookies_per_batch

theorem baker_batches_chocolate_chip (total_cookies oatmeal_cookies cookies_per_batch : ℕ) 
  (h_total : total_cookies = 10) 
  (h_oatmeal : oatmeal_cookies = 4) 
  (h_batch : cookies_per_batch = 3) : 
  number_of_batches total_cookies oatmeal_cookies cookies_per_batch = 2 :=
by
  sorry

end baker_batches_chocolate_chip_l986_98643


namespace hulk_strength_l986_98689

theorem hulk_strength:
    ∃ n: ℕ, (2^(n-1) > 1000) ∧ (∀ m: ℕ, (2^(m-1) > 1000 → n ≤ m)) := sorry

end hulk_strength_l986_98689


namespace Shannon_ratio_2_to_1_l986_98642

structure IceCreamCarton :=
  (scoops : ℕ)

structure PersonWants :=
  (vanilla : ℕ)
  (chocolate : ℕ)
  (strawberry : ℕ)

noncomputable def total_scoops_served (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants) : ℕ :=
  ethan_wants.vanilla + ethan_wants.chocolate +
  lucas_wants.chocolate +
  danny_wants.chocolate +
  connor_wants.chocolate +
  olivia_wants.vanilla + olivia_wants.strawberry

theorem Shannon_ratio_2_to_1 
    (cartons : List IceCreamCarton)
    (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants)
    (scoops_left : ℕ) : 
    -- Conditions
    (∀ carton ∈ cartons, carton.scoops = 10) →
    (cartons.length = 3) →
    (ethan_wants.vanilla = 1 ∧ ethan_wants.chocolate = 1) →
    (lucas_wants.chocolate = 2) →
    (danny_wants.chocolate = 2) →
    (connor_wants.chocolate = 2) →
    (olivia_wants.vanilla = 1 ∧ olivia_wants.strawberry = 1) →
    (scoops_left = 16) →
    -- To Prove
    4 / olivia_wants.vanilla + olivia_wants.strawberry = 2 := 
sorry

end Shannon_ratio_2_to_1_l986_98642


namespace determine_M_l986_98679

theorem determine_M (M : ℕ) (h : 12 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2) : M = 36 :=
by
  sorry

end determine_M_l986_98679


namespace team_a_took_fewer_hours_l986_98691

/-- Two dogsled teams raced across a 300-mile course. 
Team A finished the course in fewer hours than Team E. 
Team A's average speed was 5 mph greater than Team E's, which was 20 mph. 
How many fewer hours did Team A take to finish the course compared to Team E? --/

theorem team_a_took_fewer_hours :
  let distance := 300
  let speed_e := 20
  let speed_a := speed_e + 5
  let time_e := distance / speed_e
  let time_a := distance / speed_a
  time_e - time_a = 3 := by
  sorry

end team_a_took_fewer_hours_l986_98691


namespace percent_decrease_to_original_price_l986_98650

variable (x : ℝ) (p : ℝ)

def new_price (x : ℝ) : ℝ := 1.35 * x

theorem percent_decrease_to_original_price :
  ∀ (x : ℝ), x ≠ 0 → (1 - (7 / 27)) * (new_price x) = x := 
sorry

end percent_decrease_to_original_price_l986_98650


namespace imaginary_part_of_z_squared_l986_98684

-- Let i be the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number (1 - 2i)
def z : ℂ := 1 - 2 * i

-- Define the expanded form of (1 - 2i)^2
def z_squared : ℂ := z^2

-- State the problem of finding the imaginary part of (1 - 2i)^2
theorem imaginary_part_of_z_squared : (z_squared).im = -4 := by
  sorry

end imaginary_part_of_z_squared_l986_98684


namespace find_b_l986_98688

theorem find_b (a u v w : ℝ) (b : ℝ)
  (h1 : ∀ x : ℝ, 12 * x^3 + 7 * a * x^2 + 6 * b * x + b = 0 → (x = u ∨ x = v ∨ x = w))
  (h2 : 0 < u ∧ 0 < v ∧ 0 < w)
  (h3 : u ≠ v ∧ v ≠ w ∧ u ≠ w)
  (h4 : Real.log u / Real.log 3 + Real.log v / Real.log 3 + Real.log w / Real.log 3 = 3):
  b = -324 := 
sorry

end find_b_l986_98688


namespace ratio_of_sums_of_sides_and_sines_l986_98618

theorem ratio_of_sums_of_sides_and_sines (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (ha : a = 3) 
  (h : a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C) : 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 3 := 
by 
  sorry

end ratio_of_sums_of_sides_and_sines_l986_98618


namespace company_budget_salaries_degrees_l986_98661

theorem company_budget_salaries_degrees :
  let transportation := 0.20
  let research_and_development := 0.09
  let utilities := 0.05
  let equipment := 0.04
  let supplies := 0.02
  let total_budget := 1.0
  let total_percentage := transportation + research_and_development + utilities + equipment + supplies
  let salaries_percentage := total_budget - total_percentage
  let total_degrees := 360.0
  let degrees_salaries := salaries_percentage * total_degrees
  degrees_salaries = 216 :=
by
  sorry

end company_budget_salaries_degrees_l986_98661


namespace find_width_of_room_eq_l986_98603

noncomputable def total_cost : ℝ := 20625
noncomputable def rate_per_sqm : ℝ := 1000
noncomputable def length_of_room : ℝ := 5.5
noncomputable def area_paved : ℝ := total_cost / rate_per_sqm
noncomputable def width_of_room : ℝ := area_paved / length_of_room

theorem find_width_of_room_eq :
  width_of_room = 3.75 :=
sorry

end find_width_of_room_eq_l986_98603


namespace num_four_letter_initials_l986_98601

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l986_98601


namespace relationship_abc_l986_98614

noncomputable def a := (1 / 3 : ℝ) ^ (2 / 3)
noncomputable def b := (2 / 3 : ℝ) ^ (1 / 3)
noncomputable def c := Real.logb (1/2) (1/3)

theorem relationship_abc : c > b ∧ b > a :=
by
  sorry

end relationship_abc_l986_98614


namespace frank_fence_length_l986_98640

theorem frank_fence_length (L W total_fence : ℝ) 
  (hW : W = 40) 
  (hArea : L * W = 200) 
  (htotal_fence : total_fence = 2 * L + W) : 
  total_fence = 50 := 
by 
  sorry

end frank_fence_length_l986_98640


namespace point_on_curve_iff_F_eq_zero_l986_98675

variable (F : ℝ → ℝ → ℝ)
variable (a b : ℝ)

theorem point_on_curve_iff_F_eq_zero :
  (F a b = 0) ↔ (∃ P : ℝ × ℝ, P = (a, b) ∧ F P.1 P.2 = 0) :=
by
  sorry

end point_on_curve_iff_F_eq_zero_l986_98675


namespace ratio_yellow_jelly_beans_l986_98606

theorem ratio_yellow_jelly_beans :
  let bag_A_total := 24
  let bag_B_total := 30
  let bag_C_total := 32
  let bag_D_total := 34
  let bag_A_yellow_ratio := 0.40
  let bag_B_yellow_ratio := 0.30
  let bag_C_yellow_ratio := 0.25 
  let bag_D_yellow_ratio := 0.10
  let bag_A_yellow := bag_A_total * bag_A_yellow_ratio
  let bag_B_yellow := bag_B_total * bag_B_yellow_ratio
  let bag_C_yellow := bag_C_total * bag_C_yellow_ratio
  let bag_D_yellow := bag_D_total * bag_D_yellow_ratio
  let total_yellow := bag_A_yellow + bag_B_yellow + bag_C_yellow + bag_D_yellow
  let total_beans := bag_A_total + bag_B_total + bag_C_total + bag_D_total
  (total_yellow / total_beans) = 0.25 := by
  sorry

end ratio_yellow_jelly_beans_l986_98606


namespace garden_length_l986_98651

theorem garden_length 
  (W : ℕ) (small_gate_width : ℕ) (large_gate_width : ℕ) (P : ℕ)
  (hW : W = 125)
  (h_small_gate : small_gate_width = 3)
  (h_large_gate : large_gate_width = 10)
  (hP : P = 687) :
  ∃ (L : ℕ), P = 2 * L + 2 * W - (small_gate_width + large_gate_width) ∧ L = 225 := by
  sorry

end garden_length_l986_98651


namespace number_div_0_04_eq_100_9_l986_98678

theorem number_div_0_04_eq_100_9 :
  ∃ number : ℝ, (number / 0.04 = 100.9) ∧ (number = 4.036) :=
sorry

end number_div_0_04_eq_100_9_l986_98678


namespace unique_solution_arith_prog_system_l986_98673

theorem unique_solution_arith_prog_system (x y : ℝ) : 
  (6 * x + 9 * y = 12) ∧ (15 * x + 18 * y = 21) ↔ (x = -1) ∧ (y = 2) :=
by sorry

end unique_solution_arith_prog_system_l986_98673


namespace imaginary_unit_cube_l986_98647

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
by
  sorry

end imaginary_unit_cube_l986_98647


namespace notched_circle_coordinates_l986_98671

variable (a b : ℝ)

theorem notched_circle_coordinates : 
  let sq_dist_from_origin := a^2 + b^2
  let A := (a, b + 5)
  let C := (a + 3, b)
  (a^2 + (b + 5)^2 = 36 ∧ (a + 3)^2 + b^2 = 36) →
  (sq_dist_from_origin = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) :=
by
  sorry

end notched_circle_coordinates_l986_98671


namespace simplify_expression_l986_98694

theorem simplify_expression : 
  (1 / (1 / (1 / 2)^0 + 1 / (1 / 2)^1 + 1 / (1 / 2)^2 + 1 / (1 / 2)^3)) = 1 / 15 :=
by 
  sorry

end simplify_expression_l986_98694


namespace trig_identity_l986_98644

theorem trig_identity (α : ℝ) : 
  (2 * (Real.sin (4 * α))^2 - 1) / 
  (2 * (1 / Real.tan (Real.pi / 4 + 4 * α)) * (Real.cos (5 * Real.pi / 4 - 4 * α))^2) = -1 :=
by
  sorry

end trig_identity_l986_98644


namespace georgie_initial_avocados_l986_98636

-- Define the conditions
def avocados_needed_per_serving := 3
def servings_made := 3
def avocados_bought_by_sister := 4
def total_avocados_needed := avocados_needed_per_serving * servings_made

-- The statement to prove
theorem georgie_initial_avocados : (total_avocados_needed - avocados_bought_by_sister) = 5 :=
sorry

end georgie_initial_avocados_l986_98636


namespace simplify_exponent_fraction_l986_98674

theorem simplify_exponent_fraction : (3 ^ 2015 + 3 ^ 2013) / (3 ^ 2015 - 3 ^ 2013) = 5 / 4 := by
  sorry

end simplify_exponent_fraction_l986_98674


namespace find_s_l986_98620

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem find_s (p q r s : ℝ)
  (h1 : ∀ (x : ℝ), g x p q r s = (x + 1) * (x + 10) * (x + 10) * (x + 10))
  (h2 : p + q + r + s = 2673) :
  s = 1000 := 
  sorry

end find_s_l986_98620


namespace factorization_of_square_difference_l986_98632

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end factorization_of_square_difference_l986_98632


namespace log3_x_minus_1_increasing_l986_98666

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem log3_x_minus_1_increasing : is_increasing_on (fun x => log_base_3 (x - 1)) (Set.Ioi 1) :=
sorry

end log3_x_minus_1_increasing_l986_98666


namespace sum_of_cubes_mod_4_l986_98604

theorem sum_of_cubes_mod_4 :
  let b := 2
  let n := 2010
  ( (n * (n + 1) / 2) ^ 2 ) % (b ^ 2) = 1 :=
by
  let b := 2
  let n := 2010
  sorry

end sum_of_cubes_mod_4_l986_98604


namespace find_x_l986_98629

noncomputable def eq_num (x : ℝ) : Prop :=
  9 - 3 / (1 / 3) + x = 3

theorem find_x : ∃ x : ℝ, eq_num x ∧ x = 3 := 
by
  sorry

end find_x_l986_98629


namespace sum_of_first_3n_terms_l986_98616

-- Define the sums of the geometric sequence
variable (S_n S_2n S_3n : ℕ)

-- Given conditions
variable (h1 : S_n = 48)
variable (h2 : S_2n = 60)

-- The statement we need to prove
theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 := by
  sorry

end sum_of_first_3n_terms_l986_98616


namespace snail_distance_l986_98668

def speed_A : ℝ := 10
def speed_B : ℝ := 15
def time_difference : ℝ := 0.5

theorem snail_distance : 
  ∃ (D : ℝ) (t_A t_B : ℝ), 
    D = speed_A * t_A ∧ 
    D = speed_B * t_B ∧
    t_A = t_B + time_difference ∧ 
    D = 15 := 
by
  sorry

end snail_distance_l986_98668


namespace sin_75_eq_sqrt6_add_sqrt2_div4_l986_98600

theorem sin_75_eq_sqrt6_add_sqrt2_div4 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
sorry

end sin_75_eq_sqrt6_add_sqrt2_div4_l986_98600


namespace positive_distinct_solutions_conditons_l986_98605

-- Definitions corresponding to the conditions in the problem
variables {x y z a b : ℝ}

-- The statement articulates the condition
theorem positive_distinct_solutions_conditons (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = b^2) (h3 : xy = z^2) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x ≠ y) (h8 : y ≠ z) (h9 : x ≠ z) : 
  b^2 ≥ a^2 / 2 :=
sorry

end positive_distinct_solutions_conditons_l986_98605


namespace determine_M_l986_98655

theorem determine_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 := 
sorry

end determine_M_l986_98655


namespace book_area_correct_l986_98639

def book_length : ℝ := 5
def book_width : ℝ := 10
def book_area (length : ℝ) (width : ℝ) : ℝ := length * width

theorem book_area_correct :
  book_area book_length book_width = 50 :=
by
  sorry

end book_area_correct_l986_98639


namespace problem_l986_98669

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) (α1 : ℝ) (α2 : ℝ) :=
  m * Real.sin (Real.pi * x + α1) + n * Real.cos (Real.pi * x + α2)

variables (m n α1 α2 : ℝ) (h_m : m ≠ 0) (h_n : n ≠ 0) (h_α1 : α1 ≠ 0) (h_α2 : α2 ≠ 0)

theorem problem (h : f 2008 m n α1 α2 = 1) : f 2009 m n α1 α2 = -1 :=
  sorry

end problem_l986_98669


namespace least_n_divisible_by_25_and_7_l986_98625

theorem least_n_divisible_by_25_and_7 (n : ℕ) (h1 : n > 1) (h2 : n % 25 = 1) (h3 : n % 7 = 1) : n = 126 :=
by
  sorry

end least_n_divisible_by_25_and_7_l986_98625


namespace least_positive_integer_solution_l986_98626

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l986_98626


namespace no_integer_solutions_l986_98652

theorem no_integer_solutions (x y : ℤ) : ¬ (x^2 + 4 * x - 11 = 8 * y) := 
by
  sorry

end no_integer_solutions_l986_98652


namespace domain_f_l986_98634

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 9*x + 18)

theorem domain_f :
  (∀ x : ℝ, (x ≠ -6) ∧ (x ≠ -3) → ∃ y : ℝ, y = f x) ∧
  (∀ x : ℝ, x = -6 ∨ x = -3 → ¬(∃ y : ℝ, y = f x)) :=
sorry

end domain_f_l986_98634


namespace range_of_a_l986_98692

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ x1^3 - 3*x1 + a = 0 ∧ x2^3 - 3*x2 + a = 0 ∧ x3^3 - 3*x3 + a = 0) 
  ↔ -2 < a ∧ a < 2 :=
sorry

end range_of_a_l986_98692


namespace rolls_combinations_l986_98656

theorem rolls_combinations (x1 x2 x3 : ℕ) (h1 : x1 + x2 + x3 = 2) : 
  (Nat.choose (2 + 3 - 1) (3 - 1) = 6) :=
by
  sorry

end rolls_combinations_l986_98656


namespace initial_oranges_count_l986_98658

theorem initial_oranges_count
  (initial_apples : ℕ := 50)
  (apple_cost : ℝ := 0.80)
  (orange_cost : ℝ := 0.50)
  (total_earnings : ℝ := 49)
  (remaining_apples : ℕ := 10)
  (remaining_oranges : ℕ := 6)
  : initial_oranges = 40 := 
by
  sorry

end initial_oranges_count_l986_98658


namespace parabola_distance_l986_98699

open Real

theorem parabola_distance (x₀ : ℝ) (h₁ : ∃ p > 0, (x₀^2 = 2 * p * 2) ∧ (2 + p / 2 = 5 / 2)) : abs (sqrt (x₀^2 + 4)) = 2 * sqrt 2 :=
by
  rcases h₁ with ⟨p, hp, h₀, h₂⟩
  sorry

end parabola_distance_l986_98699


namespace find_measure_A_and_b_c_sum_l986_98672

open Real

noncomputable def triangle_abc (a b c A B C : ℝ) : Prop :=
  ∀ (A B C : ℝ),
  A + B + C = π ∧
  a = sin A ∧
  b = sin B ∧
  c = sin C ∧
  cos (A - C) - cos (A + C) = sqrt 3 * sin C

theorem find_measure_A_and_b_c_sum (a b c A B C : ℝ)
  (h_triangle : triangle_abc a b c A B C) 
  (h_area : (1/2) * b * c * (sin A) = (3 * sqrt 3) / 16) 
  (h_b_def : b = sin B) :
  A = π / 3 ∧ b + c = sqrt 3 := by
  sorry

end find_measure_A_and_b_c_sum_l986_98672
