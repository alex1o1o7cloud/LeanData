import Mathlib

namespace find_students_that_got_As_l578_57891

variables (Emily Frank Grace Harry : Prop)

theorem find_students_that_got_As
  (cond1 : Emily → Frank)
  (cond2 : Frank → Grace)
  (cond3 : Grace → Harry)
  (cond4 : Harry → ¬ Emily)
  (three_A_students : ¬ (Emily ∧ Frank ∧ Grace ∧ Harry) ∧
                      (Emily ∧ Frank ∧ Grace ∧ ¬ Harry ∨
                       Emily ∧ Frank ∧ ¬ Grace ∧ Harry ∨
                       Emily ∧ ¬ Frank ∧ Grace ∧ Harry ∨
                       ¬ Emily ∧ Frank ∧ Grace ∧ Harry)) :
  (¬ Emily ∧ Frank ∧ Grace ∧ Harry) :=
by {
  sorry
}

end find_students_that_got_As_l578_57891


namespace amount_p_l578_57834

variable (P : ℚ)

/-- p has $42 more than what q and r together would have had if both q and r had 1/8 of what p has.
    We need to prove that P = 56. -/
theorem amount_p (h : P = (1/8 : ℚ) * P + (1/8) * P + 42) : P = 56 :=
by
  sorry

end amount_p_l578_57834


namespace max_sum_of_factors_l578_57887

theorem max_sum_of_factors (p q : ℕ) (hpq : p * q = 100) : p + q ≤ 101 :=
sorry

end max_sum_of_factors_l578_57887


namespace remaining_pieces_l578_57831

theorem remaining_pieces (initial_pieces : ℕ) (arianna_lost : ℕ) (samantha_lost : ℕ) (diego_lost : ℕ) (lucas_lost : ℕ) :
  initial_pieces = 128 → arianna_lost = 3 → samantha_lost = 9 → diego_lost = 5 → lucas_lost = 7 →
  initial_pieces - (arianna_lost + samantha_lost + diego_lost + lucas_lost) = 104 := by
  sorry

end remaining_pieces_l578_57831


namespace sum_of_cubes_eq_neg_27_l578_57873

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l578_57873


namespace solve_for_y_l578_57822

theorem solve_for_y (y : ℚ) : 
  y + 1 / 3 = 3 / 8 - 1 / 4 → y = -5 / 24 := 
by
  sorry

end solve_for_y_l578_57822


namespace find_b_l578_57871

theorem find_b (a b : ℝ) (h1 : a * (a - 4) = 21) (h2 : b * (b - 4) = 21) (h3 : a + b = 4) (h4 : a ≠ b) :
  b = -3 :=
sorry

end find_b_l578_57871


namespace unique_root_iff_k_eq_4_l578_57869

theorem unique_root_iff_k_eq_4 (k : ℝ) : 
  (∃! x : ℝ, x^2 - 4 * x + k = 0) ↔ k = 4 := 
by {
  sorry
}

end unique_root_iff_k_eq_4_l578_57869


namespace max_sum_of_squares_l578_57862

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 86) 
  (h3 : ad + bc = 180) 
  (h4 : cd = 110) : 
  a^2 + b^2 + c^2 + d^2 ≤ 258 :=
sorry

end max_sum_of_squares_l578_57862


namespace ratio_platform_to_train_length_l578_57875

variable (L P t : ℝ)

-- Definitions based on conditions
def train_has_length (L : ℝ) : Prop := true
def train_constant_velocity : Prop := true
def train_passes_pole_in_t_seconds (L t : ℝ) : Prop := L / t = L
def train_passes_platform_in_4t_seconds (L P t : ℝ) : Prop := L / t = (L + P) / (4 * t)

-- Theorem statement: ratio of the length of the platform to the length of the train is 3:1
theorem ratio_platform_to_train_length (h1 : train_has_length L) 
                                      (h2 : train_constant_velocity) 
                                      (h3 : train_passes_pole_in_t_seconds L t)
                                      (h4 : train_passes_platform_in_4t_seconds L P t) :
  P / L = 3 := 
by sorry

end ratio_platform_to_train_length_l578_57875


namespace ratio_of_compositions_l578_57801

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem ratio_of_compositions :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 :=
by
  -- Proof will go here
  sorry

end ratio_of_compositions_l578_57801


namespace optimal_strategies_and_value_l578_57896

-- Define the payoff matrix for the two-player zero-sum game
def payoff_matrix : Matrix (Fin 2) (Fin 2) ℕ := ![![12, 22], ![32, 2]]

-- Define the optimal mixed strategies for both players
def optimal_strategy_row_player : Fin 2 → ℚ
| 0 => 3 / 4
| 1 => 1 / 4

def optimal_strategy_column_player : Fin 2 → ℚ
| 0 => 1 / 2
| 1 => 1 / 2

-- Define the value of the game
def value_of_game := (17 : ℚ)

theorem optimal_strategies_and_value :
  (∀ i j, (optimal_strategy_row_player 0 * payoff_matrix 0 j + optimal_strategy_row_player 1 * payoff_matrix 1 j = value_of_game) ∧
           (optimal_strategy_column_player 0 * payoff_matrix i 0 + optimal_strategy_column_player 1 * payoff_matrix i 1 = value_of_game)) :=
by 
  -- sorry is used as a placeholder for the proof
  sorry

end optimal_strategies_and_value_l578_57896


namespace fewer_vip_tickets_sold_l578_57885

-- Definitions based on the conditions
variables (V G : ℕ)
def tickets_sold := V + G = 320
def total_cost := 40 * V + 10 * G = 7500

-- The main statement to prove
theorem fewer_vip_tickets_sold :
  tickets_sold V G → total_cost V G → G - V = 34 := 
by
  intros h1 h2
  sorry

end fewer_vip_tickets_sold_l578_57885


namespace find_f_of_2_l578_57889

variable (f : ℝ → ℝ)

-- Given condition: f is the inverse function of the exponential function 2^x
def inv_function : Prop := ∀ x, f (2^x) = x ∧ 2^(f x) = x

theorem find_f_of_2 (h : inv_function f) : f 2 = 1 :=
by sorry

end find_f_of_2_l578_57889


namespace economy_value_after_two_years_l578_57859

/--
Given an initial amount A₀ = 3200,
that increases annually by 1/8th of itself,
with an inflation rate of 3% in the first year and 4% in the second year,
prove that the value of the amount after two years is 3771.36
-/
theorem economy_value_after_two_years :
  let A₀ := 3200 
  let increase_rate := 1 / 8
  let inflation_rate_year_1 := 0.03
  let inflation_rate_year_2 := 0.04
  let A₁ := A₀ * (1 + increase_rate)
  let V₁ := A₁ * (1 - inflation_rate_year_1)
  let A₂ := V₁ * (1 + increase_rate)
  let V₂ := A₂ * (1 - inflation_rate_year_2)
  V₂ = 3771.36 :=
by
  simp only []
  sorry

end economy_value_after_two_years_l578_57859


namespace sequence_an_solution_l578_57895

noncomputable def a_n (n : ℕ) : ℝ := (
  (1 / 2) * (2 + Real.sqrt 3)^n + 
  (1 / 2) * (2 - Real.sqrt 3)^n
)^2

theorem sequence_an_solution (n : ℕ) : 
  ∀ (a b : ℕ → ℝ),
  a 0 = 1 → 
  b 0 = 0 → 
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) → 
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) → 
  a n = a_n n := sorry

end sequence_an_solution_l578_57895


namespace smallest_possible_n_l578_57818

theorem smallest_possible_n (n : ℕ) (h1 : 0 < n) (h2 : 0 < 60) 
  (h3 : (Nat.lcm 60 n) / (Nat.gcd 60 n) = 24) : n = 20 :=
by sorry

end smallest_possible_n_l578_57818


namespace truck_dirt_road_time_l578_57838

noncomputable def time_on_dirt_road (time_paved : ℝ) (speed_increment : ℝ) (total_distance : ℝ) (dirt_speed : ℝ) : ℝ :=
  let paved_speed := dirt_speed + speed_increment
  let distance_paved := paved_speed * time_paved
  let distance_dirt := total_distance - distance_paved
  distance_dirt / dirt_speed

theorem truck_dirt_road_time :
  time_on_dirt_road 2 20 200 32 = 3 :=
by
  sorry

end truck_dirt_road_time_l578_57838


namespace find_b_l578_57843

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
by {
  -- Proof will be filled in here
  sorry
}

end find_b_l578_57843


namespace sequence_area_formula_l578_57850

open Real

noncomputable def S_n (n : ℕ) : ℝ := (8 / 5) - (3 / 5) * (4 / 9) ^ n

theorem sequence_area_formula (n : ℕ) :
  S_n n = (8 / 5) - (3 / 5) * (4 / 9) ^ n := sorry

end sequence_area_formula_l578_57850


namespace pyramid_volume_l578_57803

theorem pyramid_volume (a : ℝ) (h : a = 2)
  (b : ℝ) (hb : b = 18) :
  ∃ V, V = 2 * Real.sqrt 2 :=
by
  sorry

end pyramid_volume_l578_57803


namespace john_walks_further_than_nina_l578_57852

theorem john_walks_further_than_nina :
  let john_distance := 0.7
  let nina_distance := 0.4
  john_distance - nina_distance = 0.3 :=
by
  sorry

end john_walks_further_than_nina_l578_57852


namespace largest_possible_P10_l578_57815

noncomputable def P (x : ℤ) : ℤ := x^2 + 3*x + 3

theorem largest_possible_P10 : P 10 = 133 := by
  sorry

end largest_possible_P10_l578_57815


namespace sector_area_l578_57848

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = π / 3) (hr : r = 4) : 
  (1/2) * (r * θ) * r = 8 * π / 3 :=
by
  -- Implicitly use the given values of θ and r by substituting them in the expression.
  sorry

end sector_area_l578_57848


namespace numOxygenAtoms_l578_57829

-- Define the conditions as hypothesis
def numCarbonAtoms : ℕ := 4
def numHydrogenAtoms : ℕ := 8
def molecularWeight : ℕ := 88
def atomicWeightCarbon : ℕ := 12
def atomicWeightHydrogen : ℕ := 1
def atomicWeightOxygen : ℕ := 16

-- The statement to be proved
theorem numOxygenAtoms :
  let totalWeightC := numCarbonAtoms * atomicWeightCarbon
  let totalWeightH := numHydrogenAtoms * atomicWeightHydrogen
  let totalWeightCH := totalWeightC + totalWeightH
  let weightOxygenAtoms := molecularWeight - totalWeightCH
  let numOxygenAtoms := weightOxygenAtoms / atomicWeightOxygen
  numOxygenAtoms = 2 :=
by {
  sorry
}

end numOxygenAtoms_l578_57829


namespace solve_pow_problem_l578_57849

theorem solve_pow_problem : (-2)^1999 + (-2)^2000 = 2^1999 := 
sorry

end solve_pow_problem_l578_57849


namespace number_of_workers_who_read_all_three_books_l578_57890

theorem number_of_workers_who_read_all_three_books
  (W S K A SK SA KA SKA N : ℝ)
  (hW : W = 75)
  (hS : S = 1 / 2 * W)
  (hK : K = 1 / 4 * W)
  (hA : A = 1 / 5 * W)
  (hSK : SK = 2 * SKA)
  (hN : N = S - (SK + SA + SKA) - 1)
  (hTotal : S + K + A - (SK + SA + KA - SKA) + N = W) :
  SKA = 6 :=
by
  -- The proof steps are omitted
  sorry

end number_of_workers_who_read_all_three_books_l578_57890


namespace imaginary_part_of_fraction_l578_57866

open Complex

theorem imaginary_part_of_fraction :
  ∃ z : ℂ, z = ⟨0, 1⟩ / ⟨1, 1⟩ ∧ z.im = 1 / 2 :=
by
  sorry

end imaginary_part_of_fraction_l578_57866


namespace abs_neg_2023_l578_57894

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l578_57894


namespace estimate_red_balls_l578_57864

-- Define the conditions in Lean 4
def total_balls : ℕ := 15
def freq_red_ball : ℝ := 0.4

-- Define the proof statement without proving it
theorem estimate_red_balls (x : ℕ) 
  (h1 : x ≤ total_balls) 
  (h2 : ∃ (p : ℝ), p = x / total_balls ∧ p = freq_red_ball) :
  x = 6 :=
sorry

end estimate_red_balls_l578_57864


namespace find_q_sum_of_bn_l578_57808

-- Defining the sequences and conditions
def a (n : ℕ) (q : ℝ) : ℝ := q^(n-1)

def b (n : ℕ) (q : ℝ) : ℝ := a n q + n

-- Given that 2a_1, (1/2)a_3, a_2 form an arithmetic sequence
def condition_arithmetic_sequence (q : ℝ) : Prop :=
  2 * a 1 q + a 2 q = (1 / 2) * a 3 q + (1 / 2) * a 3 q

-- To be proved: Given conditions, prove q = 2
theorem find_q : ∃ q > 0, a 1 q = 1 ∧ a 2 q = q ∧ a 3 q = q^2 ∧ condition_arithmetic_sequence q ∧ q = 2 :=
by {
  sorry
}

-- Given b_n = a_n + n, prove T_n = (n(n+1))/2 + 2^n - 1
theorem sum_of_bn (n : ℕ) : 
  ∃ T_n : ℕ → ℝ, T_n n = (n * (n + 1)) / 2 + (2^n) - 1 :=
by {
  sorry
}

end find_q_sum_of_bn_l578_57808


namespace average_salary_proof_l578_57804

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end average_salary_proof_l578_57804


namespace find_pairs_solution_l578_57899

theorem find_pairs_solution (x y : ℝ) :
  (x^3 + x^2 * y + x * y^2 + y^3 = 8 * (x^2 + x * y + y^2 + 1)) ↔ 
  (x, y) = (8, -2) ∨ (x, y) = (-2, 8) ∨ 
  (x, y) = (4 + Real.sqrt 15, 4 - Real.sqrt 15) ∨ 
  (x, y) = (4 - Real.sqrt 15, 4 + Real.sqrt 15) :=
by 
  sorry

end find_pairs_solution_l578_57899


namespace average_output_assembly_line_l578_57833

theorem average_output_assembly_line
  (initial_rate : ℕ) (initial_cogs : ℕ) 
  (increased_rate : ℕ) (increased_cogs : ℕ)
  (h1 : initial_rate = 15)
  (h2 : initial_cogs = 60)
  (h3 : increased_rate = 60)
  (h4 : increased_cogs = 60) :
  (initial_cogs + increased_cogs) / (initial_cogs / initial_rate + increased_cogs / increased_rate) = 24 := 
by sorry

end average_output_assembly_line_l578_57833


namespace mike_profit_l578_57827

-- Define the conditions
def total_acres : ℕ := 200
def cost_per_acre : ℕ := 70
def sold_acres := total_acres / 2
def selling_price_per_acre : ℕ := 200

-- Statement to prove the profit Mike made is $6,000
theorem mike_profit :
  let total_cost := total_acres * cost_per_acre
  let total_revenue := sold_acres * selling_price_per_acre
  total_revenue - total_cost = 6000 := 
by
  sorry

end mike_profit_l578_57827


namespace suzanne_donation_l578_57811

theorem suzanne_donation :
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  total_donation = 310 :=
by
  let base_donation := 10
  let total_distance := 5
  let total_donation := (List.range total_distance).foldl (fun acc km => acc + base_donation * 2 ^ km) 0
  sorry

end suzanne_donation_l578_57811


namespace mutually_exclusive_but_not_opposite_l578_57876

-- Define the cards and the people
inductive Card
| Red
| Black
| Blue
| White

inductive Person
| A
| B
| C
| D

-- Define the events
def eventA_gets_red (distribution : Person → Card) : Prop :=
distribution Person.A = Card.Red

def eventB_gets_red (distribution : Person → Card) : Prop :=
distribution Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Prop) : Prop :=
P → ¬ Q

-- Statement of the problem
theorem mutually_exclusive_but_not_opposite :
  ∀ (distribution : Person → Card), 
    mutually_exclusive (eventA_gets_red distribution) (eventB_gets_red distribution) ∧ 
    ¬ (eventA_gets_red distribution ↔ eventB_gets_red distribution) :=
by sorry

end mutually_exclusive_but_not_opposite_l578_57876


namespace sum_of_roots_of_abs_quadratic_is_zero_l578_57823

theorem sum_of_roots_of_abs_quadratic_is_zero : 
  ∀ x : ℝ, (|x|^2 + |x| - 6 = 0) → (x = 2 ∨ x = -2) → (2 + (-2) = 0) :=
by
  intros x h h1
  sorry

end sum_of_roots_of_abs_quadratic_is_zero_l578_57823


namespace compute_expression_l578_57839

theorem compute_expression : (3 + 5) ^ 2 + (3 ^ 2 + 5 ^ 2) = 98 := by
  sorry

end compute_expression_l578_57839


namespace divisibility_criterion_l578_57860

theorem divisibility_criterion (x y : ℕ) (h_two_digit : 10 ≤ x ∧ x < 100) :
  (1207 % x = 0) ↔ (x = 10 * (x / 10) + (x % 10) ∧ (x / 10)^3 + (x % 10)^3 = 344) :=
by
  sorry

end divisibility_criterion_l578_57860


namespace price_change_38_percent_l578_57870

variables (P : ℝ) (x : ℝ)
noncomputable def final_price := P * (1 - (x / 100)^2) * 0.9
noncomputable def target_price := 0.77 * P

theorem price_change_38_percent (h : final_price P x = target_price P):
  x = 38 := sorry

end price_change_38_percent_l578_57870


namespace solve_system_of_equations_l578_57813

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ xyz = -16 ↔ 
  (x = 1 ∧ y = 4 ∧ z = -4) ∨ (x = 1 ∧ y = -4 ∧ z = 4) ∨ 
  (x = 4 ∧ y = 1 ∧ z = -4) ∨ (x = 4 ∧ y = -4 ∧ z = 1) ∨ 
  (x = -4 ∧ y = 1 ∧ z = 4) ∨ (x = -4 ∧ y = 4 ∧ z = 1) := 
by
  sorry

end solve_system_of_equations_l578_57813


namespace neznaika_made_mistake_l578_57816

-- Define the total digits used from 1 to N pages
def totalDigits (N : ℕ) : ℕ :=
  let single_digit_pages := min N 9
  let double_digit_pages := if N > 9 then N - 9 else 0
  single_digit_pages * 1 + double_digit_pages * 2

-- The main statement we want to prove
theorem neznaika_made_mistake : ¬ ∃ N : ℕ, totalDigits N = 100 :=
by
  sorry

end neznaika_made_mistake_l578_57816


namespace geometric_sequence_divisibility_l578_57809

theorem geometric_sequence_divisibility 
  (a1 : ℚ) (h1 : a1 = 1 / 2) 
  (a2 : ℚ) (h2 : a2 = 10) 
  (n : ℕ) :
  ∃ (n : ℕ), a_n = (a1 * 20^(n - 1)) ∧ (n ≥ 4) ∧ (5000 ∣ a_n) :=
by
  sorry

end geometric_sequence_divisibility_l578_57809


namespace algebraic_expression_value_l578_57888

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
sorry

end algebraic_expression_value_l578_57888


namespace total_dogs_barking_l578_57830

theorem total_dogs_barking 
  (initial_dogs : ℕ)
  (new_dogs : ℕ)
  (h1 : initial_dogs = 30)
  (h2 : new_dogs = 3 * initial_dogs) :
  initial_dogs + new_dogs = 120 :=
by
  sorry

end total_dogs_barking_l578_57830


namespace expected_value_of_unfair_die_l578_57851

-- Define the probabilities for each face of the die.
def prob_face (n : ℕ) : ℚ :=
  if n = 8 then 5/14 else 1/14

-- Define the expected value of a roll of this die.
def expected_value : ℚ :=
  (1 / 14) * 1 + (1 / 14) * 2 + (1 / 14) * 3 + (1 / 14) * 4 + (1 / 14) * 5 + (1 / 14) * 6 + (1 / 14) * 7 + (5 / 14) * 8

-- The statement to prove: the expected value of a roll of this die is 4.857.
theorem expected_value_of_unfair_die : expected_value = 4.857 := by
  sorry

end expected_value_of_unfair_die_l578_57851


namespace eval_expr_l578_57856

namespace ProofProblem

variables (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d = a + b + c)

theorem eval_expr :
  d = a + b + c →
  (a^3 + b^3 + c^3 - 3 * a * b * c) / (a * b * c) = (d * (a^2 + b^2 + c^2 - a * b - a * c - b * c)) / (a * b * c) :=
by
  intros hd
  sorry

end ProofProblem

end eval_expr_l578_57856


namespace number_of_sandwiches_l578_57892

-- Define the constants and assumptions

def soda_cost : ℤ := 1
def number_of_sodas : ℤ := 3
def cost_of_sodas : ℤ := number_of_sodas * soda_cost

def number_of_soups : ℤ := 2
def soup_cost : ℤ := cost_of_sodas
def cost_of_soups : ℤ := number_of_soups * soup_cost

def sandwich_cost : ℤ := 3 * soup_cost
def total_cost : ℤ := 18

-- The mathematical statement we want to prove
theorem number_of_sandwiches :
  ∃ n : ℤ, (n * sandwich_cost + cost_of_sodas + cost_of_soups = total_cost) ∧ n = 1 :=
by
  sorry

end number_of_sandwiches_l578_57892


namespace complex_number_powers_l578_57893

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 :=
sorry

end complex_number_powers_l578_57893


namespace john_total_spent_l578_57814

noncomputable def calculate_total_spent : ℝ :=
  let orig_price_A := 900.0
  let discount_A := 0.15 * orig_price_A
  let price_A := orig_price_A - discount_A
  let tax_A := 0.06 * price_A
  let total_A := price_A + tax_A
  let orig_price_B := 600.0
  let discount_B := 0.25 * orig_price_B
  let price_B := orig_price_B - discount_B
  let tax_B := 0.09 * price_B
  let total_B := price_B + tax_B
  let total_other_toys := total_A + total_B
  let price_lightsaber := 2 * total_other_toys
  let tax_lightsaber := 0.04 * price_lightsaber
  let total_lightsaber := price_lightsaber + tax_lightsaber
  total_other_toys + total_lightsaber

theorem john_total_spent : calculate_total_spent = 4008.312 := by
  sorry

end john_total_spent_l578_57814


namespace largest_x_by_equation_l578_57842

theorem largest_x_by_equation : ∃ x : ℚ, 
  (∀ y : ℚ, 6 * (12 * y^2 + 12 * y + 11) = y * (12 * y - 44) → y ≤ x) 
  ∧ 6 * (12 * x^2 + 12 * x + 11) = x * (12 * x - 44) 
  ∧ x = -1 := 
sorry

end largest_x_by_equation_l578_57842


namespace dodgeball_cost_l578_57844

theorem dodgeball_cost (B : ℝ) 
  (hb1 : 1.20 * B = 90) 
  (hb2 : B / 15 = 5) :
  ∃ (cost_per_dodgeball : ℝ), cost_per_dodgeball = 5 := by
sorry

end dodgeball_cost_l578_57844


namespace machines_produce_12x_boxes_in_expected_time_l578_57805

-- Definitions corresponding to the conditions
def rate_A (x : ℕ) := x / 10
def rate_B (x : ℕ) := 2 * x / 5
def rate_C (x : ℕ) := 3 * x / 8
def rate_D (x : ℕ) := x / 4

-- Total combined rate when working together
def combined_rate (x : ℕ) := rate_A x + rate_B x + rate_C x + rate_D x

-- The time taken to produce 12x boxes given their combined rate
def time_to_produce (x : ℕ) : ℕ := 12 * x / combined_rate x

-- Goal: Time taken should be 32/3 minutes
theorem machines_produce_12x_boxes_in_expected_time (x : ℕ) : time_to_produce x = 32 / 3 :=
sorry

end machines_produce_12x_boxes_in_expected_time_l578_57805


namespace find_a_l578_57826

theorem find_a (a : ℝ) : (4, -5).2 = (a - 2, a + 1).2 → a = -6 :=
by
  intro h
  sorry

end find_a_l578_57826


namespace XY_sym_diff_l578_57806

-- The sets X and Y
def X : Set ℤ := {1, 3, 5, 7}
def Y : Set ℤ := { x | x < 4 ∧ x ∈ Set.univ }

-- Definition of set operation (A - B)
def set_sub (A B : Set ℤ) : Set ℤ := { x | x ∈ A ∧ x ∉ B }

-- Definition of set operation (A * B)
def set_sym_diff (A B : Set ℤ) : Set ℤ := (set_sub A B) ∪ (set_sub B A)

-- Prove that X * Y = {-3, -2, -1, 0, 2, 5, 7}
theorem XY_sym_diff : set_sym_diff X Y = {-3, -2, -1, 0, 2, 5, 7} :=
by
  sorry

end XY_sym_diff_l578_57806


namespace distance_to_axes_l578_57820

def point (P : ℝ × ℝ) : Prop :=
  P = (3, 5)

theorem distance_to_axes (P : ℝ × ℝ) (hx : P = (3, 5)) : 
  abs P.2 = 5 ∧ abs P.1 = 3 :=
by 
  sorry

end distance_to_axes_l578_57820


namespace no_exact_cover_l578_57855

theorem no_exact_cover (large_w : ℕ) (large_h : ℕ) (small_w : ℕ) (small_h : ℕ) (n : ℕ) :
  large_w = 13 → large_h = 7 → small_w = 2 → small_h = 3 → n = 15 →
  ¬ (small_w * small_h * n = large_w * large_h) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end no_exact_cover_l578_57855


namespace population_of_metropolitan_county_l578_57846

theorem population_of_metropolitan_county : 
  let average_population := 5500
  let two_populous_cities_population := 2 * average_population
  let remaining_cities := 25 - 2
  let remaining_population := remaining_cities * average_population
  let total_population := (2 * two_populous_cities_population) + remaining_population
  total_population = 148500 := by
sorry

end population_of_metropolitan_county_l578_57846


namespace gcd_5039_3427_l578_57800

def a : ℕ := 5039
def b : ℕ := 3427

theorem gcd_5039_3427 : Nat.gcd a b = 7 := by
  sorry

end gcd_5039_3427_l578_57800


namespace total_students_at_gathering_l578_57825

theorem total_students_at_gathering (x : ℕ) 
  (h1 : ∃ x : ℕ, 0 < x)
  (h2 : (x + 6) / (2 * x + 6) = 2 / 3) : 
  (2 * x + 6) = 18 := 
  sorry

end total_students_at_gathering_l578_57825


namespace john_spent_at_candy_store_l578_57836

noncomputable def johns_allowance : ℝ := 2.40
noncomputable def arcade_spending : ℝ := (3 / 5) * johns_allowance
noncomputable def remaining_after_arcade : ℝ := johns_allowance - arcade_spending
noncomputable def toy_store_spending : ℝ := (1 / 3) * remaining_after_arcade
noncomputable def remaining_after_toy_store : ℝ := remaining_after_arcade - toy_store_spending
noncomputable def candy_store_spending : ℝ := remaining_after_toy_store

theorem john_spent_at_candy_store : candy_store_spending = 0.64 := by sorry

end john_spent_at_candy_store_l578_57836


namespace function_value_at_minus_one_l578_57863

theorem function_value_at_minus_one :
  ( -(1:ℝ)^4 + -(1:ℝ)^3 + (1:ℝ) ) / ( -(1:ℝ)^2 + (1:ℝ) ) = 1 / 2 :=
by sorry

end function_value_at_minus_one_l578_57863


namespace range_of_m_l578_57854

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0
def proposition_q (m : ℝ) : Prop := 5 - 2*m > 1

theorem range_of_m (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : m ≤ 1 :=
sorry

end range_of_m_l578_57854


namespace limes_left_l578_57857

-- Define constants
def num_limes_initial : ℕ := 9
def num_limes_given : ℕ := 4

-- Theorem to be proved
theorem limes_left : num_limes_initial - num_limes_given = 5 :=
by
  sorry

end limes_left_l578_57857


namespace number_of_houses_on_block_l578_57897

theorem number_of_houses_on_block 
  (total_mail : ℕ) 
  (white_mailboxes : ℕ) 
  (red_mailboxes : ℕ) 
  (mail_per_house : ℕ) 
  (total_white_mail : ℕ) 
  (total_red_mail : ℕ) 
  (remaining_mail : ℕ)
  (additional_houses : ℕ)
  (total_houses : ℕ) :
  total_mail = 48 ∧ 
  white_mailboxes = 2 ∧ 
  red_mailboxes = 3 ∧ 
  mail_per_house = 6 ∧ 
  total_white_mail = white_mailboxes * mail_per_house ∧
  total_red_mail = red_mailboxes * mail_per_house ∧
  remaining_mail = total_mail - (total_white_mail + total_red_mail) ∧
  additional_houses = remaining_mail / mail_per_house ∧
  total_houses = white_mailboxes + red_mailboxes + additional_houses →
  total_houses = 8 :=
by 
  sorry

end number_of_houses_on_block_l578_57897


namespace maximize_net_income_l578_57840

-- Define the conditions of the problem
def bicycles := 50
def management_cost := 115

def rental_income (x : ℕ) : ℕ :=
if x ≤ 6 then bicycles * x
else (bicycles - 3 * (x - 6)) * x

def net_income (x : ℕ) : ℤ :=
rental_income x - management_cost

-- Define the domain of the function
def domain (x : ℕ) : Prop := 3 ≤ x ∧ x ≤ 20

-- Define the piecewise function for y = f(x)
def f (x : ℕ) : ℤ :=
if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
else if 6 < x ∧ x ≤ 20 then -3 * x * x + 68 * x - 115
else 0  -- Out of domain

-- The theorem that we need to prove
theorem maximize_net_income :
  (∀ x, domain x → net_income x = f x) ∧
  (∃ x, domain x ∧ (∀ y, domain y → net_income y ≤ net_income x) ∧ x = 11) :=
by
  sorry

end maximize_net_income_l578_57840


namespace part_3_l578_57837

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) := (1/2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) := f x m + g x m

theorem part_3 (x₁ x₂ : ℝ) (m : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hm : m = -2)
  (hF : F x₁ m + F x₂ m + x₁ * x₂ = 0) : x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end part_3_l578_57837


namespace percentage_of_gold_coins_is_35_percent_l578_57807

-- Definitions of conditions
def percentage_of_objects_that_are_beads : ℝ := 0.30
def percentage_of_coins_that_are_silver : ℝ := 0.25
def percentage_of_coins_that_are_gold : ℝ := 0.50

-- Problem Statement
theorem percentage_of_gold_coins_is_35_percent 
  (h_beads : percentage_of_objects_that_are_beads = 0.30) 
  (h_silver_coins : percentage_of_coins_that_are_silver = 0.25) 
  (h_gold_coins : percentage_of_coins_that_are_gold = 0.50) :
  0.35 = 0.35 := 
sorry

end percentage_of_gold_coins_is_35_percent_l578_57807


namespace correct_student_mark_l578_57874

theorem correct_student_mark (x : ℕ) : 
  (∀ (n : ℕ), n = 30) →
  (∀ (avg correct_avg wrong_mark correct_mark : ℕ), 
    avg = 100 ∧ 
    correct_avg = 98 ∧ 
    wrong_mark = 70 ∧ 
    (n * avg) - wrong_mark + correct_mark = n * correct_avg) →
  x = 10 := by
  intros
  sorry

end correct_student_mark_l578_57874


namespace impossible_arrangement_of_numbers_l578_57861

theorem impossible_arrangement_of_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) 
(hpos : ∀ i, 0 < a i)
(hdiff : ∃ i, ∀ j ≠ i, a j = a ((j + 1) % n) - a ((j - 1 + n) % n)):
  false :=
by
  sorry

end impossible_arrangement_of_numbers_l578_57861


namespace sum_of_reflection_midpoint_coordinates_l578_57872

theorem sum_of_reflection_midpoint_coordinates (P R : ℝ × ℝ) (M : ℝ × ℝ) (P' R' M' : ℝ × ℝ) :
  P = (2, 1) → R = (12, 15) → 
  M = ((P.fst + R.fst) / 2, (P.snd + R.snd) / 2) →
  P' = (-P.fst, P.snd) → R' = (-R.fst, R.snd) →
  M' = ((P'.fst + R'.fst) / 2, (P'.snd + R'.snd) / 2) →
  (M'.fst + M'.snd) = 1 := 
by 
  intros
  sorry

end sum_of_reflection_midpoint_coordinates_l578_57872


namespace always_odd_l578_57865

theorem always_odd (p m : ℕ) (hp : p % 2 = 1) : (p^3 + 3*p*m^2 + 2*m) % 2 = 1 := 
by sorry

end always_odd_l578_57865


namespace chocolate_bar_weight_l578_57847

theorem chocolate_bar_weight :
  let square_weight := 6
  let triangles_count := 16
  let squares_count := 32
  let triangle_weight := square_weight / 2
  let total_square_weight := squares_count * square_weight
  let total_triangles_weight := triangles_count * triangle_weight
  total_square_weight + total_triangles_weight = 240 := 
by
  sorry

end chocolate_bar_weight_l578_57847


namespace number_of_classes_l578_57879

theorem number_of_classes (x : ℕ) (total_games : ℕ) (h : total_games = 45) :
  (x * (x - 1)) / 2 = total_games → x = 10 :=
by
  sorry

end number_of_classes_l578_57879


namespace emma_harry_weight_l578_57845

theorem emma_harry_weight (e f g h : ℕ) 
  (h1 : e + f = 280) 
  (h2 : f + g = 260) 
  (h3 : g + h = 290) : 
  e + h = 310 := 
sorry

end emma_harry_weight_l578_57845


namespace common_difference_arithmetic_seq_l578_57883

theorem common_difference_arithmetic_seq (a1 d : ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d) : 
  (S 5 / 5 - S 2 / 2 = 3) → d = 2 :=
by
  intros h1
  sorry

end common_difference_arithmetic_seq_l578_57883


namespace students_with_both_uncool_parents_l578_57878

theorem students_with_both_uncool_parents :
  let total_students := 35
  let cool_dads := 18
  let cool_moms := 22
  let both_cool := 11
  total_students - (cool_dads + cool_moms - both_cool) = 6 := by
sorry

end students_with_both_uncool_parents_l578_57878


namespace sum_of_non_common_roots_zero_l578_57832

theorem sum_of_non_common_roots_zero (m α β γ : ℝ) 
  (h1 : α + β = -(m + 1))
  (h2 : α * β = -3)
  (h3 : α + γ = 4)
  (h4 : α * γ = -m)
  (h_common : α^2 + (m + 1)*α - 3 = 0)
  (h_common2 : α^2 - 4*α - m = 0)
  : β + γ = 0 := sorry

end sum_of_non_common_roots_zero_l578_57832


namespace max_imag_part_of_roots_l578_57877

noncomputable def polynomial (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

theorem max_imag_part_of_roots :
  ∃ (z : ℂ), polynomial z = 0 ∧ ∀ w, polynomial w = 0 → (z.im ≤ w.im) := sorry

end max_imag_part_of_roots_l578_57877


namespace chocolate_candy_pieces_l578_57886

-- Define the initial number of boxes and the boxes given away
def initial_boxes : Nat := 12
def boxes_given : Nat := 7

-- Define the number of remaining boxes
def remaining_boxes := initial_boxes - boxes_given

-- Define the number of pieces per box
def pieces_per_box : Nat := 6

-- Calculate the total pieces Tom still has
def total_pieces := remaining_boxes * pieces_per_box

-- State the theorem
theorem chocolate_candy_pieces : total_pieces = 30 :=
by
  -- proof steps would go here
  sorry

end chocolate_candy_pieces_l578_57886


namespace option_C_correct_l578_57817

theorem option_C_correct : 5 + (-6) - (-7) = 5 - 6 + 7 := 
by
  sorry

end option_C_correct_l578_57817


namespace Amanda_money_left_l578_57835

theorem Amanda_money_left (initial_amount cost_cassette tape_count cost_headphone : ℕ) 
  (h1 : initial_amount = 50) 
  (h2 : cost_cassette = 9) 
  (h3 : tape_count = 2) 
  (h4 : cost_headphone = 25) :
  initial_amount - (tape_count * cost_cassette + cost_headphone) = 7 :=
by
  sorry

end Amanda_money_left_l578_57835


namespace inequality_solution_set_system_of_inequalities_solution_set_l578_57882

theorem inequality_solution_set (x : ℝ) (h : 3 * x - 5 > 5 * x + 3) : x < -4 :=
by sorry

theorem system_of_inequalities_solution_set (x : ℤ) 
  (h₁ : x - 1 ≥ 1 - x) 
  (h₂ : x + 8 > 4 * x - 1) : x = 1 ∨ x = 2 :=
by sorry

end inequality_solution_set_system_of_inequalities_solution_set_l578_57882


namespace ratio_problem_l578_57867

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l578_57867


namespace find_y_value_l578_57821

/-- Given angles and conditions, find the value of y in the geometric figure. -/
theorem find_y_value
  (AB_parallel_DC : true) -- AB is parallel to DC
  (ACE_straight_line : true) -- ACE is a straight line
  (angle_ACF : ℝ := 130) -- ∠ACF = 130°
  (angle_CBA : ℝ := 60) -- ∠CBA = 60°
  (angle_ACB : ℝ := 100) -- ∠ACB = 100°
  (angle_ADC : ℝ := 125) -- ∠ADC = 125°
  : 35 = 35 := -- y = 35°
by
  sorry

end find_y_value_l578_57821


namespace count_two_digit_primes_with_units_digit_3_l578_57858

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l578_57858


namespace geometric_seq_sum_l578_57853

noncomputable def a_n (n : ℕ) : ℤ :=
  (-3)^(n-1)

theorem geometric_seq_sum :
  let a1 := a_n 1
  let a2 := a_n 2
  let a3 := a_n 3
  let a4 := a_n 4
  let a5 := a_n 5
  a1 + |a2| + a3 + |a4| + a5 = 121 :=
by
  sorry

end geometric_seq_sum_l578_57853


namespace ratio_Theresa_Timothy_2010_l578_57812

def Timothy_movies_2009 : Nat := 24
def Timothy_movies_2010 := Timothy_movies_2009 + 7
def Theresa_movies_2009 := Timothy_movies_2009 / 2
def total_movies := 129
def Timothy_total_movies := Timothy_movies_2009 + Timothy_movies_2010
def Theresa_total_movies := total_movies - Timothy_total_movies
def Theresa_movies_2010 := Theresa_total_movies - Theresa_movies_2009

theorem ratio_Theresa_Timothy_2010 :
  (Theresa_movies_2010 / Timothy_movies_2010) = 2 :=
by
  sorry

end ratio_Theresa_Timothy_2010_l578_57812


namespace Juan_run_time_l578_57898

theorem Juan_run_time
  (d : ℕ) (s : ℕ) (t : ℕ)
  (H1: d = 80)
  (H2: s = 10)
  (H3: t = d / s) :
  t = 8 := 
sorry

end Juan_run_time_l578_57898


namespace part1_part2_part3_l578_57802

variable {x y : ℚ}

def star (x y : ℚ) : ℚ := x * y + 1

theorem part1 : star 2 4 = 9 := by
  sorry

theorem part2 : star (star 1 4) (-2) = -9 := by
  sorry

theorem part3 (a b c : ℚ) : star a (b + c) + 1 = star a b + star a c := by
  sorry

end part1_part2_part3_l578_57802


namespace abc_cubic_sum_identity_l578_57881

theorem abc_cubic_sum_identity (a b c : ℂ) 
  (M : Matrix (Fin 3) (Fin 3) ℂ)
  (h1 : M = fun i j => if i = 0 then (if j = 0 then a else if j = 1 then b else c)
                      else if i = 1 then (if j = 0 then b else if j = 1 then c else a)
                      else (if j = 0 then c else if j = 1 then a else b))
  (h2 : M ^ 3 = 1)
  (h3 : a * b * c = -1) :
  a^3 + b^3 + c^3 = 4 := sorry

end abc_cubic_sum_identity_l578_57881


namespace solve_equation_l578_57810

theorem solve_equation (y : ℝ) (z : ℝ) (hz : z = y^(1/3)) :
  (6 * y^(1/3) - 3 * y^(4/3) = 12 + y^(1/3) + y) ↔ (3 * z^4 + z^3 - 5 * z + 12 = 0) :=
by sorry

end solve_equation_l578_57810


namespace cube_painting_distinct_ways_l578_57828

theorem cube_painting_distinct_ways : ∃ n : ℕ, n = 7 := sorry

end cube_painting_distinct_ways_l578_57828


namespace ratio_y_to_x_l578_57841

variable (x y z : ℝ)

-- Conditions
def condition1 (x y z : ℝ) := 0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)
def condition2 (y z : ℝ) := ∃ k : ℝ, z = k * y
def condition3 (y z : ℝ) := z = 7 * y
def condition4 (x y : ℝ) := y = 5 * x / 7

theorem ratio_y_to_x (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 y z) (h3 : condition3 y z) (h4 : condition4 x y) : y / x = 5 / 7 :=
by
  sorry

end ratio_y_to_x_l578_57841


namespace reggie_marbles_l578_57819

/-- Given that Reggie and his friend played 9 games in total,
    Reggie lost 1 game, and they bet 10 marbles per game.
    Prove that Reggie has 70 marbles after all games. -/
theorem reggie_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) (marbles_initial : ℕ) 
  (h_total_games : total_games = 9) (h_lost_games : lost_games = 1) (h_marbles_per_game : marbles_per_game = 10) 
  (h_marbles_initial : marbles_initial = 0) : 
  marbles_initial + (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game = 70 :=
by
  -- We proved this in the solution steps, but will skip the proof here with sorry.
  sorry

end reggie_marbles_l578_57819


namespace tan_half_angle_is_two_l578_57880

-- Define the setup
variables (α : ℝ) (H1 : α ∈ Icc (π/2) π) (H2 : 3 * Real.sin α + 4 * Real.cos α = 0)

-- Define the main theorem
theorem tan_half_angle_is_two : Real.tan (α / 2) = 2 :=
sorry

end tan_half_angle_is_two_l578_57880


namespace tyrone_money_l578_57884

def bill_value (count : ℕ) (val : ℝ) : ℝ :=
  count * val

def total_value : ℝ :=
  bill_value 2 1 + bill_value 1 5 + bill_value 13 0.25 + bill_value 20 0.10 + bill_value 8 0.05 + bill_value 35 0.01

theorem tyrone_money : total_value = 13 := by 
  sorry

end tyrone_money_l578_57884


namespace brown_eyed_brunettes_count_l578_57824

-- Definitions of conditions
variables (total_students blue_eyed_blondes brunettes brown_eyed_students : ℕ)
variable (brown_eyed_brunettes : ℕ)

-- Initial conditions
axiom h1 : total_students = 60
axiom h2 : blue_eyed_blondes = 18
axiom h3 : brunettes = 40
axiom h4 : brown_eyed_students = 24

-- Proof objective
theorem brown_eyed_brunettes_count :
  brown_eyed_brunettes = 24 - (24 - (20 - (20 - 18))) := sorry

end brown_eyed_brunettes_count_l578_57824


namespace probability_exactly_one_instrument_l578_57868

-- Definitions of the conditions
def total_people : ℕ := 800
def frac_one_instrument : ℚ := 1 / 5
def people_two_or_more_instruments : ℕ := 64

-- Statement of the problem
theorem probability_exactly_one_instrument :
  let people_at_least_one_instrument := frac_one_instrument * total_people
  let people_exactly_one_instrument := people_at_least_one_instrument - people_two_or_more_instruments
  let probability := people_exactly_one_instrument / total_people
  probability = 3 / 25 :=
by
  -- Definitions
  let people_at_least_one_instrument : ℚ := frac_one_instrument * total_people
  let people_exactly_one_instrument : ℚ := people_at_least_one_instrument - people_two_or_more_instruments
  let probability : ℚ := people_exactly_one_instrument / total_people
  
  -- Sorry statement to skip the proof
  exact sorry

end probability_exactly_one_instrument_l578_57868
