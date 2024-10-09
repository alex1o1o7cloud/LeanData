import Mathlib

namespace problem_statement_l860_86062

variable (x y z a b c : ℝ)

-- Conditions
def condition1 := x / a + y / b + z / c = 5
def condition2 := a / x + b / y + c / z = 0

-- Proof statement
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end problem_statement_l860_86062


namespace sum_of_all_possible_values_of_g7_l860_86025

def f (x : ℝ) : ℝ := x ^ 2 - 6 * x + 14
def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g7 :
  let x1 := 3 + Real.sqrt 2;
  let x2 := 3 - Real.sqrt 2;
  let g1 := g x1;
  let g2 := g x2;
  g (f 7) = g1 + g2 := by
  sorry

end sum_of_all_possible_values_of_g7_l860_86025


namespace trisha_take_home_pay_l860_86084

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end trisha_take_home_pay_l860_86084


namespace arithmetic_sequence_a9_l860_86041

noncomputable def S (n : ℕ) (a₁ aₙ : ℝ) : ℝ := (n * (a₁ + aₙ)) / 2

theorem arithmetic_sequence_a9 (a₁ a₁₇ : ℝ) (h1 : S 17 a₁ a₁₇ = 102) : (a₁ + a₁₇) / 2 = 6 :=
by
  sorry

end arithmetic_sequence_a9_l860_86041


namespace rate_of_mangoes_is_60_l860_86082

-- Define the conditions
def kg_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 9
def total_paid : ℕ := 1100

-- Define the cost of grapes and total cost
def cost_of_grapes : ℕ := kg_grapes * rate_per_kg_grapes
def cost_of_mangoes : ℕ := total_paid - cost_of_grapes
def rate_per_kg_mangoes : ℕ := cost_of_mangoes / kg_mangoes

-- Prove that the rate of mangoes per kg is 60
theorem rate_of_mangoes_is_60 : rate_per_kg_mangoes = 60 := by
  -- Here we would provide the proof
  sorry

end rate_of_mangoes_is_60_l860_86082


namespace solve_for_y_l860_86029

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l860_86029


namespace odds_against_y_winning_l860_86019

/- 
   Define the conditions: 
   odds_w: odds against W winning is 4:1
   odds_x: odds against X winning is 5:3
-/
def odds_w : ℚ := 4 / 1
def odds_x : ℚ := 5 / 3

/- 
   Calculate the odds against Y winning 
-/
theorem odds_against_y_winning : 
  (4 / (4 + 1)) + (5 / (5 + 3)) < 1 ∧
  (1 - ((4 / (4 + 1)) + (5 / (5 + 3)))) = 17 / 40 ∧
  ((1 - (17 / 40)) / (17 / 40)) = 23 / 17 := by
  sorry

end odds_against_y_winning_l860_86019


namespace solve_triangle_l860_86087

variable {A B C : ℝ}
variable {a b c : ℝ}

noncomputable def sin_B_plus_pi_four (a b c : ℝ) : ℝ :=
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  let sin_B := Real.sqrt (1 - cos_B^2)
  sin_B * Real.sqrt 2 / 2 + cos_B * Real.sqrt 2 / 2

theorem solve_triangle 
  (a b c : ℝ)
  (h1 : b = 2 * Real.sqrt 5)
  (h2 : c = 3)
  (h3 : 3 * a * (a^2 + b^2 - c^2) / (2 * a * b) = 2 * c * (b^2 + c^2 - a^2) / (2 * b * c)) :
  a = Real.sqrt 5 ∧ 
  sin_B_plus_pi_four a b c = Real.sqrt 10 / 10 :=
by 
  sorry

end solve_triangle_l860_86087


namespace value_of_m_l860_86028

theorem value_of_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 8) (h2 : x1 = 3 * x2) : m = 12 :=
by
  -- Proof will be provided here
  sorry

end value_of_m_l860_86028


namespace ratio_of_votes_l860_86034

theorem ratio_of_votes (up_votes down_votes : ℕ) (h_up : up_votes = 18) (h_down : down_votes = 4) : (up_votes / Nat.gcd up_votes down_votes) = 9 ∧ (down_votes / Nat.gcd up_votes down_votes) = 2 :=
by
  sorry

end ratio_of_votes_l860_86034


namespace intersection_M_P_l860_86036

def M : Set ℝ := {0, 1, 2, 3}
def P : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_P : M ∩ P = {0, 1} := 
by
  -- You can fill in the proof here
  sorry

end intersection_M_P_l860_86036


namespace derivative_y_eq_l860_86080

noncomputable def y (x : ℝ) : ℝ := 
  (3 / 2) * Real.log (Real.tanh (x / 2)) + Real.cosh x - (Real.cosh x) / (2 * (Real.sinh x)^2)

theorem derivative_y_eq :
  (deriv y x) = (Real.cosh x)^4 / (Real.sinh x)^3 :=
sorry

end derivative_y_eq_l860_86080


namespace inverse_of_11_mod_1021_l860_86088

theorem inverse_of_11_mod_1021 : ∃ x : ℕ, x < 1021 ∧ 11 * x ≡ 1 [MOD 1021] := by
  use 557
  -- We leave the proof as an exercise.
  sorry

end inverse_of_11_mod_1021_l860_86088


namespace remainder_29_169_1990_mod_11_l860_86070

theorem remainder_29_169_1990_mod_11 :
  (29 * 169 ^ 1990) % 11 = 7 :=
by
  sorry

end remainder_29_169_1990_mod_11_l860_86070


namespace x_varies_as_z_raised_to_n_power_l860_86061

noncomputable def x_varies_as_cube_of_y (k y : ℝ) : ℝ := k * y ^ 3
noncomputable def y_varies_as_cube_root_of_z (j z : ℝ) : ℝ := j * z ^ (1/3 : ℝ)

theorem x_varies_as_z_raised_to_n_power (k j z : ℝ) :
  ∃ n : ℝ, x_varies_as_cube_of_y k (y_varies_as_cube_root_of_z j z) = (k * j^3) * z ^ n ∧ n = 1 :=
by
  sorry

end x_varies_as_z_raised_to_n_power_l860_86061


namespace bicycle_profit_theorem_l860_86037

def bicycle_profit_problem : Prop :=
  let CP_A : ℝ := 120
  let SP_C : ℝ := 225
  let profit_percentage_B : ℝ := 0.25
  -- intermediate calculations
  let CP_B : ℝ := SP_C / (1 + profit_percentage_B)
  let SP_A : ℝ := CP_B
  let Profit_A : ℝ := SP_A - CP_A
  let Profit_Percentage_A : ℝ := (Profit_A / CP_A) * 100
  -- final statement to prove
  Profit_Percentage_A = 50

theorem bicycle_profit_theorem : bicycle_profit_problem := 
by
  sorry

end bicycle_profit_theorem_l860_86037


namespace maximum_side_length_of_triangle_l860_86069

theorem maximum_side_length_of_triangle (a b c : ℕ) (h_diff: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter: a + b + c = 30)
  (h_triangle_inequality_1: a + b > c) 
  (h_triangle_inequality_2: a + c > b) 
  (h_triangle_inequality_3: b + c > a) : 
  c ≤ 14 :=
sorry

end maximum_side_length_of_triangle_l860_86069


namespace fewer_noodles_than_pirates_l860_86056

theorem fewer_noodles_than_pirates 
  (P : ℕ) (N : ℕ) (h1 : P = 45) (h2 : N + P = 83) : P - N = 7 := by 
  sorry

end fewer_noodles_than_pirates_l860_86056


namespace max_10a_3b_15c_l860_86038

theorem max_10a_3b_15c (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) : 
  10 * a + 3 * b + 15 * c ≤ (Real.sqrt 337) / 6 := 
sorry

end max_10a_3b_15c_l860_86038


namespace cost_of_acai_berry_juice_l860_86011

theorem cost_of_acai_berry_juice (cost_per_litre_cocktail : ℝ)
                                 (cost_per_litre_fruit_juice : ℝ)
                                 (litres_fruit_juice : ℝ)
                                 (litres_acai_juice : ℝ)
                                 (total_cost_cocktail : ℝ)
                                 (cost_per_litre_acai : ℝ) :
  cost_per_litre_cocktail = 1399.45 →
  cost_per_litre_fruit_juice = 262.85 →
  litres_fruit_juice = 34 →
  litres_acai_juice = 22.666666666666668 →
  total_cost_cocktail = (34 + 22.666666666666668) * 1399.45 →
  (litres_fruit_juice * cost_per_litre_fruit_juice + litres_acai_juice * cost_per_litre_acai) = total_cost_cocktail →
  cost_per_litre_acai = 3106.66666666666666 :=
by
  intros
  sorry

end cost_of_acai_berry_juice_l860_86011


namespace Archer_catch_total_fish_l860_86078

noncomputable def ArcherFishProblem : ℕ :=
  let firstRound := 8
  let secondRound := firstRound + 12
  let thirdRound := secondRound + (secondRound * 60 / 100)
  firstRound + secondRound + thirdRound

theorem Archer_catch_total_fish : ArcherFishProblem = 60 := by
  sorry

end Archer_catch_total_fish_l860_86078


namespace lcm_of_40_90_150_l860_86063

-- Definition to calculate the Least Common Multiple of three numbers
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Definitions for the given numbers
def n1 : ℕ := 40
def n2 : ℕ := 90
def n3 : ℕ := 150

-- The statement of the proof problem
theorem lcm_of_40_90_150 : lcm3 n1 n2 n3 = 1800 := by
  sorry

end lcm_of_40_90_150_l860_86063


namespace system_of_equations_has_no_solution_l860_86001

theorem system_of_equations_has_no_solution
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y + z = 10)
  (h2 : 6 * x - 8 * y + 2 * z = 16)
  (h3 : x + y - z = 3) :
  false :=
by 
  sorry

end system_of_equations_has_no_solution_l860_86001


namespace geometric_sequence_common_ratio_l860_86068

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) (S_n : ℕ → ℝ)
  (h₁ : S_n 3 = a₁ + a₁ * q + a₁ * q ^ 2)
  (h₂ : S_n 2 = a₁ + a₁ * q)
  (h₃ : S_n 3 / S_n 2 = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l860_86068


namespace find_plane_through_points_and_perpendicular_l860_86072

-- Definitions for points and plane conditions
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def point1 : Point3D := ⟨2, -2, 2⟩
def point2 : Point3D := ⟨0, 2, -1⟩

def normal_vector_of_given_plane : Point3D := ⟨2, -1, 2⟩

-- Lean 4 statement
theorem find_plane_through_points_and_perpendicular :
  ∃ (A B C D : ℤ), 
  (∀ (p : Point3D), (p = point1 ∨ p = point2) → A * p.x + B * p.y + C * p.z + D = 0) ∧
  (A * 2 + B * -1 + C * 2 = 0) ∧ 
  A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (A = 5 ∧ B = -2 ∧ C = 6 ∧ D = -26) :=
by
  sorry

end find_plane_through_points_and_perpendicular_l860_86072


namespace odd_n_divides_3n_plus_1_is_1_l860_86032

theorem odd_n_divides_3n_plus_1_is_1 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) (h3 : n ∣ 3^n + 1) : n = 1 :=
sorry

end odd_n_divides_3n_plus_1_is_1_l860_86032


namespace mat_weaves_problem_l860_86017

theorem mat_weaves_problem (S1 S2: ℕ) (days1 days2: ℕ) (mats1 mats2: ℕ) (H1: S1 = 1)
    (H2: S2 = 8) (H3: days1 = 4) (H4: days2 = 8) (H5: mats1 = 4) (H6: mats2 = 16) 
    (rate_consistency: (mats1 / days1) = (mats2 / days2 / S2)): S1 = 4 := 
by
  sorry

end mat_weaves_problem_l860_86017


namespace diamond_value_l860_86044

variable {a b : ℤ}

-- Define the operation diamond following the given condition.
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Define the conditions given in the problem.
axiom h1 : a + b = 10
axiom h2 : a * b = 24

-- State the target theorem.
theorem diamond_value : diamond a b = 5 / 12 :=
by
  sorry

end diamond_value_l860_86044


namespace expand_and_count_nonzero_terms_l860_86023

theorem expand_and_count_nonzero_terms (x : ℝ) : 
  (x-3)*(3*x^2-2*x+6) + 2*(x^3 + x^2 - 4*x) = 5*x^3 - 9*x^2 + 4*x - 18 ∧ 
  (5 ≠ 0 ∧ -9 ≠ 0 ∧ 4 ≠ 0 ∧ -18 ≠ 0) :=
sorry

end expand_and_count_nonzero_terms_l860_86023


namespace product_possible_values_l860_86053

theorem product_possible_values (N L M M_5: ℤ) :
  M = L + N → 
  M_5 = M - 8 → 
  ∃ L_5, L_5 = L + 5 ∧ |M_5 - L_5| = 6 →
  N = 19 ∨ N = 7 → 19 * 7 = 133 :=
by {
  sorry
}

end product_possible_values_l860_86053


namespace complex_magnitude_sixth_power_l860_86048

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end complex_magnitude_sixth_power_l860_86048


namespace jumping_bug_ways_l860_86095

-- Define the problem with given conditions and required answer
theorem jumping_bug_ways :
  let starting_position := 0
  let ending_position := 3
  let jumps := 5
  let jump_options := [1, -1]
  (∃ (jump_seq : Fin jumps → ℤ), (∀ i, jump_seq i ∈ jump_options ∧ (List.sum (List.ofFn jump_seq) = ending_position)) ∧
  (List.count (-1) (List.ofFn jump_seq) = 1)) →
  (∃ n : ℕ, n = 5) :=
by
  sorry  -- Proof to be completed

end jumping_bug_ways_l860_86095


namespace total_duration_in_seconds_l860_86099

theorem total_duration_in_seconds :
  let hours_in_seconds := 2 * 3600
  let minutes_in_seconds := 45 * 60
  let extra_seconds := 30
  hours_in_seconds + minutes_in_seconds + extra_seconds = 9930 := by
  sorry

end total_duration_in_seconds_l860_86099


namespace slope_of_line_l860_86075

theorem slope_of_line (x : ℝ) : (2 * x + 1) = 2 :=
by sorry

end slope_of_line_l860_86075


namespace prime_divides_30_l860_86018

theorem prime_divides_30 (p : ℕ) (h_prime : Prime p) (h_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
  sorry

end prime_divides_30_l860_86018


namespace cricket_team_matches_played_in_august_l860_86009

theorem cricket_team_matches_played_in_august
    (M : ℕ)
    (h1 : ∃ W : ℕ, W = 24 * M / 100)
    (h2 : ∃ W : ℕ, W + 70 = 52 * (M + 70) / 100) :
    M = 120 :=
sorry

end cricket_team_matches_played_in_august_l860_86009


namespace unique_solution_for_a_l860_86051

theorem unique_solution_for_a (a : ℝ) :
  (∃! x : ℝ, 2 ^ |2 * x - 2| - a * Real.cos (1 - x) = 0) ↔ a = 1 :=
sorry

end unique_solution_for_a_l860_86051


namespace find_FC_l860_86004

theorem find_FC (DC : ℝ) (CB : ℝ) (AB AD ED FC : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (h3 : AB = (1/3) * AD) 
  (h4 : ED = (3/4) * AD) 
  (h5 : FC = 14.625) : FC = 14.625 :=
by sorry

end find_FC_l860_86004


namespace monomials_exponents_l860_86031

theorem monomials_exponents (m n : ℕ) 
  (h₁ : 3 * x ^ 5 * y ^ m + -2 * x ^ n * y ^ 7 = 0) : m - n = 2 := 
by
  sorry

end monomials_exponents_l860_86031


namespace find_f_of_2_l860_86033

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x) = (1 + x) / x) : f 2 = 3 :=
sorry

end find_f_of_2_l860_86033


namespace universal_proposition_is_B_l860_86002

theorem universal_proposition_is_B :
  (∀ n : ℤ, (2 * n % 2 = 0)) = True :=
sorry

end universal_proposition_is_B_l860_86002


namespace solution_l860_86030

theorem solution (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) 
  : x * y * z = 8 := 
by sorry

end solution_l860_86030


namespace residue_neg_437_mod_13_l860_86012

theorem residue_neg_437_mod_13 : (-437) % 13 = 5 :=
by
  sorry

end residue_neg_437_mod_13_l860_86012


namespace compute_a_sq_sub_b_sq_l860_86090

variables {a b : (ℝ × ℝ)}

-- Conditions
axiom a_nonzero : a ≠ (0, 0)
axiom b_nonzero : b ≠ (0, 0)
axiom a_add_b_eq_neg3_6 : a + b = (-3, 6)
axiom a_sub_b_eq_neg3_2 : a - b = (-3, 2)

-- Question and the correct answer
theorem compute_a_sq_sub_b_sq : (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 :=
by sorry

end compute_a_sq_sub_b_sq_l860_86090


namespace quadratics_common_root_square_sum_6_l860_86077

theorem quadratics_common_root_square_sum_6
  (a b c : ℝ)
  (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_common_root_1: ∃ x1, x1^2 + a * x1 + b = 0 ∧ x1^2 + b * x1 + c = 0)
  (h_common_root_2: ∃ x2, x2^2 + b * x2 + c = 0 ∧ x2^2 + c * x2 + a = 0)
  (h_common_root_3: ∃ x3, x3^2 + c * x3 + a = 0 ∧ x3^2 + a * x3 + b = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratics_common_root_square_sum_6_l860_86077


namespace evaluate_expression_at_values_l860_86046

theorem evaluate_expression_at_values :
  let x := 2
  let y := -1
  let z := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
    sorry

end evaluate_expression_at_values_l860_86046


namespace initial_earning_members_l860_86007

theorem initial_earning_members (n : ℕ) (h1 : (n * 735) - ((n - 1) * 650) = 905) : n = 3 := by
  sorry

end initial_earning_members_l860_86007


namespace minimum_value_is_1297_l860_86015

noncomputable def find_minimum_value (a b c n : ℕ) : ℕ :=
  if (a + b ≠ b + c) ∧ (b + c ≠ c + a) ∧ (a + b ≠ c + a) ∧
     ((a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
      (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
      (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) then
    a^2 + b^2 + c^2
  else
    0

theorem minimum_value_is_1297 (a b c n : ℕ) :
  a ≠ b → b ≠ c → c ≠ a → (∃ a b c n, (a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
                                  (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
                                  (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) →
  (∃ a b c, a^2 + b^2 + c^2 = 1297) :=
by sorry

end minimum_value_is_1297_l860_86015


namespace smallest_sum_of_squares_l860_86060

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := 
  sorry

end smallest_sum_of_squares_l860_86060


namespace sum_of_possible_x_l860_86021

theorem sum_of_possible_x 
  (x : ℝ)
  (squareSide : ℝ) 
  (rectangleLength : ℝ) 
  (rectangleWidth : ℝ) 
  (areaCondition : (rectangleLength * rectangleWidth) = 3 * (squareSide ^ 2)) : 
  6 + 6.5 = 12.5 := 
by 
  sorry

end sum_of_possible_x_l860_86021


namespace value_of_p_l860_86067

theorem value_of_p (a : ℕ → ℚ) (m : ℕ) (p : ℚ)
  (h1 : a 1 = 111)
  (h2 : a 2 = 217)
  (h3 : ∀ n : ℕ, 3 ≤ n ∧ n ≤ m → a n = a (n - 2) - (n - p) / a (n - 1))
  (h4 : m = 220) :
  p = 110 / 109 :=
by
  sorry

end value_of_p_l860_86067


namespace last_two_digits_of_power_sequence_l860_86098

noncomputable def power_sequence (n : ℕ) : ℤ :=
  (Int.sqrt 29 + Int.sqrt 21)^(2 * n) + (Int.sqrt 29 - Int.sqrt 21)^(2 * n)

theorem last_two_digits_of_power_sequence :
  (power_sequence 992) % 100 = 71 := by
  sorry

end last_two_digits_of_power_sequence_l860_86098


namespace integral_sign_negative_l860_86096

open Topology

-- Define the problem
theorem integral_sign_negative {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_lt : ∀ x ∈ Set.Icc a b, f x < 0) (h_ab : a < b) :
  ∫ x in a..b, f x < 0 := 
sorry

end integral_sign_negative_l860_86096


namespace trajectory_of_point_l860_86045

/-- 
  Given points A and B on the coordinate plane, with |AB|=2, 
  and a moving point P such that the sum of the distances from P
  to points A and B is constantly 2, the trajectory of point P 
  is the line segment AB. 
-/
theorem trajectory_of_point (A B P : ℝ × ℝ) 
  (h_AB : dist A B = 2) 
  (h_sum : dist P A + dist P B = 2) :
  P ∈ segment ℝ A B :=
sorry

end trajectory_of_point_l860_86045


namespace union_of_A_and_B_l860_86049

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end union_of_A_and_B_l860_86049


namespace books_a_count_l860_86074

theorem books_a_count (A B : ℕ) (h1 : A + B = 20) (h2 : A = B + 4) : A = 12 :=
by
  sorry

end books_a_count_l860_86074


namespace range_of_a_l860_86089

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * a * x - 2 else x + 36 / x - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 2 a) ↔ (2 ≤ a ∧ a ≤ 5) :=
sorry

end range_of_a_l860_86089


namespace solve_for_x_l860_86073

theorem solve_for_x : (1 / 3 - 1 / 4) * 2 = 1 / 6 :=
by
  -- Sorry is used to skip the proof; the proof steps are not included.
  sorry

end solve_for_x_l860_86073


namespace joker_probability_l860_86005

-- Definition of the problem parameters according to the conditions
def total_cards := 54
def jokers := 2

-- Calculate the probability
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

-- State the theorem that we want to prove
theorem joker_probability : probability jokers total_cards = 1 / 27 := by
  sorry

end joker_probability_l860_86005


namespace Q_subset_P_l860_86024

def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | x^2 < 4 }

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l860_86024


namespace magnitude_of_parallel_vector_l860_86085

theorem magnitude_of_parallel_vector {x : ℝ} 
  (h_parallel : 2 / x = -1 / 3) : 
  (Real.sqrt (x^2 + 3^2)) = 3 * Real.sqrt 5 := 
sorry

end magnitude_of_parallel_vector_l860_86085


namespace fill_bucket_completely_l860_86064

theorem fill_bucket_completely (t : ℕ) : (2/3 : ℚ) * t = 100 → t = 150 :=
by
  intro h
  sorry

end fill_bucket_completely_l860_86064


namespace gcd_fx_x_l860_86065

-- Let x be an instance of ℤ
variable (x : ℤ)

-- Define that x is a multiple of 46200
def is_multiple_of_46200 := ∃ k : ℤ, x = 46200 * k

-- Define the function f(x) = (3x + 5)(5x + 3)(11x + 6)(x + 11)
def f (x : ℤ) := (3 * x + 5) * (5 * x + 3) * (11 * x + 6) * (x + 11)

-- The statement to prove
theorem gcd_fx_x (h : is_multiple_of_46200 x) : Int.gcd (f x) x = 990 := 
by
  -- Placeholder for the proof
  sorry

end gcd_fx_x_l860_86065


namespace solve_for_n_l860_86097

-- Define the equation as a Lean expression
def equation (n : ℚ) : Prop :=
  (2 - n) / (n + 1) + (2 * n - 4) / (2 - n) = 1

theorem solve_for_n : ∃ n : ℚ, equation n ∧ n = -1 / 4 := by
  sorry

end solve_for_n_l860_86097


namespace find_percentage_l860_86058

theorem find_percentage (P : ℕ) (h1 : P * 64 = 320 * 10) : P = 5 := 
  by
  sorry

end find_percentage_l860_86058


namespace units_digit_13_pow_2003_l860_86054

theorem units_digit_13_pow_2003 : (13 ^ 2003) % 10 = 7 := by
  sorry

end units_digit_13_pow_2003_l860_86054


namespace foci_equality_ellipse_hyperbola_l860_86003

theorem foci_equality_ellipse_hyperbola (m : ℝ) (h : m > 0) 
  (hl: ∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (4 - m^2)) 
  (hh: ∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (m^2 + 2)) : 
  m = 1 :=
by {
  sorry
}

end foci_equality_ellipse_hyperbola_l860_86003


namespace lily_of_the_valley_bushes_needed_l860_86000

theorem lily_of_the_valley_bushes_needed 
  (r l : ℕ) (h_radius : r = 20) (h_length : l = 400) : 
  l / (2 * r) = 10 := 
by 
  sorry

end lily_of_the_valley_bushes_needed_l860_86000


namespace circle_radius_eq_five_l860_86052

theorem circle_radius_eq_five : 
  ∀ (x y : ℝ), (x^2 + y^2 - 6 * x + 8 * y = 0) → (∃ r : ℝ, ((x - 3)^2 + (y + 4)^2 = r^2) ∧ r = 5) :=
by
  sorry

end circle_radius_eq_five_l860_86052


namespace bananas_to_oranges_equivalence_l860_86071

noncomputable def bananas_to_apples (bananas apples : ℕ) : Prop :=
  4 * apples = 3 * bananas

noncomputable def apples_to_oranges (apples oranges : ℕ) : Prop :=
  5 * oranges = 2 * apples

theorem bananas_to_oranges_equivalence (x y : ℕ) (hx : bananas_to_apples 24 x) (hy : apples_to_oranges x y) :
  y = 72 / 10 := by
  sorry

end bananas_to_oranges_equivalence_l860_86071


namespace ellipse_hyperbola_tangent_l860_86008

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end ellipse_hyperbola_tangent_l860_86008


namespace no_such_n_exists_l860_86066

theorem no_such_n_exists : ∀ (n : ℕ), n ≥ 1 → ¬ Prime (n^n - 4 * n + 3) :=
by
  intro n hn
  sorry

end no_such_n_exists_l860_86066


namespace complex_number_a_eq_1_l860_86013

theorem complex_number_a_eq_1 
  (a : ℝ) 
  (h : ∃ b : ℝ, (a - b * I) / (1 + I) = 0 + b * I) : 
  a = 1 := 
sorry

end complex_number_a_eq_1_l860_86013


namespace correct_operation_l860_86014

theorem correct_operation (x y : ℝ) : (-x - y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 :=
sorry

end correct_operation_l860_86014


namespace final_temperature_l860_86086

variable (initial_temp : ℝ := 40)
variable (double_temp : ℝ := initial_temp * 2)
variable (reduce_by_dad : ℝ := double_temp - 30)
variable (reduce_by_mother : ℝ := reduce_by_dad * 0.70)
variable (increase_by_sister : ℝ := reduce_by_mother + 24)

theorem final_temperature : increase_by_sister = 59 := by
  sorry

end final_temperature_l860_86086


namespace randy_total_trees_l860_86043

def mango_trees : ℕ := 60
def coconut_trees : ℕ := mango_trees / 2 - 5
def total_trees (mangos coconuts : ℕ) : ℕ := mangos + coconuts

theorem randy_total_trees : total_trees mango_trees coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l860_86043


namespace same_side_of_line_l860_86010

theorem same_side_of_line (a : ℝ) :
    let point1 := (3, -1)
    let point2 := (-4, -3)
    let line_eq (x y : ℝ) := 3 * x - 2 * y + a
    (line_eq point1.1 point1.2) * (line_eq point2.1 point2.2) > 0 ↔
        (a < -11 ∨ a > 6) := sorry

end same_side_of_line_l860_86010


namespace total_weight_of_onions_l860_86026

def weight_per_bag : ℕ := 50
def bags_per_trip : ℕ := 10
def trips : ℕ := 20

theorem total_weight_of_onions : bags_per_trip * weight_per_bag * trips = 10000 := by
  sorry

end total_weight_of_onions_l860_86026


namespace An_integer_and_parity_l860_86055

theorem An_integer_and_parity (k : Nat) (h : k > 0) : 
  ∀ n ≥ 1, ∃ A : Nat, 
   (A = 1 ∨ (∀ A' : Nat, A' = ( (A * n + 2 * (n+1) ^ (2 * k)) / (n+2)))) 
  ∧ (A % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 2) := 
by 
  sorry

end An_integer_and_parity_l860_86055


namespace abs_difference_21st_term_l860_86057

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 - 14 * (n - 1)

theorem abs_difference_21st_term :
  |sequence_C 21 - sequence_D 21| = 520 := by
  sorry

end abs_difference_21st_term_l860_86057


namespace nth_number_eq_l860_86022

noncomputable def nth_number (n : Nat) : ℚ := n / (n^2 + 1)

theorem nth_number_eq (n : Nat) : nth_number n = n / (n^2 + 1) :=
by
  sorry

end nth_number_eq_l860_86022


namespace quadratic_minimum_value_proof_l860_86042

-- Define the quadratic function and its properties
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the condition that the coefficient of the squared term is positive
def coefficient_positive : Prop := (2 : ℝ) > 0

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : Prop := h = 3

-- Define the minimum value of the quadratic function
def minimum_value (y_min : ℝ) : Prop := ∀ x : ℝ, y_min ≤ quadratic_function x 

-- Define the correct answer choice
def correct_answer : Prop := minimum_value 2

-- The theorem stating the proof problem
theorem quadratic_minimum_value_proof :
  coefficient_positive ∧ axis_of_symmetry 3 → correct_answer :=
sorry

end quadratic_minimum_value_proof_l860_86042


namespace xy_z_eq_inv_sqrt2_l860_86079

noncomputable def f (t : ℝ) : ℝ := (Real.sqrt 2) * t + 1 / ((Real.sqrt 2) * t)

theorem xy_z_eq_inv_sqrt2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (Real.sqrt 2) * x + 1 / ((Real.sqrt 2) * x) 
      + (Real.sqrt 2) * y + 1 / ((Real.sqrt 2) * y) 
      + (Real.sqrt 2) * z + 1 / ((Real.sqrt 2) * z) 
      = 6 - 2 * (Real.sqrt (2 * x)) * abs (y - z) 
            - (Real.sqrt (2 * y)) * (x - z) ^ 2 
            - (Real.sqrt (2 * z)) * (Real.sqrt (abs (x - y)))) :
  x = y ∧ y = z ∧ z = 1 / (Real.sqrt 2) :=
sorry

end xy_z_eq_inv_sqrt2_l860_86079


namespace miles_driven_before_gas_stop_l860_86006

def total_distance : ℕ := 78
def distance_left : ℕ := 46

theorem miles_driven_before_gas_stop : total_distance - distance_left = 32 := by
  sorry

end miles_driven_before_gas_stop_l860_86006


namespace probability_roll_2_four_times_in_five_rolls_l860_86050

theorem probability_roll_2_four_times_in_five_rolls :
  (∃ (prob_roll_2 : ℚ) (prob_not_roll_2 : ℚ), 
   prob_roll_2 = 1/6 ∧ prob_not_roll_2 = 5/6 ∧ 
   (5 * prob_roll_2^4 * prob_not_roll_2 = 5/72)) :=
sorry

end probability_roll_2_four_times_in_five_rolls_l860_86050


namespace systematic_sampling_correct_l860_86039

-- Define the conditions for the problem
def num_employees : ℕ := 840
def num_selected : ℕ := 42
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Define systematic sampling interval
def sampling_interval := num_employees / num_selected

-- Define the length of the given interval
def interval_length := interval_end - interval_start + 1

-- The theorem to prove
theorem systematic_sampling_correct :
  (interval_length / sampling_interval) = 12 := sorry

end systematic_sampling_correct_l860_86039


namespace square_ratios_l860_86081

/-- 
  Given two squares with areas ratio 16:49, 
  prove that the ratio of their perimeters is 4:7,
  and the ratio of the sum of their perimeters to the sum of their areas is 84:13.
-/
theorem square_ratios (s₁ s₂ : ℝ) 
  (h₁ : s₁^2 / s₂^2 = 16 / 49) :
  (s₁ / s₂ = 4 / 7) ∧ ((4 * (s₁ + s₂)) / (s₁^2 + s₂^2) = 84 / 13) :=
by {
  sorry
}

end square_ratios_l860_86081


namespace abc_inequality_l860_86094

theorem abc_inequality 
  (a b c : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : 0 < c) 
  (h4 : a * b * c = 1) 
  : 
  (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) := 
by 
  sorry

end abc_inequality_l860_86094


namespace range_of_m_l860_86059

noncomputable def f (x : ℝ) : ℝ := -x^3 + 6 * x^2 - 9 * x

def tangents_condition (m : ℝ) : Prop := ∃ x : ℝ, (-3 * x^2 + 12 * x - 9) * (x + 1) + m = -x^3 + 6 * x^2 - 9 * x

theorem range_of_m (m : ℝ) : tangents_condition m → -11 < m ∧ m < 16 :=
sorry

end range_of_m_l860_86059


namespace yanna_gave_100_l860_86040

/--
Yanna buys 10 shirts at $5 each and 3 pairs of sandals at $3 each, 
and she receives $41 in change. Prove that she gave $100.
-/
theorem yanna_gave_100 :
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  total_cost + change = 100 :=
by
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  show total_cost + change = 100
  sorry

end yanna_gave_100_l860_86040


namespace Maria_waist_size_correct_l860_86091

noncomputable def waist_size_mm (waist_size_in : ℕ) (mm_per_ft : ℝ) (in_per_ft : ℕ) : ℝ :=
  (waist_size_in : ℝ) / (in_per_ft : ℝ) * mm_per_ft

theorem Maria_waist_size_correct :
  let waist_size_in := 27
  let mm_per_ft := 305
  let in_per_ft := 12
  waist_size_mm waist_size_in mm_per_ft in_per_ft = 686.3 :=
by
  sorry

end Maria_waist_size_correct_l860_86091


namespace fraction_to_decimal_terminating_l860_86093

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l860_86093


namespace roger_steps_to_minutes_l860_86027

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end roger_steps_to_minutes_l860_86027


namespace find_divisor_l860_86092

theorem find_divisor : ∃ D : ℕ, 14698 = (D * 89) + 14 ∧ D = 165 :=
by
  use 165
  sorry

end find_divisor_l860_86092


namespace cube_root_simplification_l860_86020

theorem cube_root_simplification {a b : ℕ} (h : (a * b^(1/3) : ℝ) = (2450 : ℝ)^(1/3)) 
  (a_pos : 0 < a) (b_pos : 0 < b) (h_smallest : ∀ b', 0 < b' → (∃ a', (a' * b'^(1/3) : ℝ) = (2450 : ℝ)^(1/3) → b ≤ b')) :
  a + b = 37 := 
sorry

end cube_root_simplification_l860_86020


namespace number_of_tie_games_l860_86076

def total_games (n_teams: ℕ) (games_per_matchup: ℕ) : ℕ :=
  (n_teams * (n_teams - 1) / 2) * games_per_matchup

def theoretical_max_points (total_games: ℕ) (points_per_win: ℕ): ℕ :=
  total_games * points_per_win

def actual_total_points (lions: ℕ) (tigers: ℕ) (mounties: ℕ) (royals: ℕ): ℕ :=
  lions + tigers + mounties + royals

def tie_games (theoretical_points: ℕ) (actual_points: ℕ) (points_per_tie: ℕ): ℕ :=
  (theoretical_points - actual_points) / points_per_tie

theorem number_of_tie_games
  (n_teams: ℕ)
  (games_per_matchup: ℕ)
  (points_per_win: ℕ)
  (points_per_tie: ℕ)
  (lions: ℕ)
  (tigers: ℕ)
  (mounties: ℕ)
  (royals: ℕ)
  (h_teams: n_teams = 4)
  (h_games: games_per_matchup = 4)
  (h_points_win: points_per_win = 3)
  (h_points_tie: points_per_tie = 2)
  (h_lions: lions = 22)
  (h_tigers: tigers = 19)
  (h_mounties: mounties = 14)
  (h_royals: royals = 12) :
  tie_games (theoretical_max_points (total_games n_teams games_per_matchup) points_per_win) 
  (actual_total_points lions tigers mounties royals) points_per_tie = 5 :=
by
  rw [h_teams, h_games, h_points_win, h_points_tie, h_lions, h_tigers, h_mounties, h_royals]
  simp [total_games, theoretical_max_points, actual_total_points, tie_games]
  sorry

end number_of_tie_games_l860_86076


namespace ratio_of_first_term_to_common_difference_l860_86083

theorem ratio_of_first_term_to_common_difference (a d : ℕ) (h : 15 * a + 105 * d = 3 * (5 * a + 10 * d)) : a = 5 * d :=
by
  sorry

end ratio_of_first_term_to_common_difference_l860_86083


namespace range_of_x_l860_86016

theorem range_of_x (x : ℝ) : (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_x_l860_86016


namespace justin_and_tim_play_same_game_210_times_l860_86047

def number_of_games_with_justin_and_tim : ℕ :=
  have num_players : ℕ := 12
  have game_size : ℕ := 6
  have justin_and_tim_fixed : ℕ := 2
  have remaining_players : ℕ := num_players - justin_and_tim_fixed
  have players_to_choose : ℕ := game_size - justin_and_tim_fixed
  Nat.choose remaining_players players_to_choose

theorem justin_and_tim_play_same_game_210_times :
  number_of_games_with_justin_and_tim = 210 :=
by sorry

end justin_and_tim_play_same_game_210_times_l860_86047


namespace fourth_person_height_l860_86035

theorem fourth_person_height 
  (h : ℝ)
  (height_average : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79)
  : h + 10 = 85 := 
by
  sorry

end fourth_person_height_l860_86035
