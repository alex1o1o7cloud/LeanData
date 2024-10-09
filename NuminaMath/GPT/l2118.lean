import Mathlib

namespace michael_lost_at_least_800_l2118_211849

theorem michael_lost_at_least_800 
  (T F : ℕ) 
  (h1 : T + F = 15) 
  (h2 : T = F + 1 ∨ T = F - 1) 
  (h3 : 10 * T + 50 * F = 1270) : 
  1270 - (10 * T + 50 * F) = 800 :=
by
  sorry

end michael_lost_at_least_800_l2118_211849


namespace calc_101_cubed_expression_l2118_211831

theorem calc_101_cubed_expression : 101^3 + 3 * (101^2) - 3 * 101 + 9 = 1060610 := 
by
  sorry

end calc_101_cubed_expression_l2118_211831


namespace find_a1_and_d_l2118_211866

variable (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) (a5 : ℤ := -1) (a8 : ℤ := 2)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem find_a1_and_d
  (h : arithmetic_sequence a d)
  (h_a5 : a 5 = -1)
  (h_a8 : a 8 = 2) :
  a 1 = -5 ∧ d = 1 :=
by
  sorry

end find_a1_and_d_l2118_211866


namespace equivalent_exponentiation_l2118_211884

theorem equivalent_exponentiation (h : 64 = 8^2) : 8^15 / 64^3 = 8^9 :=
by
  sorry

end equivalent_exponentiation_l2118_211884


namespace relationship_xyz_l2118_211877

theorem relationship_xyz (a b : ℝ) (x y z : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a > b) (hab_sum : a + b = 1) 
  (hx : x = Real.log b / Real.log a)
  (hy : y = Real.log (1 / b) / Real.log a)
  (hz : z = Real.log 3 / Real.log ((1 / a) + (1 / b))) : 
  y < z ∧ z < x := 
sorry

end relationship_xyz_l2118_211877


namespace percentage_hate_german_l2118_211858

def percentage_hate_math : ℝ := 0.01
def percentage_hate_english : ℝ := 0.02
def percentage_hate_french : ℝ := 0.01
def percentage_hate_all_four : ℝ := 0.08

theorem percentage_hate_german : (0.08 - (0.01 + 0.02 + 0.01)) = 0.04 :=
by
  -- Proof goes here
  sorry

end percentage_hate_german_l2118_211858


namespace problem_statement_l2118_211818

noncomputable def m (α : ℝ) : ℝ := - (Real.sqrt 2) / 4

noncomputable def tan_alpha (α : ℝ) : ℝ := 2 * Real.sqrt 2

theorem problem_statement (α : ℝ) (P : (ℝ × ℝ)) (h1 : P = (m α, 1)) (h2 : Real.cos α = - 1 / 3) :
  (P.1 = - (Real.sqrt 2) / 4) ∧ (Real.tan α = 2 * Real.sqrt 2) :=
by
  sorry

end problem_statement_l2118_211818


namespace find_z_add_inv_y_l2118_211886

theorem find_z_add_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) : z + 1 / y = 5 / 27 := by
  sorry

end find_z_add_inv_y_l2118_211886


namespace problem_solution_l2118_211872

theorem problem_solution
  (a b c d : ℕ)
  (h1 : a^6 = b^5)
  (h2 : c^4 = d^3)
  (h3 : c - a = 25) :
  d - b = 561 :=
sorry

end problem_solution_l2118_211872


namespace add_decimals_l2118_211841

theorem add_decimals : 4.3 + 3.88 = 8.18 := 
sorry

end add_decimals_l2118_211841


namespace intersection_coords_perpendicular_line_l2118_211859

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := x + y - 2 = 0

theorem intersection_coords : ∃ P : ℝ × ℝ, line1 P.1 P.2 ∧ line2 P.1 P.2 ∧ P = (1, 1) := by
  sorry

theorem perpendicular_line (x y : ℝ) (P : ℝ × ℝ) (hP: P = (1, 1)) : 
  (line2 P.1 P.2) → x - y = 0 := by
  sorry

end intersection_coords_perpendicular_line_l2118_211859


namespace logical_equivalence_l2118_211857

variables {α : Type} (A B : α → Prop)

theorem logical_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end logical_equivalence_l2118_211857


namespace certain_number_is_120_l2118_211853

theorem certain_number_is_120 : ∃ certain_number : ℤ, 346 * certain_number = 173 * 240 ∧ certain_number = 120 :=
by
  sorry

end certain_number_is_120_l2118_211853


namespace games_given_away_correct_l2118_211854

-- Define initial and remaining games
def initial_games : ℕ := 50
def remaining_games : ℕ := 35

-- Define the number of games given away
def games_given_away : ℕ := initial_games - remaining_games

-- Prove that the number of games given away is 15
theorem games_given_away_correct : games_given_away = 15 := by
  -- This is a placeholder for the actual proof
  sorry

end games_given_away_correct_l2118_211854


namespace convex_n_hedral_angle_l2118_211870

theorem convex_n_hedral_angle (n : ℕ) 
  (sum_plane_angles : ℝ) (sum_dihedral_angles : ℝ) 
  (h1 : sum_plane_angles = sum_dihedral_angles)
  (h2 : sum_plane_angles < 2 * Real.pi)
  (h3 : sum_dihedral_angles > (n - 2) * Real.pi) :
  n = 3 := 
by 
  sorry

end convex_n_hedral_angle_l2118_211870


namespace sequence_eventually_congruent_mod_l2118_211850

theorem sequence_eventually_congruent_mod (n : ℕ) (hn : n ≥ 1) : 
  ∃ N, ∀ m ≥ N, ∃ k, m = k * n + N ∧ (2^N.succ - 2^k) % n = 0 :=
by
  sorry

end sequence_eventually_congruent_mod_l2118_211850


namespace fraction_meaningful_l2118_211891

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_l2118_211891


namespace division_modulus_l2118_211803

-- Definitions using the conditions
def a : ℕ := 8 * (10^9)
def b : ℕ := 4 * (10^4)
def n : ℕ := 10^6

-- Lean statement to prove the problem
theorem division_modulus (a b n : ℕ) (h : a = 8 * (10^9) ∧ b = 4 * (10^4) ∧ n = 10^6) : 
  ((a / b) % n) = 200000 := 
by 
  sorry

end division_modulus_l2118_211803


namespace range_g_l2118_211825

def f (x: ℝ) : ℝ := 4 * x - 3
def g (x: ℝ) : ℝ := f (f (f (f (f x))))

theorem range_g (x: ℝ) (h: 0 ≤ x ∧ x ≤ 3) : -1023 ≤ g x ∧ g x ≤ 2049 :=
by
  sorry

end range_g_l2118_211825


namespace rice_mixed_grain_amount_l2118_211895

theorem rice_mixed_grain_amount (total_rice : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) (proportion : ℚ) 
    (h1 : total_rice = 1536) 
    (h2 : sample_size = 256)
    (h3 : mixed_in_sample = 18)
    (h4 : proportion = mixed_in_sample / sample_size) : 
    total_rice * proportion = 108 :=
  sorry

end rice_mixed_grain_amount_l2118_211895


namespace Joel_contributed_22_toys_l2118_211808

/-
Define the given conditions as separate variables and statements in Lean:
1. Toys collected from friends.
2. Total toys donated.
3. Relationship between Joel's and his sister's toys.
4. Prove that Joel donated 22 toys.
-/

theorem Joel_contributed_22_toys (S : ℕ) (toys_from_friends : ℕ) (total_toys : ℕ) (sisters_toys : ℕ) 
  (h1 : toys_from_friends = 18 + 42 + 2 + 13)
  (h2 : total_toys = 108)
  (h3 : S + 2 * S = total_toys - toys_from_friends)
  (h4 : sisters_toys = S) :
  2 * S = 22 :=
  sorry

end Joel_contributed_22_toys_l2118_211808


namespace avg_decreased_by_one_l2118_211874

noncomputable def avg_decrease (n : ℕ) (average_initial : ℝ) (obs_new : ℝ) : ℝ :=
  (n * average_initial + obs_new) / (n + 1)

theorem avg_decreased_by_one (init_avg : ℝ) (obs_new : ℝ) (num_obs : ℕ)
  (h₁ : num_obs = 6)
  (h₂ : init_avg = 12)
  (h₃ : obs_new = 5) :
  init_avg - avg_decrease num_obs init_avg obs_new = 1 :=
by
  sorry

end avg_decreased_by_one_l2118_211874


namespace inverse_of_73_mod_74_l2118_211855

theorem inverse_of_73_mod_74 :
  73 * 73 ≡ 1 [MOD 74] :=
by
  sorry

end inverse_of_73_mod_74_l2118_211855


namespace benny_turnips_l2118_211889

theorem benny_turnips (M B : ℕ) (h1 : M = 139) (h2 : M = B + 26) : B = 113 := 
by 
  sorry

end benny_turnips_l2118_211889


namespace distance_from_home_to_high_school_l2118_211883

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end distance_from_home_to_high_school_l2118_211883


namespace probability_top_card_is_star_l2118_211876

theorem probability_top_card_is_star :
  let total_cards := 65
  let suits := 5
  let ranks_per_suit := 13
  let star_cards := 13
  (star_cards / total_cards) = 1 / 5 :=
by
  sorry

end probability_top_card_is_star_l2118_211876


namespace complement_union_result_l2118_211834

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3})
variable (hA : A = {1, 2})
variable (hB : B = {2, 3})

theorem complement_union_result : compl A ∪ B = {0, 2, 3} :=
by
  -- Our proof steps would go here
  sorry

end complement_union_result_l2118_211834


namespace initial_roses_in_vase_l2118_211873

theorem initial_roses_in_vase (added_roses current_roses : ℕ) (h1 : added_roses = 8) (h2 : current_roses = 18) : 
  current_roses - added_roses = 10 :=
by
  sorry

end initial_roses_in_vase_l2118_211873


namespace sum_of_digits_B_equals_4_l2118_211878

theorem sum_of_digits_B_equals_4 (A B : ℕ) (N : ℕ) (hN : N = 4444 ^ 4444)
    (hA : A = (N.digits 10).sum) (hB : B = (A.digits 10).sum) :
    (B.digits 10).sum = 4 := by
  sorry

end sum_of_digits_B_equals_4_l2118_211878


namespace cost_of_two_burritos_and_five_quesadillas_l2118_211852

theorem cost_of_two_burritos_and_five_quesadillas
  (b q : ℝ)
  (h1 : b + 4 * q = 3.50)
  (h2 : 4 * b + q = 4.10) :
  2 * b + 5 * q = 5.02 := 
sorry

end cost_of_two_burritos_and_five_quesadillas_l2118_211852


namespace intersection_A_B_l2118_211832

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | abs (x - 2) ≥ 1}
def answer : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_A_B :
  A ∩ B = answer :=
sorry

end intersection_A_B_l2118_211832


namespace panda_on_stilts_height_l2118_211801

theorem panda_on_stilts_height (x : ℕ) (h_A : ℕ) 
  (h1 : h_A = x / 4) -- A Bao's height accounts for 1/4 of initial total height
  (h2 : x - 40 = 3 * h_A) -- After breaking 20 dm off each stilt, the new total height is such that A Bao's height accounts for 1/3 of this new height
  : x = 160 := 
by
  sorry

end panda_on_stilts_height_l2118_211801


namespace possible_degrees_of_remainder_l2118_211842

theorem possible_degrees_of_remainder (p : Polynomial ℝ) :
  ∃ r q : Polynomial ℝ, p = q * (3 * X^3 - 4 * X^2 + 5 * X - 6) + r ∧ r.degree < 3 :=
sorry

end possible_degrees_of_remainder_l2118_211842


namespace math_problem_l2118_211805

noncomputable def triangle_conditions (a b c A B C : ℝ) := 
  (2 * b - c) / a = (Real.cos C) / (Real.cos A) ∧ 
  a = Real.sqrt 5 ∧
  1 / 2 * b * c * (Real.sin A) = Real.sqrt 3 / 2

theorem math_problem (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  A = π / 3 ∧ a + b + c = Real.sqrt 5 + Real.sqrt 11 :=
by
  sorry

end math_problem_l2118_211805


namespace jim_total_payment_l2118_211830

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end jim_total_payment_l2118_211830


namespace find_x_eq_eight_l2118_211840

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l2118_211840


namespace parabola_tangent_circle_radius_l2118_211871

noncomputable def radius_of_tangent_circle : ℝ :=
  let r := 1 / 4
  r

theorem parabola_tangent_circle_radius :
  ∃ (r : ℝ), (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 1 - 4 * r) ∧ r = 1 / 4 :=
by
  use 1 / 4
  sorry

end parabola_tangent_circle_radius_l2118_211871


namespace total_blocks_fell_l2118_211816

-- Definitions based on the conditions
def first_stack_height := 7
def second_stack_height := first_stack_height + 5
def third_stack_height := second_stack_height + 7

def first_stack_fallen_blocks := first_stack_height  -- All blocks fell down
def second_stack_fallen_blocks := second_stack_height - 2  -- 2 blocks left standing
def third_stack_fallen_blocks := third_stack_height - 3  -- 3 blocks left standing

-- Total fallen blocks
def total_fallen_blocks := first_stack_fallen_blocks + second_stack_fallen_blocks + third_stack_fallen_blocks

-- Theorem to prove the total number of fallen blocks
theorem total_blocks_fell : total_fallen_blocks = 33 :=
by
  -- Proof omitted, statement given as required
  sorry

end total_blocks_fell_l2118_211816


namespace triple_layer_area_l2118_211847

theorem triple_layer_area (A B C X Y : ℕ) 
  (h1 : A + B + C = 204) 
  (h2 : 140 = (A + B + C) - X - 2 * Y + X + Y)
  (h3 : X = 24) : 
  Y = 64 := by
  sorry

end triple_layer_area_l2118_211847


namespace ratio_of_ages_l2118_211875

variable (x : Nat) -- The multiple of Marie's age
variable (marco_age marie_age : Nat) -- Marco's and Marie's ages

-- Conditions from (a)
axiom h1 : marie_age = 12
axiom h2 : marco_age = (12 * x) + 1
axiom h3 : marco_age + marie_age = 37

-- Statement to be proved
theorem ratio_of_ages : (marco_age : Nat) / (marie_age : Nat) = (25 / 12) :=
by
  -- Proof steps here
  sorry

end ratio_of_ages_l2118_211875


namespace phyllis_marbles_l2118_211899

theorem phyllis_marbles (num_groups : ℕ) (num_marbles_per_group : ℕ) (h1 : num_groups = 32) (h2 : num_marbles_per_group = 2) : 
  num_groups * num_marbles_per_group = 64 :=
by
  sorry

end phyllis_marbles_l2118_211899


namespace intersection_eq_l2118_211848

open Set

-- Define the sets M and N
def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {-3, -1, 1, 3, 5}

-- The goal is to prove that M ∩ N = {-1, 1, 3}
theorem intersection_eq : M ∩ N = {-1, 1, 3} :=
  sorry

end intersection_eq_l2118_211848


namespace num_div_divided_by_10_l2118_211820

-- Given condition: the number divided by 10 equals 12
def number_divided_by_10_gives_12 (x : ℝ) : Prop :=
  x / 10 = 12

-- Lean statement for the mathematical problem
theorem num_div_divided_by_10 (x : ℝ) (h : number_divided_by_10_gives_12 x) : x = 120 :=
by
  sorry

end num_div_divided_by_10_l2118_211820


namespace hall_length_l2118_211824

theorem hall_length (L : ℝ) (H : ℝ) 
  (h1 : 2 * (L * 15) = 2 * (L * H) + 2 * (15 * H)) 
  (h2 : L * 15 * H = 1687.5) : 
  L = 15 :=
by 
  sorry

end hall_length_l2118_211824


namespace sqrt_12_lt_4_l2118_211851

theorem sqrt_12_lt_4 : Real.sqrt 12 < 4 := sorry

end sqrt_12_lt_4_l2118_211851


namespace sequence_sixth_term_l2118_211807

theorem sequence_sixth_term (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) 
  (h2 : ∀ n :ℕ, n > 0 → a (n + 1) = 2 * a n) 
  (h3 : a 1 = 3) : 
  a 6 = 96 := 
by
  sorry

end sequence_sixth_term_l2118_211807


namespace minimum_value_of_4a_plus_b_l2118_211813

noncomputable def minimum_value (a b : ℝ) :=
  if a > 0 ∧ b > 0 ∧ a^2 + a*b - 3 = 0 then 4*a + b else 0

theorem minimum_value_of_4a_plus_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + a*b - 3 = 0 → 4*a + b ≥ 6 :=
by
  intros a b ha hb hab
  sorry

end minimum_value_of_4a_plus_b_l2118_211813


namespace equation_one_solution_equation_two_solution_l2118_211814

theorem equation_one_solution (x : ℝ) : (6 * x - 7 = 4 * x - 5) ↔ (x = 1) := by
  sorry

theorem equation_two_solution (x : ℝ) : ((x + 1) / 2 - 1 = 2 + (2 - x) / 4) ↔ (x = 4) := by
  sorry

end equation_one_solution_equation_two_solution_l2118_211814


namespace sandy_total_money_l2118_211829

def half_dollar_value := 0.5
def quarter_value := 0.25
def dime_value := 0.1
def nickel_value := 0.05
def dollar_value := 1.0

def monday_total := 12 * half_dollar_value + 5 * quarter_value + 10 * dime_value
def tuesday_total := 8 * half_dollar_value + 15 * quarter_value + 5 * dime_value
def wednesday_total := 3 * dollar_value + 4 * half_dollar_value + 10 * quarter_value + 7 * nickel_value
def thursday_total := 5 * dollar_value + 6 * half_dollar_value + 8 * quarter_value + 5 * dime_value + 12 * nickel_value
def friday_total := 2 * dollar_value + 7 * half_dollar_value + 20 * nickel_value + 25 * dime_value

def total_amount := monday_total + tuesday_total + wednesday_total + thursday_total + friday_total

theorem sandy_total_money : total_amount = 44.45 := by
  sorry

end sandy_total_money_l2118_211829


namespace polynomial_j_value_l2118_211845

noncomputable def polynomial_roots_in_ap (a d : ℝ) : Prop :=
  let r1 := a
  let r2 := a + d
  let r3 := a + 2 * d
  let r4 := a + 3 * d
  ∀ (r : ℝ), r = r1 ∨ r = r2 ∨ r = r3 ∨ r = r4

theorem polynomial_j_value (a d : ℝ) (h_ap : polynomial_roots_in_ap a d)
  (h_poly : ∀ (x : ℝ), (x - (a)) * (x - (a + d)) * (x - (a + 2 * d)) * (x - (a + 3 * d)) = x^4 + j * x^2 + k * x + 256) :
  j = -80 :=
by
  sorry

end polynomial_j_value_l2118_211845


namespace part1_correct_part2_correct_part3_correct_l2118_211865

-- Example survival rates data (provided conditions)
def survivalRatesA : List (Option Float) := [some 95.5, some 92, some 96.5, some 91.6, some 96.3, some 94.6, none, none, none, none]
def survivalRatesB : List (Option Float) := [some 95.1, some 91.6, some 93.2, some 97.8, some 95.6, some 92.3, some 96.6, none, none, none]
def survivalRatesC : List (Option Float) := [some 97, some 95.4, some 98.2, some 93.5, some 94.8, some 95.5, some 94.5, some 93.5, some 98, some 92.5]

-- Define high-quality project condition
def isHighQuality (rate : Float) : Bool := rate > 95.0

-- Problem 1: Probability of two high-quality years from farm B
noncomputable def probabilityTwoHighQualityB : Float := (4.0 * 3.0) / (7.0 * 6.0)

-- Problem 2: Distribution of high-quality projects from farms A, B, and C
structure DistributionX := 
(P0 : Float) -- probability of 0 high-quality years
(P1 : Float) -- probability of 1 high-quality year
(P2 : Float) -- probability of 2 high-quality years
(P3 : Float) -- probability of 3 high-quality years

noncomputable def distributionX : DistributionX := 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
}

-- Problem 3: Inference of average survival rate from high-quality project probabilities
structure AverageSurvivalRates := 
(avgB : Float) 
(avgC : Float)
(probHighQualityB : Float)
(probHighQualityC : Float)
(canInfer : Bool)

noncomputable def avgSurvivalRates : AverageSurvivalRates := 
{ avgB := (95.1 + 91.6 + 93.2 + 97.8 + 95.6 + 92.3 + 96.6) / 7.0,
  avgC := (97 + 95.4 + 98.2 + 93.5 + 94.8 + 95.5 + 94.5 + 93.5 + 98 + 92.5) / 10.0,
  probHighQualityB := 4.0 / 7.0,
  probHighQualityC := 5.0 / 10.0,
  canInfer := false
}

-- Definitions for proof statements indicating correctness
theorem part1_correct : probabilityTwoHighQualityB = (2.0 / 7.0) := sorry

theorem part2_correct : distributionX = 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
} := sorry

theorem part3_correct : avgSurvivalRates.canInfer = false := sorry

end part1_correct_part2_correct_part3_correct_l2118_211865


namespace min_e1_plus_2e2_l2118_211863

noncomputable def e₁ (r : ℝ) : ℝ := 2 / (4 - r)
noncomputable def e₂ (r : ℝ) : ℝ := 2 / (4 + r)

theorem min_e1_plus_2e2 (r : ℝ) (h₀ : 0 < r) (h₂ : r < 2) :
  e₁ r + 2 * e₂ r = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_e1_plus_2e2_l2118_211863


namespace birds_meeting_distance_l2118_211836

theorem birds_meeting_distance :
  ∀ (d distance speed1 speed2: ℕ),
  distance = 20 →
  speed1 = 4 →
  speed2 = 1 →
  (d / speed1) = ((distance - d) / speed2) →
  d = 16 :=
by
  intros d distance speed1 speed2 hdist hspeed1 hspeed2 htime
  sorry

end birds_meeting_distance_l2118_211836


namespace find_f_2023_l2118_211827

noncomputable def f : ℤ → ℤ := sorry

theorem find_f_2023 (h1 : ∀ x : ℤ, f (x+2) + f x = 3) (h2 : f 1 = 0) : f 2023 = 3 := sorry

end find_f_2023_l2118_211827


namespace calc_expression_eq_3_solve_quadratic_eq_l2118_211860

-- Problem 1
theorem calc_expression_eq_3 :
  (-1 : ℝ) ^ 2020 + (- (1 / 2)⁻¹) - (3.14 - Real.pi) ^ 0 + abs (-3) = 3 :=
by
  sorry

-- Problem 2
theorem solve_quadratic_eq {x : ℝ} :
  (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2 / 3) :=
by
  sorry

end calc_expression_eq_3_solve_quadratic_eq_l2118_211860


namespace rise_in_water_level_l2118_211804

theorem rise_in_water_level (edge base_length base_width : ℝ) (cube_volume base_area rise : ℝ) 
  (h₁ : edge = 5) (h₂ : base_length = 10) (h₃ : base_width = 5)
  (h₄ : cube_volume = edge^3) (h₅ : base_area = base_length * base_width) 
  (h₆ : rise = cube_volume / base_area) : 
  rise = 2.5 := 
by 
  -- add proof here 
  sorry

end rise_in_water_level_l2118_211804


namespace cadence_old_company_salary_l2118_211837

variable (S : ℝ)

def oldCompanyMonths : ℝ := 36
def newCompanyMonths : ℝ := 41
def newSalaryMultiplier : ℝ := 1.20
def totalEarnings : ℝ := 426000

theorem cadence_old_company_salary :
  (oldCompanyMonths * S) + (newCompanyMonths * newSalaryMultiplier * S) = totalEarnings → 
  S = 5000 :=
by
  sorry

end cadence_old_company_salary_l2118_211837


namespace boys_belong_to_other_communities_l2118_211862

/-- In a school of 300 boys, if 44% are Muslims, 28% are Hindus, and 10% are Sikhs,
then the number of boys belonging to other communities is 54. -/
theorem boys_belong_to_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℕ)
  (b : total_boys = 300)
  (m : percentage_muslims = 44)
  (h : percentage_hindus = 28)
  (s : percentage_sikhs = 10) :
  total_boys * ((100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 54 := 
sorry

end boys_belong_to_other_communities_l2118_211862


namespace not_rain_probability_l2118_211879

-- Define the probability of rain tomorrow
def prob_rain : ℚ := 3 / 10

-- Define the complementary probability (probability that it will not rain tomorrow)
def prob_no_rain : ℚ := 1 - prob_rain

-- Statement to prove: probability that it will not rain tomorrow equals 7/10 
theorem not_rain_probability : prob_no_rain = 7 / 10 := 
by sorry

end not_rain_probability_l2118_211879


namespace total_fencing_cost_is_correct_l2118_211844

-- Defining the lengths of each side
def length1 : ℝ := 50
def length2 : ℝ := 75
def length3 : ℝ := 60
def length4 : ℝ := 80
def length5 : ℝ := 65

-- Defining the cost per unit length for each side
def cost_per_meter1 : ℝ := 2
def cost_per_meter2 : ℝ := 3
def cost_per_meter3 : ℝ := 4
def cost_per_meter4 : ℝ := 3.5
def cost_per_meter5 : ℝ := 5

-- Calculating the total cost for each side
def cost1 : ℝ := length1 * cost_per_meter1
def cost2 : ℝ := length2 * cost_per_meter2
def cost3 : ℝ := length3 * cost_per_meter3
def cost4 : ℝ := length4 * cost_per_meter4
def cost5 : ℝ := length5 * cost_per_meter5

-- Summing up the total cost for all sides
def total_cost : ℝ := cost1 + cost2 + cost3 + cost4 + cost5

-- The theorem to be proven
theorem total_fencing_cost_is_correct : total_cost = 1170 := by
  sorry

end total_fencing_cost_is_correct_l2118_211844


namespace isosceles_triangle_l2118_211806

noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

variables {A B C : ℝ}
variable (h : sin C = 2 * sin (B + C) * cos B)

theorem isosceles_triangle (h : sin C = 2 * sin (B + C) * cos B) : A = B :=
by
  sorry

end isosceles_triangle_l2118_211806


namespace correlation_highly_related_l2118_211890

-- Conditions:
-- Let corr be the linear correlation coefficient of product output and unit cost.
-- Let rel be the relationship between product output and unit cost.

def corr : ℝ := -0.87

-- Proof Goal:
-- If corr = -0.87, then the relationship is "highly related".

theorem correlation_highly_related (h : corr = -0.87) : rel = "highly related" := by
  sorry

end correlation_highly_related_l2118_211890


namespace geometric_sequence_common_ratio_l2118_211861

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_start : a 1 < 0)
  (h_increasing : ∀ n, a n < a (n + 1)) : 0 < q ∧ q < 1 :=
by
  sorry

end geometric_sequence_common_ratio_l2118_211861


namespace points_on_ellipse_l2118_211819

theorem points_on_ellipse (u : ℝ) :
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  (x^2 / 2 + y^2 / 32 = 1) :=
by
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  sorry

end points_on_ellipse_l2118_211819


namespace Carla_pays_more_than_Bob_l2118_211838

theorem Carla_pays_more_than_Bob
  (slices : ℕ := 12)
  (veg_slices : ℕ := slices / 2)
  (non_veg_slices : ℕ := slices / 2)
  (base_cost : ℝ := 10)
  (extra_cost : ℝ := 3)
  (total_cost : ℝ := base_cost + extra_cost)
  (per_slice_cost : ℝ := total_cost / slices)
  (carla_slices : ℕ := veg_slices + 2)
  (bob_slices : ℕ := 3)
  (carla_payment : ℝ := carla_slices * per_slice_cost)
  (bob_payment : ℝ := bob_slices * per_slice_cost) :
  (carla_payment - bob_payment) = 5.41665 :=
sorry

end Carla_pays_more_than_Bob_l2118_211838


namespace root_line_discriminant_curve_intersection_l2118_211887

theorem root_line_discriminant_curve_intersection (a p q : ℝ) :
  (4 * p^3 + 27 * q^2 = 0) ∧ (ap + q + a^3 = 0) →
  (a = 0 ∧ ∀ p q, 4 * p^3 + 27 * q^2 = 0 → ap + q + a^3 = 0 → (p = 0 ∧ q = 0)) ∨
  (a ≠ 0 ∧ (∃ p1 q1 p2 q2, 
             4 * p1^3 + 27 * q1^2 = 0 ∧ ap + q1 + a^3 = 0 ∧ 
             4 * p2^3 + 27 * q2^2 = 0 ∧ ap + q2 + a^3 = 0 ∧ 
             (p1, q1) ≠ (p2, q2))) := 
sorry

end root_line_discriminant_curve_intersection_l2118_211887


namespace cos_triangle_inequality_l2118_211823

theorem cos_triangle_inequality (α β γ : ℝ) (h_sum : α + β + γ = Real.pi) 
    (h_α : 0 < α) (h_β : 0 < β) (h_γ : 0 < γ) (h_α_lt : α < Real.pi) (h_β_lt : β < Real.pi) (h_γ_lt : γ < Real.pi) : 
    (Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α) ≤ 3 / 4 :=
by
  sorry

end cos_triangle_inequality_l2118_211823


namespace ab_value_l2118_211811

theorem ab_value (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 33) : a * b = 18 := 
by
  sorry

end ab_value_l2118_211811


namespace average_distance_is_600_l2118_211880

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l2118_211880


namespace count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l2118_211892

theorem count_of_numbers_less_than_100_divisible_by_2_but_not_by_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

theorem count_of_numbers_less_than_100_divisible_by_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∨ n % 3 = 0) (Finset.range 100)) = 66 :=
sorry

theorem count_of_numbers_less_than_100_not_divisible_by_either_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 ≠ 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

end count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l2118_211892


namespace find_positive_integer_n_l2118_211846

noncomputable def is_largest_prime_divisor (p n : ℕ) : Prop :=
  (∃ k, n = p * k) ∧ ∀ q, Prime q ∧ q ∣ n → q ≤ p

noncomputable def is_least_prime_divisor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n ∧ ∀ q, Prime q ∧ q ∣ n → p ≤ q

theorem find_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ 
    (∃ p, is_largest_prime_divisor p (n^2 + 3) ∧ is_least_prime_divisor p (n^4 + 6)) ∧
    ∀ m : ℕ, m > 0 ∧ 
      (∃ q, is_largest_prime_divisor q (m^2 + 3) ∧ is_least_prime_divisor q (m^4 + 6)) → m = 3 :=
by sorry

end find_positive_integer_n_l2118_211846


namespace product_of_primes_sum_85_l2118_211802

open Nat

theorem product_of_primes_sum_85 :
  ∃ (p q : ℕ), p.Prime ∧ q.Prime ∧ p + q = 85 ∧ p * q = 166 :=
sorry

end product_of_primes_sum_85_l2118_211802


namespace intersection_points_on_ellipse_l2118_211821

theorem intersection_points_on_ellipse (s x y : ℝ)
  (h_line1 : s * x - 3 * y - 4 * s = 0)
  (h_line2 : x - 3 * s * y + 4 = 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry

end intersection_points_on_ellipse_l2118_211821


namespace blocks_per_tree_l2118_211869

def trees_per_day : ℕ := 2
def blocks_after_5_days : ℕ := 30
def days : ℕ := 5

theorem blocks_per_tree : (blocks_after_5_days / (trees_per_day * days)) = 3 :=
by
  sorry

end blocks_per_tree_l2118_211869


namespace problem_statement_l2118_211835

noncomputable def f : ℝ → ℝ := sorry

axiom func_condition : ∀ a b : ℝ, b^2 * f a = a^2 * f b
axiom f2_nonzero : f 2 ≠ 0

theorem problem_statement : (f 6 - f 3) / f 2 = 27 / 4 := 
by 
  sorry

end problem_statement_l2118_211835


namespace smallest_positive_integer_l2118_211815

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l2118_211815


namespace almonds_walnuts_ratio_l2118_211839

-- Define the given weights and parts
def w_a : ℝ := 107.14285714285714
def w_m : ℝ := 150
def p_a : ℝ := 5

-- Now we will formulate the statement to prove the ratio of almonds to walnuts
theorem almonds_walnuts_ratio : 
  ∃ (p_w : ℝ), p_a / p_w = 5 / 2 :=
by
  -- It is given that p_a / p_w = 5 / 2, we need to find p_w
  sorry

end almonds_walnuts_ratio_l2118_211839


namespace find_a_l2118_211885

noncomputable def a : ℚ := ((68^3 - 65^3) * (32^3 + 18^3)) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2))

theorem find_a : a = 150 := 
  sorry

end find_a_l2118_211885


namespace find_point_P_coordinates_l2118_211810

noncomputable def coordinates_of_point (x y : ℝ) : Prop :=
  y > 0 ∧ x < 0 ∧ abs x = 4 ∧ abs y = 4

theorem find_point_P_coordinates : ∃ (x y : ℝ), coordinates_of_point x y ∧ (x, y) = (-4, 4) :=
by
  sorry

end find_point_P_coordinates_l2118_211810


namespace place_b_left_of_a_forms_correct_number_l2118_211864

noncomputable def form_three_digit_number (a b : ℕ) : ℕ :=
  100 * b + a

theorem place_b_left_of_a_forms_correct_number (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 1 ≤ b ∧ b < 10) :
  form_three_digit_number a b = 100 * b + a :=
by sorry

end place_b_left_of_a_forms_correct_number_l2118_211864


namespace original_cylinder_weight_is_24_l2118_211826

noncomputable def weight_of_original_cylinder (cylinder_weight cone_weight : ℝ) : Prop :=
  cylinder_weight = 3 * cone_weight

-- Given conditions in Lean 4
variables (cone_weight : ℝ) (h_cone_weight : cone_weight = 8)

-- Proof problem statement
theorem original_cylinder_weight_is_24 :
  weight_of_original_cylinder 24 cone_weight :=
by
  sorry

end original_cylinder_weight_is_24_l2118_211826


namespace evaluate_sets_are_equal_l2118_211896

theorem evaluate_sets_are_equal :
  (-3^5) = ((-3)^5) ∧
  ¬ ((-2^2) = ((-2)^2)) ∧
  ¬ ((-4 * 2^3) = (-4^2 * 3)) ∧
  ¬ ((- (-3)^2) = (- (-2)^3)) :=
by
  sorry

end evaluate_sets_are_equal_l2118_211896


namespace value_of_x_plus_2y_l2118_211897

theorem value_of_x_plus_2y (x y : ℝ) (h1 : (x + y) / 3 = 1.6666666666666667) (h2 : 2 * x + y = 7) : x + 2 * y = 8 := by
  sorry

end value_of_x_plus_2y_l2118_211897


namespace emily_small_gardens_l2118_211843

theorem emily_small_gardens 
  (total_seeds : Nat)
  (big_garden_seeds : Nat)
  (small_garden_seeds : Nat)
  (remaining_seeds : total_seeds = big_garden_seeds + (small_garden_seeds * 3)) :
  3 = (total_seeds - big_garden_seeds) / small_garden_seeds :=
by
  have h1 : total_seeds = 42 := by sorry
  have h2 : big_garden_seeds = 36 := by sorry
  have h3 : small_garden_seeds = 2 := by sorry
  have h4 : 6 = total_seeds - big_garden_seeds := by sorry
  have h5 : 3 = 6 / small_garden_seeds := by sorry
  sorry

end emily_small_gardens_l2118_211843


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l2118_211888

theorem solve_eq1 :
  ∀ x : ℝ, 6 * x - 7 = 4 * x - 5 ↔ x = 1 := by
  intro x
  sorry

theorem solve_eq2 :
  ∀ x : ℝ, 5 * (x + 8) - 5 = 6 * (2 * x - 7) ↔ x = 11 := by
  intro x
  sorry

theorem solve_eq3 :
  ∀ x : ℝ, x - (x - 1) / 2 = 2 - (x + 2) / 5 ↔ x = 11 / 7 := by
  intro x
  sorry

theorem solve_eq4 :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  intro x
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l2118_211888


namespace find_m_containing_2015_l2118_211893

theorem find_m_containing_2015 : 
  ∃ n : ℕ, ∀ k, 0 ≤ k ∧ k < n → 2015 = n^3 → (1979 + 2*k < 2015 ∧ 2015 < 1979 + 2*k + 2*n) :=
by
  sorry

end find_m_containing_2015_l2118_211893


namespace roots_pure_imaginary_if_negative_real_k_l2118_211881

theorem roots_pure_imaginary_if_negative_real_k (k : ℝ) (h_neg : k < 0) :
  (∃ (z : ℂ), 10 * z^2 - 3 * Complex.I * z - (k : ℂ) = 0 ∧ z.im ≠ 0 ∧ z.re = 0) :=
sorry

end roots_pure_imaginary_if_negative_real_k_l2118_211881


namespace min_value_xy_l2118_211898

-- Defining the operation ⊗
def otimes (a b : ℝ) : ℝ := a * b - a - b

theorem min_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : otimes x y = 3) : 9 ≤ x * y := by
  sorry

end min_value_xy_l2118_211898


namespace train_pass_bridge_time_l2118_211828

noncomputable def totalDistance (trainLength bridgeLength : ℕ) : ℕ :=
  trainLength + bridgeLength

noncomputable def speedInMPerSecond (speedInKmPerHour : ℕ) : ℝ :=
  (speedInKmPerHour * 1000) / 3600

noncomputable def timeToPass (totalDistance : ℕ) (speedInMPerSecond : ℝ) : ℝ :=
  totalDistance / speedInMPerSecond

theorem train_pass_bridge_time
  (trainLength : ℕ) (bridgeLength : ℕ) (speedInKmPerHour : ℕ)
  (h_train : trainLength = 300)
  (h_bridge : bridgeLength = 115)
  (h_speed : speedInKmPerHour = 35) :
  timeToPass (totalDistance trainLength bridgeLength) (speedInMPerSecond speedInKmPerHour) = 42.7 :=
by
  sorry

end train_pass_bridge_time_l2118_211828


namespace evaporation_period_days_l2118_211856

theorem evaporation_period_days
    (initial_water : ℝ)
    (daily_evaporation : ℝ)
    (evaporation_percentage : ℝ)
    (total_evaporated_water : ℝ)
    (number_of_days : ℝ) :
    initial_water = 10 ∧
    daily_evaporation = 0.06 ∧
    evaporation_percentage = 0.12 ∧
    total_evaporated_water = initial_water * evaporation_percentage ∧
    number_of_days = total_evaporated_water / daily_evaporation →
    number_of_days = 20 :=
by
  sorry

end evaporation_period_days_l2118_211856


namespace squared_sum_l2118_211822

theorem squared_sum (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 :=
by
  sorry

end squared_sum_l2118_211822


namespace exists_f_gcd_form_l2118_211809

noncomputable def f : ℤ → ℕ := sorry

theorem exists_f_gcd_form :
  (∀ x y : ℤ, Nat.gcd (f x) (f y) = Nat.gcd (f x) (Int.natAbs (x - y))) →
  ∃ m n : ℕ, (0 < m ∧ 0 < n) ∧ (∀ x : ℤ, f x = Nat.gcd (m + Int.natAbs x) n) :=
sorry

end exists_f_gcd_form_l2118_211809


namespace range_of_a_l2118_211867
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

theorem range_of_a (h : ∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x a = -g x) : 
  0 ≤ a ∧ a ≤ Real.exp 3 - 4 := 
sorry

end range_of_a_l2118_211867


namespace part1_l2118_211817

def U : Set ℝ := Set.univ
def P (a : ℝ) : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem part1 (a : ℝ) (P_def : P 3 = {x | 4 ≤ x ∧ x ≤ 7}) :
  ((U \ P a) ∩ Q = {x | -2 ≤ x ∧ x < 4}) := by
  sorry

end part1_l2118_211817


namespace sandy_total_spent_l2118_211800

def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def total_spent : ℝ := shorts_price + shirt_price + jacket_price

theorem sandy_total_spent : total_spent = 33.56 :=
by
  sorry

end sandy_total_spent_l2118_211800


namespace remainder_27_pow_482_div_13_l2118_211868

theorem remainder_27_pow_482_div_13 :
  27^482 % 13 = 1 :=
sorry

end remainder_27_pow_482_div_13_l2118_211868


namespace cost_of_pears_l2118_211833

theorem cost_of_pears 
  (initial_amount : ℕ := 55) 
  (left_amount : ℕ := 28) 
  (banana_count : ℕ := 2) 
  (banana_price : ℕ := 4) 
  (asparagus_price : ℕ := 6) 
  (chicken_price : ℕ := 11) 
  (total_spent : ℕ := 27) :
  initial_amount - left_amount - (banana_count * banana_price + asparagus_price + chicken_price) = 2 := 
by
  sorry

end cost_of_pears_l2118_211833


namespace more_geese_than_ducks_l2118_211894

def mallard_start := 25
def wood_start := 15
def geese_start := 2 * mallard_start - 10
def swan_start := 3 * wood_start + 8

def mallard_after_morning := mallard_start + 4
def wood_after_morning := wood_start + 8
def geese_after_morning := geese_start + 7
def swan_after_morning := swan_start

def mallard_after_noon := mallard_after_morning
def wood_after_noon := wood_after_morning - 6
def geese_after_noon := geese_after_morning - 5
def swan_after_noon := swan_after_morning - 9

def mallard_after_later := mallard_after_noon + 8
def wood_after_later := wood_after_noon + 10
def geese_after_later := geese_after_noon
def swan_after_later := swan_after_noon + 4

def mallard_after_evening := mallard_after_later + 5
def wood_after_evening := wood_after_later + 3
def geese_after_evening := geese_after_later + 15
def swan_after_evening := swan_after_later + 11

def mallard_final := 0
def wood_final := wood_after_evening - (3 / 4 : ℚ) * wood_after_evening
def geese_final := geese_after_evening - (1 / 5 : ℚ) * geese_after_evening
def swan_final := swan_after_evening - (1 / 2 : ℚ) * swan_after_evening

theorem more_geese_than_ducks :
  (geese_final - (mallard_final + wood_final)) = 38 :=
by sorry

end more_geese_than_ducks_l2118_211894


namespace intersection_of_sets_l2118_211812

variable {x : ℝ}

def SetA : Set ℝ := {x | x + 1 > 0}
def SetB : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_sets : SetA ∩ SetB = {x | -1 < x ∧ x < 3} :=
by sorry

end intersection_of_sets_l2118_211812


namespace student_B_speed_l2118_211882

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l2118_211882
