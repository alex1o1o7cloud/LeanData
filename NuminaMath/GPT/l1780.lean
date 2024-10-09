import Mathlib

namespace total_blocks_correct_l1780_178043

-- Definitions given by the conditions in the problem
def red_blocks : ℕ := 18
def yellow_blocks : ℕ := red_blocks + 7
def blue_blocks : ℕ := red_blocks + 14

-- Theorem stating the goal to prove
theorem total_blocks_correct : red_blocks + yellow_blocks + blue_blocks = 75 := by
  -- Skipping the proof for now
  sorry

end total_blocks_correct_l1780_178043


namespace total_opponents_runs_l1780_178073

theorem total_opponents_runs (team_scores : List ℕ) (opponent_scores : List ℕ) :
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  ∃ lost_games won_games opponent_lost_scores opponent_won_scores,
    lost_games = [1, 3, 5, 7, 9, 11] ∧
    won_games = [2, 4, 6, 8, 10, 12] ∧
    (∀ (t : ℕ), t ∈ lost_games → ∃ o : ℕ, o = t + 1 ∧ o ∈ opponent_scores) ∧
    (∀ (t : ℕ), t ∈ won_games → ∃ o : ℕ, o = t / 2 ∧ o ∈ opponent_scores) ∧
    opponent_scores = opponent_lost_scores ++ opponent_won_scores ∧
    opponent_lost_scores = [2, 4, 6, 8, 10, 12] ∧
    opponent_won_scores = [1, 2, 3, 4, 5, 6] →
  opponent_scores.sum = 63 :=
by
  sorry

end total_opponents_runs_l1780_178073


namespace solvable_system_of_inequalities_l1780_178069

theorem solvable_system_of_inequalities (n : ℕ) : 
  (∃ x : ℝ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k < x ^ k ∧ x ^ k < k + 1)) ∧ (1 < x ∧ x < 2)) ↔ (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end solvable_system_of_inequalities_l1780_178069


namespace ratio_of_a_and_b_l1780_178080

theorem ratio_of_a_and_b (x y a b : ℝ) (h1 : x / y = 3) (h2 : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end ratio_of_a_and_b_l1780_178080


namespace weight_of_fish_in_barrel_l1780_178012

/-- 
Given a barrel with an initial weight of 54 kg when full of fish,
and a weight of 29 kg after removing half of the fish,
prove that the initial weight of the fish in the barrel was 50 kg.
-/
theorem weight_of_fish_in_barrel (B F : ℝ)
  (h1: B + F = 54)
  (h2: B + F / 2 = 29) : F = 50 := 
sorry

end weight_of_fish_in_barrel_l1780_178012


namespace coffee_mix_price_l1780_178030

theorem coffee_mix_price (
  weight1 price1 weight2 price2 total_weight : ℝ)
  (h1 : weight1 = 9)
  (h2 : price1 = 2.15)
  (h3 : weight2 = 9)
  (h4 : price2 = 2.45)
  (h5 : total_weight = 18)
  :
  (weight1 * price1 + weight2 * price2) / total_weight = 2.30 :=
by
  sorry

end coffee_mix_price_l1780_178030


namespace answer_l1780_178047

def p : Prop := ∃ x > Real.exp 1, (1 / 2)^x > Real.log x
def q : Prop := ∀ a b : Real, a > 1 → b > 1 → Real.log a / Real.log b + 2 * (Real.log b / Real.log a) ≥ 2 * Real.sqrt 2

theorem answer : ¬ p ∧ q :=
by
  have h1 : ¬ p := sorry
  have h2 : q := sorry
  exact ⟨h1, h2⟩

end answer_l1780_178047


namespace quarterback_sacked_times_l1780_178008

theorem quarterback_sacked_times
    (total_throws : ℕ)
    (no_pass_percentage : ℚ)
    (half_sacked : ℚ)
    (no_passes : ℕ)
    (sacks : ℕ) :
    total_throws = 80 →
    no_pass_percentage = 0.30 →
    half_sacked = 0.50 →
    no_passes = total_throws * no_pass_percentage →
    sacks = no_passes / 2 →
    sacks = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end quarterback_sacked_times_l1780_178008


namespace possible_measures_for_angle_A_l1780_178019

-- Definition of angles A and B, and their relationship
def is_supplementary_angles (A B : ℕ) : Prop := A + B = 180

def is_multiple_of (A B : ℕ) : Prop := ∃ k : ℕ, k ≥ 1 ∧ A = k * B

-- Prove there are 17 possible measures for angle A.
theorem possible_measures_for_angle_A : 
  (∀ (A B : ℕ), (A > 0) ∧ (B > 0) ∧ is_multiple_of A B ∧ is_supplementary_angles A B → 
  A = B * 17) := 
sorry

end possible_measures_for_angle_A_l1780_178019


namespace composite_polynomial_l1780_178039

-- Definition that checks whether a number is composite
def is_composite (a : ℕ) : Prop := ∃ (b c : ℕ), b > 1 ∧ c > 1 ∧ a = b * c

-- Problem translated into a Lean 4 statement
theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  is_composite (n ^ (5 * n - 1) + n ^ (5 * n - 2) + n ^ (5 * n - 3) + n + 1) :=
sorry

end composite_polynomial_l1780_178039


namespace trigonometric_identity_l1780_178078

open Real

theorem trigonometric_identity
  (theta : ℝ)
  (h : cos (π / 6 - theta) = 2 * sqrt 2 / 3) : 
  cos (π / 3 + theta) = 1 / 3 ∨ cos (π / 3 + theta) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l1780_178078


namespace product_of_means_eq_pm20_l1780_178016

theorem product_of_means_eq_pm20 :
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  a * b = 20 ∨ a * b = -20 :=
by
  -- Placeholders for the actual proof
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  sorry

end product_of_means_eq_pm20_l1780_178016


namespace april_roses_l1780_178003

theorem april_roses (R : ℕ) (h1 : 7 * (R - 4) = 35) : R = 9 :=
sorry

end april_roses_l1780_178003


namespace find_the_number_l1780_178028

theorem find_the_number (x : ℝ) (h : 150 - x = x + 68) : x = 41 :=
sorry

end find_the_number_l1780_178028


namespace inequality_generalization_l1780_178018

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) :
  x + n^n / x^n ≥ n + 1 :=
sorry

end inequality_generalization_l1780_178018


namespace village_population_l1780_178082

theorem village_population (P : ℝ) (h : 0.9 * P = 45000) : P = 50000 :=
by
  sorry

end village_population_l1780_178082


namespace packs_of_string_cheese_l1780_178051

theorem packs_of_string_cheese (cost_per_piece: ℕ) (pieces_per_pack: ℕ) (total_cost_dollars: ℕ) 
                                (h1: cost_per_piece = 10) 
                                (h2: pieces_per_pack = 20) 
                                (h3: total_cost_dollars = 6) : 
  (total_cost_dollars * 100) / (cost_per_piece * pieces_per_pack) = 3 := 
by
  -- Insert proof here
  sorry

end packs_of_string_cheese_l1780_178051


namespace vertex_y_coordinate_l1780_178005

theorem vertex_y_coordinate (x : ℝ) : 
    let a := -6
    let b := 24
    let c := -7
    ∃ k : ℝ, k = 17 ∧ ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x - 2)^2 + k) := 
by 
  sorry

end vertex_y_coordinate_l1780_178005


namespace f_zero_f_odd_f_range_l1780_178048

-- Condition 1: The function f is defined on ℝ
-- Condition 2: f(x + y) = f(x) + f(y)
-- Condition 3: f(1/3) = 1
-- Condition 4: f(x) < 0 when x > 0

variables (f : ℝ → ℝ)
axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_third : f (1/3) = 1
axiom f_neg_positive : ∀ x : ℝ, 0 < x → f x < 0

-- Question 1: Find the value of f(0)
theorem f_zero : f 0 = 0 := sorry

-- Question 2: Prove that f is an odd function
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

-- Question 3: Find the range of x where f(x) + f(2 + x) < 2
theorem f_range : ∀ x : ℝ, f x + f (2 + x) < 2 → -2/3 < x := sorry

end f_zero_f_odd_f_range_l1780_178048


namespace find_a5_l1780_178011

noncomputable def arithmetic_sequence (n : ℕ) (a d : ℤ) : ℤ :=
a + n * d

theorem find_a5 (a d : ℤ) (a_2_a_4_sum : arithmetic_sequence 1 a d + arithmetic_sequence 3 a d = 16)
  (a1 : arithmetic_sequence 0 a d = 1) :
  arithmetic_sequence 4 a d = 15 :=
by
  sorry

end find_a5_l1780_178011


namespace dormitory_to_city_distance_l1780_178038

theorem dormitory_to_city_distance
  (D : ℝ)
  (h1 : (1/5) * D + (2/3) * D + 14 = D) :
  D = 105 :=
by
  sorry

end dormitory_to_city_distance_l1780_178038


namespace expand_and_simplify_l1780_178087

-- Define the two polynomials P and Q.
def P (x : ℝ) := 5 * x + 3
def Q (x : ℝ) := 2 * x^2 - x + 4

-- State the theorem we want to prove.
theorem expand_and_simplify (x : ℝ) : (P x * Q x) = 10 * x^3 + x^2 + 17 * x + 12 := 
by
  sorry

end expand_and_simplify_l1780_178087


namespace fibonacci_arith_sequence_a_eq_665_l1780_178052

theorem fibonacci_arith_sequence_a_eq_665 (F : ℕ → ℕ) (a b c : ℕ) :
  (F 1 = 1) →
  (F 2 = 1) →
  (∀ n, n ≥ 3 → F n = F (n - 1) + F (n - 2)) →
  (a + b + c = 2000) →
  (F a < F b ∧ F b < F c ∧ F b - F a = F c - F b) →
  a = 665 :=
by
  sorry

end fibonacci_arith_sequence_a_eq_665_l1780_178052


namespace emani_money_l1780_178060

def emani_has_30_more (E H : ℝ) : Prop := E = H + 30
def equal_share (E H : ℝ) : Prop := (E + H) / 2 = 135

theorem emani_money (E H : ℝ) (h1: emani_has_30_more E H) (h2: equal_share E H) : E = 150 :=
by
  sorry

end emani_money_l1780_178060


namespace cube_surface_divisible_into_12_squares_l1780_178090

theorem cube_surface_divisible_into_12_squares (a : ℝ) :
  (∃ b : ℝ, b = a / Real.sqrt 2 ∧
  ∀ cube_surface_area: ℝ, cube_surface_area = 6 * a^2 →
  ∀ smaller_square_area: ℝ, smaller_square_area = b^2 →
  12 * smaller_square_area = cube_surface_area) :=
sorry

end cube_surface_divisible_into_12_squares_l1780_178090


namespace new_babysitter_rate_l1780_178059

theorem new_babysitter_rate (x : ℝ) :
  (6 * 16) - 18 = 6 * x + 3 * 2 → x = 12 :=
by
  intros h
  sorry

end new_babysitter_rate_l1780_178059


namespace specific_value_of_n_l1780_178092

theorem specific_value_of_n (n : ℕ) 
  (A_n : ℕ → ℕ)
  (C_n : ℕ → ℕ → ℕ)
  (h1 : A_n n ^ 2 = C_n n (n-3)) :
  n = 8 :=
sorry

end specific_value_of_n_l1780_178092


namespace problem_a_l1780_178033

variable {S : Type*}
variables (a b : S)
variables [Inhabited S] -- Ensures S has at least one element
variables (op : S → S → S) -- Defines the binary operation

-- Condition: binary operation a * (b * a) = b holds for all a, b in S
axiom binary_condition : ∀ a b : S, op a (op b a) = b

-- Theorem to prove: (a * b) * a ≠ a
theorem problem_a : (op (op a b) a) ≠ a :=
sorry

end problem_a_l1780_178033


namespace equation_of_circle_l1780_178076

-- Definitions directly based on conditions 
noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)
noncomputable def directrix_of_parabola : ℝ × ℝ -> Prop
  | (x, _) => x = -1

-- The statement of the problem: equation of the circle with given conditions
theorem equation_of_circle : ∃ (r : ℝ), (∀ (x y : ℝ), (x - 1)^2 + y^2 = r^2) ∧ r = 2 :=
sorry

end equation_of_circle_l1780_178076


namespace maximum_books_l1780_178037

theorem maximum_books (dollars : ℝ) (price_per_book : ℝ) (n : ℕ) 
    (h1 : dollars = 12) (h2 : price_per_book = 1.25) : n ≤ 9 :=
    sorry

end maximum_books_l1780_178037


namespace age_problem_l1780_178086

variables (K M A B : ℕ)

theorem age_problem
  (h1 : K + 7 = 3 * M)
  (h2 : M = 5)
  (h3 : A + B = 2 * M + 4)
  (h4 : A = B - 3)
  (h5 : K + B = M + 9) :
  K = 8 ∧ M = 5 ∧ B = 6 ∧ A = 3 :=
sorry

end age_problem_l1780_178086


namespace inequality_ab_ab2_a_l1780_178044

theorem inequality_ab_ab2_a (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_ab_ab2_a_l1780_178044


namespace maximum_value_of_expression_l1780_178063

variable (x y z : ℝ)

theorem maximum_value_of_expression (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) : 
  x + y^3 + z^4 ≤ 1 :=
sorry

end maximum_value_of_expression_l1780_178063


namespace sum_arithmetic_series_eq_250500_l1780_178004

theorem sum_arithmetic_series_eq_250500 :
  let a1 := 2
  let d := 2
  let an := 1000
  let n := 500
  (a1 + (n-1) * d = an) →
  ((n * (a1 + an)) / 2 = 250500) :=
by
  sorry

end sum_arithmetic_series_eq_250500_l1780_178004


namespace largest_angle_in_triangle_l1780_178064

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A = 45) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : 
  max A (max B C) = 75 :=
by
  -- Since no proof is needed, we mark it as sorry
  sorry

end largest_angle_in_triangle_l1780_178064


namespace rate_of_current_is_8_5_l1780_178093

-- Define the constants for the problem
def downstream_speed : ℝ := 24
def upstream_speed : ℝ := 7
def rate_still_water : ℝ := 15.5

-- Define the rate of the current calculation
def rate_of_current : ℝ := downstream_speed - rate_still_water

-- Define the rate of the current proof statement
theorem rate_of_current_is_8_5 :
  rate_of_current = 8.5 :=
by
  -- This skip the actual proof
  sorry

end rate_of_current_is_8_5_l1780_178093


namespace alex_wins_if_picks_two_l1780_178032

theorem alex_wins_if_picks_two (matches_left : ℕ) (alex_picks bob_picks : ℕ) :
  matches_left = 30 →
  1 ≤ alex_picks ∧ alex_picks ≤ 6 →
  1 ≤ bob_picks ∧ bob_picks ≤ 6 →
  alex_picks = 2 →
  (∀ n, (n % 7 ≠ 0) → ¬ (∃ k, matches_left - k ≤ 0 ∧ (matches_left - k) % 7 = 0)) :=
by sorry

end alex_wins_if_picks_two_l1780_178032


namespace scientific_notation_correct_l1780_178068

noncomputable def significant_figures : ℝ := 274
noncomputable def decimal_places : ℝ := 8
noncomputable def scientific_notation_rep : ℝ := 2.74 * (10^8)

theorem scientific_notation_correct :
  274000000 = scientific_notation_rep :=
sorry

end scientific_notation_correct_l1780_178068


namespace asha_borrowed_from_mother_l1780_178010

def total_money (M : ℕ) : ℕ := 20 + 40 + 70 + 100 + M

def remaining_money_after_spending_3_4 (total : ℕ) : ℕ := total * 1 / 4

theorem asha_borrowed_from_mother : ∃ M : ℕ, total_money M = 260 ∧ remaining_money_after_spending_3_4 (total_money M) = 65 :=
by
  sorry

end asha_borrowed_from_mother_l1780_178010


namespace Mrs_Hilt_remaining_money_l1780_178065

theorem Mrs_Hilt_remaining_money :
  let initial_amount : ℝ := 3.75
  let pencil_cost : ℝ := 1.15
  let eraser_cost : ℝ := 0.85
  let notebook_cost : ℝ := 2.25
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -0.50 :=
by
  sorry

end Mrs_Hilt_remaining_money_l1780_178065


namespace find_x_l1780_178014

theorem find_x (x : ℝ) (h : 0.65 * x = 0.2 * 617.50) : x = 190 :=
by
  sorry

end find_x_l1780_178014


namespace problem1_l1780_178034

theorem problem1 (x : ℝ) : (2 * x - 1) * (2 * x - 3) - (1 - 2 * x) * (2 - x) = 2 * x^2 - 3 * x + 1 :=
by
  sorry

end problem1_l1780_178034


namespace repeating_decimals_product_l1780_178088

-- Definitions to represent the conditions
def repeating_decimal_03_as_frac : ℚ := 1 / 33
def repeating_decimal_36_as_frac : ℚ := 4 / 11

-- The statement to be proven
theorem repeating_decimals_product : (repeating_decimal_03_as_frac * repeating_decimal_36_as_frac) = (4 / 363) :=
by {
  sorry
}

end repeating_decimals_product_l1780_178088


namespace sum_of_interior_angles_eq_1440_l1780_178023

theorem sum_of_interior_angles_eq_1440 (h : ∀ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ)) : 
    (∃ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ) ∧ (n - 2) * 180 = 1440) :=
by
  sorry

end sum_of_interior_angles_eq_1440_l1780_178023


namespace original_cost_price_l1780_178089

theorem original_cost_price (C : ℝ) 
  (h1 : 0.87 * C > 0) 
  (h2 : 1.2 * (0.87 * C) = 54000) : 
  C = 51724.14 :=
by
  sorry

end original_cost_price_l1780_178089


namespace find_smaller_number_l1780_178040

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l1780_178040


namespace range_g_a_values_l1780_178054

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem range_g : ∀ x : ℝ, -1 ≤ g x ∧ g x ≤ 1 :=
sorry

theorem a_values (a : ℝ) : (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) :=
sorry

end range_g_a_values_l1780_178054


namespace paula_remaining_money_l1780_178015

-- Definitions based on the conditions
def initialMoney : ℕ := 1000
def shirtCost : ℕ := 45
def pantsCost : ℕ := 85
def jacketCost : ℕ := 120
def shoeCost : ℕ := 95
def jeansOriginalPrice : ℕ := 140
def jeansDiscount : ℕ := 30 / 100  -- 30%

-- Using definitions to compute the spending and remaining money
def totalShirtCost : ℕ := 6 * shirtCost
def totalPantsCost : ℕ := 2 * pantsCost
def totalShoeCost : ℕ := 3 * shoeCost
def jeansDiscountValue : ℕ := jeansDiscount * jeansOriginalPrice
def jeansDiscountedPrice : ℕ := jeansOriginalPrice - jeansDiscountValue
def totalSpent : ℕ := totalShirtCost + totalPantsCost + jacketCost + totalShoeCost
def remainingMoney : ℕ := initialMoney - totalSpent - jeansDiscountedPrice

-- Proof problem statement
theorem paula_remaining_money : remainingMoney = 57 := by
  sorry

end paula_remaining_money_l1780_178015


namespace least_number_to_add_l1780_178099

theorem least_number_to_add (n : ℕ) :
  (exists n, 1202 + n % 4 = 0 ∧ (∀ m, (1202 + m) % 4 = 0 → n ≤ m)) → n = 2 :=
by
  sorry

end least_number_to_add_l1780_178099


namespace triangle_sides_external_tangent_l1780_178061

theorem triangle_sides_external_tangent (R r : ℝ) (h : R > r) :
  ∃ (AB BC AC : ℝ),
    AB = 2 * Real.sqrt (R * r) ∧
    AC = 2 * r * Real.sqrt (R / (R + r)) ∧
    BC = 2 * R * Real.sqrt (r / (R + r)) :=
by
  sorry

end triangle_sides_external_tangent_l1780_178061


namespace solution_set_of_inequality_l1780_178021

theorem solution_set_of_inequality :
  {x : ℝ | |x - 5| + |x + 3| >= 10} = {x : ℝ | x ≤ -4} ∪ {x : ℝ | x ≥ 6} :=
by
  sorry

end solution_set_of_inequality_l1780_178021


namespace probability_both_tell_truth_l1780_178027

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_tell_truth (hA : P_A = 0.75) (hB : P_B = 0.60) : P_A * P_B = 0.45 :=
by
  rw [hA, hB]
  norm_num

end probability_both_tell_truth_l1780_178027


namespace find_multiplier_l1780_178085

/-- Define the number -/
def number : ℝ := -10.0

/-- Define the multiplier m -/
def m : ℝ := 0.4

/-- Given conditions and prove the correct multiplier -/
theorem find_multiplier (number : ℝ) (m : ℝ) 
  (h1 : ∃ m : ℝ, m * number - 8 = -12) 
  (h2 : number = -10.0) : m = 0.4 :=
by
  -- We skip the actual steps and provide the answer using sorry
  sorry

end find_multiplier_l1780_178085


namespace group_D_forms_a_definite_set_l1780_178013

theorem group_D_forms_a_definite_set : 
  ∃ (S : Set ℝ), S = { x : ℝ | x = 1 ∨ x = -1 } :=
by
  sorry

end group_D_forms_a_definite_set_l1780_178013


namespace bivalid_positions_count_l1780_178058

/-- 
A position of the hands of a (12-hour, analog) clock is called valid if it occurs in the course of a day.
A position of the hands is called bivalid if it is valid and, in addition, the position formed by interchanging the hour and minute hands is valid.
-/
def is_valid (h m : ℕ) : Prop := 
  0 ≤ h ∧ h < 360 ∧ 
  0 ≤ m ∧ m < 360

def satisfies_conditions (h m : Int) (a b : Int) : Prop :=
  m = 12 * h - 360 * a ∧ h = 12 * m - 360 * b

def is_bivalid (h m : ℕ) : Prop := 
  ∃ (a b : Int), satisfies_conditions (h : Int) (m : Int) a b ∧ satisfies_conditions (m : Int) (h : Int) b a

theorem bivalid_positions_count : 
  ∃ (n : ℕ), n = 143 ∧ 
  ∀ (h m : ℕ), is_bivalid h m → n = 143 :=
sorry

end bivalid_positions_count_l1780_178058


namespace positive_numbers_with_cube_root_lt_10_l1780_178002

def cube_root_lt_10 (n : ℕ) : Prop :=
  (↑n : ℝ)^(1 / 3 : ℝ) < 10

theorem positive_numbers_with_cube_root_lt_10 : 
  ∃ (count : ℕ), (count = 999) ∧ ∀ n : ℕ, (1 ≤ n ∧ n ≤ 999) → cube_root_lt_10 n :=
by
  sorry

end positive_numbers_with_cube_root_lt_10_l1780_178002


namespace swimming_speed_in_still_water_l1780_178031

-- Given conditions
def water_speed : ℝ := 4
def swim_time_against_current : ℝ := 2
def swim_distance_against_current : ℝ := 8

-- What we are trying to prove
theorem swimming_speed_in_still_water (v : ℝ) 
    (h1 : swim_distance_against_current = 8) 
    (h2 : swim_time_against_current = 2)
    (h3 : water_speed = 4) :
    v - water_speed = swim_distance_against_current / swim_time_against_current → v = 8 :=
by
  sorry

end swimming_speed_in_still_water_l1780_178031


namespace lucas_total_assignments_l1780_178091

theorem lucas_total_assignments : 
  ∃ (total_assignments : ℕ), 
  (∀ (points : ℕ), 
    (points ≤ 10 → total_assignments = points * 1) ∧
    (10 < points ∧ points ≤ 20 → total_assignments = 10 * 1 + (points - 10) * 2) ∧
    (20 < points ∧ points ≤ 30 → total_assignments = 10 * 1 + 10 * 2 + (points - 20) * 3)
  ) ∧
  total_assignments = 60 :=
by
  sorry

end lucas_total_assignments_l1780_178091


namespace graph_passes_through_point_l1780_178053

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 3

theorem graph_passes_through_point (a : ℝ) : f a 1 = 4 := by
  sorry

end graph_passes_through_point_l1780_178053


namespace cost_price_of_each_clock_l1780_178025

theorem cost_price_of_each_clock
  (C : ℝ)
  (h1 : 40 * C * 1.1 + 50 * C * 1.2 - 90 * C * 1.15 = 40) :
  C = 80 :=
sorry

end cost_price_of_each_clock_l1780_178025


namespace measure_angle_F_correct_l1780_178074

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end measure_angle_F_correct_l1780_178074


namespace holly_pills_per_week_l1780_178062

theorem holly_pills_per_week 
  (insulin_pills_per_day : ℕ)
  (blood_pressure_pills_per_day : ℕ)
  (anticonvulsants_per_day : ℕ)
  (H1 : insulin_pills_per_day = 2)
  (H2 : blood_pressure_pills_per_day = 3)
  (H3 : anticonvulsants_per_day = 2 * blood_pressure_pills_per_day) :
  (insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsants_per_day) * 7 = 77 := 
by
  sorry

end holly_pills_per_week_l1780_178062


namespace number_of_people_in_team_l1780_178041

def total_distance : ℕ := 150
def distance_per_member : ℕ := 30

theorem number_of_people_in_team :
  (total_distance / distance_per_member) = 5 := by
  sorry

end number_of_people_in_team_l1780_178041


namespace ariana_total_owe_l1780_178057

-- Definitions based on the conditions
def first_bill_principal : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_overdue_months : ℕ := 2

def second_bill_principal : ℕ := 130
def second_bill_late_fee : ℕ := 50
def second_bill_overdue_months : ℕ := 6

def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80

-- Theorem
theorem ariana_total_owe : 
  first_bill_principal + 
    (first_bill_principal : ℝ) * first_bill_interest_rate * (first_bill_overdue_months : ℝ) +
    second_bill_principal + 
    second_bill_late_fee * second_bill_overdue_months + 
    third_bill_first_month_fee + 
    third_bill_second_month_fee = 790 := 
by 
  sorry

end ariana_total_owe_l1780_178057


namespace ellipse_eq_from_hyperbola_l1780_178046

noncomputable def hyperbola_eq : Prop :=
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = -1) →
  (x^2 / 4 + y^2 / 16 = 1)

theorem ellipse_eq_from_hyperbola :
  hyperbola_eq :=
by
  sorry

end ellipse_eq_from_hyperbola_l1780_178046


namespace remainder_of_7n_mod_4_l1780_178077

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l1780_178077


namespace gasoline_price_increase_percent_l1780_178070

theorem gasoline_price_increase_percent {P Q : ℝ}
  (h₁ : P > 0)
  (h₂: Q > 0)
  (x : ℝ)
  (condition : P * Q * 1.08 = P * (1 + x/100) * Q * 0.90) :
  x = 20 :=
by {
  sorry
}

end gasoline_price_increase_percent_l1780_178070


namespace standard_equation_of_ellipse_locus_of_midpoint_M_l1780_178035

-- Define the conditions of the ellipse
def isEllipse (a b c : ℝ) : Prop :=
  a = 2 ∧ c = Real.sqrt 3 ∧ b = Real.sqrt (a^2 - c^2)

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the locus of the midpoint M
def locus_midpoint (x y : ℝ) : Prop :=
  x^2 / 4 + 4 * y^2 = 1

theorem standard_equation_of_ellipse :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, ellipse_equation x y) :=
sorry

theorem locus_of_midpoint_M :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, locus_midpoint x y) :=
sorry

end standard_equation_of_ellipse_locus_of_midpoint_M_l1780_178035


namespace number_of_valid_n_l1780_178079

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m n : ℕ, n = 2^m * 5^n

def has_nonzero_thousandths_digit (n : ℕ) : Prop :=
  -- Placeholder for a formal definition to check the non-zero thousandths digit.
  sorry

theorem number_of_valid_n : 
  (∃ l : List ℕ, 
    l.length = 10 ∧ 
    ∀ n ∈ l, n <= 200 ∧ is_terminating_decimal n ∧ has_nonzero_thousandths_digit n) :=
sorry

end number_of_valid_n_l1780_178079


namespace tickets_required_l1780_178050

theorem tickets_required (cost_ferris_wheel : ℝ) (cost_roller_coaster : ℝ) 
  (discount_multiple_rides : ℝ) (coupon_value : ℝ) 
  (total_cost_with_discounts : ℝ) : 
  cost_ferris_wheel = 2.0 ∧ 
  cost_roller_coaster = 7.0 ∧ 
  discount_multiple_rides = 1.0 ∧ 
  coupon_value = 1.0 → 
  total_cost_with_discounts = 7.0 :=
by
  sorry

end tickets_required_l1780_178050


namespace marble_leftovers_l1780_178071

theorem marble_leftovers :
  ∃ r p : ℕ, (r % 8 = 5) ∧ (p % 8 = 7) ∧ ((r + p) % 10 = 0) → ((r + p) % 8 = 4) :=
by { sorry }

end marble_leftovers_l1780_178071


namespace roots_relationship_l1780_178001

theorem roots_relationship (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0)
  (h_triple : β = 3 * α)
  (h_vieta1 : α + β = -b / a)
  (h_vieta2 : α * β = c / a) : 
  3 * b^2 = 16 * a * c :=
sorry

end roots_relationship_l1780_178001


namespace children_on_bus_l1780_178042

/-- Prove the number of children on the bus after the bus stop equals 14 given the initial conditions -/
theorem children_on_bus (initial_children : ℕ) (children_got_off : ℕ) (extra_children_got_on : ℕ) (final_children : ℤ) :
  initial_children = 5 →
  children_got_off = 63 →
  extra_children_got_on = 9 →
  final_children = (initial_children - children_got_off) + (children_got_off + extra_children_got_on) →
  final_children = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end children_on_bus_l1780_178042


namespace maximize_area_l1780_178024

noncomputable def optimal_fencing (L W : ℝ) : Prop :=
  (2 * L + W = 1200) ∧ (∀ L1 W1, 2 * L1 + W1 = 1200 → L * W ≥ L1 * W1)

theorem maximize_area : ∃ L W, optimal_fencing L W ∧ L + W = 900 := sorry

end maximize_area_l1780_178024


namespace triangle_angle_A_l1780_178098

theorem triangle_angle_A (a c : ℝ) (C A : ℝ) 
  (h1 : a = 4 * Real.sqrt 3)
  (h2 : c = 12)
  (h3 : C = Real.pi / 3)
  (h4 : a < c) :
  A = Real.pi / 6 :=
sorry

end triangle_angle_A_l1780_178098


namespace composite_number_l1780_178007

theorem composite_number (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (10 ^ n + 1) * (10 ^ (n + 1) - 1) / 9 :=
by sorry

end composite_number_l1780_178007


namespace infinite_rationals_sqrt_rational_l1780_178072

theorem infinite_rationals_sqrt_rational : ∃ᶠ x : ℚ in Filter.atTop, ∃ y : ℚ, y = Real.sqrt (x^2 + x + 1) :=
sorry

end infinite_rationals_sqrt_rational_l1780_178072


namespace alice_net_amount_spent_l1780_178022

noncomputable def net_amount_spent : ℝ :=
  let price_per_pint := 4
  let sunday_pints := 4
  let sunday_cost := sunday_pints * price_per_pint

  let monday_discount := 0.1
  let monday_pints := 3 * sunday_pints
  let monday_price_per_pint := price_per_pint * (1 - monday_discount)
  let monday_cost := monday_pints * monday_price_per_pint

  let tuesday_discount := 0.2
  let tuesday_pints := monday_pints / 3
  let tuesday_price_per_pint := price_per_pint * (1 - tuesday_discount)
  let tuesday_cost := tuesday_pints * tuesday_price_per_pint

  let wednesday_returned_pints := tuesday_pints / 2
  let wednesday_refund := wednesday_returned_pints * tuesday_price_per_pint

  sunday_cost + monday_cost + tuesday_cost - wednesday_refund

theorem alice_net_amount_spent : net_amount_spent = 65.60 := by
  sorry

end alice_net_amount_spent_l1780_178022


namespace simplified_expression_correct_l1780_178075

noncomputable def simplified_expression : ℝ := 0.3 * 0.8 + 0.1 * 0.5

theorem simplified_expression_correct : simplified_expression = 0.29 := by 
  sorry

end simplified_expression_correct_l1780_178075


namespace total_budget_l1780_178083

theorem total_budget (s_ticket : ℕ) (s_drinks_food : ℕ) (k_ticket : ℕ) (k_drinks : ℕ) (k_food : ℕ) 
  (h1 : s_ticket = 14) (h2 : s_drinks_food = 6) (h3 : k_ticket = 14) (h4 : k_drinks = 2) (h5 : k_food = 4) : 
  s_ticket + s_drinks_food + k_ticket + k_drinks + k_food = 40 := 
by
  sorry

end total_budget_l1780_178083


namespace remainder_x_squared_mod_25_l1780_178006

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end remainder_x_squared_mod_25_l1780_178006


namespace sum_of_two_digit_factors_l1780_178055

theorem sum_of_two_digit_factors (a b : ℕ) (h : a * b = 5681) (h1 : 10 ≤ a) (h2 : a < 100) (h3 : 10 ≤ b) (h4 : b < 100) : a + b = 154 :=
by
  sorry

end sum_of_two_digit_factors_l1780_178055


namespace sum_ages_of_brothers_l1780_178094

theorem sum_ages_of_brothers (x : ℝ) (ages : List ℝ) 
  (h1 : ages = [x, x + 1.5, x + 3, x + 4.5, x + 6, x + 7.5, x + 9])
  (h2 : x + 9 = 4 * x) : 
    List.sum ages = 52.5 := 
  sorry

end sum_ages_of_brothers_l1780_178094


namespace original_difference_of_weights_l1780_178020

variable (F S T : ℝ)

theorem original_difference_of_weights :
  (F + S + T = 75) →
  (F - 2 = 0.7 * (S + 2)) →
  (S + 1 = 0.8 * (T + 1)) →
  T - F = 10.16 :=
by
  intro h1 h2 h3
  sorry

end original_difference_of_weights_l1780_178020


namespace unique_solution_to_functional_eq_l1780_178084

theorem unique_solution_to_functional_eq :
  (∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2) :=
by
  sorry

end unique_solution_to_functional_eq_l1780_178084


namespace range_of_a_l1780_178067

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) :=
by
  sorry

end range_of_a_l1780_178067


namespace distance_incenters_ACD_BCD_l1780_178056

noncomputable def distance_between_incenters (AC : ℝ) (angle_ABC : ℝ) (angle_BAC : ℝ) : ℝ :=
  -- Use the given conditions to derive the distance value
  -- Skipping the detailed calculations, denoted by "sorry"
  sorry

theorem distance_incenters_ACD_BCD :
  distance_between_incenters 1 (30 : ℝ) (60 : ℝ) = 0.5177 := sorry

end distance_incenters_ACD_BCD_l1780_178056


namespace fibonacci_rabbits_l1780_178009

theorem fibonacci_rabbits : 
  ∀ (F : ℕ → ℕ), 
    (F 0 = 1) ∧ 
    (F 1 = 1) ∧ 
    (∀ n, F (n + 2) = F n + F (n + 1)) → 
    F 12 = 233 := 
by 
  intro F h; sorry

end fibonacci_rabbits_l1780_178009


namespace nate_search_time_l1780_178029

theorem nate_search_time
  (rowsG : Nat) (cars_per_rowG : Nat)
  (rowsH : Nat) (cars_per_rowH : Nat)
  (rowsI : Nat) (cars_per_rowI : Nat)
  (walk_speed : Nat) : Nat :=
  let total_cars : Nat := rowsG * cars_per_rowG + rowsH * cars_per_rowH + rowsI * cars_per_rowI
  let total_minutes : Nat := total_cars / walk_speed
  if total_cars % walk_speed == 0 then total_minutes else total_minutes + 1

/-- Given:
- rows in Section G = 18, cars per row in Section G = 12
- rows in Section H = 25, cars per row in Section H = 10
- rows in Section I = 17, cars per row in Section I = 11
- Nate's walking speed is 8 cars per minute
Prove: Nate took 82 minutes to search the parking lot
-/
example : nate_search_time 18 12 25 10 17 11 8 = 82 := by
  sorry

end nate_search_time_l1780_178029


namespace solve_log_eq_l1780_178097

theorem solve_log_eq (x : ℝ) (h : 0 < x) :
  (1 / (Real.sqrt (Real.logb 5 (5 * x)) + Real.sqrt (Real.logb 5 x)) + Real.sqrt (Real.logb 5 x) = 2) ↔ x = 125 := 
  sorry

end solve_log_eq_l1780_178097


namespace area_of_tangency_triangle_l1780_178000

noncomputable def area_of_triangle : ℝ :=
  let r1 := 2
  let r2 := 3
  let r3 := 4
  let s := (r1 + r2 + r3) / 2
  let A := Real.sqrt (s * (s - (r1 + r2)) * (s - (r2 + r3)) * (s - (r1 + r3)))
  let inradius := A / s
  let area_points_of_tangency := A * (inradius / r1) * (inradius / r2) * (inradius / r3)
  area_points_of_tangency

theorem area_of_tangency_triangle :
  area_of_triangle = (16 * Real.sqrt 6) / 3 :=
sorry

end area_of_tangency_triangle_l1780_178000


namespace simplify_sin_cos_expr_cos_pi_six_alpha_expr_l1780_178045

open Real

-- Problem (1)
theorem simplify_sin_cos_expr (x : ℝ) :
  (sin x ^ 2 / (sin x - cos x)) - ((sin x + cos x) / (tan x ^ 2 - 1)) - sin x = cos x :=
sorry

-- Problem (2)
theorem cos_pi_six_alpha_expr (α : ℝ) (h : cos (π / 6 - α) = sqrt 3 / 3) :
  cos (5 * π / 6 + α) + cos (4 * π / 3 + α) ^ 2 = (2 - sqrt 3) / 3 :=
sorry

end simplify_sin_cos_expr_cos_pi_six_alpha_expr_l1780_178045


namespace neg_pow_eq_pow_four_l1780_178081

variable (a : ℝ)

theorem neg_pow_eq_pow_four (a : ℝ) : (-a)^4 = a^4 :=
sorry

end neg_pow_eq_pow_four_l1780_178081


namespace winner_beats_by_16_secons_l1780_178096

-- Definitions of the times for mathematician and physicist
variables (x y : ℕ)

-- Conditions based on the given problem
def condition1 := 2 * y - x = 24
def condition2 := 2 * x - y = 72

-- The statement to prove
theorem winner_beats_by_16_secons (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x - 2 * y = 16 := 
sorry

end winner_beats_by_16_secons_l1780_178096


namespace combined_time_alligators_walked_l1780_178095

-- Define the conditions
def original_time : ℕ := 4
def return_time := original_time + 2 * Int.sqrt original_time

-- State the theorem to be proven
theorem combined_time_alligators_walked : original_time + return_time = 12 := by
  sorry

end combined_time_alligators_walked_l1780_178095


namespace average_greater_than_median_by_22_l1780_178017

/-- Define the weights of the siblings -/
def hammie_weight : ℕ := 120
def triplet1_weight : ℕ := 4
def triplet2_weight : ℕ := 4
def triplet3_weight : ℕ := 7
def brother_weight : ℕ := 10

/-- Define the list of weights -/
def weights : List ℕ := [hammie_weight, triplet1_weight, triplet2_weight, triplet3_weight, brother_weight]

/-- Define the median and average weight -/
def median_weight : ℕ := 7
def average_weight : ℕ := 29

theorem average_greater_than_median_by_22 : average_weight - median_weight = 22 := by
  sorry

end average_greater_than_median_by_22_l1780_178017


namespace at_least_two_equal_l1780_178026

noncomputable def positive_reals (x y z : ℝ) : Prop :=
x > 0 ∧ y > 0 ∧ z > 0

noncomputable def triangle_inequality_for_n (x y z : ℝ) (n : ℕ) : Prop :=
(x^n + y^n > z^n) ∧ (y^n + z^n > x^n) ∧ (z^n + x^n > y^n)

theorem at_least_two_equal (x y z : ℝ) 
  (pos : positive_reals x y z) 
  (triangle_ineq: ∀ n : ℕ, n > 0 → triangle_inequality_for_n x y z n) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l1780_178026


namespace find_slower_train_speed_l1780_178049

theorem find_slower_train_speed (l : ℝ) (vf : ℝ) (t : ℝ) (v_s : ℝ) 
  (h1 : l = 37.5)   -- Length of each train
  (h2 : vf = 46)   -- Speed of the faster train in km/hr
  (h3 : t = 27)    -- Time in seconds to pass the slower train
  (h4 : (2 * l) = ((46 - v_s) * (5 / 18) * 27))   -- Distance covered at relative speed
  : v_s = 36 := 
sorry

end find_slower_train_speed_l1780_178049


namespace arithmetic_seq_a7_l1780_178066

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (d : ℕ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 8)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 7 = 6 := by
  sorry

end arithmetic_seq_a7_l1780_178066


namespace area_of_region_l1780_178036

noncomputable def region_area : ℝ :=
  sorry

theorem area_of_region :
  region_area = sorry := 
sorry

end area_of_region_l1780_178036
