import Mathlib

namespace jeremy_uncle_money_l49_49862

def total_cost (num_jerseys : Nat) (cost_per_jersey : Nat) (basketball_cost : Nat) (shorts_cost : Nat) : Nat :=
  (num_jerseys * cost_per_jersey) + basketball_cost + shorts_cost

def total_money_given (total_cost : Nat) (money_left : Nat) : Nat :=
  total_cost + money_left

theorem jeremy_uncle_money :
  total_money_given (total_cost 5 2 18 8) 14 = 50 :=
by
  sorry

end jeremy_uncle_money_l49_49862


namespace shifted_function_l49_49147

def initial_fun (x : ℝ) : ℝ := 5 * (x - 1) ^ 2 + 1

theorem shifted_function :
  (∀ x, initial_fun (x - 2) - 3 = 5 * (x + 1) ^ 2 - 2) :=
by
  intro x
  -- sorry statement to indicate proof should be here
  sorry

end shifted_function_l49_49147


namespace right_angled_triangle_l49_49385
  
theorem right_angled_triangle (x : ℝ) (hx : 0 < x) :
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  a^2 + b^2 = c^2 :=
by
  let a := 5 * x
  let b := 12 * x
  let c := 13 * x
  sorry

end right_angled_triangle_l49_49385


namespace number_of_maple_trees_planted_today_l49_49437

-- Define the initial conditions
def initial_maple_trees : ℕ := 2
def poplar_trees : ℕ := 5
def final_maple_trees : ℕ := 11

-- State the main proposition
theorem number_of_maple_trees_planted_today : 
  (final_maple_trees - initial_maple_trees) = 9 := by
  sorry

end number_of_maple_trees_planted_today_l49_49437


namespace problem_l49_49663

noncomputable def vector_a (ω φ x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x + φ), 1)
noncomputable def vector_b (ω φ x : ℝ) : ℝ × ℝ := (1, Real.cos (ω / 2 * x + φ))
noncomputable def f (ω φ x : ℝ) : ℝ := 
  let a := vector_a ω φ x
  let b := vector_b ω φ x
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)

theorem problem 
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 4)
  (h_period : Function.Periodic (f ω φ) 4)
  (h_point1 : f ω φ 1 = 1 / 2) : 
  ω = π / 2 ∧ ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f (π / 2) (π / 12) x ∧ f (π / 2) (π / 12) x ≤ 1 / 2 := 
by
  sorry

end problem_l49_49663


namespace total_value_of_coins_is_correct_l49_49840

def rolls_dollars : ℕ := 6
def rolls_half_dollars : ℕ := 5
def rolls_quarters : ℕ := 7
def rolls_dimes : ℕ := 4
def rolls_nickels : ℕ := 3
def rolls_pennies : ℕ := 2

def coins_per_dollar_roll : ℕ := 20
def coins_per_half_dollar_roll : ℕ := 25
def coins_per_quarter_roll : ℕ := 40
def coins_per_dime_roll : ℕ := 50
def coins_per_nickel_roll : ℕ := 40
def coins_per_penny_roll : ℕ := 50

def value_per_dollar : ℚ := 1
def value_per_half_dollar : ℚ := 0.5
def value_per_quarter : ℚ := 0.25
def value_per_dime : ℚ := 0.10
def value_per_nickel : ℚ := 0.05
def value_per_penny : ℚ := 0.01

theorem total_value_of_coins_is_correct : 
  rolls_dollars * coins_per_dollar_roll * value_per_dollar +
  rolls_half_dollars * coins_per_half_dollar_roll * value_per_half_dollar +
  rolls_quarters * coins_per_quarter_roll * value_per_quarter +
  rolls_dimes * coins_per_dime_roll * value_per_dime +
  rolls_nickels * coins_per_nickel_roll * value_per_nickel +
  rolls_pennies * coins_per_penny_roll * value_per_penny = 279.50 := 
sorry

end total_value_of_coins_is_correct_l49_49840


namespace width_of_river_l49_49348

def ferry_problem (v1 v2 W t1 t2 : ℝ) : Prop :=
  v1 * t1 + v2 * t1 = W ∧
  v1 * t1 = 720 ∧
  v2 * t1 = W - 720 ∧
  (v1 * t2 + v2 * t2 = 3 * W) ∧
  v1 * t2 = 2 * W - 400 ∧
  v2 * t2 = W + 400

theorem width_of_river 
  (v1 v2 W t1 t2 : ℝ)
  (h : ferry_problem v1 v2 W t1 t2) :
  W = 1280 :=
by
  sorry

end width_of_river_l49_49348


namespace range_of_a_l49_49592

noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n <= 7 then (3 - a) * n - 3 else a ^ (n - 6)

def increasing_seq (a : ℝ) (n : ℕ) : Prop :=
  a_n a n < a_n a (n + 1)

theorem range_of_a (a : ℝ) :
  (∀ n, increasing_seq a n) ↔ (9 / 4 < a ∧ a < 3) :=
sorry

end range_of_a_l49_49592


namespace tulip_area_of_flower_bed_l49_49060

theorem tulip_area_of_flower_bed 
  (CD CF : ℝ) (DE : ℝ := 4) (EF : ℝ := 3) 
  (triangle : ∀ (A B C : ℝ), A = B + C) : 
  CD * CF = 12 :=
by sorry

end tulip_area_of_flower_bed_l49_49060


namespace converse_statement_2_true_implies_option_A_l49_49764

theorem converse_statement_2_true_implies_option_A :
  (∀ x : ℕ, x = 1 ∨ x = 2 → (x^2 - 3 * x + 2 = 0)) →
  (x = 1 ∨ x = 2) :=
by
  intro h
  sorry

end converse_statement_2_true_implies_option_A_l49_49764


namespace conner_ties_sydney_l49_49805

def sydney_initial_collect := 837
def conner_initial_collect := 723

def sydney_collect_day_one := 4
def conner_collect_day_one := 8 * sydney_collect_day_one / 2

def sydney_collect_day_two := (sydney_initial_collect + sydney_collect_day_one) - ((sydney_initial_collect + sydney_collect_day_one) / 10)
def conner_collect_day_two := conner_initial_collect + conner_collect_day_one + 123

def sydney_collect_day_three := sydney_collect_day_two + 2 * conner_collect_day_one
def conner_collect_day_three := (conner_collect_day_two - (123 / 4))

theorem conner_ties_sydney :
  sydney_collect_day_three <= conner_collect_day_three :=
by
  sorry

end conner_ties_sydney_l49_49805


namespace trisha_walked_distance_l49_49588

theorem trisha_walked_distance :
  ∃ x : ℝ, (x + x + 0.67 = 0.89) ∧ (x = 0.11) :=
by sorry

end trisha_walked_distance_l49_49588


namespace sum_of_two_numbers_l49_49429

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end sum_of_two_numbers_l49_49429


namespace coordinates_of_point_P_l49_49613

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end coordinates_of_point_P_l49_49613


namespace investment_ratio_l49_49874

noncomputable def ratio_A_B (profit : ℝ) (profit_C : ℝ) (ratio_A_C : ℝ) (ratio_C_A : ℝ) := 
  3 / 1

theorem investment_ratio (total_profit : ℝ) (C_profit : ℝ) (A_C_ratio : ℝ) (C_A_ratio : ℝ) :
  total_profit = 60000 → C_profit = 20000 → A_C_ratio = 3 / 2 → ratio_A_B total_profit C_profit A_C_ratio C_A_ratio = 3 / 1 :=
by 
  intros h1 h2 h3
  sorry

end investment_ratio_l49_49874


namespace original_deck_size_l49_49922

-- Let's define the number of red and black cards initially
def numRedCards (r : ℕ) : ℕ := r
def numBlackCards (b : ℕ) : ℕ := b

-- Define the initial condition as given in the problem
def initial_prob_red (r b : ℕ) : Prop :=
  r / (r + b) = 2 / 5

-- Define the condition after adding 7 black cards
def prob_red_after_adding_black (r b : ℕ) : Prop :=
  r / (r + (b + 7)) = 1 / 3

-- The proof statement to verify original number of cards in the deck
theorem original_deck_size (r b : ℕ) (h1 : initial_prob_red r b) (h2 : prob_red_after_adding_black r b) : r + b = 35 := by
  sorry

end original_deck_size_l49_49922


namespace purple_gumdrops_after_replacement_l49_49861

def total_gumdrops : Nat := 200
def orange_percentage : Nat := 40
def purple_percentage : Nat := 10
def yellow_percentage : Nat := 25
def white_percentage : Nat := 15
def black_percentage : Nat := 10

def initial_orange_gumdrops := (orange_percentage * total_gumdrops) / 100
def initial_purple_gumdrops := (purple_percentage * total_gumdrops) / 100
def orange_to_purple := initial_orange_gumdrops / 3
def final_purple_gumdrops := initial_purple_gumdrops + orange_to_purple

theorem purple_gumdrops_after_replacement : final_purple_gumdrops = 47 := by
  sorry

end purple_gumdrops_after_replacement_l49_49861


namespace tangency_point_l49_49024

theorem tangency_point (x y : ℝ) : 
  y = x ^ 2 + 20 * x + 70 ∧ x = y ^ 2 + 70 * y + 1225 →
  (x, y) = (-19 / 2, -69 / 2) :=
by {
  sorry
}

end tangency_point_l49_49024


namespace arithmetic_sequence_a6_l49_49304

theorem arithmetic_sequence_a6 (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, ∃ d, a (n+1) = a n + d)
  (h_sum : a 4 + a 8 = 16) : a 6 = 8 :=
sorry

end arithmetic_sequence_a6_l49_49304


namespace rhombus_diagonal_length_l49_49214

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end rhombus_diagonal_length_l49_49214


namespace hunting_dogs_theorem_l49_49786

noncomputable def hunting_dogs_problem : Prop :=
  ∃ (courtiers : Finset (Finset (Fin 100))) (h1 : courtiers.card = 100),
  ∀ (c1 c2 : Finset (Fin 100)), c1 ∈ courtiers → c2 ∈ courtiers → c1 ≠ c2 → (c1 ∩ c2).card ≥ 2 → 
  ∃ (c₁ c₂ : Finset (Fin 100)), c₁ ∈ courtiers ∧ c₂ ∈ courtiers ∧ c₁ = c₂

theorem hunting_dogs_theorem : hunting_dogs_problem :=
sorry

end hunting_dogs_theorem_l49_49786


namespace roxy_garden_problem_l49_49219

variable (initial_flowering : ℕ)
variable (multiplier : ℕ)
variable (bought_flowering : ℕ)
variable (bought_fruiting : ℕ)
variable (given_flowering : ℕ)
variable (given_fruiting : ℕ)

def initial_fruiting (initial_flowering : ℕ) (multiplier : ℕ) : ℕ :=
  initial_flowering * multiplier

def saturday_flowering (initial_flowering : ℕ) (bought_flowering : ℕ) : ℕ :=
  initial_flowering + bought_flowering

def saturday_fruiting (initial_fruiting : ℕ) (bought_fruiting : ℕ) : ℕ :=
  initial_fruiting + bought_fruiting

def sunday_flowering (saturday_flowering : ℕ) (given_flowering : ℕ) : ℕ :=
  saturday_flowering - given_flowering

def sunday_fruiting (saturday_fruiting : ℕ) (given_fruiting : ℕ) : ℕ :=
  saturday_fruiting - given_fruiting

def total_plants_remaining (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  sunday_flowering + sunday_fruiting

theorem roxy_garden_problem 
  (h1 : initial_flowering = 7)
  (h2 : multiplier = 2)
  (h3 : bought_flowering = 3)
  (h4 : bought_fruiting = 2)
  (h5 : given_flowering = 1)
  (h6 : given_fruiting = 4) :
  total_plants_remaining 
    (sunday_flowering 
      (saturday_flowering initial_flowering bought_flowering) 
      given_flowering) 
    (sunday_fruiting 
      (saturday_fruiting 
        (initial_fruiting initial_flowering multiplier) 
        bought_fruiting) 
      given_fruiting) = 21 := 
  sorry

end roxy_garden_problem_l49_49219


namespace incorrect_operation_B_l49_49320

variables (a b c : ℝ)

theorem incorrect_operation_B : (c - 2 * (a + b)) ≠ (c - 2 * a + 2 * b) := by
  sorry

end incorrect_operation_B_l49_49320


namespace real_values_of_x_l49_49470

theorem real_values_of_x :
  {x : ℝ | (∃ y, y = (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ∧ y ≥ -1)} =
  {x | -1 ≤ x ∧ x < -1/3 ∨ -1/3 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 1 < x} := 
sorry

end real_values_of_x_l49_49470


namespace determine_k_for_quadratic_eq_l49_49571

theorem determine_k_for_quadratic_eq {k : ℝ} :
  (∀ r s : ℝ, 3 * r^2 + 5 * r + k = 0 ∧ 3 * s^2 + 5 * s + k = 0 →
    (|r + s| = r^2 + s^2)) ↔ k = -10/3 := by
sorry

end determine_k_for_quadratic_eq_l49_49571


namespace value_2_std_devs_below_mean_l49_49138

theorem value_2_std_devs_below_mean {μ σ : ℝ} (h_mean : μ = 10.5) (h_std_dev : σ = 1) : μ - 2 * σ = 8.5 :=
by
  sorry

end value_2_std_devs_below_mean_l49_49138


namespace bill_drew_12_triangles_l49_49145

theorem bill_drew_12_triangles 
  (T : ℕ)
  (total_lines : T * 3 + 8 * 4 + 4 * 5 = 88) : 
  T = 12 :=
sorry

end bill_drew_12_triangles_l49_49145


namespace find_hanyoung_weight_l49_49062

variable (H J : ℝ)

def hanyoung_is_lighter (H J : ℝ) : Prop := H = J - 4
def sum_of_weights (H J : ℝ) : Prop := H + J = 88

theorem find_hanyoung_weight (H J : ℝ) (h1 : hanyoung_is_lighter H J) (h2 : sum_of_weights H J) : H = 42 :=
by
  sorry

end find_hanyoung_weight_l49_49062


namespace max_surface_area_of_rectangular_solid_l49_49600

theorem max_surface_area_of_rectangular_solid {r a b c : ℝ} (h_sphere : 4 * π * r^2 = 4 * π)
  (h_diagonal : a^2 + b^2 + c^2 = (2 * r)^2) :
  2 * (a * b + a * c + b * c) ≤ 8 :=
by
  sorry

end max_surface_area_of_rectangular_solid_l49_49600


namespace fraction_of_300_greater_than_3_fifths_of_125_l49_49586

theorem fraction_of_300_greater_than_3_fifths_of_125 (f : ℚ)
    (h : f * 300 = 3 / 5 * 125 + 45) : 
    f = 2 / 5 :=
sorry

end fraction_of_300_greater_than_3_fifths_of_125_l49_49586


namespace sqrt_computation_l49_49668

open Real

theorem sqrt_computation : sqrt ((5: ℝ)^2 * (7: ℝ)^4) = 245 :=
by
  -- Proof here
  sorry

end sqrt_computation_l49_49668


namespace Kristyna_number_l49_49039

theorem Kristyna_number (k n : ℕ) (h1 : k = 6 * n + 3) (h2 : 3 * n + 1 + 2 * n = 1681) : k = 2019 := 
by
  -- Proof goes here
  sorry

end Kristyna_number_l49_49039


namespace ellipse_chord_slope_relation_l49_49671

theorem ellipse_chord_slope_relation
    (a b : ℝ) (h : a > b) (h1 : b > 0)
    (A B M : ℝ × ℝ)
    (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (hAB_slope : A.1 ≠ B.1)
    (K_AB K_OM : ℝ)
    (hK_AB : K_AB = (B.2 - A.2) / (B.1 - A.1))
    (hK_OM : K_OM = (M.2 - 0) / (M.1 - 0)) :
  K_AB * K_OM = - (b ^ 2) / (a ^ 2) := 
  sorry

end ellipse_chord_slope_relation_l49_49671


namespace product_of_B_coordinates_l49_49202

theorem product_of_B_coordinates :
  (∃ (x y : ℝ), (1 / 3 * x + 2 / 3 * 4 = 1 ∧ 1 / 3 * y + 2 / 3 * 2 = 7) ∧ x * y = -85) :=
by
  sorry

end product_of_B_coordinates_l49_49202


namespace count_valid_pairs_l49_49736

open Nat

-- Define the conditions
def room_conditions (p q : ℕ) : Prop :=
  q > p ∧
  (∃ (p' q' : ℕ), p = p' + 6 ∧ q = q' + 6 ∧ p' * q' = 48)

-- State the theorem to prove the number of valid pairs (p, q)
theorem count_valid_pairs : 
  (∃ l : List (ℕ × ℕ), 
    (∀ pq ∈ l, room_conditions pq.fst pq.snd) ∧ 
    l.length = 5) := 
sorry

end count_valid_pairs_l49_49736


namespace x_plus_p_eq_2p_plus_3_l49_49707

theorem x_plus_p_eq_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 := by
  sorry

end x_plus_p_eq_2p_plus_3_l49_49707


namespace solution_correct_l49_49453

noncomputable def solve_system (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let x := (3 * c - a - b) / 4
  let y := (3 * b - a - c) / 4
  let z := (3 * a - b - c) / 4
  (x, y, z)

theorem solution_correct (a b c : ℝ) (x y z : ℝ) :
  (x + y + 2 * z = a) →
  (x + 2 * y + z = b) →
  (2 * x + y + z = c) →
  (x, y, z) = solve_system a b c :=
by sorry

end solution_correct_l49_49453


namespace percentage_passed_exam_l49_49659

theorem percentage_passed_exam (total_students failed_students : ℕ) (h_total : total_students = 540) (h_failed : failed_students = 351) :
  (total_students - failed_students) * 100 / total_students = 35 :=
by
  sorry

end percentage_passed_exam_l49_49659


namespace polynomial_evaluation_l49_49335

theorem polynomial_evaluation (a : ℝ) (h : a^2 + 3 * a = 2) : 2 * a^2 + 6 * a - 10 = -6 := by
  sorry

end polynomial_evaluation_l49_49335


namespace minimum_value_l49_49966

theorem minimum_value (x : ℝ) (hx : 0 < x) : ∃ y, (y = x + 4 / (x + 1)) ∧ (∀ z, (x > 0 → z = x + 4 / (x + 1)) → 3 ≤ z) := sorry

end minimum_value_l49_49966


namespace polyomino_count_5_l49_49411

-- Definition of distinct polyomino counts for n = 2, 3, and 4.
def polyomino_count (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 5
  else 0

-- Theorem stating the distinct polyomino count for n = 5
theorem polyomino_count_5 : polyomino_count 5 = 12 :=
by {
  -- Proof steps would go here, but for now we use sorry.
  sorry
}

end polyomino_count_5_l49_49411


namespace points_lie_on_hyperbola_l49_49655

noncomputable
def point_on_hyperbola (t : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ 
    (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) }

theorem points_lie_on_hyperbola : 
  ∀ t : ℝ, ∀ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0 ∧ x - 3 * t * y + 4 = 0) → (x^2 / 16) - (y^2 / 1) = 1 :=
by 
  intro t x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end points_lie_on_hyperbola_l49_49655


namespace find_time_eating_dinner_l49_49914

def total_flight_time : ℕ := 11 * 60 + 20
def time_reading : ℕ := 2 * 60
def time_watching_movies : ℕ := 4 * 60
def time_listening_radio : ℕ := 40
def time_playing_games : ℕ := 1 * 60 + 10
def time_nap : ℕ := 3 * 60

theorem find_time_eating_dinner : 
  total_flight_time - (time_reading + time_watching_movies + time_listening_radio + time_playing_games + time_nap) = 30 := 
by
  sorry

end find_time_eating_dinner_l49_49914


namespace rectangle_area_l49_49808

theorem rectangle_area (P L W : ℝ) (hP : P = 2 * (L + W)) (hRatio : L / W = 5 / 2) (hP_val : P = 280) : 
  L * W = 4000 :=
by 
  sorry

end rectangle_area_l49_49808


namespace alpha_plus_beta_l49_49766

theorem alpha_plus_beta :
  (∃ α β : ℝ, 
    (∀ x : ℝ, x ≠ -β ∧ x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1980) / (x^2 + 70 * x - 3570))
  ) → (∃ α β : ℝ, α + β = 123) :=
by {
  sorry
}

end alpha_plus_beta_l49_49766


namespace knights_on_red_chairs_l49_49181

variable (K L Kr Lb : ℕ)
variable (h1 : K + L = 20)
variable (h2 : Kr + Lb = 10)
variable (h3 : Kr = L - Lb)

/-- Given the conditions:
1. There are 20 seats with knights and liars such that K + L = 20.
2. Half of the individuals claim to be sitting on blue chairs, and half on red chairs such that Kr + Lb = 10.
3. Knights on red chairs (Kr) must be equal to liars minus liars on blue chairs (Lb).
Prove that the number of knights now sitting on red chairs is 5. -/
theorem knights_on_red_chairs : Kr = 5 :=
by
  sorry

end knights_on_red_chairs_l49_49181


namespace infinite_solutions_of_system_l49_49036

theorem infinite_solutions_of_system :
  ∃x y : ℝ, (3 * x - 4 * y = 10 ∧ 6 * x - 8 * y = 20) :=
by
  sorry

end infinite_solutions_of_system_l49_49036


namespace tshirt_costs_more_than_jersey_l49_49033

open Nat

def cost_tshirt : ℕ := 192
def cost_jersey : ℕ := 34

theorem tshirt_costs_more_than_jersey :
  cost_tshirt - cost_jersey = 158 :=
by sorry

end tshirt_costs_more_than_jersey_l49_49033


namespace remainder_sum_division_by_9_l49_49306

theorem remainder_sum_division_by_9 :
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 :=
by
  sorry

end remainder_sum_division_by_9_l49_49306


namespace roots_are_distinct_and_negative_l49_49234

theorem roots_are_distinct_and_negative : 
  (∀ x : ℝ, x^2 + m * x + 1 = 0 → ∃! (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2) ↔ m > 2 :=
by
  sorry

end roots_are_distinct_and_negative_l49_49234


namespace total_groups_l49_49640

-- Define the problem conditions
def boys : ℕ := 9
def girls : ℕ := 12

-- Calculate the required combinations
def C (n k: ℕ) : ℕ := n.choose k
def groups_with_two_boys_one_girl : ℕ := C boys 2 * C girls 1
def groups_with_two_girls_one_boy : ℕ := C girls 2 * C boys 1

-- Statement of the theorem to prove
theorem total_groups : groups_with_two_boys_one_girl + groups_with_two_girls_one_boy = 1026 := 
by sorry

end total_groups_l49_49640


namespace number_of_terms_in_arithmetic_sequence_l49_49264

noncomputable def arithmetic_sequence_terms (a d n : ℕ) : Prop :=
  let sum_first_three := 3 * a + 3 * d = 34
  let sum_last_three := 3 * a + 3 * (n - 1) * d = 146
  let sum_all := n * (2 * a + (n - 1) * d) / 2 = 390
  (sum_first_three ∧ sum_last_three ∧ sum_all) → n = 13

theorem number_of_terms_in_arithmetic_sequence (a d n : ℕ) : arithmetic_sequence_terms a d n → n = 13 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l49_49264


namespace necessary_but_not_sufficient_l49_49142

theorem necessary_but_not_sufficient (A B : Prop) (h : A → B) : ¬ (B → A) :=
sorry

end necessary_but_not_sufficient_l49_49142


namespace isosceles_right_triangle_area_l49_49696

theorem isosceles_right_triangle_area (p : ℝ) : 
  ∃ (A : ℝ), A = (3 - 2 * Real.sqrt 2) * p^2 
  → (∃ (x : ℝ), 2 * x + x * Real.sqrt 2 = 2 * p ∧ A = 1 / 2 * x^2) := 
sorry

end isosceles_right_triangle_area_l49_49696


namespace logical_equivalence_l49_49199

theorem logical_equivalence (P Q R : Prop) :
  ((¬ P ∧ ¬ Q) → ¬ R) ↔ (R → (P ∨ Q)) :=
by sorry

end logical_equivalence_l49_49199


namespace sophomores_stratified_sampling_l49_49811

theorem sophomores_stratified_sampling 
  (total_students freshmen sophomores seniors selected_total : ℕ) 
  (H1 : total_students = 2800) 
  (H2 : freshmen = 970) 
  (H3 : sophomores = 930) 
  (H4 : seniors = 900) 
  (H_selected_total : selected_total = 280) : 
  (sophomores / total_students) * selected_total = 93 :=
by sorry

end sophomores_stratified_sampling_l49_49811


namespace rectangle_width_is_4_l49_49339

-- Definitions of conditions
variable (w : ℝ) -- width of the rectangle
def length := w + 2 -- length of the rectangle
def perimeter := 2 * w + 2 * (w + 2) -- perimeter of the rectangle, using given conditions

-- The theorem to be proved
theorem rectangle_width_is_4 (h : perimeter = 20) : w = 4 :=
by {
  sorry -- To be proved
}

end rectangle_width_is_4_l49_49339


namespace polynomial_evaluation_l49_49365

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 3 * x - 10 = 0) (h2 : 0 < x) : 
  x^3 - 3 * x^2 - 10 * x + 5 = 5 :=
sorry

end polynomial_evaluation_l49_49365


namespace max_xy_min_function_l49_49019

-- Problem 1: Prove that the maximum value of xy is 8 given the conditions
theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 8) : xy ≤ 8 :=
sorry

-- Problem 2: Prove that the minimum value of the function is 9 given the conditions
theorem min_function (x : ℝ) (hx : -1 < x) : (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

end max_xy_min_function_l49_49019


namespace last_digit_of_2_pow_2010_l49_49020

theorem last_digit_of_2_pow_2010 : (2 ^ 2010) % 10 = 4 :=
by
  sorry

end last_digit_of_2_pow_2010_l49_49020


namespace sum_of_reciprocals_squares_l49_49192

theorem sum_of_reciprocals_squares (a b : ℕ) (h : a * b = 17) :
  (1 : ℚ) / (a * a) + 1 / (b * b) = 290 / 289 :=
sorry

end sum_of_reciprocals_squares_l49_49192


namespace purple_marble_probability_l49_49097

theorem purple_marble_probability (P_blue P_green P_purple : ℝ) (h1 : P_blue = 0.35) (h2 : P_green = 0.45) (h3 : P_blue + P_green + P_purple = 1) :
  P_purple = 0.2 := 
by sorry

end purple_marble_probability_l49_49097


namespace Alton_profit_l49_49227

variable (earnings_per_day : ℕ)
variable (days_per_week : ℕ)
variable (rent_per_week : ℕ)

theorem Alton_profit (h1 : earnings_per_day = 8) (h2 : days_per_week = 7) (h3 : rent_per_week = 20) :
  earnings_per_day * days_per_week - rent_per_week = 36 := 
by sorry

end Alton_profit_l49_49227


namespace cookies_per_child_l49_49965

theorem cookies_per_child (total_cookies : ℕ) (adults : ℕ) (children : ℕ) (fraction_eaten_by_adults : ℚ) 
  (h1 : total_cookies = 120) (h2 : adults = 2) (h3 : children = 4) (h4 : fraction_eaten_by_adults = 1/3) :
  total_cookies * (1 - fraction_eaten_by_adults) / children = 20 := 
by
  sorry

end cookies_per_child_l49_49965


namespace store_breaks_even_l49_49963

-- Defining the conditions based on the problem statement.
def cost_price_piece1 (profitable : ℝ → Prop) : Prop :=
  ∃ x, profitable x ∧ 1.5 * x = 150

def cost_price_piece2 (loss : ℝ → Prop) : Prop :=
  ∃ y, loss y ∧ 0.75 * y = 150

def profitable (x : ℝ) : Prop := x + 0.5 * x = 150
def loss (y : ℝ) : Prop := y - 0.25 * y = 150

-- Store breaks even if the total cost price equals the total selling price
theorem store_breaks_even (x y : ℝ)
  (P1 : cost_price_piece1 profitable)
  (P2 : cost_price_piece2 loss) :
  (x + y = 100 + 200) → (150 + 150) = 300 :=
by
  sorry

end store_breaks_even_l49_49963


namespace perimeter_correct_l49_49104

open EuclideanGeometry

noncomputable def perimeter_of_figure : ℝ := 
  let AB : ℝ := 6
  let BC : ℝ := AB
  let AD : ℝ := AB / 2
  let DC : ℝ := AD
  let DE : ℝ := AD
  let EA : ℝ := DE
  let EF : ℝ := EA / 2
  let FG : ℝ := EF
  let GH : ℝ := FG / 2
  let HJ : ℝ := GH
  let JA : ℝ := HJ
  AB + BC + DC + DE + EF + FG + GH + HJ + JA

theorem perimeter_correct : perimeter_of_figure = 23.25 :=
by
  -- proof steps would go here, but are not required for this problem transformation
  sorry

end perimeter_correct_l49_49104


namespace Aryan_owes_1200_l49_49362

variables (A K : ℝ) -- A represents Aryan's debt, K represents Kyro's debt

-- Condition 1: Aryan's debt is twice Kyro's debt
axiom condition1 : A = 2 * K

-- Condition 2: Aryan pays 60% of her debt
axiom condition2 : (0.60 * A) + (0.80 * K) = 1500 - 300

theorem Aryan_owes_1200 : A = 1200 :=
by
  sorry

end Aryan_owes_1200_l49_49362


namespace no_positive_integer_solution_l49_49612

theorem no_positive_integer_solution (p x y : ℕ) (hp : Nat.Prime p) (hp_gt3 : p > 3) 
  (h_p_div_x : p ∣ x) (hx_pos : 0 < x) (hy_pos : 0 < y) : x^2 - 1 ≠ y^p :=
sorry

end no_positive_integer_solution_l49_49612


namespace problem_statement_l49_49018

def Omega (n : ℕ) : ℕ := 
  -- Number of prime factors of n, counting multiplicity
  sorry

def f1 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 1 (mod 4)
  sorry

def f3 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 3 (mod 4)
  sorry

theorem problem_statement : 
  f3 (6 ^ 2020) - f1 (6 ^ 2020) = (1 / 10 : ℚ) * (6 ^ 2021 - 3 ^ 2021 - 2 ^ 2021 - 1) :=
sorry

end problem_statement_l49_49018


namespace set_A_main_inequality_l49_49852

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 2|
def A : Set ℝ := {x | f x < 3}

theorem set_A :
  A = {x | -2 / 3 < x ∧ x < 0} :=
sorry

theorem main_inequality (s t : ℝ) (hs : -2 / 3 < s ∧ s < 0) (ht : -2 / 3 < t ∧ t < 0) :
  |1 - t / s| < |t - 1 / s| :=
sorry

end set_A_main_inequality_l49_49852


namespace faster_car_distance_l49_49357

theorem faster_car_distance (d v : ℝ) (h_dist: d + 2 * d = 4) (h_faster: 2 * v = 2 * (d / v)) : 
  d = 4 / 3 → 2 * d = 8 / 3 :=
by sorry

end faster_car_distance_l49_49357


namespace slope_of_line_l49_49769

theorem slope_of_line (s x y : ℝ) (h1 : 2 * x + 3 * y = 8 * s + 5) (h2 : x + 2 * y = 3 * s + 2) :
  ∃ m c : ℝ, ∀ x y, x = m * y + c ∧ m = -7/2 :=
by
  sorry

end slope_of_line_l49_49769


namespace tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l49_49444

-- Condition: Given tan(α) = 2
variable (α : ℝ) (h₀ : Real.tan α = 2)

-- Statement (1): Prove tan(2α + π/4) = 9
theorem tan_double_alpha_plus_pi_over_four :
  Real.tan (2 * α + Real.pi / 4) = 9 := by
  sorry

-- Statement (2): Prove (6 sin α + cos α) / (3 sin α - 2 cos α) = 13 / 4
theorem sin_cos_fraction :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 := by
  sorry

end tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l49_49444


namespace extremum_points_l49_49400

noncomputable def f (x1 x2 : ℝ) : ℝ := x1 * x2 / (1 + x1^2 * x2^2)

theorem extremum_points :
  (f 0 0 = 0) ∧
  (∀ x1 : ℝ, f x1 (-1 / x1) = -1 / 2) ∧
  (∀ x1 : ℝ, f x1 (1 / x1) = 1 / 2) ∧
  ∀ y1 y2 : ℝ, (f 0 0 < f y1 y2 → (0 < y1 ∧ 0 < y2)) ∧ 
             (f 0 0 > f y1 y2 → (0 > y1 ∧ 0 > y2)) :=
by
  sorry

end extremum_points_l49_49400


namespace x_to_the_12_eq_14449_l49_49821

/-
Given the condition x + 1/x = 2*sqrt(2), prove that x^12 = 14449.
-/

theorem x_to_the_12_eq_14449 (x : ℂ) (hx : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := 
sorry

end x_to_the_12_eq_14449_l49_49821


namespace find_a_l49_49032

theorem find_a 
  (a : ℝ) 
  (h : 1 - 2 * a = a - 2) 
  (h1 : 1 - 2 * a = a - 2) 
  : a = 1 := 
by 
  -- proof goes here
  sorry

end find_a_l49_49032


namespace mandy_used_nutmeg_l49_49483

theorem mandy_used_nutmeg (x : ℝ) (h1 : 0.67 = x + 0.17) : x = 0.50 :=
  by
  sorry

end mandy_used_nutmeg_l49_49483


namespace number_of_students_in_third_batch_l49_49678

theorem number_of_students_in_third_batch
  (avg1 avg2 avg3 : ℕ)
  (total_avg : ℚ)
  (students1 students2 : ℕ)
  (h_avg1 : avg1 = 45)
  (h_avg2 : avg2 = 55)
  (h_avg3 : avg3 = 65)
  (h_total_avg : total_avg = 56.333333333333336)
  (h_students1 : students1 = 40)
  (h_students2 : students2 = 50) :
  ∃ x : ℕ, (students1 * avg1 + students2 * avg2 + x * avg3 = total_avg * (students1 + students2 + x) ∧ x = 60) :=
by
  sorry

end number_of_students_in_third_batch_l49_49678


namespace evaluate_expression_l49_49280

theorem evaluate_expression : (2014 - 2013) * (2013 - 2012) = 1 := 
by sorry

end evaluate_expression_l49_49280


namespace range_of_d_l49_49957

theorem range_of_d (d : ℝ) : (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) ↔ d ≥ 1 :=
sorry

end range_of_d_l49_49957


namespace pencil_eraser_cost_l49_49700

theorem pencil_eraser_cost (p e : ℕ) (h1 : 15 * p + 5 * e = 200) (h2 : p > e) (h_p_pos : p > 0) (h_e_pos : e > 0) :
  p + e = 18 :=
  sorry

end pencil_eraser_cost_l49_49700


namespace marbles_remainder_l49_49405

theorem marbles_remainder 
  (g r p : ℕ) 
  (hg : g % 8 = 5) 
  (hr : r % 7 = 2) 
  (hp : p % 7 = 4) : 
  (r + p + g) % 7 = 4 := 
sorry

end marbles_remainder_l49_49405


namespace actual_distance_traveled_l49_49743

variable (t : ℝ) -- let t be the actual time in hours
variable (d : ℝ) -- let d be the actual distance traveled at 12 km/hr

-- Conditions
def condition1 := 20 * t = 12 * t + 30
def condition2 := d = 12 * t

-- The target we want to prove
theorem actual_distance_traveled (t : ℝ) (d : ℝ) (h1 : condition1 t) (h2 : condition2 t d) : 
  d = 45 := by
  sorry

end actual_distance_traveled_l49_49743


namespace minimum_value_of_f_l49_49903

def f (x : ℝ) : ℝ := |3 - x| + |x - 2|

theorem minimum_value_of_f : ∃ x0 : ℝ, (∀ x : ℝ, f x0 ≤ f x) ∧ f x0 = 1 := 
by
  sorry

end minimum_value_of_f_l49_49903


namespace max_value_phi_l49_49686

theorem max_value_phi (φ : ℝ) (hφ : -Real.pi / 2 < φ ∧ φ < Real.pi / 2) :
  (∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2 - Real.pi / 3) →
  φ = Real.pi / 6 :=
by 
  intro h
  sorry

end max_value_phi_l49_49686


namespace largest_angle_in_pentagon_l49_49779

theorem largest_angle_in_pentagon (P Q R S T : ℝ) 
          (h1 : P = 70) 
          (h2 : Q = 100)
          (h3 : R = S) 
          (h4 : T = 3 * R - 25)
          (h5 : P + Q + R + S + T = 540) : 
          T = 212 :=
by
  sorry

end largest_angle_in_pentagon_l49_49779


namespace max_non_attacking_rooks_l49_49484

theorem max_non_attacking_rooks (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 299) (h3 : 1 ≤ b) (h4 : b ≤ 299) :
  ∃ max_rooks : ℕ, max_rooks = 400 :=
  sorry

end max_non_attacking_rooks_l49_49484


namespace timeAfter2687Minutes_l49_49249

-- We define a structure for representing time in hours and minutes.
structure Time :=
  (hour : Nat)
  (minute : Nat)

-- Define the current time
def currentTime : Time := {hour := 7, minute := 0}

-- Define a function that computes the time after adding a given number of minutes to a given time
noncomputable def addMinutes (t : Time) (minutesToAdd : Nat) : Time :=
  let totalMinutes := t.minute + minutesToAdd
  let extraHours := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  let totalHours := t.hour + extraHours
  let effectiveHours := totalHours % 24
  {hour := effectiveHours, minute := remainingMinutes}

-- The theorem to state that 2687 minutes after 7:00 a.m. is 3:47 a.m.
theorem timeAfter2687Minutes : addMinutes currentTime 2687 = { hour := 3, minute := 47 } :=
  sorry

end timeAfter2687Minutes_l49_49249


namespace find_function_expression_find_range_of_m_l49_49013

-- Statement for Part 1
theorem find_function_expression (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) : 
  y = -1/2 * x - 2 := 
sorry

-- Statement for Part 2
theorem find_range_of_m (m x : ℝ) (hx : x > -2) (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) :
  (-x + m < -1/2 * x - 2) ↔ (m ≤ -3) := 
sorry

end find_function_expression_find_range_of_m_l49_49013


namespace integer_solutions_exist_l49_49832

theorem integer_solutions_exist (a : ℕ) (ha : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 := 
sorry

end integer_solutions_exist_l49_49832


namespace find_radius_of_cone_l49_49682

def slant_height : ℝ := 10
def curved_surface_area : ℝ := 157.07963267948966

theorem find_radius_of_cone
    (l : ℝ) (CSA : ℝ) (h1 : l = slant_height) (h2 : CSA = curved_surface_area) :
    ∃ r : ℝ, r = 5 := 
by
  sorry

end find_radius_of_cone_l49_49682


namespace work_rate_c_l49_49846

theorem work_rate_c (A B C : ℝ) 
  (h1 : A + B = 1 / 15) 
  (h2 : A + B + C = 1 / 5) :
  (1 / C) = 7.5 :=
by 
  sorry

end work_rate_c_l49_49846


namespace intersect_in_third_quadrant_l49_49551

theorem intersect_in_third_quadrant (b : ℝ) : (¬ (∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0)) ↔ b > 3 / 2 := sorry

end intersect_in_third_quadrant_l49_49551


namespace nancy_kept_tortilla_chips_l49_49294

theorem nancy_kept_tortilla_chips (initial_chips : ℕ) (chips_to_brother : ℕ) (chips_to_sister : ℕ) (remaining_chips : ℕ) 
  (h1 : initial_chips = 22) 
  (h2 : chips_to_brother = 7) 
  (h3 : chips_to_sister = 5) 
  (h_total_given : initial_chips - (chips_to_brother + chips_to_sister) = remaining_chips) :
  remaining_chips = 10 :=
sorry

end nancy_kept_tortilla_chips_l49_49294


namespace andrei_stamps_l49_49855

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end andrei_stamps_l49_49855


namespace steven_falls_correct_l49_49102

/-
  We will model the problem where we are given the conditions about the falls of Steven, Stephanie,
  and Sonya, and then prove that the number of times Steven fell is 3.
-/

variables (S : ℕ) -- Steven's falls

-- Conditions
def stephanie_falls := S + 13
def sonya_falls := 6 
def sonya_condition := 6 = (stephanie_falls / 2) - 2

-- Theorem statement
theorem steven_falls_correct : S = 3 :=
by {
  -- Note: the actual proof steps would go here, but are omitted per instructions
  sorry
}

end steven_falls_correct_l49_49102


namespace oblique_prism_volume_l49_49285

noncomputable def volume_of_oblique_prism 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : ℝ :=
  a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2)

theorem oblique_prism_volume 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : volume_of_oblique_prism a b c α β hα hβ = a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2) := 
by
  -- Proof will be completed here
  sorry

end oblique_prism_volume_l49_49285


namespace mario_hibiscus_l49_49239

def hibiscus_flowers (F : ℕ) : Prop :=
  let F2 := 2 * F
  let F3 := 4 * F2
  F + F2 + F3 = 22 → F = 2

theorem mario_hibiscus (F : ℕ) : hibiscus_flowers F :=
  sorry

end mario_hibiscus_l49_49239


namespace weekly_income_l49_49619

-- Defining the daily catches
def blue_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 10
  | "Tuesday"   => 8
  | "Wednesday" => 12
  | "Thursday"  => 6
  | "Friday"    => 14
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

def red_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 14
  | "Tuesday"   => 16
  | "Wednesday" => 10
  | "Thursday"  => 18
  | "Friday"    => 12
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

-- Prices per crab
def price_per_blue_crab : ℕ := 6
def price_per_red_crab : ℕ := 4
def buckets : ℕ := 8

-- Daily income calculation
def daily_income (day : String) : ℕ :=
  let blue_income := (blue_crabs_per_bucket day) * buckets * price_per_blue_crab
  let red_income := (red_crabs_per_bucket day) * buckets * price_per_red_crab
  blue_income + red_income

-- Proving the weekly income is $6080
theorem weekly_income : 
  (daily_income "Monday" +
  daily_income "Tuesday" +
  daily_income "Wednesday" +
  daily_income "Thursday" +
  daily_income "Friday" +
  daily_income "Saturday" +
  daily_income "Sunday") = 6080 :=
by sorry

end weekly_income_l49_49619


namespace opposite_event_of_hitting_at_least_once_is_missing_both_times_l49_49211

theorem opposite_event_of_hitting_at_least_once_is_missing_both_times
  (A B : Prop) :
  ¬(A ∨ B) ↔ (¬A ∧ ¬B) :=
by
  sorry

end opposite_event_of_hitting_at_least_once_is_missing_both_times_l49_49211


namespace hexagon_colorings_correct_l49_49491

def valid_hexagon_colorings : Prop :=
  ∃ (colors : Fin 6 → Fin 7),
    (colors 0 ≠ colors 1) ∧
    (colors 1 ≠ colors 2) ∧
    (colors 2 ≠ colors 3) ∧
    (colors 3 ≠ colors 4) ∧
    (colors 4 ≠ colors 5) ∧
    (colors 5 ≠ colors 0) ∧
    (colors 0 ≠ colors 2) ∧
    (colors 1 ≠ colors 3) ∧
    (colors 2 ≠ colors 4) ∧
    (colors 3 ≠ colors 5) ∧
    ∃! (n : Nat), n = 12600

theorem hexagon_colorings_correct : valid_hexagon_colorings :=
sorry

end hexagon_colorings_correct_l49_49491


namespace grain_spilled_correct_l49_49942

variable (original_grain : ℕ) (remaining_grain : ℕ) (grain_spilled : ℕ)

theorem grain_spilled_correct : 
  original_grain = 50870 → remaining_grain = 918 → grain_spilled = original_grain - remaining_grain → grain_spilled = 49952 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end grain_spilled_correct_l49_49942


namespace jogs_per_day_l49_49845

-- Definitions of conditions
def weekdays_per_week : ℕ := 5
def total_weeks : ℕ := 3
def total_miles : ℕ := 75

-- Define the number of weekdays in total weeks
def total_weekdays : ℕ := total_weeks * weekdays_per_week

-- Theorem to prove Damien jogs 5 miles per day on weekdays
theorem jogs_per_day : total_miles / total_weekdays = 5 := by
  sorry

end jogs_per_day_l49_49845


namespace expected_value_m_plus_n_l49_49445

-- Define the main structures and conditions
def spinner_sectors : List ℚ := [-1.25, -1, 0, 1, 1.25]
def initial_value : ℚ := 1

-- Define a function that returns the largest expected value on the paper
noncomputable def expected_largest_written_value (sectors : List ℚ) (initial : ℚ) : ℚ :=
  -- The expected value calculation based on the problem and solution analysis
  11/6  -- This is derived from the correct solution steps not shown here

-- Define the final claim
theorem expected_value_m_plus_n :
  let m := 11
  let n := 6
  expected_largest_written_value spinner_sectors initial_value = 11/6 → m + n = 17 :=
by sorry

end expected_value_m_plus_n_l49_49445


namespace hike_down_distance_l49_49908

theorem hike_down_distance :
  let rate_up := 4 -- rate going up in miles per day
  let time := 2    -- time in days
  let rate_down := 1.5 * rate_up -- rate going down in miles per day
  let distance_down := rate_down * time -- distance going down in miles
  distance_down = 12 :=
by
  sorry

end hike_down_distance_l49_49908


namespace minimum_m_l49_49565

/-
  Given that for all 2 ≤ x ≤ 3, 3 ≤ y ≤ 6, the inequality mx^2 - xy + y^2 ≥ 0 always holds,
  prove that the minimum value of the real number m is 0.
-/
theorem minimum_m (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x * y + y^2 ≥ 0) → m = 0 :=
sorry -- proof to be provided

end minimum_m_l49_49565


namespace not_divisible_by_3_l49_49790

theorem not_divisible_by_3 (n : ℤ) : (n^2 + 1) % 3 ≠ 0 := by
  sorry

end not_divisible_by_3_l49_49790


namespace Jaco_total_gift_budget_l49_49830

theorem Jaco_total_gift_budget :
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  friends_gifts + parents_gifts = 100 :=
by
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  show friends_gifts + parents_gifts = 100
  sorry

end Jaco_total_gift_budget_l49_49830


namespace problem1_problem2_l49_49150

open Real

variables {α β γ : ℝ}

theorem problem1 (α β : ℝ) :
  abs (cos (α + β)) ≤ abs (cos α) + abs (sin β) ∧
  abs (sin (α + β)) ≤ abs (cos α) + abs (cos β) :=
sorry

theorem problem2 (h : α + β + γ = 0) :
  abs (cos α) + abs (cos β) + abs (cos γ) ≥ 1 :=
sorry

end problem1_problem2_l49_49150


namespace ted_speed_l49_49756

variables (T F : ℝ)

-- Ted runs two-thirds as fast as Frank
def condition1 : Prop := T = (2 / 3) * F

-- In two hours, Frank runs eight miles farther than Ted
def condition2 : Prop := 2 * F = 2 * T + 8

-- Prove that Ted runs at a speed of 8 mph
theorem ted_speed (h1 : condition1 T F) (h2 : condition2 T F) : T = 8 :=
by
  sorry

end ted_speed_l49_49756


namespace probability_one_white_one_black_l49_49880

def white_ball_count : ℕ := 8
def black_ball_count : ℕ := 7
def total_ball_count : ℕ := white_ball_count + black_ball_count
def total_ways_to_choose_2_balls : ℕ := total_ball_count.choose 2
def favorable_ways : ℕ := white_ball_count * black_ball_count

theorem probability_one_white_one_black : 
  (favorable_ways : ℚ) / (total_ways_to_choose_2_balls : ℚ) = 8 / 15 :=
by
  sorry

end probability_one_white_one_black_l49_49880


namespace inequality_solution_set_l49_49639

noncomputable def solution_set : Set ℝ := { x : ℝ | x > 5 ∨ x < -2 }

theorem inequality_solution_set (x : ℝ) :
  x^2 - 3 * x - 10 > 0 ↔ x > 5 ∨ x < -2 :=
by
  sorry

end inequality_solution_set_l49_49639


namespace original_radius_of_cylinder_l49_49645

theorem original_radius_of_cylinder (r z : ℝ) (h : ℝ := 3) :
  z = 3 * π * ((r + 8)^2 - r^2) → z = 8 * π * r^2 → r = 8 :=
by
  intros hz1 hz2
  -- Translate given conditions into their equivalent expressions and equations
  sorry

end original_radius_of_cylinder_l49_49645


namespace fractional_part_of_blue_square_four_changes_l49_49165

theorem fractional_part_of_blue_square_four_changes 
  (initial_area : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ (a : ℝ), f a = (8 / 9) * a) :
  (f^[4]) initial_area / initial_area = 4096 / 6561 :=
by
  sorry

end fractional_part_of_blue_square_four_changes_l49_49165


namespace exists_integer_multiple_of_3_2008_l49_49133

theorem exists_integer_multiple_of_3_2008 :
  ∃ k : ℤ, 3 ^ 2008 ∣ (k ^ 3 - 36 * k ^ 2 + 51 * k - 97) :=
sorry

end exists_integer_multiple_of_3_2008_l49_49133


namespace total_weight_is_28_87_l49_49328

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12
def green_ball_weight : ℝ := 4.25

def red_ball_weight : ℝ := 2 * green_ball_weight
def yellow_ball_weight : ℝ := red_ball_weight - 1.5

def total_weight : ℝ := blue_ball_weight + brown_ball_weight + green_ball_weight + red_ball_weight + yellow_ball_weight

theorem total_weight_is_28_87 : total_weight = 28.87 :=
by
  /- proof goes here -/
  sorry

end total_weight_is_28_87_l49_49328


namespace distribution_properties_l49_49422

theorem distribution_properties (m d j s k : ℝ) (h1 : True)
  (h2 : True)
  (h3 : True)
  (h4 : 68 ≤ 100 ∧ 68 ≥ 0) -- 68% being a valid percentage
  : j = 84 ∧ s = s ∧ k = k :=
by
  -- sorry is used to highlight the proof is not included
  sorry

end distribution_properties_l49_49422


namespace problem_equiv_none_of_these_l49_49174

variable {x y : ℝ}

theorem problem_equiv_none_of_these (hx : x ≠ 0) (hx3 : x ≠ 3) (hy : y ≠ 0) (hy5 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) →
  ¬(3 * x + 2 * y = x * y) ∧
  ¬(y = 3 * x / (5 - y)) ∧
  ¬(x / 3 + y / 2 = 3) ∧
  ¬(3 * y / (y - 5) = x) :=
sorry

end problem_equiv_none_of_these_l49_49174


namespace problem_l49_49077

variable (a : ℕ → ℝ) -- {a_n} is a sequence
variable (S : ℕ → ℝ) -- S_n represents the sum of the first n terms
variable (d : ℝ) -- non-zero common difference
variable (a1 : ℝ) -- first term of the sequence

-- Define an arithmetic sequence with common difference d and first term a1
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (a1 : ℝ) 
  (h_non_zero : d ≠ 0)
  (h_sequence : is_arithmetic_sequence a d a1)
  (h_sum : sum_of_arithmetic_sequence S a)
  (h_S5_eq_S6 : S 5 = S 6) :
  S 11 = 0 := 
sorry

end problem_l49_49077


namespace equal_intercepts_l49_49158

theorem equal_intercepts (a : ℝ) (h : ∃p, (a * p, 0) = (0, a - 2)) : a = 1 ∨ a = 2 :=
sorry

end equal_intercepts_l49_49158


namespace M_subset_N_l49_49116

def M (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 2) + (Real.pi / 4)
def N (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 4) + (Real.pi / 2)

theorem M_subset_N : ∀ x, M x → N x := 
by
  sorry

end M_subset_N_l49_49116


namespace fernanda_savings_calc_l49_49495

noncomputable def aryan_debt : ℝ := 1200
noncomputable def kyro_debt : ℝ := aryan_debt / 2
noncomputable def aryan_payment : ℝ := 0.60 * aryan_debt
noncomputable def kyro_payment : ℝ := 0.80 * kyro_debt
noncomputable def initial_savings : ℝ := 300
noncomputable def total_payment_received : ℝ := aryan_payment + kyro_payment
noncomputable def total_savings : ℝ := initial_savings + total_payment_received

theorem fernanda_savings_calc : total_savings = 1500 := by
  sorry

end fernanda_savings_calc_l49_49495


namespace expand_expression_l49_49627

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 :=
by 
  sorry

end expand_expression_l49_49627


namespace c_negative_l49_49130

theorem c_negative (a b c : ℝ) (h₁ : a + b + c < 0) (h₂ : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 :=
sorry

end c_negative_l49_49130


namespace optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l49_49526

-- Definitions of the options
def optionA : ℕ := 2019^2 - 2014^2
def optionB : ℕ := 2019^2 * 10^2
def optionC : ℕ := 2020^2 / 101^2
def optionD : ℕ := 2010^2 - 2005^2
def optionE : ℕ := 2015^2 / 5^2

-- Statements to be proven
theorem optionA_is_multiple_of_5 : optionA % 5 = 0 := by sorry
theorem optionB_is_multiple_of_5 : optionB % 5 = 0 := by sorry
theorem optionC_is_multiple_of_5 : optionC % 5 = 0 := by sorry
theorem optionD_is_multiple_of_5 : optionD % 5 = 0 := by sorry
theorem optionE_is_not_multiple_of_5 : optionE % 5 ≠ 0 := by sorry

end optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l49_49526


namespace all_lucky_years_l49_49223

def is_lucky_year (y : ℕ) : Prop :=
  ∃ m d : ℕ, 1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 31 ∧ (m * d = y % 100)

theorem all_lucky_years :
  (is_lucky_year 2024) ∧ (is_lucky_year 2025) ∧ (is_lucky_year 2026) ∧ (is_lucky_year 2027) ∧ (is_lucky_year 2028) :=
sorry

end all_lucky_years_l49_49223


namespace shaded_area_fraction_l49_49816

-- Define the problem conditions
def total_squares : ℕ := 18
def half_squares : ℕ := 10
def whole_squares : ℕ := 3

-- Define the total shaded area given the conditions
def shaded_area := (half_squares * (1/2) + whole_squares)

-- Define the total area of the rectangle
def total_area := total_squares

-- Lean 4 theorem statement
theorem shaded_area_fraction :
  shaded_area / total_area = (4 : ℚ) / 9 :=
by sorry

end shaded_area_fraction_l49_49816


namespace graph_passes_through_point_l49_49205

noncomputable def exponential_shift (a : ℝ) (x : ℝ) := a^(x - 2)

theorem graph_passes_through_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : exponential_shift a 2 = 1 :=
by
  unfold exponential_shift
  sorry

end graph_passes_through_point_l49_49205


namespace cone_slice_ratio_l49_49263

theorem cone_slice_ratio (h r : ℝ) (hb : h > 0) (hr : r > 0) :
    let V1 := (1/3) * π * (5*r)^2 * (5*h) - (1/3) * π * (4*r)^2 * (4*h)
    let V2 := (1/3) * π * (4*r)^2 * (4*h) - (1/3) * π * (3*r)^2 * (3*h)
    V2 / V1 = 37 / 61 := by {
  sorry
}

end cone_slice_ratio_l49_49263


namespace shobha_current_age_l49_49953

variable (S B : ℕ)
variable (h_ratio : 4 * B = 3 * S)
variable (h_future_age : S + 6 = 26)

theorem shobha_current_age : B = 15 :=
by
  sorry

end shobha_current_age_l49_49953


namespace fraction_subtraction_l49_49193

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) 
  = 9 / 20 := by
  sorry

end fraction_subtraction_l49_49193


namespace sum_of_reciprocals_l49_49477

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) : 
  (1 / x) + (1 / y) = 8 := 
by 
  sorry

end sum_of_reciprocals_l49_49477


namespace remainder_zero_when_divided_by_condition_l49_49066

noncomputable def remainder_problem (x : ℂ) : ℂ :=
  (2 * x^5 - x^4 + x^2 - 1) * (x^3 - 1)

theorem remainder_zero_when_divided_by_condition (x : ℂ) (h : x^2 - x + 1 = 0) :
  remainder_problem x % (x^2 - x + 1) = 0 := by
  sorry

end remainder_zero_when_divided_by_condition_l49_49066


namespace maxwells_walking_speed_l49_49434

theorem maxwells_walking_speed 
    (brad_speed : ℕ) 
    (distance_between_homes : ℕ) 
    (maxwell_distance : ℕ)
    (meeting : maxwell_distance = 12)
    (brad_speed_condition : brad_speed = 6)
    (distance_between_homes_condition: distance_between_homes = 36) : 
    (maxwell_distance / (distance_between_homes - maxwell_distance) * brad_speed ) = 3 := by
  sorry

end maxwells_walking_speed_l49_49434


namespace part1_part2_case1_part2_case2_part2_case3_part3_l49_49481

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end part1_part2_case1_part2_case2_part2_case3_part3_l49_49481


namespace peanuts_added_l49_49073

theorem peanuts_added (initial_peanuts final_peanuts added_peanuts : ℕ) 
(h1 : initial_peanuts = 10) 
(h2 : final_peanuts = 18) 
(h3 : final_peanuts = initial_peanuts + added_peanuts) : 
added_peanuts = 8 := 
by {
  sorry
}

end peanuts_added_l49_49073


namespace appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l49_49309

-- Definitions for binomial coefficient and Pascal's triangle

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Check occurrences in Pascal's triangle more than three times
theorem appears_more_than_three_times_in_Pascal (n : ℕ) :
  n = 10 ∨ n = 15 ∨ n = 21 → ∃ a b c : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ 
    (binomial_coeff a 2 = n ∨ binomial_coeff a 3 = n) ∧
    (binomial_coeff b 2 = n ∨ binomial_coeff b 3 = n) ∧
    (binomial_coeff c 2 = n ∨ binomial_coeff c 3 = n) := 
by
  sorry

-- Check occurrences in Pascal's triangle more than four times
theorem appears_more_than_four_times_in_Pascal (n : ℕ) :
  n = 120 ∨ n = 210 ∨ n = 3003 → ∃ a b c d : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ (1 < d) ∧ 
    (binomial_coeff a 3 = n ∨ binomial_coeff a 4 = n) ∧
    (binomial_coeff b 3 = n ∨ binomial_coeff b 4 = n) ∧
    (binomial_coeff c 3 = n ∨ binomial_coeff c 4 = n) ∧
    (binomial_coeff d 3 = n ∨ binomial_coeff d 4 = n) := 
by
  sorry

end appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l49_49309


namespace man_speed_l49_49762

theorem man_speed (time_in_minutes : ℝ) (distance_in_km : ℝ) (T : time_in_minutes = 24) (D : distance_in_km = 4) : 
  (distance_in_km / (time_in_minutes / 60)) = 10 := by
  sorry

end man_speed_l49_49762


namespace derivative_of_f_l49_49072

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) :=
by
  intro x
  -- We skip the proof here
  sorry

end derivative_of_f_l49_49072


namespace MsSatosClassRatioProof_l49_49402

variable (g b : ℕ) -- g is the number of girls, b is the number of boys

def MsSatosClassRatioProblem : Prop :=
  (g = b + 6) ∧ (g + b = 32) → g / b = 19 / 13

theorem MsSatosClassRatioProof : MsSatosClassRatioProblem g b := by
  sorry

end MsSatosClassRatioProof_l49_49402


namespace b5_b9_equal_16_l49_49674

-- Define the arithmetic sequence and conditions
variables {a : ℕ → ℝ} (h_arith : ∀ n m, a m = a n + (m - n) * (a 1 - a 0))
variable (h_non_zero : ∀ n, a n ≠ 0)
variable (h_cond : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)

-- Define the geometric sequence and condition
variables {b : ℕ → ℝ} (h_geom : ∀ n, b (n + 1) = b n * (b 1 / b 0))
variable (h_b7 : b 7 = a 7)

-- State the theorem to prove
theorem b5_b9_equal_16 : b 5 * b 9 = 16 :=
sorry

end b5_b9_equal_16_l49_49674


namespace find_union_of_sets_l49_49604

-- Define the sets A and B in terms of a
def A (a : ℤ) : Set ℤ := { n | n = |a + 1| ∨ n = 3 ∨ n = 5 }
def B (a : ℤ) : Set ℤ := { n | n = 2 * a + 1 ∨ n = a^2 + 2 * a ∨ n = a^2 + 2 * a - 1 }

-- Given condition: A ∩ B = {2, 3}
def condition (a : ℤ) : Prop := A a ∩ B a = {2, 3}

-- The correct answer: A ∪ B = {-5, 2, 3, 5}
theorem find_union_of_sets (a : ℤ) (h : condition a) : A a ∪ B a = {-5, 2, 3, 5} :=
sorry

end find_union_of_sets_l49_49604


namespace find_x_l49_49933

theorem find_x (x : ℝ) : 0.003 + 0.158 + x = 2.911 → x = 2.750 :=
by
  sorry

end find_x_l49_49933


namespace remainder_when_divided_by_20_l49_49436

theorem remainder_when_divided_by_20
  (a b : ℤ) 
  (h1 : a % 60 = 49)
  (h2 : b % 40 = 29) :
  (a + b) % 20 = 18 :=
by
  sorry

end remainder_when_divided_by_20_l49_49436


namespace length_XW_l49_49112

theorem length_XW {XY XZ YZ XW : ℝ}
  (hXY : XY = 15)
  (hXZ : XZ = 17)
  (hAngle : XY^2 + YZ^2 = XZ^2)
  (hYZ : YZ = 8) :
  XW = 15 :=
by
  sorry

end length_XW_l49_49112


namespace at_least_one_is_one_l49_49446

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a + b + c = (1 / a) + (1 / b) + (1 / c)) 
  (h2 : a * b * c = 1) : a = 1 ∨ b = 1 ∨ c = 1 := 
by 
  sorry

end at_least_one_is_one_l49_49446


namespace sum_interior_angles_l49_49154

theorem sum_interior_angles (n : ℕ) (h : 180 * (n - 2) = 3240) : 180 * ((n + 3) - 2) = 3780 := by
  sorry

end sum_interior_angles_l49_49154


namespace grocer_display_rows_l49_49355

theorem grocer_display_rows (n : ℕ)
  (h1 : ∃ k, k = 2 + 3 * (n - 1))
  (h2 : ∃ s, s = (n / 2) * (2 + (3 * n - 1))):
  (3 * n^2 + n) / 2 = 225 → n = 12 :=
by
  sorry

end grocer_display_rows_l49_49355


namespace possible_measures_of_angle_X_l49_49501

theorem possible_measures_of_angle_X : 
  ∃ n : ℕ, n = 17 ∧ (∀ (X Y : ℕ), 
    X > 0 ∧ Y > 0 ∧ X + Y = 180 ∧ 
    ∃ m : ℕ, m ≥ 1 ∧ X = m * Y) :=
sorry

end possible_measures_of_angle_X_l49_49501


namespace eve_discovers_secret_l49_49454

theorem eve_discovers_secret (x : ℕ) : ∃ (n : ℕ), ∃ (is_prime : ℕ → Prop), (∀ m : ℕ, (is_prime (x + n * m)) ∨ (¬is_prime (x + n * m))) :=
  sorry

end eve_discovers_secret_l49_49454


namespace equal_a_b_l49_49296

theorem equal_a_b (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_n : 0 < n) 
  (h_eq : (a + b)^n - (a - b)^n = (a / b) * ((a + b)^n + (a - b)^n)) : a = b :=
sorry

end equal_a_b_l49_49296


namespace determine_phi_l49_49986

theorem determine_phi (f : ℝ → ℝ) (φ : ℝ): 
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + 3 * φ)) ∧ 
  (∀ x : ℝ, f (-x) = -f x) → 
  (∃ k : ℤ, φ = k * Real.pi / 3) :=
by 
  sorry

end determine_phi_l49_49986


namespace pascal_triangle_fifth_number_l49_49375

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l49_49375


namespace iron_weighs_more_l49_49043

-- Define the weights of the metal pieces
def weight_iron : ℝ := 11.17
def weight_aluminum : ℝ := 0.83

-- State the theorem to prove that the difference in weights is 10.34 pounds
theorem iron_weighs_more : weight_iron - weight_aluminum = 10.34 :=
by sorry

end iron_weighs_more_l49_49043


namespace wall_building_time_l49_49699

theorem wall_building_time (m1 m2 d1 d2 k : ℕ) (h1 : m1 = 12) (h2 : d1 = 6) (h3 : m2 = 18) (h4 : k = 72) 
  (condition : m1 * d1 = k) (rate_const : m2 * d2 = k) : d2 = 4 := by
  sorry

end wall_building_time_l49_49699


namespace rectangle_area_l49_49394

theorem rectangle_area (b l : ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end rectangle_area_l49_49394


namespace smallest_b_factor_2020_l49_49308

theorem smallest_b_factor_2020 :
  ∃ b : ℕ, b > 0 ∧
  (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ b = r + s) ∧
  (∀ c : ℕ, c > 0 → (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ c = r + s) → b ≤ c) ∧
  b = 121 :=
sorry

end smallest_b_factor_2020_l49_49308


namespace combined_income_is_16800_l49_49990

-- Given conditions
def ErnieOldIncome : ℕ := 6000
def ErnieCurrentIncome : ℕ := (4 * ErnieOldIncome) / 5
def JackCurrentIncome : ℕ := 2 * ErnieOldIncome

-- Proof that their combined income is $16800
theorem combined_income_is_16800 : ErnieCurrentIncome + JackCurrentIncome = 16800 := by
  sorry

end combined_income_is_16800_l49_49990


namespace alpha_beta_roots_l49_49590

theorem alpha_beta_roots (α β : ℝ) (hαβ1 : α^2 + α - 1 = 0) (hαβ2 : β^2 + β - 1 = 0) (h_sum : α + β = -1) :
  α^4 - 3 * β = 5 :=
by
  sorry

end alpha_beta_roots_l49_49590


namespace suitable_survey_set_l49_49124

def Survey1 := "Investigate the lifespan of a batch of light bulbs"
def Survey2 := "Investigate the household income situation in a city"
def Survey3 := "Investigate the vision of students in a class"
def Survey4 := "Investigate the efficacy of a certain drug"

-- Define what it means for a survey to be suitable for sample surveys
def suitable_for_sample_survey (survey : String) : Prop :=
  survey = Survey1 ∨ survey = Survey2 ∨ survey = Survey4

-- The question is to prove that the surveys suitable for sample surveys include exactly (1), (2), and (4).
theorem suitable_survey_set :
  {Survey1, Survey2, Survey4} = {s : String | suitable_for_sample_survey s} :=
by
  sorry

end suitable_survey_set_l49_49124


namespace paul_books_sold_l49_49279

theorem paul_books_sold:
  ∀ (initial_books friend_books sold_per_day days final_books sold_books: ℝ),
    initial_books = 284.5 →
    friend_books = 63.7 →
    sold_per_day = 16.25 →
    days = 8 →
    final_books = 112.3 →
    sold_books = initial_books - friend_books - final_books →
    sold_books = 108.5 :=
by intros initial_books friend_books sold_per_day days final_books sold_books
   sorry

end paul_books_sold_l49_49279


namespace algebraic_expression_value_l49_49620

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 1 + Real.sqrt 2) (h2 : b = Real.sqrt 3) : 
  a^2 + b^2 - 2 * a + 1 = 5 := 
by
  sorry

end algebraic_expression_value_l49_49620


namespace quadratic_discriminant_l49_49185

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/5) (1/5) = 576 / 25 := by
  sorry

end quadratic_discriminant_l49_49185


namespace number_of_representations_l49_49314

-- Definitions of the conditions
def is_valid_b (b : ℕ) : Prop :=
  b ≤ 99

def is_representation (b3 b2 b1 b0 : ℕ) : Prop :=
  3152 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0

-- The theorem to prove
theorem number_of_representations : 
  ∃ (N' : ℕ), (N' = 316) ∧ 
  (∀ (b3 b2 b1 b0 : ℕ), is_representation b3 b2 b1 b0 → is_valid_b b0 → is_valid_b b1 → is_valid_b b2 → is_valid_b b3) :=
sorry

end number_of_representations_l49_49314


namespace sufficient_but_not_necessary_l49_49638

-- Definitions of propositions p and q
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2
def q (a b : ℝ) : Prop := a < b

-- Problem statement as a Lean theorem
theorem sufficient_but_not_necessary (a b m : ℝ) : 
  (p a b m → q a b) ∧ (¬ (q a b → p a b m)) :=
by
  sorry

end sufficient_but_not_necessary_l49_49638


namespace sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l49_49190

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l49_49190


namespace triangle_area_via_line_eq_l49_49153

theorem triangle_area_via_line_eq (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  area = 1 / (2 * |a * b|) :=
by
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  sorry

end triangle_area_via_line_eq_l49_49153


namespace arman_is_6_times_older_than_sister_l49_49931

def sisterWasTwoYearsOldFourYearsAgo := 2
def yearsAgo := 4
def armansAgeInFourYears := 40

def currentAgeOfSister := sisterWasTwoYearsOldFourYearsAgo + yearsAgo
def currentAgeOfArman := armansAgeInFourYears - yearsAgo

theorem arman_is_6_times_older_than_sister :
  currentAgeOfArman = 6 * currentAgeOfSister :=
by
  sorry

end arman_is_6_times_older_than_sister_l49_49931


namespace find_g_l49_49836

noncomputable def g : ℝ → ℝ
| x => 2 * (4^x - 3^x)

theorem find_g :
  (g 1 = 2) ∧
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) →
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end find_g_l49_49836


namespace redPoints_l49_49726

open Nat

def isRedPoint (x y : ℕ) : Prop :=
  (y = (x - 36) * (x - 144) - 1991) ∧ (∃ m : ℕ, y = m * m)

theorem redPoints :
  {p : ℕ × ℕ | isRedPoint p.1 p.2} = { (2544, 6017209), (444, 120409) } :=
by
  sorry

end redPoints_l49_49726


namespace complement_setP_in_U_l49_49063

def setU : Set ℝ := {x | -1 < x ∧ x < 3}
def setP : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem complement_setP_in_U : (setU \ setP) = {x | 2 < x ∧ x < 3} :=
by
  sorry

end complement_setP_in_U_l49_49063


namespace rate_per_sq_meter_l49_49042

theorem rate_per_sq_meter
  (L : ℝ) (W : ℝ) (total_cost : ℝ)
  (hL : L = 6) (hW : W = 4.75) (h_total_cost : total_cost = 25650) :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_l49_49042


namespace Isabel_reading_pages_l49_49275

def pages_of_math_homework : ℕ := 2
def problems_per_page : ℕ := 5
def total_problems : ℕ := 30

def math_problems : ℕ := pages_of_math_homework * problems_per_page
def reading_problems : ℕ := total_problems - math_problems

theorem Isabel_reading_pages : (reading_problems / problems_per_page) = 4 :=
by
  sorry

end Isabel_reading_pages_l49_49275


namespace saved_money_is_30_l49_49860

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end saved_money_is_30_l49_49860


namespace adults_attended_l49_49164

def adult_ticket_cost : ℕ := 25
def children_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400

theorem adults_attended (A C: ℕ) (h1 : adult_ticket_cost * A + children_ticket_cost * C = total_receipts)
                       (h2 : A + C = total_attendance) : A = 120 :=
by
  sorry

end adults_attended_l49_49164


namespace two_painters_days_l49_49408

-- Define the conditions and the proof problem
def five_painters_days : ℕ := 5
def days_per_five_painters : ℕ := 2
def total_painter_days : ℕ := five_painters_days * days_per_five_painters -- Total painter-days for the original scenario
def two_painters : ℕ := 2
def last_day_painter_half_day : ℕ := 1 -- Indicating that one painter works half a day on the last day
def last_day_work : ℕ := two_painters - last_day_painter_half_day / 2 -- Total work on the last day is equivalent to 1.5 painter-days

theorem two_painters_days : total_painter_days = 5 :=
by
  sorry -- Mathematical proof goes here

end two_painters_days_l49_49408


namespace principal_amount_l49_49763

theorem principal_amount (P : ℕ) (R : ℕ) (T : ℕ) (SI : ℕ) 
  (h1 : R = 12)
  (h2 : T = 10)
  (h3 : SI = 1500) 
  (h4 : SI = (P * R * T) / 100) : P = 1250 :=
by sorry

end principal_amount_l49_49763


namespace equation_solution_l49_49172

theorem equation_solution (x y : ℕ) (h : x^3 - y^3 = x * y + 61) : x = 6 ∧ y = 5 :=
by
  sorry

end equation_solution_l49_49172


namespace xy_sum_values_l49_49004

theorem xy_sum_values (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end xy_sum_values_l49_49004


namespace find_g_neg_3_l49_49349

def g (x : ℤ) : ℤ :=
if x < 1 then 3 * x - 4 else x + 6

theorem find_g_neg_3 : g (-3) = -13 :=
by
  -- proof omitted: sorry
  sorry

end find_g_neg_3_l49_49349


namespace cos_double_angle_l49_49123

theorem cos_double_angle (α : ℝ) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : Real.cos α = 1 / 3 :=
sorry

end cos_double_angle_l49_49123


namespace b_catches_A_distance_l49_49899

noncomputable def speed_A := 10 -- kmph
noncomputable def speed_B := 20 -- kmph
noncomputable def time_diff := 7 -- hours
noncomputable def distance_A := speed_A * time_diff -- km
noncomputable def relative_speed := speed_B - speed_A -- kmph
noncomputable def catch_up_time := distance_A / relative_speed -- hours
noncomputable def distance_B := speed_B * catch_up_time -- km

theorem b_catches_A_distance :
  distance_B = 140 := by
  sorry

end b_catches_A_distance_l49_49899


namespace find_starting_number_l49_49386

theorem find_starting_number (x : ℝ) (h : ((x - 2 + 4) / 1) / 2 * 8 = 77) : x = 17.25 := by
  sorry

end find_starting_number_l49_49386


namespace problem_a2_b_c_in_M_l49_49795

def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem problem_a2_b_c_in_M (a b c : ℤ) (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
sorry

end problem_a2_b_c_in_M_l49_49795


namespace aaron_already_had_lids_l49_49581

-- Definitions for conditions
def number_of_boxes : ℕ := 3
def can_lids_per_box : ℕ := 13
def total_can_lids : ℕ := 53
def lids_from_boxes : ℕ := number_of_boxes * can_lids_per_box

-- The statement to be proven
theorem aaron_already_had_lids : total_can_lids - lids_from_boxes = 14 := 
by
  sorry

end aaron_already_had_lids_l49_49581


namespace cubic_yard_to_cubic_meter_l49_49101

/-- Define the conversion from yards to meters. -/
def yard_to_meter : ℝ := 0.9144

/-- Theorem stating how many cubic meters are in one cubic yard. -/
theorem cubic_yard_to_cubic_meter :
  (yard_to_meter ^ 3 : ℝ) = 0.7636 :=
by
  sorry

end cubic_yard_to_cubic_meter_l49_49101


namespace rational_numbers_inequality_l49_49340

theorem rational_numbers_inequality (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 :=
sorry

end rational_numbers_inequality_l49_49340


namespace problem_part1_problem_part2_l49_49740

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l49_49740


namespace divisible_bc_ad_l49_49938

theorem divisible_bc_ad
  (a b c d u : ℤ)
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) :
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end divisible_bc_ad_l49_49938


namespace central_angle_star_in_polygon_l49_49528

theorem central_angle_star_in_polygon (n : ℕ) (h : 2 < n) : 
  ∃ C, C = 720 / n :=
by sorry

end central_angle_star_in_polygon_l49_49528


namespace least_value_r_minus_p_l49_49001

theorem least_value_r_minus_p (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 5) :
  ∃ r p, r = 5 ∧ p = 1/2 ∧ r - p = 9 / 2 :=
by
  sorry

end least_value_r_minus_p_l49_49001


namespace negation_of_universal_proposition_l49_49146

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^3 - 8 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^3 - 8 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l49_49146


namespace vec_eqn_solution_l49_49910

theorem vec_eqn_solution :
  ∀ m : ℝ, let a : ℝ × ℝ := (1, -2) 
           let b : ℝ × ℝ := (m, 4) 
           (a.1 * b.2 = a.2 * b.1) → 2 • a - b = (4, -8) :=
by
  intro m a b h_parallel
  sorry

end vec_eqn_solution_l49_49910


namespace work_completion_in_8_days_l49_49739

/-- Definition of the individual work rates and the combined work rate. -/
def work_rate_A := 1 / 12
def work_rate_B := 1 / 24
def combined_work_rate := work_rate_A + work_rate_B

/-- The main theorem stating that A and B together complete the job in 8 days. -/
theorem work_completion_in_8_days (h1 : work_rate_A = 1 / 12) (h2 : work_rate_B = 1 / 24) : 
  1 / combined_work_rate = 8 :=
by
  sorry

end work_completion_in_8_days_l49_49739


namespace result_of_dividing_295_by_5_and_adding_6_is_65_l49_49392

theorem result_of_dividing_295_by_5_and_adding_6_is_65 : (295 / 5) + 6 = 65 := by
  sorry

end result_of_dividing_295_by_5_and_adding_6_is_65_l49_49392


namespace two_bedroom_units_l49_49005

theorem two_bedroom_units {x y : ℕ} 
  (h1 : x + y = 12) 
  (h2 : 360 * x + 450 * y = 4950) : 
  y = 7 := 
by
  sorry

end two_bedroom_units_l49_49005


namespace large_pizza_slices_l49_49366

-- Definitions and conditions based on the given problem
def slicesEatenByPhilAndre : ℕ := 9 * 2
def slicesLeft : ℕ := 2 * 2
def slicesOnSmallCheesePizza : ℕ := 8
def totalSlices : ℕ := slicesEatenByPhilAndre + slicesLeft

-- The theorem to be proven
theorem large_pizza_slices (slicesEatenByPhilAndre slicesLeft slicesOnSmallCheesePizza : ℕ) :
  slicesEatenByPhilAndre = 18 ∧ slicesLeft = 4 ∧ slicesOnSmallCheesePizza = 8 →
  totalSlices - slicesOnSmallCheesePizza = 14 :=
by
  intros h
  sorry

end large_pizza_slices_l49_49366


namespace functional_relationship_minimum_wage_l49_49465

/-- Problem setup and conditions --/
def total_area : ℝ := 1200
def team_A_rate : ℝ := 100
def team_B_rate : ℝ := 50
def team_A_wage : ℝ := 4000
def team_B_wage : ℝ := 3000
def min_days_A : ℝ := 3

/-- Prove Part 1: y as a function of x --/
def y_of_x (x : ℝ) : ℝ := 24 - 2 * x

theorem functional_relationship (x : ℝ) :
  100 * x + 50 * y_of_x x = total_area := by
  sorry

/-- Prove Part 2: Minimum wage calculation --/
def total_wage (a b : ℝ) : ℝ := team_A_wage * a + team_B_wage * b

theorem minimum_wage :
  ∀ (a b : ℝ), 3 ≤ a → a ≤ b → b = 24 - 2 * a → 
  total_wage a b = 56000 → a = 8 ∧ b = 8 := by
  sorry

end functional_relationship_minimum_wage_l49_49465


namespace calculate_f_g2_l49_49787

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x^3 - 1

theorem calculate_f_g2 : f (g 2) = 226 := by
  sorry

end calculate_f_g2_l49_49787


namespace geometric_series_sum_l49_49189

theorem geometric_series_sum :
  (1 / 5 - 1 / 25 + 1 / 125 - 1 / 625 + 1 / 3125) = 521 / 3125 :=
by
  sorry

end geometric_series_sum_l49_49189


namespace sin_double_angle_l49_49170

noncomputable def r := Real.sqrt 5
noncomputable def sin_α := -2 / r
noncomputable def cos_α := 1 / r
noncomputable def sin_2α := 2 * sin_α * cos_α

theorem sin_double_angle (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (1, -2) ∧ ∃ α : ℝ, true) → sin_2α = -4 / 5 :=
by
  sorry

end sin_double_angle_l49_49170


namespace max_possible_salary_l49_49240

-- Definition of the conditions
def num_players : ℕ := 25
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 800000

-- The theorem we want to prove: the maximum possible salary for a single player is $320,000
theorem max_possible_salary (total_salary_cap : ℕ) (num_players : ℕ) (min_salary : ℕ) :
  total_salary_cap - (num_players - 1) * min_salary = 320000 :=
by sorry

end max_possible_salary_l49_49240


namespace find_value_l49_49728

def set_condition (s : Set ℕ) : Prop := s = {0, 1, 2}

def one_relationship_correct (a b c : ℕ) : Prop :=
  (a ≠ 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b = 2 ∧ c = 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c ≠ 0)
  ∨ (a ≠ 2 ∧ b = 0 ∧ c ≠ 0)

theorem find_value (a b c : ℕ) (h1 : set_condition {a, b, c}) (h2 : one_relationship_correct a b c) :
  100 * c + 10 * b + a = 102 :=
sorry

end find_value_l49_49728


namespace simplify_fraction_l49_49761

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2) - (4 * a - 4) / (a - 2)) = a - 2 :=
  sorry

end simplify_fraction_l49_49761


namespace invalid_votes_percentage_l49_49909

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l49_49909


namespace degrees_for_lemon_pie_l49_49532

theorem degrees_for_lemon_pie 
    (total_students : ℕ)
    (chocolate_lovers : ℕ)
    (apple_lovers : ℕ)
    (blueberry_lovers : ℕ)
    (remaining_students : ℕ)
    (lemon_pie_degrees : ℝ) :
    total_students = 42 →
    chocolate_lovers = 15 →
    apple_lovers = 9 →
    blueberry_lovers = 7 →
    remaining_students = total_students - (chocolate_lovers + apple_lovers + blueberry_lovers) →
    lemon_pie_degrees = (remaining_students / 2 / total_students * 360) →
    lemon_pie_degrees = 47.14 :=
by
  intros _ _ _ _ _ _
  sorry

end degrees_for_lemon_pie_l49_49532


namespace interest_equality_l49_49541

theorem interest_equality (total_sum : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) (n : ℝ) :
  total_sum = 2730 ∧ part1 = 1050 ∧ part2 = 1680 ∧
  rate1 = 3 ∧ time1 = 8 ∧ rate2 = 5 ∧ part1 * rate1 * time1 = part2 * rate2 * n →
  n = 3 :=
by
  sorry

end interest_equality_l49_49541


namespace Yoongi_score_is_53_l49_49566

-- Define the scores of the three students
variables (score_Yoongi score_Eunji score_Yuna : ℕ)

-- Define the conditions given in the problem
axiom Yoongi_Eunji : score_Eunji = score_Yoongi - 25
axiom Eunji_Yuna  : score_Yuna = score_Eunji - 20
axiom Yuna_score  : score_Yuna = 8

theorem Yoongi_score_is_53 : score_Yoongi = 53 := by
  sorry

end Yoongi_score_is_53_l49_49566


namespace find_n_for_integer_roots_l49_49166

theorem find_n_for_integer_roots (n : ℤ):
    (∃ x y : ℤ, x ≠ y ∧ x^2 + (n+1)*x + (2*n - 1) = 0 ∧ y^2 + (n+1)*y + (2*n - 1) = 0) →
    (n = 1 ∨ n = 5) :=
sorry

end find_n_for_integer_roots_l49_49166


namespace reciprocal_of_complex_power_l49_49350

noncomputable def complex_num_reciprocal : ℂ :=
  (Complex.I) ^ 2023

theorem reciprocal_of_complex_power :
  ∀ z : ℂ, z = (Complex.I) ^ 2023 -> (1 / z) = Complex.I :=
by
  intro z
  intro hz
  have h_power : z = Complex.I ^ 2023 := by assumption
  sorry

end reciprocal_of_complex_power_l49_49350


namespace probability_of_three_even_numbers_l49_49094

theorem probability_of_three_even_numbers (n : ℕ) (k : ℕ) (p_even : ℚ) (p_odd : ℚ) (comb : ℕ → ℕ → ℕ) 
    (h_n : n = 5) (h_k : k = 3) (h_p_even : p_even = 1/2) (h_p_odd : p_odd = 1/2) 
    (h_comb : comb 5 3 = 10) :
    comb n k * (p_even ^ k) * (p_odd ^ (n - k)) = 5 / 16 :=
by sorry

end probability_of_three_even_numbers_l49_49094


namespace fencing_required_l49_49084

theorem fencing_required
  (L : ℝ) (A : ℝ) (h_L : L = 20) (h_A : A = 400) : 
  (2 * (A / L) + L) = 60 :=
by
  sorry

end fencing_required_l49_49084


namespace lowest_price_eq_195_l49_49901

def cost_per_component : ℕ := 80
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_costs : ℕ := 16500
def num_components : ℕ := 150

theorem lowest_price_eq_195 
  (cost_per_component shipping_cost_per_unit fixed_monthly_costs num_components : ℕ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 5)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : num_components = 150) :
  (fixed_monthly_costs + num_components * (cost_per_component + shipping_cost_per_unit)) / num_components = 195 :=
by
  sorry

end lowest_price_eq_195_l49_49901


namespace additional_votes_in_revote_l49_49770

theorem additional_votes_in_revote (a b a' b' n : ℕ) :
  a + b = 300 →
  b - a = n →
  a' - b' = 3 * n →
  a' + b' = 300 →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end additional_votes_in_revote_l49_49770


namespace ball_hits_ground_time_l49_49803

def ball_height (t : ℝ) : ℝ := -20 * t^2 + 30 * t + 60

theorem ball_hits_ground_time :
  ∃ t : ℝ, ball_height t = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
sorry

end ball_hits_ground_time_l49_49803


namespace find_asterisk_value_l49_49537

theorem find_asterisk_value : 
  (∃ x : ℕ, (x / 21) * (x / 189) = 1) → x = 63 :=
by
  intro h
  sorry

end find_asterisk_value_l49_49537


namespace train_length_is_250_l49_49331

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) (station_length : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600 * time_sec) - station_length

theorem train_length_is_250 :
  train_length 36 45 200 = 250 :=
by
  sorry

end train_length_is_250_l49_49331


namespace wait_time_difference_l49_49372

noncomputable def kids_waiting_for_swings : ℕ := 3
noncomputable def kids_waiting_for_slide : ℕ := 2 * kids_waiting_for_swings
noncomputable def wait_per_kid_swings : ℕ := 2 * 60 -- 2 minutes in seconds
noncomputable def wait_per_kid_slide : ℕ := 15 -- in seconds

noncomputable def total_wait_swings : ℕ := kids_waiting_for_swings * wait_per_kid_swings
noncomputable def total_wait_slide : ℕ := kids_waiting_for_slide * wait_per_kid_slide

theorem wait_time_difference : total_wait_swings - total_wait_slide = 270 := by
  sorry

end wait_time_difference_l49_49372


namespace inequality_hold_l49_49650

theorem inequality_hold (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - |c| > b - |c| :=
sorry

end inequality_hold_l49_49650


namespace noah_large_paintings_last_month_l49_49978

-- problem definitions
def large_painting_price : ℕ := 60
def small_painting_price : ℕ := 30
def small_paintings_sold_last_month : ℕ := 4
def sales_this_month : ℕ := 1200

-- to be proven
theorem noah_large_paintings_last_month (L : ℕ) (last_month_sales_eq : large_painting_price * L + small_painting_price * small_paintings_sold_last_month = S) 
   (this_month_sales_eq : 2 * S = sales_this_month) :
  L = 8 :=
sorry

end noah_large_paintings_last_month_l49_49978


namespace remainder_of_50_pow_2019_plus_1_mod_7_l49_49421

theorem remainder_of_50_pow_2019_plus_1_mod_7 :
  (50 ^ 2019 + 1) % 7 = 2 :=
by
  sorry

end remainder_of_50_pow_2019_plus_1_mod_7_l49_49421


namespace find_value_of_expression_l49_49259

theorem find_value_of_expression (a b c : ℝ) (h : (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0) : a + 2*b + 3*c = 18 := 
sorry

end find_value_of_expression_l49_49259


namespace sum_of_first_9_terms_l49_49672

-- Define the arithmetic sequence {a_n} and the sum S_n of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

-- Define the conditions given in the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom arith_seq : arithmetic_sequence a
axiom sum_terms : sum_of_first_n_terms a S
axiom S3 : S 3 = 30
axiom S6 : S 6 = 100

-- Goal: Prove that S 9 = 170
theorem sum_of_first_9_terms : S 9 = 170 :=
sorry -- Placeholder for the proof

end sum_of_first_9_terms_l49_49672


namespace proof_problem_l49_49128

-- Defining a right triangle ΔABC with ∠BCA=90°
structure RightTriangle :=
(a b c : ℝ)  -- sides a, b, c with c as the hypotenuse
(hypotenuse_eq : c^2 = a^2 + b^2)  -- Pythagorean relation

-- Define the circles K1 and K2 with radii r1 and r2 respectively
structure CirclesOnTriangle (Δ : RightTriangle) :=
(r1 r2 : ℝ)  -- radii of the circles K1 and K2

-- Prove the relationship r1 + r2 = a + b - c
theorem proof_problem (Δ : RightTriangle) (C : CirclesOnTriangle Δ) :
  C.r1 + C.r2 = Δ.a + Δ.b - Δ.c := by
  sorry

end proof_problem_l49_49128


namespace investment_ratio_l49_49026

theorem investment_ratio (total_investment Jim_investment : ℕ) (h₁ : total_investment = 80000) (h₂ : Jim_investment = 36000) :
  (total_investment - Jim_investment) / Nat.gcd (total_investment - Jim_investment) Jim_investment = 11 ∧ Jim_investment / Nat.gcd (total_investment - Jim_investment) Jim_investment = 9 :=
by
  sorry

end investment_ratio_l49_49026


namespace katya_solves_enough_l49_49844

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l49_49844


namespace min_cuts_for_30_sided_polygons_l49_49256

theorem min_cuts_for_30_sided_polygons (n : ℕ) (h : n = 73) : 
  ∃ k : ℕ, (∀ m : ℕ, m < k → (m + 1) ≤ 2 * m - 1972) ∧ (k = 1970) :=
sorry

end min_cuts_for_30_sided_polygons_l49_49256


namespace meeting_time_l49_49301

def time_Cassie_leaves : ℕ := 495 -- 8:15 AM in minutes past midnight
def speed_Cassie : ℕ := 12 -- mph
def break_Cassie : ℚ := 0.25 -- hours
def time_Brian_leaves : ℕ := 540 -- 9:00 AM in minutes past midnight
def speed_Brian : ℕ := 14 -- mph
def total_distance : ℕ := 74 -- miles

def time_in_minutes (h m : ℕ) : ℕ := h * 60 + m

theorem meeting_time : time_Cassie_leaves + (87 : ℚ) / 26 * 60 = time_in_minutes 11 37 := 
by sorry

end meeting_time_l49_49301


namespace foundation_cost_l49_49948

theorem foundation_cost (volume_per_house : ℝ)
    (density : ℝ)
    (cost_per_pound : ℝ)
    (num_houses : ℕ) 
    (dimension_len : ℝ)
    (dimension_wid : ℝ)
    (dimension_height : ℝ)
    : cost_per_pound = 0.02 → density = 150 → dimension_len = 100 → dimension_wid = 100 → dimension_height = 0.5 → num_houses = 3
    → volume_per_house = dimension_len * dimension_wid * dimension_height 
    → (num_houses : ℝ) * (volume_per_house * density * cost_per_pound) = 45000 := 
by 
  sorry

end foundation_cost_l49_49948


namespace find_floor_l49_49003

-- Define the total number of floors
def totalFloors : ℕ := 9

-- Define the total number of entrances
def totalEntrances : ℕ := 10

-- Each floor has the same number of apartments
-- The claim we are to prove is that for entrance 10 and apartment 333, Petya needs to go to the 3rd floor.

theorem find_floor (apartment_number : ℕ) (entrance_number : ℕ) (floor : ℕ)
  (h1 : entrance_number = 10)
  (h2 : apartment_number = 333)
  (h3 : ∀ (f : ℕ), 0 < f ∧ f ≤ totalFloors)
  (h4 : ∃ (n : ℕ), totalEntrances * totalFloors * n >= apartment_number)
  : floor = 3 :=
  sorry

end find_floor_l49_49003


namespace max_n_for_factored_polynomial_l49_49535

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l49_49535


namespace arithmetic_seq_sum_l49_49452

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l49_49452


namespace marbles_remaining_l49_49480

theorem marbles_remaining 
  (initial_remaining : ℕ := 400)
  (num_customers : ℕ := 20)
  (marbles_per_customer : ℕ := 15) :
  initial_remaining - (num_customers * marbles_per_customer) = 100 :=
by
  sorry

end marbles_remaining_l49_49480


namespace smallest_gcd_bc_l49_49912

theorem smallest_gcd_bc (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (gcd_ab : Nat.gcd a b = 168) (gcd_ac : Nat.gcd a c = 693) : Nat.gcd b c = 21 := 
sorry

end smallest_gcd_bc_l49_49912


namespace commutative_star_l49_49871

def star (a b : ℤ) : ℤ := a^2 + b^2

theorem commutative_star (a b : ℤ) : star a b = star b a :=
by sorry

end commutative_star_l49_49871


namespace tangent_line_and_area_l49_49168

noncomputable def tangent_line_equation (t : ℝ) : String := 
  "x + e^t * y - t - 1 = 0"

noncomputable def area_triangle_MON (t : ℝ) : ℝ :=
  (t + 1)^2 / (2 * Real.exp t)

theorem tangent_line_and_area (t : ℝ) (ht : t > 0) :
  tangent_line_equation t = "x + e^t * y - t - 1 = 0" ∧
  area_triangle_MON t = (t + 1)^2 / (2 * Real.exp t) := by
  sorry

end tangent_line_and_area_l49_49168


namespace shoveling_driveway_time_l49_49995

theorem shoveling_driveway_time (S : ℝ) (Wayne_rate : ℝ) (combined_rate : ℝ) :
  (S = 1 / 7) → (Wayne_rate = 6 * S) → (combined_rate = Wayne_rate + S) → (combined_rate = 1) :=
by { sorry }

end shoveling_driveway_time_l49_49995


namespace greatest_x_lcm_105_l49_49129

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l49_49129


namespace lunch_to_read_ratio_l49_49754

theorem lunch_to_read_ratio 
  (total_pages : ℕ) (pages_per_hour : ℕ) (lunch_hours : ℕ)
  (h₁ : total_pages = 4000)
  (h₂ : pages_per_hour = 250)
  (h₃ : lunch_hours = 4) :
  lunch_hours / (total_pages / pages_per_hour) = 1 / 4 := by
  sorry

end lunch_to_read_ratio_l49_49754


namespace find_number_l49_49694

theorem find_number (x : ℝ) (h : 0.05 * x = 12.75) : x = 255 :=
by
  sorry

end find_number_l49_49694


namespace pizzas_served_today_l49_49837

theorem pizzas_served_today (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (h1 : lunch_pizzas = 9) (h2 : dinner_pizzas = 6) : lunch_pizzas + dinner_pizzas = 15 :=
by sorry

end pizzas_served_today_l49_49837


namespace radius_of_inscribed_circle_in_quarter_circle_l49_49472

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ :=
  R * (Real.sqrt 2 - 1)

theorem radius_of_inscribed_circle_in_quarter_circle 
  (R : ℝ) (hR : R = 6) : inscribed_circle_radius R = 6 * Real.sqrt 2 - 6 :=
by
  rw [inscribed_circle_radius, hR]
  -- Apply the necessary simplifications and manipulations from the given solution steps here
  sorry

end radius_of_inscribed_circle_in_quarter_circle_l49_49472


namespace second_smallest_integer_l49_49410

theorem second_smallest_integer (x y z w v : ℤ) (h_avg : (x + y + z + w + v) / 5 = 69)
  (h_median : z = 83) (h_mode : w = 85 ∧ v = 85) (h_range : 85 - x = 70) :
  y = 77 :=
by
  sorry

end second_smallest_integer_l49_49410


namespace total_cost_to_replace_floor_l49_49151

def removal_cost : ℝ := 50
def cost_per_sqft : ℝ := 1.25
def room_dimensions : (ℝ × ℝ) := (8, 7)

theorem total_cost_to_replace_floor :
  removal_cost + (cost_per_sqft * (room_dimensions.1 * room_dimensions.2)) = 120 := by
  sorry

end total_cost_to_replace_floor_l49_49151


namespace unique_prime_solution_l49_49875

-- Define the problem in terms of prime numbers and checking the conditions
open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_solution (p : ℕ) (hp : is_prime p) (h1 : is_prime (p^2 - 6)) (h2 : is_prime (p^2 + 6)) : p = 5 := 
sorry

end unique_prime_solution_l49_49875


namespace inequality_proof_l49_49407

theorem inequality_proof (x y : ℝ) :
  abs ((x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2))) ≤ 1 / 2 := 
sorry

end inequality_proof_l49_49407


namespace percentage_proof_l49_49610

theorem percentage_proof (n : ℝ) (h : 0.3 * 0.4 * n = 24) : 0.4 * 0.3 * n = 24 :=
sorry

end percentage_proof_l49_49610


namespace neg_p_l49_49496

-- Define the initial proposition p
def p : Prop := ∀ (m : ℝ), m ≥ 0 → 4^m ≥ 4 * m

-- State the theorem to prove the negation of p
theorem neg_p : ¬p ↔ ∃ (m_0 : ℝ), m_0 ≥ 0 ∧ 4^m_0 < 4 * m_0 :=
by
  sorry

end neg_p_l49_49496


namespace player_A_elimination_after_third_round_at_least_one_player_passes_all_l49_49287

-- Define probabilities for Player A's success in each round
def P_A1 : ℚ := 4 / 5
def P_A2 : ℚ := 3 / 4
def P_A3 : ℚ := 2 / 3

-- Define probabilities for Player B's success in each round
def P_B1 : ℚ := 2 / 3
def P_B2 : ℚ := 2 / 3
def P_B3 : ℚ := 1 / 2

-- Define theorems
theorem player_A_elimination_after_third_round :
  P_A1 * P_A2 * (1 - P_A3) = 1 / 5 := by
  sorry

theorem at_least_one_player_passes_all :
  1 - ((1 - (P_A1 * P_A2 * P_A3)) * (1 - (P_B1 * P_B2 * P_B3))) = 8 / 15 := by
  sorry


end player_A_elimination_after_third_round_at_least_one_player_passes_all_l49_49287


namespace sin_of_cos_of_angle_l49_49115

-- We need to assume that A is an angle of a triangle, hence A is in the range (0, π).
theorem sin_of_cos_of_angle (A : ℝ) (hA : 0 < A ∧ A < π) (h_cos : Real.cos A = -3/5) : Real.sin A = 4/5 := by
  sorry

end sin_of_cos_of_angle_l49_49115


namespace jack_keeps_10800_pounds_l49_49706

def number_of_months_in_a_quarter := 12 / 4
def monthly_hunting_trips := 6
def total_hunting_trips := monthly_hunting_trips * number_of_months_in_a_quarter
def deers_per_trip := 2
def total_deers := total_hunting_trips * deers_per_trip
def weight_per_deer := 600
def total_weight := total_deers * weight_per_deer
def kept_weight_fraction := 1 / 2
def kept_weight := total_weight * kept_weight_fraction

theorem jack_keeps_10800_pounds :
  kept_weight = 10800 :=
by
  -- This is a stub for the automated proof
  sorry

end jack_keeps_10800_pounds_l49_49706


namespace quoted_value_stock_l49_49265

-- Define the conditions
def face_value : ℕ := 100
def dividend_percentage : ℝ := 0.14
def yield_percentage : ℝ := 0.1

-- Define the computed dividend per share
def dividend_per_share : ℝ := dividend_percentage * face_value

-- State the theorem to prove the quoted value
theorem quoted_value_stock : (dividend_per_share / yield_percentage) * 100 = 140 :=
by
  sorry  -- Placeholder for the proof

end quoted_value_stock_l49_49265


namespace sum_of_intercepts_l49_49260

theorem sum_of_intercepts (x y : ℝ) (h : x / 3 - y / 4 = 1) : (x / 3 = 1 ∧ y / (-4) = 1) → 3 + (-4) = -1 :=
by
  sorry

end sum_of_intercepts_l49_49260


namespace one_girl_made_a_mistake_l49_49074

variables (c_M c_K c_L c_O : ℤ)

theorem one_girl_made_a_mistake (h₁ : c_M + c_K = c_L + c_O + 12) (h₂ : c_K + c_L = c_M + c_O - 7) :
  false := by
  -- Proof intentionally missing
  sorry

end one_girl_made_a_mistake_l49_49074


namespace total_area_of_sheet_l49_49293

theorem total_area_of_sheet (A B : ℝ) (h1 : A = 4 * B) (h2 : A = B + 2208) : A + B = 3680 :=
by
  sorry

end total_area_of_sheet_l49_49293


namespace platform_length_259_9584_l49_49497

noncomputable def length_of_platform (speed_kmph time_sec train_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600  -- conversion from kmph to m/s
  let distance_covered := speed_mps * time_sec
  distance_covered - train_length_m

theorem platform_length_259_9584 :
  length_of_platform 72 26 260.0416 = 259.9584 :=
by sorry

end platform_length_259_9584_l49_49497


namespace greatest_possible_median_l49_49601

theorem greatest_possible_median : 
  ∀ (k m r s t : ℕ),
    k < m → m < r → r < s → s < t →
    (k + m + r + s + t = 90) →
    (t = 40) →
    (r = 23) :=
by
  intros k m r s t h1 h2 h3 h4 h_sum h_t
  sorry

end greatest_possible_median_l49_49601


namespace composite_sum_l49_49822

open Nat

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) : ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + c = x * y :=
by
  sorry

end composite_sum_l49_49822


namespace find_y_l49_49947

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℝ) / y = 86.12) : y = 75 :=
sorry

end find_y_l49_49947


namespace jade_more_transactions_l49_49367

theorem jade_more_transactions 
    (mabel_transactions : ℕ) 
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + (mabel_transactions / 10))
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = 82) :
    jade_transactions - cal_transactions = 16 :=
sorry

end jade_more_transactions_l49_49367


namespace tangent_line_equation_l49_49999

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + a * x^2 + (a - 3) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_equation (a : ℝ) (h : ∀ x : ℝ, f a (-x) = f a x) :
    9 * (2 : ℝ) - f a 2 - 16 = 0 :=
by
  sorry

end tangent_line_equation_l49_49999


namespace simplify_parentheses_l49_49607

theorem simplify_parentheses (a b c x y : ℝ) : (3 * a - (2 * a - c) = 3 * a - 2 * a + c) := 
by 
  sorry

end simplify_parentheses_l49_49607


namespace common_point_eq_l49_49723

theorem common_point_eq (a b c d : ℝ) (h₀ : a ≠ b) 
  (h₁ : ∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) : 
  d = c :=
by
  sorry

end common_point_eq_l49_49723


namespace solve_cubic_eq_l49_49858

theorem solve_cubic_eq (x : ℝ) : (8 - x)^3 = x^3 → x = 8 :=
by
  sorry

end solve_cubic_eq_l49_49858


namespace ones_digit_of_sum_of_powers_l49_49269

theorem ones_digit_of_sum_of_powers :
  (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 :=
by
  sorry

end ones_digit_of_sum_of_powers_l49_49269


namespace volume_removed_percentage_l49_49876

noncomputable def original_volume : ℕ := 20 * 15 * 10

noncomputable def cube_volume : ℕ := 4 * 4 * 4

noncomputable def total_volume_removed : ℕ := 8 * cube_volume

noncomputable def percentage_volume_removed : ℝ :=
  (total_volume_removed : ℝ) / (original_volume : ℝ) * 100

theorem volume_removed_percentage :
  percentage_volume_removed = 512 / 30 := sorry

end volume_removed_percentage_l49_49876


namespace single_point_graph_value_of_d_l49_49818

theorem single_point_graph_value_of_d (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + d = 0 → x = -2 ∧ y = 3) ↔ d = 21 := 
by 
  sorry

end single_point_graph_value_of_d_l49_49818


namespace sum_of_coordinates_point_D_l49_49851

theorem sum_of_coordinates_point_D 
(M : ℝ × ℝ) (C D : ℝ × ℝ) 
(hM : M = (3, 5)) 
(hC : C = (1, 10)) 
(hmid : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
: D.1 + D.2 = 5 :=
sorry

end sum_of_coordinates_point_D_l49_49851


namespace sum_of_all_possible_values_of_abs_b_l49_49614

theorem sum_of_all_possible_values_of_abs_b {a b : ℝ}
  {r s : ℝ} (hr : r^3 + a * r + b = 0) (hs : s^3 + a * s + b = 0)
  (hr4 : (r + 4)^3 + a * (r + 4) + b + 240 = 0) (hs3 : (s - 3)^3 + a * (s - 3) + b + 240 = 0) :
  |b| = 20 ∨ |b| = 42 →
  20 + 42 = 62 :=
by
  sorry

end sum_of_all_possible_values_of_abs_b_l49_49614


namespace secret_reaches_2186_students_on_seventh_day_l49_49970

/-- 
Alice tells a secret to three friends on Sunday. The next day, each of those friends tells the secret to three new friends.
Each time a person hears the secret, they tell three other new friends the following day.
On what day will 2186 students know the secret?
-/
theorem secret_reaches_2186_students_on_seventh_day :
  ∃ (n : ℕ), 1 + 3 * ((3^n - 1)/2) = 2186 ∧ n = 7 :=
by
  sorry

end secret_reaches_2186_students_on_seventh_day_l49_49970


namespace find_certain_number_multiplied_by_24_l49_49334

-- Define the conditions
theorem find_certain_number_multiplied_by_24 :
  (∃ x : ℤ, 37 - x = 24) →
  ∀ x : ℤ, (37 - x = 24) → (x * 24 = 312) :=
by
  intros h x hx
  -- Here we will have the proof using the assumption and the theorem.
  sorry

end find_certain_number_multiplied_by_24_l49_49334


namespace cost_per_text_message_for_first_plan_l49_49479

theorem cost_per_text_message_for_first_plan (x : ℝ) : 
  (9 + 60 * x = 60 * 0.40) → (x = 0.25) :=
by
  intro h
  sorry

end cost_per_text_message_for_first_plan_l49_49479


namespace adults_eat_one_third_l49_49796

theorem adults_eat_one_third (n c k : ℕ) (hn : n = 120) (hc : c = 4) (hk : k = 20) :
  ((n - c * k) / n : ℚ) = 1 / 3 :=
by
  sorry

end adults_eat_one_third_l49_49796


namespace marble_count_l49_49471

theorem marble_count (r g b : ℝ) (h1 : g + b = 9) (h2 : r + b = 7) (h3 : r + g = 5) :
  r + g + b = 10.5 :=
by sorry

end marble_count_l49_49471


namespace perpendicular_condition_l49_49685

theorem perpendicular_condition (a : ℝ) :
  (a = 1) ↔ (∀ x : ℝ, (a*x + 1 - ((a - 2)*x - 1)) * ((a * x + 1 - (a * x + 1))) = 0) :=
by
  sorry

end perpendicular_condition_l49_49685


namespace license_plate_difference_l49_49468

theorem license_plate_difference :
  (26^4 * 10^3 - 26^5 * 10^2 = -731161600) :=
sorry

end license_plate_difference_l49_49468


namespace monic_polynomial_roots_l49_49973

theorem monic_polynomial_roots (r1 r2 r3 : ℝ) (h : ∀ x : ℝ, x^3 - 4*x^2 + 5 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x : ℝ, x^3 - 12*x^2 + 135 = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
by
  sorry

end monic_polynomial_roots_l49_49973


namespace classroom_lamps_total_ways_l49_49738

theorem classroom_lamps_total_ways (n : ℕ) (h : n = 4) : (2^n - 1) = 15 :=
by
  sorry

end classroom_lamps_total_ways_l49_49738


namespace part1_a_eq_zero_part2_range_of_a_l49_49729

noncomputable def f (x : ℝ) := abs (x + 1)
noncomputable def g (x : ℝ) (a : ℝ) := 2 * abs x + a

theorem part1_a_eq_zero :
  ∀ x, 0 < x + 1 → 0 < 2 * abs x → a = 0 →
  f x ≥ g x a ↔ (-1 / 3 : ℝ) ≤ x ∧ x ≤ 1 :=
sorry

theorem part2_range_of_a :
  ∃ x, f x ≥ g x a ↔ a ≤ 1 :=
sorry

end part1_a_eq_zero_part2_range_of_a_l49_49729


namespace double_seven_eighth_l49_49730

theorem double_seven_eighth (n : ℕ) (h : n = 48) : 2 * (7 / 8 * n) = 84 := by
  sorry

end double_seven_eighth_l49_49730


namespace solve_for_x_l49_49911

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
    5 * y ^ 2 + 2 * y + 3 = 3 * (9 * x ^ 2 + y + 1) ↔ x = 0 ∨ x = 1 / 6 := 
by
  sorry

end solve_for_x_l49_49911


namespace prod2025_min_sum_l49_49715

theorem prod2025_min_sum : ∃ (a b : ℕ), a * b = 2025 ∧ a > 0 ∧ b > 0 ∧ (∀ (x y : ℕ), x * y = 2025 → x > 0 → y > 0 → x + y ≥ a + b) ∧ a + b = 90 :=
sorry

end prod2025_min_sum_l49_49715


namespace problem_statement_l49_49972

def U : Set ℤ := {x | True}
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}
def complement (B : Set ℤ) : Set ℤ := {x | x ∉ B}

theorem problem_statement : (A ∩ (complement B)) = {1, 3, 9} :=
by {
  sorry
}

end problem_statement_l49_49972


namespace nonnegative_for_interval_l49_49704

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 * (x - 2)^2) / ((1 - x) * (1 + x + x^2))

theorem nonnegative_for_interval (x : ℝ) : (f x >= 0) ↔ (0 <= x) :=
by
  sorry

end nonnegative_for_interval_l49_49704


namespace dave_coins_l49_49867

theorem dave_coins :
  ∃ n : ℕ, n ≡ 2 [MOD 7] ∧ n ≡ 3 [MOD 5] ∧ n ≡ 1 [MOD 3] ∧ n = 58 :=
sorry

end dave_coins_l49_49867


namespace oranges_left_l49_49489

-- Main theorem statement: number of oranges left after specified increases and losses
theorem oranges_left (Mary Jason Tom Sarah : ℕ)
  (hMary : Mary = 122)
  (hJason : Jason = 105)
  (hTom : Tom = 85)
  (hSarah : Sarah = 134) 
  (round : ℝ → ℕ) 
  : round (round ( (Mary : ℝ) * 1.1) 
         + round ((Jason : ℝ) * 1.1) 
         + round ((Tom : ℝ) * 1.1) 
         + round ((Sarah : ℝ) * 1.1) 
         - round (0.15 * (round ((Mary : ℝ) * 1.1) 
                         + round ((Jason : ℝ) * 1.1)
                         + round ((Tom : ℝ) * 1.1) 
                         + round ((Sarah : ℝ) * 1.1)) )) = 417  := 
sorry

end oranges_left_l49_49489


namespace regular_train_passes_by_in_4_seconds_l49_49987

theorem regular_train_passes_by_in_4_seconds
    (l_high_speed : ℕ)
    (l_regular : ℕ)
    (t_observed : ℕ)
    (v_relative : ℕ)
    (h_length_high_speed : l_high_speed = 80)
    (h_length_regular : l_regular = 100)
    (h_time_observed : t_observed = 5)
    (h_velocity : v_relative = l_regular / t_observed) :
    v_relative * 4 = l_high_speed :=
by
  sorry

end regular_train_passes_by_in_4_seconds_l49_49987


namespace ratio_is_five_thirds_l49_49547

noncomputable def ratio_of_numbers (a b : ℝ) : Prop :=
  (a + b = 4 * (a - b)) → (a = 2 * b) → (a / b = 5 / 3)

theorem ratio_is_five_thirds {a b : ℝ} (h1 : a + b = 4 * (a - b)) (h2 : a = 2 * b) :
  a / b = 5 / 3 :=
  sorry

end ratio_is_five_thirds_l49_49547


namespace sum_even_and_odd_numbers_up_to_50_l49_49200

def sum_even_numbers (n : ℕ) : ℕ :=
  (2 + 50) * n / 2

def sum_odd_numbers (n : ℕ) : ℕ :=
  (1 + 49) * n / 2

theorem sum_even_and_odd_numbers_up_to_50 : 
  sum_even_numbers 25 + sum_odd_numbers 25 = 1275 :=
by
  sorry

end sum_even_and_odd_numbers_up_to_50_l49_49200


namespace trapezoid_perimeter_l49_49330

theorem trapezoid_perimeter (AB CD BC DA : ℝ) (BCD_angle : ℝ)
  (h1 : AB = 60) (h2 : CD = 40) (h3 : BC = DA) (h4 : BCD_angle = 120) :
  AB + BC + CD + DA = 220 := 
sorry

end trapezoid_perimeter_l49_49330


namespace gcd_two_5_digit_integers_l49_49974

theorem gcd_two_5_digit_integers (a b : ℕ) 
  (h1 : 10^4 ≤ a ∧ a < 10^5)
  (h2 : 10^4 ≤ b ∧ b < 10^5)
  (h3 : 10^8 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^9) :
  Nat.gcd a b < 10^2 :=
by
  sorry  -- Skip the proof

end gcd_two_5_digit_integers_l49_49974


namespace difference_of_decimal_and_fraction_l49_49208

theorem difference_of_decimal_and_fraction :
  0.127 - (1 / 8) = 0.002 := 
by
  sorry

end difference_of_decimal_and_fraction_l49_49208


namespace vertical_angles_equal_l49_49937

-- Given: Definition for pairs of adjacent angles summing up to 180 degrees
def adjacent_add_to_straight_angle (α β : ℝ) : Prop := 
  α + β = 180

-- Given: Two intersecting lines forming angles
variables (α β γ δ : ℝ)

-- Given: Relationship of adjacent angles being supplementary
axiom adj1 : adjacent_add_to_straight_angle α β
axiom adj2 : adjacent_add_to_straight_angle β γ
axiom adj3 : adjacent_add_to_straight_angle γ δ
axiom adj4 : adjacent_add_to_straight_angle δ α

-- Question: Prove that vertical angles are equal
theorem vertical_angles_equal : α = γ :=
by sorry

end vertical_angles_equal_l49_49937


namespace annual_growth_rate_l49_49817

-- definitions based on the conditions in the problem
def FirstYear : ℝ := 400
def ThirdYear : ℝ := 625
def n : ℕ := 2

-- the main statement to prove the corresponding equation
theorem annual_growth_rate (x : ℝ) : 400 * (1 + x)^2 = 625 :=
sorry

end annual_growth_rate_l49_49817


namespace smallest_m_l49_49343

-- Defining the remainder function
def r (m n : ℕ) : ℕ := m % n

-- Main theorem stating the problem needed to be proved
theorem smallest_m (m : ℕ) (h : m > 0) 
  (H : (r m 1 + r m 2 + r m 3 + r m 4 + r m 5 + r m 6 + r m 7 + r m 8 + r m 9 + r m 10) = 4) : 
  m = 120 :=
sorry

end smallest_m_l49_49343


namespace sum_of_smallest_and_largest_eq_2y_l49_49985

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l49_49985


namespace percentage_loss_15_l49_49160

theorem percentage_loss_15
  (sold_at_loss : ℝ)
  (sold_at_profit : ℝ)
  (percentage_profit : ℝ)
  (cost_price : ℝ)
  (percentage_loss : ℝ)
  (H1 : sold_at_loss = 12)
  (H2 : sold_at_profit = 14.823529411764707)
  (H3 : percentage_profit = 5)
  (H4 : cost_price = sold_at_profit / (1 + percentage_profit / 100))
  (H5 : percentage_loss = (cost_price - sold_at_loss) / cost_price * 100) :
  percentage_loss = 15 :=
by
  sorry

end percentage_loss_15_l49_49160


namespace total_results_count_l49_49209

theorem total_results_count (N : ℕ) (S : ℕ) 
  (h1 : S = 50 * N) 
  (h2 : (12 * 14) + (12 * 17) = 372)
  (h3 : S = 372 + 878) : N = 25 := 
by 
  sorry

end total_results_count_l49_49209


namespace cube_volume_l49_49516

theorem cube_volume (A : ℝ) (hA : A = 294) : ∃ (V : ℝ), V = 343 :=
by
  sorry

end cube_volume_l49_49516


namespace factorize_ab_factorize_x_l49_49096

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end factorize_ab_factorize_x_l49_49096


namespace ninth_term_arithmetic_sequence_l49_49246

-- Definitions based on conditions:
def first_term : ℚ := 5 / 6
def seventeenth_term : ℚ := 5 / 8

-- Here is the main statement we need to prove:
theorem ninth_term_arithmetic_sequence : (first_term + 8 * ((seventeenth_term - first_term) / 16) = 15 / 16) :=
by
  sorry

end ninth_term_arithmetic_sequence_l49_49246


namespace option_c_same_function_l49_49346

-- Definitions based on conditions
def f_c (x : ℝ) : ℝ := x^2
def g_c (x : ℝ) : ℝ := 3 * x^6

-- Theorem statement that Option C f(x) and g(x) represent the same function
theorem option_c_same_function : ∀ x : ℝ, f_c x = g_c x := by
  sorry

end option_c_same_function_l49_49346


namespace sin_order_l49_49093

theorem sin_order :
  ∀ (sin₁ sin₂ sin₃ sin₄ sin₆ : ℝ),
  sin₁ = Real.sin 1 ∧ 
  sin₂ = Real.sin 2 ∧ 
  sin₃ = Real.sin 3 ∧ 
  sin₄ = Real.sin 4 ∧ 
  sin₆ = Real.sin 6 →
  sin₂ > sin₁ ∧ sin₁ > sin₃ ∧ sin₃ > sin₆ ∧ sin₆ > sin₄ :=
by
  sorry

end sin_order_l49_49093


namespace sum_of_eggs_is_3712_l49_49540

-- Definitions based on the conditions
def eggs_yesterday : ℕ := 1925
def eggs_fewer_today : ℕ := 138
def eggs_today : ℕ := eggs_yesterday - eggs_fewer_today

-- Theorem stating the equivalence of the sum of eggs
theorem sum_of_eggs_is_3712 : eggs_yesterday + eggs_today = 3712 :=
by
  sorry

end sum_of_eggs_is_3712_l49_49540


namespace total_percent_sample_candy_l49_49574

theorem total_percent_sample_candy (total_customers : ℕ) (percent_caught : ℝ) (percent_not_caught : ℝ)
  (h1 : percent_caught = 0.22)
  (h2 : percent_not_caught = 0.20)
  (h3 : total_customers = 100) :
  percent_caught + percent_not_caught = 0.28 :=
by
  sorry

end total_percent_sample_candy_l49_49574


namespace bear_weight_gain_l49_49616

theorem bear_weight_gain :
  let total_weight := 1000
  let weight_from_berries := total_weight / 5
  let weight_from_acorns := 2 * weight_from_berries
  let weight_from_salmon := (total_weight - weight_from_berries - weight_from_acorns) / 2
  let weight_from_small_animals := total_weight - (weight_from_berries + weight_from_acorns + weight_from_salmon)
  weight_from_small_animals = 200 :=
by sorry

end bear_weight_gain_l49_49616


namespace participating_girls_l49_49107

theorem participating_girls (total_students boys_participation girls_participation participating_students : ℕ)
  (h1 : total_students = 800)
  (h2 : boys_participation = 2)
  (h3 : girls_participation = 3)
  (h4 : participating_students = 550) :
  (4 / total_students) * (boys_participation / 3) * total_students + (4 * girls_participation / 4) * total_students = 4 * 150 :=
by
  sorry

end participating_girls_l49_49107


namespace points_in_quadrants_l49_49418

theorem points_in_quadrants (x y : ℝ) (h₁ : y > 3 * x) (h₂ : y > 6 - x) : 
  (0 <= x ∧ 0 <= y) ∨ (x <= 0 ∧ 0 <= y) :=
by
  sorry

end points_in_quadrants_l49_49418


namespace inradius_of_triangle_l49_49313

theorem inradius_of_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 16) (h3 : c = 17) : 
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    let r := area / s
    r = Real.sqrt 21 := by
  sorry

end inradius_of_triangle_l49_49313


namespace symmetric_graph_l49_49341

variable (f : ℝ → ℝ)
variable (c : ℝ)
variable (h_nonzero : c ≠ 0)
variable (h_fx_plus_y : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * f x * f y)
variable (h_f_half_c : f (c / 2) = 0)
variable (h_f_zero : f 0 ≠ 0)

theorem symmetric_graph (k : ℤ) : 
  ∀ (x : ℝ), f (x) = f (2*k*c - x) :=
sorry

end symmetric_graph_l49_49341


namespace minimum_value_of_f_l49_49709

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.cos x)^2) + (1 / (Real.sin x)^2)

theorem minimum_value_of_f : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ y = 4 :=
by
  sorry

end minimum_value_of_f_l49_49709


namespace carpet_shaded_area_l49_49733

theorem carpet_shaded_area
  (side_length_carpet : ℝ)
  (S : ℝ)
  (T : ℝ)
  (h1 : side_length_carpet = 12)
  (h2 : 12 / S = 4)
  (h3 : S / T = 2) :
  let area_big_square := S^2
  let area_small_squares := 4 * T^2
  area_big_square + area_small_squares = 18 := by
  sorry

end carpet_shaded_area_l49_49733


namespace min_value_expr_l49_49969

theorem min_value_expr : ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by
  sorry

end min_value_expr_l49_49969


namespace area_ratio_of_region_A_and_C_l49_49419

theorem area_ratio_of_region_A_and_C
  (pA : ℕ) (pC : ℕ) 
  (hA : pA = 16)
  (hC : pC = 24) :
  let sA := pA / 4
  let sC := pC / 6
  let areaA := sA * sA
  let areaC := (3 * Real.sqrt 3 / 2) * sC * sC
  (areaA / areaC) = (2 * Real.sqrt 3 / 9) :=
by
  sorry

end area_ratio_of_region_A_and_C_l49_49419


namespace eval_expression_l49_49634

theorem eval_expression : 1999^2 - 1998 * 2002 = -3991 := 
by
  sorry

end eval_expression_l49_49634


namespace equation_of_line_BC_l49_49882

/-
Given:
1. Point A(3, -1)
2. The line containing the median from A to side BC: 6x + 10y - 59 = 0
3. The line containing the angle bisector of ∠B: x - 4y + 10 = 0

Prove:
The equation of the line containing side BC is 2x + 9y - 65 = 0.
-/

noncomputable def point_A : (ℝ × ℝ) := (3, -1)

noncomputable def median_line (x y : ℝ) : Prop := 6 * x + 10 * y - 59 = 0

noncomputable def angle_bisector_line_B (x y : ℝ) : Prop := x - 4 * y + 10 = 0

theorem equation_of_line_BC :
  ∃ (x y : ℝ), 2 * x + 9 * y - 65 = 0 :=
sorry

end equation_of_line_BC_l49_49882


namespace yogurt_banana_slices_l49_49891

/--
Given:
1. Each banana yields 10 slices.
2. Vivian needs to make 5 yogurts.
3. She needs to buy 4 bananas.

Prove:
The number of banana slices needed for each yogurt is 8.
-/
theorem yogurt_banana_slices 
    (slices_per_banana : ℕ)
    (bananas_bought : ℕ)
    (yogurts_needed : ℕ)
    (h1 : slices_per_banana = 10)
    (h2 : yogurts_needed = 5)
    (h3 : bananas_bought = 4) : 
    (bananas_bought * slices_per_banana) / yogurts_needed = 8 :=
by
  sorry

end yogurt_banana_slices_l49_49891


namespace minimum_value_proof_l49_49055

noncomputable def minimum_value : ℝ :=
  3 + 2 * Real.sqrt 2

theorem minimum_value_proof (a b : ℝ) (h_line_eq : ∀ x y : ℝ, a * x + b * y = 1)
  (h_ab_pos : a * b > 0)
  (h_center_bisect : ∃ x y : ℝ, (x - 1)^2 + (y - 2)^2 <= x^2 + y^2) :
  (1 / a + 1 / b) ≥ minimum_value :=
by
  -- Sorry placeholder for the proof
  sorry

end minimum_value_proof_l49_49055


namespace modular_inverse_of_17_mod_800_l49_49493

    theorem modular_inverse_of_17_mod_800 :
      ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ (17 * x) % 800 = 1 :=
    by
      use 47
      sorry
    
end modular_inverse_of_17_mod_800_l49_49493


namespace xiaoming_correct_answers_l49_49980

theorem xiaoming_correct_answers (x : ℕ) (h1 : x ≤ 10) (h2 : 5 * x - (10 - x) > 30) : x ≥ 7 := 
by
  sorry

end xiaoming_correct_answers_l49_49980


namespace initial_sentences_today_l49_49490

-- Definitions of the given conditions
def typing_rate : ℕ := 6
def initial_typing_time : ℕ := 20
def additional_typing_time : ℕ := 15
def erased_sentences : ℕ := 40
def post_meeting_typing_time : ℕ := 18
def total_sentences_end_of_day : ℕ := 536

def sentences_typed_before_break := initial_typing_time * typing_rate
def sentences_typed_after_break := additional_typing_time * typing_rate
def sentences_typed_post_meeting := post_meeting_typing_time * typing_rate
def sentences_today := sentences_typed_before_break + sentences_typed_after_break - erased_sentences + sentences_typed_post_meeting

theorem initial_sentences_today : total_sentences_end_of_day - sentences_today = 258 := by
  -- proof here
  sorry

end initial_sentences_today_l49_49490


namespace cost_of_first_variety_l49_49943

theorem cost_of_first_variety (x : ℝ) (cost2 : ℝ) (cost_mix : ℝ) (ratio : ℝ) :
    cost2 = 8.75 →
    cost_mix = 7.50 →
    ratio = 0.625 →
    (x - cost_mix) / (cost2 - cost_mix) = ratio →
    x = 8.28125 := 
by
  intros h1 h2 h3 h4
  sorry

end cost_of_first_variety_l49_49943


namespace not_prime_a_l49_49670

theorem not_prime_a 
  (a b : ℕ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : ∃ k : ℤ, (5 * a^4 + a^2) = k * (b^4 + 3 * b^2 + 4))
  : ¬ Nat.Prime a := 
sorry

end not_prime_a_l49_49670


namespace fraction_is_one_fourth_l49_49603

-- Defining the numbers
def num1 : ℕ := 16
def num2 : ℕ := 8

-- Conditions
def difference_correct : Prop := num1 - num2 = 8
def sum_of_numbers : ℕ := num1 + num2
def fraction_of_sum (f : ℚ) : Prop := f * sum_of_numbers = 6

-- Theorem stating the fraction
theorem fraction_is_one_fourth (f : ℚ) (h1 : difference_correct) (h2 : fraction_of_sum f) : f = 1 / 4 :=
by {
  -- This will use the conditions and show that f = 1/4
  sorry
}

end fraction_is_one_fourth_l49_49603


namespace intersection_point_finv_l49_49218

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b

theorem intersection_point_finv (a b : ℤ) : 
  (∀ x : ℝ, f (f x b) b = x) → 
  (∀ y : ℝ, f (f y b) b = y) → 
  (f (-4) b = a) → 
  (f a b = -4) → 
  a = -4 := 
by
  intros
  sorry

end intersection_point_finv_l49_49218


namespace total_pennies_after_addition_l49_49594

def initial_pennies_per_compartment : ℕ := 10
def compartments : ℕ := 20
def added_pennies_per_compartment : ℕ := 15

theorem total_pennies_after_addition :
  (initial_pennies_per_compartment + added_pennies_per_compartment) * compartments = 500 :=
by 
  sorry

end total_pennies_after_addition_l49_49594


namespace sum_of_solutions_l49_49232

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l49_49232


namespace min_units_l49_49925

theorem min_units (x : ℕ) (h1 : 5500 * 60 + 5000 * (x - 60) > 550000) : x ≥ 105 := 
by {
  sorry
}

end min_units_l49_49925


namespace halfway_fraction_l49_49388

theorem halfway_fraction : 
  ∃ (x : ℚ), x = 1/2 * ((2/3) + (4/5)) ∧ x = 11/15 :=
by
  sorry

end halfway_fraction_l49_49388


namespace find_m_l49_49869

-- Define the conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 4 = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- Statement of the problem
theorem find_m (m : ℝ) (e : ℝ) (h1 : eccentricity e) (h2 : ∀ x y : ℝ, ellipse_eq x y m) :
  m = 3 ∨ m = 5 :=
sorry

end find_m_l49_49869


namespace common_root_rational_l49_49323

variable (a b c d e f g : ℚ) -- coefficient variables

def poly1 (x : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 18

def poly2 (x : ℚ) : ℚ := 18 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_rational (k : ℚ) (h1 : poly1 a b c k = 0) (h2 : poly2 d e f g k = 0) 
  (hn : k < 0) (hi : ∀ (m n : ℤ), k ≠ m / n) : k = -1/3 := sorry

end common_root_rational_l49_49323


namespace required_speed_remaining_l49_49226

theorem required_speed_remaining (total_distance : ℕ) (total_time : ℕ) (initial_speed : ℕ) (initial_time : ℕ) 
  (h1 : total_distance = 24) (h2 : total_time = 8) (h3 : initial_speed = 4) (h4 : initial_time = 4) :
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

end required_speed_remaining_l49_49226


namespace number_b_is_three_times_number_a_l49_49023

theorem number_b_is_three_times_number_a (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 :=
by
  -- This is where the proof would go
  sorry

end number_b_is_three_times_number_a_l49_49023


namespace costPrice_of_bat_is_152_l49_49680

noncomputable def costPriceOfBatForA (priceC : ℝ) (profitA : ℝ) (profitB : ℝ) : ℝ :=
  priceC / (1 + profitB) / (1 + profitA)

theorem costPrice_of_bat_is_152 :
  costPriceOfBatForA 228 0.20 0.25 = 152 :=
by
  -- Placeholder for the proof
  sorry

end costPrice_of_bat_is_152_l49_49680


namespace sum_geometric_series_l49_49751

-- Given the conditions
def q : ℕ := 2
def a3 : ℕ := 16
def n : ℕ := 2017
def a1 : ℕ := 4

-- Define the sum of the first n terms of a geometric series
noncomputable def geometricSeriesSum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

-- State the problem
theorem sum_geometric_series :
  geometricSeriesSum a1 q n = 2^2019 - 4 :=
sorry

end sum_geometric_series_l49_49751


namespace original_price_of_book_l49_49442

theorem original_price_of_book (final_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) 
  (h1 : final_price = 360) (h2 : increase_percentage = 0.20) 
  (h3 : final_price = (1 + increase_percentage) * original_price) : original_price = 300 := 
by
  sorry

end original_price_of_book_l49_49442


namespace Haleigh_can_make_3_candles_l49_49688

variable (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ)

def wax_leftover (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ) : ℝ := 
  n20 * w20 + n5 * w5 + n1 * w1 

theorem Haleigh_can_make_3_candles :
  ∀ (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ), 
  n20 = 5 →
  w20 = 2 →
  n5 = 5 →
  w5 = 0.5 →
  n1 = 25 →
  w1 = 0.1 →
  oz10 = 10 →
  (wax_leftover n20 n5 n1 w20 w5 w1 oz10) / 5 = 3 := 
by
  intros n20 n5 n1 w20 w5 w1 oz10 hn20 hw20 hn5 hw5 hn1 hw1 hoz10
  rw [hn20, hw20, hn5, hw5, hn1, hw1, hoz10]
  sorry

end Haleigh_can_make_3_candles_l49_49688


namespace maisie_flyers_count_l49_49430

theorem maisie_flyers_count (M : ℕ) (h1 : 71 = 2 * M + 5) : M = 33 :=
by
  sorry

end maisie_flyers_count_l49_49430


namespace tropical_island_parrots_l49_49431

theorem tropical_island_parrots :
  let total_parrots := 150
  let red_fraction := 4 / 5
  let yellow_fraction := 1 - red_fraction
  let yellow_parrots := yellow_fraction * total_parrots
  yellow_parrots = 30 := sorry

end tropical_island_parrots_l49_49431


namespace blue_notebook_cost_l49_49744

theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (cost_per_red : ℕ)
  (green_notebooks : ℕ)
  (cost_per_green : ℕ)
  (blue_notebooks : ℕ)
  (total_cost_blue : ℕ)
  (cost_per_blue : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : cost_per_red = 4)
  (h5 : green_notebooks = 2)
  (h6 : cost_per_green = 2)
  (h7 : total_cost_blue = total_spent - (red_notebooks * cost_per_red + green_notebooks * cost_per_green))
  (h8 : blue_notebooks = total_notebooks - (red_notebooks + green_notebooks))
  (h9 : cost_per_blue = total_cost_blue / blue_notebooks)
  : cost_per_blue = 3 :=
sorry

end blue_notebook_cost_l49_49744


namespace simplify_expression_l49_49993

theorem simplify_expression (x y : ℝ) : 
  (5 * x ^ 2 - 3 * x + 2) * (107 - 107) + (7 * y ^ 2 + 4 * y - 1) * (93 - 93) = 0 := 
by 
  sorry

end simplify_expression_l49_49993


namespace average_last_4_matches_l49_49511

theorem average_last_4_matches 
  (avg_10 : ℝ) (avg_6 : ℝ) (result : ℝ)
  (h1 : avg_10 = 38.9)
  (h2 : avg_6 = 42)
  (h3 : result = 34.25) :
  let total_runs_10 := avg_10 * 10
  let total_runs_6 := avg_6 * 6
  let total_runs_4 := total_runs_10 - total_runs_6
  let avg_4 := total_runs_4 / 4
  avg_4 = result :=
  sorry

end average_last_4_matches_l49_49511


namespace max_min_x_sub_2y_l49_49646

theorem max_min_x_sub_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 0 ≤ x - 2*y ∧ x - 2*y ≤ 10 :=
sorry

end max_min_x_sub_2y_l49_49646


namespace RU_eq_825_l49_49087

variables (P Q R S T U : Type)
variables (PQ QR RP QS SR : ℝ)
variables (RU : ℝ)
variables (hPQ : PQ = 13)
variables (hQR : QR = 30)
variables (hRP : RP = 26)
variables (hQS : QS = 10)
variables (hSR : SR = 20)

theorem RU_eq_825 :
  RU = 8.25 :=
sorry

end RU_eq_825_l49_49087


namespace union_P_complement_Q_l49_49117

open Set

def P : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }
def R : Set ℝ := { x | -2 < x ∧ x < 2 }
def PQ_union : Set ℝ := P ∪ R

theorem union_P_complement_Q : PQ_union = { x | -2 < x ∧ x ≤ 3 } :=
by sorry

end union_P_complement_Q_l49_49117


namespace maoming_population_scientific_notation_l49_49460

-- Definitions for conditions
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- The main theorem to prove
theorem maoming_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 6800000 ∧ a = 6.8 ∧ n = 6 :=
sorry

end maoming_population_scientific_notation_l49_49460


namespace box_calories_l49_49939

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end box_calories_l49_49939


namespace num_4digit_special_integers_l49_49649

noncomputable def count_valid_4digit_integers : ℕ :=
  let first_two_options := 3 * 3 -- options for the first two digits
  let valid_last_two_pairs := 4 -- (6,9), (7,8), (8,7), (9,6)
  first_two_options * valid_last_two_pairs

theorem num_4digit_special_integers : count_valid_4digit_integers = 36 :=
by
  sorry

end num_4digit_special_integers_l49_49649


namespace sum_of_roots_l49_49924

-- Define the quadratic equation whose roots are the excluded domain values C and D
def quadratic_eq (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

-- Define C and D as the roots of the quadratic equation
def is_root (x : ℝ) : Prop := quadratic_eq x

-- Define C and D as the specific roots of the given quadratic equation
axiom C : ℝ
axiom D : ℝ

-- Assert that C and D are the roots of the quadratic equation
axiom hC : is_root C
axiom hD : is_root D

-- Statement to prove
theorem sum_of_roots : C + D = 3 :=
by sorry

end sum_of_roots_l49_49924


namespace green_light_probability_l49_49056

-- Define the durations of the red, green, and yellow lights
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

-- Define the total cycle time
def total_cycle_time : ℕ := red_light_duration + green_light_duration + yellow_light_duration

-- Define the expected probability
def expected_probability : ℚ := 5 / 12

-- Prove the probability of seeing a green light equals the expected_probability
theorem green_light_probability :
  (green_light_duration : ℚ) / (total_cycle_time : ℚ) = expected_probability :=
by
  sorry

end green_light_probability_l49_49056


namespace price_per_sq_ft_l49_49828

def house_sq_ft : ℕ := 2400
def barn_sq_ft : ℕ := 1000
def total_property_value : ℝ := 333200

theorem price_per_sq_ft : 
  (total_property_value / (house_sq_ft + barn_sq_ft)) = 98 := 
by 
  sorry

end price_per_sq_ft_l49_49828


namespace linear_function_quadrant_l49_49807

theorem linear_function_quadrant (x y : ℝ) (h : y = 2 * x - 3) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = 2 * x - 3) :=
sorry

end linear_function_quadrant_l49_49807


namespace minimum_value_of_expression_l49_49231

noncomputable def f (x : ℝ) : ℝ := 16^x - 2^x + x^2 + 1

theorem minimum_value_of_expression : ∃ (x : ℝ), f x = 1 ∧ ∀ y : ℝ, f y ≥ 1 := 
sorry

end minimum_value_of_expression_l49_49231


namespace intersection_empty_set_l49_49463

def M : Set ℝ := { y | ∃ x, x > 0 ∧ y = 2^x }
def N : Set ℝ := { y | ∃ x, y = Real.sqrt (2*x - x^2) }

theorem intersection_empty_set :
  M ∩ N = ∅ :=
by
  sorry

end intersection_empty_set_l49_49463


namespace cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l49_49781

-- Problem 1
theorem cylinder_lateral_area (C H : ℝ) (hC : C = 1.8) (hH : H = 1.5) :
  C * H = 2.7 := by sorry 

-- Problem 2
theorem cylinder_volume (D H : ℝ) (hD : D = 3) (hH : H = 8) :
  (3.14 * ((D * 10 / 2) ^ 2) * H) = 5652 :=
by sorry

-- Problem 3
theorem cylinder_surface_area (r h : ℝ) (hr : r = 6) (hh : h = 5) :
    (3.14 * r * 2 * h + 3.14 * r ^ 2 * 2) = 414.48 :=
by sorry

-- Problem 4
theorem cone_volume (B H : ℝ) (hB : B = 18.84) (hH : H = 6) :
  (1 / 3 * B * H) = 37.68 :=
by sorry

end cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l49_49781


namespace new_job_hourly_wage_l49_49188

def current_job_weekly_earnings : ℝ := 8 * 10
def new_job_hours_per_week : ℝ := 4
def new_job_bonus : ℝ := 35
def new_job_expected_additional_wage : ℝ := 15

theorem new_job_hourly_wage (W : ℝ) 
  (h_current_job : current_job_weekly_earnings = 80)
  (h_new_job : new_job_hours_per_week * W + new_job_bonus = current_job_weekly_earnings + new_job_expected_additional_wage) : 
  W = 15 :=
by 
  sorry

end new_job_hourly_wage_l49_49188


namespace min_value_fraction_l49_49661

theorem min_value_fraction (a b : ℝ) (h1 : 2 * a + b = 3) (h2 : a > 0) (h3 : b > 0) (h4 : ∃ n : ℕ, b = n) : 
  (∃ a b : ℝ, 2 * a + b = 3 ∧ a > 0 ∧ b > 0 ∧ (∃ n : ℕ, b = n) ∧ ((1/(2*a) + 2/b) = 2)) := 
by
  sorry

end min_value_fraction_l49_49661


namespace tank_full_time_l49_49575

def tank_capacity : ℕ := 900
def fill_rate_A : ℕ := 40
def fill_rate_B : ℕ := 30
def drain_rate_C : ℕ := 20
def cycle_time : ℕ := 3
def net_fill_per_cycle : ℕ := fill_rate_A + fill_rate_B - drain_rate_C

theorem tank_full_time :
  (tank_capacity / net_fill_per_cycle) * cycle_time = 54 :=
by
  sorry

end tank_full_time_l49_49575


namespace cherry_sodas_l49_49038

theorem cherry_sodas (C O : ℕ) (h1 : O = 2 * C) (h2 : C + O = 24) : C = 8 :=
by sorry

end cherry_sodas_l49_49038


namespace scientific_notation_of_274M_l49_49379

theorem scientific_notation_of_274M :
  274000000 = 2.74 * 10^8 := 
by 
  sorry

end scientific_notation_of_274M_l49_49379


namespace arrange_banana_l49_49780

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l49_49780


namespace find_perpendicular_vector_l49_49653

def vector_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_magnitude_equal (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 ^ 2 + v1.2 ^ 2) = (v2.1 ^ 2 + v2.2 ^ 2)

theorem find_perpendicular_vector (a b : ℝ) :
  ∃ n : ℝ × ℝ, vector_perpendicular (a, b) n ∧ vector_magnitude_equal (a, b) n ∧ n = (b, -a) :=
by
  sorry

end find_perpendicular_vector_l49_49653


namespace turnip_mixture_l49_49487

theorem turnip_mixture (cups_potatoes total_turnips : ℕ) (h_ratio : 20 = 5 * 4) (h_turnips : total_turnips = 8) :
    cups_potatoes = 2 :=
by
    have ratio := h_ratio
    have turnips := h_turnips
    sorry

end turnip_mixture_l49_49487


namespace polygon_sides_eq_six_l49_49373

theorem polygon_sides_eq_six (n : ℕ) (S_i S_e : ℕ) :
  S_i = 2 * S_e →
  S_e = 360 →
  (n - 2) * 180 = S_i →
  n = 6 :=
by
  sorry

end polygon_sides_eq_six_l49_49373


namespace rahul_batting_average_before_match_l49_49864

open Nat

theorem rahul_batting_average_before_match (R : ℕ) (A : ℕ) :
  (R + 69 = 6 * 54) ∧ (A = R / 5) → (A = 51) :=
by
  sorry

end rahul_batting_average_before_match_l49_49864


namespace moles_HCl_formed_l49_49630

-- Define the initial moles of CH4 and Cl2
def CH4_initial : ℕ := 2
def Cl2_initial : ℕ := 4

-- Define the balanced chemical equation in terms of the number of moles
def balanced_equation (CH4 : ℕ) (Cl2 : ℕ) : Prop :=
  CH4 + 4 * Cl2 = 1 * CH4 + 4 * Cl2

-- Theorem statement: Given the conditions, prove the number of moles of HCl formed is 4
theorem moles_HCl_formed (CH4_initial Cl2_initial : ℕ) (h_CH4 : CH4_initial = 2) (h_Cl2 : Cl2_initial = 4) :
  ∃ (HCl : ℕ), HCl = 4 :=
  sorry

end moles_HCl_formed_l49_49630


namespace sum_of_two_primes_is_multiple_of_six_l49_49081

theorem sum_of_two_primes_is_multiple_of_six
  (p q r : ℕ)
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) (hr_gt_3 : r > 3)
  (h_sum_prime : Nat.Prime (p + q + r)) : 
  (p + q) % 6 = 0 ∨ (p + r) % 6 = 0 ∨ (q + r) % 6 = 0 :=
sorry

end sum_of_two_primes_is_multiple_of_six_l49_49081


namespace identify_heaviest_and_lightest_13_weighings_l49_49690

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end identify_heaviest_and_lightest_13_weighings_l49_49690


namespace angle_triple_supplement_l49_49161

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l49_49161


namespace dennis_pants_purchase_l49_49920

theorem dennis_pants_purchase
  (pants_cost : ℝ) 
  (pants_discount : ℝ) 
  (socks_cost : ℝ) 
  (socks_discount : ℝ) 
  (socks_quantity : ℕ)
  (total_spent : ℝ)
  (discounted_pants_cost : ℝ)
  (discounted_socks_cost : ℝ)
  (pants_quantity : ℕ) :
  pants_cost = 110.00 →
  pants_discount = 0.30 →
  socks_cost = 60.00 →
  socks_discount = 0.30 →
  socks_quantity = 2 →
  total_spent = 392.00 →
  discounted_pants_cost = pants_cost * (1 - pants_discount) →
  discounted_socks_cost = socks_cost * (1 - socks_discount) →
  total_spent = socks_quantity * discounted_socks_cost + pants_quantity * discounted_ppants_cost →
  pants_quantity = 4 :=
by
  intros
  sorry

end dennis_pants_purchase_l49_49920


namespace sheep_count_l49_49550

-- Define the conditions
def TotalAnimals : ℕ := 200
def NumberCows : ℕ := 40
def NumberGoats : ℕ := 104

-- Define the question and its corresponding answer
def NumberSheep : ℕ := TotalAnimals - (NumberCows + NumberGoats)

-- State the theorem
theorem sheep_count : NumberSheep = 56 := by
  -- Skipping the proof
  sorry

end sheep_count_l49_49550


namespace smallest_solution_l49_49048

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l49_49048


namespace bert_total_stamp_cost_l49_49621

theorem bert_total_stamp_cost :
    let numA := 150
    let numB := 90
    let numC := 60
    let priceA := 2
    let priceB := 3
    let priceC := 5
    let costA := numA * priceA
    let costB := numB * priceB
    let costC := numC * priceC
    let total_cost := costA + costB + costC
    total_cost = 870 := 
by
    sorry

end bert_total_stamp_cost_l49_49621


namespace seq_contains_exactly_16_twos_l49_49121

-- Define a helper function to count occurrences of a digit in a number
def count_digit (d : Nat) (n : Nat) : Nat :=
  (n.digits 10).count d

-- Define a function to sum occurrences of the digit '2' in a list of numbers
def total_twos_in_sequence (seq : List Nat) : Nat :=
  seq.foldl (λ acc n => acc + count_digit 2 n) 0

-- Define the sequence we are interested in
def seq : List Nat := [2215, 2216, 2217, 2218, 2219, 2220, 2221]

-- State the theorem we need to prove
theorem seq_contains_exactly_16_twos : total_twos_in_sequence seq = 16 := 
by
  -- We do not provide the proof here according to the given instructions
  sorry

end seq_contains_exactly_16_twos_l49_49121


namespace initial_sand_in_bucket_A_l49_49009

theorem initial_sand_in_bucket_A (C : ℝ) : 
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  x / C = 1 / 4 := by
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  show x / C = 1 / 4
  sorry

end initial_sand_in_bucket_A_l49_49009


namespace original_number_doubled_added_trebled_l49_49513

theorem original_number_doubled_added_trebled (x : ℤ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by
  intro h
  -- The proof is omitted as instructed.
  sorry

end original_number_doubled_added_trebled_l49_49513


namespace blue_balls_count_l49_49702

def num_purple : Nat := 7
def num_yellow : Nat := 11
def min_tries : Nat := 19

theorem blue_balls_count (num_blue: Nat): num_blue = 1 :=
by
  have worst_case_picks := num_purple + num_yellow
  have h := min_tries
  sorry

end blue_balls_count_l49_49702


namespace length_of_place_mat_l49_49406

noncomputable def length_of_mat
  (R : ℝ)
  (w : ℝ)
  (n : ℕ)
  (θ : ℝ) : ℝ :=
  2 * R * Real.sin (θ / 2)

theorem length_of_place_mat :
  ∃ y : ℝ, y = length_of_mat 5 1 7 (360 / 7) := by
  use 4.38
  sorry

end length_of_place_mat_l49_49406


namespace problem_statement_l49_49889

theorem problem_statement :
  ∃ (w x y z : ℕ), (2^w * 3^x * 5^y * 7^z = 588) ∧ (2 * w + 3 * x + 5 * y + 7 * z = 21) :=
by
  sorry

end problem_statement_l49_49889


namespace socks_count_l49_49759

theorem socks_count :
  ∃ (x y z : ℕ), x + y + z = 12 ∧ x + 3 * y + 4 * z = 24 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 <= z ∧ x = 7 :=
by
  sorry

end socks_count_l49_49759


namespace possible_strings_after_moves_l49_49008

theorem possible_strings_after_moves : 
  let initial_string := "HHMMMMTT"
  let moves := [("HM", "MH"), ("MT", "TM"), ("TH", "HT")]
  let binom := Nat.choose 8 4
  binom = 70 := by
  sorry

end possible_strings_after_moves_l49_49008


namespace necessary_not_sufficient_for_circle_l49_49559

theorem necessary_not_sufficient_for_circle (a : ℝ) :
  (a ≤ 2 → (x^2 + y^2 - 2*x + 2*y + a = 0 → ∃ r : ℝ, r > 0)) ∧
  (a ≤ 2 ∧ ∃ b, b < 2 → a = b) := sorry

end necessary_not_sufficient_for_circle_l49_49559


namespace hyperbola_equation_l49_49967

theorem hyperbola_equation
  (a b m n e e' c' : ℝ)
  (h1 : 2 * a^2 + b^2 = 2)
  (h2 : e * e' = 1)
  (h_c : c' = e * m)
  (h_b : b^2 = m^2 - n^2)
  (h_e : e = n / m) : 
  y^2 - x^2 = 2 := 
sorry

end hyperbola_equation_l49_49967


namespace mother_l49_49176

def problem_conditions (D M : ℤ) : Prop :=
  (2 * D + M = 70) ∧ (D + 2 * M = 95)

theorem mother's_age_is_40 (D M : ℤ) (h : problem_conditions D M) : M = 40 :=
by sorry

end mother_l49_49176


namespace problem1_problem2_problem3_l49_49988

-- Definition of the sequence
def a (n : ℕ) (k : ℚ) : ℚ := (k * n - 3) / (n - 3 / 2)

-- The first condition proof problem
theorem problem1 (k : ℚ) : (∀ n : ℕ, a n k = (a (n + 1) k + a (n - 1) k) / 2) → k = 2 :=
sorry

-- The second condition proof problem
theorem problem2 (k : ℚ) : 
  k ≠ 2 → 
  (if k > 2 then (a 1 k < k ∧ a 2 k = max (a 1 k) (a 2 k))
   else if k < 2 then (a 2 k < k ∧ a 1 k = max (a 1 k) (a 2 k))
   else False) :=
sorry

-- The third condition proof problem
theorem problem3 (k : ℚ) : 
  (∀ n : ℕ, n > 0 → a n k > (k * 2^n + (-1)^n) / 2^n) → 
  101 / 48 < k ∧ k < 13 / 6 :=
sorry

end problem1_problem2_problem3_l49_49988


namespace cos_double_angle_l49_49887

theorem cos_double_angle (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : Real.cos (2 * x) = 1 / 2 := 
sorry

end cos_double_angle_l49_49887


namespace problem_one_problem_two_l49_49126

variable {α : ℝ}

theorem problem_one (h : Real.tan (π + α) = -1 / 2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2 * π) + Real.sin (4 * π - α)) = -7 / 9 :=
sorry

theorem problem_two (h : Real.tan (π + α) = -1 / 2) :
  Real.sin (α - 7 * π) * Real.cos (α + 5 * π) = -2 / 5 :=
sorry

end problem_one_problem_two_l49_49126


namespace milton_apple_pie_slices_l49_49813

theorem milton_apple_pie_slices :
  ∀ (A : ℕ),
  (∀ (peach_pie_slices_per : ℕ), peach_pie_slices_per = 6) →
  (∀ (apple_pie_slices_sold : ℕ), apple_pie_slices_sold = 56) →
  (∀ (peach_pie_slices_sold : ℕ), peach_pie_slices_sold = 48) →
  (∀ (total_pies_sold : ℕ), total_pies_sold = 15) →
  (∃ (apple_pie_slices : ℕ), apple_pie_slices = 56 / (total_pies_sold - (peach_pie_slices_sold / peach_pie_slices_per))) → 
  A = 8 :=
by sorry

end milton_apple_pie_slices_l49_49813


namespace interval_sum_l49_49548

theorem interval_sum (a b : ℝ) (h : ∀ x,  |3 * x - 80| ≤ |2 * x - 105| ↔ (a ≤ x ∧ x ≤ b)) :
  a + b = 12 :=
sorry

end interval_sum_l49_49548


namespace find_limit_of_hours_l49_49243

def regular_rate : ℝ := 16
def overtime_rate (r : ℝ) : ℝ := r * 1.75
def total_compensation : ℝ := 920
def total_hours : ℝ := 50

theorem find_limit_of_hours : 
  ∃ (L : ℝ), 
    total_compensation = (regular_rate * L) + ((overtime_rate regular_rate) * (total_hours - L)) →
    L = 40 :=
by
  sorry

end find_limit_of_hours_l49_49243


namespace range_of_a_l49_49225

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 :=
by
  sorry

end range_of_a_l49_49225


namespace C_investment_l49_49695

theorem C_investment (A B C_profit total_profit : ℝ) (hA : A = 24000) (hB : B = 32000) (hC_profit : C_profit = 36000) (h_total_profit : total_profit = 92000) (x : ℝ) (h : x / (A + B + x) = C_profit / total_profit) : x = 36000 := 
by
  sorry

end C_investment_l49_49695


namespace percentage_increase_consumption_l49_49810

theorem percentage_increase_consumption
  (T C : ℝ) 
  (h_tax : ∀ t, t = 0.60 * T)
  (h_revenue : ∀ r, r = 0.75 * T * C) :
  1.25 * C = (0.75 * T * C) / (0.60 * T) := by
sorry

end percentage_increase_consumption_l49_49810


namespace prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l49_49317

-- Definitions
def classes := 4
def students := 4
def total_distributions := classes ^ students

-- Problem 1
theorem prob_each_class_receives_one : 
  (A_4 ^ 4) / total_distributions = 3 / 32 := sorry

-- Problem 2
theorem prob_at_least_one_class_empty : 
  1 - (A_4 ^ 4) / total_distributions = 29 / 32 := sorry

-- Problem 3
theorem prob_exactly_one_class_empty :
  (C_4 ^ 1 * C_4 ^ 2 * C_3 ^ 1 * C_2 ^ 1) / total_distributions = 9 / 16 := sorry

end prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l49_49317


namespace value_of_f_f_2_l49_49050

def f (x : ℤ) : ℤ := 4 * x^2 + 2 * x - 1

theorem value_of_f_f_2 : f (f 2) = 1481 := by
  sorry

end value_of_f_f_2_l49_49050


namespace number_of_ninth_graders_l49_49224

def num_students_total := 50
def num_students_7th (x : Int) := 2 * x - 1
def num_students_8th (x : Int) := x

theorem number_of_ninth_graders (x : Int) :
  num_students_7th x + num_students_8th x + (51 - 3 * x) = num_students_total := by
  sorry

end number_of_ninth_graders_l49_49224


namespace problem_solution_l49_49241

def is_desirable_n (n : ℕ) : Prop :=
  ∃ (r b : ℕ), n = r + b ∧ r^2 - r*b + b^2 = 2007 ∧ 3 ∣ r ∧ 3 ∣ b

theorem problem_solution :
  ∀ n : ℕ, (is_desirable_n n → n = 69 ∨ n = 84) :=
by
  sorry

end problem_solution_l49_49241


namespace police_coverage_l49_49215

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define the streets
def Streets : List (List Intersection) :=
  [ [A, B, C, D],    -- Horizontal street 1
    [E, F, G],       -- Horizontal street 2
    [H, I, J, K],    -- Horizontal street 3
    [A, E, H],       -- Vertical street 1
    [B, F, I],       -- Vertical street 2
    [D, G, J],       -- Vertical street 3
    [H, F, C],       -- Diagonal street 1
    [C, G, K]        -- Diagonal street 2
  ]

-- Define the set of intersections where police officers are 
def policeIntersections : List Intersection := [B, G, H]

-- State the theorem to be proved
theorem police_coverage : 
  ∀ (street : List Intersection), street ∈ Streets → 
  ∃ (i : Intersection), i ∈ policeIntersections ∧ i ∈ street := 
sorry

end police_coverage_l49_49215


namespace calculate_expression_l49_49396

theorem calculate_expression : (-1 : ℝ) ^ 2 + (1 / 3 : ℝ) ^ 0 = 2 := 
by
  sorry

end calculate_expression_l49_49396


namespace marble_total_weight_l49_49435

theorem marble_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 + 0.21666666666666667 + 0.4583333333333333 + 0.12777777777777778 = 1.5527777777777777 :=
by
  sorry

end marble_total_weight_l49_49435


namespace distance_Reims_to_Chaumont_l49_49213

noncomputable def distance_Chalons_Vitry : ℝ := 30
noncomputable def distance_Vitry_Chaumont : ℝ := 80
noncomputable def distance_Chaumont_SaintQuentin : ℝ := 236
noncomputable def distance_SaintQuentin_Reims : ℝ := 86
noncomputable def distance_Reims_Chalons : ℝ := 40

theorem distance_Reims_to_Chaumont :
  distance_Reims_Chalons + 
  distance_Chalons_Vitry + 
  distance_Vitry_Chaumont = 150 :=
sorry

end distance_Reims_to_Chaumont_l49_49213


namespace min_value_x_add_y_div_2_l49_49825

theorem min_value_x_add_y_div_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 2 * x - y = 0) :
  ∃ x y, 0 < x ∧ 0 < y ∧ (x * y - 2 * x - y = 0 ∧ x + y / 2 = 4) :=
sorry

end min_value_x_add_y_div_2_l49_49825


namespace sqrt_4_eq_pm2_l49_49767

theorem sqrt_4_eq_pm2 : {y : ℝ | y^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_4_eq_pm2_l49_49767


namespace evaluate_expression_l49_49244

theorem evaluate_expression (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 2018) 
  (h2 : 3 * a + 8 * b + 24 * c + 37 * d = 2018) : 
  3 * b + 8 * c + 24 * d + 37 * a = 1215 :=
by 
  sorry

end evaluate_expression_l49_49244


namespace sandy_savings_l49_49417

-- Definition and conditions
def last_year_savings (S : ℝ) : ℝ := 0.06 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_savings (S : ℝ) : ℝ := 1.8333333333333333 * last_year_savings S

-- The percentage P of this year's salary that Sandy saved
def this_year_savings_perc (S : ℝ) (P : ℝ) : Prop :=
  P * this_year_salary S = this_year_savings S

-- The proof statement: Sandy saved 10% of her salary this year
theorem sandy_savings (S : ℝ) (P : ℝ) (h: this_year_savings_perc S P) : P = 0.10 :=
  sorry

end sandy_savings_l49_49417


namespace speed_in_still_water_l49_49307

-- Given conditions
def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 41

-- Question: Prove the speed of the man in still water is 33 kmph.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 33 := 
by 
  sorry

end speed_in_still_water_l49_49307


namespace prove_y_l49_49629

theorem prove_y (x y : ℝ) (h1 : 3 * x^2 - 4 * x + 7 * y + 3 = 0) (h2 : 3 * x - 5 * y + 6 = 0) :
  25 * y^2 - 39 * y + 69 = 0 := sorry

end prove_y_l49_49629


namespace range_of_x_l49_49926

theorem range_of_x (total_students math_club chemistry_club : ℕ) (h_total : total_students = 45) 
(h_math : math_club = 28) (h_chemistry : chemistry_club = 21) (x : ℕ) :
  4 ≤ x ∧ x ≤ 21 ↔ (28 + 21 - x ≤ 45) :=
by sorry

end range_of_x_l49_49926


namespace suit_price_after_discount_l49_49371

-- Definitions based on given conditions 
def original_price : ℝ := 200
def price_increase : ℝ := 0.30 * original_price
def new_price : ℝ := original_price + price_increase
def discount : ℝ := 0.30 * new_price
def final_price : ℝ := new_price - discount

-- The theorem
theorem suit_price_after_discount :
  final_price = 182 :=
by
  -- Here we would provide the proof if required
  sorry

end suit_price_after_discount_l49_49371


namespace checkered_square_division_l49_49378

theorem checkered_square_division (m n k d m1 n1 : ℕ) (h1 : m^2 = n * k)
  (h2 : d = Nat.gcd m n) (hm : m = m1 * d) (hn : n = n1 * d)
  (h3 : Nat.gcd m1 n1 = 1) : 
  ∃ (part_size : ℕ), 
    part_size = n ∧ (∃ (pieces : ℕ), pieces = k) ∧ m^2 = pieces * part_size := 
sorry

end checkered_square_division_l49_49378


namespace fraction_product_eq_l49_49029

theorem fraction_product_eq :
  (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) = 4 / 9 :=
by
  sorry

end fraction_product_eq_l49_49029


namespace max_value_expression_l49_49059

noncomputable def f : Real → Real := λ x => 3 * Real.sin x + 4 * Real.cos x

theorem max_value_expression (θ : Real) (h_max : ∀ x, f x ≤ 5) :
  (3 * Real.sin θ + 4 * Real.cos θ = 5) →
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end max_value_expression_l49_49059


namespace inequality_proof_l49_49125

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l49_49125


namespace factorable_polynomial_l49_49217

theorem factorable_polynomial (a b : ℝ) :
  (∀ x y : ℝ, ∃ u v p q : ℝ, (x + uy + v) * (x + py + q) = x * (x + 4) + a * (y^2 - 1) + 2 * b * y) ↔
  (a + 2)^2 + b^2 = 4 :=
  sorry

end factorable_polynomial_l49_49217


namespace integers_with_product_72_and_difference_4_have_sum_20_l49_49662

theorem integers_with_product_72_and_difference_4_have_sum_20 :
  ∃ (x y : ℕ), (x * y = 72) ∧ (x - y = 4) ∧ (x + y = 20) :=
sorry

end integers_with_product_72_and_difference_4_have_sum_20_l49_49662


namespace quadratic_roots_m_value_l49_49906

theorem quadratic_roots_m_value
  (x1 x2 m : ℝ)
  (h1 : x1^2 + 2 * x1 + m = 0)
  (h2 : x2^2 + 2 * x2 + m = 0)
  (h3 : x1 + x2 = x1 * x2 - 1) :
  m = -1 :=
sorry

end quadratic_roots_m_value_l49_49906


namespace total_students_correct_l49_49461

def num_first_graders : ℕ := 358
def num_second_graders : ℕ := num_first_graders - 64
def total_students : ℕ := num_first_graders + num_second_graders

theorem total_students_correct : total_students = 652 :=
by
  sorry

end total_students_correct_l49_49461


namespace faye_pencils_allocation_l49_49676

theorem faye_pencils_allocation (pencils total_pencils rows : ℕ) (h_pencils : total_pencils = 6) (h_rows : rows = 2) (h_allocation : pencils = total_pencils / rows) : pencils = 3 := by
  sorry

end faye_pencils_allocation_l49_49676


namespace range_of_a_l49_49057

noncomputable def discriminant (a : ℝ) : ℝ :=
  (2 * a)^2 - 4 * 1 * 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end range_of_a_l49_49057


namespace expression_value_l49_49258

noncomputable def evaluate_expression : ℝ :=
  Real.logb 2 (3 * 11 + Real.exp (4 - 8)) + 3 * Real.sin (Real.pi^2 - Real.sqrt ((6 * 4) / 3 - 4))

theorem expression_value : evaluate_expression = 3.832 := by
  sorry

end expression_value_l49_49258


namespace cannot_tile_remaining_with_dominoes_l49_49615

def can_tile_remaining_board (pieces : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j

theorem cannot_tile_remaining_with_dominoes : 
  ∃ (pieces : List (ℕ × ℕ)), (∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 10) ∧ (1 ≤ j ∧ j ≤ 10) → ∃ (piece : ℕ × ℕ), piece ∈ pieces ∧ piece.1 = i ∧ piece.2 = j) ∧ ¬ can_tile_remaining_board pieces :=
sorry

end cannot_tile_remaining_with_dominoes_l49_49615


namespace simplify_and_evaluate_l49_49741

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -1) (hy : y = -1/3) :
  ((3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y)) = 8 := by
  sorry

end simplify_and_evaluate_l49_49741


namespace pages_read_per_hour_l49_49693

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l49_49693


namespace value_of_a_star_b_l49_49252

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l49_49252


namespace yard_length_is_correct_l49_49257

-- Definitions based on the conditions
def trees : ℕ := 26
def distance_between_trees : ℕ := 11

-- Theorem stating that the length of the yard is 275 meters
theorem yard_length_is_correct : (trees - 1) * distance_between_trees = 275 :=
by sorry

end yard_length_is_correct_l49_49257


namespace tan_half_sum_l49_49806

theorem tan_half_sum (p q : ℝ)
  (h1 : Real.cos p + Real.cos q = (1:ℝ)/3)
  (h2 : Real.sin p + Real.sin q = (8:ℝ)/17) :
  Real.tan ((p + q) / 2) = (24:ℝ)/17 := 
sorry

end tan_half_sum_l49_49806


namespace roots_abs_gt_4_or_l49_49440

theorem roots_abs_gt_4_or
    (r1 r2 : ℝ)
    (q : ℝ) 
    (h1 : r1 ≠ r2)
    (h2 : r1 + r2 = -q)
    (h3 : r1 * r2 = -10) :
    |r1| > 4 ∨ |r2| > 4 :=
sorry

end roots_abs_gt_4_or_l49_49440


namespace F_equiv_A_l49_49835

-- Define the function F
def F : ℝ → ℝ := sorry

-- Given condition
axiom F_property (x : ℝ) : F ((1 - x) / (1 + x)) = x

-- The theorem that needs to be proved
theorem F_equiv_A (x : ℝ) : F (-2 - x) = -2 - F x := sorry

end F_equiv_A_l49_49835


namespace hyperbola_find_a_b_l49_49447

def hyperbola_conditions (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (∃ e : ℝ, e = 2) ∧ (∃ c : ℝ, c = 4)

theorem hyperbola_find_a_b (a b : ℝ) : hyperbola_conditions a b → a = 2 ∧ b = 2 * Real.sqrt 3 := 
sorry

end hyperbola_find_a_b_l49_49447


namespace compute_g_five_times_l49_49382

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then - x^3 else x + 10

theorem compute_g_five_times (x : ℤ) (h : x = 2) : g (g (g (g (g x)))) = -8 := by
  sorry

end compute_g_five_times_l49_49382


namespace decision_block_has_two_exits_l49_49451

-- Define the conditions based on the problem
def output_block_exits := 1
def processing_block_exits := 1
def start_end_block_exits := 0
def decision_block_exits := 2

-- The proof statement
theorem decision_block_has_two_exits :
  (output_block_exits = 1) ∧
  (processing_block_exits = 1) ∧
  (start_end_block_exits = 0) ∧
  (decision_block_exits = 2) →
  decision_block_exits = 2 :=
by
  sorry

end decision_block_has_two_exits_l49_49451


namespace profit_percent_l49_49216

theorem profit_percent (cost_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (n_pens : ℕ) 
  (h1 : n_pens = 60) (h2 : marked_price = 1) (h3 : cost_price = (46 : ℝ) / (60 : ℝ)) 
  (h4 : selling_price = 0.99 * marked_price) : 
  (selling_price - cost_price) / cost_price * 100 = 29.11 :=
by
  sorry

end profit_percent_l49_49216


namespace g_function_ratio_l49_49144

theorem g_function_ratio (g : ℝ → ℝ) (h : ∀ c d : ℝ, c^3 * g d = d^3 * g c) (hg3 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := 
by
  sorry

end g_function_ratio_l49_49144


namespace geometric_sequence_a3_eq_2_l49_49802

theorem geometric_sequence_a3_eq_2 
  (a_1 a_3 a_5 : ℝ) 
  (h1 : a_1 * a_3 * a_5 = 8) 
  (h2 : a_3^2 = a_1 * a_5) : 
  a_3 = 2 :=
by 
  sorry

end geometric_sequence_a3_eq_2_l49_49802


namespace geometric_sequence_common_ratio_l49_49499

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = -1) 
  (h2 : a 2 + a 3 = -2) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  q = -2 ∨ q = 1 := 
by sorry

end geometric_sequence_common_ratio_l49_49499


namespace mrs_sheridan_gave_away_14_cats_l49_49608

def num_initial_cats : ℝ := 17.0
def num_left_cats : ℝ := 3.0
def num_given_away (x : ℝ) : Prop := num_initial_cats - x = num_left_cats

theorem mrs_sheridan_gave_away_14_cats : num_given_away 14.0 :=
by
  sorry

end mrs_sheridan_gave_away_14_cats_l49_49608


namespace geom_seq_min_value_l49_49731

theorem geom_seq_min_value :
  let a1 := 2
  ∃ r : ℝ, ∀ a2 a3,
    a2 = 2 * r ∧ 
    a3 = 2 * r^2 →
    3 * a2 + 6 * a3 = -3/2 := by
  sorry

end geom_seq_min_value_l49_49731


namespace greatest_integer_solution_l49_49951

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 12 * n + 28 ≤ 0) : 6 ≤ n :=
sorry

end greatest_integer_solution_l49_49951


namespace tan_A_tan_B_l49_49494

theorem tan_A_tan_B (A B C : ℝ) (R : ℝ) (H F : ℝ)
  (HF : H + F = 26) (h1 : 2 * R * Real.cos A * Real.cos B = 8)
  (h2 : 2 * R * Real.sin A * Real.sin B = 26) :
  Real.tan A * Real.tan B = 13 / 4 :=
by
  sorry

end tan_A_tan_B_l49_49494


namespace owners_riding_to_total_ratio_l49_49114

theorem owners_riding_to_total_ratio (R W : ℕ) (h1 : 4 * R + 6 * W = 90) (h2 : R + W = 18) : R / (R + W) = 1 / 2 :=
by
  sorry

end owners_riding_to_total_ratio_l49_49114


namespace invalid_votes_percentage_is_correct_l49_49994

-- Definitions based on conditions
def total_votes : ℕ := 5500
def other_candidate_votes : ℕ := 1980
def valid_votes_percentage_other : ℚ := 0.45

-- Derived values
def valid_votes : ℚ := other_candidate_votes / valid_votes_percentage_other
def invalid_votes : ℚ := total_votes - valid_votes
def invalid_votes_percentage : ℚ := (invalid_votes / total_votes) * 100

-- Proof statement
theorem invalid_votes_percentage_is_correct :
  invalid_votes_percentage = 20 := sorry

end invalid_votes_percentage_is_correct_l49_49994


namespace find_first_term_and_common_difference_l49_49654

variable (n : ℕ)
variable (a_1 d : ℚ)

-- Definition of the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (n : ℕ) (a_1 d : ℚ) : ℚ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2

-- Theorem to prove
theorem find_first_term_and_common_difference 
  (a_1 d : ℚ) 
  (sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2) 
: a_1 = 1/2 ∧ d = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end find_first_term_and_common_difference_l49_49654


namespace line_passes_through_fixed_point_l49_49561

-- Statement to prove that the line always passes through the point (2, 2)
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, ∃ x y : ℝ, 
  (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0 ∧ x = 2 ∧ y = 2 :=
sorry

end line_passes_through_fixed_point_l49_49561


namespace number_of_integers_between_sqrt10_and_sqrt100_l49_49703

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end number_of_integers_between_sqrt10_and_sqrt100_l49_49703


namespace abs_non_positive_eq_zero_l49_49814

theorem abs_non_positive_eq_zero (y : ℚ) (h : |4 * y - 7| ≤ 0) : y = 7 / 4 :=
by
  sorry

end abs_non_positive_eq_zero_l49_49814


namespace rulers_in_drawer_l49_49134

-- conditions
def initial_rulers : ℕ := 46
def additional_rulers : ℕ := 25

-- question: total rulers in the drawer
def total_rulers : ℕ := initial_rulers + additional_rulers

-- proof statement: prove that total_rulers is 71
theorem rulers_in_drawer : total_rulers = 71 := by
  sorry

end rulers_in_drawer_l49_49134


namespace mean_weight_correct_l49_49873

def weights := [51, 60, 62, 64, 64, 65, 67, 73, 74, 74, 75, 76, 77, 78, 79]

noncomputable def mean_weight (weights : List ℕ) : ℚ :=
  (weights.sum : ℚ) / weights.length

theorem mean_weight_correct :
  mean_weight weights = 69.27 := by
  sorry

end mean_weight_correct_l49_49873


namespace find_f2_l49_49414

theorem find_f2 :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 2 * f x - 3 * f (1 / x) = x ^ 2) ∧ f 2 = 93 / 32) :=
sorry

end find_f2_l49_49414


namespace volume_relationship_l49_49557

open Real

theorem volume_relationship (r : ℝ) (A M C : ℝ)
  (hA : A = (1/3) * π * r^3)
  (hM : M = π * r^3)
  (hC : C = (4/3) * π * r^3) :
  A + M + (1/2) * C = 2 * π * r^3 :=
by
  sorry

end volume_relationship_l49_49557


namespace real_part_of_z_given_condition_l49_49070

open Complex

noncomputable def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_of_z_given_condition :
  ∀ (z : ℂ), (i * (z + 1) = -3 + 2 * i) → real_part_of_z z = 1 :=
by
  intro z h
  sorry

end real_part_of_z_given_condition_l49_49070


namespace parabola_opens_downwards_iff_l49_49895

theorem parabola_opens_downwards_iff (a : ℝ) : (∀ x : ℝ, (a - 1) * x^2 + 2 * x ≤ 0) ↔ a < 1 := 
sorry

end parabola_opens_downwards_iff_l49_49895


namespace point_in_second_quadrant_l49_49092

def isInSecondQuadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : isInSecondQuadrant (-1) 1 :=
by
  sorry

end point_in_second_quadrant_l49_49092


namespace profit_percentage_l49_49122

-- Given conditions
def CP : ℚ := 25 / 15
def SP : ℚ := 32 / 12

-- To prove profit percentage is 60%
theorem profit_percentage (CP SP : ℚ) (hCP : CP = 25 / 15) (hSP : SP = 32 / 12) :
  (SP - CP) / CP * 100 = 60 := 
by 
  sorry

end profit_percentage_l49_49122


namespace intersection_of_P_and_Q_l49_49401

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2 ∧ x ∈ Set.univ}
def Q : Set ℤ := {x | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {-2, -1, 0} :=
sorry

end intersection_of_P_and_Q_l49_49401


namespace range_of_distance_l49_49110

noncomputable def A (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def B (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

theorem range_of_distance (α β : ℝ) :
  1 ≤ Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ∧
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ≤ 5 :=
by
  sorry

end range_of_distance_l49_49110


namespace seating_arrangements_l49_49255

-- Define the participants
inductive Person : Type
| xiaoMing
| parent1
| parent2
| grandparent1
| grandparent2

open Person

-- Define the function to count seating arrangements
noncomputable def count_seating_arrangements : Nat :=
  let arrangements := [
    -- (Only one parent next to Xiao Ming, parents not next to each other)
    12,
    -- (Only one parent next to Xiao Ming, parents next to each other)
    24,
    -- (Both parents next to Xiao Ming)
    12
  ]
  arrangements.foldr (· + ·) 0

theorem seating_arrangements : count_seating_arrangements = 48 := by
  sorry

end seating_arrangements_l49_49255


namespace elementary_school_classes_count_l49_49894

theorem elementary_school_classes_count (E : ℕ) (donate_per_class : ℕ) (middle_school_classes : ℕ) (total_balls : ℕ) :
  donate_per_class = 5 →
  middle_school_classes = 5 →
  total_balls = 90 →
  5 * 2 * E + 5 * 2 * middle_school_classes = total_balls →
  E = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end elementary_school_classes_count_l49_49894


namespace unique_position_of_chess_piece_l49_49691

theorem unique_position_of_chess_piece (x y : ℕ) (h : x^2 + x * y - 2 * y^2 = 13) : (x = 5) ∧ (y = 4) :=
sorry

end unique_position_of_chess_piece_l49_49691


namespace total_amount_is_152_l49_49478

noncomputable def total_amount (p q r s t : ℝ) : ℝ := p + q + r + s + t

noncomputable def p_share (x : ℝ) : ℝ := 2 * x
noncomputable def q_share (x : ℝ) : ℝ := 1.75 * x
noncomputable def r_share (x : ℝ) : ℝ := 1.5 * x
noncomputable def s_share (x : ℝ) : ℝ := 1.25 * x
noncomputable def t_share (x : ℝ) : ℝ := 1.1 * x

theorem total_amount_is_152 (x : ℝ) (h1 : q_share x = 35) :
  total_amount (p_share x) (q_share x) (r_share x) (s_share x) (t_share x) = 152 := by
  sorry

end total_amount_is_152_l49_49478


namespace intersection_complement_B_l49_49618

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - 3 * x < 0 }
def B : Set ℝ := { x | abs x > 2 }

-- Complement of B
def complement_B : Set ℝ := { x | x ≥ -2 ∧ x ≤ 2 }

-- Final statement to prove the intersection equals the given set
theorem intersection_complement_B :
  A ∩ complement_B = { x : ℝ | 0 < x ∧ x ≤ 2 } := 
by 
  -- Proof omitted
  sorry

end intersection_complement_B_l49_49618


namespace compute_expression_l49_49282

-- Definition of the operation "minus the reciprocal of"
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement to prove the given problem
theorem compute_expression :
  ((diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5))) = -71 / 380 := 
sorry

end compute_expression_l49_49282


namespace median_CD_eq_altitude_from_C_eq_centroid_G_eq_l49_49520

namespace Geometry

/-- Vertices of the triangle -/
def A : ℝ × ℝ := (4, 4)
def B : ℝ × ℝ := (-4, 2)
def C : ℝ × ℝ := (2, 0)

/-- Proof of the equation of the median CD on the side AB -/
theorem median_CD_eq : ∀ (x y : ℝ), 3 * x + 2 * y - 6 = 0 :=
sorry

/-- Proof of the equation of the altitude from C to AB -/
theorem altitude_from_C_eq : ∀ (x y : ℝ), 4 * x + y - 8 = 0 :=
sorry

/-- Proof of the coordinates of the centroid G of triangle ABC -/
theorem centroid_G_eq : ∃ (x y : ℝ), x = 2 / 3 ∧ y = 2 :=
sorry

end Geometry

end median_CD_eq_altitude_from_C_eq_centroid_G_eq_l49_49520


namespace phil_cards_left_l49_49625

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end phil_cards_left_l49_49625


namespace area_intersection_A_B_l49_49426

noncomputable def A : Set (Real × Real) := {
  p | ∃ α β : ℝ, p.1 = 2 * Real.sin α + 2 * Real.sin β ∧ p.2 = 2 * Real.cos α + 2 * Real.cos β
}

noncomputable def B : Set (Real × Real) := {
  p | Real.sin (p.1 + p.2) * Real.cos (p.1 + p.2) ≥ 0
}

theorem area_intersection_A_B :
  let intersection := Set.inter A B
  let area : ℝ := 8 * Real.pi
  ∀ (x y : ℝ), (x, y) ∈ intersection → True := sorry

end area_intersection_A_B_l49_49426


namespace exp_arbitrarily_large_l49_49178

theorem exp_arbitrarily_large (a : ℝ) (h : a > 1) : ∀ y > 0, ∃ x > 0, a^x > y := by
  sorry

end exp_arbitrarily_large_l49_49178


namespace Micheal_completion_time_l49_49380

variable (W M A : ℝ)

-- Conditions
def condition1 (W M A : ℝ) : Prop := M + A = W / 20
def condition2 (W M A : ℝ) : Prop := A = (W - 14 * (M + A)) / 10

-- Goal
theorem Micheal_completion_time :
  (condition1 W M A) →
  (condition2 W M A) →
  M = W / 50 :=
by
  intros h1 h2
  sorry

end Micheal_completion_time_l49_49380


namespace emma_ate_more_than_liam_l49_49771

-- Definitions based on conditions
def emma_oranges : ℕ := 8
def liam_oranges : ℕ := 1

-- Lean statement to prove the question
theorem emma_ate_more_than_liam : emma_oranges - liam_oranges = 7 := by
  sorry

end emma_ate_more_than_liam_l49_49771


namespace count_factors_multiple_of_150_l49_49404

theorem count_factors_multiple_of_150 (n : ℕ) (h : n = 2^10 * 3^14 * 5^8) : 
  ∃ k, k = 980 ∧ ∀ d : ℕ, d ∣ n → 150 ∣ d → (d.factors.card = k) := sorry

end count_factors_multiple_of_150_l49_49404


namespace Alyssa_puppies_l49_49507

theorem Alyssa_puppies (initial_puppies give_away_puppies : ℕ) (h_initial : initial_puppies = 12) (h_give_away : give_away_puppies = 7) :
  initial_puppies - give_away_puppies = 5 :=
by
  sorry

end Alyssa_puppies_l49_49507


namespace optimal_position_theorem_l49_49917

noncomputable def optimal_position (a b a1 b1 : ℝ) : ℝ :=
  (b / 2) + (b1 / (2 * a1)) * (a - a1)

theorem optimal_position_theorem 
  (a b a1 b1 : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0) :
  ∃ x, x = optimal_position a b a1 b1 := by
  sorry

end optimal_position_theorem_l49_49917


namespace find_f_of_7_l49_49250

-- Defining the conditions in the problem.
variables (f : ℝ → ℝ)
variables (odd_f : ∀ x : ℝ, f (-x) = -f x)
variables (periodic_f : ∀ x : ℝ, f (x + 4) = f x)
variables (f_eqn : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = x + 2)

-- The statement of the problem, to prove f(7) = -3.
theorem find_f_of_7 : f 7 = -3 :=
by
  sorry

end find_f_of_7_l49_49250


namespace words_per_page_l49_49866

theorem words_per_page (p : ℕ) :
  (p ≤ 120) ∧ (154 * p % 221 = 145) → p = 96 := by
  sorry

end words_per_page_l49_49866


namespace max_and_min_of_expression_l49_49905

variable {x y : ℝ}

theorem max_and_min_of_expression (h : |5 * x + y| + |5 * x - y| = 20) : 
  (∃ (maxQ minQ : ℝ), maxQ = 124 ∧ minQ = 3 ∧ 
  (∀ z, z = x^2 - x * y + y^2 → z <= 124 ∧ z >= 3)) :=
sorry

end max_and_min_of_expression_l49_49905


namespace kyle_lift_weight_l49_49475

theorem kyle_lift_weight (this_year_weight last_year_weight : ℕ) 
  (h1 : this_year_weight = 80) 
  (h2 : this_year_weight = 3 * last_year_weight) : 
  (this_year_weight - last_year_weight) = 53 := by
  sorry

end kyle_lift_weight_l49_49475


namespace tile_difference_l49_49960

theorem tile_difference :
  let initial_blue_tiles := 20
  let initial_green_tiles := 15
  let first_border_tiles := 18
  let second_border_tiles := 18
  let total_green_tiles := initial_green_tiles + first_border_tiles + second_border_tiles
  let total_blue_tiles := initial_blue_tiles
  total_green_tiles - total_blue_tiles = 31 := 
by
  sorry

end tile_difference_l49_49960


namespace no_d1_d2_multiple_of_7_l49_49099
open Function

theorem no_d1_d2_multiple_of_7 (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 100) :
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  ¬(d1 * d2 % 7 = 0) :=
by
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  sorry

end no_d1_d2_multiple_of_7_l49_49099


namespace shaded_area_quadrilateral_l49_49793

theorem shaded_area_quadrilateral :
  let large_square_area := 11 * 11
  let small_square_area_1 := 1 * 1
  let small_square_area_2 := 2 * 2
  let small_square_area_3 := 3 * 3
  let small_square_area_4 := 4 * 4
  let other_non_shaded_areas := 12 + 15 + 14
  let total_non_shaded := small_square_area_1 + small_square_area_2 + small_square_area_3 + small_square_area_4 + other_non_shaded_areas
  let shaded_area := large_square_area - total_non_shaded
  shaded_area = 35 := by
  sorry

end shaded_area_quadrilateral_l49_49793


namespace proof_problem_l49_49567

-- Definition of the condition
def condition (y : ℝ) : Prop := 6 * y^2 + 5 = 2 * y + 10

-- Stating the theorem
theorem proof_problem : ∀ y : ℝ, condition y → (12 * y - 5)^2 = 133 :=
by
  intro y
  intro h
  sorry

end proof_problem_l49_49567


namespace find_omega_find_period_and_intervals_find_solution_set_l49_49979

noncomputable def omega_condition (ω : ℝ) :=
  0 < ω ∧ ω < 2

noncomputable def function_fx (ω : ℝ) (x : ℝ) := 
  3 * Real.sin (2 * ω * x + Real.pi / 3)

noncomputable def center_of_symmetry_condition (ω : ℝ) := 
  function_fx ω (-Real.pi / 6) = 0

noncomputable def period_condition (ω : ℝ) :=
  Real.pi / abs ω

noncomputable def intervals_of_increase (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, ((Real.pi / 12 + k * Real.pi) ≤ x) ∧ (x < (5 * Real.pi / 12 + k * Real.pi))

noncomputable def solution_set_fx_ge_half (x : ℝ) : Prop :=
  ∃ k : ℤ, (Real.pi / 12 + k * Real.pi) ≤ x ∧ (x ≤ 5 * Real.pi / 12 + k * Real.pi)

theorem find_omega : ∀ ω : ℝ, omega_condition ω ∧ center_of_symmetry_condition ω → omega = 1 := sorry

theorem find_period_and_intervals : 
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → period_condition ω = Real.pi :=
sorry

theorem find_solution_set :
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → (∀ x, solution_set_fx_ge_half x) :=
sorry

end find_omega_find_period_and_intervals_find_solution_set_l49_49979


namespace determine_right_triangle_l49_49012

variable (A B C : ℝ)
variable (AB BC AC : ℝ)

-- Conditions as definitions
def condition1 : Prop := A + C = B
def condition2 : Prop := A = 30 ∧ B = 60 ∧ C = 90 -- Since ratio 1:2:3 means A = 30, B = 60, C = 90

-- Proof problem statement
theorem determine_right_triangle (h1 : condition1 A B C) (h2 : condition2 A B C) : (B = 90) :=
sorry

end determine_right_triangle_l49_49012


namespace geometric_sequence_fourth_term_l49_49448

/-- In a geometric sequence with common ratio 2, where the sequence is denoted as {a_n},
and it is given that a_1 * a_3 = 6 * a_2, prove that a_4 = 24. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n)
  (h1 : a 1 * a 3 = 6 * a 2) : a 4 = 24 :=
sorry

end geometric_sequence_fourth_term_l49_49448


namespace solve_system_l49_49462

theorem solve_system :
  ∀ (x y : ℝ) (triangle : ℝ), 
  (2 * x - 3 * y = 5) ∧ (x + y = triangle) ∧ (x = 4) →
  (y = 1) ∧ (triangle = 5) :=
by
  -- Skipping the proof steps
  sorry

end solve_system_l49_49462


namespace minimize_expression_l49_49381

theorem minimize_expression (x : ℝ) : 3 * x^2 - 12 * x + 1 ≥ 3 * 2^2 - 12 * 2 + 1 :=
by sorry

end minimize_expression_l49_49381


namespace tetrahedron_altitude_exsphere_eq_l49_49403

variable (h₁ h₂ h₃ h₄ r₁ r₂ r₃ r₄ : ℝ)

/-- The equality of the sum of the reciprocals of the heights and the radii of the exspheres of 
a tetrahedron -/
theorem tetrahedron_altitude_exsphere_eq :
  2 * (1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄) = (1 / r₁ + 1 / r₂ + 1 / r₃ + 1 / r₄) :=
sorry

end tetrahedron_altitude_exsphere_eq_l49_49403


namespace average_book_width_l49_49390

noncomputable def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]
def number_of_books : ℕ := 7
def total_sum_of_widths : ℚ := 34.5

theorem average_book_width :
  ((book_widths.sum) / number_of_books) = 241/49 :=
by
  sorry

end average_book_width_l49_49390


namespace sqrt_of_9_fact_over_84_eq_24_sqrt_15_l49_49359

theorem sqrt_of_9_fact_over_84_eq_24_sqrt_15 :
  Real.sqrt (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (2^2 * 3 * 7)) = 24 * Real.sqrt 15 :=
by
  sorry

end sqrt_of_9_fact_over_84_eq_24_sqrt_15_l49_49359


namespace smallest_arithmetic_mean_divisible_by_1111_l49_49103

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l49_49103


namespace geometric_sum_eight_terms_l49_49598

noncomputable def geometric_series_sum_8 (a r : ℝ) : ℝ :=
  a * (1 - r^8) / (1 - r)

theorem geometric_sum_eight_terms
  (a r : ℝ) (h_geom_pos : r > 0)
  (h_sum_two : a + a * r = 2)
  (h_sum_eight : a * r^2 + a * r^3 = 8) :
  geometric_series_sum_8 a r = 170 := 
sorry

end geometric_sum_eight_terms_l49_49598


namespace solve_for_x_l49_49201

theorem solve_for_x (x : ℕ) : (3 : ℝ)^(27^x) = (27 : ℝ)^(3^x) → x = 0 :=
by
  sorry

end solve_for_x_l49_49201


namespace problem1_solution_problem2_solution_l49_49684

-- Proof for Problem 1
theorem problem1_solution (x y : ℝ) 
(h1 : x - y - 1 = 4)
(h2 : 4 * (x - y) - y = 5) : 
x = 20 ∧ y = 15 := sorry

-- Proof for Problem 2
theorem problem2_solution (x : ℝ) 
(h1 : 4 * x - 1 ≥ x + 1)
(h2 : (1 - x) / 2 < x) : 
x ≥ 2 / 3 := sorry

end problem1_solution_problem2_solution_l49_49684


namespace f_neg_l49_49420

/-- Define f(x) as an odd function --/
def f : ℝ → ℝ := sorry

/-- The property of odd functions: f(-x) = -f(x) --/
axiom odd_fn_property (x : ℝ) : f (-x) = -f x

/-- Define the function for non-negative x --/
axiom f_nonneg (x : ℝ) (hx : 0 ≤ x) : f x = x + 1

/-- The goal is to determine f(x) when x < 0 --/
theorem f_neg (x : ℝ) (h : x < 0) : f x = x - 1 :=
by
  sorry

end f_neg_l49_49420


namespace inequality_proof_l49_49389

theorem inequality_proof 
  (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
  sorry

end inequality_proof_l49_49389


namespace largest_int_lt_100_div_9_rem_5_l49_49859

theorem largest_int_lt_100_div_9_rem_5 :
  ∃ a, a < 100 ∧ (a % 9 = 5) ∧ ∀ b, b < 100 ∧ (b % 9 = 5) → b ≤ 95 := by
sorry

end largest_int_lt_100_div_9_rem_5_l49_49859


namespace shortest_distance_D_to_V_l49_49449

-- Define distances
def distance_A_to_G : ℕ := 12
def distance_G_to_B : ℕ := 10
def distance_A_to_B : ℕ := 8
def distance_D_to_G : ℕ := 15
def distance_V_to_G : ℕ := 17

-- Prove the shortest distance from Dasha to Vasya
theorem shortest_distance_D_to_V : 
  let dD_to_V := distance_D_to_G + distance_V_to_G
  let dAlt := dD_to_V + distance_A_to_B - distance_A_to_G - distance_G_to_B
  (dAlt < dD_to_V) -> dAlt = 18 :=
by
  sorry

end shortest_distance_D_to_V_l49_49449


namespace intersecting_lines_c_plus_d_l49_49030

theorem intersecting_lines_c_plus_d (c d : ℝ) 
  (h1 : ∀ y, ∃ x, x = (1/3) * y + c) 
  (h2 : ∀ x, ∃ y, y = (1/3) * x + d)
  (P : (3:ℝ) = (1 / 3) * (3:ℝ) + c) 
  (Q : (3:ℝ) = (1 / 3) * (3:ℝ) + d) : 
  c + d = 4 := 
by
  sorry

end intersecting_lines_c_plus_d_l49_49030


namespace evaluate_b3_l49_49318

variable (b1 q : ℤ)
variable (b1_cond : b1 = 5 ∨ b1 = -5)
variable (q_cond : q = 3 ∨ q = -3)
def b3 : ℤ := b1 * q^2

theorem evaluate_b3 (h : b1^2 * (1 + q^2 + q^4) = 2275) : b3 = 45 ∨ b3 = -45 :=
by sorry

end evaluate_b3_l49_49318


namespace points_on_opposite_sides_l49_49545

-- Definitions and the conditions written to Lean
def satisfies_A (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 2 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

def satisfies_B (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 8 * a^2 * x - 2 * a^3 * y + 12 * a * y + a^4 + 36 = 0

def opposite_sides_of_line (y_A y_B : ℝ) : Prop :=
  (y_A - 1) * (y_B - 1) < 0

theorem points_on_opposite_sides (a : ℝ) (x_A y_A x_B y_B : ℝ) :
  satisfies_A a x_A y_A →
  satisfies_B a x_B y_B →
  -2 > a ∨ (-1 < a ∧ a < 0) ∨ 3 < a →
  opposite_sides_of_line y_A y_B → 
  x_A = 2 * a ∧ y_A = -a ∧ x_B = 4 ∧ y_B = a - 6/a :=
sorry

end points_on_opposite_sides_l49_49545


namespace hash_hash_hash_72_eq_12_5_l49_49997

def hash (N : ℝ) : ℝ := 0.5 * N + 2

theorem hash_hash_hash_72_eq_12_5 : hash (hash (hash 72)) = 12.5 := 
by
  sorry

end hash_hash_hash_72_eq_12_5_l49_49997


namespace find_k_values_l49_49956

theorem find_k_values (k : ℚ) 
  (h1 : ∀ k, ∃ m, m = (3 * k + 9) / (7 - k))
  (h2 : ∀ k, m = 2 * k) : 
  (k = 9 / 2 ∨ k = 1) :=
by
  sorry

end find_k_values_l49_49956


namespace polygon_area_is_correct_l49_49752

def points : List (ℕ × ℕ) := [
  (0, 0), (10, 0), (20, 0), (30, 10),
  (0, 20), (10, 20), (20, 30), (10, 30),
  (0, 30), (20, 10), (30, 20), (10, 10)
]

def polygon_area (ps : List (ℕ × ℕ)) : ℕ := sorry

theorem polygon_area_is_correct :
  polygon_area points = 9 := sorry

end polygon_area_is_correct_l49_49752


namespace count_perfect_squares_lt_10_pow_9_multiple_36_l49_49425

theorem count_perfect_squares_lt_10_pow_9_multiple_36 : 
  ∃ N : ℕ, ∀ n < 31622, (n % 6 = 0 → n^2 < 10^9 ∧ 36 ∣ n^2 → n ≤ 31620 → N = 5270) :=
by
  sorry

end count_perfect_squares_lt_10_pow_9_multiple_36_l49_49425


namespace count_divisible_by_35_l49_49521

theorem count_divisible_by_35 : 
  ∃! (n : ℕ), n = 13 ∧ (∀ (ab : ℕ), 10 ≤ ab ∧ ab ≤ 99 ∧ (∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ ab = 10 * a + b) →
    (ab * 100 + 35) % 35 = 0 ↔ ab % 7 = 0) :=
by {
  sorry
}

end count_divisible_by_35_l49_49521


namespace taxi_range_l49_49197

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 
    5
  else if x <= 10 then
    5 + (x - 3) * 2
  else
    5 + 7 * 2 + (x - 10) * 3

theorem taxi_range (x : ℝ) (h : fare x + 1 = 38) : 15 < x ∧ x ≤ 16 := 
  sorry

end taxi_range_l49_49197


namespace problem_f_2005_value_l49_49877

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_2005_value (h_even : ∀ x : ℝ, f (-x) = f x)
                            (h_periodic : ∀ x : ℝ, f (x + 8) = f x + f 4)
                            (h_initial : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = 4 - x) :
  f 2005 = 0 :=
sorry

end problem_f_2005_value_l49_49877


namespace total_pieces_l49_49079

def gum_packages : ℕ := 28
def candy_packages : ℕ := 14
def pieces_per_package : ℕ := 6

theorem total_pieces : (gum_packages * pieces_per_package) + (candy_packages * pieces_per_package) = 252 :=
by
  sorry

end total_pieces_l49_49079


namespace ratio_of_first_term_to_common_difference_l49_49848

theorem ratio_of_first_term_to_common_difference
  (a d : ℝ)
  (h : (8 / 2 * (2 * a + 7 * d)) = 3 * (5 / 2 * (2 * a + 4 * d))) :
  a / d = 2 / 7 :=
by
  sorry

end ratio_of_first_term_to_common_difference_l49_49848


namespace find_p_l49_49946

/-- Given conditions about the coordinates of points on a line, we want to prove p = 3. -/
theorem find_p (m n p : ℝ) 
  (h1 : m = n / 3 - 2 / 5)
  (h2 : m + p = (n + 9) / 3 - 2 / 5) 
  : p = 3 := by 
  sorry

end find_p_l49_49946


namespace verify_p_q_l49_49525

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-5, 2]]

def p : ℤ := 5
def q : ℤ := -26

theorem verify_p_q :
  N * N = p • N + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  -- Skipping the proof
  sorry

end verify_p_q_l49_49525


namespace batsman_average_after_17th_innings_l49_49919

theorem batsman_average_after_17th_innings :
  ∀ (A : ℕ), (80 + 16 * A) = 17 * (A + 2) → A + 2 = 48 := by
  intro A h
  sorry

end batsman_average_after_17th_innings_l49_49919


namespace area_of_smaller_circle_l49_49583

noncomputable def radius_smaller_circle : ℝ := sorry
noncomputable def radius_larger_circle : ℝ := 3 * radius_smaller_circle

-- Given: PA = AB = 5
def PA : ℝ := 5
def AB : ℝ := 5

-- Final goal: The area of the smaller circle is 5/3 * π
theorem area_of_smaller_circle (r_s : ℝ) (rsq : r_s^2 = 5 / 3) : (π * r_s^2 = 5/3 * π) :=
by
  exact sorry

end area_of_smaller_circle_l49_49583


namespace smallest_positive_period_of_sin_2x_l49_49195

noncomputable def period_of_sine (B : ℝ) : ℝ := (2 * Real.pi) / B

theorem smallest_positive_period_of_sin_2x :
  period_of_sine 2 = Real.pi := sorry

end smallest_positive_period_of_sin_2x_l49_49195


namespace taqeesha_grade_l49_49085

theorem taqeesha_grade (s : ℕ → ℕ) (h1 : (s 16) = 77) (h2 : (s 17) = 78) : s 17 - s 16 = 94 :=
by
  -- Add definitions and sorry to skip the proof
  sorry

end taqeesha_grade_l49_49085


namespace equation_two_roots_iff_l49_49303

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end equation_two_roots_iff_l49_49303


namespace no_parallelogram_on_convex_graph_l49_49958

-- Definition of strictly convex function
def is_strictly_convex (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x t y : ℝ⦄, (x < t ∧ t < y) → f t < ((f y - f x) / (y - x)) * (t - x) + f x

-- The main statement of the problem
theorem no_parallelogram_on_convex_graph (f : ℝ → ℝ) :
  is_strictly_convex f →
  ¬ ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (f b < (f c - f a) / (c - a) * (b - a) + f a) ∧
    (f c < (f d - f b) / (d - b) * (c - b) + f b) :=
sorry

end no_parallelogram_on_convex_graph_l49_49958


namespace anayet_speed_is_61_l49_49319

-- Define the problem conditions
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_time : ℝ := 2
def total_distance : ℝ := 369
def remaining_distance : ℝ := 121

-- Calculate derived values
def amoli_distance : ℝ := amoli_speed * amoli_time
def covered_distance : ℝ := total_distance - remaining_distance
def anayet_distance : ℝ := covered_distance - amoli_distance

-- Define the theorem to prove Anayet's speed
theorem anayet_speed_is_61 : anayet_distance / anayet_time = 61 :=
by
  -- sorry is a placeholder for the proof
  sorry

end anayet_speed_is_61_l49_49319


namespace number_of_shirts_made_today_l49_49517

-- Define the rate of shirts made per minute.
def shirts_per_minute : ℕ := 6

-- Define the number of minutes the machine worked today.
def minutes_today : ℕ := 12

-- Define the total number of shirts made today.
def shirts_made_today : ℕ := shirts_per_minute * minutes_today

-- State the theorem for the number of shirts made today.
theorem number_of_shirts_made_today : shirts_made_today = 72 := 
by
  -- Proof is omitted
  sorry

end number_of_shirts_made_today_l49_49517


namespace stapler_machines_l49_49220

theorem stapler_machines (x : ℝ) :
  (∃ (x : ℝ), x > 0) ∧
  ((∀ r1 r2 : ℝ, (r1 = 800 / 6) → (r2 = 800 / x) → (r1 + r2 = 800 / 3)) ↔
    (1 / 6 + 1 / x = 1 / 3)) :=
by sorry

end stapler_machines_l49_49220


namespace percentage_markup_l49_49245

theorem percentage_markup (SP CP : ℕ) (h1 : SP = 8340) (h2 : CP = 6672) :
  ((SP - CP) / CP * 100) = 25 :=
by
  -- Before proving, we state our assumptions
  sorry

end percentage_markup_l49_49245


namespace unique_n_divisors_satisfies_condition_l49_49879

theorem unique_n_divisors_satisfies_condition:
  ∃ (n : ℕ), (∃ d1 d2 d3 : ℕ, d1 = 1 ∧ d2 > d1 ∧ d3 > d2 ∧ n = d3 ∧
  n = d2^2 + d3^3) ∧ n = 68 := by
  sorry

end unique_n_divisors_satisfies_condition_l49_49879


namespace min_dot_product_l49_49333

noncomputable def ellipse_eq_p (x y : ℝ) : Prop :=
    x^2 / 9 + y^2 / 8 = 1

noncomputable def dot_product_op_fp (x y : ℝ) : ℝ :=
    x^2 + x + y^2

theorem min_dot_product : 
    (∀ x y : ℝ, ellipse_eq_p x y → dot_product_op_fp x y = 6) := 
sorry

end min_dot_product_l49_49333


namespace proof_math_problem_l49_49376

noncomputable def math_problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧ 
  ω^4 = 1 ∧ ω ≠ 1 ∧ 
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2)

theorem proof_math_problem (a b c d : ℝ) (ω : ℂ) (h: math_problem a b c d ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
sorry

end proof_math_problem_l49_49376


namespace cube_probability_l49_49230

def prob_same_color_vertical_faces : ℕ := sorry

theorem cube_probability :
  prob_same_color_vertical_faces = 1 / 27 := 
sorry

end cube_probability_l49_49230


namespace ratio_Umar_Yusaf_l49_49921

variable (AliAge YusafAge UmarAge : ℕ)

-- Given conditions:
def Ali_is_8_years_old : Prop := AliAge = 8
def Ali_is_3_years_older_than_Yusaf : Prop := AliAge = YusafAge + 3
def Umar_is_10_years_old : Prop := UmarAge = 10

-- Proof statement:
theorem ratio_Umar_Yusaf (h1 : Ali_is_8_years_old AliAge)
                         (h2 : Ali_is_3_years_older_than_Yusaf AliAge YusafAge)
                         (h3 : Umar_is_10_years_old UmarAge) :
  UmarAge / YusafAge = 2 :=
by
  sorry

end ratio_Umar_Yusaf_l49_49921


namespace unique_triple_solution_l49_49799

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end unique_triple_solution_l49_49799


namespace loss_percentage_l49_49064

/-
Books Problem:
Determine the loss percentage on the first book given:
1. The cost of the first book (C1) is Rs. 280.
2. The total cost of two books is Rs. 480.
3. The second book is sold at a gain of 19%.
4. Both books are sold at the same price.
-/

theorem loss_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 = 280)
  (h2 : C1 + C2 = 480) (h3 : SP2 = C2 * 1.19) (h4 : SP1 = SP2) : 
  (C1 - SP1) / C1 * 100 = 15 := 
by
  sorry

end loss_percentage_l49_49064


namespace simplify_expr1_simplify_expr2_l49_49021

theorem simplify_expr1 : 
  (1:ℝ) * (-3:ℝ) ^ 0 + (- (1/2:ℝ)) ^ (-2:ℝ) - (-3:ℝ) ^ (-1:ℝ) = 16 / 3 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 
  ((-2 * x^3) ^ 2 * (-x^2)) / ((-x)^2) ^ 3 = -4 * x^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l49_49021


namespace least_common_multiple_first_ten_l49_49635

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l49_49635


namespace ef_length_l49_49457

theorem ef_length (FR RG : ℝ) (cos_ERH : ℝ) (h1 : FR = 12) (h2 : RG = 6) (h3 : cos_ERH = 1 / 5) : EF = 30 :=
by
  sorry

end ef_length_l49_49457


namespace student_selection_l49_49283

theorem student_selection (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 4) : a + b + c = 12 :=
by {
  sorry
}

end student_selection_l49_49283


namespace min_max_transformation_a_min_max_transformation_b_l49_49941

theorem min_max_transformation_a {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≥ a) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≤ b) :=
sorry

theorem min_max_transformation_b {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≥ -b) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≤ -a) :=
sorry

end min_max_transformation_a_min_max_transformation_b_l49_49941


namespace minimal_time_for_horses_l49_49139

/-- Define the individual periods of the horses' runs -/
def periods : List ℕ := [2, 3, 4, 5, 6, 7, 9, 10]

/-- Define a function to calculate the LCM of a list of numbers -/
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

/-- Conjecture: proving that 60 is the minimal time until at least 6 out of 8 horses meet at the starting point -/
theorem minimal_time_for_horses : lcm_list [2, 3, 4, 5, 6, 10] = 60 :=
by
  sorry

end minimal_time_for_horses_l49_49139


namespace problem_solution_l49_49760

theorem problem_solution
  (p q : ℝ)
  (h₁ : p ≠ q)
  (h₂ : (x : ℝ) → (x - 5) * (x + 3) = 24 * x - 72 → x = p ∨ x = q)
  (h₃ : p > q) :
  p - q = 20 :=
sorry

end problem_solution_l49_49760


namespace bed_length_l49_49105

noncomputable def volume (length width height : ℝ) : ℝ :=
  length * width * height

theorem bed_length
  (width height : ℝ)
  (bags_of_soil soil_volume_per_bag total_volume : ℝ)
  (needed_bags : ℝ)
  (L : ℝ) :
  width = 4 →
  height = 1 →
  needed_bags = 16 →
  soil_volume_per_bag = 4 →
  total_volume = needed_bags * soil_volume_per_bag →
  total_volume = 2 * volume L width height →
  L = 8 :=
by
  intros
  sorry

end bed_length_l49_49105


namespace roof_collapse_days_l49_49801

def leaves_per_pound : ℕ := 1000
def pounds_limit_of_roof : ℕ := 500
def leaves_per_day : ℕ := 100

theorem roof_collapse_days : (pounds_limit_of_roof * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

end roof_collapse_days_l49_49801


namespace probability_of_drawing_letter_in_name_l49_49078

theorem probability_of_drawing_letter_in_name :
  let total_letters := 26
  let alonso_letters := ['a', 'l', 'o', 'n', 's']
  let number_of_alonso_letters := alonso_letters.length
  number_of_alonso_letters / total_letters = 5 / 26 :=
by
  sorry

end probability_of_drawing_letter_in_name_l49_49078


namespace problem1_solution_problem2_solution_l49_49353

-- Statement for Problem 1
theorem problem1_solution (x : ℝ) : (1 / 2 * (x - 3) ^ 2 = 18) ↔ (x = 9 ∨ x = -3) :=
by sorry

-- Statement for Problem 2
theorem problem2_solution (x : ℝ) : (x ^ 2 + 6 * x = 5) ↔ (x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end problem1_solution_problem2_solution_l49_49353


namespace jesse_total_carpet_l49_49815

theorem jesse_total_carpet : 
  let length_rect := 12
  let width_rect := 8
  let base_tri := 10
  let height_tri := 6
  let area_rect := length_rect * width_rect
  let area_tri := (base_tri * height_tri) / 2
  area_rect + area_tri = 126 :=
by
  sorry

end jesse_total_carpet_l49_49815


namespace triangle_length_product_square_l49_49370

theorem triangle_length_product_square 
  (a1 : ℝ) (b1 : ℝ) (c1 : ℝ) (a2 : ℝ) (b2 : ℝ) (c2 : ℝ) 
  (h1 : a1 * b1 / 2 = 3)
  (h2 : a2 * b2 / 2 = 4)
  (h3 : a1 = a2)
  (h4 : c1 = 2 * c2) 
  (h5 : c1^2 = a1^2 + b1^2)
  (h6 : c2^2 = a2^2 + b2^2) :
  (b1 * b2)^2 = (2304 / 25 : ℝ) :=
by
  sorry

end triangle_length_product_square_l49_49370


namespace Cl_invalid_electrons_l49_49322

noncomputable def Cl_mass_number : ℕ := 35
noncomputable def Cl_protons : ℕ := 17
noncomputable def Cl_neutrons : ℕ := Cl_mass_number - Cl_protons
noncomputable def Cl_electrons : ℕ := Cl_protons

theorem Cl_invalid_electrons : Cl_electrons ≠ 18 :=
by
  sorry

end Cl_invalid_electrons_l49_49322


namespace problem_l49_49868

variable (a b : ℝ)

theorem problem (h : a = 1.25 * b) : (4 * b) / a = 3.2 :=
by
  sorry

end problem_l49_49868


namespace area_of_triangle_PQR_l49_49443

-- Define the problem conditions
def PQ : ℝ := 4
def PR : ℝ := 4
def angle_P : ℝ := 45 -- degrees

-- Define the main problem
theorem area_of_triangle_PQR : 
  (PQ = PR) ∧ (angle_P = 45) ∧ (PR = 4) → 
  ∃ A, A = 8 := 
by
  sorry

end area_of_triangle_PQR_l49_49443


namespace could_not_be_diagonal_lengths_l49_49284

-- Definitions of the diagonal conditions
def diagonal_condition (s : List ℕ) : Prop :=
  match s with
  | [x, y, z] => x^2 + y^2 > z^2 ∧ x^2 + z^2 > y^2 ∧ y^2 + z^2 > x^2
  | _ => false

-- Statement of the problem
theorem could_not_be_diagonal_lengths : 
  ¬ diagonal_condition [5, 6, 8] :=
by 
  sorry

end could_not_be_diagonal_lengths_l49_49284


namespace total_votes_4500_l49_49180

theorem total_votes_4500 (V : ℝ) 
  (h : 0.60 * V - 0.40 * V = 900) : V = 4500 :=
by
  sorry

end total_votes_4500_l49_49180


namespace tiling_possible_values_of_n_l49_49383

-- Define the sizes of the grid and the tiles
def grid_size : ℕ × ℕ := (9, 7)
def l_tile_size : ℕ := 3  -- L-shaped tile composed of three unit squares
def square_tile_size : ℕ := 4  -- square tile composed of four unit squares

-- Formalize the properties of the grid and the constraints for the tiling
def total_squares : ℕ := grid_size.1 * grid_size.2
def white_squares (n : ℕ) : ℕ := 3 * n
def black_squares (n : ℕ) : ℕ := n
def total_black_squares : ℕ := 20
def total_white_squares : ℕ := total_squares - total_black_squares

-- The main theorem statement
theorem tiling_possible_values_of_n (n : ℕ) : 
  (n = 2 ∨ n = 5 ∨ n = 8 ∨ n = 11 ∨ n = 14 ∨ n = 17 ∨ n = 20) ↔
  (3 * (total_white_squares - 2 * (20 - n)) / 3 + n = 23 ∧ n + (total_black_squares - n) = 20) :=
sorry

end tiling_possible_values_of_n_l49_49383


namespace find_x_when_y_equals_two_l49_49854

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l49_49854


namespace roots_of_polynomial_l49_49631

-- Define the polynomial
def poly := fun (x : ℝ) => x^3 - 7 * x^2 + 14 * x - 8

-- Define the statement
theorem roots_of_polynomial : (poly 1 = 0) ∧ (poly 2 = 0) ∧ (poly 4 = 0) :=
  by
  sorry

end roots_of_polynomial_l49_49631


namespace intended_profit_l49_49611

variables (C P : ℝ)

theorem intended_profit (L S : ℝ) (h1 : L = C * (1 + P)) (h2 : S = 0.90 * L) (h3 : S = 1.17 * C) :
  P = 0.3 + 1 / 3 :=
by
  sorry

end intended_profit_l49_49611


namespace b_must_be_one_l49_49865

theorem b_must_be_one (a b : ℝ) (h1 : a + b - a * b = 1) (h2 : ∀ n : ℤ, a ≠ n) : b = 1 :=
sorry

end b_must_be_one_l49_49865


namespace range_of_m_l49_49300

def cond1 (x : ℝ) : Prop := x^2 - 4 * x + 3 < 0
def cond2 (x : ℝ) : Prop := x^2 - 6 * x + 8 < 0
def cond3 (x m : ℝ) : Prop := 2 * x^2 - 9 * x + m < 0

theorem range_of_m (m : ℝ) : (∀ x, cond1 x → cond2 x → cond3 x m) → m < 9 :=
by
  sorry

end range_of_m_l49_49300


namespace fixed_point_f_l49_49467

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log (2 * x + 1) / Real.log a) + 2

theorem fixed_point_f (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : f a 0 = 2 :=
by
  sorry

end fixed_point_f_l49_49467


namespace classify_numbers_l49_49902

def isDecimal (n : ℝ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), n = i + f ∧ i ≠ 0

def isNatural (n : ℕ) : Prop :=
  n ≥ 0

theorem classify_numbers :
  (isDecimal 7.42) ∧ (isDecimal 3.6) ∧ (isDecimal 5.23) ∧ (isDecimal 37.8) ∧
  (isNatural 5) ∧ (isNatural 100) ∧ (isNatural 502) ∧ (isNatural 460) :=
by
  sorry

end classify_numbers_l49_49902


namespace log_inequality_l49_49714

theorem log_inequality (a b c : ℝ) (h1 : b^2 - a * c < 0) :
  ∀ x y : ℝ, a * (Real.log x)^2 + 2 * b * (Real.log x) * (Real.log y) + c * (Real.log y)^2 = 1 
  → a * 1^2 + 2 * b * 1 * (-1) + c * (-1)^2 = 1 → 
  -1 / Real.sqrt (a * c - b^2) ≤ Real.log (x * y) ∧ Real.log (x * y) ≤ 1 / Real.sqrt (a * c - b^2) :=
by
  sorry

end log_inequality_l49_49714


namespace proof_problem_l49_49904

noncomputable def arithmetic_mean (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem proof_problem (a b c x y z m : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (m_pos : 0 < m) (m_ne_one : m ≠ 1) 
  (h_b : b = arithmetic_mean a c) (h_y : y = geometric_mean x z) :
  (b - c) * Real.logb m x + (c - a) * Real.logb m y + (a - b) * Real.logb m z = 0 := by
  sorry

end proof_problem_l49_49904


namespace Eric_white_marbles_l49_49617

theorem Eric_white_marbles (total_marbles blue_marbles green_marbles : ℕ) (h1 : total_marbles = 20) (h2 : blue_marbles = 6) (h3 : green_marbles = 2) : 
  total_marbles - (blue_marbles + green_marbles) = 12 := by
  sorry

end Eric_white_marbles_l49_49617


namespace manager_salary_is_3600_l49_49578

noncomputable def manager_salary (M : ℕ) : ℕ :=
  let total_salary_20 := 20 * 1500
  let new_average_salary := 1600
  let total_salary_21 := 21 * new_average_salary
  total_salary_21 - total_salary_20

theorem manager_salary_is_3600 : manager_salary 3600 = 3600 := by
  sorry

end manager_salary_is_3600_l49_49578


namespace find_fourth_number_l49_49705

theorem find_fourth_number (x : ℝ) (h : 3 + 33 + 333 + x = 369.63) : x = 0.63 :=
sorry

end find_fourth_number_l49_49705


namespace distinct_solutions_subtraction_eq_two_l49_49221

theorem distinct_solutions_subtraction_eq_two :
  ∃ p q : ℝ, (p ≠ q) ∧ (p > q) ∧ ((6 * p - 18) / (p^2 + 4 * p - 21) = p + 3) ∧ ((6 * q - 18) / (q^2 + 4 * q - 21) = q + 3) ∧ (p - q = 2) :=
by
  have p := -3
  have q := -5
  exists p, q
  sorry

end distinct_solutions_subtraction_eq_two_l49_49221


namespace subsets_union_l49_49131

theorem subsets_union (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) 
  (A : Fin m → Finset (Fin n)) (hA : ∀ i j, i ≠ j → A i ≠ A j) 
  (hB : ∀ i, A i ≠ ∅) : 
  ∃ i j k, i ≠ j ∧ A i ∪ A j = A k := 
sorry

end subsets_union_l49_49131


namespace lcm_48_180_l49_49745

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l49_49745


namespace Nina_has_16dollars65_l49_49633

-- Definitions based on given conditions
variables (W M : ℝ)

-- Condition 1: Nina has exactly enough money to purchase 5 widgets
def condition1 : Prop := 5 * W = M

-- Condition 2: If the cost of each widget were reduced by $1.25, Nina would have exactly enough money to purchase 8 widgets
def condition2 : Prop := 8 * (W - 1.25) = M

-- Statement: Proving the amount of money Nina has is $16.65
theorem Nina_has_16dollars65 (h1 : condition1 W M) (h2 : condition2 W M) : M = 16.65 :=
sorry

end Nina_has_16dollars65_l49_49633


namespace revenue_from_full_price_tickets_l49_49542

theorem revenue_from_full_price_tickets (f h p : ℕ) (H1 : f + h = 150) (H2 : f * p + h * (p / 2) = 2450) : 
  f * p = 1150 :=
by 
  sorry

end revenue_from_full_price_tickets_l49_49542


namespace ellipse_minor_axis_length_l49_49412

theorem ellipse_minor_axis_length
  (semi_focal_distance : ℝ)
  (eccentricity : ℝ)
  (semi_focal_distance_eq : semi_focal_distance = 2)
  (eccentricity_eq : eccentricity = 2 / 3) :
  ∃ minor_axis_length : ℝ, minor_axis_length = 2 * Real.sqrt 5 :=
by
  sorry

end ellipse_minor_axis_length_l49_49412


namespace at_least_one_not_less_than_two_l49_49276

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) :=
sorry

end at_least_one_not_less_than_two_l49_49276


namespace carrie_spent_money_l49_49878

variable (cost_per_tshirt : ℝ) (num_tshirts : ℕ)

theorem carrie_spent_money (h1 : cost_per_tshirt = 9.95) (h2 : num_tshirts = 20) :
  cost_per_tshirt * num_tshirts = 199 := by
  sorry

end carrie_spent_money_l49_49878


namespace range_of_m_tangent_not_parallel_l49_49562

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := (1 / 2) * x^2 - k * x
noncomputable def h (x : ℝ) (m : ℝ) : ℝ := f x + g x (m + (1 / m))
noncomputable def M (x : ℝ) (m : ℝ) : ℝ := f x - g x (m + (1 / m))

theorem range_of_m (m : ℝ) (h_extreme : ∃ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, h y m ≤ h x m) : 
  (0 < m ∧ m ≤ 1 / 2) ∨ (m ≥ 2) :=
  sorry

theorem tangent_not_parallel (x1 x2 x0 : ℝ) (m : ℝ) (h_zeros : M x1 m = 0 ∧ M x2 m = 0 ∧ x1 > x2 ∧ 2 * x0 = x1 + x2) :
  ¬ (∃ l : ℝ, ∀ x : ℝ, M x m = l * (x - x0) + M x0 m ∧ l = 0) :=
  sorry

end range_of_m_tangent_not_parallel_l49_49562


namespace which_two_students_donated_l49_49198

theorem which_two_students_donated (A B C D : Prop) 
  (h1 : A ∨ D) 
  (h2 : ¬(A ∧ D)) 
  (h3 : (A ∧ B) ∨ (A ∧ D) ∨ (B ∧ D))
  (h4 : ¬(A ∧ B ∧ D)) 
  : B ∧ D :=
sorry

end which_two_students_donated_l49_49198


namespace simplify_expression_l49_49135

theorem simplify_expression (a b : ℤ) : 
  (18 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 40 * b) = 21 * a + 41 * b := 
by
  sorry

end simplify_expression_l49_49135


namespace prime_a_b_l49_49804

theorem prime_a_b (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^11 + b = 2089) : 49 * b - a = 2007 :=
sorry

end prime_a_b_l49_49804


namespace p_and_q_work_together_l49_49109

-- Given conditions
variable (Wp Wq : ℝ)

-- Condition that p is 50% more efficient than q
def efficiency_relation : Prop := Wp = 1.5 * Wq

-- Condition that p can complete the work in 25 days
def work_completion_by_p : Prop := Wp = 1 / 25

-- To be proved that p and q working together can complete the work in 15 days
theorem p_and_q_work_together (h1 : efficiency_relation Wp Wq)
                              (h2 : work_completion_by_p Wp) :
                              1 / (Wp + (Wp / 1.5)) = 15 :=
by
  sorry

end p_and_q_work_together_l49_49109


namespace plot_length_l49_49075

def breadth : ℝ := 40 -- Derived from conditions and cost equation solution
def length : ℝ := breadth + 20
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300

theorem plot_length :
  (2 * (breadth + (breadth + 20))) * cost_per_meter = total_cost → length = 60 :=
by {
  sorry
}

end plot_length_l49_49075


namespace max_gold_coins_l49_49940

theorem max_gold_coins (k : ℕ) (n : ℕ) (h : n = 13 * k + 3 ∧ n < 150) : n = 146 :=
by 
  sorry

end max_gold_coins_l49_49940


namespace line_points_sum_slope_and_intercept_l49_49599

-- Definition of the problem
theorem line_points_sum_slope_and_intercept (a b : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 3) ∨ (x = 10 ∧ y = 19) → y = a * x + b) →
  a + b = 1 :=
by
  intro h
  sorry

end line_points_sum_slope_and_intercept_l49_49599


namespace ordered_pairs_count_l49_49534

theorem ordered_pairs_count :
  (∃ (a b : ℝ), (∃ (x y : ℤ),
    a * (x : ℝ) + b * (y : ℝ) = 1 ∧
    (x : ℝ)^2 + (y : ℝ)^2 = 65)) →
  ∃ (n : ℕ), n = 128 :=
by
  sorry

end ordered_pairs_count_l49_49534


namespace goose_price_remains_affordable_l49_49593

theorem goose_price_remains_affordable :
  ∀ (h v : ℝ),
  h + v = 1 →
  h + (v / 2) = 1 →
  h * 1.2 ≤ 1 :=
by
  intros h v h_eq v_eq
  /- Proof will go here -/
  sorry

end goose_price_remains_affordable_l49_49593


namespace cakes_remaining_l49_49106

theorem cakes_remaining (initial_cakes : ℕ) (bought_cakes : ℕ) (h1 : initial_cakes = 169) (h2 : bought_cakes = 137) : initial_cakes - bought_cakes = 32 :=
by
  sorry

end cakes_remaining_l49_49106


namespace travel_time_home_to_community_center_l49_49502

-- Definitions and assumptions based on the conditions
def time_to_library := 30 -- in minutes
def distance_to_library := 5 -- in miles
def time_spent_at_library := 15 -- in minutes
def distance_to_community_center := 3 -- in miles
noncomputable def cycling_speed := time_to_library / distance_to_library -- in minutes per mile

-- Time calculation to reach the community center from the library
noncomputable def time_from_library_to_community_center := distance_to_community_center * cycling_speed -- in minutes

-- Total time spent to travel from home to the community center
noncomputable def total_time_home_to_community_center :=
  time_to_library + time_spent_at_library + time_from_library_to_community_center

-- The proof statement verifying the total time
theorem travel_time_home_to_community_center : total_time_home_to_community_center = 63 := by
  sorry

end travel_time_home_to_community_center_l49_49502


namespace s_plough_time_l49_49687

theorem s_plough_time (r_s_combined_time : ℝ) (r_time : ℝ) (t_time : ℝ) (s_time : ℝ) :
  r_s_combined_time = 10 → r_time = 15 → t_time = 20 → s_time = 30 :=
by
  sorry

end s_plough_time_l49_49687


namespace minimum_value_of_sum_of_squares_l49_49329

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 4 * x + 3 * y + 12 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 169 :=
by
  sorry

end minimum_value_of_sum_of_squares_l49_49329


namespace moving_circle_fixed_point_l49_49791

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end moving_circle_fixed_point_l49_49791


namespace find_y_l49_49998

def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y : ∃ y : ℕ, F 3 y 5 15 = 490 ∧ y = 6 := by
  sorry

end find_y_l49_49998


namespace gena_hits_target_l49_49492

-- Definitions from the problem conditions
def initial_shots : ℕ := 5
def total_shots : ℕ := 17
def shots_per_hit : ℕ := 2

-- Mathematical equivalent proof statement
theorem gena_hits_target (G : ℕ) (H : G * shots_per_hit + initial_shots = total_shots) : G = 6 :=
by
  sorry

end gena_hits_target_l49_49492


namespace find_t_l49_49554

theorem find_t (p q r s t : ℤ)
  (h₁ : p - q - r + s - t = -t)
  (h₂ : p - (q - (r - (s - t))) = -4 + t) :
  t = 2 := 
sorry

end find_t_l49_49554


namespace simplify_and_sum_of_exponents_l49_49955

-- Define the given expression
def radicand (x y z : ℝ) : ℝ := 40 * x ^ 5 * y ^ 7 * z ^ 9

-- Define what cube root stands for
noncomputable def cbrt (a : ℝ) := a ^ (1 / 3 : ℝ)

-- Define the simplified expression outside the cube root
noncomputable def simplified_outside_exponents (x y z : ℝ) : ℝ := x * y * z ^ 3

-- Define the sum of the exponents outside the radical
def sum_of_exponents_outside (x y z : ℝ) : ℝ := (1 + 1 + 3 : ℝ)

-- Statement of the problem in Lean
theorem simplify_and_sum_of_exponents (x y z : ℝ) :
  sum_of_exponents_outside x y z = 5 :=
by 
  sorry

end simplify_and_sum_of_exponents_l49_49955


namespace grasshopper_max_reach_points_l49_49664

theorem grasshopper_max_reach_points
  (α : ℝ) (α_eq : α = 36 * Real.pi / 180)
  (L : ℕ)
  (jump_constant : ∀ (n : ℕ), L = L) :
  ∃ (N : ℕ), N ≤ 10 :=
by 
  sorry

end grasshopper_max_reach_points_l49_49664


namespace volume_of_prism_l49_49393

theorem volume_of_prism (l w h : ℝ) (hlw : l * w = 10) (hwh : w * h = 15) (hlh : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 :=
by
  sorry

end volume_of_prism_l49_49393


namespace determine_x_l49_49000

theorem determine_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
by
  sorry

end determine_x_l49_49000


namespace ab_cd_zero_l49_49883

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : ac + bd = 0) : 
  ab + cd = 0 := 
sorry

end ab_cd_zero_l49_49883


namespace solution_set_of_inequality_l49_49288

variables {R : Type*} [LinearOrderedField R]

-- Define f as an even function
def even_function (f : R → R) := ∀ x : R, f x = f (-x)

-- Define f as an increasing function on [0, +∞)
def increasing_on_nonneg (f : R → R) := ∀ ⦃x y : R⦄, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the hypothesis and the theorem
theorem solution_set_of_inequality (f : R → R)
  (h_even : even_function f)
  (h_inc : increasing_on_nonneg f) :
  { x : R | f x > f 1 } = { x : R | x > 1 ∨ x < -1 } :=
by {
  sorry
}

end solution_set_of_inequality_l49_49288


namespace no_equilateral_triangle_on_integer_lattice_l49_49579

theorem no_equilateral_triangle_on_integer_lattice :
  ∀ (A B C : ℤ × ℤ), 
  A ≠ B → B ≠ C → C ≠ A →
  (dist A B = dist B C ∧ dist B C = dist C A) → 
  false :=
by sorry

end no_equilateral_triangle_on_integer_lattice_l49_49579


namespace smallest_four_digit_number_divisible_by_35_l49_49892

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def ends_with_0_or_5 (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

def divisibility_rule_for_7 (n : ℕ) : Prop := is_divisible_by (n / 10 - 2 * (n % 10)) 7

def smallest_four_digit_number := 1000

theorem smallest_four_digit_number_divisible_by_35 : ∃ n : ℕ, 
  n ≥ smallest_four_digit_number ∧ 
  ends_with_0_or_5 n ∧ 
  divisibility_rule_for_7 n ∧ 
  is_divisible_by n 35 ∧ 
  n = 1015 := 
by
  unfold smallest_four_digit_number ends_with_0_or_5 divisibility_rule_for_7 is_divisible_by
  sorry

end smallest_four_digit_number_divisible_by_35_l49_49892


namespace amy_money_left_l49_49950

def amount_left (initial_amount doll_price board_game_price comic_book_price doll_qty board_game_qty comic_book_qty board_game_discount sales_tax_rate : ℝ) :
    ℝ :=
  let cost_dolls := doll_qty * doll_price
  let cost_board_games := board_game_qty * board_game_price
  let cost_comic_books := comic_book_qty * comic_book_price
  let discounted_cost_board_games := cost_board_games * (1 - board_game_discount)
  let total_cost_before_tax := cost_dolls + discounted_cost_board_games + cost_comic_books
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  initial_amount - total_cost_after_tax

theorem amy_money_left :
  amount_left 100 1.25 12.75 3.50 3 2 4 0.10 0.08 = 56.04 :=
by
  sorry

end amy_money_left_l49_49950


namespace symmetric_point_origin_l49_49450

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l49_49450


namespace problem_2003_divisibility_l49_49229

theorem problem_2003_divisibility :
  let N := (List.range' 1 1001).prod + (List.range' 1002 1001).prod
  N % 2003 = 0 := by
  sorry

end problem_2003_divisibility_l49_49229


namespace cubic_roots_inequalities_l49_49108

theorem cubic_roots_inequalities 
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ z : ℂ, (a * z^3 + b * z^2 + c * z + d = 0) → z.re < 0) :
  a * b > 0 ∧ b * c - a * d > 0 ∧ a * d > 0 :=
by
  sorry

end cubic_roots_inequalities_l49_49108


namespace min_value_expression_l49_49292

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  ∃ A : ℝ, A = 3 * Real.sqrt 2 ∧ 
  (A = (Real.sqrt (a^6 + b^4 * c^6) / b) + 
       (Real.sqrt (b^6 + c^4 * a^6) / c) + 
       (Real.sqrt (c^6 + a^4 * b^6) / a)) :=
sorry

end min_value_expression_l49_49292


namespace garden_width_l49_49789

theorem garden_width (L W : ℕ) 
  (area_playground : 192 = 16 * 12)
  (area_garden : 192 = L * W)
  (perimeter_garden : 64 = 2 * L + 2 * W) :
  W = 12 :=
by
  sorry

end garden_width_l49_49789


namespace parabola_directrix_l49_49641

theorem parabola_directrix (x : ℝ) :
  (y = (x^2 - 8 * x + 12) / 16) →
  (∃ y, y = -17/4) :=
by
  intro h
  sorry

end parabola_directrix_l49_49641


namespace problem1_problem2_problem3_problem4_l49_49089

-- Problem (1)
theorem problem1 : (-8 - 6 + 24) = 10 :=
by sorry

-- Problem (2)
theorem problem2 : (-48 / 6 + -21 * (-1 / 3)) = -1 :=
by sorry

-- Problem (3)
theorem problem3 : ((1 / 8 - 1 / 3 + 1 / 4) * -24) = -1 :=
by sorry

-- Problem (4)
theorem problem4 : (-1^4 - (1 + 0.5) * (1 / 3) * (1 - (-2)^2)) = 0.5 :=
by sorry

end problem1_problem2_problem3_problem4_l49_49089


namespace quadratic_real_roots_and_value_l49_49964

theorem quadratic_real_roots_and_value (m x1 x2: ℝ) 
  (h1: ∀ (a: ℝ), ∃ (b c: ℝ), a = x^2 - 4 * x - 2 * m + 5) 
  (h2: x1 * x2 + x1 + x2 = m^2 + 6):
  m ≥ 1/2 ∧ m = 1 := 
by
  sorry

end quadratic_real_roots_and_value_l49_49964


namespace water_fraction_final_l49_49538

noncomputable def initial_water_volume : ℚ := 25
noncomputable def first_removal_water : ℚ := 5
noncomputable def first_add_antifreeze : ℚ := 5
noncomputable def first_water_fraction : ℚ := (initial_water_volume - first_removal_water) / initial_water_volume

noncomputable def second_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def second_water_fraction : ℚ := (initial_water_volume - first_removal_water - second_removal_fraction * (initial_water_volume - first_removal_water)) / initial_water_volume

noncomputable def third_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def third_water_fraction := (second_water_fraction * (initial_water_volume - 5) + 2) / initial_water_volume

theorem water_fraction_final :
  third_water_fraction = 14.8 / 25 := sorry

end water_fraction_final_l49_49538


namespace max_value_at_log2_one_l49_49514

noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * (4 : ℝ) ^ x
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

theorem max_value_at_log2_one :
  (∃ x, domain x ∧ f x = 0) ∧ (∀ y, domain y → f y ≤ 0) :=
by
  sorry

end max_value_at_log2_one_l49_49514


namespace solve_inequality_l49_49034

-- Declare the necessary conditions as variables in Lean
variables (a c : ℝ)

-- State the Lean theorem
theorem solve_inequality :
  (∀ x : ℝ, (ax^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) →
  a < 0 →
  a = -6 ∧ c = -1 :=
  sorry

end solve_inequality_l49_49034


namespace find_c_in_parabola_l49_49222

theorem find_c_in_parabola (b c : ℝ) (h₁ : 2 = (-1) ^ 2 + b * (-1) + c) (h₂ : 2 = 3 ^ 2 + b * 3 + c) : c = -1 :=
sorry

end find_c_in_parabola_l49_49222


namespace triangle_evaluation_l49_49945

def triangle (a b : ℤ) : ℤ := a^2 - 2 * b

theorem triangle_evaluation : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end triangle_evaluation_l49_49945


namespace max_x_squared_plus_y_squared_l49_49983

theorem max_x_squared_plus_y_squared (x y : ℝ) 
  (h : 3 * x^2 + 2 * y^2 = 2 * x) : x^2 + y^2 ≤ 4 / 9 :=
sorry

end max_x_squared_plus_y_squared_l49_49983


namespace money_received_from_mom_l49_49119

-- Define the given conditions
def initial_amount : ℕ := 48
def amount_spent : ℕ := 11
def amount_after_getting_money : ℕ := 58
def amount_left_after_spending : ℕ := initial_amount - amount_spent

-- Define the proof statement
theorem money_received_from_mom : (amount_after_getting_money - amount_left_after_spending) = 21 :=
by
  -- placeholder for the proof
  sorry

end money_received_from_mom_l49_49119


namespace find_C_l49_49842

theorem find_C
  (A B C D : ℕ)
  (h1 : 0 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 4 * 1000 + A * 100 + 5 * 10 + B + (C * 1000 + 2 * 100 + D * 10 + 7) = 8070) :
  C = 3 :=
by
  sorry

end find_C_l49_49842


namespace calculate_expression_l49_49749

theorem calculate_expression :
  let a := 2^4
  let b := 2^2
  let c := 2^3
  (a^2 / b^3) * c^3 = 2048 :=
by
  sorry -- Proof is omitted as per instructions

end calculate_expression_l49_49749


namespace averageSpeed_is_45_l49_49044

/-- Define the upstream and downstream speeds of the fish --/
def fishA_upstream_speed := 40
def fishA_downstream_speed := 60
def fishB_upstream_speed := 30
def fishB_downstream_speed := 50
def fishC_upstream_speed := 45
def fishC_downstream_speed := 65
def fishD_upstream_speed := 35
def fishD_downstream_speed := 55
def fishE_upstream_speed := 25
def fishE_downstream_speed := 45

/-- Define a function to calculate the speed in still water --/
def stillWaterSpeed (upstream_speed : ℕ) (downstream_speed : ℕ) : ℕ :=
  (upstream_speed + downstream_speed) / 2

/-- Calculate the still water speed for each fish --/
def fishA_speed := stillWaterSpeed fishA_upstream_speed fishA_downstream_speed
def fishB_speed := stillWaterSpeed fishB_upstream_speed fishB_downstream_speed
def fishC_speed := stillWaterSpeed fishC_upstream_speed fishC_downstream_speed
def fishD_speed := stillWaterSpeed fishD_upstream_speed fishD_downstream_speed
def fishE_speed := stillWaterSpeed fishE_upstream_speed fishE_downstream_speed

/-- Calculate the average speed of all fish in still water --/
def averageSpeedInStillWater :=
  (fishA_speed + fishB_speed + fishC_speed + fishD_speed + fishE_speed) / 5

/-- The statement to prove --/
theorem averageSpeed_is_45 : averageSpeedInStillWater = 45 :=
  sorry

end averageSpeed_is_45_l49_49044


namespace triangle_height_l49_49718

theorem triangle_height (base height area : ℝ) 
(h_base : base = 3) (h_area : area = 9) 
(h_area_eq : area = (base * height) / 2) :
  height = 6 := 
by 
  sorry

end triangle_height_l49_49718


namespace scientific_notation_of_number_l49_49140

theorem scientific_notation_of_number :
  ∀ (n : ℕ), n = 450000000 -> n = 45 * 10^7 := 
by
  sorry

end scientific_notation_of_number_l49_49140


namespace solve_system_solve_equation_l49_49602

-- 1. System of Equations
theorem solve_system :
  ∀ (x y : ℝ), (x + 2 * y = 9) ∧ (3 * x - 2 * y = 3) → (x = 3) ∧ (y = 3) :=
by sorry

-- 2. Single Equation
theorem solve_equation :
  ∀ (x : ℝ), (2 - x) / (x - 3) + 3 = 2 / (3 - x) → x = 5 / 2 :=
by sorry

end solve_system_solve_equation_l49_49602


namespace find_V_l49_49585

theorem find_V 
  (c : ℝ)
  (R₁ V₁ W₁ R₂ W₂ V₂ : ℝ)
  (h1 : R₁ = c * (V₁ / W₁))
  (h2 : R₁ = 6)
  (h3 : V₁ = 2)
  (h4 : W₁ = 3)
  (h5 : R₂ = 25)
  (h6 : W₂ = 5)
  (h7 : V₂ = R₂ * W₂ / 9) :
  V₂ = 125 / 9 :=
by sorry

end find_V_l49_49585


namespace polynomial_solution_l49_49080

noncomputable def P : ℝ → ℝ := sorry

theorem polynomial_solution (x : ℝ) :
  (∃ P : ℝ → ℝ, (∀ x, P x = (P 0) + (P 1) * x + (P 2) * x^2) ∧ 
  (P (-2) = 4)) →
  (P x = (4 * x^2 - 6 * x) / 7) :=
by
  sorry

end polynomial_solution_l49_49080


namespace boat_trip_l49_49642

variable {v v_T : ℝ}

theorem boat_trip (d_total t_total : ℝ) (h1 : d_total = 10) (h2 : t_total = 5) (h3 : 2 / (v - v_T) = 3 / (v + v_T)) :
  v_T = 5 / 12 ∧ (5 / (v - v_T)) = 3 ∧ (5 / (v + v_T)) = 2 :=
by
  have h4 : 1 / (d_total / t_total) = v - v_T := sorry
  have h5 : 1 / (d_total / t_total) = v + v_T := sorry
  have h6 : v = 5 * v_T := sorry
  have h7 : v_T = 5 / 12 := sorry
  have t_upstream : 5 / (v - v_T) = 3 := sorry
  have t_downstream : 5 / (v + v_T) = 2 := sorry
  exact ⟨h7, t_upstream, t_downstream⟩

end boat_trip_l49_49642


namespace weekly_milk_consumption_l49_49204

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end weekly_milk_consumption_l49_49204


namespace parabola_directrix_l49_49784

noncomputable def directrix_value (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_directrix (a : ℝ) (h : directrix_value a = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l49_49784


namespace determine_original_price_l49_49251

namespace PriceProblem

variable (x : ℝ)

def final_price (x : ℝ) : ℝ := 0.98175 * x

theorem determine_original_price (h : final_price x = 100) : x = 101.86 :=
by
  sorry

end PriceProblem

end determine_original_price_l49_49251


namespace monkey_slip_distance_l49_49656

theorem monkey_slip_distance
  (height : ℕ)
  (climb_per_hour : ℕ)
  (hours : ℕ)
  (s : ℕ)
  (total_hours : ℕ)
  (final_climb : ℕ)
  (reach_top : height = hours * (climb_per_hour - s) + final_climb)
  (total_hours_constraint : total_hours = 17)
  (climb_per_hour_constraint : climb_per_hour = 3)
  (height_constraint : height = 19)
  (final_climb_constraint : final_climb = 3)
  (hours_constraint : hours = 16) :
  s = 2 := sorry

end monkey_slip_distance_l49_49656


namespace negation_of_p_l49_49358

theorem negation_of_p :
  (∃ x : ℝ, x < 0 ∧ x + (1 / x) > -2) ↔ ¬ (∀ x : ℝ, x < 0 → x + (1 / x) ≤ -2) :=
by {
  sorry
}

end negation_of_p_l49_49358


namespace length_of_bridge_correct_l49_49182

noncomputable def length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60  -- Convert speed from km/hr to m/min
  speed_mpm * time_min  -- Length of the bridge in meters

theorem length_of_bridge_correct :
  length_of_bridge 10 10 = 1666.7 :=
by
  sorry

end length_of_bridge_correct_l49_49182


namespace cells_after_3_hours_l49_49949

noncomputable def cell_division_problem (t : ℕ) : ℕ :=
  2 ^ (t * 2)

theorem cells_after_3_hours : cell_division_problem 3 = 64 := by
  sorry

end cells_after_3_hours_l49_49949


namespace num_solutions_in_interval_l49_49536

theorem num_solutions_in_interval : 
  ∃ n : ℕ, n = 2 ∧ ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  2 ^ Real.cos θ = Real.sin θ → n = 2 := 
sorry

end num_solutions_in_interval_l49_49536


namespace line_tangent_to_ellipse_l49_49774

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 → ∃! y : ℝ, 3 * x^2 + 6 * y^2 = 6) →
  m^2 = 3 / 2 :=
by
  sorry

end line_tangent_to_ellipse_l49_49774


namespace product_of_solutions_l49_49560

-- Definitions based on given conditions
def equation (x : ℝ) : Prop := |x| = 3 * (|x| - 2)

-- Statement of the proof problem
theorem product_of_solutions : ∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 * x2 = -9 := by
  sorry

end product_of_solutions_l49_49560


namespace roots_difference_one_l49_49896

theorem roots_difference_one (p : ℝ) :
  (∃ (x y : ℝ), (x^3 - 7 * x + p = 0) ∧ (y^3 - 7 * y + p = 0) ∧ (x - y = 1)) ↔ (p = 6 ∨ p = -6) :=
sorry

end roots_difference_one_l49_49896


namespace julia_average_speed_l49_49098

theorem julia_average_speed :
  let distance1 := 45
  let speed1 := 15
  let distance2 := 15
  let speed2 := 45
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 18 := by
sorry

end julia_average_speed_l49_49098


namespace ticket_distribution_count_l49_49881

theorem ticket_distribution_count :
  let A := 2
  let B := 2
  let C := 1
  let D := 1
  let total_tickets := A + B + C + D
  ∃ (num_dist : ℕ), num_dist = 180 :=
by {
  sorry
}

end ticket_distribution_count_l49_49881


namespace final_ratio_l49_49710

-- Define initial conditions
def initial_milk_ratio : ℕ := 1
def initial_water_ratio : ℕ := 5
def total_parts : ℕ := initial_milk_ratio + initial_water_ratio
def can_capacity : ℕ := 8
def additional_milk : ℕ := 2
def initial_volume : ℕ := can_capacity - additional_milk
def part_volume : ℕ := initial_volume / total_parts

-- Define initial quantities
def initial_milk_quantity : ℕ := part_volume * initial_milk_ratio
def initial_water_quantity : ℕ := part_volume * initial_water_ratio

-- Define final quantities
def final_milk_quantity : ℕ := initial_milk_quantity + additional_milk
def final_water_quantity : ℕ := initial_water_quantity

-- Hypothesis: final ratios of milk and water
def final_ratio_of_milk_to_water : ℕ × ℕ := (final_milk_quantity, final_water_quantity)

-- Final ratio should be 3:5
theorem final_ratio (h : final_ratio_of_milk_to_water = (3, 5)) : final_ratio_of_milk_to_water = (3, 5) :=
  by
  sorry

end final_ratio_l49_49710


namespace find_p_l49_49647

/-- Given the points Q(0, 15), A(3, 15), B(15, 0), O(0, 0), and C(0, p).
The area of triangle ABC is given as 45.
We need to prove that p = 11.25. -/
theorem find_p (ABC_area : ℝ) (p : ℝ) (h : ABC_area = 45) :
  p = 11.25 :=
by
  sorry

end find_p_l49_49647


namespace Q_root_l49_49228

def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_root : Q (3^(1 / 3 : ℝ) + 2) = 0 := sorry

end Q_root_l49_49228


namespace area_of_field_l49_49439

theorem area_of_field (b l : ℝ) (h1 : l = b + 30) (h2 : 2 * (l + b) = 540) : l * b = 18000 := 
by
  sorry

end area_of_field_l49_49439


namespace complement_intersection_l49_49543

-- Defining the universal set U and subsets A and B
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 3, 4}
def B : Finset ℕ := {3, 4, 5}

-- Proving the complement of the intersection of A and B in U
theorem complement_intersection : (U \ (A ∩ B)) = {1, 2, 5} :=
by sorry

end complement_intersection_l49_49543


namespace central_angle_measures_l49_49299

-- Definitions for the conditions
def perimeter_eq (r l : ℝ) : Prop := l + 2 * r = 6
def area_eq (r l : ℝ) : Prop := (1 / 2) * l * r = 2
def central_angle (r l α : ℝ) : Prop := α = l / r

-- The final proof statement
theorem central_angle_measures (r l α : ℝ) (h1 : perimeter_eq r l) (h2 : area_eq r l) :
  central_angle r l α → (α = 1 ∨ α = 4) :=
sorry

end central_angle_measures_l49_49299


namespace original_number_proof_l49_49302

-- Define the conditions
variables (x y : ℕ)
-- Given conditions
def condition1 : Prop := y = 13
def condition2 : Prop := 7 * x + 5 * y = 146

-- Goal: the original number (sum of the parts x and y)
def original_number : ℕ := x + y

-- State the problem as a theorem
theorem original_number_proof (x y : ℕ) (h1 : condition1 y) (h2 : condition2 x y) : original_number x y = 24 := by
  -- The proof will be written here
  sorry

end original_number_proof_l49_49302


namespace find_slope_and_intercept_l49_49040

noncomputable def line_equation_to_slope_intercept_form 
  (x y : ℝ) : Prop :=
  (3 * (x - 2) - 4 * (y + 3) = 0) ↔ (y = (3 / 4) * x - 4.5)

theorem find_slope_and_intercept : 
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), (line_equation_to_slope_intercept_form x y) → m = 3/4 ∧ b = -4.5) :=
sorry

end find_slope_and_intercept_l49_49040


namespace trebled_result_of_original_number_is_72_l49_49747

theorem trebled_result_of_original_number_is_72:
  ∀ (x : ℕ), x = 9 → 3 * (2 * x + 6) = 72 :=
by
  intro x h
  sorry

end trebled_result_of_original_number_is_72_l49_49747


namespace time_bob_cleans_room_l49_49242

variable (timeAlice : ℕ) (fractionBob : ℚ)

-- Definitions based on conditions from the problem
def timeAliceCleaningRoom : ℕ := 40
def fractionOfTimeBob : ℚ := 3 / 8

-- Prove the time it takes Bob to clean his room
theorem time_bob_cleans_room : (timeAliceCleaningRoom * fractionOfTimeBob : ℚ) = 15 := 
by
  sorry

end time_bob_cleans_room_l49_49242


namespace smallest_four_digit_divisible_by_35_l49_49843

theorem smallest_four_digit_divisible_by_35 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 35 = 0 → n ≤ m ∧ n = 1006 :=
by
  sorry

end smallest_four_digit_divisible_by_35_l49_49843


namespace xiao_hong_home_to_school_distance_l49_49698

-- Definition of conditions
def distance_from_drop_to_school := 1000 -- in meters
def time_from_home_to_school_walking := 22.5 -- in minutes
def time_from_home_to_school_biking := 40 -- in minutes
def walking_speed := 80 -- in meters per minute
def bike_speed_slowdown := 800 -- in meters per minute

-- The main theorem statement
theorem xiao_hong_home_to_school_distance :
  ∃ d : ℝ, d = 12000 ∧ 
            distance_from_drop_to_school = 1000 ∧
            time_from_home_to_school_walking = 22.5 ∧
            time_from_home_to_school_biking = 40 ∧
            walking_speed = 80 ∧
            bike_speed_slowdown = 800 := 
sorry

end xiao_hong_home_to_school_distance_l49_49698


namespace difference_between_Annette_and_Sara_l49_49006

-- Define the weights of the individuals
variables (A C S B E : ℝ)

-- Conditions given in the problem
def condition1 := A + C = 95
def condition2 := C + S = 87
def condition3 := A + S = 97
def condition4 := C + B = 100
def condition5 := A + C + B = 155
def condition6 := A + S + B + E = 240
def condition7 := E = 1.25 * C

-- The theorem that we want to prove
theorem difference_between_Annette_and_Sara (A C S B E : ℝ)
  (h1 : condition1 A C)
  (h2 : condition2 C S)
  (h3 : condition3 A S)
  (h4 : condition4 C B)
  (h5 : condition5 A C B)
  (h6 : condition6 A S B E)
  (h7 : condition7 C E) :
  A - S = 8 :=
by {
  sorry
}

end difference_between_Annette_and_Sara_l49_49006


namespace pink_highlighters_count_l49_49262

-- Definitions for the problem's conditions
def total_highlighters : Nat := 11
def yellow_highlighters : Nat := 2
def blue_highlighters : Nat := 5
def non_pink_highlighters : Nat := yellow_highlighters + blue_highlighters

-- Statement of the problem as a theorem
theorem pink_highlighters_count : total_highlighters - non_pink_highlighters = 4 :=
by
  sorry

end pink_highlighters_count_l49_49262


namespace symmetry_with_respect_to_line_x_eq_1_l49_49100

theorem symmetry_with_respect_to_line_x_eq_1 (f : ℝ → ℝ) :
  ∀ x, f (x - 1) = f (1 - x) ↔ x - 1 = 1 - x :=
by
  sorry

end symmetry_with_respect_to_line_x_eq_1_l49_49100


namespace xyz_value_l49_49794

theorem xyz_value (x y z : ℕ) (h1 : x + 2 * y = z) (h2 : x^2 - 4 * y^2 + z^2 = 310) :
  xyz = 4030 ∨ xyz = 23870 :=
by
  -- placeholder for proof steps
  sorry

end xyz_value_l49_49794


namespace percent_problem_l49_49156

theorem percent_problem
  (X : ℝ)
  (h1 : 0.28 * 400 = 112)
  (h2 : 0.45 * X + 112 = 224.5) :
  X = 250 := 
sorry

end percent_problem_l49_49156


namespace smallest_integer_divisible_20_perfect_cube_square_l49_49936

theorem smallest_integer_divisible_20_perfect_cube_square :
  ∃ (n : ℕ), n > 0 ∧ n % 20 = 0 ∧ (∃ (m : ℕ), n^2 = m^3) ∧ (∃ (k : ℕ), n^3 = k^2) ∧ n = 1000000 :=
by {
  sorry -- Replace this placeholder with an appropriate proof.
}

end smallest_integer_divisible_20_perfect_cube_square_l49_49936


namespace triangle_sides_ratio_l49_49025

theorem triangle_sides_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a)
  (ha_pos : a > 0) : b / a = Real.sqrt 2 :=
sorry

end triangle_sides_ratio_l49_49025


namespace original_fraction_l49_49929

def fraction (a b c : ℕ) := 10 * a + b / 10 * c + a

theorem original_fraction (a b c : ℕ) (ha: a < 10) (hb : b < 10) (hc : c < 10) (h : b ≠ c):
  (fraction a b c = b / c) →
  (fraction 6 4 1 = 64 / 16) ∨ (fraction 9 8 4 = 98 / 49) ∨
  (fraction 9 5 1 = 95 / 19) ∨ (fraction 6 5 2 = 65 / 26) :=
sorry

end original_fraction_l49_49929


namespace polar_to_rectangular_l49_49191

open Real

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 8) (h_θ : θ = π / 4) :
    (r * cos θ, r * sin θ) = (4 * sqrt 2, 4 * sqrt 2) :=
by
  rw [h_r, h_θ]
  rw [cos_pi_div_four, sin_pi_div_four]
  norm_num
  field_simp [sqrt_eq_rpow]
  sorry

end polar_to_rectangular_l49_49191


namespace minimum_value_l49_49088

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) : 
  (1 / m + 2 / n) ≥ 4 :=
sorry

end minimum_value_l49_49088


namespace fifty_third_card_is_A_s_l49_49605

def sequence_position (n : ℕ) : String :=
  let cycle_length := 26
  let pos_in_cycle := (n - 1) % cycle_length + 1
  if pos_in_cycle <= 13 then
    "A_s"
  else
    "A_h"

theorem fifty_third_card_is_A_s : sequence_position 53 = "A_s" := by
  sorry  -- proof placeholder

end fifty_third_card_is_A_s_l49_49605


namespace distance_between_pulley_centers_l49_49163

theorem distance_between_pulley_centers (R1 R2 CD : ℝ) (R1_pos : R1 = 10) (R2_pos : R2 = 6) (CD_pos : CD = 30) :
  ∃ AB : ℝ, AB = 2 * Real.sqrt 229 :=
by
  sorry

end distance_between_pulley_centers_l49_49163


namespace percentage_increase_l49_49091

theorem percentage_increase (a : ℕ) (x : ℝ) (b : ℝ) (r : ℝ) 
    (h1 : a = 1500) 
    (h2 : r = 0.6) 
    (h3 : b = 1080) 
    (h4 : a * (1 + x / 100) * r = b) : 
    x = 20 := 
by 
  sorry

end percentage_increase_l49_49091


namespace no_solution_l49_49563

theorem no_solution (n : ℕ) (x y k : ℕ) (h1 : n ≥ 1) (h2 : x > 0) (h3 : y > 0) (h4 : k > 1) (h5 : Nat.gcd x y = 1) (h6 : 3^n = x^k + y^k) : False :=
by
  sorry

end no_solution_l49_49563


namespace max_surface_area_of_rectangular_solid_on_sphere_l49_49853

noncomputable def max_surface_area_rectangular_solid (a b c : ℝ) :=
  2 * a * b + 2 * a * c + 2 * b * c

theorem max_surface_area_of_rectangular_solid_on_sphere :
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 36 → max_surface_area_rectangular_solid a b c ≤ 72) :=
by
  intros a b c h
  sorry

end max_surface_area_of_rectangular_solid_on_sphere_l49_49853


namespace simplify_fraction_l49_49286

-- Define what it means for a fraction to be in simplest form
def coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Define what it means for a fraction to be reducible
def reducible_fraction (num den : ℕ) : Prop := ∃ d > 1, d ∣ num ∧ d ∣ den

-- Main theorem statement
theorem simplify_fraction 
  (m n : ℕ) (h_coprime : coprime m n) 
  (h_reducible : reducible_fraction (4 * m + 3 * n) (5 * m + 2 * n)) : ∃ d, d = 7 :=
by {
  sorry
}

end simplify_fraction_l49_49286


namespace sum_of_first_100_positive_odd_integers_is_correct_l49_49826

def sum_first_100_positive_odd_integers : ℕ :=
  10000

theorem sum_of_first_100_positive_odd_integers_is_correct :
  sum_first_100_positive_odd_integers = 10000 :=
by
  sorry

end sum_of_first_100_positive_odd_integers_is_correct_l49_49826


namespace pet_store_customers_buy_different_pets_l49_49338

theorem pet_store_customers_buy_different_pets :
  let puppies := 20
  let kittens := 10
  let hamsters := 12
  let rabbits := 5
  let customers := 4
  (puppies * kittens * hamsters * rabbits * Nat.factorial customers = 288000) := 
by
  sorry

end pet_store_customers_buy_different_pets_l49_49338


namespace positive_int_solution_is_perfect_square_l49_49387

variable (t n : ℤ)

theorem positive_int_solution_is_perfect_square (ht : ∃ n : ℕ, n > 0 ∧ n^2 + (4 * t - 1) * n + 4 * t^2 = 0) : ∃ k : ℕ, n = k^2 :=
  sorry

end positive_int_solution_is_perfect_square_l49_49387


namespace gear_B_turns_l49_49552

theorem gear_B_turns (teeth_A teeth_B turns_A: ℕ) (h₁: teeth_A = 6) (h₂: teeth_B = 8) (h₃: turns_A = 12) :
(turn_A * teeth_A) / teeth_B = 9 :=
by  sorry

end gear_B_turns_l49_49552


namespace quadratic_root_conditions_l49_49748

theorem quadratic_root_conditions (a b : ℝ)
    (h1 : ∃ k : ℝ, ∀ x : ℝ, x^2 + 2 * x + 3 - k = 0)
    (h2 : ∀ α β : ℝ, α * β = 3 - k ∧ k^2 = α * β + 3 * k) : 
    k = 3 := 
sorry

end quadratic_root_conditions_l49_49748


namespace factors_of_180_l49_49505

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l49_49505


namespace distance_between_stations_l49_49989

theorem distance_between_stations :
  ∀ (x t : ℕ), 
    (20 * t = x) ∧ 
    (25 * t = x + 70) →
    (2 * x + 70 = 630) :=
by
  sorry

end distance_between_stations_l49_49989


namespace range_of_a_for_solution_set_l49_49067

theorem range_of_a_for_solution_set (a : ℝ) :
  ((∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1)) :=
sorry

end range_of_a_for_solution_set_l49_49067


namespace Bryce_raisins_l49_49352

theorem Bryce_raisins (B C : ℚ) (h1 : B = C + 10) (h2 : C = B / 4) : B = 40 / 3 :=
by
 -- The proof goes here, but we skip it for now
 sorry

end Bryce_raisins_l49_49352


namespace roller_coaster_cost_l49_49090

variable (ferris_wheel_rides : Nat) (log_ride_rides : Nat) (rc_rides : Nat)
variable (ferris_wheel_cost : Nat) (log_ride_cost : Nat)
variable (initial_tickets : Nat) (additional_tickets : Nat)
variable (total_needed_tickets : Nat)

theorem roller_coaster_cost :
  ferris_wheel_rides = 2 →
  log_ride_rides = 7 →
  rc_rides = 3 →
  ferris_wheel_cost = 2 →
  log_ride_cost = 1 →
  initial_tickets = 20 →
  additional_tickets = 6 →
  total_needed_tickets = initial_tickets + additional_tickets →
  let total_ride_costs := ferris_wheel_rides * ferris_wheel_cost + log_ride_rides * log_ride_cost
  let rc_cost := (total_needed_tickets - total_ride_costs) / rc_rides
  rc_cost = 5 := by
  sorry

end roller_coaster_cost_l49_49090


namespace quadrilateral_perimeter_l49_49556

-- Define the basic conditions
variables (a b : ℝ)

-- Let's define what happens when Xiao Ming selected 2 pieces of type A, 7 pieces of type B, and 3 pieces of type C
theorem quadrilateral_perimeter (a b : ℝ) : 2 * (a + 3 * b + 2 * a + b) = 6 * a + 8 * b :=
by sorry

end quadrilateral_perimeter_l49_49556


namespace cost_of_traveling_all_roads_l49_49809

noncomputable def total_cost_of_roads (length width road_width : ℝ) (cost_per_sq_m : ℝ) : ℝ :=
  let area_road_parallel_length := length * road_width
  let area_road_parallel_breadth := width * road_width
  let diagonal_length := Real.sqrt (length^2 + width^2)
  let area_road_diagonal := diagonal_length * road_width
  let total_area := area_road_parallel_length + area_road_parallel_breadth + area_road_diagonal
  total_area * cost_per_sq_m

theorem cost_of_traveling_all_roads :
  total_cost_of_roads 80 50 10 3 = 6730.2 :=
by
  sorry

end cost_of_traveling_all_roads_l49_49809


namespace arithmetic_sequence_a5_l49_49310

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_a5 (a₁ d : ℝ) (h1 : a 2 a₁ d = 2 * a 3 a₁ d + 1) (h2 : a 4 a₁ d = 2 * a 3 a₁ d + 7) :
  a 5 a₁ d = 2 :=
by
  sorry

end arithmetic_sequence_a5_l49_49310


namespace distance_from_pointM_to_xaxis_l49_49136

-- Define the point M with coordinates (2, -3)
def pointM : ℝ × ℝ := (2, -3)

-- Define the function to compute the distance from a point to the x-axis.
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

-- Formalize the proof statement.
theorem distance_from_pointM_to_xaxis : distanceToXAxis pointM = 3 := by
  -- Proof goes here
  sorry

end distance_from_pointM_to_xaxis_l49_49136


namespace owl_cost_in_gold_l49_49015

-- Definitions for conditions
def spellbook_cost_gold := 5
def potionkit_cost_silver := 20
def num_spellbooks := 5
def num_potionkits := 3
def silver_per_gold := 9
def total_payment_silver := 537

-- Function to convert gold to silver
def gold_to_silver (gold : ℕ) : ℕ := gold * silver_per_gold

-- Function to compute total cost in silver for spellbooks and potion kits
def total_spellbook_cost_silver : ℕ :=
  gold_to_silver spellbook_cost_gold * num_spellbooks

def total_potionkit_cost_silver : ℕ :=
  potionkit_cost_silver * num_potionkits

-- Function to calculate the cost of the owl in silver
def owl_cost_silver : ℕ :=
  total_payment_silver - (total_spellbook_cost_silver + total_potionkit_cost_silver)

-- Function to convert the owl's cost from silver to gold
def owl_cost_gold : ℕ :=
  owl_cost_silver / silver_per_gold

-- The proof statement
theorem owl_cost_in_gold : owl_cost_gold = 28 :=
  by
    sorry

end owl_cost_in_gold_l49_49015


namespace length_of_field_l49_49082

variable (w l : ℝ)
variable (H1 : l = 2 * w)
variable (pond_area : ℝ := 64)
variable (field_area : ℝ := l * w)
variable (H2 : pond_area = (1 / 98) * field_area)

theorem length_of_field : l = 112 :=
by
  sorry

end length_of_field_l49_49082


namespace students_neither_l49_49277

-- Define the conditions
def total_students : ℕ := 60
def students_math : ℕ := 40
def students_physics : ℕ := 35
def students_both : ℕ := 25

-- Define the problem statement
theorem students_neither : total_students - ((students_math - students_both) + (students_physics - students_both) + students_both) = 10 :=
by
  sorry

end students_neither_l49_49277


namespace andy_loss_more_likely_than_win_l49_49458

def prob_win_first := 0.30
def prob_lose_first := 0.70

def prob_win_second := 0.50
def prob_lose_second := 0.50

def prob_win_both := prob_win_first * prob_win_second
def prob_lose_both := prob_lose_first * prob_lose_second
def diff_probability := prob_lose_both - prob_win_both
def percentage_more_likely := (diff_probability / prob_win_both) * 100

theorem andy_loss_more_likely_than_win :
  percentage_more_likely = 133.33 := sorry

end andy_loss_more_likely_than_win_l49_49458


namespace centroid_of_triangle_l49_49692

-- Definitions and conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := 
  true -- Placeholder for a more specific definition if necessary

def triangle (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder for defining a triangle with vertices at integer grid points

def no_other_nodes_on_sides (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert no other integer grid points on the sides

def exactly_one_node_inside (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert exactly one integer grid point inside the triangle

def medians_intersection_is_point_O (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert \(O\) is the intersection point of the medians

-- Theorem statement
theorem centroid_of_triangle 
  (A B C O : ℤ × ℤ)
  (h1 : is_lattice_point A)
  (h2 : is_lattice_point B)
  (h3 : is_lattice_point C)
  (h4 : triangle A B C)
  (h5 : no_other_nodes_on_sides A B C)
  (h6 : exactly_one_node_inside A B C O) : 
  medians_intersection_is_point_O A B C O :=
sorry

end centroid_of_triangle_l49_49692


namespace triangle_properties_l49_49427

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (D : ℝ) : 
  (a + c) * Real.sin A = Real.sin A + Real.sin C →
  c^2 + c = b^2 - 1 →
  D = (a + c) / 2 →
  BD = Real.sqrt 3 / 2 →
  B = 2 * Real.pi / 3 ∧ (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry

end triangle_properties_l49_49427


namespace eighth_term_geometric_sequence_l49_49918

theorem eighth_term_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 12) (h_r : r = 1/4) (h_n : n = 8) :
  a * r^(n - 1) = 3 / 4096 := 
by 
  sorry

end eighth_term_geometric_sequence_l49_49918


namespace find_remainder_in_division_l49_49666

theorem find_remainder_in_division
  (D : ℕ)
  (r : ℕ) -- the remainder when using the incorrect divisor
  (R : ℕ) -- the remainder when using the correct divisor
  (h1 : D = 12 * 63 + r)
  (h2 : D = 21 * 36 + R)
  : R = 0 :=
by
  sorry

end find_remainder_in_division_l49_49666


namespace colored_pencils_more_than_erasers_l49_49007

def colored_pencils_initial := 67
def erasers_initial := 38

def colored_pencils_final := 50
def erasers_final := 28

theorem colored_pencils_more_than_erasers :
  colored_pencils_final - erasers_final = 22 := by
  sorry

end colored_pencils_more_than_erasers_l49_49007


namespace Albert_more_than_Joshua_l49_49031

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_than_Joshua_l49_49031


namespace sequence_identical_l49_49186

noncomputable def a (n : ℕ) : ℝ :=
  (1 / (2 * Real.sqrt 3)) * ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n)

theorem sequence_identical (n : ℕ) :
  a (n + 1) = (a n + a (n + 2)) / 4 :=
by
  sorry

end sequence_identical_l49_49186


namespace parabola_hyperbola_focus_l49_49364

noncomputable def focus_left (p : ℝ) : ℝ × ℝ :=
  (-p / 2, 0)

theorem parabola_hyperbola_focus (p : ℝ) (hp : p > 0) : 
  focus_left p = (-2, 0) ↔ p = 4 :=
by 
  sorry

end parabola_hyperbola_focus_l49_49364


namespace average_pages_per_day_l49_49035

variable (total_pages : ℕ := 160)
variable (pages_read : ℕ := 60)
variable (days_left : ℕ := 5)

theorem average_pages_per_day : (total_pages - pages_read) / days_left = 20 := by
  sorry

end average_pages_per_day_l49_49035


namespace common_ratio_of_gp_l49_49549

variable (r : ℝ)(n : ℕ)

theorem common_ratio_of_gp (h1 : 9 * r ^ (n - 1) = 1/3) 
                           (h2 : 9 * (1 - r ^ n) / (1 - r) = 40 / 3) : 
                           r = 1/3 := 
sorry

end common_ratio_of_gp_l49_49549


namespace arithmetic_sequence_a5_l49_49596

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ a_1 d, ∀ n, a (n + 1) = a_1 + n * d

theorem arithmetic_sequence_a5 (a : ℕ → α) (h_seq : is_arithmetic_sequence a) (h_cond : a 1 + a 7 = 12) :
  a 4 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l49_49596


namespace infinite_castle_hall_unique_l49_49776

theorem infinite_castle_hall_unique :
  (∀ (n : ℕ), ∃ hall : ℕ, ∀ m : ℕ, ((m = 2 * n + 1) ∨ (m = 3 * n + 1)) → hall = m) →
  ∀ (hall1 hall2 : ℕ), hall1 = hall2 :=
by
  sorry

end infinite_castle_hall_unique_l49_49776


namespace power_modulo_l49_49568

theorem power_modulo {a : ℤ} : a^561 ≡ a [ZMOD 561] :=
sorry

end power_modulo_l49_49568


namespace agent_takes_19_percent_l49_49141

def agentPercentage (copies_sold : ℕ) (advance_copies : ℕ) (price_per_copy : ℕ) (steve_earnings : ℕ) : ℕ :=
  let total_earnings := copies_sold * price_per_copy
  let agent_earnings := total_earnings - steve_earnings
  let percentage_agent := 100 * agent_earnings / total_earnings
  percentage_agent

theorem agent_takes_19_percent :
  agentPercentage 1000000 100000 2 1620000 = 19 :=
by 
  sorry

end agent_takes_19_percent_l49_49141


namespace max_value_x2_y2_l49_49539

noncomputable def max_x2_y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : ℝ := 2

theorem max_value_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ max_x2_y2 x y hx hy h :=
by
  sorry

end max_value_x2_y2_l49_49539


namespace solve_equation_l49_49628

theorem solve_equation (x : ℝ) :
  3 * x + 6 = abs (-20 + x^2) →
  x = (3 + Real.sqrt 113) / 2 ∨ x = (3 - Real.sqrt 113) / 2 :=
by
  sorry

end solve_equation_l49_49628


namespace john_spending_l49_49295

theorem john_spending (X : ℝ) 
  (H1 : X * (1 / 4) + X * (1 / 3) + X * (1 / 6) + 6 = X) : 
  X = 24 := 
sorry

end john_spending_l49_49295


namespace equations_have_one_contact_point_l49_49515

theorem equations_have_one_contact_point (c : ℝ):
  (∃ x : ℝ, x^2 + 1 = 4 * x + c) ∧ (∀ x1 x2 : ℝ, (x1 ≠ x2) → ¬(x1^2 + 1 = 4 * x1 + c ∧ x2^2 + 1 = 4 * x2 + c)) ↔ c = -3 :=
by
  sorry

end equations_have_one_contact_point_l49_49515


namespace tan_diff_l49_49203

theorem tan_diff (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) : Real.tan (α - β) = 1 / 7 := by
  sorry

end tan_diff_l49_49203


namespace point_M_first_quadrant_distances_length_of_segment_MN_l49_49833

-- Proof problem 1
theorem point_M_first_quadrant_distances (m : ℝ) (h1 : 2 * m + 1 > 0) (h2 : m + 3 > 0) (h3 : m + 3 = 2 * (2 * m + 1)) :
  m = 1 / 3 :=
by
  sorry

-- Proof problem 2
theorem length_of_segment_MN (m : ℝ) (h4 : m + 3 = 1) :
  let Mx := 2 * m + 1
  let My := m + 3
  let Nx := 2
  let Ny := 1
  let distMN := abs (Nx - Mx)
  distMN = 5 :=
by
  sorry

end point_M_first_quadrant_distances_length_of_segment_MN_l49_49833


namespace polynomial_value_l49_49792

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end polynomial_value_l49_49792


namespace inequality_triangle_area_l49_49913

-- Define the triangles and their properties
variables {α β γ : Real} -- Internal angles of triangle ABC
variables {r : Real} -- Circumradius of triangle ABC
variables {P Q : Real} -- Areas of triangles ABC and A'B'C' respectively

-- Define the bisectors and intersect points
-- Note: For the purpose of this proof, we're not explicitly defining the geometry
-- of the inner bisectors and intersect points but working from the given conditions.

theorem inequality_triangle_area
  (h1 : P = r^2 * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) / 2)
  (h2 : Q = r^2 * (Real.sin (β + γ) + Real.sin (γ + α) + Real.sin (α + β)) / 2) :
  16 * Q^3 ≥ 27 * r^4 * P :=
sorry

end inequality_triangle_area_l49_49913


namespace imaginary_unit_power_l49_49746

theorem imaginary_unit_power (i : ℂ) (n : ℕ) (h_i : i^2 = -1) : ∃ (n : ℕ), i^n = -1 :=
by
  use 6
  have h1 : i^4 = 1 := by sorry  -- Need to show i^4 = 1
  have h2 : i^6 = -1 := by sorry  -- Use i^4 and additional steps to show i^6 = -1
  exact h2

end imaginary_unit_power_l49_49746


namespace cube_face_problem_l49_49898

theorem cube_face_problem (n : ℕ) (h : 0 < n) :
  ((6 * n^2) : ℚ) / (6 * n^3) = 1 / 3 → n = 3 :=
by
  sorry

end cube_face_problem_l49_49898


namespace unique_integer_solution_l49_49052

theorem unique_integer_solution (n : ℤ) :
  (⌊n^2 / 4 + n⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 10) :=
by sorry

end unique_integer_solution_l49_49052


namespace total_fruit_weight_l49_49011

def melon_weight : Real := 0.35
def berries_weight : Real := 0.48
def grapes_weight : Real := 0.29
def pineapple_weight : Real := 0.56
def oranges_weight : Real := 0.17

theorem total_fruit_weight : melon_weight + berries_weight + grapes_weight + pineapple_weight + oranges_weight = 1.85 :=
by
  unfold melon_weight berries_weight grapes_weight pineapple_weight oranges_weight
  sorry

end total_fruit_weight_l49_49011


namespace project_completion_time_l49_49194

theorem project_completion_time
  (x y z : ℝ)
  (h1 : x + y = 1 / 2)
  (h2 : y + z = 1 / 4)
  (h3 : z + x = 1 / 2.4) :
  (1 / x) = 3 :=
by
  sorry

end project_completion_time_l49_49194


namespace smallest_pythagorean_sum_square_l49_49086

theorem smallest_pythagorean_sum_square (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) :
  ∃ (k : ℤ), k = 4 ∧ (p + q + r)^2 ≥ k :=
by
  sorry

end smallest_pythagorean_sum_square_l49_49086


namespace mike_coins_value_l49_49010

theorem mike_coins_value (d q : ℕ)
  (h1 : d + q = 17)
  (h2 : q + 3 = 2 * d) :
  10 * d + 25 * q = 345 :=
by
  sorry

end mike_coins_value_l49_49010


namespace num_cars_in_parking_lot_l49_49915

-- Define the conditions
variable (C : ℕ) -- Number of cars
def number_of_bikes := 5 -- Number of bikes given
def total_wheels := 66 -- Total number of wheels given
def wheels_per_bike := 2 -- Number of wheels per bike
def wheels_per_car := 4 -- Number of wheels per car

-- Define the proof statement
theorem num_cars_in_parking_lot 
  (h1 : total_wheels = 66) 
  (h2 : number_of_bikes = 5) 
  (h3 : wheels_per_bike = 2)
  (h4 : wheels_per_car = 4) 
  (h5 : C * wheels_per_car + number_of_bikes * wheels_per_bike = total_wheels) :
  C = 14 :=
by
  sorry

end num_cars_in_parking_lot_l49_49915


namespace factorize_m_cubed_minus_16m_l49_49247

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_m_cubed_minus_16m_l49_49247


namespace total_cost_is_67_15_l49_49522

noncomputable def calculate_total_cost : ℝ :=
  let caramel_cost := 3
  let candy_bar_cost := 2 * caramel_cost
  let cotton_candy_cost := (candy_bar_cost * 4) / 2
  let chocolate_bar_cost := candy_bar_cost + caramel_cost
  let lollipop_cost := candy_bar_cost / 3

  let candy_bar_total := 6 * candy_bar_cost
  let caramel_total := 3 * caramel_cost
  let cotton_candy_total := 1 * cotton_candy_cost
  let chocolate_bar_total := 2 * chocolate_bar_cost
  let lollipop_total := 2 * lollipop_cost

  let discounted_candy_bar_total := candy_bar_total * 0.9
  let discounted_caramel_total := caramel_total * 0.85
  let discounted_cotton_candy_total := cotton_candy_total * 0.8
  let discounted_chocolate_bar_total := chocolate_bar_total * 0.75
  let discounted_lollipop_total := lollipop_total -- No additional discount

  discounted_candy_bar_total +
  discounted_caramel_total +
  discounted_cotton_candy_total +
  discounted_chocolate_bar_total +
  discounted_lollipop_total

theorem total_cost_is_67_15 : calculate_total_cost = 67.15 := by
  sorry

end total_cost_is_67_15_l49_49522


namespace library_pupils_count_l49_49834

-- Definitions for the conditions provided in the problem
def num_rectangular_tables : Nat := 7
def num_pupils_per_rectangular_table : Nat := 10
def num_square_tables : Nat := 5
def num_pupils_per_square_table : Nat := 4

-- Theorem stating the problem's question and the required proof
theorem library_pupils_count :
  num_rectangular_tables * num_pupils_per_rectangular_table + 
  num_square_tables * num_pupils_per_square_table = 90 :=
sorry

end library_pupils_count_l49_49834


namespace powers_of_i_sum_l49_49369

theorem powers_of_i_sum :
  ∀ (i : ℂ), 
  (i^1 = i) ∧ (i^2 = -1) ∧ (i^3 = -i) ∧ (i^4 = 1) →
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 :=
by
  intros i h
  sorry

end powers_of_i_sum_l49_49369


namespace calories_per_cookie_l49_49498

theorem calories_per_cookie (C : ℝ) (h1 : ∀ cracker, cracker = 15)
    (h2 : ∀ cookie, cookie = C)
    (h3 : 7 * C + 10 * 15 = 500) :
    C = 50 :=
  by
    sorry

end calories_per_cookie_l49_49498


namespace sum_of_coefficients_l49_49775

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℝ) :
  (∀ x, (x^2 + 1) * (x - 2)^9 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 +
        a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7 + a8 * (x - 1)^8 + a9 * (x - 1)^9 + a10 * (x - 1)^10 + a11 * (x - 1)^11) →
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 = 2 := 
sorry

end sum_of_coefficients_l49_49775


namespace odd_positive_integer_minus_twenty_l49_49626

theorem odd_positive_integer_minus_twenty (x : ℕ) (h : x = 53) : (2 * x - 1) - 20 = 85 := by
  subst h
  rfl

end odd_positive_integer_minus_twenty_l49_49626


namespace spurs_team_players_l49_49721

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end spurs_team_players_l49_49721


namespace average_goals_per_game_l49_49391

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end average_goals_per_game_l49_49391


namespace rectangle_same_color_exists_l49_49289

theorem rectangle_same_color_exists (grid : Fin 3 → Fin 7 → Bool) : 
  ∃ (r1 r2 c1 c2 : Fin 3), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid r1 c1 = grid r1 c2 ∧ grid r1 c1 = grid r2 c1 ∧ grid r1 c1 = grid r2 c2 :=
by
  sorry

end rectangle_same_color_exists_l49_49289


namespace original_profit_percentage_l49_49667

theorem original_profit_percentage (C S : ℝ) (hC : C = 70)
(h1 : S - 14.70 = 1.30 * (C * 0.80)) :
  (S - C) / C * 100 = 25 := by
  sorry

end original_profit_percentage_l49_49667


namespace min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l49_49677

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * Real.log x + b

theorem min_value_f 
  (a b : ℝ) 
  (a_non_pos : a ≤ 1) : 
  f 1 a b = (1 / 2) + b :=
sorry

theorem min_value_f_sqrt 
  (a b : ℝ) 
  (a_pos_range : 1 < a ∧ a < 4) : 
  f (Real.sqrt a) a b = (a / 2) - a * Real.log (Real.sqrt a) + b :=
sorry

theorem min_value_f_2 
  (a b : ℝ) 
  (a_ge_4 : 4 ≤ a) : 
  f 2 a b = 2 - a * Real.log 2 + b :=
sorry

theorem min_m 
  (a : ℝ) 
  (a_range : -2 ≤ a ∧ a < 0):
  ∀x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 2) ∧ (0 < x2 ∧ x2 ≤ 2) →
  ∃m : ℝ, m = 12 ∧ abs (f x1 a 0 - f x2 a 0) ≤ m ^ abs (1 / x1 - 1 / x2) :=
sorry

end min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l49_49677


namespace total_wet_surface_area_correct_l49_49016

namespace Cistern

-- Define the dimensions of the cistern and the depth of the water
def length : ℝ := 10
def width : ℝ := 8
def depth : ℝ := 1.5

-- Calculate the individual surface areas
def bottom_surface_area : ℝ := length * width
def longer_side_surface_area : ℝ := length * depth * 2
def shorter_side_surface_area : ℝ := width * depth * 2

-- The total wet surface area is the sum of all individual wet surface areas
def total_wet_surface_area : ℝ := 
  bottom_surface_area + longer_side_surface_area + shorter_side_surface_area

-- Prove that the total wet surface area is 134 m^2
theorem total_wet_surface_area_correct : 
  total_wet_surface_area = 134 := 
by sorry

end Cistern

end total_wet_surface_area_correct_l49_49016


namespace intersection_nonempty_range_b_l49_49897

noncomputable def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
noncomputable def B (b : ℝ) (a : ℝ) : Set ℝ := {x | (x - b)^2 < a}

theorem intersection_nonempty_range_b (b : ℝ) : 
  A ∩ B b 1 ≠ ∅ ↔ -2 < b ∧ b < 2 := 
by
  sorry

end intersection_nonempty_range_b_l49_49897


namespace boss_spends_7600_per_month_l49_49606

def hoursPerWeekFiona : ℕ := 40
def hoursPerWeekJohn : ℕ := 30
def hoursPerWeekJeremy : ℕ := 25
def hourlyRate : ℕ := 20
def weeksPerMonth : ℕ := 4

def weeklyEarnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def monthlyEarnings (weekly : ℕ) (weeks : ℕ) : ℕ := weekly * weeks

def totalMonthlyExpenditure : ℕ :=
  monthlyEarnings (weeklyEarnings hoursPerWeekFiona hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJohn hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJeremy hourlyRate) weeksPerMonth

theorem boss_spends_7600_per_month :
  totalMonthlyExpenditure = 7600 :=
by
  sorry

end boss_spends_7600_per_month_l49_49606


namespace parallel_lines_condition_l49_49758

theorem parallel_lines_condition (a : ℝ) :
  (a = 3 / 2) ↔ (∀ x y : ℝ, (x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → (a = 3 / 2)) :=
sorry

end parallel_lines_condition_l49_49758


namespace danny_bottle_caps_l49_49065

variable (caps_found : Nat) (caps_existing : Nat)
variable (wrappers_found : Nat) (wrappers_existing : Nat)

theorem danny_bottle_caps:
  caps_found = 58 → caps_existing = 12 →
  wrappers_found = 25 → wrappers_existing = 11 →
  (caps_found + caps_existing) - (wrappers_found + wrappers_existing) = 34 := 
by
  intros h1 h2 h3 h4
  sorry

end danny_bottle_caps_l49_49065


namespace B_join_time_l49_49273

theorem B_join_time (x : ℕ) (hx : (45000 * 12) / (27000 * (12 - x)) = 2) : x = 2 :=
sorry

end B_join_time_l49_49273


namespace original_price_of_radio_l49_49712

theorem original_price_of_radio (P : ℝ) (h : 0.95 * P = 465.5) : P = 490 :=
sorry

end original_price_of_radio_l49_49712


namespace third_chest_coin_difference_l49_49658

variable (g1 g2 g3 s1 s2 s3 : ℕ)

-- Conditions
axiom h1 : g1 + g2 + g3 = 40
axiom h2 : s1 + s2 + s3 = 40
axiom h3 : g1 = s1 + 7
axiom h4 : g2 = s2 + 15

-- Goal
theorem third_chest_coin_difference : s3 = g3 + 22 :=
sorry

end third_chest_coin_difference_l49_49658


namespace product_of_solutions_abs_eq_four_l49_49053

theorem product_of_solutions_abs_eq_four :
  (∀ x : ℝ, (|x - 5| - 4 = 0) → (x = 9 ∨ x = 1)) →
  (9 * 1 = 9) :=
by
  intros h
  sorry

end product_of_solutions_abs_eq_four_l49_49053


namespace input_value_for_output_16_l49_49236

theorem input_value_for_output_16 (x : ℝ) (y : ℝ) (h1 : x < 0 → y = (x + 1)^2) (h2 : x ≥ 0 → y = (x - 1)^2) (h3 : y = 16) : x = 5 ∨ x = -5 := by
  sorry

end input_value_for_output_16_l49_49236


namespace speed_of_second_cyclist_l49_49076

theorem speed_of_second_cyclist (v : ℝ) 
  (circumference : ℝ) 
  (time : ℝ) 
  (speed_first_cyclist : ℝ)
  (meet_time : ℝ)
  (circ_full: circumference = 300) 
  (time_full: time = 20)
  (speed_first: speed_first_cyclist = 7)
  (meet_full: meet_time = time):

  v = 8 := 
by
  sorry

end speed_of_second_cyclist_l49_49076


namespace final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l49_49609

-- Define the movements as a list of integers
def movements : List ℤ := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

-- Define the function to calculate the final position
def final_position (movements : List ℤ) : ℤ :=
  movements.foldl (· + ·) 0

-- Define the function to find the total distance walked (absolute sum)
def total_distance (movements : List ℤ) : ℕ :=
  movements.foldl (fun acc x => acc + x.natAbs) 0

-- Calorie consumption rate per kilometer (1000 meters)
def calories_per_kilometer : ℕ := 7000

-- Calculate the calories consumed
def calories_consumed (total_meters : ℕ) : ℕ :=
  (total_meters / 1000) * calories_per_kilometer

-- Lean 4 theorem statements

theorem final_position_west_of_bus_stop : final_position movements = -400 := by
  sorry

theorem distance_from_bus_stop : |final_position movements| = 400 := by
  sorry

theorem total_calories_consumed : calories_consumed (total_distance movements) = 44800 := by
  sorry

end final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l49_49609


namespace common_roots_product_l49_49798

theorem common_roots_product
  (p q r s : ℝ)
  (hpqrs1 : p + q + r = 0)
  (hpqrs2 : pqr = -20)
  (hpqrs3 : p + q + s = -4)
  (hpqrs4 : pqs = -80)
  : p * q = 20 :=
sorry

end common_roots_product_l49_49798


namespace boys_girls_relation_l49_49187

theorem boys_girls_relation (b g : ℕ) :
  (∃ b, 3 + (b - 1) * 2 = g) → b = (g - 1) / 2 :=
by
  intro h
  sorry

end boys_girls_relation_l49_49187


namespace donation_value_l49_49360

def donation_in_yuan (usd: ℝ) (exchange_rate: ℝ): ℝ :=
  usd * exchange_rate

theorem donation_value :
  donation_in_yuan 1.2 6.25 = 7.5 :=
by
  -- Proof to be filled in
  sorry

end donation_value_l49_49360


namespace divides_expression_l49_49954

theorem divides_expression (y : ℕ) (hy : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end divides_expression_l49_49954


namespace files_rem_nat_eq_two_l49_49582

-- Conditions
def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23

-- Correct Answer
def files_remaining : ℕ := initial_music_files + initial_video_files - files_deleted

theorem files_rem_nat_eq_two : files_remaining = 2 := by
  sorry

end files_rem_nat_eq_two_l49_49582


namespace jill_spent_on_clothing_l49_49297

-- Define the total amount spent excluding taxes, T.
variable (T : ℝ)
-- Define the percentage of T Jill spent on clothing, C.
variable (C : ℝ)

-- Define the conditions based on the problem statement.
def jill_tax_conditions : Prop :=
  let food_percent := 0.20
  let other_items_percent := 0.30
  let clothing_tax := 0.04
  let food_tax := 0
  let other_tax := 0.10
  let total_tax := 0.05
  let food_amount := food_percent * T
  let other_items_amount := other_items_percent * T
  let clothing_amount := C * T
  let clothing_tax_amount := clothing_tax * clothing_amount
  let other_tax_amount := other_tax * other_items_amount
  let total_tax_amount := clothing_tax_amount + food_tax * food_amount + other_tax_amount
  C * T + food_percent * T + other_items_percent * T = T ∧
  total_tax_amount / T = total_tax

-- The goal is to prove that C = 0.50.
theorem jill_spent_on_clothing (h : jill_tax_conditions T C) : C = 0.50 :=
by
  sorry

end jill_spent_on_clothing_l49_49297


namespace messages_per_member_per_day_l49_49900

theorem messages_per_member_per_day (initial_members removed_members remaining_members total_weekly_messages total_daily_messages : ℕ)
  (h1 : initial_members = 150)
  (h2 : removed_members = 20)
  (h3 : remaining_members = initial_members - removed_members)
  (h4 : total_weekly_messages = 45500)
  (h5 : total_daily_messages = total_weekly_messages / 7)
  (h6 : 7 * total_daily_messages = total_weekly_messages) -- ensures that total_daily_messages calculated is correct
  : total_daily_messages / remaining_members = 50 := 
by
  sorry

end messages_per_member_per_day_l49_49900


namespace geometric_sequence_first_term_l49_49058

theorem geometric_sequence_first_term (a r : ℚ) (third_term fourth_term : ℚ) 
  (h1 : third_term = a * r^2)
  (h2 : fourth_term = a * r^3)
  (h3 : third_term = 27)
  (h4 : fourth_term = 36) : 
  a = 243 / 16 :=
by
  sorry

end geometric_sequence_first_term_l49_49058


namespace total_books_l49_49773

def keith_books : ℕ := 20
def jason_books : ℕ := 21

theorem total_books : keith_books + jason_books = 41 :=
by
  sorry

end total_books_l49_49773


namespace can_restore_axes_l49_49486

noncomputable def restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : Prop :=
  ∃ (B C D : ℝ×ℝ),
    (B.fst = A.fst ∧ B.snd = 0) ∧
    (C.fst = A.fst ∧ C.snd = A.snd) ∧
    (D.fst = A.fst ∧ D.snd = 3 ^ C.fst) ∧
    (∃ (extend_perpendicular : ∀ (x: ℝ), ℝ→ℝ), extend_perpendicular A.snd B.fst = D.snd)

theorem can_restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : restore_axes A hA :=
  sorry

end can_restore_axes_l49_49486


namespace matt_peanut_revenue_l49_49428

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end matt_peanut_revenue_l49_49428


namespace lee_sold_action_figures_l49_49347

-- Defining variables and conditions based on the problem
def sneaker_cost : ℕ := 90
def saved_money : ℕ := 15
def price_per_action_figure : ℕ := 10
def remaining_money : ℕ := 25

-- Theorem statement asserting that Lee sold 10 action figures
theorem lee_sold_action_figures : 
  (sneaker_cost - saved_money + remaining_money) / price_per_action_figure = 10  :=
by
  sorry

end lee_sold_action_figures_l49_49347


namespace orchids_cut_l49_49466

-- Define initial and final number of orchids in the vase
def initialOrchids : ℕ := 2
def finalOrchids : ℕ := 21

-- Formulate the claim to prove the number of orchids Jessica cut
theorem orchids_cut : finalOrchids - initialOrchids = 19 := by
  sorry

end orchids_cut_l49_49466


namespace sin_expression_value_l49_49982

theorem sin_expression_value (α : ℝ) (h : Real.cos (α + π / 5) = 4 / 5) :
  Real.sin (2 * α + 9 * π / 10) = 7 / 25 :=
sorry

end sin_expression_value_l49_49982


namespace neg_p_is_exists_x_l49_49485

variable (x : ℝ)

def p : Prop := ∀ x, x^2 + x + 1 ≠ 0

theorem neg_p_is_exists_x : ¬ p ↔ ∃ x, x^2 + x + 1 = 0 := by
  sorry

end neg_p_is_exists_x_l49_49485


namespace grapefruits_orchards_proof_l49_49321

/-- 
Given the following conditions:
1. There are 40 orchards in total.
2. 15 orchards are dedicated to lemons.
3. The number of orchards for oranges is two-thirds of the number of orchards for lemons.
4. Limes and grapefruits have an equal number of orchards.
5. Mandarins have half as many orchards as limes or grapefruits.
Prove that the number of citrus orchards growing grapefruits is 6.
-/
def num_grapefruit_orchards (TotalOrchards Lemons Oranges L G M : ℕ) : Prop :=
  TotalOrchards = 40 ∧
  Lemons = 15 ∧
  Oranges = 2 * Lemons / 3 ∧
  L = G ∧
  M = G / 2 ∧
  L + G + M = TotalOrchards - (Lemons + Oranges) ∧
  G = 6

theorem grapefruits_orchards_proof : ∃ (TotalOrchards Lemons Oranges L G M : ℕ), num_grapefruit_orchards TotalOrchards Lemons Oranges L G M :=
by
  sorry

end grapefruits_orchards_proof_l49_49321


namespace find_julios_bonus_l49_49722

def commission (customers: ℕ) : ℕ :=
  customers * 1

def total_commission (week1: ℕ) (week2: ℕ) (week3: ℕ) : ℕ :=
  commission week1 + commission week2 + commission week3

noncomputable def julios_bonus (total_earnings salary total_commission: ℕ) : ℕ :=
  total_earnings - salary - total_commission

theorem find_julios_bonus :
  let week1 := 35
  let week2 := 2 * week1
  let week3 := 3 * week1
  let salary := 500
  let total_earnings := 760
  let total_comm := total_commission week1 week2 week3
  julios_bonus total_earnings salary total_comm = 50 :=
by
  sorry

end find_julios_bonus_l49_49722


namespace factorize_x_squared_minus_one_l49_49580

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l49_49580


namespace digit_for_multiple_of_9_l49_49002

theorem digit_for_multiple_of_9 : 
  -- Condition: Sum of the digits 4, 5, 6, 7, 8, and d must be divisible by 9.
  (∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (4 + 5 + 6 + 7 + 8 + d) % 9 = 0) →
  -- Result: The digit d that makes 45678d a multiple of 9 is 6.
  d = 6 :=
by
  sorry

end digit_for_multiple_of_9_l49_49002


namespace greatest_integer_b_l49_49111

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ↔ b ≤ 6 := 
by
  sorry

end greatest_integer_b_l49_49111


namespace sqrt_9_eq_3_and_neg3_l49_49290

theorem sqrt_9_eq_3_and_neg3 : { x : ℝ | x^2 = 9 } = {3, -3} :=
by
  sorry

end sqrt_9_eq_3_and_neg3_l49_49290


namespace arc_length_l49_49143

theorem arc_length 
  (a : ℝ) 
  (α β : ℝ) 
  (hα : 0 < α) 
  (hβ : 0 < β) 
  (h1 : α + β < π) 
  :  ∃ l : ℝ, l = (a * (π - α - β) * (Real.sin α) * (Real.sin β)) / (Real.sin (α + β)) :=
sorry

end arc_length_l49_49143


namespace problem_solution_l49_49683

-- Define the problem conditions and state the theorem
variable (a b : ℝ)
variable (h1 : a^2 - 4 * a + 3 = 0)
variable (h2 : b^2 - 4 * b + 3 = 0)
variable (h3 : a ≠ b)

theorem problem_solution : (a+1)*(b+1) = 8 := by
  sorry

end problem_solution_l49_49683


namespace increase_in_average_l49_49476

theorem increase_in_average (s1 s2 s3 s4 s5: ℝ)
  (h1: s1 = 92) (h2: s2 = 86) (h3: s3 = 89) (h4: s4 = 94) (h5: s5 = 91):
  ( ((s1 + s2 + s3 + s4 + s5) / 5) - ((s1 + s2 + s3) / 3) ) = 1.4 :=
by
  sorry

end increase_in_average_l49_49476


namespace cylinder_radius_unique_l49_49570

theorem cylinder_radius_unique
  (r : ℝ) (h : ℝ) (V : ℝ) (y : ℝ)
  (h_eq : h = 2)
  (V_eq : V = 2 * Real.pi * r ^ 2)
  (y_eq_increase_radius : y = 2 * Real.pi * ((r + 6) ^ 2 - r ^ 2))
  (y_eq_increase_height : y = 6 * Real.pi * r ^ 2) :
  r = 6 :=
by
  sorry

end cylinder_radius_unique_l49_49570


namespace smallest_n_for_nonzero_constant_term_l49_49398

theorem smallest_n_for_nonzero_constant_term : 
  ∃ n : ℕ, (∃ r : ℕ, n = 5 * r / 3) ∧ (n > 0) ∧ ∀ m : ℕ, (m > 0) → (∃ s : ℕ, m = 5 * s / 3) → n ≤ m :=
by sorry

end smallest_n_for_nonzero_constant_term_l49_49398


namespace find_ordered_pair_l49_49432

-- Definitions based on the conditions
variable (a c : ℝ)
def has_exactly_one_solution :=
  (-6)^2 - 4 * a * c = 0

def sum_is_twelve :=
  a + c = 12

def a_less_than_c :=
  a < c

-- The proof statement
theorem find_ordered_pair
  (h₁ : has_exactly_one_solution a c)
  (h₂ : sum_is_twelve a c)
  (h₃ : a_less_than_c a c) :
  a = 3 ∧ c = 9 := 
sorry

end find_ordered_pair_l49_49432


namespace probability_range_inequality_l49_49356

theorem probability_range_inequality :
  ∀ p : ℝ, 0 ≤ p → p ≤ 1 →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2 → 0.4 ≤ p ∧ p < 1) := sorry

end probability_range_inequality_l49_49356


namespace polygon_sides_l49_49266

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end polygon_sides_l49_49266


namespace cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l49_49725

-- Definitions based on conditions
def distanceAB := 18  -- km
def speedCarA := 54   -- km/h
def speedCarB := 36   -- km/h
def targetDistance := 45  -- km

-- Proof problem statements
theorem cars_towards_each_other {y : ℝ} : 54 * y + 36 * y = 18 + 45 ↔ y = 0.7 :=
by sorry

theorem cars_same_direction_A_to_B {x : ℝ} : 54 * x - (36 * x + 18) = 45 ↔ x = 3.5 :=
by sorry

theorem cars_same_direction_B_to_A {x : ℝ} : 54 * x + 18 - 36 * x = 45 ↔ x = 1.5 :=
by sorry

end cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l49_49725


namespace tank_filling_l49_49113

theorem tank_filling (A_rate B_rate : ℚ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) :
  (1 / (A_rate - B_rate)) = 18 :=
by
  sorry

end tank_filling_l49_49113


namespace fault_line_movement_l49_49051

theorem fault_line_movement
  (moved_past_year : ℝ)
  (moved_year_before : ℝ)
  (h1 : moved_past_year = 1.25)
  (h2 : moved_year_before = 5.25) :
  moved_past_year + moved_year_before = 6.50 :=
by
  sorry

end fault_line_movement_l49_49051


namespace xyz_sum_eq_40_l49_49384

theorem xyz_sum_eq_40
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + x * z + x^2 = 91) :
  x * y + y * z + z * x = 40 :=
sorry

end xyz_sum_eq_40_l49_49384


namespace problem_part1_problem_part2_problem_part3_l49_49415

section
variables (a b : ℚ)

-- Define the operation
def otimes (a b : ℚ) : ℚ := a * b + abs a - b

-- Prove the three statements
theorem problem_part1 : otimes (-5) 4 = -19 :=
sorry

theorem problem_part2 : otimes (otimes 2 (-3)) 4 = -7 :=
sorry

theorem problem_part3 : otimes 3 (-2) > otimes (-2) 3 :=
sorry
end

end problem_part1_problem_part2_problem_part3_l49_49415


namespace radius_of_sphere_l49_49962

theorem radius_of_sphere {r x : ℝ} (h1 : 15^2 + x^2 = r^2) (h2 : r = x + 12) :
    r = 123 / 8 :=
  by
  sorry

end radius_of_sphere_l49_49962


namespace triangle_inequality_example_l49_49500

theorem triangle_inequality_example {x : ℝ} (h1: 3 + 4 > x) (h2: abs (3 - 4) < x) : 1 < x ∧ x < 7 :=
  sorry

end triangle_inequality_example_l49_49500


namespace vasya_fraction_is_0_4_l49_49577

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l49_49577


namespace eagle_speed_l49_49368

theorem eagle_speed (E : ℕ) 
  (falcon_speed : ℕ := 46)
  (pelican_speed : ℕ := 33)
  (hummingbird_speed : ℕ := 30)
  (total_distance : ℕ := 248)
  (flight_time : ℕ := 2)
  (falcon_distance := falcon_speed * flight_time)
  (pelican_distance := pelican_speed * flight_time)
  (hummingbird_distance := hummingbird_speed * flight_time) :
  2 * E + falcon_distance + pelican_distance + hummingbird_distance = total_distance →
  E = 15 :=
by
  -- Proof will be provided here
  sorry

end eagle_speed_l49_49368


namespace determine_n_between_sqrt3_l49_49529

theorem determine_n_between_sqrt3 (n : ℕ) (hpos : 0 < n)
  (hineq : (n + 3) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4) / (n + 1)) :
  n = 4 :=
sorry

end determine_n_between_sqrt3_l49_49529


namespace train_journey_time_l49_49768

theorem train_journey_time :
  ∃ T : ℝ, (30 : ℝ) / 60 = (7 / 6 * T) - T ∧ T = 3 :=
by
  sorry

end train_journey_time_l49_49768


namespace minimum_expression_l49_49508

variable (a b : ℝ)

theorem minimum_expression (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 3) :
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x + y = 3 → 
  x = a ∧ y = b  → ∃ m : ℝ, m ≥ 1 ∧ (m = (1/(a+1)) + 1/b))) := sorry

end minimum_expression_l49_49508


namespace shift_right_symmetric_l49_49927

open Real

/-- Given the function y = sin(2x + π/3), after shifting the graph of the function right
    by φ (0 < φ < π/2) units, the resulting graph is symmetric about the y-axis.
    Prove that the value of φ is 5π/12.
-/
theorem shift_right_symmetric (φ : ℝ) (hφ₁ : 0 < φ) (hφ₂ : φ < π / 2)
  (h_sym : ∃ k : ℤ, -2 * φ + π / 3 = k * π + π / 2) : φ = 5 * π / 12 :=
sorry

end shift_right_symmetric_l49_49927


namespace youngest_sibling_is_42_l49_49149

-- Definitions for the problem conditions
def consecutive_even_integers (a : ℤ) := [a, a + 2, a + 4, a + 6]
def sum_of_ages_is_180 (ages : List ℤ) := ages.sum = 180

-- Main statement
theorem youngest_sibling_is_42 (a : ℤ) 
  (h1 : sum_of_ages_is_180 (consecutive_even_integers a)) :
  a = 42 := 
sorry

end youngest_sibling_is_42_l49_49149


namespace derivative_at_neg_one_l49_49555

theorem derivative_at_neg_one (a b c : ℝ) (h : (4*a*(1:ℝ)^3 + 2*b*(1:ℝ)) = 2) :
  (4*a*(-1:ℝ)^3 + 2*b*(-1:ℝ)) = -2 :=
by
  sorry

end derivative_at_neg_one_l49_49555


namespace total_remaining_books_l49_49207

-- Define the initial conditions as constants
def total_books_crazy_silly_school : ℕ := 14
def read_books_crazy_silly_school : ℕ := 8
def total_books_mystical_adventures : ℕ := 10
def read_books_mystical_adventures : ℕ := 5
def total_books_sci_fi_universe : ℕ := 18
def read_books_sci_fi_universe : ℕ := 12

-- Define the remaining books calculation
def remaining_books_crazy_silly_school : ℕ :=
  total_books_crazy_silly_school - read_books_crazy_silly_school

def remaining_books_mystical_adventures : ℕ :=
  total_books_mystical_adventures - read_books_mystical_adventures

def remaining_books_sci_fi_universe : ℕ :=
  total_books_sci_fi_universe - read_books_sci_fi_universe

-- Define the proof statement
theorem total_remaining_books : 
  remaining_books_crazy_silly_school + remaining_books_mystical_adventures + remaining_books_sci_fi_universe = 17 := by
  sorry

end total_remaining_books_l49_49207


namespace interval_between_prizes_l49_49336

theorem interval_between_prizes (total_prize : ℝ) (first_place : ℝ) (interval : ℝ) :
  total_prize = 4800 ∧
  first_place = 2000 ∧
  (first_place - interval) + (first_place - 2 * interval) = total_prize - 2000 →
  interval = 400 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2] at h3
  sorry

end interval_between_prizes_l49_49336


namespace find_number_l49_49573

theorem find_number :
  ∃ n : ℤ,
    (n % 12 = 11) ∧ 
    (n % 11 = 10) ∧ 
    (n % 10 = 9) ∧ 
    (n % 9 = 8) ∧ 
    (n % 8 = 7) ∧ 
    (n % 7 = 6) ∧ 
    (n % 6 = 5) ∧ 
    (n % 5 = 4) ∧ 
    (n % 4 = 3) ∧ 
    (n % 3 = 2) ∧ 
    (n % 2 = 1) ∧
    n = 27719 :=
sorry

end find_number_l49_49573


namespace total_students_in_high_school_l49_49237

theorem total_students_in_high_school 
  (num_freshmen : ℕ)
  (num_sample : ℕ) 
  (num_sophomores : ℕ)
  (num_seniors : ℕ)
  (freshmen_drawn : ℕ)
  (sampling_ratio : ℕ)
  (total_students : ℕ)
  (h1 : num_freshmen = 600)
  (h2 : num_sample = 45)
  (h3 : num_sophomores = 20)
  (h4 : num_seniors = 10)
  (h5 : freshmen_drawn = 15)
  (h6 : sampling_ratio = 40)
  (h7 : freshmen_drawn * sampling_ratio = num_freshmen)
  : total_students = 1800 :=
sorry

end total_students_in_high_school_l49_49237


namespace perpendicular_vectors_m_val_l49_49171

theorem perpendicular_vectors_m_val (m : ℝ) 
  (a : ℝ × ℝ := (-1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 2 := 
by 
  sorry

end perpendicular_vectors_m_val_l49_49171


namespace total_bottle_caps_l49_49777

-- Define the conditions
def bottle_caps_per_child : ℕ := 5
def number_of_children : ℕ := 9

-- Define the main statement to be proven
theorem total_bottle_caps : bottle_caps_per_child * number_of_children = 45 :=
by sorry

end total_bottle_caps_l49_49777


namespace paco_cookie_problem_l49_49482

theorem paco_cookie_problem (x : ℕ) (hx : x + 9 = 18) : x = 9 :=
by sorry

end paco_cookie_problem_l49_49482


namespace a3_plus_a4_value_l49_49675

theorem a3_plus_a4_value
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : (1 - 2*x)^5 = a_0 + a_1*(1 + x) + a_2*(1 + x)^2 + a_3*(1 + x)^3 + a_4*(1 + x)^4 + a_5*(1 + x)^5) :
  a_3 + a_4 = -480 := 
sorry

end a3_plus_a4_value_l49_49675


namespace tangent_line_equation_at_x_zero_l49_49027

noncomputable def curve (x : ℝ) : ℝ := x + Real.exp (2 * x)

theorem tangent_line_equation_at_x_zero :
  ∃ (k b : ℝ), (∀ x : ℝ, curve x = k * x + b) :=
by
  let df := fun (x : ℝ) => (deriv curve x)
  have k : ℝ := df 0
  have b : ℝ := curve 0 - k * 0
  use k, b
  sorry

end tangent_line_equation_at_x_zero_l49_49027


namespace count_8_digit_even_ending_l49_49819

theorem count_8_digit_even_ending : 
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  (choices_first_digit * choices_middle_digits * choices_last_digit) = 45000000 :=
by
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  sorry

end count_8_digit_even_ending_l49_49819


namespace decimal_6_to_binary_is_110_l49_49337

def decimal_to_binary (n : ℕ) : ℕ :=
  -- This is just a placeholder definition. Adjust as needed for formalization.
  sorry

theorem decimal_6_to_binary_is_110 :
  decimal_to_binary 6 = 110 := 
sorry

end decimal_6_to_binary_is_110_l49_49337


namespace total_cases_after_three_weeks_l49_49118

theorem total_cases_after_three_weeks (week1_cases week2_cases week3_cases : ℕ) 
  (h1 : week1_cases = 5000)
  (h2 : week2_cases = week1_cases + week1_cases / 10 * 3)
  (h3 : week3_cases = week2_cases - week2_cases / 10 * 2) :
  week1_cases + week2_cases + week3_cases = 16700 := 
by
  sorry

end total_cases_after_three_weeks_l49_49118


namespace application_outcomes_l49_49311

theorem application_outcomes :
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  (choices_A * choices_B * choices_C) = 18 :=
by
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  show (choices_A * choices_B * choices_C = 18)
  sorry

end application_outcomes_l49_49311


namespace man_l49_49576

variable (v : ℝ) (speed_with_current : ℝ) (speed_of_current : ℝ)

theorem man's_speed_against_current :
  speed_with_current = 12 ∧ speed_of_current = 2 → v - speed_of_current = 8 :=
by
  sorry

end man_l49_49576


namespace fixed_chord_property_l49_49961

theorem fixed_chord_property (d : ℝ) (h₁ : d = 3 / 2) :
  ∀ (x1 x2 m : ℝ) (h₀ : x1 + x2 = m) (h₂ : x1 * x2 = 1 - d),
    ((1 / ((x1 ^ 2) + (m * x1) ^ 2)) + (1 / ((x2 ^ 2) + (m * x2) ^ 2))) = 4 / 9 :=
by
  sorry

end fixed_chord_property_l49_49961


namespace sumOfTrianglesIs34_l49_49734

def triangleOp (a b c : ℕ) : ℕ := a * b - c

theorem sumOfTrianglesIs34 : 
  triangleOp 3 5 2 + triangleOp 4 6 3 = 34 := 
by
  sorry

end sumOfTrianglesIs34_l49_49734


namespace no_partition_of_six_consecutive_numbers_product_equal_l49_49916

theorem no_partition_of_six_consecutive_numbers_product_equal (n : ℕ) :
  ¬ ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (n+6) ∧ 
    A ∩ B = ∅ ∧ 
    A.prod id = B.prod id :=
by
  sorry

end no_partition_of_six_consecutive_numbers_product_equal_l49_49916


namespace paper_area_l49_49459

theorem paper_area (L W : ℝ) 
(h1 : 2 * L + W = 34) 
(h2 : L + 2 * W = 38) : 
L * W = 140 := by
  sorry

end paper_area_l49_49459


namespace fraction_of_buttons_l49_49587

variable (K S M : ℕ)  -- Kendra's buttons, Sue's buttons, Mari's buttons

theorem fraction_of_buttons (H1 : M = 5 * K + 4) 
                            (H2 : S = 6)
                            (H3 : M = 64) :
  S / K = 1 / 2 := by
  sorry

end fraction_of_buttons_l49_49587


namespace investment_amount_first_rate_l49_49643

theorem investment_amount_first_rate : ∀ (x y : ℝ) (r : ℝ),
  x + y = 15000 → -- Condition 1 (Total investments)
  8200 * r + 6800 * 0.075 = 1023 → -- Condition 2 (Interest yield)
  x = 8200 → -- Condition 3 (Amount invested at first rate)
  x = 8200 := -- Question (How much was invested)
by
  intros x y r h₁ h₂ h₃
  exact h₃

end investment_amount_first_rate_l49_49643


namespace find_third_number_l49_49305

noncomputable def third_number := 9.110300000000005

theorem find_third_number :
  12.1212 + 17.0005 - third_number = 20.011399999999995 :=
sorry

end find_third_number_l49_49305


namespace staffing_ways_l49_49409

def total_resumes : ℕ := 30
def unsuitable_resumes : ℕ := 10
def suitable_resumes : ℕ := total_resumes - unsuitable_resumes
def position_count : ℕ := 5

theorem staffing_ways :
  20 * 19 * 18 * 17 * 16 = 1860480 := by
  sorry

end staffing_ways_l49_49409


namespace expression_evaluation_l49_49831

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x^2 - 4 * y + 5 = 24 :=
by
  sorry

end expression_evaluation_l49_49831


namespace divisibility_56786730_polynomial_inequality_l49_49727

theorem divisibility_56786730 (m n : ℤ) : 56786730 ∣ m * n * (m^60 - n^60) :=
sorry

theorem polynomial_inequality (m n : ℤ) : m^5 + 3 * m^4 * n - 5 * m^3 * n^2 - 15 * m^2 * n^3 + 4 * m * n^4 + 12 * n^5 ≠ 33 :=
sorry

end divisibility_56786730_polynomial_inequality_l49_49727


namespace min_solution_of_x_abs_x_eq_3x_plus_4_l49_49907

theorem min_solution_of_x_abs_x_eq_3x_plus_4 : 
  ∃ x : ℝ, (x * |x| = 3 * x + 4) ∧ ∀ y : ℝ, (y * |y| = 3 * y + 4) → x ≤ y :=
sorry

end min_solution_of_x_abs_x_eq_3x_plus_4_l49_49907


namespace matrix_determinant_6_l49_49886

theorem matrix_determinant_6 (x y z w : ℝ)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 2 * w) - z * (5 * x + 2 * y)) = 6 :=
by
  sorry

end matrix_determinant_6_l49_49886


namespace plane_determination_l49_49651

inductive Propositions : Type where
  | p1 : Propositions
  | p2 : Propositions
  | p3 : Propositions
  | p4 : Propositions

open Propositions

def correct_proposition := p4

theorem plane_determination (H: correct_proposition = p4): correct_proposition = p4 := 
by 
  exact H

end plane_determination_l49_49651


namespace die_roll_probability_div_3_l49_49893

noncomputable def probability_divisible_by_3 : ℚ :=
  1 - ((2 : ℚ) / 3) ^ 8

theorem die_roll_probability_div_3 :
  probability_divisible_by_3 = 6305 / 6561 :=
by
  sorry

end die_roll_probability_div_3_l49_49893


namespace work_efficiency_ratio_l49_49544

variable (A B : ℝ)
variable (h1 : A = 1 / 2 * B) 
variable (h2 : 1 / (A + B) = 13)
variable (h3 : B = 1 / 19.5)

theorem work_efficiency_ratio : A / B = 1 / 2 := by
  sorry

end work_efficiency_ratio_l49_49544


namespace base12_addition_example_l49_49167

theorem base12_addition_example : 
  (5 * 12^2 + 2 * 12^1 + 8 * 12^0) + (2 * 12^2 + 7 * 12^1 + 3 * 12^0) = (7 * 12^2 + 9 * 12^1 + 11 * 12^0) :=
by sorry

end base12_addition_example_l49_49167


namespace work_rates_l49_49558

theorem work_rates (A B : ℝ) (combined_days : ℝ) (b_rate: B = 35) 
(combined_rate: combined_days = 20 / 11):
    A = 700 / 365 :=
by
  have h1 : B = 35 := by sorry
  have h2 : combined_days = 20 / 11 := by sorry
  have : 1/A + 1/B = 11/20 := by sorry
  have : 1/A = 11/20 - 1/B := by sorry
  have : 1/A =  365 / 700:= by sorry
  have : A = 700 / 365 := by sorry
  assumption

end work_rates_l49_49558


namespace problem_solution_l49_49433

theorem problem_solution
  (m : ℝ) (n : ℝ)
  (h1 : m = 1 / (Real.sqrt 3 + Real.sqrt 2))
  (h2 : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) :
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 :=
by sorry

end problem_solution_l49_49433


namespace load_transportable_l49_49841

theorem load_transportable :
  ∃ (n : ℕ), n ≤ 11 ∧ (∀ (box_weight : ℕ) (total_weight : ℕ),
    total_weight = 13500 ∧ 
    box_weight ≤ 350 ∧ 
    (n * 1500) ≥ total_weight) :=
by
  sorry

end load_transportable_l49_49841


namespace parabola_coordinates_l49_49782

theorem parabola_coordinates (x y : ℝ) (h_parabola : y^2 = 4 * x) (h_distance : (x - 1)^2 + y^2 = 100) :
  (x = 9 ∧ y = 6) ∨ (x = 9 ∧ y = -6) :=
by
  sorry

end parabola_coordinates_l49_49782


namespace batsman_avg_increase_l49_49689

theorem batsman_avg_increase (R : ℕ) (A : ℕ) : 
  (R + 48 = 12 * 26) ∧ (R = 11 * A) → 26 - A = 2 :=
by
  intro h
  have h1 : R + 48 = 312 := h.1
  have h2 : R = 11 * A := h.2
  sorry

end batsman_avg_increase_l49_49689


namespace max_value_fraction_l49_49361

theorem max_value_fraction (x : ℝ) : 
  (∃ x, (x^4 / (x^8 + 4 * x^6 - 8 * x^4 + 16 * x^2 + 64)) = (1 / 24)) := 
sorry

end max_value_fraction_l49_49361


namespace novels_per_month_l49_49827

theorem novels_per_month (pages_per_novel : ℕ) (total_pages_per_year : ℕ) (months_in_year : ℕ) 
  (h1 : pages_per_novel = 200) (h2 : total_pages_per_year = 9600) (h3 : months_in_year = 12) : 
  (total_pages_per_year / pages_per_novel) / months_in_year = 4 :=
by
  have novels_per_year := total_pages_per_year / pages_per_novel
  have novels_per_month := novels_per_year / months_in_year
  sorry

end novels_per_month_l49_49827


namespace cancel_terms_valid_equation_l49_49177

theorem cancel_terms_valid_equation {m n : ℕ} 
  (x : Fin n → ℕ) (y : Fin m → ℕ) 
  (h_sum_eq : (Finset.univ.sum x) = (Finset.univ.sum y))
  (h_sum_lt : (Finset.univ.sum x) < (m * n)) : 
  ∃ x' : Fin n → ℕ, ∃ y' : Fin m → ℕ, 
    (Finset.univ.sum x' = Finset.univ.sum y') ∧ x' ≠ x ∧ y' ≠ y :=
sorry

end cancel_terms_valid_equation_l49_49177


namespace g_at_8_l49_49523

def g (x : ℝ) : ℝ := sorry

axiom g_property : ∀ x y : ℝ, x * g y = y * g x

axiom g_at_24 : g 24 = 12

theorem g_at_8 : g 8 = 4 := by
  sorry

end g_at_8_l49_49523


namespace proof_problem_l49_49271

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end proof_problem_l49_49271


namespace sum_of_three_squares_l49_49778

theorem sum_of_three_squares (n : ℕ) (h : n = 100) : 
  ∃ (a b c : ℕ), a = 4 ∧ b^2 + c^2 = 84 ∧ a^2 + b^2 + c^2 = 100 ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c ∨ (b = c ∧ a ≠ b)) ∧
  (4^2 + 7^2 + 6^2 = 100 ∧ 4^2 + 8^2 + 5^2 = 100 ∧ 4^2 + 9^2 + 1^2 = 100) ∧
  (4^2 + 6^2 + 7^2 ≠ 100 ∧ 4^2 + 5^2 + 8^2 ≠ 100 ∧ 4^2 + 1^2 + 9^2 ≠ 100 ∧ 
   4^2 + 4^2 + 8^2 ≠ 100 ∨ 4^2 + 8^2 + 4^2 ≠ 100) :=
sorry

end sum_of_three_squares_l49_49778


namespace emmy_rosa_ipods_total_l49_49753

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end emmy_rosa_ipods_total_l49_49753


namespace remainder_of_division_l49_49045

theorem remainder_of_division :
  ∀ (x : ℝ), (3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8) % (x ^ 2 - 3 * x + 2) = 74 * x - 76 :=
by
  sorry

end remainder_of_division_l49_49045


namespace probability_both_selected_l49_49624

/- 
Problem statement: Given that the probability of selection of Ram is 5/7 and that of Ravi is 1/5,
prove that the probability that both Ram and Ravi are selected is 1/7.
-/

theorem probability_both_selected (pRam : ℚ) (pRavi : ℚ) (hRam : pRam = 5 / 7) (hRavi : pRavi = 1 / 5) :
  (pRam * pRavi) = 1 / 7 :=
by
  sorry

end probability_both_selected_l49_49624


namespace trapezoid_area_l49_49952

-- Definitions based on the given conditions
variable (BD AC h : ℝ)
variable (BD_perpendicular_AC : BD * AC = 0)
variable (BD_val : BD = 13)
variable (h_val : h = 12)

-- Statement of the theorem to prove the area of the trapezoid
theorem trapezoid_area (BD AC h : ℝ)
  (BD_perpendicular_AC : BD * AC = 0)
  (BD_val : BD = 13)
  (h_val : h = 12) :
  0.5 * 13 * 12 = 1014 / 5 := sorry

end trapezoid_area_l49_49952


namespace range_of_a_l49_49326

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - a = 0) ↔ a ≥ -1 :=
by
  sorry

end range_of_a_l49_49326


namespace find_m_of_lcm_conditions_l49_49238

theorem find_m_of_lcm_conditions (m : ℕ) (h_pos : 0 < m)
  (h1 : Int.lcm 18 m = 54)
  (h2 : Int.lcm m 45 = 180) : m = 36 :=
sorry

end find_m_of_lcm_conditions_l49_49238


namespace fraction_sequence_calc_l49_49857

theorem fraction_sequence_calc : 
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) - 1 = -(7 / 9) := 
by 
  sorry

end fraction_sequence_calc_l49_49857


namespace find_stamps_l49_49553

def stamps_problem (x y : ℕ) : Prop :=
  (x + y = 70) ∧ (y = 4 * x + 5)

theorem find_stamps (x y : ℕ) (h : stamps_problem x y) : 
  x = 13 ∧ y = 57 :=
sorry

end find_stamps_l49_49553


namespace inequality_proof_l49_49212

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 :=
by
  sorry

end inequality_proof_l49_49212


namespace relationship_above_l49_49981

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 15 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt 2

theorem relationship_above (ha : a = Real.log 5 / Real.log 2) 
                           (hb : b = Real.log 15 / (2 * Real.log 2))
                           (hc : c = Real.sqrt 2) : a > b ∧ b > c :=
by
  sorry

end relationship_above_l49_49981


namespace star_polygon_edges_congruent_l49_49397

theorem star_polygon_edges_congruent
  (n : ℕ)
  (α β : ℝ)
  (h1 : ∀ i j : ℕ, i ≠ j → (n = 133))
  (h2 : α = (5 / 14) * β)
  (h3 : n * (α + β) = 360) :
n = 133 :=
by sorry

end star_polygon_edges_congruent_l49_49397


namespace probability_three_cards_l49_49152

theorem probability_three_cards (S : Type) [Fintype S]
  (deck : Finset S) (n : ℕ) (hn : n = 52)
  (hearts : Finset S) (spades : Finset S)
  (tens: Finset S)
  (hhearts_count : ∃ k, hearts.card = k ∧ k = 13)
  (hspades_count : ∃ k, spades.card = k ∧ k = 13)
  (htens_count : ∃ k, tens.card = k ∧ k = 4)
  (hdeck_partition : ∀ x ∈ deck, x ∈ hearts ∨ x ∈ spades ∨ x ∈ tens ∨ (x ∉ hearts ∧ x ∉ spades ∧ x ∉ tens)) :
  (12 / 52 * 13 / 51 * 4 / 50 + 1 / 52 * 13 / 51 * 3 / 50 = 221 / 44200) :=
by {
  sorry
}

end probability_three_cards_l49_49152


namespace number_of_adults_in_sleeper_class_l49_49054

-- Number of passengers in the train
def total_passengers : ℕ := 320

-- Percentage of passengers who are adults
def percentage_adults : ℚ := 75 / 100

-- Percentage of adults who are in the sleeper class
def percentage_adults_sleeper_class : ℚ := 15 / 100

-- Mathematical statement to prove
theorem number_of_adults_in_sleeper_class :
  (total_passengers * percentage_adults * percentage_adults_sleeper_class) = 36 :=
by
  sorry

end number_of_adults_in_sleeper_class_l49_49054


namespace depth_of_well_l49_49971

noncomputable def volume_of_cylinder (radius : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * radius^2 * depth

theorem depth_of_well (volume depth : ℝ) (r : ℝ) : 
  r = 1 ∧ volume = 25.132741228718345 ∧ 2 * r = 2 → depth = 8 :=
by
  intros h
  sorry

end depth_of_well_l49_49971


namespace negation_universal_to_existential_l49_49885

theorem negation_universal_to_existential :
  ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_universal_to_existential_l49_49885


namespace value_of_y_l49_49235

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(2*y) = 4) : y = 1 :=
by
  sorry

end value_of_y_l49_49235


namespace justin_reading_ratio_l49_49717

theorem justin_reading_ratio
  (pages_total : ℝ)
  (pages_first_day : ℝ)
  (pages_left : ℝ)
  (days_remaining : ℝ) :
  pages_total = 130 → 
  pages_first_day = 10 → 
  pages_left = pages_total - pages_first_day →
  days_remaining = 6 →
  (∃ R : ℝ, 60 * R = pages_left) → 
  ∃ R : ℝ, 60 * R = pages_left ∧ R = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end justin_reading_ratio_l49_49717


namespace find_A_find_B_l49_49281

-- First problem: Prove A = 10 given 100A = 35^2 - 15^2
theorem find_A (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) : A = 10 := by
  sorry

-- Second problem: Prove B = 4 given (A-1)^6 = 27^B and A = 10
theorem find_B (B : ℕ) (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) (h₂ : (A - 1) ^ 6 = 27 ^ B) : B = 4 := by
  have A_is_10 : A = 10 := by
    apply find_A
    assumption
  sorry

end find_A_find_B_l49_49281


namespace range_of_a_l49_49148

theorem range_of_a (f : ℝ → ℝ) (a : ℝ):
  (∀ x, f x = f (-x)) →
  (∀ x y, 0 ≤ x → x < y → f x ≤ f y) →
  (∀ x, 1/2 ≤ x ∧ x ≤ 1 → f (a * x + 1) ≤ f (x - 2)) →
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l49_49148


namespace max_cookies_Andy_can_eat_l49_49327

theorem max_cookies_Andy_can_eat 
  (x y : ℕ) 
  (h1 : x + y = 36)
  (h2 : y ≥ 2 * x) : 
  x ≤ 12 := by
  sorry

end max_cookies_Andy_can_eat_l49_49327


namespace unique_solution_to_equation_l49_49183

theorem unique_solution_to_equation (x y z : ℤ) 
    (h : 5 * x^3 + 11 * y^3 + 13 * z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end unique_solution_to_equation_l49_49183


namespace fraction_of_trunks_l49_49173

theorem fraction_of_trunks (h1 : 0.38 ≤ 1) (h2 : 0.63 ≤ 1) : 
  0.63 - 0.38 = 0.25 :=
by
  sorry

end fraction_of_trunks_l49_49173


namespace max_value_l49_49719

theorem max_value (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) : 
  8 * x + 3 * y + 15 * z ≤ Real.sqrt 298 :=
sorry

end max_value_l49_49719


namespace g_property_l49_49648

theorem g_property (g : ℝ → ℝ) (h : ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y) :
  let n := 2
  let s := 14 / 3
  n = 2 ∧ s = 14 / 3 ∧ n * s = 28 / 3 :=
by {
  sorry
}

end g_property_l49_49648


namespace division_example_l49_49569

theorem division_example : 72 / (6 / 3) = 36 :=
by sorry

end division_example_l49_49569


namespace best_fit_line_slope_l49_49589

theorem best_fit_line_slope (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (d : ℝ) 
  (h1 : x2 - x1 = 2 * d) (h2 : x3 - x2 = 3 * d) (h3 : x4 - x3 = d) : 
  ((y4 - y1) / (x4 - x1)) = (y4 - y1) / (x4 - x1) :=
by
  sorry

end best_fit_line_slope_l49_49589


namespace find_x_l49_49737

theorem find_x {x : ℝ} (hx : x^2 - 5 * x = -4) : x = 1 ∨ x = 4 :=
sorry

end find_x_l49_49737


namespace length_QF_l49_49783

-- Define parabola C as y^2 = 8x
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 * P.2 = 8 * P.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the condition that Q is on the parabola and the line PF in the first quadrant
def is_intersection_and_in_first_quadrant (Q : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  is_on_parabola Q ∧ Q.1 - Q.2 - 2 = 0 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the vector relation between P, Q, and F
def vector_relation (P Q F : ℝ × ℝ) : Prop :=
  let vPQ := (Q.1 - P.1, Q.2 - P.2)
  let vQF := (F.1 - Q.1, F.2 - Q.2)
  (vPQ.1^2 + vPQ.2^2) = 2 * (vQF.1^2 + vQF.2^2)

-- Lean 4 statement of the proof problem
theorem length_QF (Q : ℝ × ℝ) (P : ℝ × ℝ) :
  is_on_parabola Q ∧ is_intersection_and_in_first_quadrant Q P ∧ vector_relation P Q focus → 
  dist Q focus = 8 + 4 * Real.sqrt 2 :=
by
  sorry

end length_QF_l49_49783


namespace fraction_of_married_men_l49_49332

-- We start by defining the conditions given in the problem.
def only_single_women_and_married_couples (total_women total_married_women : ℕ) :=
  total_women - total_married_women + total_married_women * 2

def probability_single_woman_single (total_women total_single_women : ℕ) :=
  total_single_women / total_women = 3 / 7

-- The main theorem we need to prove under the given conditions.
theorem fraction_of_married_men (total_women total_married_women : ℕ)
  (h1 : probability_single_woman_single total_women (total_women - total_married_women))
  : (total_married_women * 2) / (total_women + total_married_women) = 4 / 11 := sorry

end fraction_of_married_men_l49_49332


namespace platform_length_is_correct_l49_49474

-- Given Definitions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 42
def time_to_cross_pole : ℝ := 18

-- Definition to prove
theorem platform_length_is_correct :
  ∃ L : ℝ, L = 400 ∧ (length_of_train + L) / time_to_cross_platform = length_of_train / time_to_cross_pole :=
by
  sorry

end platform_length_is_correct_l49_49474


namespace not_B_l49_49829

def op (x y : ℝ) := (x - y) ^ 2

theorem not_B (x y : ℝ) : 2 * (op x y) ≠ op (2 * x) (2 * y) :=
by
  sorry

end not_B_l49_49829


namespace round_robin_teams_l49_49399

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 :=
sorry

end round_robin_teams_l49_49399


namespace initial_quantity_of_A_l49_49464

noncomputable def initial_quantity_of_A_in_can (initial_total_mixture : ℤ) (x : ℤ) := 7 * x

theorem initial_quantity_of_A
  (initial_ratio_A : ℤ) (initial_ratio_B : ℤ) (initial_ratio_C : ℤ)
  (initial_total_mixture : ℤ) (drawn_off_mixture : ℤ) (new_quantity_of_B : ℤ)
  (new_ratio_A : ℤ) (new_ratio_B : ℤ) (new_ratio_C : ℤ)
  (h1 : initial_ratio_A = 7) (h2 : initial_ratio_B = 5) (h3 : initial_ratio_C = 3)
  (h4 : initial_total_mixture = 15 * x)
  (h5 : new_ratio_A = 7) (h6 : new_ratio_B = 9) (h7 : new_ratio_C = 3)
  (h8 : drawn_off_mixture = 18)
  (h9 : new_quantity_of_B = 5 * x - (5 / 15) * 18 + 18)
  (h10 : (7 * x - (7 / 15) * 18) / new_quantity_of_B = 7 / 9) :
  initial_quantity_of_A_in_can initial_total_mixture x = 54 :=
by
  sorry

end initial_quantity_of_A_l49_49464


namespace negation_P1_is_false_negation_P2_is_false_l49_49504

-- Define the propositions
def isMultiDigitNumber (n : ℕ) : Prop := n >= 10
def lastDigitIsZero (n : ℕ) : Prop := n % 10 = 0
def isMultipleOfFive (n : ℕ) : Prop := n % 5 = 0
def isEven (n : ℕ) : Prop := n % 2 = 0

-- The propositions
def P1 (n : ℕ) : Prop := isMultiDigitNumber n → (lastDigitIsZero n → isMultipleOfFive n)
def P2 : Prop := ∀ n, isEven n → n % 2 = 0

-- The negations
def notP1 (n : ℕ) : Prop := isMultiDigitNumber n ∧ lastDigitIsZero n → ¬isMultipleOfFive n
def notP2 : Prop := ∃ n, isEven n ∧ ¬(n % 2 = 0)

-- The proof problems
theorem negation_P1_is_false (n : ℕ) : notP1 n → False := by
  sorry

theorem negation_P2_is_false : notP2 → False := by
  sorry

end negation_P1_is_false_negation_P2_is_false_l49_49504


namespace matrix_determinant_zero_l49_49137

theorem matrix_determinant_zero (a b : ℝ) : 
  Matrix.det ![
    ![1, Real.sin (2 * a), Real.sin a],
    ![Real.sin (2 * a), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 := 
by 
  sorry

end matrix_determinant_zero_l49_49137


namespace value_of_y_l49_49325

theorem value_of_y (y : ℝ) (α : ℝ) (h₁ : (-3, y) = (x, y)) (h₂ : Real.sin α = -3 / 4) : 
  y = -9 * Real.sqrt 7 / 7 := 
  sorry

end value_of_y_l49_49325


namespace proof_two_digit_number_l49_49354

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end proof_two_digit_number_l49_49354


namespace cube_sphere_surface_area_l49_49984

open Real

noncomputable def cube_edge_length := 1
noncomputable def cube_space_diagonal := sqrt 3
noncomputable def sphere_radius := cube_space_diagonal / 2
noncomputable def sphere_surface_area := 4 * π * (sphere_radius ^ 2)

theorem cube_sphere_surface_area :
  sphere_surface_area = 3 * π :=
by
  sorry

end cube_sphere_surface_area_l49_49984


namespace find_paycheck_l49_49584

variable (P : ℝ) -- P represents the paycheck amount

def initial_balance : ℝ := 800
def rent_payment : ℝ := 450
def electricity_bill : ℝ := 117
def internet_bill : ℝ := 100
def phone_bill : ℝ := 70
def final_balance : ℝ := 1563

theorem find_paycheck :
  initial_balance - rent_payment + P - (electricity_bill + internet_bill) - phone_bill = final_balance → 
    P = 1563 :=
by
  sorry

end find_paycheck_l49_49584


namespace mass_of_man_is_120_l49_49711

def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def height_water_rise : ℝ := 0.02
def density_of_water : ℝ := 1000
def volume_displaced : ℝ := length_of_boat * breadth_of_boat * height_water_rise
def mass_of_man := density_of_water * volume_displaced

theorem mass_of_man_is_120 : mass_of_man = 120 :=
by
  -- insert the detailed proof here
  sorry

end mass_of_man_is_120_l49_49711


namespace initial_mixture_l49_49441

theorem initial_mixture (M : ℝ) (h1 : 0.20 * M + 20 = 0.36 * (M + 20)) : 
  M = 80 :=
by
  sorry

end initial_mixture_l49_49441


namespace parabola_directrix_l49_49527

theorem parabola_directrix (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end parabola_directrix_l49_49527


namespace gcd_lcm_sum_l49_49732

-- Definitions
def gcd_42_70 := Nat.gcd 42 70
def lcm_8_32 := Nat.lcm 8 32

-- Theorem statement
theorem gcd_lcm_sum : gcd_42_70 + lcm_8_32 = 46 := by
  sorry

end gcd_lcm_sum_l49_49732


namespace min_value_of_function_l49_49413

open Real

theorem min_value_of_function (x y : ℝ) (h : 2 * x + 8 * y = 3) : ∃ (min_value : ℝ), min_value = -19 / 20 ∧ ∀ (x y : ℝ), 2 * x + 8 * y = 3 → x^2 + 4 * y^2 - 2 * x ≥ -19 / 20 :=
by
  sorry

end min_value_of_function_l49_49413


namespace remainder_when_divided_by_seven_l49_49022

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end remainder_when_divided_by_seven_l49_49022


namespace value_of_a_l49_49820

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 24 - 4 * a) : a = 3 :=
by
  sorry

end value_of_a_l49_49820


namespace largest_term_at_k_31_l49_49312

noncomputable def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.15)^k

theorem largest_term_at_k_31 : 
  ∀ k : ℕ, (k ≤ 500) →
    (B_k 31 ≥ B_k k) :=
by
  intro k hk
  sorry

end largest_term_at_k_31_l49_49312


namespace possible_amounts_l49_49772

theorem possible_amounts (n : ℕ) : 
  ¬ (∃ x y : ℕ, 3 * x + 5 * y = n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 7 :=
sorry

end possible_amounts_l49_49772


namespace zeroes_in_base_81_l49_49316

-- Definitions based on the conditions:
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question: How many zeroes does 15! end with in base 81?
-- Lean 4 proof statement:
theorem zeroes_in_base_81 (n : ℕ) : n = 15 → Nat.factorial n = 
  (81 : ℕ) ^ k * m → k = 1 :=
by
  sorry

end zeroes_in_base_81_l49_49316


namespace monotonicity_decreasing_range_l49_49681

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem monotonicity_decreasing_range (ω : ℝ) :
  (∀ x y : ℝ, (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) → f ω x > f ω y) ↔ (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
sorry

end monotonicity_decreasing_range_l49_49681


namespace PetyaWinsAgainstSasha_l49_49049

def MatchesPlayed (name : String) : Nat :=
if name = "Petya" then 12 else if name = "Sasha" then 7 else if name = "Misha" then 11 else 0

def TotalGames : Nat := 15

def GamesMissed (name : String) : Nat :=
if name = "Petya" then TotalGames - MatchesPlayed name else 
if name = "Sasha" then TotalGames - MatchesPlayed name else
if name = "Misha" then TotalGames - MatchesPlayed name else 0

def CanNotMissConsecutiveGames : Prop := True

theorem PetyaWinsAgainstSasha : (GamesMissed "Misha" = 4) ∧ CanNotMissConsecutiveGames → 
  ∃ (winsByPetya : Nat), winsByPetya = 4 :=
by
  sorry

end PetyaWinsAgainstSasha_l49_49049


namespace tan_alpha_through_point_l49_49518

theorem tan_alpha_through_point (α : ℝ) (x y : ℝ) (h : (x, y) = (3, 4)) : Real.tan α = 4 / 3 :=
sorry

end tan_alpha_through_point_l49_49518


namespace arccos_one_over_sqrt_two_l49_49546

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l49_49546


namespace Bernoulli_inequality_l49_49657

theorem Bernoulli_inequality (p : ℝ) (k : ℚ) (hp : 0 < p) (hk : 1 < k) : 
  (1 + p) ^ (k : ℝ) > 1 + p * (k : ℝ) := by
sorry

end Bernoulli_inequality_l49_49657


namespace arccos_zero_eq_pi_div_two_l49_49977

theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l49_49977


namespace other_train_length_l49_49524

-- Define a theorem to prove that the length of the other train (L) is 413.95 meters
theorem other_train_length (length_first_train : ℝ) (speed_first_train_kmph : ℝ) 
                           (speed_second_train_kmph: ℝ) (time_crossing_seconds : ℝ) : 
                           length_first_train = 350 → 
                           speed_first_train_kmph = 150 →
                           speed_second_train_kmph = 100 →
                           time_crossing_seconds = 11 →
                           ∃ (L : ℝ), L = 413.95 :=
by
  intros h1 h2 h3 h4
  sorry

end other_train_length_l49_49524


namespace speed_ratio_l49_49046

theorem speed_ratio (L v_a v_b : ℝ) (h1 : v_a = c * v_b) (h2 : (L / v_a) = (0.8 * L / v_b)) :
  v_a / v_b = 5 / 4 :=
by
  sorry

end speed_ratio_l49_49046


namespace find_particular_number_l49_49975

theorem find_particular_number (x : ℤ) (h : x - 29 + 64 = 76) : x = 41 :=
by
  sorry

end find_particular_number_l49_49975


namespace commutating_matrices_l49_49488

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=  ![![2, 3], ![4, 5]]
noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem commutating_matrices (x y z w : ℝ) (h1 : A * (B x y z w) = (B x y z w) * A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 / 2 := 
by
  sorry

end commutating_matrices_l49_49488


namespace sqrt_five_gt_two_l49_49597

theorem sqrt_five_gt_two : Real.sqrt 5 > 2 :=
by
  -- Proof goes here
  sorry

end sqrt_five_gt_two_l49_49597


namespace total_practice_hours_correct_l49_49788

-- Define the conditions
def daily_practice_hours : ℕ := 5 -- The team practices 5 hours daily
def missed_days : ℕ := 1 -- They missed practicing 1 day this week
def days_in_week : ℕ := 7 -- There are 7 days in a week

-- Calculate the number of days they practiced
def practiced_days : ℕ := days_in_week - missed_days

-- Calculate the total hours practiced
def total_practice_hours : ℕ := practiced_days * daily_practice_hours

-- Theorem to prove the total hours practiced is 30
theorem total_practice_hours_correct : total_practice_hours = 30 := by
  -- Start the proof; skipping the actual proof steps
  sorry

end total_practice_hours_correct_l49_49788


namespace least_square_of_conditions_l49_49455

theorem least_square_of_conditions :
  ∃ (a x y : ℕ), 0 < a ∧ 0 < x ∧ 0 < y ∧ 
  (15 * a + 165 = x^2) ∧ 
  (16 * a - 155 = y^2) ∧ 
  (min (x^2) (y^2) = 481) := 
sorry

end least_square_of_conditions_l49_49455


namespace problem_value_l49_49509

theorem problem_value :
  4 * (8 - 3) / 2 - 7 = 3 := 
by
  sorry

end problem_value_l49_49509


namespace composite_expression_l49_49701

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^3 + 6 * n^2 + 12 * n + 7 = a * b :=
by
  sorry

end composite_expression_l49_49701


namespace particle_speed_interval_l49_49530

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 7)

theorem particle_speed_interval (k : ℝ) :
  let start_pos := particle_position k
  let end_pos := particle_position (k + 2)
  let delta_x := end_pos.1 - start_pos.1
  let delta_y := end_pos.2 - start_pos.2
  let speed := Real.sqrt (delta_x^2 + delta_y^2)
  speed = 2 * Real.sqrt 34 := by
  sorry

end particle_speed_interval_l49_49530


namespace general_term_formula_l49_49755

noncomputable def xSeq : ℕ → ℝ
| 0       => 3
| (n + 1) => (xSeq n)^2 + 2 / (2 * (xSeq n) - 1)

theorem general_term_formula (n : ℕ) : 
  xSeq n = (2 * 2^2^n + 1) / (2^2^n - 1) := 
sorry

end general_term_formula_l49_49755


namespace billy_horses_l49_49992

theorem billy_horses (each_horse_oats_per_meal : ℕ) (meals_per_day : ℕ) (total_oats_needed : ℕ) (days : ℕ) 
    (h_each_horse_oats_per_meal : each_horse_oats_per_meal = 4)
    (h_meals_per_day : meals_per_day = 2)
    (h_total_oats_needed : total_oats_needed = 96)
    (h_days : days = 3) :
    (total_oats_needed / (days * (each_horse_oats_per_meal * meals_per_day)) = 4) :=
by
  sorry

end billy_horses_l49_49992


namespace set_union_inter_example_l49_49697

open Set

theorem set_union_inter_example :
  let A := ({1, 2} : Set ℕ)
  let B := ({1, 2, 3} : Set ℕ)
  let C := ({2, 3, 4} : Set ℕ)
  (A ∩ B) ∪ C = ({1, 2, 3, 4} : Set ℕ) := by
    let A := ({1, 2} : Set ℕ)
    let B := ({1, 2, 3} : Set ℕ)
    let C := ({2, 3, 4} : Set ℕ)
    sorry

end set_union_inter_example_l49_49697


namespace min_num_edges_chromatic_l49_49870

-- Definition of chromatic number.
def chromatic_number (G : SimpleGraph V) : ℕ := sorry

-- Definition of the number of edges in a graph as a function.
def num_edges (G : SimpleGraph V) : ℕ := sorry

-- Statement of the theorem.
theorem min_num_edges_chromatic (G : SimpleGraph V) (n : ℕ) 
  (chrom_num_G : chromatic_number G = n) : 
  num_edges G ≥ n * (n - 1) / 2 :=
sorry

end min_num_edges_chromatic_l49_49870


namespace value_two_sd_below_mean_l49_49291

theorem value_two_sd_below_mean :
  let mean := 14.5
  let stdev := 1.7
  mean - 2 * stdev = 11.1 :=
by
  sorry

end value_two_sd_below_mean_l49_49291


namespace geometric_sequence_a1_value_l49_49423

variable {a_1 q : ℝ}

theorem geometric_sequence_a1_value
  (h1 : a_1 * q^2 = 1)
  (h2 : a_1 * q^4 + (3 / 2) * a_1 * q^3 = 1) :
  a_1 = 4 := by
  sorry

end geometric_sequence_a1_value_l49_49423


namespace xy_value_l49_49823

variable (x y : ℕ)

def condition1 : Prop := 8^x / 4^(x + y) = 16
def condition2 : Prop := 16^(x + y) / 4^(7 * y) = 256

theorem xy_value (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 48 := by
  sorry

end xy_value_l49_49823


namespace op_4_6_l49_49928

-- Define the operation @ in Lean
def op (a b : ℕ) : ℤ := 2 * (a : ℤ)^2 - 2 * (b : ℤ)^2

-- State the theorem to prove
theorem op_4_6 : op 4 6 = -40 :=
by sorry

end op_4_6_l49_49928


namespace operation_example_l49_49735

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end operation_example_l49_49735


namespace treasure_hunt_distance_l49_49708

theorem treasure_hunt_distance (d : ℝ) : 
  (d < 8) → (d > 7) → (d > 9) → False :=
by
  intros h1 h2 h3
  sorry

end treasure_hunt_distance_l49_49708


namespace smallest_n_for_inequality_l49_49028

theorem smallest_n_for_inequality :
  ∃ n : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧ 
    (∀ m : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ m * (w^6 + x^6 + y^6 + z^6)) → m ≥ 64) :=
by
  sorry

end smallest_n_for_inequality_l49_49028


namespace constant_term_expansion_l49_49572

theorem constant_term_expansion : 
  ∃ r : ℕ, (9 - 3 * r / 2 = 0) ∧ 
  ∀ (x : ℝ) (hx : x ≠ 0), (2 * x - 1 / Real.sqrt x) ^ 9 = 672 := 
by sorry

end constant_term_expansion_l49_49572


namespace common_difference_l49_49416

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : a 2 = 1 + d) (h4 : a 4 = 1 + 3 * d) (h5 : a 5 = 1 + 4 * d) 
  (h_geometric : (a 4)^2 = a 2 * a 5) 
  (h_nonzero : d ≠ 0) : 
  d = 1 / 5 :=
by sorry

end common_difference_l49_49416


namespace Genevieve_drinks_pints_l49_49395

theorem Genevieve_drinks_pints :
  ∀ (total_gallons : ℝ) (num_thermoses : ℕ) (pints_per_gallon : ℝ) (genevieve_thermoses : ℕ),
  total_gallons = 4.5 → num_thermoses = 18 → pints_per_gallon = 8 → genevieve_thermoses = 3 →
  (genevieve_thermoses * ((total_gallons / num_thermoses) * pints_per_gallon) = 6) :=
by
  intros total_gallons num_thermoses pints_per_gallon genevieve_thermoses
  intros h1 h2 h3 h4
  sorry

end Genevieve_drinks_pints_l49_49395


namespace volume_of_cube_l49_49564

theorem volume_of_cube (a : ℕ) (h : ((a - 2) * a * (a + 2)) = a^3 - 16) : a^3 = 64 :=
sorry

end volume_of_cube_l49_49564


namespace water_left_after_experiment_l49_49660

theorem water_left_after_experiment (initial_water : ℝ) (used_water : ℝ) (result_water : ℝ) 
  (h1 : initial_water = 3) 
  (h2 : used_water = 9 / 4) 
  (h3 : result_water = 3 / 4) : 
  initial_water - used_water = result_water := by
  sorry

end water_left_after_experiment_l49_49660


namespace cos_diff_angle_l49_49196

theorem cos_diff_angle
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (h : 3 * Real.sin α = Real.tan α) :
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 :=
sorry

end cos_diff_angle_l49_49196


namespace quadratic_equation_solutions_l49_49622

theorem quadratic_equation_solutions (x : ℝ) : x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 :=
by
  sorry

end quadratic_equation_solutions_l49_49622


namespace inequality_proof_l49_49847

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l49_49847


namespace find_abs_ab_l49_49047

def ellipse_foci_distance := 5
def hyperbola_foci_distance := 7

def ellipse_condition (a b : ℝ) := b^2 - a^2 = ellipse_foci_distance^2
def hyperbola_condition (a b : ℝ) := a^2 + b^2 = hyperbola_foci_distance^2

theorem find_abs_ab (a b : ℝ) (h_ellipse : ellipse_condition a b) (h_hyperbola : hyperbola_condition a b) :
  |a * b| = 2 * Real.sqrt 111 :=
by
  sorry

end find_abs_ab_l49_49047


namespace conditional_probabilities_l49_49724

def PA : ℝ := 0.20
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

theorem conditional_probabilities :
  PAB / PB = 2 / 3 ∧ PAB / PA = 3 / 5 := by
  sorry

end conditional_probabilities_l49_49724


namespace ratio_of_rooms_l49_49872

theorem ratio_of_rooms (rooms_danielle : ℕ) (rooms_grant : ℕ) (ratio_grant_heidi : ℚ)
  (h1 : rooms_danielle = 6)
  (h2 : rooms_grant = 2)
  (h3 : ratio_grant_heidi = 1/9) :
  (18 : ℚ) / rooms_danielle = 3 :=
by
  sorry

end ratio_of_rooms_l49_49872


namespace quadratic_equation_completing_square_l49_49797

theorem quadratic_equation_completing_square :
  ∃ a b c : ℤ, a > 0 ∧ (25 * x^2 + 30 * x - 75 = 0 → (a * x + b)^2 = c) ∧ a + b + c = -58 :=
  sorry

end quadratic_equation_completing_square_l49_49797


namespace train_length_is_150_l49_49169

noncomputable def train_length_crossing_post (t_post : ℕ := 10) : ℕ := 10
noncomputable def train_length_crossing_platform (length_platform : ℕ := 150) (t_platform : ℕ := 20) : ℕ := 20
def train_constant_speed (L v : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) : Prop :=
  v = L / t_post ∧ v = (L + length_platform) / t_platform

theorem train_length_is_150 (L : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) (H : train_constant_speed L v t_post t_platform length_platform) : 
  L = 150 :=
by
  sorry

end train_length_is_150_l49_49169


namespace cubes_of_roots_l49_49278

theorem cubes_of_roots (a b c : ℝ) (h1 : a + b + c = 2) (h2 : ab + ac + bc = 2) (h3 : abc = 3) : 
  a^3 + b^3 + c^3 = 9 :=
by
  sorry

end cubes_of_roots_l49_49278


namespace union_sets_l49_49652

namespace Proof

def setA : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
sorry

end Proof

end union_sets_l49_49652


namespace proof_4_minus_a_l49_49069

theorem proof_4_minus_a :
  ∀ (a b : ℚ),
    (5 + a = 7 - b) →
    (3 + b = 8 + a) →
    4 - a = 11 / 2 :=
by
  intros a b h1 h2
  sorry

end proof_4_minus_a_l49_49069


namespace fence_poles_placement_l49_49506

def total_bridges_length (bridges : List ℕ) : ℕ :=
  bridges.sum

def effective_path_length (path_length : ℕ) (bridges_length : ℕ) : ℕ :=
  path_length - bridges_length

def poles_on_one_side (effective_length : ℕ) (interval : ℕ) : ℕ :=
  effective_length / interval

def total_poles (path_length : ℕ) (interval : ℕ) (bridges : List ℕ) : ℕ :=
  let bridges_length := total_bridges_length bridges
  let effective_length := effective_path_length path_length bridges_length
  let poles_one_side := poles_on_one_side effective_length interval
  2 * poles_one_side + 2

theorem fence_poles_placement :
  total_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end fence_poles_placement_l49_49506


namespace total_amount_l49_49041

theorem total_amount (P Q R : ℝ) (h1 : R = 2 / 3 * (P + Q)) (h2 : R = 3200) : P + Q + R = 8000 := 
by
  sorry

end total_amount_l49_49041


namespace no_savings_if_purchased_together_l49_49298

def window_price : ℕ := 120

def free_windows (purchased_windows : ℕ) : ℕ :=
  (purchased_windows / 10) * 2

def total_cost (windows_needed : ℕ) : ℕ :=
  (windows_needed - free_windows windows_needed) * window_price

def separate_cost : ℕ :=
  total_cost 9 + total_cost 11 + total_cost 10

def joint_cost : ℕ :=
  total_cost 30

theorem no_savings_if_purchased_together :
  separate_cost = joint_cost :=
by
  -- Proof will be provided here, currently skipped.
  sorry

end no_savings_if_purchased_together_l49_49298


namespace segment_problem_l49_49274

theorem segment_problem 
  (A C : ℝ) (B D : ℝ) (P Q : ℝ) (x y k : ℝ)
  (hA : A = 0) (hC : C = 0) 
  (hB : B = 6) (hD : D = 9)
  (hx : x = P - A) (hy : y = Q - C) 
  (hxk : x = 3 * k)
  (hxyk : x + y = 12 * k) :
  k = 2 :=
  sorry

end segment_problem_l49_49274


namespace total_emails_received_l49_49623

theorem total_emails_received (emails_morning emails_afternoon : ℕ) 
  (h1 : emails_morning = 3) 
  (h2 : emails_afternoon = 5) : 
  emails_morning + emails_afternoon = 8 := 
by 
  sorry

end total_emails_received_l49_49623


namespace gcd_in_base3_l49_49812

def gcd_2134_1455_is_97 : ℕ :=
  gcd 2134 1455

def base3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) : List ℕ :=
      if n = 0 then [] else aux (n / 3) ++ [n % 3]
    aux n

theorem gcd_in_base3 :
  gcd_2134_1455_is_97 = 97 ∧ base3 97 = [1, 0, 1, 2, 1] :=
by
  sorry

end gcd_in_base3_l49_49812


namespace radius_of_sphere_touching_four_l49_49888

noncomputable def r_sphere_internally_touching_four := Real.sqrt (3 / 2) + 1
noncomputable def r_sphere_externally_touching_four := Real.sqrt (3 / 2) - 1

theorem radius_of_sphere_touching_four (r : ℝ) (R := Real.sqrt (3 / 2)) :
  r = R + 1 ∨ r = R - 1 :=
by
  sorry

end radius_of_sphere_touching_four_l49_49888


namespace unique_function_satisfying_conditions_l49_49636

theorem unique_function_satisfying_conditions (f : ℤ → ℤ) :
  (∀ n : ℤ, f (f n) + f n = 2 * n + 3) → 
  (f 0 = 1) → 
  (∀ n : ℤ, f n = n + 1) :=
by
  intro h1 h2
  sorry

end unique_function_satisfying_conditions_l49_49636


namespace factorize_diff_of_squares_l49_49679

theorem factorize_diff_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
  sorry

end factorize_diff_of_squares_l49_49679


namespace unique_point_value_l49_49959

noncomputable def unique_point_condition : Prop :=
  ∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + 12 = 0

theorem unique_point_value (d : ℝ) : unique_point_condition ↔ d = 12 := 
sorry

end unique_point_value_l49_49959


namespace ratio_proof_l49_49800

theorem ratio_proof (a b c d : ℝ) (h1 : a / b = 20) (h2 : c / b = 5) (h3 : c / d = 1 / 8) : 
  a / d = 1 / 2 :=
by
  sorry

end ratio_proof_l49_49800


namespace initial_weight_l49_49342

theorem initial_weight (W : ℝ) (current_weight : ℝ) (future_weight : ℝ) (months : ℝ) (additional_months : ℝ) 
  (constant_rate : Prop) :
  current_weight = 198 →
  future_weight = 170 →
  months = 3 →
  additional_months = 3.5 →
  constant_rate →
  W = 222 :=
by
  intros h_current_weight h_future_weight h_months h_additional_months h_constant_rate
  -- proof would go here
  sorry

end initial_weight_l49_49342


namespace sequence_contains_composite_l49_49935

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem sequence_contains_composite (a : ℕ → ℕ) (h : ∀ n, a (n+1) = 2 * a n + 1 ∨ a (n+1) = 2 * a n - 1) :
  ∃ n, is_composite (a n) :=
sorry

end sequence_contains_composite_l49_49935


namespace graph_passes_through_point_l49_49268

theorem graph_passes_through_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
    ∃ y : ℝ, y = a^0 + 1 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end graph_passes_through_point_l49_49268


namespace tom_first_part_speed_l49_49632

theorem tom_first_part_speed 
  (total_distance : ℕ)
  (distance_first_part : ℕ)
  (speed_second_part : ℕ)
  (average_speed : ℕ)
  (total_time : ℕ)
  (distance_remaining : ℕ)
  (T2 : ℕ)
  (v : ℕ) :
  total_distance = 80 →
  distance_first_part = 30 →
  speed_second_part = 50 →
  average_speed = 40 →
  total_time = 2 →
  distance_remaining = 50 →
  T2 = 1 →
  total_time = distance_first_part / v + T2 →
  v = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, we need to prove that v = 30 given the above conditions.
  sorry

end tom_first_part_speed_l49_49632


namespace geometric_sequence_a8_eq_pm1_l49_49510

variable {R : Type*} [LinearOrderedField R]

theorem geometric_sequence_a8_eq_pm1 :
  ∀ (a : ℕ → R), (∀ n : ℕ, ∃ r : R, r ≠ 0 ∧ a n = a 0 * r ^ n) → 
  (a 4 + a 12 = -3) ∧ (a 4 * a 12 = 1) → 
  (a 8 = 1 ∨ a 8 = -1) := by
  sorry

end geometric_sequence_a8_eq_pm1_l49_49510


namespace cuboid_volume_l49_49159

theorem cuboid_volume (length width height : ℕ) (h_length : length = 4) (h_width : width = 4) (h_height : height = 6) : (length * width * height = 96) :=
by 
  -- Sorry places a placeholder for the actual proof
  sorry

end cuboid_volume_l49_49159


namespace trig_eq_solutions_l49_49155

theorem trig_eq_solutions (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  3 * Real.sin x = 1 + Real.cos (2 * x) ↔ x = Real.pi / 6 ∨ x = 5 * Real.pi / 6 :=
by
  sorry

end trig_eq_solutions_l49_49155


namespace find_acute_angle_as_pi_over_4_l49_49206
open Real

-- Definitions from the problem's conditions
variables (x : ℝ)
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2
def trig_eq (x : ℝ) : Prop := (sin x) ^ 3 + (cos x) ^ 3 = sqrt 2 / 2

-- The math proof problem statement
theorem find_acute_angle_as_pi_over_4 (h_acute : is_acute x) (h_trig_eq : trig_eq x) : x = π / 4 := 
sorry

end find_acute_angle_as_pi_over_4_l49_49206


namespace fraction_pos_integer_l49_49363

theorem fraction_pos_integer (p : ℕ) (hp : 0 < p) : (∃ (k : ℕ), k = 1 + (2 * p + 53) / (3 * p - 8)) ↔ p = 3 := 
by
  sorry

end fraction_pos_integer_l49_49363


namespace total_distance_in_land_miles_l49_49863

-- Definitions based on conditions
def speed_one_sail : ℕ := 25
def time_one_sail : ℕ := 4
def distance_one_sail := speed_one_sail * time_one_sail

def speed_two_sails : ℕ := 50
def time_two_sails : ℕ := 4
def distance_two_sails := speed_two_sails * time_two_sails

def conversion_factor : ℕ := 115  -- Note: 1.15 * 100 for simplicity with integers

-- Theorem to prove the total distance in land miles
theorem total_distance_in_land_miles : (distance_one_sail + distance_two_sails) * conversion_factor / 100 = 345 := by
  sorry

end total_distance_in_land_miles_l49_49863


namespace line_form_l49_49923

-- Given vector equation for a line
def line_eq (x y : ℝ) : Prop :=
  (3 * (x - 4) + 7 * (y - 14)) = 0

-- Prove that the line can be written in the form y = mx + b
theorem line_form (x y : ℝ) (h : line_eq x y) :
  y = (-3/7) * x + (110/7) :=
sorry

end line_form_l49_49923


namespace triangle_inequality_l49_49716

theorem triangle_inequality (a b c p S r : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b)
  (hp : p = (a + b + c) / 2)
  (hS : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (hr : r = S / p):
  1 / (p - a) ^ 2 + 1 / (p - b) ^ 2 + 1 / (p - c) ^ 2 ≥ 1 / r ^ 2 :=
sorry

end triangle_inequality_l49_49716


namespace find_ab_l49_49531

variables (a b c : ℝ)

-- Defining the conditions
def cond1 : Prop := a - b = 5
def cond2 : Prop := a^2 + b^2 = 34
def cond3 : Prop := a^3 - b^3 = 30
def cond4 : Prop := a^2 + b^2 - c^2 = 50

theorem find_ab (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 a b) (h4 : cond4 a b c) :
  a * b = 4.5 :=
sorry

end find_ab_l49_49531


namespace triangle_OAB_area_range_l49_49424

noncomputable def area_of_triangle_OAB (m : ℝ) : ℝ :=
  4 * Real.sqrt (64 * m^2 + 4 * 64)

theorem triangle_OAB_area_range :
  ∀ m : ℝ, 64 ≤ area_of_triangle_OAB m :=
by
  intro m
  sorry

end triangle_OAB_area_range_l49_49424


namespace next_bell_ringing_time_l49_49713

theorem next_bell_ringing_time (post_office_interval train_station_interval town_hall_interval start_time : ℕ)
  (h1 : post_office_interval = 18)
  (h2 : train_station_interval = 24)
  (h3 : town_hall_interval = 30)
  (h4 : start_time = 9) :
  let lcm := Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval)
  lcm + start_time = 15 := by
  sorry

end next_bell_ringing_time_l49_49713


namespace necessary_not_sufficient_condition_t_for_b_l49_49345

variable (x y : ℝ)

def condition_t : Prop := x ≤ 12 ∨ y ≤ 16
def condition_b : Prop := x + y ≤ 28 ∨ x * y ≤ 192

theorem necessary_not_sufficient_condition_t_for_b (h : condition_b x y) : condition_t x y ∧ ¬ (condition_t x y → condition_b x y) := by
  sorry

end necessary_not_sufficient_condition_t_for_b_l49_49345


namespace consecutive_lucky_years_l49_49932

def is_lucky (Y : ℕ) : Prop := 
  let first_two_digits := Y / 100
  let last_two_digits := Y % 100
  Y % (first_two_digits + last_two_digits) = 0

theorem consecutive_lucky_years : ∃ Y : ℕ, is_lucky Y ∧ is_lucky (Y + 1) :=
by
  sorry

end consecutive_lucky_years_l49_49932


namespace find_numbers_l49_49233

theorem find_numbers (a b c : ℝ) (x y z: ℝ) (h1 : x + y = z + a) (h2 : x + z = y + b) (h3 : y + z = x + c) :
    x = (a + b - c) / 2 ∧ y = (a - b + c) / 2 ∧ z = (-a + b + c) / 2 := by
  sorry

end find_numbers_l49_49233


namespace height_of_given_cylinder_l49_49930

noncomputable def height_of_cylinder (P d : ℝ) : ℝ :=
  let r := P / (2 * Real.pi)
  let l := P
  let h := Real.sqrt (d^2 - l^2)
  h

theorem height_of_given_cylinder : height_of_cylinder 6 10 = 8 :=
by
  show height_of_cylinder 6 10 = 8
  sorry

end height_of_given_cylinder_l49_49930


namespace inequality_for_positive_integer_l49_49849

theorem inequality_for_positive_integer (n : ℕ) (h : n > 0) :
  n^n ≤ (n!)^2 ∧ (n!)^2 ≤ ((n + 1) * (n + 2) / 6)^n := by
  sorry

end inequality_for_positive_integer_l49_49849


namespace percent_increase_l49_49750

variable (P : ℝ)
def firstQuarterPrice := 1.20 * P
def secondQuarterPrice := 1.50 * P

theorem percent_increase:
  ((secondQuarterPrice P - firstQuarterPrice P) / firstQuarterPrice P) * 100 = 25 := by
  sorry

end percent_increase_l49_49750


namespace rainfall_comparison_l49_49469

-- Define the conditions
def rainfall_mondays (n_mondays : ℕ) (rain_monday : ℝ) : ℝ :=
  n_mondays * rain_monday

def rainfall_tuesdays (n_tuesdays : ℕ) (rain_tuesday : ℝ) : ℝ :=
  n_tuesdays * rain_tuesday

def rainfall_difference (total_monday : ℝ) (total_tuesday : ℝ) : ℝ :=
  total_tuesday - total_monday

-- The proof statement
theorem rainfall_comparison :
  rainfall_difference (rainfall_mondays 13 1.75) (rainfall_tuesdays 16 2.65) = 19.65 := by
  sorry

end rainfall_comparison_l49_49469


namespace sum_consecutive_even_l49_49184

theorem sum_consecutive_even (m : ℤ) : m + (m + 2) + (m + 4) + (m + 6) + (m + 8) + (m + 10) = 6 * m + 30 :=
by
  sorry

end sum_consecutive_even_l49_49184


namespace pedestrians_speed_ratio_l49_49061

-- Definitions based on conditions
variable (v v1 v2 : ℝ)

-- Conditions
def first_meeting (v1 v : ℝ) := (1 / 3) * v1 = (1 / 4) * v
def second_meeting (v2 v : ℝ) := (5 / 12) * v2 = (1 / 6) * v

-- Theorem Statement
theorem pedestrians_speed_ratio (h1 : first_meeting v1 v) (h2 : second_meeting v2 v) : v1 / v2 = 15 / 8 :=
by
  -- Proof will go here
  sorry

end pedestrians_speed_ratio_l49_49061


namespace arithmetic_sequence_middle_term_l49_49968

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l49_49968


namespace ordered_pairs_of_positive_integers_l49_49595

theorem ordered_pairs_of_positive_integers (x y : ℕ) (h : x * y = 2800) :
  2^4 * 5^2 * 7 = 2800 → ∃ (n : ℕ), n = 30 ∧ (∃ x y : ℕ, x * y = 2800 ∧ n = 30) :=
by
  sorry

end ordered_pairs_of_positive_integers_l49_49595


namespace probability_value_expr_is_7_l49_49157

theorem probability_value_expr_is_7 : 
  let num_ones : ℕ := 15
  let num_ops : ℕ := 14
  let target_value : ℤ := 7
  let total_ways := 2 ^ num_ops
  let favorable_ways := (Nat.choose num_ops 11)  -- Ways to choose positions for +1's
  let prob := (favorable_ways : ℝ) / total_ways
  prob = 91 / 4096 := sorry

end probability_value_expr_is_7_l49_49157


namespace greatest_possible_y_l49_49248

theorem greatest_possible_y (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : y ≤ 18 :=
sorry

end greatest_possible_y_l49_49248


namespace smallest_a_divisible_by_65_l49_49438

theorem smallest_a_divisible_by_65 (a : ℤ) 
  (h : ∀ (n : ℤ), (5 * n ^ 13 + 13 * n ^ 5 + 9 * a * n) % 65 = 0) : 
  a = 63 := 
by {
  sorry
}

end smallest_a_divisible_by_65_l49_49438


namespace odd_and_increasing_function_l49_49162

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f (x) ≤ f (y)

def function_D (x : ℝ) : ℝ := x * abs x

theorem odd_and_increasing_function : 
  (is_odd function_D) ∧ (is_increasing function_D) :=
sorry

end odd_and_increasing_function_l49_49162


namespace geometric_arithmetic_sequences_sum_l49_49673

theorem geometric_arithmetic_sequences_sum (a b : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (q d : ℝ) (h1 : 0 < q) 
  (h2 : a 1 = 1) (h3 : b 1 = 1) 
  (h4 : a 5 + b 3 = 21) 
  (h5 : a 3 + b 5 = 13) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2*n - 1) ∧ (∀ n, S_n n = 3 - (2*n + 3)/(2^n)) := 
sorry

end geometric_arithmetic_sequences_sum_l49_49673


namespace trapezoid_diagonal_comparison_l49_49253

variable {A B C D: Type}
variable (α β : Real) -- Representing angles
variable (AB CD BD AC : Real) -- Representing lengths of sides and diagonals
variable (h : Real) -- Height
variable (A' B' : Real) -- Projections

noncomputable def trapezoid (AB CD: Real) := True -- Trapezoid definition placeholder
noncomputable def angle_relation (α β : Real) := α < β -- Angle relationship

theorem trapezoid_diagonal_comparison
  (trapezoid_ABCD: trapezoid AB CD)
  (angle_relation_ABC_DCB : angle_relation α β)
  : BD > AC :=
sorry

end trapezoid_diagonal_comparison_l49_49253


namespace sara_initial_pears_l49_49665

theorem sara_initial_pears (given_to_dan : ℕ) (left_with_sara : ℕ) (total : ℕ) :
  given_to_dan = 28 ∧ left_with_sara = 7 ∧ total = given_to_dan + left_with_sara → total = 35 :=
by
  sorry

end sara_initial_pears_l49_49665


namespace triangle_inequality_expression_non_negative_l49_49261

theorem triangle_inequality_expression_non_negative
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end triangle_inequality_expression_non_negative_l49_49261


namespace smallest_integer_min_value_l49_49742

theorem smallest_integer_min_value :
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
  B ≠ C ∧ B ≠ D ∧ 
  C ≠ D ∧ 
  (A + B + C + D) = 288 ∧ 
  D = 90 ∧ 
  (A = 21) := 
sorry

end smallest_integer_min_value_l49_49742


namespace necessary_but_not_sufficient_l49_49512

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  -2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0

theorem necessary_but_not_sufficient (x : ℝ) : 
-2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0 := 
by
  sorry

end necessary_but_not_sufficient_l49_49512


namespace cid_earnings_l49_49591

theorem cid_earnings :
  let model_a_oil_change_cost := 20
  let model_a_repair_cost := 30
  let model_a_wash_cost := 5
  let model_b_oil_change_cost := 25
  let model_b_repair_cost := 40
  let model_b_wash_cost := 8
  let model_c_oil_change_cost := 30
  let model_c_repair_cost := 50
  let model_c_wash_cost := 10

  let model_a_oil_changes := 5
  let model_a_repairs := 10
  let model_a_washes := 15
  let model_b_oil_changes := 3
  let model_b_repairs := 4
  let model_b_washes := 10
  let model_c_oil_changes := 2
  let model_c_repairs := 6
  let model_c_washes := 5

  let total_earnings := 
      (model_a_oil_change_cost * model_a_oil_changes) +
      (model_a_repair_cost * model_a_repairs) +
      (model_a_wash_cost * model_a_washes) +
      (model_b_oil_change_cost * model_b_oil_changes) +
      (model_b_repair_cost * model_b_repairs) +
      (model_b_wash_cost * model_b_washes) +
      (model_c_oil_change_cost * model_c_oil_changes) +
      (model_c_repair_cost * model_c_repairs) +
      (model_c_wash_cost * model_c_washes)

  total_earnings = 1200 := by
  sorry

end cid_earnings_l49_49591


namespace problem_1_problem_2_l49_49644

-- Condition for Question 1
def f (x : ℝ) (a : ℝ) := |x - a|

-- Proof Problem for Question 1
theorem problem_1 (a : ℝ) (h : a = 1) : {x : ℝ | f x a > 1/2 * (x + 1)} = {x | x > 3 ∨ x < 1/3} :=
sorry

-- Condition for Question 2
def g (x : ℝ) (a : ℝ) := |x - a| + |x - 2|

-- Proof Problem for Question 2
theorem problem_2 (a : ℝ) : (∃ x : ℝ, g x a ≤ 3) → (-1 ≤ a ∧ a ≤ 5) :=
sorry

end problem_1_problem_2_l49_49644


namespace math_problem_statements_are_correct_l49_49856

theorem math_problem_statements_are_correct (a b : ℝ) (h : a > b ∧ b > 0) :
  (¬ (b / a > (b + 3) / (a + 3))) ∧ ((3 * a + 2 * b) / (2 * a + 3 * b) < a / b) ∧
  (¬ (2 * Real.sqrt a < Real.sqrt (a - b) + Real.sqrt b)) ∧ 
  (Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2) :=
by
  sorry

end math_problem_statements_are_correct_l49_49856


namespace paperboy_delivery_count_l49_49757

def no_miss_four_consecutive (n : ℕ) (E : ℕ → ℕ) : Prop :=
  ∀ k > 3, E k = E (k - 1) + E (k - 2) + E (k - 3)

def base_conditions (E : ℕ → ℕ) : Prop :=
  E 1 = 2 ∧ E 2 = 4 ∧ E 3 = 8

theorem paperboy_delivery_count : ∃ (E : ℕ → ℕ), 
  base_conditions E ∧ no_miss_four_consecutive 12 E ∧ E 12 = 1854 :=
by
  sorry

end paperboy_delivery_count_l49_49757


namespace jogging_walking_ratio_l49_49519

theorem jogging_walking_ratio (total_time walk_time jog_time: ℕ) (h1 : total_time = 21) (h2 : walk_time = 9) (h3 : jog_time = total_time - walk_time) :
  (jog_time : ℚ) / walk_time = 4 / 3 :=
by
  sorry

end jogging_walking_ratio_l49_49519


namespace sum_outer_equal_sum_inner_l49_49179

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem sum_outer_equal_sum_inner (M N : ℕ) (a b c d : ℕ) 
  (h1 : is_four_digit M)
  (h2 : M = 1000 * a + 100 * b + 10 * c + d) 
  (h3 : N = reverse_digits M) 
  (h4 : M + N % 101 = 0) 
  (h5 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  a + d = b + c :=
  sorry

end sum_outer_equal_sum_inner_l49_49179


namespace non_zero_digits_of_fraction_l49_49344

def fraction : ℚ := 80 / (2^4 * 5^9)

def decimal_expansion (x : ℚ) : String :=
  -- some function to compute the decimal expansion of a fraction as a string
  "0.00000256" -- placeholder

def non_zero_digits_to_right (s : String) : ℕ :=
  -- some function to count the number of non-zero digits to the right of the decimal point in the string
  3 -- placeholder

theorem non_zero_digits_of_fraction : non_zero_digits_to_right (decimal_expansion fraction) = 3 := by
  sorry

end non_zero_digits_of_fraction_l49_49344


namespace white_rabbit_hop_distance_per_minute_l49_49270

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end white_rabbit_hop_distance_per_minute_l49_49270


namespace corn_height_after_three_weeks_l49_49839

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l49_49839


namespace curves_intersection_four_points_l49_49503

theorem curves_intersection_four_points (b : ℝ) :
  (∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
    x1^2 + y1^2 = b^2 ∧ y1 = x1^2 - b + 1 ∧
    x2^2 + y2^2 = b^2 ∧ y2 = x2^2 - b + 1 ∧
    x3^2 + y3^2 = b^2 ∧ y3 = x3^2 - b + 1 ∧
    x4^2 + y4^2 = b^2 ∧ y4 = x4^2 - b + 1 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧
    (x2, y2) ≠ (x3, y3) ∧ (x2, y2) ≠ (x4, y4) ∧
    (x3, y3) ≠ (x4, y4)) →
  b > 2 :=
sorry

end curves_intersection_four_points_l49_49503


namespace cat_mouse_position_after_300_moves_l49_49315

def move_pattern_cat_mouse :=
  let cat_cycle_length := 4
  let mouse_cycle_length := 8
  let cat_moves := 300
  let mouse_moves := (3 / 2) * cat_moves
  let cat_position := (cat_moves % cat_cycle_length)
  let mouse_position := (mouse_moves % mouse_cycle_length)
  (cat_position, mouse_position)

theorem cat_mouse_position_after_300_moves :
  move_pattern_cat_mouse = (0, 2) :=
by
  sorry

end cat_mouse_position_after_300_moves_l49_49315


namespace quadratic_two_distinct_real_roots_l49_49533

theorem quadratic_two_distinct_real_roots (m : ℝ) (h : -4 * m > 0) : m = -1 :=
sorry

end quadratic_two_distinct_real_roots_l49_49533


namespace regular_polygon_sides_l49_49014

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ m : ℕ, m = 360 / n → n ≠ 0 → m = 30) : n = 12 :=
  sorry

end regular_polygon_sides_l49_49014


namespace escalator_length_l49_49272

theorem escalator_length
  (escalator_speed : ℕ)
  (person_speed : ℕ)
  (time_taken : ℕ)
  (combined_speed : ℕ)
  (condition1 : escalator_speed = 12)
  (condition2 : person_speed = 2)
  (condition3 : time_taken = 14)
  (condition4 : combined_speed = escalator_speed + person_speed)
  (condition5 : combined_speed * time_taken = 196) :
  combined_speed * time_taken = 196 := 
by
  -- the proof would go here
  sorry

end escalator_length_l49_49272


namespace melanie_attended_games_l49_49473

theorem melanie_attended_games 
(missed_games total_games attended_games : ℕ) 
(h1 : total_games = 64) 
(h2 : missed_games = 32)
(h3 : attended_games = total_games - missed_games) 
: attended_games = 32 :=
by sorry

end melanie_attended_games_l49_49473


namespace maximum_value_of_expression_is_4_l49_49669

noncomputable def maximimum_integer_value (x : ℝ) : ℝ :=
    (5 * x^2 + 10 * x + 12) / (5 * x^2 + 10 * x + 2)

theorem maximum_value_of_expression_is_4 :
    ∃ x : ℝ, ∀ y : ℝ, maximimum_integer_value y ≤ 4 ∧ maximimum_integer_value x = 4 := 
by 
  -- Proof omitted for now
  sorry

end maximum_value_of_expression_is_4_l49_49669


namespace white_longer_than_blue_l49_49351

noncomputable def whiteLineInches : ℝ := 7.666666666666667
noncomputable def blueLineInches : ℝ := 3.3333333333333335
noncomputable def inchToCm : ℝ := 2.54
noncomputable def cmToMm : ℝ := 10

theorem white_longer_than_blue :
  let whiteLineCm := whiteLineInches * inchToCm
  let blueLineCm := blueLineInches * inchToCm
  let differenceCm := whiteLineCm - blueLineCm
  let differenceMm := differenceCm * cmToMm
  differenceMm = 110.05555555555553 := by
  sorry

end white_longer_than_blue_l49_49351


namespace total_amount_shared_l49_49120

theorem total_amount_shared (J Jo B : ℝ) (r1 r2 r3 : ℝ)
  (H1 : r1 = 2) (H2 : r2 = 4) (H3 : r3 = 6) (H4 : J = 1600) (part_value : ℝ)
  (H5 : part_value = J / r1) (H6 : Jo = r2 * part_value) (H7 : B = r3 * part_value) :
  J + Jo + B = 9600 :=
sorry

end total_amount_shared_l49_49120


namespace operation_example_result_l49_49068

def myOperation (A B : ℕ) : ℕ := (A^2 + B^2) / 3

theorem operation_example_result : myOperation (myOperation 6 3) 9 = 102 := by
  sorry

end operation_example_result_l49_49068


namespace geometric_sum_sequence_l49_49850

theorem geometric_sum_sequence (n : ℕ) (a : ℕ → ℕ) (a1 : a 1 = 2) (a4 : a 4 = 16) :
    (∃ q : ℕ, a 2 = a 1 * q) → (∃ S_n : ℕ, S_n = 2 * (2 ^ n - 1)) :=
by
  sorry

end geometric_sum_sequence_l49_49850


namespace simplify_and_rationalize_denominator_l49_49095

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l49_49095


namespace sum_cubes_l49_49210

variables (a b : ℝ)
noncomputable def calculate_sum_cubes (a b : ℝ) : ℝ :=
a^3 + b^3

theorem sum_cubes (h1 : a + b = 11) (h2 : a * b = 21) : calculate_sum_cubes a b = 638 :=
by
  sorry

end sum_cubes_l49_49210


namespace range_of_c_l49_49824

theorem range_of_c (x y c : ℝ) (h1 : x^2 + (y - 2)^2 = 1) (h2 : x^2 + y^2 + c ≤ 0) : c ≤ -9 :=
by
  -- Proof goes here
  sorry

end range_of_c_l49_49824


namespace perpendicular_slope_l49_49083

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end perpendicular_slope_l49_49083


namespace post_height_l49_49017

-- Conditions
def spiral_path (circuit_per_rise rise_distance : ℝ) := ∀ (total_distance circ_circumference height : ℝ), 
  circuit_per_rise = total_distance / circ_circumference ∧ 
  height = circuit_per_rise * rise_distance

-- Given conditions
def cylinder_post : Prop := 
  ∀ (total_distance circ_circumference rise_distance : ℝ), 
    spiral_path (total_distance / circ_circumference) rise_distance ∧ 
    circ_circumference = 3 ∧ 
    rise_distance = 4 ∧ 
    total_distance = 12

-- Proof problem: Post height
theorem post_height : cylinder_post → ∃ height : ℝ, height = 16 := 
by sorry

end post_height_l49_49017


namespace evaluate_expression_l49_49884

noncomputable def cuberoot (x : ℝ) : ℝ := x ^ (1 / 3)

theorem evaluate_expression : 
  cuberoot (1 + 27) * cuberoot (1 + cuberoot 27) = cuberoot 112 := 
by 
  sorry

end evaluate_expression_l49_49884


namespace servings_needed_l49_49890

theorem servings_needed
  (pieces_per_serving : ℕ)
  (jared_consumption : ℕ)
  (three_friends_consumption : ℕ)
  (another_three_friends_consumption : ℕ)
  (last_four_friends_consumption : ℕ) : 
  pieces_per_serving = 60 →
  jared_consumption = 150 →
  three_friends_consumption = 3 * 80 →
  another_three_friends_consumption = 3 * 200 →
  last_four_friends_consumption = 4 * 100 →
  ∃ (s : ℕ), s = 24 :=
by
  intros
  sorry

end servings_needed_l49_49890


namespace count_squares_with_dot_l49_49456

theorem count_squares_with_dot (n : ℕ) (dot_center : (n = 5)) :
  n = 5 → ∃ k, k = 19 :=
by sorry

end count_squares_with_dot_l49_49456


namespace probability_all_vertical_faces_green_l49_49838

theorem probability_all_vertical_faces_green :
  let color_prob := (1 / 2 : ℚ)
  let total_arrangements := 2^6
  let valid_arrangements := 2 + 12 + 6
  ((valid_arrangements : ℚ) / total_arrangements) = 5 / 16 := by
  sorry

end probability_all_vertical_faces_green_l49_49838


namespace translate_line_upwards_l49_49934

-- Define the original line equation
def original_line_eq (x : ℝ) : ℝ := 3 * x - 3

-- Define the translation operation
def translate_upwards (y_translation : ℝ) (line_eq : ℝ → ℝ) (x : ℝ) : ℝ :=
  line_eq x + y_translation

-- Define the proof problem
theorem translate_line_upwards :
  ∀ (x : ℝ), translate_upwards 5 original_line_eq x = 3 * x + 2 :=
by
  intros x
  simp [translate_upwards, original_line_eq]
  sorry

end translate_line_upwards_l49_49934


namespace library_visit_period_l49_49324

noncomputable def dance_class_days := 6
noncomputable def karate_class_days := 12
noncomputable def common_days := 36

theorem library_visit_period (library_days : ℕ) 
  (hdance : ∀ (n : ℕ), n * dance_class_days = common_days)
  (hkarate : ∀ (n : ℕ), n * karate_class_days = common_days)
  (hcommon : ∀ (n : ℕ), n * library_days = common_days) : 
  library_days = 18 := 
sorry

end library_visit_period_l49_49324


namespace binomial_parameters_l49_49374

theorem binomial_parameters
  (n : ℕ) (p : ℚ)
  (hE : n * p = 12) (hD : n * p * (1 - p) = 2.4) :
  n = 15 ∧ p = 4 / 5 :=
by
  sorry

end binomial_parameters_l49_49374


namespace apple_equals_pear_l49_49944

-- Define the masses of the apple and pear.
variable (A G : ℝ)

-- The equilibrium condition on the balance scale.
axiom equilibrium_condition : A + 2 * G = 2 * A + G

-- Prove the mass of an apple equals the mass of a pear.
theorem apple_equals_pear (A G : ℝ) (h : A + 2 * G = 2 * A + G) : A = G :=
by
  -- Proof goes here, but we use sorry to indicate the proof's need.
  sorry

end apple_equals_pear_l49_49944


namespace log2_a_div_b_squared_l49_49037

variable (a b : ℝ)
variable (ha_ne_1 : a ≠ 1) (hb_ne_1 : b ≠ 1)
variable (ha_pos : 0 < a) (hb_pos : 0 < b)
variable (h1 : 2 ^ (Real.log 32 / Real.log b) = a)
variable (h2 : a * b = 128)

theorem log2_a_div_b_squared :
  (Real.log ((a / b) : ℝ) / Real.log 2) ^ 2 = 29 + (49 / 4) :=
sorry

end log2_a_div_b_squared_l49_49037


namespace solution_set_of_bx2_minus_ax_minus_1_gt_0_l49_49377

theorem solution_set_of_bx2_minus_ax_minus_1_gt_0
  (a b : ℝ)
  (h1 : ∀ (x : ℝ), 2 < x ∧ x < 3 ↔ x^2 - a * x - b < 0) :
  ∀ (x : ℝ), -1 / 2 < x ∧ x < -1 / 3 ↔ b * x^2 - a * x - 1 > 0 :=
by
  sorry

end solution_set_of_bx2_minus_ax_minus_1_gt_0_l49_49377


namespace arithmetic_sequence_sum_l49_49765

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (a 3 = 7) ∧ (a 5 + a 7 = 26) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((a n)^2 - 1)) →
  (∀ n, T n = n / (4 * (n + 1))) := sorry

end arithmetic_sequence_sum_l49_49765


namespace top_three_probability_l49_49720

-- Definitions for the real-world problem
def total_ways_to_choose_three_cards : ℕ :=
  52 * 51 * 50

def favorable_ways_to_choose_three_specific_suits : ℕ :=
  13 * 13 * 13 * 6

def probability_top_three_inclusive (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- The mathematically equivalent proof problem's Lean statement
theorem top_three_probability:
  probability_top_three_inclusive total_ways_to_choose_three_cards favorable_ways_to_choose_three_specific_suits = 2197 / 22100 :=
by
  sorry

end top_three_probability_l49_49720


namespace box_combination_is_correct_l49_49637

variables (C A S T t u : ℕ)

theorem box_combination_is_correct
    (h1 : 3 * S % t = C)
    (h2 : 2 * A + C = T)
    (h3 : 2 * C + A + u = T) :
  (1000 * C + 100 * A + 10 * S + T = 7252) :=
sorry

end box_combination_is_correct_l49_49637


namespace salesman_bonus_l49_49996

theorem salesman_bonus (S B : ℝ) 
  (h1 : S > 10000) 
  (h2 : 0.09 * S + 0.03 * (S - 10000) = 1380) 
  : B = 0.03 * (S - 10000) :=
sorry

end salesman_bonus_l49_49996


namespace percentage_of_oysters_with_pearls_l49_49071

def jamie_collects_oysters (oysters_per_dive dives total_pearls : ℕ) : ℕ :=
  oysters_per_dive * dives

def percentage_with_pearls (total_pearls total_oysters : ℕ) : ℕ :=
  (total_pearls * 100) / total_oysters

theorem percentage_of_oysters_with_pearls :
  ∀ (oysters_per_dive dives total_pearls : ℕ),
  oysters_per_dive = 16 →
  dives = 14 →
  total_pearls = 56 →
  percentage_with_pearls total_pearls (jamie_collects_oysters oysters_per_dive dives total_pearls) = 25 :=
by
  intros
  sorry

end percentage_of_oysters_with_pearls_l49_49071


namespace rick_savings_ratio_proof_l49_49267

-- Define the conditions
def erika_savings : ℤ := 155
def cost_of_gift : ℤ := 250
def cost_of_cake : ℤ := 25
def amount_left : ℤ := 5

-- Define the total amount they have together
def total_amount : ℤ := cost_of_gift + cost_of_cake - amount_left

-- Define Rick's savings based on the conditions
def rick_savings : ℤ := total_amount - erika_savings

-- Define the ratio of Rick's savings to the cost of the gift
def rick_gift_ratio : ℚ := rick_savings / cost_of_gift

-- Prove the ratio is 23/50
theorem rick_savings_ratio_proof : rick_gift_ratio = 23 / 50 :=
  by
    have h1 : total_amount = 270 := by sorry
    have h2 : rick_savings = 115 := by sorry
    have h3 : rick_gift_ratio = 23 / 50 := by sorry
    exact h3

end rick_savings_ratio_proof_l49_49267


namespace initial_ratio_l49_49254

theorem initial_ratio (partners associates associates_after_hiring : ℕ)
  (h_partners : partners = 20)
  (h_associates_after_hiring : associates_after_hiring = 20 * 34)
  (h_assoc_equation : associates + 50 = associates_after_hiring) :
  (partners : ℚ) / associates = 2 / 63 :=
by
  sorry

end initial_ratio_l49_49254


namespace trajectory_of_midpoint_l49_49132

theorem trajectory_of_midpoint (M : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) ∧
  (P.1 = M.1 ∧ P.2 = 2 * M.2) ∧ 
  (N.1 = P.1 ∧ N.2 = 0) ∧ 
  (M.1 = (P.1 + N.1) / 2 ∧ M.2 = (P.2 + N.2) / 2)
  → M.1^2 + 4 * M.2^2 = 1 := 
by
  sorry

end trajectory_of_midpoint_l49_49132


namespace parallel_vectors_eq_l49_49785

theorem parallel_vectors_eq (x : ℝ) :
  let a := (x, 1)
  let b := (2, 4)
  (a.1 / b.1 = a.2 / b.2) → x = 1 / 2 :=
by
  intros h
  sorry

end parallel_vectors_eq_l49_49785


namespace total_gray_trees_l49_49127

theorem total_gray_trees :
  (∃ trees_first trees_second trees_third gray1 gray2,
    trees_first = 100 ∧
    trees_second = 90 ∧
    trees_third = 82 ∧
    gray1 = trees_first - trees_third ∧
    gray2 = trees_second - trees_third ∧
    trees_first + trees_second - 2 * trees_third = gray1 + gray2) →
  (gray1 + gray2 = 26) :=
by
  intros
  sorry

end total_gray_trees_l49_49127


namespace sum_of_three_sqrt_139_l49_49991

theorem sum_of_three_sqrt_139 {x y z : ℝ} (h1 : x >= 0) (h2 : y >= 0) (h3 : z >= 0)
  (hx : x^2 + y^2 + z^2 = 75) (hy : x * y + y * z + z * x = 32) : x + y + z = Real.sqrt 139 := 
by
  sorry

end sum_of_three_sqrt_139_l49_49991


namespace x_squared_plus_y_squared_l49_49976

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + x + y = 11) : 
  x^2 + y^2 = 2893 / 36 := 
by 
  sorry

end x_squared_plus_y_squared_l49_49976


namespace sum_of_surface_areas_of_two_smaller_cuboids_l49_49175

theorem sum_of_surface_areas_of_two_smaller_cuboids
  (L W H : ℝ) (hL : L = 3) (hW : W = 2) (hH : H = 1) :
  ∃ S, (S = 26 ∨ S = 28 ∨ S = 34) ∧ (∀ l w h, (l = L / 2 ∨ w = W / 2 ∨ h = H / 2) →
  (S = 2 * 2 * (l * W + w * H + h * L))) :=
by
  sorry

end sum_of_surface_areas_of_two_smaller_cuboids_l49_49175
