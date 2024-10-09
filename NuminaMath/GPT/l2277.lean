import Mathlib

namespace simplify_first_expression_simplify_second_expression_l2277_227727

theorem simplify_first_expression (x y : ℝ) : 3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4 :=
sorry

theorem simplify_second_expression (x : ℝ) : (2 * x ^ 4 - 5 * x ^ 2 - 4 * x + 3) - (3 * x ^ 3 - 5 * x ^ 2 - 4 * x) = 2 * x ^ 4 - 3 * x ^ 3 + 3 :=
sorry

end simplify_first_expression_simplify_second_expression_l2277_227727


namespace ratio_f_l2277_227779

variable (f : ℝ → ℝ)

-- Hypothesis: For all x in ℝ^+, f'(x) = 3/x * f(x)
axiom hyp1 : ∀ x : ℝ, x > 0 → deriv f x = (3 / x) * f x

-- Hypothesis: f(2^2016) ≠ 0
axiom hyp2 : f (2^2016) ≠ 0

-- Prove that f(2^2017) / f(2^2016) = 8
theorem ratio_f : f (2^2017) / f (2^2016) = 8 :=
sorry

end ratio_f_l2277_227779


namespace least_number_correct_l2277_227762

def least_number_to_add_to_make_perfect_square (x : ℝ) : ℝ :=
  let y := 1 - x -- since 1 is the smallest whole number > sqrt(0.0320)
  y

theorem least_number_correct (x : ℝ) (h : x = 0.0320) : least_number_to_add_to_make_perfect_square x = 0.9680 :=
by {
  -- Proof is skipped
  -- The proof would involve verifying that adding this number to x results in a perfect square (1 in this case).
  sorry
}

end least_number_correct_l2277_227762


namespace find_xyz_l2277_227715

theorem find_xyz (x y z : ℝ) 
  (h1 : x * (y + z) = 180) 
  (h2 : y * (z + x) = 192) 
  (h3 : z * (x + y) = 204) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) : 
  x * y * z = 168 * Real.sqrt 6 :=
sorry

end find_xyz_l2277_227715


namespace find_value_of_expression_l2277_227752

variables (a b : ℝ)

-- Given the condition that 2a - 3b = 5, prove that 2a - 3b + 3 = 8.
theorem find_value_of_expression
  (h : 2 * a - 3 * b = 5) : 2 * a - 3 * b + 3 = 8 :=
by sorry

end find_value_of_expression_l2277_227752


namespace ratio_of_length_to_width_l2277_227716

-- Definitions of conditions
def width := 5
def area := 75

-- Theorem statement proving the ratio is 3
theorem ratio_of_length_to_width {l : ℕ} (h1 : l * width = area) : l / width = 3 :=
by sorry

end ratio_of_length_to_width_l2277_227716


namespace probability_four_coins_l2277_227768

-- Define four fair coin flips, having 2 possible outcomes for each coin
def four_coin_flips_outcomes : ℕ := 2 ^ 4

-- Define the favorable outcomes: all heads or all tails
def favorable_outcomes : ℕ := 2

-- The probability of getting all heads or all tails
def probability_all_heads_or_tails : ℚ := favorable_outcomes / four_coin_flips_outcomes

-- The theorem stating the answer to the problem
theorem probability_four_coins:
  probability_all_heads_or_tails = 1 / 8 := by
  sorry

end probability_four_coins_l2277_227768


namespace tank_full_capacity_l2277_227713

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end tank_full_capacity_l2277_227713


namespace gloria_money_left_l2277_227725

theorem gloria_money_left 
  (cost_of_cabin : ℕ) (cash : ℕ)
  (num_cypress_trees num_pine_trees num_maple_trees : ℕ)
  (price_per_cypress_tree price_per_pine_tree price_per_maple_tree : ℕ)
  (money_left : ℕ)
  (h_cost_of_cabin : cost_of_cabin = 129000)
  (h_cash : cash = 150)
  (h_num_cypress_trees : num_cypress_trees = 20)
  (h_num_pine_trees : num_pine_trees = 600)
  (h_num_maple_trees : num_maple_trees = 24)
  (h_price_per_cypress_tree : price_per_cypress_tree = 100)
  (h_price_per_pine_tree : price_per_pine_tree = 200)
  (h_price_per_maple_tree : price_per_maple_tree = 300)
  (h_money_left : money_left = (num_cypress_trees * price_per_cypress_tree + 
                                num_pine_trees * price_per_pine_tree + 
                                num_maple_trees * price_per_maple_tree + 
                                cash) - cost_of_cabin)
  : money_left = 350 :=
by
  sorry

end gloria_money_left_l2277_227725


namespace sqrt_200_eq_10_l2277_227792

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l2277_227792


namespace alyssa_hike_total_distance_l2277_227720

theorem alyssa_hike_total_distance
  (e f g h i : ℝ)
  (h1 : e + f + g = 40)
  (h2 : f + g + h = 48)
  (h3 : g + h + i = 54)
  (h4 : e + h = 30) :
  e + f + g + h + i = 118 :=
by
  sorry

end alyssa_hike_total_distance_l2277_227720


namespace trajectory_of_P_l2277_227793

-- Define points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the condition |PF2| - |PF1| = 4 for a moving point P
def condition (P : ℝ × ℝ) : Prop :=
  let PF1 := Real.sqrt ((P.1 + 4)^2 + P.2^2)
  let PF2 := Real.sqrt ((P.1 - 4)^2 + P.2^2)
  abs (PF2 - PF1) = 4

-- The target equation of the trajectory
def target_eq (P : ℝ × ℝ) : Prop :=
  P.1 * P.1 / 4 - P.2 * P.2 / 12 = 1 ∧ P.1 ≤ -2

theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, condition P → target_eq P := by
  sorry

end trajectory_of_P_l2277_227793


namespace start_A_to_B_l2277_227700

theorem start_A_to_B (x : ℝ)
  (A_to_C : x = 1000 * (1000 / 571.43) - 1000)
  (h1 : 1000 / (1000 - 600) = 1000 / (1000 - 428.57))
  (h2 : x = 1750 - 1000) :
  x = 750 :=
by
  rw [h2]
  sorry   -- Proof to be filled in.

end start_A_to_B_l2277_227700


namespace product_of_repeating_decimal_l2277_227742

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l2277_227742


namespace relationship_among_abc_l2277_227786

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_among_abc : c > a ∧ a > b :=
by
  sorry

end relationship_among_abc_l2277_227786


namespace total_profit_l2277_227771

theorem total_profit (A_investment : ℝ) (B_investment : ℝ) (C_investment : ℝ) 
                     (A_months : ℝ) (B_months : ℝ) (C_months : ℝ)
                     (C_share : ℝ) (A_profit_percentage : ℝ) : ℝ :=
  let A_capital_months := A_investment * A_months
  let B_capital_months := B_investment * B_months
  let C_capital_months := C_investment * C_months
  let total_capital_months := A_capital_months + B_capital_months + C_capital_months
  let P := (C_share * total_capital_months) / (C_capital_months * (1 - A_profit_percentage))
  P

example : total_profit 6500 8400 10000 6 5 3 1900 0.05 = 24667 := by
  sorry

end total_profit_l2277_227771


namespace probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l2277_227759

def probability_of_at_least_one_head (p : ℚ) (n : ℕ) : ℚ := 
  1 - (1 - p)^n

theorem probability_of_at_least_one_head_in_three_tosses_is_7_over_8 :
  probability_of_at_least_one_head (1/2) 3 = 7/8 :=
by 
  sorry

end probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l2277_227759


namespace find_value_of_expression_l2277_227783

variable (p q r s : ℝ)

def g (x : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

-- We state the condition that g(1) = 1
axiom g_at_one : g p q r s 1 = 1

-- Now, we state the problem we need to prove:
theorem find_value_of_expression : 5 * p - 3 * q + 2 * r - s = 5 :=
by
  -- We skip the proof here
  exact sorry

end find_value_of_expression_l2277_227783


namespace students_not_playing_either_game_l2277_227744

theorem students_not_playing_either_game
  (total_students : ℕ) -- There are 20 students in the class
  (play_basketball : ℕ) -- Half of them play basketball
  (play_volleyball : ℕ) -- Two-fifths of them play volleyball
  (play_both : ℕ) -- One-tenth of them play both basketball and volleyball
  (h_total : total_students = 20)
  (h_basketball : play_basketball = 10)
  (h_volleyball : play_volleyball = 8)
  (h_both : play_both = 2) :
  total_students - (play_basketball + play_volleyball - play_both) = 4 := by
  sorry

end students_not_playing_either_game_l2277_227744


namespace min_total_weight_l2277_227704

theorem min_total_weight (crates: Nat) (weight_per_crate: Nat) (h1: crates = 6) (h2: weight_per_crate ≥ 120): 
  crates * weight_per_crate ≥ 720 :=
by
  sorry

end min_total_weight_l2277_227704


namespace grid_problem_l2277_227749

theorem grid_problem 
  (n m : ℕ) 
  (h1 : ∀ (blue_cells : ℕ), blue_cells = m + n - 1 → (n * m ≠ 0) → (blue_cells = (n * m) / 2010)) :
  ∃ (k : ℕ), k = 96 :=
by
  sorry

end grid_problem_l2277_227749


namespace inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l2277_227776

theorem inequality_8xyz_leq_1 (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_cases_8xyz_eq_1 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ 
  (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨ 
  (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l2277_227776


namespace smallest_divisor_l2277_227755

theorem smallest_divisor (k n : ℕ) (x y : ℤ) :
  (∃ n : ℕ, k ∣ 2^n + 15) ∧ (∃ x y : ℤ, k = 3 * x^2 - 4 * x * y + 3 * y^2) → k = 23 := by
  sorry

end smallest_divisor_l2277_227755


namespace number_of_positive_expressions_l2277_227769

-- Define the conditions
variable (a b c : ℝ)
variable (h_a : a < 0)
variable (h_b : b > 0)
variable (h_c : c < 0)

-- Define the expressions
def ab := a * b
def ac := a * c
def a_b_c := a + b + c
def a_minus_b_c := a - b + c
def two_a_plus_b := 2 * a + b
def two_a_minus_b := 2 * a - b

-- Problem statement
theorem number_of_positive_expressions :
  (ab < 0) → (ac > 0) → (a_b_c > 0) → (a_minus_b_c < 0) → (two_a_plus_b < 0) → (two_a_minus_b < 0)
  → (2 = 2) :=
by
  sorry

end number_of_positive_expressions_l2277_227769


namespace kendalls_total_distance_l2277_227741

-- Definitions of the conditions
def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5

-- The theorem to prove the total distance
theorem kendalls_total_distance : distance_with_mother + distance_with_father = 0.67 :=
by
  sorry

end kendalls_total_distance_l2277_227741


namespace bug_visits_tiles_l2277_227765

theorem bug_visits_tiles (width length : ℕ) (gcd_width_length : ℕ) (broken_tile : ℕ × ℕ)
  (h_width : width = 12) (h_length : length = 25) (h_gcd : gcd_width_length = Nat.gcd width length)
  (h_broken_tile : broken_tile = (12, 18)) :
  width + length - gcd_width_length = 36 := by
  sorry

end bug_visits_tiles_l2277_227765


namespace weavers_problem_l2277_227791

theorem weavers_problem 
  (W : ℕ) 
  (H1 : 1 = W / 4) 
  (H2 : 3.5 = 49 / 14) :
  W = 4 :=
by
  sorry

end weavers_problem_l2277_227791


namespace scientific_notation_of_53_96_billion_l2277_227799

theorem scientific_notation_of_53_96_billion :
  (53.96 * 10^9) = (5.396 * 10^10) :=
sorry

end scientific_notation_of_53_96_billion_l2277_227799


namespace prob_at_least_one_heart_spade_or_king_l2277_227703

theorem prob_at_least_one_heart_spade_or_king :
  let total_cards := 52
  let hearts := 13
  let spades := 13
  let kings := 4
  let unique_hsk := hearts + spades + 2  -- Two unique kings from other suits
  let prob_not_hsk := (total_cards - unique_hsk) / total_cards
  let prob_not_hsk_two_draws := prob_not_hsk * prob_not_hsk
  let prob_at_least_one_hsk := 1 - prob_not_hsk_two_draws
  prob_at_least_one_hsk = 133 / 169 :=
by sorry

end prob_at_least_one_heart_spade_or_king_l2277_227703


namespace interest_difference_correct_l2277_227758

noncomputable def principal : ℝ := 1000
noncomputable def rate : ℝ := 0.10
noncomputable def time : ℝ := 4

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r)^t - P

noncomputable def interest_difference (P r t : ℝ) : ℝ := 
  compound_interest P r t - simple_interest P r t

theorem interest_difference_correct :
  interest_difference principal rate time = 64.10 :=
by
  sorry

end interest_difference_correct_l2277_227758


namespace first_step_induction_l2277_227775

theorem first_step_induction (n : ℕ) (h : 1 < n) : 1 + 1/2 + 1/3 < 2 :=
by
  sorry

end first_step_induction_l2277_227775


namespace simplify_sqrt_sum_l2277_227707

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l2277_227707


namespace ball_bounces_before_vertex_l2277_227739

def bounces_to_vertex (v h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ) : ℕ :=
units_per_bounce_vert * v / units_per_bounce_hor * h

theorem ball_bounces_before_vertex (verts : ℕ) (h : ℕ) (units_per_bounce_vert units_per_bounce_hor : ℕ)
    (H_vert : verts = 10)
    (H_units_vert : units_per_bounce_vert = 2)
    (H_units_hor : units_per_bounce_hor = 7) :
    bounces_to_vertex verts h units_per_bounce_vert units_per_bounce_hor = 5 := 
by
  sorry

end ball_bounces_before_vertex_l2277_227739


namespace acrobats_count_l2277_227795

theorem acrobats_count
  (a e c : ℕ)
  (h1 : 2 * a + 4 * e + 2 * c = 58)
  (h2 : a + e + c = 25) :
  a = 11 :=
by
  -- Proof skipped
  sorry

end acrobats_count_l2277_227795


namespace part1_part2_l2277_227751

variable {a b c : ℚ}

theorem part1 (ha : a < 0) : (a / |a|) = -1 :=
sorry

theorem part2 (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  min (a * b / |a * b| + |b * c| / (b * c) + a * c / |a * c| + |a * b * c| / (a * b * c)) (-2) = -2 :=
sorry

end part1_part2_l2277_227751


namespace probability_no_two_boys_same_cinema_l2277_227761

-- Definitions
def total_cinemas := 10
def total_boys := 7

def total_arrangements : ℕ := total_cinemas ^ total_boys
def favorable_arrangements : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4
def probability := (favorable_arrangements : ℚ) / total_arrangements

-- Mathematical proof problem
theorem probability_no_two_boys_same_cinema : 
  probability = 0.06048 := 
by {
  sorry -- Proof goes here
}

end probability_no_two_boys_same_cinema_l2277_227761


namespace max_soap_boxes_in_carton_l2277_227764

def carton_volume (length width height : ℕ) : ℕ :=
  length * width * height

def soap_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def max_soap_boxes (carton_volume soap_box_volume : ℕ) : ℕ :=
  carton_volume / soap_box_volume

theorem max_soap_boxes_in_carton :
  max_soap_boxes (carton_volume 25 42 60) (soap_box_volume 7 6 6) = 250 :=
by
  sorry

end max_soap_boxes_in_carton_l2277_227764


namespace denis_sum_of_numbers_l2277_227774

theorem denis_sum_of_numbers :
  ∃ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ a*d = 32 ∧ b*c = 14 ∧ a + b + c + d = 42 :=
sorry

end denis_sum_of_numbers_l2277_227774


namespace largest_five_digit_palindromic_number_l2277_227757

theorem largest_five_digit_palindromic_number (a b c d e : ℕ)
  (h1 : ∃ a b c, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
                 ∃ d e, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
                 (10001 * a + 1010 * b + 100 * c = 45 * (1001 * d + 110 * e))) :
  10001 * 5 + 1010 * 9 + 100 * 8 = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l2277_227757


namespace average_weight_increase_l2277_227709

theorem average_weight_increase
  (A : ℝ) -- Average weight of the two persons
  (w1 : ℝ) (h1 : w1 = 65) -- One person's weight is 65 kg 
  (w2 : ℝ) (h2 : w2 = 74) -- The new person's weight is 74 kg
  :
  ((A * 2 - w1 + w2) / 2 - A = 4.5) :=
by
  simp [h1, h2]
  sorry

end average_weight_increase_l2277_227709


namespace emily_fishes_correct_l2277_227730

/-- Given conditions:
1. Emily caught 4 trout weighing 2 pounds each.
2. Emily caught 3 catfish weighing 1.5 pounds each.
3. Bluegills weigh 2.5 pounds each.
4. Emily caught a total of 25 pounds of fish. -/
def emilyCatches : Prop :=
  ∃ (trout_count catfish_count bluegill_count : ℕ)
    (trout_weight catfish_weight bluegill_weight total_weight : ℝ),
    trout_count = 4 ∧ catfish_count = 3 ∧ 
    trout_weight = 2 ∧ catfish_weight = 1.5 ∧ 
    bluegill_weight = 2.5 ∧ 
    total_weight = 25 ∧
    (total_weight = (trout_count * trout_weight) + (catfish_count * catfish_weight) + (bluegill_count * bluegill_weight)) ∧
    bluegill_count = 5

theorem emily_fishes_correct : emilyCatches := by
  sorry

end emily_fishes_correct_l2277_227730


namespace rachel_points_product_l2277_227781

-- Define the scores in the first 10 games
def scores_first_10_games := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

-- Define the conditions as given in the problem
def total_score_first_10_games := scores_first_10_games.sum = 55
def points_scored_in_game_11 (P₁₁ : ℕ) : Prop := P₁₁ < 10 ∧ (55 + P₁₁) % 11 = 0
def points_scored_in_game_12 (P₁₁ P₁₂ : ℕ) : Prop := P₁₂ < 10 ∧ (55 + P₁₁ + P₁₂) % 12 = 0

-- Prove the product of the points scored in eleventh and twelfth games
theorem rachel_points_product : ∃ P₁₁ P₁₂ : ℕ, total_score_first_10_games ∧ points_scored_in_game_11 P₁₁ ∧ points_scored_in_game_12 P₁₁ P₁₂ ∧ P₁₁ * P₁₂ = 0 :=
by 
  sorry -- proof not required

end rachel_points_product_l2277_227781


namespace Eunji_total_wrong_questions_l2277_227794

theorem Eunji_total_wrong_questions 
  (solved_A : ℕ) (solved_B : ℕ) (wrong_A : ℕ) (right_diff : ℕ) 
  (h1 : solved_A = 12) 
  (h2 : solved_B = 15) 
  (h3 : wrong_A = 4) 
  (h4 : right_diff = 2) :
  (solved_A - (solved_A - (solved_A - wrong_A) + right_diff) + (solved_A - wrong_A) + right_diff - solved_B - (solved_B - (solved_A - (solved_A - wrong_A) + right_diff))) = 9 :=
by {
  sorry
}

end Eunji_total_wrong_questions_l2277_227794


namespace intersecting_lines_l2277_227777

theorem intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ (x = 0 ∨ y = 0) := by
  sorry

end intersecting_lines_l2277_227777


namespace symmetry_axis_of_function_l2277_227763

theorem symmetry_axis_of_function {x : ℝ} :
  (∃ k : ℤ, ∃ x : ℝ, (y = 2 * (Real.cos ((x / 2) + (Real.pi / 3))) ^ 2 - 1) ∧ (x + (2 * Real.pi) / 3 = k * Real.pi)) →
    x = (Real.pi / 3) ∧ 0 = y :=
sorry

end symmetry_axis_of_function_l2277_227763


namespace expression_value_l2277_227719

theorem expression_value 
  (a b c : ℕ) 
  (ha : a = 12) 
  (hb : b = 2) 
  (hc : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := 
by 
  sorry

end expression_value_l2277_227719


namespace square_area_l2277_227785

theorem square_area (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (h_eq : y = 8) : 
  (|x1 - x2|) ^ 2 = 36 :=
sorry

end square_area_l2277_227785


namespace solve_sys_eqns_l2277_227770

def sys_eqns_solution (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

theorem solve_sys_eqns :
  ∃ (x y : ℝ),
  (sys_eqns_solution x y ∧
  ((x = 0 ∧ y = 0) ∨
  (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
  (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2))) :=
by
  sorry

end solve_sys_eqns_l2277_227770


namespace simplify_and_evaluate_expression_l2277_227790

-- Definitions of the variables and their values
def x : ℤ := -2
def y : ℚ := 1 / 2

-- Theorem statement
theorem simplify_and_evaluate_expression : 
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 
  (1 : ℚ) / 2 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_expression_l2277_227790


namespace cottage_cheese_quantity_l2277_227782

theorem cottage_cheese_quantity (x : ℝ) 
    (milk_fat : ℝ := 0.05) 
    (curd_fat : ℝ := 0.155) 
    (whey_fat : ℝ := 0.005) 
    (milk_mass : ℝ := 1) 
    (h : (curd_fat * x + whey_fat * (milk_mass - x)) = milk_fat * milk_mass) : 
    x = 0.3 :=
    sorry

end cottage_cheese_quantity_l2277_227782


namespace greatest_m_div_36_and_7_l2277_227724

def reverse_digits (m : ℕ) : ℕ :=
  let d1 := (m / 1000) % 10
  let d2 := (m / 100) % 10
  let d3 := (m / 10) % 10
  let d4 := m % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_m_div_36_and_7
  (m : ℕ) (n : ℕ := reverse_digits m)
  (h1 : 1000 ≤ m ∧ m < 10000)
  (h2 : 1000 ≤ n ∧ n < 10000)
  (h3 : 36 ∣ m ∧ 36 ∣ n)
  (h4 : 7 ∣ m) :
  m = 9828 := 
sorry

end greatest_m_div_36_and_7_l2277_227724


namespace intersection_point_unique_l2277_227780

theorem intersection_point_unique (k : ℝ) :
  (∃ y : ℝ, k = -2 * y^2 - 3 * y + 5) ∧ (∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → -2 * y₁^2 - 3 * y₁ + 5 ≠ k ∨ -2 * y₂^2 - 3 * y₂ + 5 ≠ k)
  ↔ k = 49 / 8 := 
by sorry

end intersection_point_unique_l2277_227780


namespace bacon_percentage_l2277_227797

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end bacon_percentage_l2277_227797


namespace exists_digit_a_l2277_227736

theorem exists_digit_a : 
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (1111 * a - 1 = (a - 1) ^ (a - 2)) :=
by {
  sorry
}

end exists_digit_a_l2277_227736


namespace multiply_difference_of_cubes_l2277_227747

def multiply_and_simplify (x : ℝ) : ℝ :=
  (x^4 + 25 * x^2 + 625) * (x^2 - 25)

theorem multiply_difference_of_cubes (x : ℝ) :
  multiply_and_simplify x = x^6 - 15625 :=
by
  sorry

end multiply_difference_of_cubes_l2277_227747


namespace ratio_of_sides_l2277_227745

theorem ratio_of_sides 
  (a b c d : ℝ) 
  (h1 : (a * b) / (c * d) = 0.16) 
  (h2 : b / d = 2 / 5) : 
  a / c = 0.4 := 
by 
  sorry

end ratio_of_sides_l2277_227745


namespace sufficient_but_not_necessary_l2277_227706

noncomputable def problem_statement (a : ℝ) : Prop :=
(a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2)

theorem sufficient_but_not_necessary (a : ℝ) : problem_statement a := 
sorry

end sufficient_but_not_necessary_l2277_227706


namespace find_b_l2277_227712

theorem find_b 
    (x1 x2 b c : ℝ)
    (h_distinct : x1 ≠ x2)
    (h_root_x : ∀ x, (x^2 + 5 * b * x + c = 0) → x = x1 ∨ x = x2)
    (h_common_root : ∃ y, (y^2 + 2 * x1 * y + 2 * x2 = 0) ∧ (y^2 + 2 * x2 * y + 2 * x1 = 0)) :
  b = 1 / 10 := 
sorry

end find_b_l2277_227712


namespace possible_values_of_y_l2277_227788

theorem possible_values_of_y (x : ℝ) (hx : x^2 + 5 * (x / (x - 3)) ^ 2 = 50) :
  ∃ (y : ℝ), y = (x - 3)^2 * (x + 4) / (3 * x - 4) ∧ (y = 0 ∨ y = 15 ∨ y = 49) :=
sorry

end possible_values_of_y_l2277_227788


namespace region_area_l2277_227718

theorem region_area (x y : ℝ) : 
  (x^2 + y^2 + 14 * x + 18 * y = 0) → 
  (π * 130) = 130 * π :=
by 
  sorry

end region_area_l2277_227718


namespace initial_water_amount_l2277_227784

open Real

theorem initial_water_amount (W : ℝ)
  (h1 : ∀ (d : ℝ), d = 0.03 * 20)
  (h2 : ∀ (W : ℝ) (d : ℝ), d = 0.06 * W) :
  W = 10 :=
by
  sorry

end initial_water_amount_l2277_227784


namespace ways_to_select_books_l2277_227705

theorem ways_to_select_books (nChinese nMath nEnglish : ℕ) (h1 : nChinese = 9) (h2 : nMath = 7) (h3 : nEnglish = 5) :
  (nChinese * nMath + nChinese * nEnglish + nMath * nEnglish) = 143 :=
by
  sorry

end ways_to_select_books_l2277_227705


namespace sum_of_square_roots_l2277_227710

theorem sum_of_square_roots : 
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) + 
  Real.sqrt (1 + 3 + 5 + 7 + 9 + 11)) = 21 :=
by
  -- Proof here
  sorry

end sum_of_square_roots_l2277_227710


namespace negate_neg_two_l2277_227714

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two_l2277_227714


namespace at_most_two_zero_points_l2277_227753

noncomputable def f (x a : ℝ) := x^3 - 12 * x + a

theorem at_most_two_zero_points (a : ℝ) (h : a ≥ 16) : ∃ l u : ℝ, (∀ x : ℝ, f x a = 0 → x < l ∨ l ≤ x ∧ x ≤ u ∨ u < x) := sorry

end at_most_two_zero_points_l2277_227753


namespace range_of_a_l2277_227750

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1
def intersects_at_single_point (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
∃! x, f x a = 3

theorem range_of_a (a : ℝ) :
  intersects_at_single_point f a ↔ -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l2277_227750


namespace circle_intersection_l2277_227717

theorem circle_intersection (m : ℝ) :
  (x^2 + y^2 - 2*m*x + m^2 - 4 = 0 ∧ x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0) →
  (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
by sorry

end circle_intersection_l2277_227717


namespace pure_imaginary_a_zero_l2277_227756

theorem pure_imaginary_a_zero (a : ℝ) (h : ∃ b : ℝ, (i : ℂ) * (1 + (a : ℂ) * i) = (b : ℂ) * i) : a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l2277_227756


namespace hyperbola_eccentricity_correct_l2277_227734

noncomputable def hyperbola_eccentricity : ℝ := 2

variables {a b : ℝ}
variables (ha_pos : 0 < a) (hb_pos : 0 < b)
variables (h_hyperbola : ∃ x y, x^2/a^2 - y^2/b^2 = 1)
variables (h_circle_chord_len : ∃ d, d = 2 ∧ ∃ x y, ((x - 2)^2 + y^2 = 4) ∧ (x * b/a = -y))

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ), 0 < a → 0 < b → (∃ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  ∧ (∃ d, d = 2 ∧ ∃ x y, (x - 2)^2 + y^2 = 4 ∧ (x * b / a = -y)) →
  (eccentricity = 2) :=
by
  intro a b ha_pos hb_pos h_conditions
  have e := hyperbola_eccentricity
  sorry


end hyperbola_eccentricity_correct_l2277_227734


namespace apple_order_for_month_l2277_227767

def Chandler_apples (week : ℕ) : ℕ :=
  23 + 2 * week

def Lucy_apples (week : ℕ) : ℕ :=
  19 - week

def Ross_apples : ℕ :=
  15

noncomputable def total_apples : ℕ :=
  (Chandler_apples 0 + Chandler_apples 1 + Chandler_apples 2 + Chandler_apples 3) +
  (Lucy_apples 0 + Lucy_apples 1 + Lucy_apples 2 + Lucy_apples 3) +
  (Ross_apples * 4)

theorem apple_order_for_month : total_apples = 234 := by
  sorry

end apple_order_for_month_l2277_227767


namespace sabi_share_removed_l2277_227737

theorem sabi_share_removed :
  ∀ (N S M x : ℝ), N - 5 = 2 * (S - x) / 8 ∧ S - x = 4 * (6 * (M - 4)) / 16 ∧ M = 102 ∧ N + S + M = 1100 
  → x = 829.67 := by
  sorry

end sabi_share_removed_l2277_227737


namespace standard_deviation_of_distribution_l2277_227711

theorem standard_deviation_of_distribution (μ σ : ℝ) 
    (h₁ : μ = 15) (h₂ : μ - 2 * σ = 12) : σ = 1.5 := by
  sorry

end standard_deviation_of_distribution_l2277_227711


namespace find_triplets_l2277_227796

theorem find_triplets (u v w : ℝ):
  (u + v * w = 12) ∧ 
  (v + w * u = 12) ∧ 
  (w + u * v = 12) ↔ 
  (u = 3 ∧ v = 3 ∧ w = 3) ∨ 
  (u = -4 ∧ v = -4 ∧ w = -4) ∨ 
  (u = 1 ∧ v = 1 ∧ w = 11) ∨ 
  (u = 11 ∧ v = 1 ∧ w = 1) ∨ 
  (u = 1 ∧ v = 11 ∧ w = 1) := 
sorry

end find_triplets_l2277_227796


namespace lcm_12_35_l2277_227728

theorem lcm_12_35 : Nat.lcm 12 35 = 420 :=
by
  sorry

end lcm_12_35_l2277_227728


namespace evaluate_expression_l2277_227787

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^a + (b^a)^b = 793) := by
  -- The following lines skip the proof but outline the structure:
  sorry

end evaluate_expression_l2277_227787


namespace find_floor_of_apt_l2277_227729

-- Define the conditions:
-- Number of stories
def num_stories : Nat := 9
-- Number of entrances
def num_entrances : Nat := 10
-- Total apartments in entrance 10
def apt_num : Nat := 333
-- Number of apartments per floor in each entrance (which is to be found)
def apts_per_floor_per_entrance : Nat := 4 -- from solution b)

-- Assertion: The floor number that apartment number 333 is on in entrance 10
theorem find_floor_of_apt (num_stories num_entrances apt_num apts_per_floor_per_entrance : ℕ) :
  1 ≤ apt_num ∧ apt_num ≤ num_stories * num_entrances * apts_per_floor_per_entrance →
  (apt_num - 1) / apts_per_floor_per_entrance + 1 = 3 :=
by
  sorry

end find_floor_of_apt_l2277_227729


namespace probability_htth_l2277_227760

def probability_of_sequence_HTTH := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)

theorem probability_htth : probability_of_sequence_HTTH = 1 / 16 := by
  sorry

end probability_htth_l2277_227760


namespace smallest_solution_proof_l2277_227778

noncomputable def smallest_solution (x : ℝ) : ℝ :=
  if x = (1 - Real.sqrt 65) / 4 then x else x

theorem smallest_solution_proof :
  ∃ x : ℝ, (2 * x / (x - 2) + (2 * x^2 - 24) / x = 11) ∧
           (∀ y : ℝ, 2 * y / (y - 2) + (2 * y^2 - 24) / y = 11 → y ≥ (1 - Real.sqrt 65) / 4) ∧
           x = (1 - Real.sqrt 65) /4 :=
sorry

end smallest_solution_proof_l2277_227778


namespace max_value_f_l2277_227732

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem max_value_f (h : ∀ ε > (0 : ℝ), ∃ x : ℝ, x < 1 ∧ ε < f x) : ∀ x : ℝ, x < 1 → f x ≤ -1 :=
by
  intros x hx
  dsimp [f]
  -- Proof steps are omitted.
  sorry

example (h: ∀ ε > 0, ∃ x : ℝ, x < 1 ∧ ε < f x) : ∃ x : ℝ, x < 1 ∧ f x = -1 :=
by
  use 0
  -- Proof steps are omitted.
  sorry

end max_value_f_l2277_227732


namespace expression_result_l2277_227740

-- We denote k as a natural number representing the number of digits in A, B, C, and D.
variable (k : ℕ)

-- Definitions of the numbers A, B, C, D, and E based on the problem statement.
def A : ℕ := 3 * ((10 ^ (k - 1) - 1) / 9)
def B : ℕ := 4 * ((10 ^ (k - 1) - 1) / 9)
def C : ℕ := 6 * ((10 ^ (k - 1) - 1) / 9)
def D : ℕ := 7 * ((10 ^ (k - 1) - 1) / 9)
def E : ℕ := 5 * ((10 ^ (2 * k) - 1) / 9)

-- The statement we want to prove.
theorem expression_result :
  E - A * D - B * C + 1 = (10 ^ (k + 1) - 1) / 9 :=
by
  sorry

end expression_result_l2277_227740


namespace volume_of_TABC_l2277_227748

noncomputable def volume_pyramid_TABC : ℝ :=
  let TA : ℝ := 15
  let TB : ℝ := 15
  let TC : ℝ := 5 * Real.sqrt 3
  let area_ABT : ℝ := (1 / 2) * TA * TB
  (1 / 3) * area_ABT * TC

theorem volume_of_TABC :
  volume_pyramid_TABC = 187.5 * Real.sqrt 3 :=
sorry

end volume_of_TABC_l2277_227748


namespace solve_for_f_1988_l2277_227773

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom functional_eq (m n : ℕ+) : f (f m + f n) = m + n

theorem solve_for_f_1988 : f 1988 = 1988 :=
sorry

end solve_for_f_1988_l2277_227773


namespace even_sum_probability_correct_l2277_227733

-- Definition: Calculate probabilities based on the given wheels
def even_probability_wheel_one : ℚ := 1/3
def odd_probability_wheel_one : ℚ := 2/3
def even_probability_wheel_two : ℚ := 1/4
def odd_probability_wheel_two : ℚ := 3/4

-- Probability of both numbers being even
def both_even_probability : ℚ := even_probability_wheel_one * even_probability_wheel_two

-- Probability of both numbers being odd
def both_odd_probability : ℚ := odd_probability_wheel_one * odd_probability_wheel_two

-- Final probability of the sum being even
def even_sum_probability : ℚ := both_even_probability + both_odd_probability

theorem even_sum_probability_correct : even_sum_probability = 7/12 := 
sorry

end even_sum_probability_correct_l2277_227733


namespace solve_for_x_l2277_227721

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 := 
by sorry

end solve_for_x_l2277_227721


namespace soccer_teams_participation_l2277_227723

theorem soccer_teams_participation (total_games : ℕ) (teams_play : ℕ → ℕ) (x : ℕ) :
  (total_games = 20) → (teams_play x = x * (x - 1)) → x = 5 :=
by
  sorry

end soccer_teams_participation_l2277_227723


namespace correct_calculation_l2277_227746

theorem correct_calculation (x : ℕ) (h : x / 9 = 30) : x - 37 = 233 :=
by sorry

end correct_calculation_l2277_227746


namespace circular_garden_area_l2277_227766

theorem circular_garden_area (AD DB DC R : ℝ) 
  (h1 : AD = 10) 
  (h2 : DB = 10) 
  (h3 : DC = 12) 
  (h4 : AD^2 + DC^2 = R^2) : 
  π * R^2 = 244 * π := 
  by 
    sorry

end circular_garden_area_l2277_227766


namespace sum_of_cubes_of_roots_l2277_227735

theorem sum_of_cubes_of_roots (P : Polynomial ℝ)
  (hP : P = Polynomial.C (-1) + Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X) 
  (x1 x2 x3 : ℝ) 
  (hr : P.eval x1 = 0 ∧ P.eval x2 = 0 ∧ P.eval x3 = 0) :
  x1^3 + x2^3 + x3^3 = 3 := 
sorry

end sum_of_cubes_of_roots_l2277_227735


namespace vegetables_harvest_problem_l2277_227789

theorem vegetables_harvest_problem
  (same_area : ∀ (a b : ℕ), a = b)
  (first_field_harvest : ℕ := 900)
  (second_field_harvest : ℕ := 1500)
  (less_harvest_per_acre : ∀ (x : ℕ), x - 300 = y) :
  x = y ->
  900 / x = 1500 / (x + 300) :=
by
  sorry

end vegetables_harvest_problem_l2277_227789


namespace arithmetic_sequence_9th_term_l2277_227701

theorem arithmetic_sequence_9th_term (S : ℕ → ℕ) (d : ℕ) (Sn : ℕ) (a9 : ℕ) :
  (∀ n, S n = (n * (2 * S 1 + (n - 1) * d)) / 2) →
  d = 2 →
  Sn = 81 →
  S 9 = Sn →
  a9 = S 1 + 8 * d →
  a9 = 17 :=
by
  sorry

end arithmetic_sequence_9th_term_l2277_227701


namespace probability_correct_l2277_227754

noncomputable def probability_two_queens_or_at_least_one_jack : ℚ :=
  let total_cards := 52
  let queens := 3
  let jacks := 1
  let prob_two_queens := (queens * (queens - 1)) / (total_cards * (total_cards - 1))
  let prob_one_jack := jacks / total_cards * (total_cards - jacks) / (total_cards - 1) + (total_cards - jacks) / total_cards * jacks / (total_cards - 1)
  prob_two_queens + prob_one_jack

theorem probability_correct : probability_two_queens_or_at_least_one_jack = 9 / 221 := by
  sorry

end probability_correct_l2277_227754


namespace subtraction_of_decimals_l2277_227731

theorem subtraction_of_decimals : (3.75 - 0.48) = 3.27 :=
by
  sorry

end subtraction_of_decimals_l2277_227731


namespace girls_bought_balloons_l2277_227722

theorem girls_bought_balloons (initial_balloons boys_bought girls_bought remaining_balloons : ℕ)
  (h1 : initial_balloons = 36)
  (h2 : boys_bought = 3)
  (h3 : remaining_balloons = 21)
  (h4 : initial_balloons - remaining_balloons = boys_bought + girls_bought) :
  girls_bought = 12 := by
  sorry

end girls_bought_balloons_l2277_227722


namespace cistern_problem_l2277_227702

theorem cistern_problem (fill_rate empty_rate net_rate : ℝ) (T : ℝ) : 
  fill_rate = 1 / 3 →
  net_rate = 7 / 30 →
  empty_rate = 1 / T →
  net_rate = fill_rate - empty_rate →
  T = 10 :=
by
  intros
  sorry

end cistern_problem_l2277_227702


namespace set_operations_l2277_227726

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | 0 < x ∧ x < 5 }
def U : Set ℝ := Set.univ  -- Universal set ℝ
def complement (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem set_operations :
  (A ∩ B = { x | 0 < x ∧ x < 2 }) ∧ 
  (complement A ∪ B = { x | 0 < x }) :=
by {
  sorry
}

end set_operations_l2277_227726


namespace pipe_flow_rate_is_correct_l2277_227708

-- Definitions for the conditions
def tank_capacity : ℕ := 10000
def initial_water : ℕ := tank_capacity / 2
def fill_time : ℕ := 60
def drain1_rate : ℕ := 1000
def drain1_interval : ℕ := 4
def drain2_rate : ℕ := 1000
def drain2_interval : ℕ := 6

-- Calculation based on conditions
def total_water_needed : ℕ := tank_capacity - initial_water
def drain1_loss (time : ℕ) : ℕ := (time / drain1_interval) * drain1_rate
def drain2_loss (time : ℕ) : ℕ := (time / drain2_interval) * drain2_rate
def total_drain_loss (time : ℕ) : ℕ := drain1_loss time + drain2_loss time

-- Target flow rate for the proof
def total_fill (time : ℕ) : ℕ := total_water_needed + total_drain_loss time
def pipe_flow_rate : ℕ := total_fill fill_time / fill_time

-- Statement to prove
theorem pipe_flow_rate_is_correct : pipe_flow_rate = 500 := by  
  sorry

end pipe_flow_rate_is_correct_l2277_227708


namespace trader_gain_percentage_l2277_227798

variable (x : ℝ) (cost_of_one_pen : ℝ := x) (selling_cost_90_pens : ℝ := 90 * x) (gain : ℝ := 30 * x)

theorem trader_gain_percentage :
  30 * cost_of_one_pen / (90 * cost_of_one_pen) * 100 = 33.33 := by
  sorry

end trader_gain_percentage_l2277_227798


namespace initial_population_l2277_227743

theorem initial_population (P : ℝ) (h1 : 1.05 * (0.765 * P + 50) = 3213) : P = 3935 :=
by
  have h2 : 1.05 * (0.765 * P + 50) = 3213 := h1
  sorry

end initial_population_l2277_227743


namespace opposite_face_number_l2277_227772

theorem opposite_face_number (sum_faces : ℕ → ℕ → ℕ) (face_number : ℕ → ℕ) :
  (face_number 1 = 6) ∧ (face_number 2 = 7) ∧ (face_number 3 = 8) ∧ 
  (face_number 4 = 9) ∧ (face_number 5 = 10) ∧ (face_number 6 = 11) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 33 + 18) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 35 + 16) →
  (face_number 2 ≠ 9 ∨ face_number 2 ≠ 11) → 
  face_number 2 = 9 ∨ face_number 2 = 11 :=
by
  intros hface_numbers hsum1 hsum2 hnot_possible
  sorry

end opposite_face_number_l2277_227772


namespace tangent_product_eq_three_l2277_227738

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end tangent_product_eq_three_l2277_227738
