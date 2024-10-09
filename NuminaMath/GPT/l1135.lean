import Mathlib

namespace total_tips_l1135_113514

def tips_per_customer := 2
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

theorem total_tips : 
  (tips_per_customer * (customers_friday + customers_saturday + customers_sunday) = 296) :=
by
  sorry

end total_tips_l1135_113514


namespace b_contribution_l1135_113535

/-- A starts business with Rs. 3500.
    After 9 months, B joins as a partner.
    After a year, the profit is divided in the ratio 2:3.
    Prove that B's contribution to the capital is Rs. 21000. -/
theorem b_contribution (a_capital : ℕ) (months_a : ℕ) (b_time : ℕ) (profit_ratio_num : ℕ) (profit_ratio_den : ℕ)
  (h_a_capital : a_capital = 3500)
  (h_months_a : months_a = 12)
  (h_b_time : b_time = 3)
  (h_profit_ratio : profit_ratio_num = 2 ∧ profit_ratio_den = 3) :
  (21000 * b_time * profit_ratio_num) / (3 * profit_ratio_den) = 3500 * months_a := by
  sorry

end b_contribution_l1135_113535


namespace seashells_total_l1135_113596

theorem seashells_total (tim_seashells sally_seashells : ℕ) (ht : tim_seashells = 37) (hs : sally_seashells = 13) :
  tim_seashells + sally_seashells = 50 := 
by 
  sorry

end seashells_total_l1135_113596


namespace value_of_a8_l1135_113560

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → 2 * a n + a (n + 1) = 0

theorem value_of_a8 (a : ℕ → ℝ) (h1 : seq a) (h2 : a 3 = -2) : a 8 = 64 :=
sorry

end value_of_a8_l1135_113560


namespace num_handshakes_l1135_113542

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l1135_113542


namespace sum_base_6_l1135_113564

-- Define base 6 numbers
def n1 : ℕ := 1 * 6^3 + 4 * 6^2 + 5 * 6^1 + 2 * 6^0
def n2 : ℕ := 2 * 6^3 + 3 * 6^2 + 5 * 6^1 + 4 * 6^0

-- Define the expected result in base 6
def expected_sum : ℕ := 4 * 6^3 + 2 * 6^2 + 5 * 6^1 + 0 * 6^0

-- The theorem to prove
theorem sum_base_6 : n1 + n2 = expected_sum := by
    sorry

end sum_base_6_l1135_113564


namespace total_leftover_tarts_l1135_113559

variable (cherry_tart blueberry_tart peach_tart : ℝ)
variable (h1 : cherry_tart = 0.08)
variable (h2 : blueberry_tart = 0.75)
variable (h3 : peach_tart = 0.08)

theorem total_leftover_tarts : 
  cherry_tart + blueberry_tart + peach_tart = 0.91 := 
by 
  sorry

end total_leftover_tarts_l1135_113559


namespace find_k_and_x2_l1135_113558

theorem find_k_and_x2 (k : ℝ) (x2 : ℝ)
  (h1 : 2 * x2 = k)
  (h2 : 2 + x2 = 6) :
  k = 8 ∧ x2 = 4 :=
by
  sorry

end find_k_and_x2_l1135_113558


namespace band_member_earnings_l1135_113555

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l1135_113555


namespace almost_perfect_numbers_l1135_113585

def d (n : Nat) : Nat := 
  -- Implement the function to count the number of positive divisors of n
  sorry

def f (n : Nat) : Nat := 
  -- Implement the function f(n) as given in the problem statement
  sorry

def isAlmostPerfect (n : Nat) : Prop := 
  f n = n

theorem almost_perfect_numbers :
  ∀ n, isAlmostPerfect n → n = 1 ∨ n = 3 ∨ n = 18 ∨ n = 36 :=
by
  sorry

end almost_perfect_numbers_l1135_113585


namespace relationship_between_a_and_b_l1135_113593

theorem relationship_between_a_and_b (a b : ℝ) (h₀ : a ≠ 0) (max_point : ∃ x, (x = 0 ∨ x = 1/3) ∧ (∀ y, (y = 0 ∨ y = 1/3) → (3 * a * y^2 + 2 * b * y) = 0)) : a + 2 * b = 0 :=
sorry

end relationship_between_a_and_b_l1135_113593


namespace ratio_a_to_c_l1135_113531

-- Define the variables and ratios
variables (x y z a b c d : ℝ)

-- Define the conditions as given ratios
variables (h1 : a / b = 2 * x / (3 * y))
variables (h2 : b / c = z / (5 * z))
variables (h3 : a / d = 4 * x / (7 * y))
variables (h4 : d / c = 7 * y / (3 * z))

-- Statement to prove the ratio of a to c
theorem ratio_a_to_c (x y z a b c d : ℝ) 
  (h1 : a / b = 2 * x / (3 * y)) 
  (h2 : b / c = z / (5 * z)) 
  (h3 : a / d = 4 * x / (7 * y)) 
  (h4 : d / c = 7 * y / (3 * z)) : a / c = 2 * x / (15 * y) :=
sorry

end ratio_a_to_c_l1135_113531


namespace rectangle_area_l1135_113562

theorem rectangle_area
  (line : ∀ x, 6 = x * x + 4 * x + 3 → x = -2 + Real.sqrt 7 ∨ x = -2 - Real.sqrt 7)
  (shorter_side : ∃ l, l = 2 * Real.sqrt 7 ∧ ∃ s, s = l + 3) :
  ∃ a, a = 28 + 12 * Real.sqrt 7 :=
by
  sorry

end rectangle_area_l1135_113562


namespace common_ratio_of_geometric_seq_l1135_113534

variable {α : Type*} [Field α]

-- Definition of the geometric sequence
def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ n

-- Sum of the first three terms of the geometric sequence
def sum_first_three_terms (a q: α) : α :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

theorem common_ratio_of_geometric_seq (a q : α) (h : sum_first_three_terms a q = 3 * a) : q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_seq_l1135_113534


namespace log_equivalence_l1135_113552

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end log_equivalence_l1135_113552


namespace a100_gt_2pow99_l1135_113519

theorem a100_gt_2pow99 (a : Fin 101 → ℕ) 
  (h_pos : ∀ i, a i > 0) 
  (h_initial : a 1 > a 0) 
  (h_rec : ∀ k, 2 ≤ k → a k = 3 * a (k - 1) - 2 * a (k - 2)) 
  : a 100 > 2 ^ 99 :=
by
  sorry

end a100_gt_2pow99_l1135_113519


namespace find_n_l1135_113524

noncomputable def arctan_sum_eq_pi_over_2 (n : ℕ) : Prop :=
  Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2

theorem find_n (h : ∃ n, arctan_sum_eq_pi_over_2 n) : ∃ n, n = 54 := by
  obtain ⟨n, hn⟩ := h
  have H : 1 / 3 + 1 / 4 + 1 / 7 < 1 := by sorry
  sorry

end find_n_l1135_113524


namespace angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l1135_113527

open Real

variable {A B C a b c : ℝ}
variable (AM BM MC : ℝ)

-- Conditions
axiom triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)
axiom BM_MC_relation : BM = (1 / 2) * MC

-- Part 1: Measure of angle A
theorem angle_A_is_pi_over_3 (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) : 
  A = π / 3 :=
by sorry

-- Part 2: Minimum value of |AM|^2 / S
noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * sin A

axiom condition_b_eq_2c : b = 2 * c

theorem minimum_value_AM_sq_div_S (AM BM MC : ℝ) (S : ℝ) (H : BM = (1 / 2) * MC) 
  (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) 
  (area : S = area_triangle a b c A)
  (condition_b_eq_2c : b = 2 * c) : 
  (AM ^ 2) / S ≥ (8 * sqrt 3) / 9 :=
by sorry

end angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l1135_113527


namespace find_train_speed_l1135_113556

variable (bridge_length train_length train_crossing_time : ℕ)

def speed_of_train (bridge_length train_length train_crossing_time : ℕ) : ℕ :=
  (bridge_length + train_length) / train_crossing_time

theorem find_train_speed
  (bridge_length : ℕ) (train_length : ℕ) (train_crossing_time : ℕ)
  (h_bridge_length : bridge_length = 180)
  (h_train_length : train_length = 120)
  (h_train_crossing_time : train_crossing_time = 20) :
  speed_of_train bridge_length train_length train_crossing_time = 15 := by
  sorry

end find_train_speed_l1135_113556


namespace urea_formation_l1135_113565

theorem urea_formation (CO2 NH3 : ℕ) (OCN2 H2O : ℕ) (h1 : CO2 = 3) (h2 : NH3 = 6) :
  (∀ x, CO2 * 1 + NH3 * 2 = x + (2 * x) + x) →
  OCN2 = 3 :=
by
  sorry

end urea_formation_l1135_113565


namespace investment_ratio_correct_l1135_113538

-- Constants representing the savings and investments
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 4
def cost_per_share : ℕ := 50
def shares_bought : ℕ := 25

-- Derived quantities from the conditions
def total_savings_wife : ℕ := weekly_savings_wife * weeks_in_month * months_saving
def total_savings_husband : ℕ := monthly_savings_husband * months_saving
def total_savings : ℕ := total_savings_wife + total_savings_husband
def total_invested_in_stocks : ℕ := shares_bought * cost_per_share
def investment_ratio_nat : ℚ := (total_invested_in_stocks : ℚ) / (total_savings : ℚ)

-- Proof statement
theorem investment_ratio_correct : investment_ratio_nat = 1 / 2 := by
  sorry

end investment_ratio_correct_l1135_113538


namespace valid_votes_other_candidate_l1135_113571

theorem valid_votes_other_candidate (total_votes : ℕ)
  (invalid_percentage valid_percentage candidate1_percentage candidate2_percentage : ℕ)
  (h_invalid_valid_sum : invalid_percentage + valid_percentage = 100)
  (h_candidates_sum : candidate1_percentage + candidate2_percentage = 100)
  (h_invalid_percentage : invalid_percentage = 20)
  (h_candidate1_percentage : candidate1_percentage = 55)
  (h_total_votes : total_votes = 7500)
  (h_valid_percentage_eq : valid_percentage = 100 - invalid_percentage)
  (h_candidate2_percentage_eq : candidate2_percentage = 100 - candidate1_percentage) :
  ( ( candidate2_percentage * ( valid_percentage * total_votes / 100) ) / 100 ) = 2700 :=
  sorry

end valid_votes_other_candidate_l1135_113571


namespace sum_not_zero_l1135_113548

theorem sum_not_zero (a b c d : ℝ) (h1 : a * b * c - d = 1) (h2 : b * c * d - a = 2) 
  (h3 : c * d * a - b = 3) (h4 : d * a * b - c = -6) : a + b + c + d ≠ 0 :=
sorry

end sum_not_zero_l1135_113548


namespace total_number_of_fish_l1135_113586

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l1135_113586


namespace maria_threw_out_carrots_l1135_113505

theorem maria_threw_out_carrots (initially_picked: ℕ) (picked_next_day: ℕ) (total_now: ℕ) (carrots_thrown_out: ℕ) :
  initially_picked = 48 → 
  picked_next_day = 15 → 
  total_now = 52 → 
  (initially_picked + picked_next_day - total_now = carrots_thrown_out) → 
  carrots_thrown_out = 11 :=
by
  intros
  sorry

end maria_threw_out_carrots_l1135_113505


namespace ratio_B_C_l1135_113546

variable (A B C : ℕ)
variable (h1 : A = B + 2)
variable (h2 : A + B + C = 37)
variable (h3 : B = 14)

theorem ratio_B_C : B / C = 2 := by
  sorry

end ratio_B_C_l1135_113546


namespace clubsuit_commute_l1135_113580

-- Define the operation a ♣ b = a^3 * b - a * b^3
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the proposition to prove
theorem clubsuit_commute (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end clubsuit_commute_l1135_113580


namespace ratio_mets_to_redsox_l1135_113530

theorem ratio_mets_to_redsox (Y M R : ℕ)
  (h1 : Y / M = 3 / 2)
  (h2 : M = 96)
  (h3 : Y + M + R = 360) :
  M / R = 4 / 5 :=
by sorry

end ratio_mets_to_redsox_l1135_113530


namespace baby_frogs_on_rock_l1135_113549

theorem baby_frogs_on_rock (f_l f_L f_T : ℕ) (h1 : f_l = 5) (h2 : f_L = 3) (h3 : f_T = 32) : 
  f_T - (f_l + f_L) = 24 :=
by sorry

end baby_frogs_on_rock_l1135_113549


namespace sqrt_of_square_of_neg_five_eq_five_l1135_113568

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l1135_113568


namespace evaluate_expression_l1135_113517

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

axiom g_inverse : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x

axiom g_1_2 : g 1 = 2
axiom g_4_7 : g 4 = 7
axiom g_3_8 : g 3 = 8

theorem evaluate_expression :
  g_inv (g_inv 8 * g_inv 2) = 3 :=
by
  sorry

end evaluate_expression_l1135_113517


namespace total_games_to_determine_winner_l1135_113522

-- Conditions: Initial number of teams in the preliminary round
def initial_teams : ℕ := 24

-- Condition: Preliminary round eliminates 50% of the teams
def preliminary_round_elimination (n : ℕ) : ℕ := n / 2

-- Function to compute the required games for any single elimination tournament
def single_elimination_games (teams : ℕ) : ℕ :=
  if teams = 0 then 0
  else teams - 1

-- Proof Statement: Total number of games to determine the winner
theorem total_games_to_determine_winner (n : ℕ) (h : n = 24) :
  preliminary_round_elimination n + single_elimination_games (preliminary_round_elimination n) = 23 :=
by
  sorry

end total_games_to_determine_winner_l1135_113522


namespace farm_owns_60_more_horses_than_cows_l1135_113508

-- Let x be the number of cows initially
-- The number of horses initially is 4x
-- After selling 15 horses and buying 15 cows, the ratio of horses to cows becomes 7:3

theorem farm_owns_60_more_horses_than_cows (x : ℕ) (h_pos : 0 < x)
  (h_ratio : (4 * x - 15) / (x + 15) = 7 / 3) :
  (4 * x - 15) - (x + 15) = 60 :=
by
  sorry

end farm_owns_60_more_horses_than_cows_l1135_113508


namespace cube_edge_length_l1135_113587

-- Define edge length and surface area
variables (edge_length surface_area : ℝ)

-- Given condition
def surface_area_condition : Prop := surface_area = 294

-- Cube surface area formula
def cube_surface_area : Prop := surface_area = 6 * edge_length^2

-- Proof statement
theorem cube_edge_length (h1: surface_area_condition surface_area) (h2: cube_surface_area edge_length surface_area) : edge_length = 7 := 
by
  sorry

end cube_edge_length_l1135_113587


namespace calculate_f_17_69_l1135_113526

noncomputable def f (x y: ℕ) : ℚ := sorry

axiom f_self : ∀ x, f x x = x
axiom f_symm : ∀ x y, f x y = f y x
axiom f_add : ∀ x y, (x + y) * f x y = y * f x (x + y)

theorem calculate_f_17_69 : f 17 69 = 73.3125 := sorry

end calculate_f_17_69_l1135_113526


namespace quadratic_equation_general_form_l1135_113545

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 2 * (x + 2)^2 + (x + 3) * (x - 2) = -11 ↔ 3 * x^2 + 9 * x + 13 = 0 :=
sorry

end quadratic_equation_general_form_l1135_113545


namespace total_amount_proof_l1135_113500

-- Define the relationships between x, y, and z in terms of the amounts received
variables (x y z : ℝ)

-- Given: For each rupee x gets, y gets 0.45 rupees and z gets 0.50 rupees
def relationship1 : Prop := ∀ (k : ℝ), y = 0.45 * k ∧ z = 0.50 * k ∧ x = k

-- Given: The share of y is Rs. 54
def condition1 : Prop := y = 54

-- The total amount x + y + z is Rs. 234
def total_amount (x y z : ℝ) : ℝ := x + y + z

-- Prove that the total amount is Rs. 234
theorem total_amount_proof (x y z : ℝ) (h1: relationship1 x y z) (h2: condition1 y) : total_amount x y z = 234 :=
sorry

end total_amount_proof_l1135_113500


namespace correct_operation_l1135_113553

theorem correct_operation :
  (3 * a^3 - 2 * a^3 = a^3) ∧ ¬(m - 4 * m = -3) ∧ ¬(a^2 * b - a * b^2 = 0) ∧ ¬(2 * x + 3 * x = 5 * x^2) :=
by
  sorry

end correct_operation_l1135_113553


namespace ellipse_foci_x_axis_l1135_113539

theorem ellipse_foci_x_axis (m n : ℝ) (h_eq : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1)
  (h_foci : ∃ (c : ℝ), c = 0 ∧ (c^2 = 1 - n/m)) : n > m ∧ m > 0 ∧ n > 0 :=
sorry

end ellipse_foci_x_axis_l1135_113539


namespace savings_increase_is_100_percent_l1135_113544

variable (I : ℝ) -- Initial income
variable (S : ℝ) -- Initial savings
variable (I2 : ℝ) -- Income in the second year
variable (E1 : ℝ) -- Expenditure in the first year
variable (E2 : ℝ) -- Expenditure in the second year
variable (S2 : ℝ) -- Second year savings

-- Initial conditions
def initial_savings (I : ℝ) : ℝ := 0.25 * I
def first_year_expenditure (I : ℝ) (S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.25 * I

-- Total expenditure condition
def total_expenditure_condition (E1 : ℝ) (E2 : ℝ) : Prop := E1 + E2 = 2 * E1

-- Prove that the savings increase in the second year is 100%
theorem savings_increase_is_100_percent :
   ∀ (I S E1 I2 E2 S2 : ℝ),
     S = initial_savings I →
     E1 = first_year_expenditure I S →
     I2 = second_year_income I →
     total_expenditure_condition E1 E2 →
     S2 = I2 - E2 →
     ((S2 - S) / S) * 100 = 100 := by
  sorry

end savings_increase_is_100_percent_l1135_113544


namespace equation_true_l1135_113576

variables {AB BC CD AD AC BD : ℝ}

theorem equation_true :
  (AD * BC + AB * CD = AC * BD) ∧
  (AD * BC - AB * CD ≠ AC * BD) ∧
  (AB * BC + AC * CD ≠ AC * BD) ∧
  (AB * BC - AC * CD ≠ AC * BD) :=
by
  sorry

end equation_true_l1135_113576


namespace hexagon_area_correct_l1135_113595

structure Point where
  x : ℝ
  y : ℝ

def hexagon : List Point := [
  { x := 0, y := 0 },
  { x := 2, y := 4 },
  { x := 6, y := 4 },
  { x := 8, y := 0 },
  { x := 6, y := -4 },
  { x := 2, y := -4 }
]

def area_of_hexagon (hex : List Point) : ℝ :=
  -- Assume a function that calculates the area of a polygon given a list of vertices
  sorry

theorem hexagon_area_correct : area_of_hexagon hexagon = 16 :=
  sorry

end hexagon_area_correct_l1135_113595


namespace range_of_f3_l1135_113532

def f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_of_f3 (a c : ℝ)
  (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
  (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end range_of_f3_l1135_113532


namespace add_n_to_constant_l1135_113598

theorem add_n_to_constant (y n : ℝ) (h_eq : y^4 - 20 * y + 1 = 22) (h_n : n = 3) : y^4 - 20 * y + 4 = 25 :=
by
  sorry

end add_n_to_constant_l1135_113598


namespace equation_of_line_l1135_113584

theorem equation_of_line 
  (a : ℝ) (h : a < 3) 
  (C : ℝ × ℝ) 
  (hC : C = (-2, 3)) 
  (l_intersects_circle : ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 + 2 * A.1 - 4 * A.2 + a = 0) ∧ 
    (B.1^2 + B.2^2 + 2 * B.1 - 4 * B.2 + a = 0) ∧ 
    (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) : 
  ∃ (m b : ℝ), 
    (m = 1) ∧ 
    (b = -5) ∧ 
    (∀ x y, y - 3 = m * (x + 2) ↔ x - y + 5 = 0) :=
by
  sorry

end equation_of_line_l1135_113584


namespace jill_trips_to_fill_tank_l1135_113563

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end jill_trips_to_fill_tank_l1135_113563


namespace shoe_cost_on_monday_l1135_113533

theorem shoe_cost_on_monday 
  (price_thursday : ℝ) 
  (increase_rate : ℝ) 
  (decrease_rate : ℝ) 
  (price_thursday_eq : price_thursday = 40)
  (increase_rate_eq : increase_rate = 0.10)
  (decrease_rate_eq : decrease_rate = 0.10)
  :
  let price_friday := price_thursday * (1 + increase_rate)
  let discount := price_friday * decrease_rate
  let price_monday := price_friday - discount
  price_monday = 39.60 :=
by
  sorry

end shoe_cost_on_monday_l1135_113533


namespace num_of_negative_x_l1135_113594

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l1135_113594


namespace total_splash_width_l1135_113529

theorem total_splash_width :
  let pebble_splash := 1 / 4
  let rock_splash := 1 / 2
  let boulder_splash := 2
  let pebbles := 6
  let rocks := 3
  let boulders := 2
  let total_pebble_splash := pebbles * pebble_splash
  let total_rock_splash := rocks * rock_splash
  let total_boulder_splash := boulders * boulder_splash
  let total_splash := total_pebble_splash + total_rock_splash + total_boulder_splash
  total_splash = 7 := by
  sorry

end total_splash_width_l1135_113529


namespace gratuity_is_four_l1135_113547

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end gratuity_is_four_l1135_113547


namespace total_birds_on_fence_l1135_113569

variable (initial_birds : ℕ := 1)
variable (added_birds : ℕ := 4)

theorem total_birds_on_fence : initial_birds + added_birds = 5 := by
  sorry

end total_birds_on_fence_l1135_113569


namespace isosceles_triangle_perimeter_l1135_113592

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a = 6 ∨ b = 6) 
(h_isosceles : a = b ∨ b = a) : 
  a + b + a = 15 ∨ b + a + b = 15 :=
by sorry

end isosceles_triangle_perimeter_l1135_113592


namespace complex_root_product_value_l1135_113506

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end complex_root_product_value_l1135_113506


namespace largest_value_among_given_numbers_l1135_113516

theorem largest_value_among_given_numbers :
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  d >= a ∧ d >= b ∧ d >= c ∧ d >= e :=
by
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  sorry

end largest_value_among_given_numbers_l1135_113516


namespace total_crayons_lost_or_given_away_l1135_113512

/-
Paul gave 52 crayons to his friends.
Paul lost 535 crayons.
Paul had 492 crayons left.
Prove that the total number of crayons lost or given away is 587.
-/
theorem total_crayons_lost_or_given_away
  (crayons_given : ℕ)
  (crayons_lost : ℕ)
  (crayons_left : ℕ)
  (h_crayons_given : crayons_given = 52)
  (h_crayons_lost : crayons_lost = 535)
  (h_crayons_left : crayons_left = 492) :
  crayons_given + crayons_lost = 587 := 
sorry

end total_crayons_lost_or_given_away_l1135_113512


namespace dot_product_result_l1135_113582

open scoped BigOperators

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (-1, 2)

-- Define the addition of two vectors
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_result : dot_product (vector_add a b) a = 5 := by
  sorry

end dot_product_result_l1135_113582


namespace last_number_l1135_113567

theorem last_number (A B C D E F G : ℕ)
  (h1 : A + B + C + D = 52)
  (h2 : D + E + F + G = 60)
  (h3 : E + F + G = 55)
  (h4 : D^2 = G) : G = 25 :=
by
  sorry

end last_number_l1135_113567


namespace sum_of_roots_of_quadratic_l1135_113501

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, x^2 + 2000*x - 2000 = 0 ->
  (∃ x1 x2 : ℝ, (x1 ≠ x2 ∧ x1^2 + 2000*x1 - 2000 = 0 ∧ x2^2 + 2000*x2 - 2000 = 0 ∧ x1 + x2 = -2000)) :=
sorry

end sum_of_roots_of_quadratic_l1135_113501


namespace muffin_price_proof_l1135_113521

noncomputable def price_per_muffin (s m t : ℕ) (contribution : ℕ) : ℕ :=
  contribution / (s + m + t)

theorem muffin_price_proof :
  ∀ (sasha_muffins melissa_muffins : ℕ) (h1 : sasha_muffins = 30) (h2 : melissa_muffins = 4 * sasha_muffins)
  (tiffany_muffins total_muffins : ℕ) (h3 : total_muffins = sasha_muffins + melissa_muffins)
  (h4 : tiffany_muffins = total_muffins / 2)
  (h5 : total_muffins = sasha_muffins + melissa_muffins + tiffany_muffins)
  (contribution : ℕ) (h6 : contribution = 900),
  price_per_muffin sasha_muffins melissa_muffins tiffany_muffins contribution = 4 :=
by
  intros sasha_muffins melissa_muffins h1 h2 tiffany_muffins total_muffins h3 h4 h5 contribution h6
  simp [price_per_muffin]
  sorry

end muffin_price_proof_l1135_113521


namespace store_profit_l1135_113572

theorem store_profit :
  let selling_price : ℝ := 80
  let cost_price_profitable : ℝ := (selling_price - 0.60 * selling_price)
  let cost_price_loss : ℝ := (selling_price + 0.20 * selling_price)
  selling_price + selling_price - cost_price_profitable - cost_price_loss = 10 := by
  sorry

end store_profit_l1135_113572


namespace Madeline_hours_left_over_l1135_113570

theorem Madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  total_hours_per_week - total_busy_hours = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  have : total_hours_per_week - total_busy_hours = 168 - 122 := by rfl
  have : 168 - 122 = 46 := by rfl
  exact this

end Madeline_hours_left_over_l1135_113570


namespace inequality_problem_l1135_113590

noncomputable def a := (3 / 4) * Real.exp (2 / 5)
noncomputable def b := 2 / 5
noncomputable def c := (2 / 5) * Real.exp (3 / 4)

theorem inequality_problem : b < c ∧ c < a := by
  sorry

end inequality_problem_l1135_113590


namespace total_pie_eaten_l1135_113554

theorem total_pie_eaten (s1 s2 s3 : ℚ) (h1 : s1 = 8/9) (h2 : s2 = 5/6) (h3 : s3 = 2/3) :
  s1 + s2 + s3 = 43/18 := by
  sorry

end total_pie_eaten_l1135_113554


namespace collinear_magnitude_a_perpendicular_magnitude_b_l1135_113515

noncomputable section

open Real

-- Defining the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Defining the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Given conditions and respective proofs
theorem collinear_magnitude_a (x : ℝ) (h : 1 * 3 = x ^ 2) : magnitude (a x) = 2 :=
by sorry

theorem perpendicular_magnitude_b (x : ℝ) (h : 1 * x + x * 3 = 0) : magnitude (b x) = 3 :=
by sorry

end collinear_magnitude_a_perpendicular_magnitude_b_l1135_113515


namespace fifteenth_term_geometric_sequence_l1135_113502

theorem fifteenth_term_geometric_sequence :
  let a1 := 5
  let r := (1 : ℝ) / 2
  let fifteenth_term := a1 * r^(14 : ℕ)
  fifteenth_term = (5 : ℝ) / 16384 := by
sorry

end fifteenth_term_geometric_sequence_l1135_113502


namespace cover_square_with_rectangles_l1135_113520

theorem cover_square_with_rectangles :
  ∃ n : ℕ, n = 24 ∧
  ∀ (rect_area : ℕ) (square_area : ℕ), rect_area = 2 * 3 → square_area = 12 * 12 → square_area / rect_area = n :=
by
  use 24
  sorry

end cover_square_with_rectangles_l1135_113520


namespace time_after_classes_l1135_113588

def time_after_maths : Nat := 60
def time_after_history : Nat := 60 + 90
def time_after_break1 : Nat := time_after_history + 25
def time_after_geography : Nat := time_after_break1 + 45
def time_after_break2 : Nat := time_after_geography + 15
def time_after_science : Nat := time_after_break2 + 75

theorem time_after_classes (start_time : Nat := 12 * 60) : (start_time + time_after_science) % 1440 = 17 * 60 + 10 :=
by
  sorry

end time_after_classes_l1135_113588


namespace mass_percentage_C_is_54_55_l1135_113589

def mass_percentage (C: String) (percentage: ℝ) : Prop :=
  percentage = 54.55

theorem mass_percentage_C_is_54_55 :
  mass_percentage "C" 54.55 :=
by
  unfold mass_percentage
  rfl

end mass_percentage_C_is_54_55_l1135_113589


namespace alice_has_ball_after_two_turns_l1135_113513

def prob_alice_keeps_ball : ℚ := (2/3 * 1/2) + (1/3 * 1/3)

theorem alice_has_ball_after_two_turns :
  prob_alice_keeps_ball = 4 / 9 :=
by
  -- This line is just a placeholder for the actual proof
  sorry

end alice_has_ball_after_two_turns_l1135_113513


namespace max_principals_ten_years_l1135_113577

theorem max_principals_ten_years : 
  (∀ (P : ℕ → Prop), (∀ n, n ≥ 10 → ∀ i, ¬P (n - i)) → ∀ p, p ≤ 4 → 
  (∃ n ≤ 10, ∀ k, k ≥ n → P k)) :=
sorry

end max_principals_ten_years_l1135_113577


namespace odd_factors_360_l1135_113537

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l1135_113537


namespace triangle_inequality_l1135_113509

variable {a b c : ℝ}
variable {x y z : ℝ}

theorem triangle_inequality (ha : a ≥ b) (hb : b ≥ c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hx_yz_sum : x + y + z = π) :
  bc + ca - ab < bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ∧
  bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 := sorry

end triangle_inequality_l1135_113509


namespace electric_fan_wattage_l1135_113528

theorem electric_fan_wattage (hours_per_day : ℕ) (energy_per_month : ℝ) (days_per_month : ℕ) 
  (h1 : hours_per_day = 8) (h2 : energy_per_month = 18) (h3 : days_per_month = 30) : 
  (energy_per_month * 1000) / (days_per_month * hours_per_day) = 75 := 
by { 
  -- Placeholder for the proof
  sorry 
}

end electric_fan_wattage_l1135_113528


namespace find_a_l1135_113536

theorem find_a (x y a : ℤ) (h1 : a * x + y = 40) (h2 : 2 * x - y = 20) (h3 : 3 * y^2 = 48) : a = 3 :=
sorry

end find_a_l1135_113536


namespace gcd_of_18_and_30_l1135_113575

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l1135_113575


namespace new_ratio_books_to_clothes_l1135_113510

-- Given initial conditions
def initial_ratio := (7, 4, 3)
def electronics_weight : ℕ := 12
def clothes_removed : ℕ := 8

-- Definitions based on the problem
def part_weight : ℕ := electronics_weight / initial_ratio.2.2
def initial_books_weight : ℕ := initial_ratio.1 * part_weight
def initial_clothes_weight : ℕ := initial_ratio.2.1 * part_weight
def new_clothes_weight : ℕ := initial_clothes_weight - clothes_removed

-- Proof of the new ratio
theorem new_ratio_books_to_clothes : (initial_books_weight, new_clothes_weight) = (7 * part_weight, 2 * part_weight) :=
sorry

end new_ratio_books_to_clothes_l1135_113510


namespace solve_system_l1135_113561

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem solve_system (a b : ℝ) :
  (f a b 1 = 4) ∧ (f a b 0 = 2) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_system_l1135_113561


namespace well_depth_l1135_113504

theorem well_depth (e x a b c d : ℝ)
  (h1 : x = 2 * a + b)
  (h2 : x = 3 * b + c)
  (h3 : x = 4 * c + d)
  (h4 : x = 5 * d + e)
  (h5 : x = 6 * e + a) :
  x = 721 / 76 * e ∧
  a = 265 / 76 * e ∧
  b = 191 / 76 * e ∧
  c = 37 / 19 * e ∧
  d = 129 / 76 * e :=
sorry

end well_depth_l1135_113504


namespace xy_sum_l1135_113540

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 20) : x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by
  sorry

end xy_sum_l1135_113540


namespace tan_add_pi_over_4_l1135_113503

variable {α : ℝ}

theorem tan_add_pi_over_4 (h : Real.tan (α - Real.pi / 4) = 1 / 4) : Real.tan (α + Real.pi / 4) = -4 :=
sorry

end tan_add_pi_over_4_l1135_113503


namespace solution_set_of_inequality_l1135_113573

theorem solution_set_of_inequality : 
  { x : ℝ | (x + 2) * (1 - x) > 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l1135_113573


namespace distinct_real_roots_iff_l1135_113591

-- Define f(x, a) := |x^2 - a| - x + 2
noncomputable def f (x a : ℝ) : ℝ := abs (x^2 - a) - x + 2

-- The proposition we need to prove
theorem distinct_real_roots_iff (a : ℝ) (h : 0 < a) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 4 < a :=
by
  sorry

end distinct_real_roots_iff_l1135_113591


namespace lines_identical_pairs_count_l1135_113541

theorem lines_identical_pairs_count :
  (∃ a d : ℝ, (4 * x + a * y + d = 0 ∧ d * x - 3 * y + 15 = 0)) →
  (∃! n : ℕ, n = 2) := 
sorry

end lines_identical_pairs_count_l1135_113541


namespace temple_run_red_coins_l1135_113543

variables (x y z : ℕ)

theorem temple_run_red_coins :
  x + y + z = 2800 →
  x + 3 * y + 5 * z = 7800 →
  z = y + 200 →
  y = 700 := 
by 
  intro h1 h2 h3
  sorry

end temple_run_red_coins_l1135_113543


namespace find_height_of_triangular_prism_l1135_113551

-- Define the conditions
def volume (V : ℝ) : Prop := V = 120
def base_side1 (a : ℝ) : Prop := a = 3
def base_side2 (b : ℝ) : Prop := b = 4

-- The final proof problem
theorem find_height_of_triangular_prism (V : ℝ) (a : ℝ) (b : ℝ) (h : ℝ) 
  (h1 : volume V) (h2 : base_side1 a) (h3 : base_side2 b) : h = 20 :=
by
  -- The actual proof goes here
  sorry

end find_height_of_triangular_prism_l1135_113551


namespace composite_quadratic_l1135_113557

theorem composite_quadratic (m n : ℤ) (x1 x2 : ℤ)
  (h1 : 2 * x1^2 + m * x1 + 2 - n = 0)
  (h2 : 2 * x2^2 + m * x2 + 2 - n = 0)
  (h3 : x1 ≠ 0) 
  (h4 : x2 ≠ 0) :
  ∃ (k : ℕ), ∃ (l : ℕ), 
    (k > 1) ∧ (l > 1) ∧ (k * l = (m^2 + n^2) / 4) := sorry

end composite_quadratic_l1135_113557


namespace average_speed_is_correct_l1135_113579

-- Definitions for the conditions
def speed_first_hour : ℕ := 140
def speed_second_hour : ℕ := 40
def total_distance : ℕ := speed_first_hour + speed_second_hour
def total_time : ℕ := 2

-- The statement we need to prove
theorem average_speed_is_correct : total_distance / total_time = 90 := by
  -- We would place the proof here
  sorry

end average_speed_is_correct_l1135_113579


namespace chord_equation_l1135_113583

variable {x y k b : ℝ}

-- Define the condition of the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 - 4 = 0

-- Define the condition that the point M(1, 1) is the midpoint
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- Define the line equation in terms of its slope k and y-intercept b
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

theorem chord_equation :
  (∃ (x₁ x₂ y₁ y₂ : ℝ), ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ midpoint_condition x₁ y₁ x₂ y₂) →
  (∃ (k b : ℝ), line k b x y ∧ k + b = 1 ∧ b = 1 - k) →
  y = -0.5 * x + 1.5 ↔ x + 2 * y - 3 = 0 :=
by
  sorry

end chord_equation_l1135_113583


namespace proof_problem_l1135_113511

theorem proof_problem (x : ℕ) (h : (x - 4) / 10 = 5) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l1135_113511


namespace translate_line_upwards_by_3_translate_line_right_by_3_l1135_113599

theorem translate_line_upwards_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y' := y + 3
  y' = 2 * x - 1 := 
by
  let y := 2 * x - 4
  let y' := y + 3
  sorry

theorem translate_line_right_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  y_right = 2 * x - 10 :=
by
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  sorry

end translate_line_upwards_by_3_translate_line_right_by_3_l1135_113599


namespace binom_10_3_eq_120_l1135_113566

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l1135_113566


namespace total_cost_family_visit_l1135_113550

/-
Conditions:
1. entrance_ticket_cost: $5 per person
2. attraction_ticket_cost_kid: $2 per kid
3. attraction_ticket_cost_parent: $4 per parent
4. family_discount_threshold: A family of 6 or more gets a 10% discount on entrance tickets
5. senior_discount: Senior citizens get a 50% discount on attraction tickets
6. family_composition: 4 children, 2 parents, and 1 grandmother
7. visit_attraction: The family plans to visit at least one attraction
-/

def entrance_ticket_cost : ℝ := 5
def attraction_ticket_cost_kid : ℝ := 2
def attraction_ticket_cost_parent : ℝ := 4
def family_discount_threshold : ℕ := 6
def family_discount_rate : ℝ := 0.10
def senior_discount_rate : ℝ := 0.50
def number_of_kids : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_seniors : ℕ := 1

theorem total_cost_family_visit : 
  let total_entrance_fee := (number_of_kids + number_of_parents + number_of_seniors) * entrance_ticket_cost 
  let total_entrance_fee_discounted := total_entrance_fee * (1 - family_discount_rate)
  let total_attraction_fee_kids := number_of_kids * attraction_ticket_cost_kid
  let total_attraction_fee_parents := number_of_parents * attraction_ticket_cost_parent
  let total_attraction_fee_seniors := number_of_seniors * attraction_ticket_cost_parent * (1 - senior_discount_rate)
  let total_attraction_fee := total_attraction_fee_kids + total_attraction_fee_parents + total_attraction_fee_seniors
  (number_of_kids + number_of_parents + number_of_seniors ≥ family_discount_threshold) → 
  (total_entrance_fee_discounted + total_attraction_fee = 49.50) :=
by
  -- Assuming we calculate entrance fee and attraction fee correctly, state the theorem
  sorry

end total_cost_family_visit_l1135_113550


namespace border_material_length_l1135_113574

noncomputable def area (r : ℝ) : ℝ := (22 / 7) * r^2

theorem border_material_length (r : ℝ) (C : ℝ) (border : ℝ) : 
  area r = 616 →
  C = 2 * (22 / 7) * r →
  border = C + 3 →
  border = 91 :=
by
  intro h_area h_circumference h_border
  sorry

end border_material_length_l1135_113574


namespace taxi_ride_cost_l1135_113525

-- Definitions based on conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def minimum_charge : ℝ := 5.00
def fare (miles : ℝ) : ℝ := base_fare + miles * cost_per_mile

-- Theorem statement reflecting the problem
theorem taxi_ride_cost (miles : ℝ) (h : miles < 4) : fare miles < minimum_charge → fare miles = minimum_charge :=
by
  sorry

end taxi_ride_cost_l1135_113525


namespace louis_current_age_l1135_113578

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l1135_113578


namespace B_completes_remaining_work_in_23_days_l1135_113597

noncomputable def A_work_rate : ℝ := 1 / 45
noncomputable def B_work_rate : ℝ := 1 / 40
noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate
noncomputable def work_done_together_in_9_days : ℝ := combined_work_rate * 9
noncomputable def remaining_work : ℝ := 1 - work_done_together_in_9_days
noncomputable def days_B_completes_remaining_work : ℝ := remaining_work / B_work_rate

theorem B_completes_remaining_work_in_23_days :
  days_B_completes_remaining_work = 23 :=
by 
  -- Proof omitted - please fill in the proof steps
  sorry

end B_completes_remaining_work_in_23_days_l1135_113597


namespace medians_sum_square_l1135_113581

-- Define the sides of the triangle
variables {a b c : ℝ}

-- Define diameters
variables {D : ℝ}

-- Define medians of the triangle
variables {m_a m_b m_c : ℝ}

-- Defining the theorem statement
theorem medians_sum_square :
  m_a ^ 2 + m_b ^ 2 + m_c ^ 2 = (3 / 4) * (a ^ 2 + b ^ 2 + c ^ 2) + (3 / 4) * D ^ 2 :=
sorry

end medians_sum_square_l1135_113581


namespace hounds_score_points_l1135_113507

theorem hounds_score_points (x y : ℕ) (h_total : x + y = 82) (h_margin : x - y = 18) : y = 32 :=
sorry

end hounds_score_points_l1135_113507


namespace remainder_of_largest_divided_by_second_smallest_l1135_113518

theorem remainder_of_largest_divided_by_second_smallest 
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  c % b = 1 :=
by
  -- We assume and/or prove the necessary statements here.
  -- The core of the proof uses existing facts or assumptions.
  -- We insert the proof strategy or intermediate steps here.
  
  sorry

end remainder_of_largest_divided_by_second_smallest_l1135_113518


namespace multiply_72517_9999_l1135_113523

theorem multiply_72517_9999 : 72517 * 9999 = 725097483 :=
by
  sorry

end multiply_72517_9999_l1135_113523
